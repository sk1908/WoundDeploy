"""
SAM (Segment Anything Model) Mask Generator
Generates initial segmentation masks for wound images
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from tqdm import tqdm
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    UNIFIED_DIR, PROCESSED_DIR, WEIGHTS_DIR,
    sam_config, DEVICE
)


class SAMMaskGenerator:
    """
    Uses Segment Anything Model to generate wound segmentation masks.
    These can be used as initial pseudo-labels for training the segmentation model.
    """
    
    def __init__(self, checkpoint_path: Optional[Path] = None):
        self.checkpoint_path = checkpoint_path or sam_config.checkpoint_path
        self.model_type = sam_config.model_type
        self.device = DEVICE
        
        self.sam = None
        self.mask_generator = None
        self.predictor = None
        
    def load_model(self):
        """Load SAM model"""
        try:
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
            
            if not self.checkpoint_path.exists():
                print(f"SAM checkpoint not found at {self.checkpoint_path}")
                print("Please download from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
                print(f"Expected path: {self.checkpoint_path}")
                return False
            
            print(f"Loading SAM model ({self.model_type})...")
            self.sam = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint_path))
            self.sam.to(self.device)
            
            # Automatic mask generator for whole image
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=sam_config.points_per_side,
                pred_iou_thresh=sam_config.pred_iou_thresh,
                stability_score_thresh=sam_config.stability_score_thresh,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )
            
            # Point/box predictor for interactive segmentation
            self.predictor = SamPredictor(self.sam)
            
            print("SAM model loaded successfully!")
            return True
            
        except ImportError:
            print("segment-anything not installed. Install with:")
            print("pip install git+https://github.com/facebookresearch/segment-anything.git")
            return False
        except Exception as e:
            print(f"Error loading SAM: {e}")
            return False
    
    def generate_masks_auto(self, image: np.ndarray) -> List[Dict]:
        """
        Generate all masks automatically using SAM
        
        Returns:
            List of mask dictionaries with keys:
            - segmentation: binary mask
            - area: mask area in pixels
            - bbox: [x, y, w, h]
            - predicted_iou: model's confidence
            - stability_score: mask stability
        """
        if self.mask_generator is None:
            if not self.load_model():
                return []
        
        # SAM expects RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        masks = self.mask_generator.generate(image_rgb)
        
        # Sort by area (largest first, likely to be the wound)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        return masks
    
    def generate_mask_with_point(self, image: np.ndarray, 
                                  point: Tuple[int, int],
                                  point_label: int = 1) -> np.ndarray:
        """
        Generate mask using a point prompt
        
        Args:
            image: Input image (BGR)
            point: (x, y) coordinate
            point_label: 1 for foreground, 0 for background
            
        Returns:
            Binary mask
        """
        if self.predictor is None:
            if not self.load_model():
                return np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Set image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        
        # Predict
        input_point = np.array([[point[0], point[1]]])
        input_label = np.array([point_label])
        
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        # Return best mask
        best_idx = np.argmax(scores)
        return masks[best_idx].astype(np.uint8)
    
    def generate_mask_with_box(self, image: np.ndarray,
                                box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Generate mask using a bounding box prompt
        
        Args:
            image: Input image (BGR)
            box: (x1, y1, x2, y2) bounding box
            
        Returns:
            Binary mask
        """
        if self.predictor is None:
            if not self.load_model():
                return np.zeros(image.shape[:2], dtype=np.uint8)
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        
        input_box = np.array([box[0], box[1], box[2], box[3]])
        
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,
        )
        
        best_idx = np.argmax(scores)
        return masks[best_idx].astype(np.uint8)
    
    def select_wound_mask(self, masks: List[Dict], 
                          image_shape: Tuple[int, int],
                          min_area_ratio: float = 0.01,
                          max_area_ratio: float = 0.9) -> Optional[np.ndarray]:
        """
        Select the most likely wound mask from auto-generated masks
        
        Heuristics:
        - Not too small (> 1% of image)
        - Not too large (< 90% of image, i.e., not background)
        - High stability score
        - Central location preference
        """
        total_area = image_shape[0] * image_shape[1]
        center_x, center_y = image_shape[1] // 2, image_shape[0] // 2
        
        best_mask = None
        best_score = -1
        
        for mask_data in masks:
            area = mask_data['area']
            area_ratio = area / total_area
            
            # Filter by area
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue
            
            # Get mask centroid
            mask = mask_data['segmentation']
            coords = np.where(mask)
            if len(coords[0]) == 0:
                continue
                
            centroid_y = np.mean(coords[0])
            centroid_x = np.mean(coords[1])
            
            # Distance from center (normalized)
            center_dist = np.sqrt(
                ((centroid_x - center_x) / image_shape[1]) ** 2 +
                ((centroid_y - center_y) / image_shape[0]) ** 2
            )
            
            # Scoring: prefer central, medium-sized, stable masks
            stability = mask_data.get('stability_score', 0.5)
            iou = mask_data.get('predicted_iou', 0.5)
            
            # Size score (prefer medium-sized masks, ~5-30% of image)
            optimal_size = 0.15
            size_score = 1 - abs(area_ratio - optimal_size) / optimal_size
            
            # Center score (prefer central masks)
            center_score = 1 - center_dist
            
            # Combined score
            score = (stability * 0.3 + iou * 0.3 + size_score * 0.2 + center_score * 0.2)
            
            if score > best_score:
                best_score = score
                best_mask = mask.astype(np.uint8) * 255
                
        return best_mask
    
    def process_dataset(self, input_dir: Path, output_dir: Path, 
                        use_auto: bool = True):
        """
        Process all images in a directory and generate masks
        
        Args:
            input_dir: Directory with images
            output_dir: Directory to save masks
            use_auto: Use automatic mask generation (vs manual point selection)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(input_dir.rglob("*.jpg")) + list(input_dir.rglob("*.png"))
        
        print(f"Processing {len(image_files)} images...")
        
        results = []
        
        for img_path in tqdm(image_files):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Generate relative path for output
            rel_path = img_path.relative_to(input_dir)
            mask_path = output_dir / rel_path.with_suffix(".png")
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            
            if use_auto:
                masks = self.generate_masks_auto(image)
                wound_mask = self.select_wound_mask(masks, image.shape[:2])
            else:
                # Use center point as default prompt
                h, w = image.shape[:2]
                wound_mask = self.generate_mask_with_point(image, (w//2, h//2))
                wound_mask = wound_mask * 255
            
            if wound_mask is not None:
                cv2.imwrite(str(mask_path), wound_mask)
                results.append({
                    "image": str(img_path),
                    "mask": str(mask_path),
                    "success": True
                })
            else:
                results.append({
                    "image": str(img_path),
                    "mask": None,
                    "success": False
                })
        
        # Save results log
        log_path = output_dir / "generation_log.json"
        with open(log_path, "w") as f:
            json.dump(results, f, indent=2)
            
        success_count = sum(1 for r in results if r["success"])
        print(f"\nMask generation complete!")
        print(f"Success: {success_count}/{len(results)}")
        
        return results


def download_sam_checkpoint(model_type: str = "vit_h"):
    """Download SAM checkpoint"""
    import urllib.request
    
    checkpoints = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }
    
    if model_type not in checkpoints:
        print(f"Unknown model type: {model_type}")
        return
    
    url = checkpoints[model_type]
    filename = url.split("/")[-1]
    output_path = WEIGHTS_DIR / filename
    
    if output_path.exists():
        print(f"Checkpoint already exists: {output_path}")
        return output_path
    
    print(f"Downloading {model_type} checkpoint...")
    print(f"URL: {url}")
    print(f"Output: {output_path}")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print("Download complete!")
        return output_path
    except Exception as e:
        print(f"Download failed: {e}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SAM Mask Generator")
    parser.add_argument("--download", action="store_true", help="Download SAM checkpoint")
    parser.add_argument("--model", default="vit_h", help="Model type (vit_h, vit_l, vit_b)")
    parser.add_argument("--process", action="store_true", help="Process dataset")
    
    args = parser.parse_args()
    
    if args.download:
        download_sam_checkpoint(args.model)
    
    if args.process:
        generator = SAMMaskGenerator()
        if generator.load_model():
            generator.process_dataset(
                input_dir=UNIFIED_DIR,
                output_dir=PROCESSED_DIR / "sam_masks"
            )
