"""
Tissue Segmentation Inference Module
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import json
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import segmentation_config, TISSUE_CLASSES, DEVICE
from .model import TissueSegmentor
from .color_correction import AdaptiveColorCorrector


@dataclass
class SegmentationResult:
    """Segmentation result container"""
    mask: np.ndarray
    class_areas: Dict[str, int]
    class_percentages: Dict[str, float]
    colored_mask: Optional[np.ndarray] = None
    overlay: Optional[np.ndarray] = None
    sam_mask: Optional[np.ndarray] = None  # Binary wound boundary from SAM
    sam_confidence: float = 0.0


class SegmentationInference:
    """
    High-level inference wrapper for tissue segmentation
    Now with SAM-enhanced wound boundary detection
    """
    
    def __init__(self,
                 weights_path: Optional[Path] = None,
                 device: Optional[torch.device] = None,
                 use_sam: bool = True):
        self.segmentor = TissueSegmentor(
            weights_path=weights_path,
            device=device
        )
        self.use_sam = use_sam
        self._sam_segmentor = None
        self._color_corrector = AdaptiveColorCorrector(confidence_threshold=0.75)
        self._is_loaded = False
        
    def load(self) -> bool:
        """Load model"""
        self._is_loaded = self.segmentor.load()
        
        # Load MedSAM if enabled (better than regular SAM for medical images)
        if self.use_sam:
            try:
                from .medsam_segmentor import MedSAMWoundSegmentor
                self._sam_segmentor = MedSAMWoundSegmentor()
                # Don't load MedSAM yet - lazy load on first use
                print("MedSAM segmentor initialized (will load on first use)")
            except Exception as e:
                print(f"MedSAM initialization failed: {e}")
                # Fallback to regular SAM
                try:
                    from .sam_segmentor import SAMWoundSegmentor
                    self._sam_segmentor = SAMWoundSegmentor(model_type="vit_b")
                    print("Falling back to regular SAM")
                except:
                    self._sam_segmentor = None
        
        return self._is_loaded
    
    def segment(self, 
                image: Union[np.ndarray, str, Path],
                return_visualization: bool = True,
                detection_bbox: Optional[Tuple[int, int, int, int]] = None) -> SegmentationResult:
        """
        Segment wound image
        
        Args:
            image: Input image (BGR numpy array or path)
            return_visualization: Whether to generate visualization
            detection_bbox: Optional (x1, y1, x2, y2) from YOLO detection for SAM
            
        Returns:
            SegmentationResult with mask and statistics
        """
        if not self._is_loaded:
            self.load()
        
        # Load image if path
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        
        if image is None:
            return SegmentationResult(
                mask=np.zeros((1, 1), dtype=np.uint8),
                class_areas={},
                class_percentages={}
            )
        
        # Get SAM wound boundary if available
        sam_mask = None
        sam_confidence = 0.0
        
        if self._sam_segmentor is not None:
            sam_mask, sam_confidence = self._get_sam_mask(image, detection_bbox)
        
        # Run tissue segmentation
        result = self.segmentor.segment(image)
        
        # Refine mask with SAM if available
        final_mask = result["mask"]
        if sam_mask is not None:
            final_mask = self._combine_masks(result["mask"], sam_mask, image)
        
        # Apply color-based correction for low-confidence pixels
        if "probs" in result and result["probs"] is not None:
            correction_result = self._color_corrector.correct_fast(
                image=image,
                mask=final_mask,
                probs=result["probs"],
                wound_mask=sam_mask
            )
            final_mask = correction_result.corrected_mask
            print(f"Color correction: {correction_result.corrections_made} pixels corrected")
        
        # Recalculate areas with corrected mask
        class_areas = self._calculate_class_areas(final_mask)
        class_percentages = self._calculate_percentages(class_areas)
        
        # Create result object
        seg_result = SegmentationResult(
            mask=final_mask,
            class_areas=class_areas,
            class_percentages=class_percentages,
            sam_mask=sam_mask,
            sam_confidence=sam_confidence
        )
        
        if return_visualization:
            seg_result.colored_mask = self._create_colored_mask(final_mask)
            seg_result.overlay = self.segmentor.visualize(image, final_mask)
        
        return seg_result
    
    def _get_sam_mask(self, 
                      image: np.ndarray,
                      bbox: Optional[Tuple[int, int, int, int]] = None) -> Tuple[Optional[np.ndarray], float]:
        """Get wound boundary mask from MedSAM/SAM"""
        try:
            # Try MedSAM's unified interface first
            if hasattr(self._sam_segmentor, 'segment'):
                result = self._sam_segmentor.segment(image, bbox=bbox)
            elif bbox is not None:
                # Fallback to regular SAM with bbox
                result = self._sam_segmentor.segment_with_bbox(image, bbox)
            else:
                # Fallback to regular SAM auto
                result = self._sam_segmentor.segment_auto(image, use_center=True)
            
            return result.wound_mask, result.confidence
        except Exception as e:
            print(f"MedSAM/SAM segmentation failed: {e}")
            return None, 0.0
    
    def _combine_masks(self, 
                       tissue_mask: np.ndarray,
                       sam_mask: np.ndarray,
                       image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Combine tissue segmentation with MedSAM wound boundary.
        MedSAM handles wound detection, so we just use its mask as the boundary.
        """
        # Resize SAM mask if needed
        if sam_mask.shape != tissue_mask.shape:
            sam_mask = cv2.resize(sam_mask, (tissue_mask.shape[1], tissue_mask.shape[0]))
        
        # Create refined mask: tissue types only within MedSAM boundary
        refined_mask = tissue_mask.copy()
        
        # Color-based filtering DISABLED - MedSAM handles wound boundary detection
        # The color filter was too aggressive and caused 100% background classification
        
        # Outside MedSAM boundary → background (class 0)
        refined_mask[sam_mask == 0] = 0
        
        return refined_mask
    
    def _create_wound_color_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Create a mask of pixels that have wound-like colors.
        Key insight: EXCLUDE healthy skin first, then identify wounds.
        """
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Step 1: Identify HEALTHY SKIN to exclude
        # Healthy skin has: H=0-20, moderate saturation (20-50), high brightness (130-255)
        lower_skin = np.array([0, 20, 130])
        upper_skin = np.array([25, 90, 255])
        healthy_skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Step 2: Identify WOUND regions
        wound_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 2a. Very dark regions (necrotic tissue, eschar) - gray < 50
        very_dark_mask = (gray < 50).astype(np.uint8) * 255
        wound_mask = cv2.bitwise_or(wound_mask, very_dark_mask)
        
        # 2b. Moderately dark (potential wound) - gray 50-90
        mod_dark_mask = ((gray >= 50) & (gray < 90)).astype(np.uint8) * 255
        wound_mask = cv2.bitwise_or(wound_mask, mod_dark_mask)
        
        # 2c. Highly saturated red (blood, inflamed tissue) - H=0-8 or 175-180, S>100
        lower_blood1 = np.array([0, 100, 30])
        upper_blood1 = np.array([8, 255, 255])
        lower_blood2 = np.array([175, 100, 30])
        upper_blood2 = np.array([180, 255, 255])
        blood_mask = cv2.inRange(hsv, lower_blood1, upper_blood1) | cv2.inRange(hsv, lower_blood2, upper_blood2)
        wound_mask = cv2.bitwise_or(wound_mask, blood_mask)
        
        # 2d. Bright yellow (slough, pus) - H=20-40, S>60, V>100
        lower_slough = np.array([20, 60, 100])
        upper_slough = np.array([40, 255, 255])
        slough_mask = cv2.inRange(hsv, lower_slough, upper_slough)
        wound_mask = cv2.bitwise_or(wound_mask, slough_mask)
        
        # 2e. Very dark brown (eschar) - H=5-18, S>40, V<100
        lower_eschar = np.array([5, 40, 10])
        upper_eschar = np.array([18, 255, 100])
        eschar_mask = cv2.inRange(hsv, lower_eschar, upper_eschar)
        wound_mask = cv2.bitwise_or(wound_mask, eschar_mask)
        
        # Step 3: Remove healthy skin from wound mask
        wound_mask = cv2.bitwise_and(wound_mask, cv2.bitwise_not(healthy_skin_mask))
        
        # Step 4: Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        wound_mask = cv2.morphologyEx(wound_mask, cv2.MORPH_CLOSE, kernel)
        wound_mask = cv2.morphologyEx(wound_mask, cv2.MORPH_OPEN, kernel)
        
        return wound_mask
    
    def _calculate_class_areas(self, mask: np.ndarray) -> Dict[str, int]:
        """Calculate area for each class"""
        areas = {}
        for class_id, class_name in TISSUE_CLASSES.items():
            areas[class_name] = int(np.sum(mask == class_id))
        return areas
    
    def _calculate_percentages(self, areas: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate percentage for each class WITHIN WOUND ONLY.
        Excludes background so percentages reflect wound tissue composition.
        """
        # Get wound-only total (exclude background)
        wound_total = sum(area for name, area in areas.items() if name != 'background')
        
        if wound_total == 0:
            # No wound detected, fall back to total
            total = sum(areas.values())
            if total == 0:
                return {name: 0.0 for name in areas}
            return {name: (area / total) * 100 for name, area in areas.items()}
        
        # Calculate percentages within wound region only
        percentages = {}
        for name, area in areas.items():
            if name == 'background':
                # Background as percentage of total image
                total = sum(areas.values())
                percentages[name] = (area / total) * 100 if total > 0 else 0.0
            else:
                # Tissue types as percentage of wound area
                percentages[name] = (area / wound_total) * 100
        
        return percentages
    
    def _create_colored_mask(self, mask: np.ndarray) -> np.ndarray:
        """Create RGB colored mask"""
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in self.segmentor.class_colors.items():
            colored[mask == class_id] = color
        
        return colored
    
    def segment_batch(self,
                      images: List[Union[np.ndarray, str, Path]],
                      return_visualization: bool = False) -> List[SegmentationResult]:
        """Segment multiple images"""
        results = []
        
        for image in tqdm(images, desc="Segmenting"):
            result = self.segment(image, return_visualization)
            results.append(result)
        
        return results
    
    def get_tissue_features(self, result: SegmentationResult) -> Dict[str, float]:
        """
        Extract features from segmentation for downstream models
        
        Returns:
            Dictionary of features:
            - Tissue fractions
            - Wound complexity metrics
            - Healing indicators
        """
        features = {}
        
        # Tissue fractions
        for name, pct in result.class_percentages.items():
            features[f"tissue_{name}_pct"] = pct / 100.0  # Normalize to [0, 1]
        
        # Calculate derived metrics
        granulation_pct = result.class_percentages.get("granulation", 0)
        necrotic_pct = result.class_percentages.get("necrotic", 0)
        slough_pct = result.class_percentages.get("slough", 0)
        
        # Healthy tissue ratio (granulation / (necrotic + slough + granulation))
        wound_tissue = granulation_pct + necrotic_pct + slough_pct
        if wound_tissue > 0:
            features["healthy_ratio"] = granulation_pct / wound_tissue
        else:
            features["healthy_ratio"] = 0.0
        
        # Necrotic burden
        features["necrotic_burden"] = (necrotic_pct + slough_pct) / 100.0
        
        # Wound area (non-background, non-epithelium)
        wound_area = sum([
            result.class_areas.get(name, 0) 
            for name in ["granulation", "slough", "necrotic"]
        ])
        total_area = sum(result.class_areas.values())
        features["wound_area_ratio"] = wound_area / total_area if total_area > 0 else 0
        
        return features
    
    def warmup(self, iterations: int = 3):
        """Warmup model for faster inference"""
        if not self._is_loaded:
            self.load()
        
        dummy = np.zeros((512, 512, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.segment(dummy, return_visualization=False)
        
        print("Segmentation warmup complete")


def process_dataset(input_dir: Path, 
                    output_dir: Path,
                    weights_path: Optional[Path] = None):
    """
    Process entire dataset and save segmentation results
    """
    inference = SegmentationInference(weights_path=weights_path)
    inference.load()
    inference.warmup()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    masks_dir.mkdir(exist_ok=True)
    overlays_dir.mkdir(exist_ok=True)
    
    image_files = list(input_dir.rglob("*.jpg")) + list(input_dir.rglob("*.png"))
    
    all_features = []
    
    for img_path in tqdm(image_files, desc="Processing"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        result = inference.segment(image, return_visualization=True)
        features = inference.get_tissue_features(result)
        
        # Save outputs
        rel_path = img_path.relative_to(input_dir)
        
        mask_path = masks_dir / rel_path.with_suffix(".png")
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), result.mask)
        
        overlay_path = overlays_dir / rel_path
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_path), result.overlay)
        
        all_features.append({
            "image": str(img_path),
            "mask": str(mask_path),
            "overlay": str(overlay_path),
            "class_percentages": result.class_percentages,
            "features": features
        })
    
    # Save features
    with open(output_dir / "segmentation_results.json", "w") as f:
        json.dump(all_features, f, indent=2)
    
    print(f"\nProcessed {len(all_features)} images")
    print(f"Results saved to {output_dir}")
    
    return all_features


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Segmentation Inference")
    parser.add_argument("--image", type=str, help="Single image path")
    parser.add_argument("--dataset", type=str, help="Dataset directory")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--weights", type=str, help="Model weights path")
    
    args = parser.parse_args()
    
    if args.image:
        inference = SegmentationInference(
            weights_path=Path(args.weights) if args.weights else None
        )
        result = inference.segment(args.image)
        
        print("\nTissue Composition:")
        for name, pct in result.class_percentages.items():
            print(f"  {name}: {pct:.1f}%")
        
        # Save visualization
        if result.overlay is not None:
            cv2.imwrite("segmentation_result.jpg", result.overlay)
            print("\nVisualization saved to segmentation_result.jpg")
    
    if args.dataset and args.output:
        process_dataset(
            Path(args.dataset),
            Path(args.output),
            Path(args.weights) if args.weights else None
        )
