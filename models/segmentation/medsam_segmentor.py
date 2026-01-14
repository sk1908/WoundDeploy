"""
MedSAM-Enhanced Wound Segmentor
Uses MedSAM (Medical SAM) for better wound boundary detection
with automatic bounding box generation for abnormal regions
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import urllib.request

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import WEIGHTS_DIR, DEVICE


@dataclass
class MedSAMSegmentationResult:
    """MedSAM segmentation result"""
    wound_mask: np.ndarray  # Binary mask of wound region
    confidence: float
    bbox_used: Optional[Tuple[int, int, int, int]] = None


class MedSAMWoundSegmentor:
    """
    MedSAM-based wound segmentation.
    
    MedSAM is trained on 1.5M medical image-mask pairs and is better
    at identifying pathological regions compared to regular SAM.
    
    For wound detection, we:
    1. Automatically generate bounding box around abnormal regions
    2. Use MedSAM to segment within that bounding box
    """
    
    # MedSAM checkpoint URL (hosted on Google Drive, we'll use direct link)
    CHECKPOINT_URL = "https://huggingface.co/spaces/junma/MedSAM/resolve/main/medsam_vit_b.pth"
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or DEVICE
        self.model = None
        self._is_loaded = False
        
        # Checkpoint path
        self.checkpoint_path = WEIGHTS_DIR / "medsam_vit_b.pth"
    
    def _download_checkpoint(self) -> bool:
        """Download MedSAM checkpoint if not present"""
        if self.checkpoint_path.exists():
            print(f"MedSAM checkpoint already exists at {self.checkpoint_path}")
            return True
        
        print(f"Downloading MedSAM checkpoint...")
        print(f"URL: {self.CHECKPOINT_URL}")
        print(f"This may take a few minutes (~1.5GB)...")
        
        try:
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(self.CHECKPOINT_URL, self.checkpoint_path)
            print(f"Downloaded to: {self.checkpoint_path}")
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            print("You can manually download from: https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN")
            return False
    
    def load(self) -> bool:
        """Load MedSAM model"""
        try:
            from segment_anything import sam_model_registry
            
            if not self._download_checkpoint():
                print("Could not get MedSAM checkpoint, falling back to regular SAM")
                return False
            
            print(f"Loading MedSAM model...")
            
            # MedSAM uses the same architecture as SAM vit_b
            self.model = sam_model_registry["vit_b"](checkpoint=str(self.checkpoint_path))
            self.model.to(self.device)
            self.model.eval()
            
            # Create predictor
            from segment_anything import SamPredictor
            self.predictor = SamPredictor(self.model)
            
            self._is_loaded = True
            print(f"MedSAM loaded on {self.device}")
            return True
            
        except ImportError:
            print("segment-anything not installed.")
            print("Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
            return False
        except Exception as e:
            print(f"Error loading MedSAM: {e}")
            return False
    
    def _find_abnormal_region_bbox(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Automatically find bounding box around abnormal (wound-like) regions.
        
        Uses multiple detection strategies:
        1. Dark regions (necrotic tissue)
        2. Color anomalies (not normal skin color)
        3. Texture anomalies
        """
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect abnormal regions
        abnormal_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 1. Very dark regions (necrotic, eschar)
        dark_mask = (gray < 60).astype(np.uint8) * 255
        abnormal_mask = cv2.bitwise_or(abnormal_mask, dark_mask)
        
        # 2. Non-skin colored regions
        # Normal skin: H=0-25, S=20-80, V=100-255
        lower_skin = np.array([0, 20, 100])
        upper_skin = np.array([25, 80, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Everything that's NOT skin-colored (but still in foreground)
        foreground_mask = (gray > 20).astype(np.uint8) * 255  # Not black background
        non_skin_mask = cv2.bitwise_and(foreground_mask, cv2.bitwise_not(skin_mask))
        abnormal_mask = cv2.bitwise_or(abnormal_mask, non_skin_mask)
        
        # 3. High saturation regions (blood, pus, etc.)
        high_sat = (hsv[:, :, 1] > 100).astype(np.uint8) * 255
        abnormal_mask = cv2.bitwise_or(abnormal_mask, high_sat)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        abnormal_mask = cv2.morphologyEx(abnormal_mask, cv2.MORPH_CLOSE, kernel)
        abnormal_mask = cv2.morphologyEx(abnormal_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours and get bounding box
        contours, _ = cv2.findContours(abnormal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get bounding box that encompasses all abnormal regions
        all_points = np.vstack(contours)
        x, y, bw, bh = cv2.boundingRect(all_points)
        
        # Add padding (10% on each side)
        pad_x = int(bw * 0.1)
        pad_y = int(bh * 0.1)
        
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)
        
        # Filter out bboxes that are too small or too large
        area_ratio = (x2 - x1) * (y2 - y1) / (w * h)
        if area_ratio < 0.01 or area_ratio > 0.95:
            return None
        
        return (x1, y1, x2, y2)
    
    def segment(self, 
                image: np.ndarray,
                bbox: Optional[Tuple[int, int, int, int]] = None) -> MedSAMSegmentationResult:
        """
        Segment wound using MedSAM.
        
        Args:
            image: Input image (BGR)
            bbox: Optional bounding box. If None, auto-detect abnormal region.
            
        Returns:
            MedSAMSegmentationResult with wound mask
        """
        if not self._is_loaded:
            if not self.load():
                return MedSAMSegmentationResult(
                    wound_mask=np.zeros(image.shape[:2], dtype=np.uint8),
                    confidence=0.0
                )
        
        # Auto-detect bounding box if not provided
        if bbox is None:
            bbox = self._find_abnormal_region_bbox(image)
            
        if bbox is None:
            # No abnormal region found, return empty mask
            return MedSAMSegmentationResult(
                wound_mask=np.zeros(image.shape[:2], dtype=np.uint8),
                confidence=0.0
            )
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image for predictor
        self.predictor.set_image(image_rgb)
        
        # Prepare box prompt
        input_box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
        
        # Predict mask
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,
        )
        
        # Select best mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx].astype(np.uint8)
        best_score = float(scores[best_idx])
        
        return MedSAMSegmentationResult(
            wound_mask=best_mask,
            confidence=best_score,
            bbox_used=bbox
        )
    
    def visualize(self,
                  image: np.ndarray,
                  result: MedSAMSegmentationResult,
                  alpha: float = 0.5) -> np.ndarray:
        """Create visualization overlay"""
        overlay = image.copy()
        
        # Draw mask overlay (green for wound)
        mask_colored = np.zeros_like(image)
        mask_colored[result.wound_mask > 0] = (0, 255, 0)
        overlay = cv2.addWeighted(overlay, 1, mask_colored, alpha, 0)
        
        # Draw bounding box if available
        if result.bbox_used:
            x1, y1, x2, y2 = result.bbox_used
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw contour
        contours, _ = cv2.findContours(
            result.wound_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        
        return overlay


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MedSAM Wound Segmentation")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="medsam_result.jpg")
    
    args = parser.parse_args()
    
    # Load segmentor
    segmentor = MedSAMWoundSegmentor()
    
    # Load image
    image = cv2.imread(args.image)
    
    # Segment (auto-detect bbox)
    result = segmentor.segment(image)
    
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Bbox used: {result.bbox_used}")
    
    # Visualize
    vis = segmentor.visualize(image, result)
    cv2.imwrite(args.output, vis)
    print(f"Saved to {args.output}")
