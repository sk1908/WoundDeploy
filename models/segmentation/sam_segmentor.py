"""
SAM-Enhanced Wound Segmentor
Uses Segment Anything Model for accurate wound boundary detection
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import urllib.request
import os

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import WEIGHTS_DIR, DEVICE


@dataclass
class SAMSegmentationResult:
    """SAM segmentation result"""
    wound_mask: np.ndarray  # Binary mask of wound region
    confidence: float  # Predicted IoU score
    bbox_used: Optional[Tuple[int, int, int, int]] = None


class SAMWoundSegmentor:
    """
    Segment Anything Model wrapper for wound boundary detection.
    Uses detection bounding box or center point to prompt SAM
    for accurate wound boundary segmentation.
    """
    
    # Checkpoint URLs for different SAM model sizes
    CHECKPOINT_URLS = {
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    }
    
    def __init__(self,
                 model_type: str = "vit_b",  # Use smaller model by default
                 device: Optional[torch.device] = None):
        self.model_type = model_type
        self.device = device or DEVICE
        
        self.sam = None
        self.predictor = None
        self._is_loaded = False
        
        # Checkpoint path
        self.checkpoint_path = WEIGHTS_DIR / f"sam_{model_type}.pth"
    
    def _download_checkpoint(self) -> bool:
        """Download SAM checkpoint if not present"""
        if self.checkpoint_path.exists():
            return True
        
        if self.model_type not in self.CHECKPOINT_URLS:
            print(f"Unknown SAM model type: {self.model_type}")
            return False
        
        url = self.CHECKPOINT_URLS[self.model_type]
        print(f"Downloading SAM checkpoint ({self.model_type})...")
        print(f"URL: {url}")
        print(f"This may take a few minutes...")
        
        try:
            # Create weights directory if not exists
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Download with progress
            urllib.request.urlretrieve(url, self.checkpoint_path)
            print(f"Downloaded to: {self.checkpoint_path}")
            return True
        except Exception as e:
            print(f"Download failed: {e}")
            return False
    
    def load(self) -> bool:
        """Load SAM model"""
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            # Download checkpoint if needed
            if not self._download_checkpoint():
                print("Could not get SAM checkpoint")
                return False
            
            print(f"Loading SAM model ({self.model_type})...")
            self.sam = sam_model_registry[self.model_type](
                checkpoint=str(self.checkpoint_path)
            )
            self.sam.to(self.device)
            self.sam.eval()
            
            # Create predictor for prompted segmentation
            self.predictor = SamPredictor(self.sam)
            
            self._is_loaded = True
            print(f"SAM loaded on {self.device}")
            return True
            
        except ImportError:
            print("segment-anything not installed.")
            print("Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
            return False
        except Exception as e:
            print(f"Error loading SAM: {e}")
            return False
    
    def segment_with_bbox(self,
                          image: np.ndarray,
                          bbox: Tuple[int, int, int, int]) -> SAMSegmentationResult:
        """
        Segment wound using bounding box prompt
        
        Args:
            image: Input image (BGR)
            bbox: Bounding box (x1, y1, x2, y2) from YOLO detection
            
        Returns:
            SAMSegmentationResult with wound mask
        """
        if not self._is_loaded:
            if not self.load():
                # Return empty mask if loading fails
                return SAMSegmentationResult(
                    wound_mask=np.zeros(image.shape[:2], dtype=np.uint8),
                    confidence=0.0,
                    bbox_used=bbox
                )
        
        # Convert BGR to RGB for SAM
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image for predictor
        self.predictor.set_image(image_rgb)
        
        # Prepare box prompt (SAM expects [x1, y1, x2, y2])
        input_box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
        
        # Predict mask
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,
        )
        
        # Select best mask (highest IoU score)
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx].astype(np.uint8)
        best_score = float(scores[best_idx])
        
        return SAMSegmentationResult(
            wound_mask=best_mask,
            confidence=best_score,
            bbox_used=bbox
        )
    
    def segment_with_point(self,
                           image: np.ndarray,
                           point: Tuple[int, int],
                           is_foreground: bool = True) -> SAMSegmentationResult:
        """
        Segment wound using point prompt with optional negative points
        
        Args:
            image: Input image (BGR)
            point: (x, y) coordinate inside wound
            is_foreground: True if point is inside wound
            
        Returns:
            SAMSegmentationResult with wound mask
        """
        if not self._is_loaded:
            if not self.load():
                return SAMSegmentationResult(
                    wound_mask=np.zeros(image.shape[:2], dtype=np.uint8),
                    confidence=0.0
                )
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        
        # Get negative points (healthy skin regions to exclude)
        negative_points = self._find_healthy_skin_points(image, point)
        
        # Combine positive and negative points
        all_points = [[point[0], point[1]]]
        all_labels = [1]  # 1 = foreground (wound)
        
        for neg_point in negative_points:
            all_points.append([neg_point[0], neg_point[1]])
            all_labels.append(0)  # 0 = background (healthy skin)
        
        input_points = np.array(all_points)
        input_labels = np.array(all_labels)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )
        
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx].astype(np.uint8)
        best_score = float(scores[best_idx])
        
        return SAMSegmentationResult(
            wound_mask=best_mask,
            confidence=best_score
        )
    
    def _find_healthy_skin_points(self, 
                                   image: np.ndarray,
                                   wound_point: Tuple[int, int],
                                   num_points: int = 3) -> List[Tuple[int, int]]:
        """
        Find points on healthy skin to use as negative prompts for SAM.
        Healthy skin is typically:
        - Uniform color (low texture)
        - Normal skin tone (pink/beige in LAB space)
        - Not dark (not necrotic)
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Find regions that look like healthy skin
        # Skin tones: H=0-25, S=10-170, V=60-255
        lower_skin = np.array([0, 10, 60])
        upper_skin = np.array([25, 170, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Exclude dark regions (potential wounds)
        bright_mask = (gray > 80).astype(np.uint8) * 255
        skin_mask = cv2.bitwise_and(skin_mask, bright_mask)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours of healthy skin regions
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        negative_points = []
        wound_x, wound_y = wound_point
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Too small
                continue
            
            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Make sure it's far enough from the wound point
            dist = np.sqrt((cx - wound_x)**2 + (cy - wound_y)**2)
            if dist < 50:  # Too close to wound
                continue
            
            negative_points.append((cx, cy, dist))
        
        # Sort by distance from wound and pick the closest ones
        negative_points.sort(key=lambda x: x[2])
        
        # Return just (x, y) tuples, up to num_points
        return [(p[0], p[1]) for p in negative_points[:num_points]]
    
    def segment_auto(self, 
                     image: np.ndarray,
                     use_center: bool = True) -> SAMSegmentationResult:
        """
        Automatic segmentation - tries to find wound region using color analysis
        
        Args:
            image: Input image (BGR)
            use_center: Fallback to image center if wound detection fails
            
        Returns:
            SAMSegmentationResult
        """
        h, w = image.shape[:2]
        
        # Try to find wound region using color analysis
        wound_point = self._find_wound_region(image)
        
        if wound_point is not None:
            # Use detected wound point
            return self.segment_with_point(image, wound_point, is_foreground=True)
        elif use_center:
            # Fallback to center point
            center = (w // 2, h // 2)
            return self.segment_with_point(image, center, is_foreground=True)
        else:
            # Use full image as bbox
            bbox = (0, 0, w, h)
            return self.segment_with_bbox(image, bbox)
    
    def _find_wound_region(self, image: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Find wound region using color analysis.
        Wounds typically have:
        - Darker areas (necrotic tissue)
        - Red/pink areas (granulation, blood)
        - Yellow areas (slough, pus)
        
        Returns:
            (x, y) point likely inside the wound, or None if not found
        """
        h, w = image.shape[:2]
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create masks for wound-like colors
        masks = []
        
        # 1. Dark/necrotic regions (low brightness)
        dark_mask = (gray < 60).astype(np.uint8) * 255
        masks.append(dark_mask)
        
        # 2. Red/pink regions (granulation, inflammation)
        # HSV: H=0-10 or 170-180 (red), S>50, V>50
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        masks.append(red_mask)
        
        # 3. Yellow/slough regions
        # HSV: H=20-35 (yellow), S>30, V>100
        lower_yellow = np.array([15, 30, 100])
        upper_yellow = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        masks.append(yellow_mask)
        
        # 4. Brown/necrotic regions
        # HSV: H=10-20, S>30, V<150
        lower_brown = np.array([10, 30, 30])
        upper_brown = np.array([25, 255, 150])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        masks.append(brown_mask)
        
        # Combine all wound-like masks
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for mask in masks:
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours and select the best one
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Score contours based on size and position
        best_contour = None
        best_score = 0
        
        center_x, center_y = w // 2, h // 2
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size (not too small, not too large)
            area_ratio = area / (h * w)
            if area_ratio < 0.005 or area_ratio > 0.5:  # Between 0.5% and 50% of image
                continue
            
            # Get centroid
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Score based on area and proximity to center
            dist_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            center_score = 1 - (dist_from_center / max_dist)
            
            # Prefer medium-sized contours
            size_score = min(area_ratio * 10, 1.0)  # Scales 0-10% to 0-1
            
            score = size_score * 0.6 + center_score * 0.4
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        if best_contour is None:
            return None
        
        # Return centroid of best contour
        M = cv2.moments(best_contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        return (cx, cy)
    
    def visualize(self,
                  image: np.ndarray,
                  mask: np.ndarray,
                  alpha: float = 0.5,
                  color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Create visualization overlay
        
        Args:
            image: Original image (BGR)
            mask: Binary mask
            alpha: Overlay transparency
            color: Mask color (BGR)
            
        Returns:
            Overlay image
        """
        overlay = image.copy()
        
        # Create colored mask
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = color
        
        # Blend
        overlay = cv2.addWeighted(overlay, 1, mask_colored, alpha, 0)
        
        # Draw contour
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 2)
        
        return overlay


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SAM Wound Segmentation")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="sam_result.jpg")
    parser.add_argument("--model", type=str, default="vit_b")
    
    args = parser.parse_args()
    
    # Load segmentor
    segmentor = SAMWoundSegmentor(model_type=args.model)
    
    # Load image
    image = cv2.imread(args.image)
    
    # Segment using center point
    result = segmentor.segment_auto(image)
    
    print(f"Confidence: {result.confidence:.2%}")
    
    # Visualize
    vis = segmentor.visualize(image, result.wound_mask)
    cv2.imwrite(args.output, vis)
    print(f"Saved to {args.output}")
