"""
Adaptive Color-Based Tissue Correction

Corrects low-confidence model predictions using adaptive color analysis
in CIELab and HSV color spaces with white balance and Otsu/k-means thresholding.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ColorCorrectionResult:
    """Result from color-based correction"""
    corrected_mask: np.ndarray
    confidence_map: np.ndarray  # Max softmax confidence per pixel
    corrections_made: int  # Number of pixels corrected
    adaptive_thresholds: Dict[str, float]


class AdaptiveColorCorrector:
    """
    Corrects tissue segmentation using adaptive color analysis.
    
    Only corrects pixels where model confidence < threshold.
    Uses CIELab for perceptually uniform analysis and HSV for color detection.
    """
    
    # Tissue class IDs (from config)
    BACKGROUND = 0
    GRANULATION = 1
    EPITHELIUM = 2
    NECROTIC = 3
    SLOUGH = 4
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.adaptive_thresholds = {}
    
    def correct(self,
                image: np.ndarray,
                mask: np.ndarray,
                probs: np.ndarray,
                wound_mask: Optional[np.ndarray] = None) -> ColorCorrectionResult:
        """
        Correct tissue classification using adaptive color analysis.
        
        Args:
            image: Original BGR image (H, W, 3)
            mask: Model's predicted mask (H, W)
            probs: Class probabilities (C, H, W)
            wound_mask: Binary wound boundary mask (H, W), optional
            
        Returns:
            ColorCorrectionResult with corrected mask
        """
        h, w = image.shape[:2]
        
        # Get confidence map (max probability per pixel)
        # Probs shape: (C, H, W) - need to resize to match image
        if probs.shape[1:] != (h, w):
            # Resize probs to match image size
            probs_resized = np.stack([
                cv2.resize(probs[c], (w, h), interpolation=cv2.INTER_LINEAR)
                for c in range(probs.shape[0])
            ])
        else:
            probs_resized = probs
        
        confidence_map = np.max(probs_resized, axis=0)
        
        # Find low-confidence pixels
        low_conf_mask = confidence_map < self.confidence_threshold
        
        # Apply wound mask if available
        if wound_mask is not None:
            if wound_mask.shape != (h, w):
                wound_mask = cv2.resize(wound_mask.astype(np.uint8), (w, h))
            low_conf_mask = low_conf_mask & (wound_mask > 0)
        
        # White balance the image
        balanced_image = self._white_balance(image)
        
        # Convert to color spaces
        lab = cv2.cvtColor(balanced_image, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(balanced_image, cv2.COLOR_BGR2HSV)
        
        # Compute adaptive thresholds
        self._compute_adaptive_thresholds(lab, hsv, wound_mask)
        
        # Start with model's mask
        corrected_mask = mask.copy()
        corrections_made = 0
        
        # Correct low-confidence pixels
        for y in range(h):
            for x in range(w):
                if low_conf_mask[y, x]:
                    new_class = self._classify_pixel(
                        lab[y, x], 
                        hsv[y, x],
                        balanced_image[y, x]
                    )
                    if new_class is not None and new_class != corrected_mask[y, x]:
                        corrected_mask[y, x] = new_class
                        corrections_made += 1
        
        return ColorCorrectionResult(
            corrected_mask=corrected_mask,
            confidence_map=confidence_map,
            corrections_made=corrections_made,
            adaptive_thresholds=self.adaptive_thresholds
        )
    
    def correct_fast(self,
                     image: np.ndarray,
                     mask: np.ndarray,
                     probs: np.ndarray,
                     wound_mask: Optional[np.ndarray] = None) -> ColorCorrectionResult:
        """
        Fast vectorized version of color correction.
        """
        h, w = image.shape[:2]
        
        # Resize probs if needed
        if probs.shape[1:] != (h, w):
            probs_resized = np.stack([
                cv2.resize(probs[c], (w, h), interpolation=cv2.INTER_LINEAR)
                for c in range(probs.shape[0])
            ])
        else:
            probs_resized = probs
        
        confidence_map = np.max(probs_resized, axis=0)
        low_conf_mask = confidence_map < self.confidence_threshold
        
        if wound_mask is not None:
            if wound_mask.shape != (h, w):
                wound_mask = cv2.resize(wound_mask.astype(np.uint8), (w, h))
            low_conf_mask = low_conf_mask & (wound_mask > 0)
        
        # White balance and convert
        balanced_image = self._white_balance(image)
        lab = cv2.cvtColor(balanced_image, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(balanced_image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(balanced_image, cv2.COLOR_BGR2GRAY)
        
        # Compute adaptive thresholds
        self._compute_adaptive_thresholds(lab, hsv, wound_mask)
        
        corrected_mask = mask.copy()
        
        # Vectorized necrotic detection (very dark in L-channel)
        L = lab[:, :, 0]
        gray = cv2.cvtColor(balanced_image, cv2.COLOR_BGR2GRAY)
        necrotic_threshold = self.adaptive_thresholds.get('necrotic_L', 50)
        
        # LOW-CONFIDENCE correction: correct pixels where model is unsure
        is_necrotic_lowconf = (L < necrotic_threshold) & low_conf_mask
        
        # UNCONDITIONAL OVERRIDE: Dark pixels are almost certainly necrotic
        # Increased thresholds significantly (L < 50, gray < 70)
        very_dark_necrotic = (L < 50) | (gray < 70)
        
        # Also catch dark regions within wound boundary even if model is confident
        # Use higher threshold (L < 90) for wound region
        if wound_mask is not None:
            wound_region = wound_mask > 0
            # Dark regions within wound that are classified as background should be necrotic
            is_misclassified_dark = (L < 90) & wound_region & (corrected_mask == self.BACKGROUND)
        else:
            # Even without wound mask, catch all dark pixels
            is_misclassified_dark = (L < 70) & (corrected_mask == self.BACKGROUND)
        
        # Combine all necrotic detections
        is_necrotic = is_necrotic_lowconf | very_dark_necrotic | is_misclassified_dark
        
        # MORPHOLOGICAL REFINEMENT for necrotic mask
        # Dilate to capture slightly lighter neighboring necrotic pixels
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        is_necrotic_dilated = cv2.dilate(is_necrotic.astype(np.uint8), kernel_dilate, iterations=1)
        is_necrotic = is_necrotic_dilated.astype(bool)
        
        # Morphological closing to fill gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        is_necrotic_closed = cv2.morphologyEx(is_necrotic.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)
        is_necrotic = is_necrotic_closed.astype(bool)
        
        # Remove tiny isolated regions (min area = 100px)
        from scipy import ndimage
        labeled, num_features = ndimage.label(is_necrotic)
        for i in range(1, num_features + 1):
            if np.sum(labeled == i) < 100:
                is_necrotic[labeled == i] = False
        
        # ESCHAR DETECTION: Very dark, dry necrotic tissue (L < 40)
        is_eschar = (L < 40) & is_necrotic  # Subset of necrotic
        
        # Extract HSV channels for yellow-brown detection
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        # YELLOW-BROWN NECROTIC: Darker yellow-brown (H=10-30, L=40-70)
        is_yellow_brown_necrotic = (
            (H >= 10) & (H <= 30) & 
            (L >= 40) & (L <= 80) &
            (S >= 40) &
            low_conf_mask
        )
        # Add to necrotic
        is_necrotic = is_necrotic | is_yellow_brown_necrotic
        
        # Vectorized slough detection (yellow in HSV)
        slough_H_min = self.adaptive_thresholds.get('slough_H_min', 15)
        slough_H_max = self.adaptive_thresholds.get('slough_H_max', 40)
        slough_S_min = self.adaptive_thresholds.get('slough_S_min', 50)
        is_slough = (
            (H >= slough_H_min) & (H <= slough_H_max) & 
            (S >= slough_S_min) & (V > 80) &
            low_conf_mask & ~is_necrotic  # Exclude necrotic
        )
        
        # CLASS PRIORITY ENFORCEMENT: Necrotic > Slough > Granulation > Epithelium
        # Apply in priority order - higher priority classes cannot be overwritten
        
        # First, apply necrotic (highest priority)
        corrected_mask[is_necrotic] = self.NECROTIC
        
        # Then slough (but only where NOT already necrotic)
        slough_final = is_slough & (corrected_mask != self.NECROTIC)
        corrected_mask[slough_final] = self.SLOUGH
        
        # Enforce priority: anywhere the model predicted granulation but we detected necrotic/slough, keep necrotic/slough
        # (already handled by applying necrotic first)
        
        corrections_made = int(np.sum(is_necrotic) + np.sum(slough_final))
        
        return ColorCorrectionResult(
            corrected_mask=corrected_mask,
            confidence_map=confidence_map,
            corrections_made=corrections_made,
            adaptive_thresholds=self.adaptive_thresholds
        )
    
    def _white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gray-world white balance algorithm.
        Assumes average color should be gray.
        """
        # Calculate mean of each channel
        mean_b = np.mean(image[:, :, 0])
        mean_g = np.mean(image[:, :, 1])
        mean_r = np.mean(image[:, :, 2])
        
        # Calculate gray mean
        gray_mean = (mean_b + mean_g + mean_r) / 3
        
        # Scale each channel
        if mean_b > 0 and mean_g > 0 and mean_r > 0:
            balanced = image.copy().astype(np.float32)
            balanced[:, :, 0] = np.clip(balanced[:, :, 0] * (gray_mean / mean_b), 0, 255)
            balanced[:, :, 1] = np.clip(balanced[:, :, 1] * (gray_mean / mean_g), 0, 255)
            balanced[:, :, 2] = np.clip(balanced[:, :, 2] * (gray_mean / mean_r), 0, 255)
            return balanced.astype(np.uint8)
        
        return image
    
    def _compute_adaptive_thresholds(self,
                                     lab: np.ndarray,
                                     hsv: np.ndarray,
                                     wound_mask: Optional[np.ndarray] = None):
        """
        Compute adaptive thresholds using Otsu and k-means.
        """
        L = lab[:, :, 0]
        
        # Apply wound mask if available
        if wound_mask is not None and np.sum(wound_mask > 0) > 100:
            L_wound = L[wound_mask > 0]
        else:
            L_wound = L.flatten()
        
        # Otsu threshold on L-channel for necrotic detection
        try:
            otsu_thresh, _ = cv2.threshold(
                L_wound.astype(np.uint8), 
                0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            # Necrotic is darker, use lower value
            self.adaptive_thresholds['necrotic_L'] = max(30, min(otsu_thresh - 20, 80))
        except:
            self.adaptive_thresholds['necrotic_L'] = 50
        
        # K-means for dark/mid/bright clusters
        try:
            from scipy.cluster.vq import kmeans, vq
            L_sample = L_wound[::10] if len(L_wound) > 1000 else L_wound
            centroids, _ = kmeans(L_sample.astype(np.float32), 3)
            centroids = sorted(centroids)
            self.adaptive_thresholds['dark_cluster'] = centroids[0]
            self.adaptive_thresholds['mid_cluster'] = centroids[1]
            self.adaptive_thresholds['bright_cluster'] = centroids[2]
        except:
            self.adaptive_thresholds['dark_cluster'] = 40
            self.adaptive_thresholds['mid_cluster'] = 100
            self.adaptive_thresholds['bright_cluster'] = 180
        
        # Slough thresholds (yellow in HSV)
        self.adaptive_thresholds['slough_H_min'] = 15
        self.adaptive_thresholds['slough_H_max'] = 40
        self.adaptive_thresholds['slough_S_min'] = 50
    
    def _classify_pixel(self,
                        lab_pixel: np.ndarray,
                        hsv_pixel: np.ndarray,
                        bgr_pixel: np.ndarray) -> Optional[int]:
        """
        Classify single pixel based on color.
        Returns tissue class or None if uncertain.
        """
        L, a, b_val = lab_pixel
        H, S, V = hsv_pixel
        
        necrotic_thresh = self.adaptive_thresholds.get('necrotic_L', 50)
        
        # Necrotic: very dark
        if L < necrotic_thresh:
            return self.NECROTIC
        
        # Slough: yellow/pale
        if 15 <= H <= 40 and S >= 50 and V > 80:
            return self.SLOUGH
        
        # Keep model's prediction for other cases
        return None


if __name__ == "__main__":
    # Test with dummy data
    corrector = AdaptiveColorCorrector(confidence_threshold=0.7)
    
    # Create test image
    image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    probs = np.random.rand(5, 64, 64).astype(np.float32)
    
    result = corrector.correct_fast(image, mask, probs)
    print(f"Corrections made: {result.corrections_made}")
    print(f"Adaptive thresholds: {result.adaptive_thresholds}")
