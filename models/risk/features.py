"""
Feature Extraction Module
Extracts features from wound analysis for risk prediction
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import TISSUE_CLASSES


@dataclass
class WoundFeatures:
    """Extracted wound features for risk prediction"""
    # Tissue fractions
    granulation_ratio: float
    slough_ratio: float
    necrotic_ratio: float
    epithelium_ratio: float
    
    # Geometric features
    wound_area: float
    wound_perimeter: float
    circularity: float
    
    # Depth features
    mean_depth: float
    max_depth: float
    depth_variance: float
    
    # Volume features
    total_volume: float
    necrotic_volume: float
    
    # Derived
    healthy_ratio: float  # granulation / (granulation + slough + necrotic)
    necrotic_burden: float  # (necrotic + slough) / total_wound
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input"""
        return np.array([
            self.granulation_ratio,
            self.slough_ratio,
            self.necrotic_ratio,
            self.epithelium_ratio,
            self.wound_area,
            self.wound_perimeter,
            self.circularity,
            self.mean_depth,
            self.max_depth,
            self.depth_variance,
            self.total_volume,
            self.necrotic_volume,
            self.healthy_ratio,
            self.necrotic_burden
        ], dtype=np.float32)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "granulation_ratio": self.granulation_ratio,
            "slough_ratio": self.slough_ratio,
            "necrotic_ratio": self.necrotic_ratio,
            "epithelium_ratio": self.epithelium_ratio,
            "wound_area": self.wound_area,
            "wound_perimeter": self.wound_perimeter,
            "circularity": self.circularity,
            "mean_depth": self.mean_depth,
            "max_depth": self.max_depth,
            "depth_variance": self.depth_variance,
            "total_volume": self.total_volume,
            "necrotic_volume": self.necrotic_volume,
            "healthy_ratio": self.healthy_ratio,
            "necrotic_burden": self.necrotic_burden
        }


class FeatureExtractor:
    """
    Extracts wound features from segmentation, depth, and classification results
    """
    
    def __init__(self):
        self.tissue_classes = TISSUE_CLASSES
    
    def extract(self,
                segmentation_mask: np.ndarray,
                depth_map: Optional[np.ndarray] = None,
                class_areas: Optional[Dict[str, int]] = None,
                volume_result: Optional[object] = None) -> WoundFeatures:
        """
        Extract comprehensive wound features
        
        Args:
            segmentation_mask: Tissue type mask (H, W)
            depth_map: Depth map (H, W) - optional
            class_areas: Dictionary of class pixel counts
            volume_result: VolumeResult object
            
        Returns:
            WoundFeatures object
        """
        # Tissue ratios
        if class_areas:
            total = sum(class_areas.values())
            granulation_ratio = class_areas.get("granulation", 0) / max(total, 1)
            slough_ratio = class_areas.get("slough", 0) / max(total, 1)
            necrotic_ratio = class_areas.get("necrotic", 0) / max(total, 1)
            epithelium_ratio = class_areas.get("epithelium", 0) / max(total, 1)
        else:
            granulation_ratio, slough_ratio, necrotic_ratio, epithelium_ratio = \
                self._compute_tissue_ratios(segmentation_mask)
        
        # Geometric features
        wound_area, wound_perimeter, circularity = self._compute_geometry(segmentation_mask)
        
        # Depth features
        if depth_map is not None:
            mean_depth, max_depth, depth_variance = self._compute_depth_stats(
                depth_map, segmentation_mask
            )
        else:
            mean_depth, max_depth, depth_variance = 0.0, 0.0, 0.0
        
        # Volume features
        if volume_result:
            total_volume = volume_result.total_volume
            necrotic_volume = volume_result.tissue_volumes.get("necrotic", 0)
        else:
            total_volume = wound_area * mean_depth if mean_depth > 0 else wound_area
            necrotic_volume = necrotic_ratio * total_volume
        
        # Derived features
        wound_tissue = granulation_ratio + slough_ratio + necrotic_ratio
        healthy_ratio = granulation_ratio / max(wound_tissue, 0.001)
        necrotic_burden = (necrotic_ratio + slough_ratio) / max(wound_tissue, 0.001)
        
        return WoundFeatures(
            granulation_ratio=granulation_ratio,
            slough_ratio=slough_ratio,
            necrotic_ratio=necrotic_ratio,
            epithelium_ratio=epithelium_ratio,
            wound_area=wound_area,
            wound_perimeter=wound_perimeter,
            circularity=circularity,
            mean_depth=mean_depth,
            max_depth=max_depth,
            depth_variance=depth_variance,
            total_volume=total_volume,
            necrotic_volume=necrotic_volume,
            healthy_ratio=healthy_ratio,
            necrotic_burden=necrotic_burden
        )
    
    def _compute_tissue_ratios(self, mask: np.ndarray) -> Tuple[float, float, float, float]:
        """Compute tissue ratios from mask"""
        total = mask.size
        
        granulation = np.sum(mask == 1) / total
        slough = np.sum(mask == 2) / total
        necrotic = np.sum(mask == 3) / total
        epithelium = np.sum(mask == 4) / total
        
        return granulation, slough, necrotic, epithelium
    
    def _compute_geometry(self, mask: np.ndarray) -> Tuple[float, float, float]:
        """Compute geometric features"""
        import cv2
        
        # Create wound binary mask (exclude background and epithelium)
        wound_mask = ((mask > 0) & (mask < 4)).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(wound_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0, 0.0, 0.0
        
        # Use largest contour
        largest = max(contours, key=cv2.contourArea)
        
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        
        # Circularity: 4*pi*area / perimeter^2
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
        else:
            circularity = 0.0
        
        return float(area), float(perimeter), float(circularity)
    
    def _compute_depth_stats(self, 
                              depth_map: np.ndarray,
                              mask: np.ndarray) -> Tuple[float, float, float]:
        """Compute depth statistics within wound region"""
        # Wound region mask
        wound_mask = (mask > 0) & (mask < 4)
        
        if not wound_mask.any():
            return 0.0, 0.0, 0.0
        
        wound_depths = depth_map[wound_mask]
        
        mean_depth = float(np.mean(wound_depths))
        max_depth = float(np.max(wound_depths))
        depth_variance = float(np.var(wound_depths))
        
        return mean_depth, max_depth, depth_variance
    
    def extract_temporal_features(self,
                                   current_features: WoundFeatures,
                                   previous_features: Optional[WoundFeatures] = None) -> Dict:
        """
        Extract temporal (change) features
        
        Args:
            current_features: Current wound features
            previous_features: Previous timepoint features
            
        Returns:
            Dictionary of change features
        """
        if previous_features is None:
            return {
                "area_change": 0.0,
                "volume_change": 0.0,
                "healthy_ratio_change": 0.0,
                "necrotic_burden_change": 0.0
            }
        
        return {
            "area_change": current_features.wound_area - previous_features.wound_area,
            "volume_change": current_features.total_volume - previous_features.total_volume,
            "healthy_ratio_change": current_features.healthy_ratio - previous_features.healthy_ratio,
            "necrotic_burden_change": current_features.necrotic_burden - previous_features.necrotic_burden
        }


if __name__ == "__main__":
    # Test feature extraction
    import cv2
    
    # Create dummy data
    mask = np.random.randint(0, 5, (256, 256), dtype=np.uint8)
    depth = np.random.rand(256, 256).astype(np.float32)
    
    extractor = FeatureExtractor()
    features = extractor.extract(mask, depth)
    
    print("Extracted Features:")
    for key, value in features.to_dict().items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nFeature array shape: {features.to_array().shape}")
