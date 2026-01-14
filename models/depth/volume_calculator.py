"""
Volume Calculator Module
Combines depth and segmentation for volumetric analysis
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import TISSUE_CLASSES


@dataclass
class VolumeResult:
    """Volume calculation result"""
    total_volume: float  # Total wound volume (arbitrary units)
    tissue_volumes: Dict[str, float]  # Volume per tissue type
    tissue_areas: Dict[str, float]  # Area per tissue type (pixels)
    mean_depth: float
    max_depth: float
    surface_area: float


class VolumeCalculator:
    """
    Calculate wound volume using depth and segmentation masks
    
    Volume is estimated as the integral of depth over the wound area.
    Since depth is relative (not metric), volumes are in arbitrary units
    unless calibration is provided.
    """
    
    def __init__(self,
                 pixel_size_mm: Optional[float] = None,
                 depth_scale_mm: Optional[float] = None):
        """
        Args:
            pixel_size_mm: Physical size of one pixel in mm (if known)
            depth_scale_mm: Scale factor to convert relative depth to mm
        """
        self.pixel_size_mm = pixel_size_mm
        self.depth_scale_mm = depth_scale_mm
        
    def calculate(self,
                  depth_map: np.ndarray,
                  segmentation_mask: np.ndarray,
                  background_class: int = 0) -> VolumeResult:
        """
        Calculate volume from depth and segmentation
        
        Args:
            depth_map: Depth map (H, W), higher = deeper
            segmentation_mask: Tissue class mask (H, W)
            background_class: Class ID to exclude from volume calculation
            
        Returns:
            VolumeResult with volume metrics
        """
        # Ensure same size
        if depth_map.shape != segmentation_mask.shape:
            import cv2
            segmentation_mask = cv2.resize(
                segmentation_mask.astype(np.uint8),
                (depth_map.shape[1], depth_map.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        # Normalize depth if needed
        if depth_map.max() > 1:
            depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        else:
            depth_normalized = depth_map
        
        # Create wound mask (non-background, non-epithelium)
        wound_mask = (segmentation_mask != background_class) & (segmentation_mask != 4)
        
        # Calculate pixel area
        pixel_area = self.pixel_size_mm ** 2 if self.pixel_size_mm else 1.0
        
        # Calculate depth scale
        depth_scale = self.depth_scale_mm if self.depth_scale_mm else 1.0
        
        # Calculate tissue-specific volumes and areas
        tissue_volumes = {}
        tissue_areas = {}
        
        for class_id, class_name in TISSUE_CLASSES.items():
            if class_id == background_class:
                continue
            
            class_mask = segmentation_mask == class_id
            class_depth = depth_normalized * class_mask
            
            # Area is count of pixels
            area = np.sum(class_mask)
            tissue_areas[class_name] = float(area * pixel_area)
            
            # Volume is sum of depth values * area per pixel * depth scale
            volume = np.sum(class_depth) * pixel_area * depth_scale
            tissue_volumes[class_name] = float(volume)
        
        # Total wound volume
        wound_depth = depth_normalized * wound_mask
        total_volume = np.sum(wound_depth) * pixel_area * depth_scale
        
        # Wound statistics
        wound_depths = depth_normalized[wound_mask]
        mean_depth = float(np.mean(wound_depths)) if len(wound_depths) > 0 else 0.0
        max_depth = float(np.max(wound_depths)) if len(wound_depths) > 0 else 0.0
        
        # Surface area
        surface_area = float(np.sum(wound_mask) * pixel_area)
        
        return VolumeResult(
            total_volume=total_volume,
            tissue_volumes=tissue_volumes,
            tissue_areas=tissue_areas,
            mean_depth=mean_depth * depth_scale,
            max_depth=max_depth * depth_scale,
            surface_area=surface_area
        )
    
    def calculate_from_contour(self,
                                depth_map: np.ndarray,
                                contour: np.ndarray) -> float:
        """
        Calculate volume from a wound contour
        
        Args:
            depth_map: Depth map
            contour: OpenCV contour of wound boundary
            
        Returns:
            Volume estimate
        """
        import cv2
        
        # Create mask from contour
        mask = np.zeros(depth_map.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, -1)
        
        # Calculate volume
        masked_depth = depth_map * mask
        
        pixel_area = self.pixel_size_mm ** 2 if self.pixel_size_mm else 1.0
        depth_scale = self.depth_scale_mm if self.depth_scale_mm else 1.0
        
        volume = np.sum(masked_depth) * pixel_area * depth_scale
        
        return float(volume)
    
    def estimate_healing_progress(self,
                                   current_volume: VolumeResult,
                                   previous_volume: Optional[VolumeResult] = None) -> Dict:
        """
        Estimate healing progress based on volume changes
        
        Returns:
            Dictionary with healing metrics
        """
        result = {
            "total_volume": current_volume.total_volume,
            "surface_area": current_volume.surface_area,
            "mean_depth": current_volume.mean_depth,
        }
        
        # Tissue composition
        total_tissue = sum(current_volume.tissue_volumes.values())
        if total_tissue > 0:
            for tissue, vol in current_volume.tissue_volumes.items():
                result[f"{tissue}_ratio"] = vol / total_tissue
        
        # If we have previous data, calculate changes
        if previous_volume:
            volume_change = current_volume.total_volume - previous_volume.total_volume
            area_change = current_volume.surface_area - previous_volume.surface_area
            
            result["volume_change"] = volume_change
            result["area_change"] = area_change
            
            # Positive change = healing (volume decreasing)
            if previous_volume.total_volume > 0:
                result["volume_reduction_pct"] = (-volume_change / previous_volume.total_volume) * 100
            
            if previous_volume.surface_area > 0:
                result["area_reduction_pct"] = (-area_change / previous_volume.surface_area) * 100
            
            # Simple healing score based on volume reduction
            if volume_change < 0:  # Healing
                result["healing_indicator"] = "improving"
            elif volume_change > 0:  # Worsening
                result["healing_indicator"] = "worsening"
            else:
                result["healing_indicator"] = "stable"
        
        return result
    
    def calibrate_from_reference(self,
                                  image: np.ndarray,
                                  reference_length_mm: float,
                                  reference_length_px: int) -> Tuple[float, float]:
        """
        Calibrate pixel size from a reference object in the image
        
        Args:
            image: Input image
            reference_length_mm: Known length of reference object (e.g., ruler) in mm
            reference_length_px: Length of reference object in pixels
            
        Returns:
            pixel_size_mm, depth_scale_mm
        """
        pixel_size = reference_length_mm / reference_length_px
        
        # Estimate depth scale (rough heuristic: assume max depth is ~10mm for typical wound)
        # This should be refined based on actual wound type and clinical data
        estimated_max_depth_mm = 10.0
        
        self.pixel_size_mm = pixel_size
        self.depth_scale_mm = estimated_max_depth_mm
        
        print(f"Calibration:")
        print(f"  Pixel size: {pixel_size:.4f} mm/pixel")
        print(f"  Depth scale: {estimated_max_depth_mm} mm (estimate)")
        
        return pixel_size, estimated_max_depth_mm


def calculate_wound_metrics(depth_map: np.ndarray,
                            segmentation_mask: np.ndarray,
                            pixel_size_mm: Optional[float] = None) -> Dict:
    """
    Convenience function to calculate all wound metrics
    """
    calculator = VolumeCalculator(
        pixel_size_mm=pixel_size_mm,
        depth_scale_mm=10.0 if pixel_size_mm else None  # Assume 10mm max depth
    )
    
    volume_result = calculator.calculate(depth_map, segmentation_mask)
    healing_progress = calculator.estimate_healing_progress(volume_result)
    
    return {
        "volume": volume_result,
        "healing": healing_progress
    }


if __name__ == "__main__":
    # Test with random data
    depth = np.random.rand(256, 256).astype(np.float32)
    mask = np.random.randint(0, 5, (256, 256), dtype=np.uint8)
    
    calculator = VolumeCalculator()
    result = calculator.calculate(depth, mask)
    
    print("Volume Result:")
    print(f"  Total Volume: {result.total_volume:.2f}")
    print(f"  Surface Area: {result.surface_area:.2f}")
    print(f"  Mean Depth: {result.mean_depth:.4f}")
    print(f"  Max Depth: {result.max_depth:.4f}")
    print("\nTissue Volumes:")
    for tissue, vol in result.tissue_volumes.items():
        print(f"  {tissue}: {vol:.2f}")
