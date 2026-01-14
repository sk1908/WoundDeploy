"""
End-to-End Inference Pipeline
Orchestrates all AI models for wound analysis
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import DEVICE, OUTPUT_DIR

# Import all model modules
from models.detection.detector import WoundDetector, DetectionResult
from models.segmentation.inference import SegmentationInference, SegmentationResult
from models.depth.depth_estimator import DepthEstimator, DepthResult
from models.depth.volume_calculator import VolumeCalculator, VolumeResult
from models.classification.classifier import WoundClassifier, ClassificationResult
from models.risk.risk_model import RiskPredictor, RiskResult
from models.risk.features import FeatureExtractor, WoundFeatures


@dataclass
class PipelineResult:
    """Complete pipeline analysis result"""
    # Input
    original_image: np.ndarray
    
    # Detection
    detection: Optional[DetectionResult] = None
    roi_image: Optional[np.ndarray] = None
    roi_bbox: Optional[Tuple[int, int, int, int]] = None
    
    # Segmentation
    segmentation: Optional[SegmentationResult] = None
    
    # Depth
    depth: Optional[DepthResult] = None
    volume: Optional[VolumeResult] = None
    
    # Classification
    classification: Optional[ClassificationResult] = None
    
    # Risk
    risk: Optional[RiskResult] = None
    
    # Features
    features: Optional[WoundFeatures] = None
    
    # Timing
    inference_times: Dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "detection": {
                "bbox": self.detection.bbox if self.detection else None,
                "confidence": self.detection.confidence if self.detection else None
            },
            "segmentation": {
                "class_percentages": self.segmentation.class_percentages if self.segmentation else {}
            },
            "depth": {
                "mean_depth": self.depth.mean_depth if self.depth else None,
                "max_depth": self.depth.max_depth if self.depth else None
            },
            "volume": {
                "total_volume": self.volume.total_volume if self.volume else None,
                "tissue_volumes": self.volume.tissue_volumes if self.volume else {}
            },
            "classification": {
                "wound_type": self.classification.wound_type if self.classification else None,
                "wound_type_confidence": self.classification.wound_type_confidence if self.classification else None,
                "severity": self.classification.severity if self.classification else None,
                "severity_confidence": self.classification.severity_confidence if self.classification else None
            },
            "risk": {
                "score": self.risk.risk_score if self.risk else None,
                "level": self.risk.risk_level if self.risk else None,
                "recommendations": self.risk.recommendations if self.risk else []
            },
            "features": self.features.to_dict() if self.features else {},
            "inference_times": self.inference_times,
            "total_time": self.total_time
        }


class InferencePipeline:
    """
    Orchestrates all AI models for end-to-end wound analysis
    
    Pipeline stages:
    1. Detection: Locate wound region
    2. Segmentation: Identify tissue types
    3. Depth: Estimate wound depth
    4. Volume: Calculate wound volume
    5. Classification: Determine wound type and severity
    6. Risk: Predict non-healing risk
    """
    
    def __init__(self,
                 use_detection: bool = True,
                 use_segmentation: bool = True,
                 use_depth: bool = True,
                 use_classification: bool = True,
                 use_risk: bool = True,
                 device: Optional[torch.device] = None):
        self.device = device or DEVICE
        
        # Flags
        self.use_detection = use_detection
        self.use_segmentation = use_segmentation
        self.use_depth = use_depth
        self.use_classification = use_classification
        self.use_risk = use_risk
        
        # Models (lazy loaded)
        self._detector = None
        self._segmentor = None
        self._depth_estimator = None
        self._volume_calculator = None
        self._classifier = None
        self._risk_predictor = None
        self._feature_extractor = None
        
        self._is_loaded = False
    
    def load(self) -> bool:
        """Load all models"""
        print("Loading inference pipeline...")
        
        try:
            if self.use_detection:
                self._detector = WoundDetector()
                self._detector.load()
            
            if self.use_segmentation:
                self._segmentor = SegmentationInference()
                self._segmentor.load()
            
            if self.use_depth:
                self._depth_estimator = DepthEstimator()
                self._depth_estimator.load()
                self._volume_calculator = VolumeCalculator()
            
            if self.use_classification:
                self._classifier = WoundClassifier()
                self._classifier.load()
            
            if self.use_risk:
                self._risk_predictor = RiskPredictor()
                self._risk_predictor.load()
                self._feature_extractor = FeatureExtractor()
            
            self._is_loaded = True
            print("Pipeline loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading pipeline: {e}")
            return False
    
    def analyze(self, 
                image: np.ndarray,
                skip_detection: bool = False) -> PipelineResult:
        """
        Run full analysis pipeline on image
        
        Args:
            image: Input image (BGR)
            skip_detection: Skip detection and use full image as ROI
            
        Returns:
            PipelineResult with all analysis results
        """
        if not self._is_loaded:
            self.load()
        
        start_time = time.time()
        result = PipelineResult(original_image=image)
        
        # 1. Detection
        if self.use_detection and not skip_detection:
            t0 = time.time()
            detections = self._detector.detect(image)
            if detections:
                result.detection = detections[0]
                result.roi_image, result.roi_bbox = self._detector.extract_roi(
                    image, detections[0]
                )
            else:
                result.roi_image = image
                result.roi_bbox = (0, 0, image.shape[1], image.shape[0])
            result.inference_times["detection"] = time.time() - t0
        else:
            result.roi_image = image
            result.roi_bbox = (0, 0, image.shape[1], image.shape[0])
        
        # 2. Segmentation (with SAM refinement using detection bbox)
        if self.use_segmentation:
            t0 = time.time()
            # Pass detection bbox to help SAM identify wound boundary
            detection_bbox = result.detection.bbox if result.detection else None
            result.segmentation = self._segmentor.segment(
                result.roi_image, 
                return_visualization=True,
                detection_bbox=detection_bbox
            )
            result.inference_times["segmentation"] = time.time() - t0
        
        # 3. Depth
        if self.use_depth:
            t0 = time.time()
            mask = result.segmentation.mask if result.segmentation else None
            # Pass tissue mask for necrotic-aware depth correction
            tissue_mask = result.segmentation.mask if result.segmentation else None
            result.depth = self._depth_estimator.estimate(
                result.roi_image, 
                mask=mask,
                tissue_mask=tissue_mask
            )
            
            # 4. Volume
            if result.segmentation:
                result.volume = self._volume_calculator.calculate(
                    result.depth.depth_normalized,
                    result.segmentation.mask
                )
            result.inference_times["depth"] = time.time() - t0
        
        # 5. Classification
        if self.use_classification:
            t0 = time.time()
            result.classification = self._classifier.classify(result.roi_image)
            result.inference_times["classification"] = time.time() - t0
        
        # 6. Features and Risk
        if self.use_risk and result.segmentation:
            t0 = time.time()
            result.features = self._feature_extractor.extract(
                result.segmentation.mask,
                result.depth.depth_normalized if result.depth else None,
                result.segmentation.class_areas,
                result.volume
            )
            result.risk = self._risk_predictor.predict(features=result.features)
            result.inference_times["risk"] = time.time() - t0
        
        result.total_time = time.time() - start_time
        
        return result
    
    def create_visualization(self, result: PipelineResult) -> np.ndarray:
        """
        Create a comprehensive visualization of the analysis
        """
        # Layout: Original | ROI with detection | Segmentation | Depth
        
        h = 300
        
        # Resize images to same height
        def resize_height(img, target_h):
            scale = target_h / img.shape[0]
            new_w = int(img.shape[1] * scale)
            return cv2.resize(img, (new_w, target_h))
        
        panels = []
        
        # Original with detection box
        original = result.original_image.copy()
        if result.detection:
            x1, y1, x2, y2 = result.detection.bbox
            cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 2)
        panels.append(resize_height(original, h))
        
        # ROI
        if result.roi_image is not None:
            panels.append(resize_height(result.roi_image, h))
        
        # Segmentation overlay
        if result.segmentation and result.segmentation.overlay is not None:
            panels.append(resize_height(result.segmentation.overlay, h))
        
        # Depth map
        if result.depth is not None:
            panels.append(resize_height(result.depth.depth_colored, h))
        
        # Concatenate horizontally
        if panels:
            visualization = np.hstack(panels)
        else:
            visualization = original
        
        return visualization
    
    def create_report_image(self, result: PipelineResult) -> np.ndarray:
        """Create a report-style image with metrics"""
        
        vis = self.create_visualization(result)
        
        # Add metrics panel
        h, w = vis.shape[:2]
        panel_width = 300
        report = np.ones((h, w + panel_width, 3), dtype=np.uint8) * 255
        report[:, :w] = vis
        
        # Add text metrics
        x = w + 10
        y = 20
        line_height = 25
        
        def add_text(text, color=(0, 0, 0)):
            nonlocal y
            cv2.putText(report, text, (x, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += line_height
        
        add_text("=== WOUND ANALYSIS ===")
        y += 10
        
        if result.classification:
            add_text(f"Type: {result.classification.wound_type}")
            add_text(f"Type Conf: {result.classification.wound_type_confidence:.1%}")
            add_text(f"Severity: {result.classification.severity}")
        y += 10
        
        if result.segmentation:
            add_text("Tissue Composition:")
            for tissue, pct in result.segmentation.class_percentages.items():
                add_text(f"  {tissue}: {pct:.1f}%")
        y += 10
        
        if result.risk:
            color = {
                "low": (0, 128, 0),
                "moderate": (0, 165, 255),
                "high": (0, 128, 255),
                "critical": (0, 0, 255)
            }.get(result.risk.risk_level, (0, 0, 0))
            add_text(f"Risk Score: {result.risk.risk_score:.2f}", color)
            add_text(f"Risk Level: {result.risk.risk_level.upper()}", color)
        
        y += 10
        add_text(f"Total Time: {result.total_time:.2f}s")
        
        return report
    
    def warmup(self):
        """Warmup all models"""
        if not self._is_loaded:
            self.load()
        
        dummy = np.zeros((512, 512, 3), dtype=np.uint8)
        self.analyze(dummy, skip_detection=True)
        print("Pipeline warmup complete")
    
    def batch_analyze(self,
                      images: List[np.ndarray],
                      output_dir: Optional[Path] = None) -> List[PipelineResult]:
        """Analyze multiple images"""
        results = []
        
        for i, image in enumerate(images):
            print(f"Processing image {i+1}/{len(images)}...")
            result = self.analyze(image)
            results.append(result)
            
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save visualization
                vis = self.create_report_image(result)
                cv2.imwrite(str(output_dir / f"result_{i:03d}.jpg"), vis)
                
                # Save JSON
                with open(output_dir / f"result_{i:03d}.json", "w") as f:
                    json.dump(result.to_dict(), f, indent=2, default=str)
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Wound Analysis Pipeline")
    parser.add_argument("--image", type=str, help="Input image path")
    parser.add_argument("--output", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    if args.image:
        pipeline = InferencePipeline()
        pipeline.load()
        pipeline.warmup()
        
        image = cv2.imread(args.image)
        result = pipeline.analyze(image)
        
        print("\n=== ANALYSIS RESULTS ===")
        print(json.dumps(result.to_dict(), indent=2, default=str))
        
        # Save visualization
        vis = pipeline.create_report_image(result)
        output_path = args.output or "analysis_result.jpg"
        cv2.imwrite(output_path, vis)
        print(f"\nVisualization saved to {output_path}")


if __name__ == "__main__":
    main()
