"""
Wound Detection Inference Module
Uses trained YOLOv8 for wound ROI extraction
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import torch
from dataclasses import dataclass

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import yolo_config, DEVICE


@dataclass
class DetectionResult:
    """Wound detection result"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int = 0
    class_name: str = "wound"
    
    @property
    def area(self) -> int:
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
    
    @property
    def center(self) -> Tuple[int, int]:
        return (
            (self.bbox[0] + self.bbox[2]) // 2,
            (self.bbox[1] + self.bbox[3]) // 2
        )


class WoundDetector:
    """
    Wound detection using YOLOv8
    Provides ROI extraction for downstream analysis modules
    """
    
    def __init__(self, 
                 weights_path: Optional[Path] = None,
                 conf_threshold: float = None,
                 iou_threshold: float = None):
        self.weights_path = weights_path or yolo_config.weights_path
        self.conf_threshold = conf_threshold or yolo_config.conf_threshold
        self.iou_threshold = iou_threshold or yolo_config.iou_threshold
        self.device = DEVICE
        
        self.model = None
        self._is_loaded = False
        
    def load(self) -> bool:
        """Load YOLO model"""
        try:
            from ultralytics import YOLO
            
            if not self.weights_path.exists():
                print(f"Weights not found: {self.weights_path}")
                print("Using pretrained YOLOv8 as fallback...")
                # Use pretrained as fallback for demo
                self.model = YOLO(f"{yolo_config.model_name}.pt")
            else:
                self.model = YOLO(str(self.weights_path))
            
            self._is_loaded = True
            print(f"Wound detector loaded on {self.device}")
            return True
            
        except ImportError:
            print("Please install ultralytics: pip install ultralytics")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def detect(self, 
               image: Union[np.ndarray, str, Path],
               return_all: bool = False) -> List[DetectionResult]:
        """
        Detect wound regions in image
        
        Args:
            image: Input image (BGR numpy array or path)
            return_all: Return all detections (vs just top one)
            
        Returns:
            List of DetectionResult objects
        """
        if not self._is_loaded:
            if not self.load():
                return []
        
        # Handle image input
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        
        if image is None:
            return []
        
        # Run inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                detections.append(DetectionResult(
                    bbox=tuple(bbox),
                    confidence=conf,
                    class_id=cls_id,
                    class_name="wound"
                ))
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        if not return_all and len(detections) > 1:
            detections = [detections[0]]
        
        return detections
    
    def extract_roi(self, 
                    image: np.ndarray,
                    detection: Optional[DetectionResult] = None,
                    padding: float = 0.1) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Extract wound region of interest
        
        Args:
            image: Input image
            detection: Detection result (if None, will detect first)
            padding: Relative padding around bbox (0.1 = 10%)
            
        Returns:
            Tuple of (cropped_image, padded_bbox)
        """
        if detection is None:
            detections = self.detect(image)
            if not detections:
                # Return full image if no detection
                h, w = image.shape[:2]
                return image, (0, 0, w, h)
            detection = detections[0]
        
        x1, y1, x2, y2 = detection.bbox
        h, w = image.shape[:2]
        
        # Add padding
        pad_w = int((x2 - x1) * padding)
        pad_h = int((y2 - y1) * padding)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)
        
        cropped = image[y1:y2, x1:x2]
        
        return cropped, (x1, y1, x2, y2)
    
    def draw_detections(self, 
                        image: np.ndarray,
                        detections: List[DetectionResult],
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 2) -> np.ndarray:
        """Draw detection boxes on image"""
        output = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(output, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(output, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        return output
    
    def process_batch(self, 
                      images: List[Union[np.ndarray, str, Path]]) -> List[List[DetectionResult]]:
        """Process multiple images"""
        if not self._is_loaded:
            if not self.load():
                return [[] for _ in images]
        
        all_detections = []
        
        for image in images:
            detections = self.detect(image, return_all=True)
            all_detections.append(detections)
        
        return all_detections
    
    def warmup(self, iterations: int = 3):
        """Warmup model for faster inference"""
        if not self._is_loaded:
            self.load()
        
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.detect(dummy)
        
        print("Warmup complete")


def detect_and_crop_dataset(input_dir: Path, output_dir: Path):
    """
    Process entire dataset: detect wounds and save cropped ROIs
    """
    from tqdm import tqdm
    
    detector = WoundDetector()
    detector.load()
    detector.warmup()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(input_dir.rglob("*.jpg")) + list(input_dir.rglob("*.png"))
    
    results = []
    
    for img_path in tqdm(image_files, desc="Detecting wounds"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        detections = detector.detect(image)
        
        if detections:
            roi, bbox = detector.extract_roi(image, detections[0])
            
            # Save cropped ROI
            rel_path = img_path.relative_to(input_dir)
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), roi)
            
            results.append({
                "image": str(img_path),
                "output": str(out_path),
                "bbox": bbox,
                "confidence": detections[0].confidence
            })
        else:
            # Copy original if no detection
            rel_path = img_path.relative_to(input_dir)
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), image)
            
            results.append({
                "image": str(img_path),
                "output": str(out_path),
                "bbox": None,
                "confidence": 0.0
            })
    
    # Save results
    import json
    with open(output_dir / "detection_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    detected = sum(1 for r in results if r["bbox"] is not None)
    print(f"\nDetection complete: {detected}/{len(results)} images with wounds")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Wound Detection")
    parser.add_argument("--image", type=str, help="Input image path")
    parser.add_argument("--dataset", type=str, help="Process dataset directory")
    parser.add_argument("--output", type=str, help="Output directory")
    
    args = parser.parse_args()
    
    detector = WoundDetector()
    
    if args.image:
        detector.load()
        image = cv2.imread(args.image)
        detections = detector.detect(image)
        
        if detections:
            print(f"Found {len(detections)} wound(s)")
            for det in detections:
                print(f"  - {det.class_name}: {det.confidence:.3f} at {det.bbox}")
            
            output = detector.draw_detections(image, detections)
            cv2.imwrite("detection_result.jpg", output)
            print("Result saved to detection_result.jpg")
        else:
            print("No wounds detected")
    
    if args.dataset and args.output:
        detect_and_crop_dataset(Path(args.dataset), Path(args.output))
