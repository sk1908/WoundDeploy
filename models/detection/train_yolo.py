"""
YOLO Wound Detection Training Module
Uses YOLOv8 for wound region detection
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, List
import yaml
import json

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import (
    UNIFIED_DIR, WEIGHTS_DIR, OUTPUT_DIR, PROCESSED_DIR,
    yolo_config, training_config, DEVICE
)


class YOLOTrainer:
    """
    YOLOv8 training wrapper for wound detection
    """
    
    def __init__(self, 
                 data_dir: Optional[Path] = None,
                 output_dir: Optional[Path] = None):
        self.data_dir = data_dir or UNIFIED_DIR
        self.output_dir = output_dir or OUTPUT_DIR / "yolo_training"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = yolo_config
        self.model = None
        
    def prepare_yolo_dataset(self):
        """
        Prepare dataset in YOLO format
        
        Creates:
        - images/train, images/val, images/test
        - labels/train, labels/val, labels/test
        - data.yaml configuration file
        """
        yolo_dir = PROCESSED_DIR / "yolo_dataset"
        
        for split in ["train", "val", "test"]:
            (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        # Load metadata
        metadata_path = self.data_dir / "annotations" / "metadata.json"
        if not metadata_path.exists():
            print(f"Metadata not found: {metadata_path}")
            print("Please run data harmonization first.")
            return None
            
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Process each image
        for img_data in metadata:
            split = img_data.get("split", "train")
            unified_path = Path(img_data.get("unified_path", ""))
            
            if not unified_path.exists():
                continue
            
            # Copy image
            dst_img = yolo_dir / "images" / split / unified_path.name
            if not dst_img.exists():
                shutil.copy2(unified_path, dst_img)
            
            # Create label file (full-image bounding box as placeholder)
            # Format: class_id x_center y_center width height
            label_path = yolo_dir / "labels" / split / f"{unified_path.stem}.txt"
            
            # For wound detection, we use class 0 (wound) with full image bbox
            # These are placeholders - ideally you'd have actual annotations
            with open(label_path, "w") as f:
                # Assuming wound is roughly centered and covers most of image
                f.write("0 0.5 0.5 0.8 0.8\n")
        
        # Create data.yaml
        data_yaml = {
            "path": str(yolo_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {
                0: "wound"
            },
            "nc": 1  # number of classes
        }
        
        yaml_path = yolo_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"YOLO dataset prepared at: {yolo_dir}")
        print(f"Data config: {yaml_path}")
        
        return yaml_path
    
    def train(self, data_yaml: Optional[Path] = None, resume: bool = False):
        """
        Train YOLOv8 model
        
        Args:
            data_yaml: Path to data.yaml config file
            resume: Resume from last checkpoint
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            print("Please install ultralytics: pip install ultralytics")
            return None
        
        # Prepare dataset if needed
        if data_yaml is None:
            data_yaml = self.prepare_yolo_dataset()
            if data_yaml is None:
                return None
        
        # Load or create model
        if resume and self.config.weights_path.exists():
            print(f"Resuming from {self.config.weights_path}")
            self.model = YOLO(str(self.config.weights_path))
        else:
            print(f"Loading pretrained {self.config.model_name}")
            self.model = YOLO(f"{self.config.model_name}.pt")
        
        # Training parameters
        train_args = {
            "data": str(data_yaml),
            "epochs": self.config.epochs,
            "imgsz": self.config.img_size,
            "batch": self.config.batch_size,
            "lr0": self.config.lr,
            "patience": self.config.patience,
            "project": str(self.output_dir),
            "name": "wound_detection",
            "exist_ok": True,
            "device": "0" if str(DEVICE) == "cuda" else "cpu",
            "workers": 4,
            "amp": training_config.use_amp,
            "save": True,
            "save_period": training_config.save_interval,
            "plots": True,
            "verbose": True,
        }
        
        print("\nStarting YOLO training...")
        print(f"Config: {train_args}")
        
        # Train
        results = self.model.train(**train_args)
        
        # Save best weights
        best_weights = self.output_dir / "wound_detection" / "weights" / "best.pt"
        if best_weights.exists():
            shutil.copy2(best_weights, self.config.weights_path)
            print(f"\nBest weights saved to: {self.config.weights_path}")
        
        return results
    
    def validate(self, data_yaml: Optional[Path] = None):
        """Validate model on validation set"""
        try:
            from ultralytics import YOLO
        except ImportError:
            print("Please install ultralytics: pip install ultralytics")
            return None
        
        if self.model is None:
            if self.config.weights_path.exists():
                self.model = YOLO(str(self.config.weights_path))
            else:
                print("No trained model found")
                return None
        
        if data_yaml is None:
            data_yaml = PROCESSED_DIR / "yolo_dataset" / "data.yaml"
        
        results = self.model.val(data=str(data_yaml))
        return results
    
    def export(self, format: str = "onnx"):
        """Export model to different formats for deployment"""
        try:
            from ultralytics import YOLO
        except ImportError:
            print("Please install ultralytics: pip install ultralytics")
            return None
        
        if self.model is None:
            if self.config.weights_path.exists():
                self.model = YOLO(str(self.config.weights_path))
            else:
                print("No trained model found")
                return None
        
        export_path = self.model.export(format=format)
        print(f"Model exported to: {export_path}")
        return export_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO Wound Detection Training")
    parser.add_argument("--prepare", action="store_true", help="Prepare dataset")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--validate", action="store_true", help="Validate model")
    parser.add_argument("--export", type=str, help="Export format (onnx, torchscript)")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    
    args = parser.parse_args()
    
    trainer = YOLOTrainer()
    
    if args.prepare:
        trainer.prepare_yolo_dataset()
    
    if args.train:
        trainer.train(resume=args.resume)
    
    if args.validate:
        trainer.validate()
    
    if args.export:
        trainer.export(format=args.export)


if __name__ == "__main__":
    main()
