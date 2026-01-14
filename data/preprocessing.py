"""
Image Preprocessing Module
Handles image resizing, normalization, and augmentation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    yolo_config, segmentation_config, classification_config,
    augmentation_config
)


class ImagePreprocessor:
    """
    Unified image preprocessing pipeline for all models
    """
    
    # ImageNet normalization (used by most pretrained models)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(self):
        self._setup_transforms()
        
    def _setup_transforms(self):
        """Setup augmentation pipelines for different purposes"""
        cfg = augmentation_config
        
        # Training augmentations (heavy)
        self.train_augment = A.Compose([
            A.HorizontalFlip(p=cfg.horizontal_flip),
            A.VerticalFlip(p=cfg.vertical_flip),
            A.Rotate(limit=cfg.rotation_limit, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=cfg.brightness_limit,
                contrast_limit=cfg.contrast_limit,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=cfg.hue_shift_limit,
                sat_shift_limit=int(cfg.saturation_limit * 100),
                val_shift_limit=20,
                p=0.3
            ),
            A.OneOf([
                A.GaussianBlur(blur_limit=cfg.blur_limit, p=1.0),
                A.GaussNoise(var_limit=cfg.noise_var_limit, p=1.0),
            ], p=0.2),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.CoarseDropout(
                max_holes=8, max_height=32, max_width=32,
                min_holes=1, min_height=8, min_width=8,
                fill_value=0, p=0.2
            ),
        ])
        
        # Validation/Test transforms (no augmentation)
        self.val_transform = A.Compose([])
        
    def resize_for_yolo(self, image: np.ndarray) -> np.ndarray:
        """Resize image for YOLO detection"""
        target_size = yolo_config.img_size
        return cv2.resize(image, (target_size, target_size))
    
    def resize_for_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Resize image for segmentation"""
        h, w = segmentation_config.img_size
        return cv2.resize(image, (w, h))
    
    def resize_for_classification(self, image: np.ndarray) -> np.ndarray:
        """Resize image for classification"""
        h, w = classification_config.img_size
        return cv2.resize(image, (w, h))
    
    def normalize(self, image: np.ndarray, 
                  mean: List[float] = None,
                  std: List[float] = None) -> np.ndarray:
        """Normalize image using ImageNet statistics"""
        mean = mean or self.IMAGENET_MEAN
        std = std or self.IMAGENET_STD
        
        # Convert to float [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize
        image = (image - np.array(mean)) / np.array(std)
        
        return image
    
    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor (CHW format)"""
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)
        return torch.from_numpy(image.transpose(2, 0, 1)).float()
    
    def augment(self, image: np.ndarray, mask: np.ndarray = None, 
                is_training: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply augmentations to image and optionally mask"""
        transform = self.train_augment if is_training else self.val_transform
        
        if mask is not None:
            result = transform(image=image, mask=mask)
            return result["image"], result["mask"]
        else:
            result = transform(image=image)
            return result["image"], None
    
    def preprocess_for_model(self, image: np.ndarray, 
                             model_type: str = "classification",
                             is_training: bool = False) -> torch.Tensor:
        """
        Full preprocessing pipeline for a specific model type
        
        Args:
            image: Input image (BGR or RGB, HWC format)
            model_type: One of 'yolo', 'segmentation', 'classification', 'depth'
            is_training: Whether to apply augmentations
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentations if training
        if is_training:
            image, _ = self.augment(image, is_training=True)
        
        # Resize based on model type
        if model_type == "yolo":
            image = self.resize_for_yolo(image)
        elif model_type == "segmentation":
            image = self.resize_for_segmentation(image)
        elif model_type == "classification":
            image = self.resize_for_classification(image)
        elif model_type == "depth":
            # Depth Anything uses its own preprocessing
            image = cv2.resize(image, (518, 518))
        
        # Normalize
        image = self.normalize(image)
        
        # Convert to tensor
        tensor = self.to_tensor(image)
        
        return tensor


def get_yolo_transforms(is_training: bool = True) -> A.Compose:
    """Get YOLO-specific transforms with bounding box support"""
    cfg = augmentation_config
    
    if is_training:
        return A.Compose([
            A.HorizontalFlip(p=cfg.horizontal_flip),
            A.VerticalFlip(p=cfg.vertical_flip),
            A.Rotate(limit=cfg.rotation_limit, border_mode=cv2.BORDER_CONSTANT, p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=cfg.brightness_limit,
                contrast_limit=cfg.contrast_limit,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=cfg.hue_shift_limit,
                sat_shift_limit=int(cfg.saturation_limit * 100),
                p=0.3
            ),
            A.Resize(yolo_config.img_size, yolo_config.img_size),
        ], bbox_params=A.BboxParams(
            format='yolo', 
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    else:
        return A.Compose([
            A.Resize(yolo_config.img_size, yolo_config.img_size),
        ], bbox_params=A.BboxParams(
            format='yolo', 
            label_fields=['class_labels']
        ))


def get_segmentation_transforms(is_training: bool = True) -> A.Compose:
    """Get segmentation-specific transforms with mask support"""
    cfg = augmentation_config
    h, w = segmentation_config.img_size
    
    if is_training:
        return A.Compose([
            A.Resize(h, w),
            A.HorizontalFlip(p=cfg.horizontal_flip),
            A.VerticalFlip(p=cfg.vertical_flip),
            A.Rotate(limit=cfg.rotation_limit, border_mode=cv2.BORDER_CONSTANT, p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=cfg.brightness_limit,
                contrast_limit=cfg.contrast_limit,
                p=0.5
            ),
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.Normalize(
                mean=ImagePreprocessor.IMAGENET_MEAN,
                std=ImagePreprocessor.IMAGENET_STD
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(h, w),
            A.Normalize(
                mean=ImagePreprocessor.IMAGENET_MEAN,
                std=ImagePreprocessor.IMAGENET_STD
            ),
            ToTensorV2(),
        ])


def get_classification_transforms(is_training: bool = True) -> A.Compose:
    """Get classification-specific transforms"""
    cfg = augmentation_config
    h, w = classification_config.img_size
    
    if is_training:
        return A.Compose([
            A.Resize(h, w),
            A.HorizontalFlip(p=cfg.horizontal_flip),
            A.VerticalFlip(p=cfg.vertical_flip),
            A.Rotate(limit=cfg.rotation_limit, p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=cfg.brightness_limit,
                contrast_limit=cfg.contrast_limit,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=cfg.hue_shift_limit,
                sat_shift_limit=int(cfg.saturation_limit * 100),
                p=0.3
            ),
            A.CoarseDropout(
                max_holes=8, max_height=32, max_width=32,
                fill_value=0, p=0.2
            ),
            A.Normalize(
                mean=ImagePreprocessor.IMAGENET_MEAN,
                std=ImagePreprocessor.IMAGENET_STD
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(h, w),
            A.Normalize(
                mean=ImagePreprocessor.IMAGENET_MEAN,
                std=ImagePreprocessor.IMAGENET_STD
            ),
            ToTensorV2(),
        ])


def convert_to_yolo_format(bbox: Tuple[int, int, int, int], 
                           img_width: int, img_height: int) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from [x1, y1, x2, y2] to YOLO format [x_center, y_center, width, height]
    All values normalized to [0, 1]
    """
    x1, y1, x2, y2 = bbox
    
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    return (x_center, y_center, width, height)


def create_yolo_annotations(metadata_path: Path, output_dir: Path):
    """
    Create YOLO format annotation files from metadata
    
    Note: AZH dataset has location codes but not actual bounding boxes.
    This function creates placeholder annotations that need manual refinement.
    """
    import json
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_data in metadata:
        if img_data["source"] == "azh" and img_data["bbox_label"] > 0:
            # Create YOLO annotation file
            img_path = Path(img_data["unified_path"])
            label_path = output_dir / img_data["split"] / img_data["class"] / f"{img_path.stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            
            # For now, create a full-image bounding box as placeholder
            # These should be refined with actual annotations
            with open(label_path, "w") as f:
                # class_id x_center y_center width height
                # class_id = 0 for wound (binary detection)
                f.write("0 0.5 0.5 0.9 0.9\n")
                
    print(f"Created YOLO annotations in {output_dir}")
    print("Note: These are placeholder annotations - please refine with actual bounding boxes")


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = ImagePreprocessor()
    
    # Create a test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test different preprocessing pipelines
    for model_type in ["yolo", "segmentation", "classification", "depth"]:
        tensor = preprocessor.preprocess_for_model(test_image, model_type=model_type)
        print(f"{model_type}: {tensor.shape}")
