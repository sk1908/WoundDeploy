"""
CO2Wounds-V2 Dataset Loader
Extended Chronic Wounds Dataset From Leprosy Patients
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import CO2WOUNDS_DIR


@dataclass
class CO2WoundsSample:
    """Single sample from CO2Wounds dataset"""
    image_id: int
    image: np.ndarray
    mask: Optional[np.ndarray]
    file_name: str
    bbox: Optional[List[float]]  # [x, y, width, height]
    area: Optional[float]
    

class CO2WoundsDataset(Dataset):
    """
    PyTorch Dataset for CO2Wounds-V2
    
    Loads images and their corresponding segmentation masks.
    """
    
    def __init__(
        self,
        root_dir: Optional[Path] = None,
        split: str = "all",  # "train", "val", "test", or "all"
        transform=None,
        load_masks: bool = True,
        target_size: Tuple[int, int] = (512, 512)
    ):
        """
        Args:
            root_dir: Path to CO2Wounds-V2 directory
            split: Which split to load
            transform: Optional transforms to apply
            load_masks: Whether to load segmentation masks
            target_size: Resize images to this size
        """
        self.root_dir = Path(root_dir) if root_dir else CO2WOUNDS_DIR
        self.split = split
        self.transform = transform
        self.load_masks = load_masks
        self.target_size = target_size
        
        # Directories
        self.imgs_dir = self.root_dir / "imgs"
        self.masks_dir = self.root_dir / "masks"
        self.annotations_path = self.root_dir / "annotations" / "annotations.json"
        self.split_dir = self.root_dir / "split"
        
        # Load annotations
        self.annotations = self._load_annotations()
        
        # Get image list based on split
        self.image_files = self._get_image_files()
        
        # Create lookup for annotations
        self.file_to_annotation = self._create_annotation_lookup()
        
        print(f"CO2Wounds Dataset loaded: {len(self.image_files)} images ({split} split)")
    
    def _load_annotations(self) -> Dict:
        """Load COCO-format annotations"""
        if self.annotations_path.exists():
            with open(self.annotations_path, 'r') as f:
                return json.load(f)
        return {"images": [], "annotations": []}
    
    def _get_image_files(self) -> List[str]:
        """Get list of image files based on split"""
        if self.split == "all":
            # Get all images from imgs directory
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(self.imgs_dir.glob(ext))
            return sorted([f.name for f in image_files])
        else:
            # Get from split directory
            split_subdir = self.split_dir / self.split
            if split_subdir.exists():
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    image_files.extend(split_subdir.glob(ext))
                return sorted([f.name for f in image_files])
            else:
                print(f"Warning: Split directory {split_subdir} not found, using all images")
                return self._get_image_files_all()
    
    def _get_image_files_all(self) -> List[str]:
        """Get all image files"""
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(self.imgs_dir.glob(ext))
        return sorted([f.name for f in image_files])
    
    def _create_annotation_lookup(self) -> Dict[str, Dict]:
        """Create lookup from filename to annotation data"""
        lookup = {}
        
        # Map image_id to image info
        id_to_image = {img['id']: img for img in self.annotations.get('images', [])}
        
        # Map image_id to annotations
        id_to_anns = {}
        for ann in self.annotations.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in id_to_anns:
                id_to_anns[img_id] = []
            id_to_anns[img_id].append(ann)
        
        # Create filename lookup
        for img_id, img_info in id_to_image.items():
            filename = img_info['file_name']
            lookup[filename] = {
                'image_id': img_id,
                'width': img_info.get('width'),
                'height': img_info.get('height'),
                'annotations': id_to_anns.get(img_id, [])
            }
        
        return lookup
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> CO2WoundsSample:
        """Get a single sample"""
        filename = self.image_files[idx]
        
        # Load image
        image_path = self.imgs_dir / filename
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        if self.target_size:
            image = cv2.resize(image, self.target_size)
        
        # Load mask if available
        mask = None
        if self.load_masks:
            # Try to find corresponding mask
            mask_name = Path(filename).stem + ".png"
            mask_path = self.masks_dir / mask_name
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if self.target_size:
                    mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
                # Normalize to 0-1
                mask = (mask > 127).astype(np.uint8)
        
        # Get annotation info
        ann_info = self.file_to_annotation.get(filename, {})
        image_id = ann_info.get('image_id', idx)
        
        # Get bounding box and area from first annotation
        bbox = None
        area = None
        if ann_info.get('annotations'):
            first_ann = ann_info['annotations'][0]
            bbox = first_ann.get('bbox')
            area = first_ann.get('area')
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed.get('mask', mask)
        
        return CO2WoundsSample(
            image_id=image_id,
            image=image,
            mask=mask,
            file_name=filename,
            bbox=bbox,
            area=area
        )
    
    def get_all_images_and_masks(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[str]]:
        """
        Load all images and masks into memory
        
        Returns:
            images: List of image arrays
            masks: List of mask arrays (None if mask not available)
            filenames: List of filenames
        """
        images = []
        masks = []
        filenames = []
        
        for idx in range(len(self)):
            sample = self[idx]
            images.append(sample.image)
            masks.append(sample.mask)
            filenames.append(sample.file_name)
        
        return images, masks, filenames


class CO2WoundsTorchDataset(Dataset):
    """
    PyTorch-ready version that returns tensors
    """
    
    def __init__(
        self,
        root_dir: Optional[Path] = None,
        split: str = "all",
        target_size: Tuple[int, int] = (512, 512),
        normalize: bool = True
    ):
        self.base_dataset = CO2WoundsDataset(
            root_dir=root_dir,
            split=split,
            target_size=target_size,
            load_masks=True
        )
        self.normalize = normalize
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        
        # Convert to tensor
        image = torch.from_numpy(sample.image).permute(2, 0, 1).float()
        
        # Normalize to [0, 1] or [-1, 1]
        if self.normalize:
            image = image / 255.0
        
        # Mask
        mask = None
        if sample.mask is not None:
            mask = torch.from_numpy(sample.mask).float()
        
        return {
            'image': image,
            'mask': mask,
            'filename': sample.file_name,
            'image_id': sample.image_id
        }


if __name__ == "__main__":
    # Test the dataset
    print("Testing CO2Wounds Dataset...")
    
    dataset = CO2WoundsDataset()
    print(f"Total samples: {len(dataset)}")
    
    # Load first sample
    sample = dataset[0]
    print(f"\nFirst sample:")
    print(f"  Filename: {sample.file_name}")
    print(f"  Image shape: {sample.image.shape}")
    print(f"  Mask shape: {sample.mask.shape if sample.mask is not None else 'None'}")
    print(f"  BBox: {sample.bbox}")
    print(f"  Area: {sample.area}")
    
    # Test torch dataset
    torch_dataset = CO2WoundsTorchDataset()
    batch = torch_dataset[0]
    print(f"\nTorch dataset sample:")
    print(f"  Image tensor shape: {batch['image'].shape}")
    print(f"  Mask tensor shape: {batch['mask'].shape if batch['mask'] is not None else 'None'}")
