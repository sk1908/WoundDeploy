"""
Data Harmonization Module
Merges AZH and Medetec datasets into a unified structure
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
import random

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    AZH_DIR, MEDETEC_DIR, UNIFIED_DIR, PROCESSED_DIR,
    FOLDER_TO_CLASS, MEDETEC_TO_UNIFIED, WOUND_TYPE_CLASSES,
    training_config
)


class DataHarmonizer:
    """
    Harmonizes AZH and Medetec wound image datasets into a unified structure.
    
    Output structure:
    unified/
    ├── train/
    │   ├── diabetic/
    │   ├── venous/
    │   ├── pressure/
    │   ├── surgical/
    │   ├── normal/     # N = Normal/Healthy skin (not wounds)
    │   └── other/
    ├── val/
    │   └── ...
    └── test/
        └── ...
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.azh_dir = AZH_DIR
        self.medetec_dir = MEDETEC_DIR
        self.output_dir = output_dir or UNIFIED_DIR
        
        # Class names for unified dataset
        # NOTE: 'necrotic' was WRONG - AZH 'N' = Normal (healthy skin), not necrotic
        self.classes = ["background", "diabetic", "venous", "pressure", "surgical", "normal", "other"]
        
        # Statistics
        self.stats = {
            "azh_images": 0,
            "medetec_images": 0,
            "total_images": 0,
            "class_distribution": {},
            "split_distribution": {"train": 0, "val": 0, "test": 0}
        }
        
    def _create_directory_structure(self):
        """Create the unified directory structure"""
        for split in ["train", "val", "test"]:
            for cls in self.classes:
                (self.output_dir / split / cls).mkdir(parents=True, exist_ok=True)
        
        # Also create directories for annotations
        (self.output_dir / "annotations").mkdir(parents=True, exist_ok=True)
        
    def _parse_azh_annotations(self) -> pd.DataFrame:
        """Parse AZH CSV annotations"""
        train_csv = self.azh_dir / "Train" / "Train" / "wound_locations_Labels_AZH_Train.csv"
        test_csv = self.azh_dir / "Test" / "Test" / "wound_locations_Labels_AZH_Test.csv"
        
        dfs = []
        
        if train_csv.exists():
            df_train = pd.read_csv(train_csv)
            df_train["original_split"] = "train"
            dfs.append(df_train)
            
        if test_csv.exists():
            df_test = pd.read_csv(test_csv)
            df_test["original_split"] = "test"
            dfs.append(df_test)
            
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
    
    def _get_azh_class_from_path(self, image_path: str) -> str:
        """Extract class from AZH image path (e.g., 'D\\1_0' -> 'diabetic')"""
        folder = image_path.split("\\")[0] if "\\" in image_path else image_path.split("/")[0]
        
        folder_to_name = {
            "BG": "background",
            "D": "diabetic",
            "N": "normal",  # Normal/Healthy skin - NOT necrotic!
            "P": "pressure",
            "S": "surgical",
            "V": "venous"
        }
        
        return folder_to_name.get(folder, "other")
    
    def _process_azh_dataset(self) -> List[Dict]:
        """Process AZH dataset and return image metadata"""
        images = []
        
        # Parse annotations
        annotations_df = self._parse_azh_annotations()
        
        # Create lookup for bounding boxes
        bbox_lookup = {}
        if not annotations_df.empty:
            for _, row in annotations_df.iterrows():
                idx = row["index"]
                bbox_lookup[idx] = {
                    "location": row.get("Locations", -1),
                    "label": row.get("Labels", 0)
                }
        
        # Process train and test directories
        for split_name, split_folder in [("train", "Train/Train"), ("test", "Test/Test")]:
            split_path = self.azh_dir / split_folder
            
            if not split_path.exists():
                continue
                
            for class_folder in ["D", "V", "P", "S", "N", "BG"]:
                class_path = split_path / class_folder
                if not class_path.exists():
                    continue
                    
                class_name = self._get_azh_class_from_path(class_folder)
                
                for img_file in class_path.glob("*.jpg"):
                    # Get annotation key
                    annotation_key = f"{class_folder}\\{img_file.stem}"
                    if "/" in str(img_file):
                        annotation_key = f"{class_folder}/{img_file.stem}"
                    
                    bbox_info = bbox_lookup.get(annotation_key, bbox_lookup.get(
                        annotation_key.replace("\\", "/"), {}
                    ))
                    
                    images.append({
                        "source": "azh",
                        "original_path": str(img_file),
                        "class": class_name,
                        "original_split": split_name,
                        "filename": img_file.name,
                        "bbox_location": bbox_info.get("location", -1),
                        "bbox_label": bbox_info.get("label", 0)
                    })
                    
        self.stats["azh_images"] = len(images)
        return images
    
    def _process_medetec_dataset(self) -> List[Dict]:
        """Process Medetec dataset and return image metadata"""
        images = []
        
        if not self.medetec_dir.exists():
            print(f"Medetec directory not found: {self.medetec_dir}")
            return images
            
        for category_folder in self.medetec_dir.iterdir():
            if not category_folder.is_dir():
                continue
                
            category_name = category_folder.name
            unified_class = MEDETEC_TO_UNIFIED.get(category_name, "other")
            
            for img_file in category_folder.glob("*.jpg"):
                images.append({
                    "source": "medetec",
                    "original_path": str(img_file),
                    "class": unified_class,
                    "original_split": None,  # Will be assigned during splitting
                    "filename": img_file.name,
                    "original_category": category_name,
                    "bbox_location": -1,
                    "bbox_label": 0
                })
                
        self.stats["medetec_images"] = len(images)
        return images
    
    def _split_dataset(self, images: List[Dict]) -> Dict[str, List[Dict]]:
        """Split images into train/val/test sets, stratified by class"""
        random.seed(training_config.seed)
        
        # Group by class
        class_images = {}
        for img in images:
            cls = img["class"]
            if cls not in class_images:
                class_images[cls] = []
            class_images[cls].append(img)
        
        splits = {"train": [], "val": [], "test": []}
        
        for cls, cls_images in class_images.items():
            random.shuffle(cls_images)
            
            n_total = len(cls_images)
            n_train = int(n_total * training_config.train_split)
            n_val = int(n_total * training_config.val_split)
            
            splits["train"].extend(cls_images[:n_train])
            splits["val"].extend(cls_images[n_train:n_train + n_val])
            splits["test"].extend(cls_images[n_train + n_val:])
            
        return splits
    
    def _copy_images(self, splits: Dict[str, List[Dict]]):
        """Copy images to unified directory structure"""
        all_metadata = []
        
        for split_name, images in splits.items():
            print(f"\nCopying {split_name} images...")
            
            for img in tqdm(images, desc=split_name):
                src_path = Path(img["original_path"])
                cls = img["class"]
                
                # Generate unique filename
                new_filename = f"{img['source']}_{src_path.stem}{src_path.suffix}"
                dst_path = self.output_dir / split_name / cls / new_filename
                
                # Copy file
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                    
                    # Update metadata
                    img["split"] = split_name
                    img["unified_path"] = str(dst_path)
                    all_metadata.append(img)
                    
                    self.stats["split_distribution"][split_name] += 1
                else:
                    print(f"Warning: Source file not found: {src_path}")
                    
        return all_metadata
    
    def _compute_statistics(self, metadata: List[Dict]):
        """Compute dataset statistics"""
        self.stats["total_images"] = len(metadata)
        
        # Class distribution
        for img in metadata:
            cls = img["class"]
            if cls not in self.stats["class_distribution"]:
                self.stats["class_distribution"][cls] = {"train": 0, "val": 0, "test": 0}
            self.stats["class_distribution"][cls][img["split"]] += 1
            
    def _save_metadata(self, metadata: List[Dict]):
        """Save metadata and statistics"""
        # Save full metadata
        metadata_path = self.output_dir / "annotations" / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Save statistics
        stats_path = self.output_dir / "annotations" / "statistics.json"
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)
            
        # Save class-split CSV for easy viewing
        df = pd.DataFrame(metadata)
        df.to_csv(self.output_dir / "annotations" / "dataset.csv", index=False)
        
        print("\nDataset Statistics:")
        print("=" * 50)
        print(f"AZH Images: {self.stats['azh_images']}")
        print(f"Medetec Images: {self.stats['medetec_images']}")
        print(f"Total Images: {self.stats['total_images']}")
        print(f"\nSplit Distribution:")
        for split, count in self.stats["split_distribution"].items():
            print(f"  {split}: {count}")
        print(f"\nClass Distribution:")
        for cls, dist in self.stats["class_distribution"].items():
            total = sum(dist.values())
            print(f"  {cls}: {total} (train: {dist['train']}, val: {dist['val']}, test: {dist['test']})")
            
    def harmonize(self):
        """Run the full harmonization pipeline"""
        print("=" * 60)
        print("STARTING DATA HARMONIZATION")
        print("=" * 60)
        
        # Create directory structure
        print("\n1. Creating directory structure...")
        self._create_directory_structure()
        
        # Process AZH dataset
        print("\n2. Processing AZH dataset...")
        azh_images = self._process_azh_dataset()
        print(f"   Found {len(azh_images)} AZH images")
        
        # Process Medetec dataset
        print("\n3. Processing Medetec dataset...")
        medetec_images = self._process_medetec_dataset()
        print(f"   Found {len(medetec_images)} Medetec images")
        
        # Combine all images
        all_images = azh_images + medetec_images
        print(f"\n4. Total images: {len(all_images)}")
        
        # Split dataset
        print("\n5. Splitting dataset...")
        splits = self._split_dataset(all_images)
        
        # Copy images
        print("\n6. Copying images to unified structure...")
        metadata = self._copy_images(splits)
        
        # Compute and save statistics
        print("\n7. Computing statistics...")
        self._compute_statistics(metadata)
        self._save_metadata(metadata)
        
        print("\n" + "=" * 60)
        print("HARMONIZATION COMPLETE")
        print(f"Output directory: {self.output_dir}")
        print("=" * 60)
        
        return metadata


def main():
    harmonizer = DataHarmonizer()
    harmonizer.harmonize()


if __name__ == "__main__":
    main()
