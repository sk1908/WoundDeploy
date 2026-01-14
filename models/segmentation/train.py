"""
Tissue Segmentation Training Module
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import json
from tqdm import tqdm
import time

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import (
    UNIFIED_DIR, PROCESSED_DIR, WEIGHTS_DIR, OUTPUT_DIR,
    segmentation_config, training_config, DEVICE
)
from data.preprocessing import get_segmentation_transforms
from .model import SegmentationModel, CombinedLoss, TissueSegmentor


class WoundSegmentationDataset(Dataset):
    """
    Dataset for wound tissue segmentation
    """
    
    def __init__(self,
                 image_dir: Path,
                 mask_dir: Optional[Path] = None,
                 split: str = "train",
                 transform = None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.split = split
        self.transform = transform or get_segmentation_transforms(split == "train")
        
        # Find all images
        self.image_paths = []
        self.mask_paths = []
        
        self._load_paths()
        
    def _load_paths(self):
        """Load image and mask paths"""
        # Look in split subdirectory
        split_dir = self.image_dir / self.split
        
        if split_dir.exists():
            search_dir = split_dir
        else:
            search_dir = self.image_dir
        
        # Find images
        for ext in ["*.jpg", "*.png"]:
            self.image_paths.extend(search_dir.rglob(ext))
        
        # Find corresponding masks if mask_dir provided
        if self.mask_dir:
            for img_path in self.image_paths:
                # Try to find mask with same relative path
                rel_path = img_path.relative_to(search_dir)
                mask_path = self.mask_dir / self.split / rel_path.with_suffix(".png")
                
                if mask_path.exists():
                    self.mask_paths.append(mask_path)
                else:
                    # Create dummy mask path (will generate synthetic)
                    self.mask_paths.append(None)
        else:
            self.mask_paths = [None] * len(self.image_paths)
        
        print(f"Found {len(self.image_paths)} images in {self.split} split")
    
    def _generate_synthetic_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Generate synthetic tissue mask using color-based heuristics
        This is a fallback when no ground truth masks are available
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Simple color-based heuristics (very approximate)
        # Red tones -> granulation (class 1)
        red_mask = ((hsv[:, :, 0] < 20) | (hsv[:, :, 0] > 160)) & (hsv[:, :, 1] > 50)
        mask[red_mask] = 1
        
        # Yellow tones -> slough (class 2)
        yellow_mask = (hsv[:, :, 0] >= 20) & (hsv[:, :, 0] < 40) & (hsv[:, :, 1] > 50)
        mask[yellow_mask] = 2
        
        # Dark areas -> necrotic (class 3)
        dark_mask = (hsv[:, :, 2] < 50)
        mask[dark_mask] = 3
        
        # Pink/light areas at edges -> epithelium (class 4)
        pink_mask = (hsv[:, :, 0] > 150) & (hsv[:, :, 0] < 180) & \
                    (hsv[:, :, 1] < 100) & (hsv[:, :, 2] > 150)
        mask[pink_mask] = 4
        
        return mask
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load or generate mask
        if self.mask_paths[idx] and Path(self.mask_paths[idx]).exists():
            mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        else:
            mask = self._generate_synthetic_mask(
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        # Ensure mask is long tensor for loss computation
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()
        
        return {
            "image": image,
            "mask": mask,
            "path": str(self.image_paths[idx])
        }


class SegmentationTrainer:
    """
    Training pipeline for tissue segmentation
    """
    
    def __init__(self,
                 data_dir: Optional[Path] = None,
                 mask_dir: Optional[Path] = None,
                 output_dir: Optional[Path] = None):
        self.data_dir = data_dir or UNIFIED_DIR
        self.mask_dir = mask_dir or (PROCESSED_DIR / "sam_masks")
        self.output_dir = output_dir or (OUTPUT_DIR / "segmentation")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = segmentation_config
        self.device = DEVICE
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        self.best_metric = 0.0
        self.history = {"train_loss": [], "val_loss": [], "val_dice": []}
        
    def setup(self):
        """Initialize model, optimizer, and loss"""
        # Model
        self.model = SegmentationModel(
            model_name=self.config.model_name,
            encoder=self.config.encoder,
            num_classes=self.config.num_classes
        )
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=1e-6
        )
        
        # Loss with class weights (account for imbalance)
        class_weights = torch.tensor([0.5, 1.0, 1.2, 1.5, 1.0]).to(self.device)
        self.criterion = CombinedLoss(class_weights=class_weights)
        
        print(f"Model: {self.config.model_name} with {self.config.encoder}")
        print(f"Device: {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders"""
        train_dataset = WoundSegmentationDataset(
            image_dir=self.data_dir,
            mask_dir=self.mask_dir,
            split="train",
            transform=get_segmentation_transforms(is_training=True)
        )
        
        val_dataset = WoundSegmentationDataset(
            image_dir=self.data_dir,
            mask_dir=self.mask_dir,
            split="val",
            transform=get_segmentation_transforms(is_training=False)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def compute_dice(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Compute per-class Dice scores and necrotic recall.
        
        Returns:
            Dict with 'mean_dice', per-class dice, and 'necrotic_recall'
        """
        pred_classes = torch.argmax(pred, dim=1)
        
        class_names = {0: 'background', 1: 'granulation', 2: 'epithelium', 3: 'necrotic', 4: 'slough'}
        metrics = {}
        dice_scores = []
        
        for class_id in range(self.config.num_classes):
            pred_class = (pred_classes == class_id).float()
            target_class = (target == class_id).float()
            
            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()
            
            if union > 0:
                dice = (2 * intersection / union).item()
            else:
                dice = 0.0
            
            class_name = class_names.get(class_id, f'class_{class_id}')
            metrics[f'dice_{class_name}'] = dice
            
            if class_id > 0:  # Skip background for mean
                dice_scores.append(dice)
            
            # Compute recall for necrotic (class 3)
            if class_id == 3:
                tp = intersection.item()
                fn = target_class.sum().item() - tp
                recall = tp / (tp + fn + 1e-8)
                metrics['necrotic_recall'] = recall
        
        metrics['mean_dice'] = np.mean(dice_scores) if dice_scores else 0.0
        return metrics
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate model and return per-class metrics"""
        self.model.eval()
        total_loss = 0.0
        all_metrics = {}
        
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            total_loss += loss.item()
            batch_metrics = self.compute_dice(outputs, masks)
            
            # Accumulate metrics
            for key, value in batch_metrics.items():
                all_metrics[key] = all_metrics.get(key, 0) + value
        
        # Average metrics
        avg_loss = total_loss / len(dataloader)
        for key in all_metrics:
            all_metrics[key] /= len(dataloader)
        
        # Log necrotic recall if available
        if 'necrotic_recall' in all_metrics:
            print(f"  Necrotic Recall: {all_metrics['necrotic_recall']:.4f}")
        
        return avg_loss, all_metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config.__dict__
        }
        
        # Save latest
        torch.save(checkpoint, self.output_dir / "latest.pt")
        
        # Save best
        if is_best:
            torch.save(self.model.state_dict(), self.config.weights_path)
            torch.save(checkpoint, self.output_dir / "best.pt")
            print(f"  -> Saved best model (Dice: {self.best_metric:.4f})")
    
    def train(self, resume: bool = False):
        """Full training loop"""
        self.setup()
        train_loader, val_loader = self.create_dataloaders()
        
        # Resume if requested
        start_epoch = 0
        if resume and (self.output_dir / "latest.pt").exists():
            checkpoint = torch.load(self.output_dir / "latest.pt")
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            self.best_metric = checkpoint["best_metric"]
            print(f"Resuming from epoch {start_epoch}")
        
        print(f"\nStarting training for {self.config.epochs} epochs...")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        
        patience_counter = 0
        
        for epoch in range(start_epoch, self.config.epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            print("-" * 40)
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate (now returns dict of metrics)
            val_loss, val_metrics = self.validate(val_loader)
            mean_dice = val_metrics.get('mean_dice', 0.0)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Mean Dice: {mean_dice:.4f}")
            print(f"  Dice (Gran): {val_metrics.get('dice_granulation', 0):.3f} | (Necr): {val_metrics.get('dice_necrotic', 0):.3f} | (Slough): {val_metrics.get('dice_slough', 0):.3f}")
            print(f"LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_dice"].append(mean_dice)
            
            # Save best (using mean_dice)
            is_best = mean_dice > self.best_metric
            if is_best:
                self.best_metric = mean_dice
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if patience_counter >= training_config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Save history
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining complete!")
        print(f"Best Dice: {self.best_metric:.4f}")
        
        return self.history


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Segmentation Training")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    
    args = parser.parse_args()
    
    if args.epochs:
        segmentation_config.epochs = args.epochs
    
    if args.train:
        trainer = SegmentationTrainer()
        trainer.train(resume=args.resume)


if __name__ == "__main__":
    main()
