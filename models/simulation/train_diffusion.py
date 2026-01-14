"""
Trajectory-Based Diffusion Training
Trains the diffusion model using synthetic longitudinal data
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import (
    diffusion_config, 
    DEVICE, 
    WEIGHTS_DIR,
    SYNTHETIC_LONGITUDINAL_DIR
)
from models.simulation.diffusion_model import DiffusionModel


# ============================================================================
# DATASET
# ============================================================================

class TrajectoryPairsDataset(Dataset):
    """
    Dataset for training diffusion model on trajectory pairs
    
    Each sample is a (before_image, after_image, before_severity, after_severity) pair
    """
    
    def __init__(
        self,
        pairs_json_path: Optional[Path] = None,
        images_dir: Optional[Path] = None,
        img_size: Tuple[int, int] = (256, 256),
        augment: bool = True
    ):
        self.pairs_json_path = pairs_json_path or (SYNTHETIC_LONGITUDINAL_DIR / "training_pairs" / "pairs.json")
        self.images_dir = images_dir  # CO2Wounds imgs directory
        self.img_size = img_size
        self.augment = augment
        
        # Load pairs metadata
        with open(self.pairs_json_path, 'r') as f:
            data = json.load(f)
        
        # Filter pairs with positive severity improvement (healing)
        self.pairs = [p for p in data['pairs'] if p['severity_delta'] >= 0]
        
        print(f"TrajectoryPairsDataset: {len(self.pairs)} pairs loaded (filtered healing pairs)")
        
        # If images_dir not specified, try to find from config
        if self.images_dir is None:
            from config import CO2WOUNDS_DIR
            self.images_dir = CO2WOUNDS_DIR / "imgs"
    
    def __len__(self):
        return len(self.pairs)
    
    def _load_image(self, filename: str) -> np.ndarray:
        """Load and preprocess image"""
        img_path = self.images_dir / filename
        
        if not img_path.exists():
            # Fallback: try with different extensions
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                alt_path = img_path.with_suffix(ext)
                if alt_path.exists():
                    img_path = alt_path
                    break
        
        img = cv2.imread(str(img_path))
        if img is None:
            # Return a placeholder if image not found
            return np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        
        return img
    
    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Apply data augmentation"""
        if np.random.random() > 0.5:
            img = np.fliplr(img).copy()
        if np.random.random() > 0.5:
            img = np.flipud(img).copy()
        if np.random.random() > 0.5:
            img = np.rot90(img, k=np.random.randint(1, 4)).copy()
        return img
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Load images
        before_img = self._load_image(pair['before_file'])
        after_img = self._load_image(pair['after_file'])
        
        # Augment (same augmentation for both)
        if self.augment:
            seed = np.random.randint(0, 10000)
            np.random.seed(seed)
            before_img = self._augment(before_img)
            np.random.seed(seed)
            after_img = self._augment(after_img)
        
        # Convert to tensors and normalize to [-1, 1]
        before_tensor = torch.from_numpy(before_img.transpose(2, 0, 1)).float()
        after_tensor = torch.from_numpy(after_img.transpose(2, 0, 1)).float()
        
        before_tensor = (before_tensor / 127.5) - 1
        after_tensor = (after_tensor / 127.5) - 1
        
        return {
            'before_image': before_tensor,
            'after_image': after_tensor,
            'before_severity': pair['before_severity'],
            'after_severity': pair['after_severity'],
            'severity_delta': pair['severity_delta']
        }


class ImageSeverityDataset(Dataset):
    """
    Simple dataset for training diffusion on severity-labeled images
    """
    
    def __init__(
        self,
        trajectories_dir: Optional[Path] = None,
        img_size: Tuple[int, int] = (256, 256),
        augment: bool = True
    ):
        self.trajectories_dir = trajectories_dir or (SYNTHETIC_LONGITUDINAL_DIR / "trajectories")
        self.img_size = img_size
        self.augment = augment
        
        self.samples = []
        self._load_samples()
        
        print(f"ImageSeverityDataset: {len(self.samples)} samples loaded")
    
    def _load_samples(self):
        """Load all samples from trajectory folders"""
        for traj_dir in self.trajectories_dir.iterdir():
            if not traj_dir.is_dir():
                continue
            
            meta_path = traj_dir / "metadata.json"
            if not meta_path.exists():
                continue
            
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            for frame in meta['frames']:
                day = frame['day']
                img_path = traj_dir / f"day_{day:03d}.jpg"
                
                if img_path.exists():
                    self.samples.append({
                        'image_path': img_path,
                        'severity': frame['severity_level'],
                        'healing_stage': frame['healing_stage']
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        img = cv2.imread(str(sample['image_path']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        
        if self.augment:
            if np.random.random() > 0.5:
                img = np.fliplr(img).copy()
            if np.random.random() > 0.5:
                img = np.flipud(img).copy()
        
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        img_tensor = (img_tensor / 127.5) - 1
        
        return {
            'image': img_tensor,
            'severity': sample['severity']
        }


# ============================================================================
# TRAINER
# ============================================================================

class DiffusionTrainer:
    """
    Trainer for diffusion model using trajectory data
    """
    
    def __init__(
        self,
        epochs: int = None,
        batch_size: int = None,
        lr: float = None,
        device = None,
        save_path: Path = None
    ):
        self.epochs = epochs or diffusion_config.epochs
        self.batch_size = batch_size or diffusion_config.batch_size
        self.lr = lr or diffusion_config.lr
        self.device = device or DEVICE
        self.save_path = save_path or diffusion_config.weights_path
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
    
    def train(self, dataset_type: str = "severity"):
        """
        Train the diffusion model
        
        Args:
            dataset_type: "severity" for simple severity conditioning,
                         "pairs" for trajectory pairs
        """
        print("=" * 60)
        print("DIFFUSION MODEL TRAINING")
        print("=" * 60)
        print(f"Dataset type: {dataset_type}")
        print(f"Epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.lr}")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        # Create dataset
        if dataset_type == "pairs":
            dataset = TrajectoryPairsDataset(img_size=diffusion_config.img_size)
        else:
            dataset = ImageSeverityDataset(img_size=diffusion_config.img_size)
        
        if len(dataset) == 0:
            print("ERROR: No training samples found!")
            return
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Windows compatibility
            pin_memory=True
        )
        
        # Create model
        self.model = DiffusionModel(device=self.device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.model.parameters(),
            lr=self.lr,
            weight_decay=0.01
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs * len(dataloader)
        )
        
        # Training loop
        best_loss = float('inf')
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            self.model.model.train()
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for batch in pbar:
                if dataset_type == "pairs":
                    loss = self._train_step_pairs(batch)
                else:
                    loss = self._train_step_severity(batch)
                
                epoch_loss += loss
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss:.4f}'})
            
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")
            
            # Save best
            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save()
                print(f"  Saved best model (loss: {best_loss:.4f})")
        
        print("\nTraining complete!")
        print(f"Best loss: {best_loss:.4f}")
        print(f"Model saved to: {self.save_path}")
    
    def _train_step_severity(self, batch) -> float:
        """Training step for severity-conditioned generation"""
        images = batch['image'].to(self.device)
        severity = torch.tensor(batch['severity'], device=self.device, dtype=torch.long)
        
        self.optimizer.zero_grad()
        
        loss = self.model.training_step(images, severity)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def _train_step_pairs(self, batch) -> float:
        """
        Training step for trajectory pairs
        
        Uses before image as starting point and trains to predict after image
        conditioned on target severity
        """
        before_images = batch['before_image'].to(self.device)
        after_images = batch['after_image'].to(self.device)
        after_severity = torch.tensor(batch['after_severity'], device=self.device, dtype=torch.long)
        
        self.optimizer.zero_grad()
        
        # We train the model to denoise the after_image conditioned on after_severity
        # This teaches it what wounds at different severities look like
        loss = self.model.training_step(after_images, after_severity)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def _save(self):
        """Save model"""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(self.save_path)


# ============================================================================
# CLI ENTRY
# ============================================================================

def train_diffusion(
    epochs: int = 10,
    batch_size: int = 4,
    dataset_type: str = "severity"
):
    """Entry point for training"""
    trainer = DiffusionTrainer(
        epochs=epochs,
        batch_size=batch_size
    )
    trainer.train(dataset_type=dataset_type)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train diffusion model on trajectory data")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size")
    parser.add_argument("--dataset", "-d", type=str, default="severity", 
                       choices=["severity", "pairs"],
                       help="Dataset type: severity (single images) or pairs (trajectory pairs)")
    
    args = parser.parse_args()
    
    train_diffusion(
        epochs=args.epochs,
        batch_size=args.batch_size,
        dataset_type=args.dataset
    )
