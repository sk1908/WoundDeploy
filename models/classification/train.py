"""
Wound Classification Training Module
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

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import (
    UNIFIED_DIR, OUTPUT_DIR, 
    classification_config, training_config, 
    WOUND_TYPE_CLASSES, FOLDER_TO_CLASS, DEVICE
)
from data.preprocessing import get_classification_transforms
from .classifier import ClassificationModel, MultiTaskLoss


class WoundClassificationDataset(Dataset):
    """
    Dataset for wound type and severity classification
    """
    
    def __init__(self,
                 data_dir: Path,
                 split: str = "train",
                 transform = None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform or get_classification_transforms(split == "train")
        
        # Class mappings
        self.class_to_idx = {
            "background": 0,
            "diabetic": 1,
            "necrotic": 2,
            "pressure": 3,
            "surgical": 4,
            "venous": 5,
            "other": 6
        }
        
        self.severity_to_idx = {
            "mild": 0,
            "moderate": 1,
            "severe": 2,
            "critical": 3
        }
        
        self.samples = []
        self._load_samples()
    
    def _load_samples(self):
        """Load image paths and labels"""
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            print(f"Split directory not found: {split_dir}")
            return
        
        for class_folder in split_dir.iterdir():
            if not class_folder.is_dir():
                continue
            
            class_name = class_folder.name.lower()
            if class_name not in self.class_to_idx:
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_folder.glob("*.jpg"):
                # Estimate severity from tissue composition (heuristic)
                # In a real scenario, you'd have actual severity labels
                severity_idx = self._estimate_severity(img_path)
                
                self.samples.append({
                    "path": img_path,
                    "wound_type": class_idx,
                    "severity": severity_idx
                })
        
        print(f"Loaded {len(self.samples)} samples for {self.split}")
    
    def _estimate_severity(self, img_path: Path) -> int:
        """
        Estimate severity based on filename patterns or random assignment
        In practice, this should use actual clinical labels
        """
        # Simple heuristic based on wound type
        path_str = str(img_path).lower()
        
        if "necrotic" in path_str or "_n" in path_str:
            return 3  # critical
        elif any(x in path_str for x in ["pressure", "diabetic"]):
            return np.random.choice([1, 2], p=[0.4, 0.6])  # moderate/severe
        elif "surgical" in path_str:
            return np.random.choice([0, 1], p=[0.6, 0.4])  # mild/moderate
        else:
            return np.random.choice([0, 1, 2], p=[0.3, 0.4, 0.3])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(sample["path"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        return {
            "image": image,
            "wound_type": torch.tensor(sample["wound_type"]),
            "severity": torch.tensor(sample["severity"]),
            "path": str(sample["path"])
        }


class ClassificationTrainer:
    """
    Training pipeline for wound classification
    """
    
    def __init__(self,
                 data_dir: Optional[Path] = None,
                 output_dir: Optional[Path] = None):
        self.data_dir = data_dir or UNIFIED_DIR
        self.output_dir = output_dir or (OUTPUT_DIR / "classification")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = classification_config
        self.device = DEVICE
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        self.best_metric = 0.0
        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    def setup(self):
        """Initialize model, optimizer, and loss"""
        # Model
        self.model = ClassificationModel(
            model_name=self.config.model_name,
            num_wound_types=self.config.num_wound_types,
            num_severity_levels=self.config.num_severity_levels,
            dropout=self.config.dropout
        )
        self.model.to(self.device)
        
        # Optimizer with layer-wise learning rate decay
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        self.optimizer = optim.AdamW([
            {"params": backbone_params, "lr": self.config.lr * 0.1},
            {"params": head_params, "lr": self.config.lr}
        ], weight_decay=self.config.weight_decay)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=1e-6
        )
        
        # Loss
        self.criterion = MultiTaskLoss(
            wound_weight=1.0,
            severity_weight=0.5
        )
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Model: {self.config.model_name}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders"""
        train_dataset = WoundClassificationDataset(
            data_dir=self.data_dir,
            split="train",
            transform=get_classification_transforms(is_training=True)
        )
        
        val_dataset = WoundClassificationDataset(
            data_dir=self.data_dir,
            split="val",
            transform=get_classification_transforms(is_training=False)
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
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            images = batch["image"].to(self.device)
            wound_types = batch["wound_type"].to(self.device)
            severities = batch["severity"].to(self.device)
            
            self.optimizer.zero_grad()
            
            wound_logits, severity_logits = self.model(images)
            loss, loss_dict = self.criterion(
                wound_logits, severity_logits,
                wound_types, severities
            )
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "wound": f"{loss_dict['wound_loss']:.4f}",
                "sev": f"{loss_dict['severity_loss']:.4f}"
            })
        
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct_wound = 0
        correct_severity = 0
        total = 0
        
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch["image"].to(self.device)
            wound_types = batch["wound_type"].to(self.device)
            severities = batch["severity"].to(self.device)
            
            wound_logits, severity_logits = self.model(images)
            loss, _ = self.criterion(
                wound_logits, severity_logits,
                wound_types, severities
            )
            
            total_loss += loss.item()
            
            # Accuracy
            wound_pred = torch.argmax(wound_logits, dim=1)
            severity_pred = torch.argmax(severity_logits, dim=1)
            
            correct_wound += (wound_pred == wound_types).sum().item()
            correct_severity += (severity_pred == severities).sum().item()
            total += images.size(0)
        
        avg_loss = total_loss / len(dataloader)
        wound_acc = correct_wound / total
        severity_acc = correct_severity / total
        
        return avg_loss, wound_acc, severity_acc
    
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
        
        torch.save(checkpoint, self.output_dir / "latest.pt")
        
        if is_best:
            torch.save(self.model.state_dict(), self.config.weights_path)
            torch.save(checkpoint, self.output_dir / "best.pt")
            print(f"  -> Saved best model (Acc: {self.best_metric:.4f})")
    
    def train(self, resume: bool = False):
        """Full training loop"""
        self.setup()
        train_loader, val_loader = self.create_dataloaders()
        
        # Resume
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
            
            train_loss = self.train_epoch(train_loader)
            val_loss, wound_acc, severity_acc = self.validate(val_loader)
            
            self.scheduler.step()
            
            combined_acc = (wound_acc * 0.7 + severity_acc * 0.3)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Wound Acc: {wound_acc:.4f} | Severity Acc: {severity_acc:.4f}")
            print(f"Combined Acc: {combined_acc:.4f}")
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(combined_acc)
            
            is_best = combined_acc > self.best_metric
            if is_best:
                self.best_metric = combined_acc
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.save_checkpoint(epoch, is_best)
            
            if patience_counter >= training_config.early_stopping_patience:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break
        
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining complete! Best accuracy: {self.best_metric:.4f}")
        
        return self.history


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Classification Training")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    
    args = parser.parse_args()
    
    if args.train:
        trainer = ClassificationTrainer()
        trainer.train(resume=args.resume)


if __name__ == "__main__":
    main()
