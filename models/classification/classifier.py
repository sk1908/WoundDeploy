"""
Wound Classification Model
Multi-task classification for wound type and severity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import cv2
from dataclasses import dataclass

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import (
    classification_config, WOUND_TYPE_CLASSES, SEVERITY_LEVELS, DEVICE
)


@dataclass
class ClassificationResult:
    """Classification result"""
    wound_type: str
    wound_type_id: int
    wound_type_confidence: float
    wound_type_probs: Dict[str, float]
    
    severity: str
    severity_id: int
    severity_confidence: float
    severity_probs: Dict[str, float]


class ClassificationModel(nn.Module):
    """
    Multi-task wound classification model
    
    Outputs:
    - Wound type (diabetic, venous, pressure, surgical, etc.)
    - Severity level (mild, moderate, severe, critical)
    """
    
    def __init__(self,
                 model_name: str = None,
                 num_wound_types: int = None,
                 num_severity_levels: int = None,
                 pretrained: bool = True,
                 dropout: float = None):
        super().__init__()
        
        self.model_name = model_name or classification_config.model_name
        self.num_wound_types = num_wound_types or classification_config.num_wound_types
        self.num_severity_levels = num_severity_levels or classification_config.num_severity_levels
        self.dropout = dropout or classification_config.dropout
        
        # Build backbone
        self.backbone, self.feature_dim = self._build_backbone(pretrained)
        
        # Separate heads for each task
        self.wound_type_head = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout / 2),
            nn.Linear(256, self.num_wound_types)
        )
        
        self.severity_head = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout / 2),
            nn.Linear(128, self.num_severity_levels)
        )
        
    def _build_backbone(self, pretrained: bool) -> Tuple[nn.Module, int]:
        """Build backbone network"""
        import timm
        
        # Map model names
        model_map = {
            "efficientnet_v2_s": "tf_efficientnetv2_s",
            "efficientnet_v2_m": "tf_efficientnetv2_m",
            "efficientnet_b4": "efficientnet_b4",
            "resnet50": "resnet50",
            "convnext_small": "convnext_small",
            "vit_small": "vit_small_patch16_224"
        }
        
        timm_name = model_map.get(self.model_name, "tf_efficientnetv2_s")
        
        # Create model without classifier
        model = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,  
            global_pool='avg'
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = model(dummy)
            feature_dim = features.shape[-1]
        
        return model, feature_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            wound_type_logits, severity_logits
        """
        # Extract features
        features = self.backbone(x)
        
        # Classification heads
        wound_type_logits = self.wound_type_head(features)
        severity_logits = self.severity_head(features)
        
        return wound_type_logits, severity_logits
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict classes"""
        wound_logits, severity_logits = self.forward(x)
        wound_type = torch.argmax(wound_logits, dim=1)
        severity = torch.argmax(severity_logits, dim=1)
        return wound_type, severity
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get feature embeddings for downstream use"""
        return self.backbone(x)


class WoundClassifier:
    """
    High-level classifier interface
    """
    
    def __init__(self,
                 weights_path: Optional[Path] = None,
                 device: Optional[torch.device] = None):
        self.weights_path = weights_path or classification_config.weights_path
        self.device = device or DEVICE
        self.img_size = classification_config.img_size
        
        self.model = None
        self._is_loaded = False
        
        self.wound_types = WOUND_TYPE_CLASSES
        self.severity_levels = SEVERITY_LEVELS
        
    def load(self) -> bool:
        """Load model"""
        try:
            self.model = ClassificationModel()
            
            if self.weights_path.exists():
                state_dict = torch.load(self.weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Loaded weights from {self.weights_path}")
            else:
                print("No pretrained weights, using ImageNet initialization")
            
            self.model.to(self.device)
            self.model.eval()
            self._is_loaded = True
            
            return True
        except Exception as e:
            print(f"Error loading classifier: {e}")
            return False
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image"""
        h, w = self.img_size
        image = cv2.resize(image, (w, h))
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image.astype(np.float32) / 255.0
        image = (image - mean) / std
        
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        return tensor.unsqueeze(0)
    
    def classify(self, image: np.ndarray, confidence_threshold: float = 0.4) -> ClassificationResult:
        """
        Classify wound image
        
        Args:
            image: Input image (BGR)
            confidence_threshold: Minimum confidence to accept classification.
                                  If max confidence is below this, classify as background.
            
        Returns:
            ClassificationResult with type and severity
        """
        if not self._is_loaded:
            self.load()
        
        # Preprocess
        x = self.preprocess(image).to(self.device)
        
        # Inference
        with torch.no_grad():
            wound_logits, severity_logits = self.model(x)
            
            wound_probs = F.softmax(wound_logits, dim=1)[0].cpu().numpy()
            severity_probs = F.softmax(severity_logits, dim=1)[0].cpu().numpy()
        
        # Get predictions
        wound_type_id = int(np.argmax(wound_probs))
        severity_id = int(np.argmax(severity_probs))
        
        max_wound_conf = float(wound_probs[wound_type_id])
        
        # Calculate entropy (uncertainty measure)
        # High entropy = model is uncertain across classes
        wound_entropy = -np.sum(wound_probs * np.log(wound_probs + 1e-10))
        max_entropy = np.log(len(wound_probs))  # Maximum possible entropy
        normalized_entropy = wound_entropy / max_entropy
        
        # OOD (Out-of-Distribution) Detection:
        # If confidence is low OR entropy is high, treat as background/non-wound
        is_uncertain = (max_wound_conf < confidence_threshold) or (normalized_entropy > 0.85)
        
        # Also check if the predicted class is already background (id=0)
        # Background should be accepted at lower confidence since it's the "no wound" class
        if wound_type_id == 0:  # background
            is_uncertain = False  # Accept background predictions
        elif wound_type_id == 2:  # normal (healthy skin)
            is_uncertain = False  # Accept normal skin predictions
        
        # If uncertain about wound type, default to background
        if is_uncertain:
            wound_type_id = 0  # background
            # Adjust probabilities to reflect uncertainty
            wound_probs_adjusted = np.zeros_like(wound_probs)
            wound_probs_adjusted[0] = 0.5  # Give some confidence to background
            for i in range(len(wound_probs)):
                if i != 0:
                    wound_probs_adjusted[i] = wound_probs[i] * 0.5  # Reduce other probs
            wound_probs = wound_probs_adjusted / wound_probs_adjusted.sum()
            max_wound_conf = float(wound_probs[0])
        
        wound_type = self.wound_types.get(wound_type_id, "unknown")
        severity = self.severity_levels.get(severity_id, "unknown")
        
        # Create probability dictionaries
        wound_type_probs = {
            self.wound_types[i]: float(p) 
            for i, p in enumerate(wound_probs)
            if i in self.wound_types
        }
        
        severity_probs_dict = {
            self.severity_levels[i]: float(p)
            for i, p in enumerate(severity_probs)
            if i in self.severity_levels
        }
        
        return ClassificationResult(
            wound_type=wound_type,
            wound_type_id=wound_type_id,
            wound_type_confidence=max_wound_conf,
            wound_type_probs=wound_type_probs,
            severity=severity,
            severity_id=severity_id,
            severity_confidence=float(severity_probs[severity_id]),
            severity_probs=severity_probs_dict
        )
    
    def get_embeddings(self, image: np.ndarray) -> np.ndarray:
        """Get feature embeddings for the image"""
        if not self._is_loaded:
            self.load()
        
        x = self.preprocess(image).to(self.device)
        
        with torch.no_grad():
            features = self.model.get_features(x)
        
        return features[0].cpu().numpy()
    
    def warmup(self, iterations: int = 3):
        """Warmup model"""
        if not self._is_loaded:
            self.load()
        
        dummy = np.zeros((384, 384, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.classify(dummy)
        
        print("Classifier warmup complete")
    
    def get_gradcam(self, image: np.ndarray, task: str = "wound_type") -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Grad-CAM visualization for the classification
        
        Args:
            image: Input image (BGR)
            task: "wound_type" or "severity"
            
        Returns:
            Tuple of (heatmap, overlay)
        """
        if not self._is_loaded:
            self.load()
        
        from .gradcam import GradCAM
        
        # Preprocess image
        h, w = self.img_size
        image_resized = cv2.resize(image, (w, h))
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_norm = image_rgb.astype(np.float32) / 255.0
        image_norm = (image_norm - mean) / std
        
        # To tensor
        tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).float()
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # Create Grad-CAM and generate heatmap
        gradcam = GradCAM(self.model, device=self.device)
        heatmap = gradcam.generate(tensor, target_class=None, task=task)
        
        # Create overlay visualization
        overlay = gradcam.visualize(image_resized, heatmap, alpha=0.4)
        
        return heatmap, overlay


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining wound type and severity losses
    """
    
    def __init__(self,
                 wound_weight: float = 1.0,
                 severity_weight: float = 0.5,
                 wound_class_weights: Optional[torch.Tensor] = None,
                 severity_class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        
        self.wound_weight = wound_weight
        self.severity_weight = severity_weight
        
        self.wound_loss = nn.CrossEntropyLoss(weight=wound_class_weights)
        self.severity_loss = nn.CrossEntropyLoss(weight=severity_class_weights)
    
    def forward(self,
                wound_logits: torch.Tensor,
                severity_logits: torch.Tensor,
                wound_targets: torch.Tensor,
                severity_targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined loss
        
        Returns:
            total_loss, loss_dict
        """
        wound_l = self.wound_loss(wound_logits, wound_targets)
        severity_l = self.severity_loss(severity_logits, severity_targets)
        
        total = self.wound_weight * wound_l + self.severity_weight * severity_l
        
        return total, {
            "wound_loss": wound_l.item(),
            "severity_loss": severity_l.item(),
            "total_loss": total.item()
        }


if __name__ == "__main__":
    # Test model
    model = ClassificationModel()
    print(f"Model: {model.model_name}")
    print(f"Feature dim: {model.feature_dim}")
    
    # Test forward pass
    x = torch.randn(1, 3, 384, 384)
    wound_logits, severity_logits = model(x)
    print(f"Wound logits: {wound_logits.shape}")
    print(f"Severity logits: {severity_logits.shape}")
