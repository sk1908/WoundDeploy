"""
Grad-CAM Explainability Module
Generates visual explanations for classification decisions
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import DEVICE


@dataclass
class GradCAMResult:
    """Grad-CAM visualization result"""
    heatmap: np.ndarray
    overlay: np.ndarray
    class_name: str
    class_idx: int
    confidence: float


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping
    Provides visual explanations for CNN predictions
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 target_layer: Optional[str] = None,
                 device: Optional[torch.device] = None):
        """
        Args:
            model: Classification model
            target_layer: Name of layer to visualize (default: last conv layer)
            device: Device to run on
        """
        self.model = model
        self.device = device or DEVICE
        
        # Find target layer
        self.target_layer = self._find_target_layer(target_layer)
        
        # Hooks for activations and gradients
        self.activations = None
        self.gradients = None
        
        self._register_hooks()
    
    def _find_target_layer(self, target_layer_name: Optional[str]) -> torch.nn.Module:
        """Find the target layer for Grad-CAM"""
        if target_layer_name:
            # Find by name
            for name, module in self.model.named_modules():
                if name == target_layer_name:
                    return module
        
        # Find last convolutional layer
        target = None
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                target = module
        
        if target is None:
            # Try to find in backbone
            if hasattr(self.model, 'backbone'):
                for module in self.model.backbone.modules():
                    if isinstance(module, torch.nn.Conv2d):
                        target = module
        
        return target
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        if self.target_layer:
            self.target_layer.register_forward_hook(forward_hook)
            self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self,
                 input_tensor: torch.Tensor,
                 target_class: Optional[int] = None,
                 task: str = "wound_type") -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Class to explain (None = use predicted class)
            task: "wound_type" or "severity"
            
        Returns:
            Heatmap array of shape (H, W)
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        
        # Forward pass
        wound_logits, severity_logits = self.model(input_tensor)
        
        # Select output based on task
        if task == "wound_type":
            logits = wound_logits
        else:
            logits = severity_logits
        
        # Get target class
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        # Backward pass for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(logits)
        one_hot[0, target_class] = 1
        logits.backward(gradient=one_hot, retain_graph=True)
        
        # Compute Grad-CAM
        if self.gradients is None or self.activations is None:
            print("Warning: No activations/gradients captured")
            return np.zeros(input_tensor.shape[2:])
        
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def visualize(self,
                  image: np.ndarray,
                  heatmap: np.ndarray,
                  alpha: float = 0.4) -> np.ndarray:
        """
        Create overlay visualization
        
        Args:
            image: Original image (BGR)
            heatmap: Grad-CAM heatmap
            alpha: Overlay transparency
            
        Returns:
            Overlay image
        """
        # Resize heatmap to match image
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert to colormap
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # Blend
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay


def apply_gradcam(model: torch.nn.Module,
                  image: np.ndarray,
                  class_names: Dict[int, str],
                  task: str = "wound_type") -> GradCAMResult:
    """
    Convenience function to apply Grad-CAM to an image
    
    Args:
        model: Classification model
        image: Input image (BGR, HWC)
        class_names: Dict mapping class indices to names
        task: "wound_type" or "severity"
        
    Returns:
        GradCAMResult with heatmap, overlay, and prediction info
    """
    from .classifier import WoundClassifier
    
    # Preprocess
    img_size = (384, 384)
    image_resized = cv2.resize(image, img_size)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = image_rgb.astype(np.float32) / 255.0
    image_norm = (image_norm - mean) / std
    
    # To tensor
    tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).float()
    tensor = tensor.unsqueeze(0).to(DEVICE)
    
    # Create Grad-CAM
    gradcam = GradCAM(model)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        wound_logits, severity_logits = model(tensor)
        
        if task == "wound_type":
            probs = F.softmax(wound_logits, dim=1)
        else:
            probs = F.softmax(severity_logits, dim=1)
    
    class_idx = probs.argmax().item()
    confidence = probs[0, class_idx].item()
    class_name = class_names.get(class_idx, f"class_{class_idx}")
    
    # Generate heatmap
    heatmap = gradcam.generate(tensor, target_class=class_idx, task=task)
    
    # Create overlay
    overlay = gradcam.visualize(image_resized, heatmap)
    
    return GradCAMResult(
        heatmap=heatmap,
        overlay=overlay,
        class_name=class_name,
        class_idx=class_idx,
        confidence=confidence
    )


def visualize_all_classes(model: torch.nn.Module,
                          image: np.ndarray,
                          class_names: Dict[int, str],
                          task: str = "wound_type") -> Dict[str, np.ndarray]:
    """
    Generate Grad-CAM visualizations for all classes
    
    Returns:
        Dictionary mapping class names to overlay images
    """
    from .classifier import WoundClassifier
    
    img_size = (384, 384)
    image_resized = cv2.resize(image, img_size)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_norm = image_rgb.astype(np.float32) / 255.0
    image_norm = (image_norm - mean) / std
    
    tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).float()
    tensor = tensor.unsqueeze(0).to(DEVICE)
    
    gradcam = GradCAM(model)
    
    results = {}
    
    for class_idx, class_name in class_names.items():
        heatmap = gradcam.generate(tensor, target_class=class_idx, task=task)
        overlay = gradcam.visualize(image_resized, heatmap)
        results[class_name] = overlay
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Grad-CAM Visualization")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--weights", type=str)
    parser.add_argument("--output", type=str, default="gradcam_result.jpg")
    
    args = parser.parse_args()
    
    from .classifier import ClassificationModel, WoundClassifier
    from config import classification_config, WOUND_TYPE_CLASSES
    
    # Load model
    model = ClassificationModel()
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # Load image
    image = cv2.imread(args.image)
    
    # Apply Grad-CAM
    result = apply_gradcam(model, image, WOUND_TYPE_CLASSES)
    
    print(f"Prediction: {result.class_name} ({result.confidence:.2%})")
    cv2.imwrite(args.output, result.overlay)
    print(f"Saved to {args.output}")
