"""
Classification module init
"""
from .classifier import WoundClassifier, ClassificationModel
from .train import ClassificationTrainer
from .gradcam import GradCAM, apply_gradcam

__all__ = [
    "WoundClassifier",
    "ClassificationModel",
    "ClassificationTrainer",
    "GradCAM",
    "apply_gradcam"
]
