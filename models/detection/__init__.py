"""
Detection module init
"""
from .detector import WoundDetector
from .train_yolo import YOLOTrainer

__all__ = ["WoundDetector", "YOLOTrainer"]
