"""
Digital Twin System for Chronic Wound Analysis
Central Configuration Module
"""

import os
# Fix OpenMP library conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch

# ============================================================================
# PATH CONFIGURATION
# ============================================================================

# Base paths
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
WEIGHTS_DIR = BASE_DIR / "weights"
OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_DIR = BASE_DIR / ".cache"

# Dataset paths
AZH_DIR = DATA_DIR / "azh"
MEDETEC_DIR = DATA_DIR / "medetec-dataset"
CO2WOUNDS_DIR = BASE_DIR / "CO2Wounds-V2 Extended Chronic Wounds Dataset From Leprosy Patients"
PROCESSED_DIR = DATA_DIR / "processed"
UNIFIED_DIR = DATA_DIR / "unified"
SYNTHETIC_LONGITUDINAL_DIR = OUTPUT_DIR / "synthetic_longitudinal"

# Create directories if they don't exist
for dir_path in [WEIGHTS_DIR, OUTPUT_DIR, CACHE_DIR, PROCESSED_DIR, UNIFIED_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4 if torch.cuda.is_available() else 0
PIN_MEMORY = torch.cuda.is_available()

# ============================================================================
# CLASS MAPPINGS
# ============================================================================

# Wound type classes (from AZH dataset)
# NOTE: 'N' in AZH = Normal (healthy skin), NOT necrotic
WOUND_TYPE_CLASSES = {
    0: "background",
    1: "diabetic",
    2: "normal",  # Changed from 'necrotic' - AZH 'N' = Normal/Healthy skin
    3: "pressure",
    4: "surgical",
    5: "venous"
}

# Folder to class mapping (AZH)
# N = Normal (healthy skin, no wound) - used as negative examples
FOLDER_TO_CLASS = {
    "BG": 0,
    "D": 1,
    "N": 2,  # Normal/Healthy - NOT necrotic!
    "P": 3,
    "S": 4,
    "V": 5
}

# Medetec category mapping to unified classes
MEDETEC_TO_UNIFIED = {
    "foot-ulcers": "diabetic",
    "leg-ulcer-images": "venous",
    "pressure-ulcer-images-a": "pressure",
    "pressure-ulcer-images-b": "pressure",
    "abdominal-wounds": "surgical",
    "orthopaedic-wounds": "surgical",
    "burns": "other",
    "epidermolysis-bullosa": "other",
    "extravasation-wound-images": "other",
    "haemangioma": "other",
    "malignant-wound-images": "other",
    "meningitis": "other",
    "miscellaneous": "other",
    "pilonidal-sinus": "surgical",
    "toes": "diabetic"
}

# Tissue segmentation classes
TISSUE_CLASSES = {
    0: "background",
    1: "granulation",
    2: "slough",
    3: "necrotic",
    4: "epithelium"
}

TISSUE_COLORS = {
    "background": (0, 0, 0),
    "granulation": (255, 0, 0),      # Red - healthy healing tissue
    "slough": (255, 255, 0),          # Yellow - dead tissue
    "necrotic": (0, 0, 0),            # Black - dead tissue
    "epithelium": (255, 192, 203)     # Pink - skin/edges
}

# Severity levels for classification
SEVERITY_LEVELS = {
    0: "mild",
    1: "moderate", 
    2: "severe",
    3: "critical"
}

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

@dataclass
class YOLOConfig:
    """YOLOv8 Detection Configuration"""
    model_name: str = "yolov8n"  # nano for speed, can use yolov8s/m for accuracy
    img_size: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    epochs: int = 50  # Quick demo (increase for production)
    batch_size: int = 16
    lr: float = 0.01
    patience: int = 20
    weights_path: Path = WEIGHTS_DIR / "yolo_wound_detection.pt"


@dataclass
class SegmentationConfig:
    """Segmentation Model Configuration"""
    model_name: str = "segformer"  # or "deeplabv3plus"
    encoder: str = "mit_b2"  # For SegFormer, or "resnet50" for DeepLab
    num_classes: int = 5  # tissue types
    img_size: Tuple[int, int] = (512, 512)
    epochs: int = 5 # Quick demo (increase for production)
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    weights_path: Path = WEIGHTS_DIR / "segmentation_model.pt"


@dataclass  
class DepthConfig:
    """Depth Estimation Configuration"""
    model_name: str = "depth-anything-v2-small"
    # Options: depth-anything-v2-small, depth-anything-v2-base, depth-anything-v2-large
    img_size: Tuple[int, int] = (518, 518)  # Depth Anything default
    output_size: Tuple[int, int] = (512, 512)


@dataclass
class ClassificationConfig:
    """Wound Classification Configuration"""
    model_name: str = "efficientnet_v2_s"
    num_wound_types: int = 7  # 0-6: background, diabetic, necrotic, pressure, surgical, venous, other
    num_severity_levels: int = 4
    img_size: Tuple[int, int] = (384, 384)
    epochs: int = 100  # Quick demo (increase for production)
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-5
    dropout: float = 0.3
    weights_path: Path = WEIGHTS_DIR / "classification_model.pt"


@dataclass
class RiskConfig:
    """Risk Score Model Configuration"""
    input_features: int = 15  # tissue fractions + area + depth metrics
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    dropout: float = 0.2
    weights_path: Path = WEIGHTS_DIR / "risk_model.pt"


@dataclass
class DiffusionConfig:
    """Healing Trajectory Diffusion Model Configuration"""
    model_name: str = "stabilityai/stable-diffusion-2-1"
    img_size: Tuple[int, int] = (512, 512)
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_severity_levels: int = 5
    epochs: int = 5  # Quick demo (increase for production)
    batch_size: int = 4
    lr: float = 1e-5
    weights_path: Path = WEIGHTS_DIR / "diffusion_trajectory.pt"


@dataclass
class SAMConfig:
    """Segment Anything Model Configuration"""
    model_type: str = "vit_h"  # vit_h, vit_l, vit_b
    checkpoint_path: Path = WEIGHTS_DIR / "sam_vit_h_4b8939.pth"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.95


@dataclass
class LongitudinalConfig:
    """Simulated Longitudinal Data Configuration"""
    n_clusters: int = 40  # Number of pseudo-patient clusters
    min_cluster_size: int = 5  # Minimum images per cluster
    min_trajectory_length: int = 3  # Minimum frames per trajectory
    pca_components: int = 50  # Dimensionality reduction
    feature_model: str = "efficientnet_b0"  # Feature extraction model
    healing_stages: int = 5  # Number of healing stages (0-4)
    days_between_visits: Tuple[int, int] = (3, 7)  # Random range for simulated days


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

@dataclass 
class TrainingConfig:
    """General Training Configuration"""
    seed: int = 42
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    early_stopping_patience: int = 3  # Quick demo
    save_best_only: bool = True
    use_amp: bool = True  # Automatic Mixed Precision
    gradient_accumulation_steps: int = 1
    log_interval: int = 10
    save_interval: int = 5


@dataclass
class AugmentationConfig:
    """Data Augmentation Configuration"""
    horizontal_flip: float = 0.5
    vertical_flip: float = 0.3
    rotation_limit: int = 45
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    hue_shift_limit: int = 20
    saturation_limit: float = 0.2
    blur_limit: int = 7
    noise_var_limit: Tuple[float, float] = (10.0, 50.0)


# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

@dataclass
class InferenceConfig:
    """Real-time Inference Configuration"""
    batch_size: int = 1
    use_tensorrt: bool = False  # Enable for faster inference
    use_onnx: bool = False
    warmup_iterations: int = 3
    max_queue_size: int = 10


# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================

@dataclass
class DashboardConfig:
    """Streamlit Dashboard Configuration"""
    page_title: str = "Digital Twin - Wound Analysis"
    page_icon: str = "🏥"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    theme_primary_color: str = "#1E88E5"
    theme_background_color: str = "#FFFFFF"


# ============================================================================
# INSTANTIATE DEFAULT CONFIGS
# ============================================================================

yolo_config = YOLOConfig()
segmentation_config = SegmentationConfig()
depth_config = DepthConfig()
classification_config = ClassificationConfig()
risk_config = RiskConfig()
diffusion_config = DiffusionConfig()
sam_config = SAMConfig()
longitudinal_config = LongitudinalConfig()
training_config = TrainingConfig()
augmentation_config = AugmentationConfig()
inference_config = InferenceConfig()
dashboard_config = DashboardConfig()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_config():
    """Print all configurations"""
    print("=" * 60)
    print("DIGITAL TWIN SYSTEM CONFIGURATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Weights Directory: {WEIGHTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
