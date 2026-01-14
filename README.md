# Digital Twin System for Chronic Wound Analysis

A comprehensive AI-powered system for analyzing chronic wounds, predicting healing trajectories, and providing clinical decision support.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Data Harmonization
```bash
python run.py harmonize
```

### 3. Train Models (Optional - uses pretrained weights)
```bash
python run.py train-all
```

### 4. Launch Dashboard
```bash
python run.py dashboard
```

## 📁 Project Structure

```
majorer/
├── app/                    # Streamlit dashboard
│   └── main.py            # Main dashboard application
├── data/                   # Datasets and preprocessing
│   ├── azh/               # AZH wound dataset
│   ├── medetec-dataset/   # Medetec wound dataset
│   ├── harmonize.py       # Dataset merging
│   ├── preprocessing.py   # Image preprocessing
│   └── sam_mask_generator.py
├── models/                 # AI modules
│   ├── detection/         # YOLOv8 wound detection
│   ├── segmentation/      # Tissue segmentation
│   ├── depth/             # Depth estimation
│   ├── classification/    # Wound type/severity
│   ├── risk/              # Non-healing risk
│   └── simulation/        # Healing trajectory
├── pipeline/               # Integration
│   ├── inference.py       # End-to-end pipeline
│   └── digital_twin.py    # State management
├── config.py              # Configuration
├── requirements.txt       # Dependencies
└── run.py                 # CLI entry point
```

## 🔧 Commands

| Command | Description |
|---------|-------------|
| `python run.py dashboard` | Launch Streamlit UI |
| `python run.py harmonize` | Merge datasets |
| `python run.py train-yolo` | Train detection |
| `python run.py train-seg` | Train segmentation |
| `python run.py train-cls` | Train classification |
| `python run.py analyze <image>` | Analyze single image |

## 🏥 Features

1. **Wound Detection** - YOLOv8-based ROI extraction
2. **Tissue Segmentation** - SegFormer/DeepLabV3+ tissue mapping
3. **Depth Estimation** - Depth Anything V2 for 3D analysis
4. **Classification** - Multi-task wound type + severity
5. **Risk Scoring** - Non-healing likelihood prediction
6. **Healing Simulation** - Diffusion-based trajectory generation
7. **Digital Twin** - Temporal tracking and trend analysis

## 📊 Dashboard

The Streamlit dashboard provides:
- **Analyze Wound**: Upload and analyze wound images
- **Dashboard**: View trends and predictions
- **Simulate Healing**: AI-generated healing trajectories
- **History**: Track wound progression over time

## 🔬 Models

| Model | Architecture | Purpose |
|-------|-------------|---------|
| Detection | YOLOv8n | Wound localization |
| Segmentation | SegFormer-B2 | Tissue classification |
| Depth | Depth Anything V2 | 3D volume estimation |
| Classification | EfficientNetV2-S | Type + severity |
| Risk | MLP | Non-healing prediction |
| Simulation | Conditional Diffusion | Healing trajectory |
