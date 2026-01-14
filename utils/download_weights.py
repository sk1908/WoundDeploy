"""
Utility to download model weights from HuggingFace Hub.
Weights are cached locally after first download.
"""

import os
from pathlib import Path
from typing import Optional
import requests
from tqdm import tqdm

# HuggingFace repository containing the model weights
# Format: "username/repo-name"
HUGGINGFACE_REPO = os.environ.get("HF_REPO", "sk1908/wound-analysis-weights")
HF_BASE_URL = f"https://huggingface.co/{HUGGINGFACE_REPO}/resolve/main"

# Model files to download
MODEL_FILES = [
    "yolo_wound_detection.pt",
    "segmentation_model.pt", 
    "classification_model.pt",
    "diffusion_best.pt",
    # "medsam_vit_b.pth",  # Optional - large file, uncomment if needed
]

def get_weights_dir() -> Path:
    """Get the weights directory path."""
    # Check if running in Streamlit Cloud (has specific env vars)
    if os.environ.get("STREAMLIT_SHARING_MODE"):
        # Use a cache directory that persists between reruns
        cache_dir = Path("/tmp/weights")
    else:
        # Local development - use project weights folder
        cache_dir = Path(__file__).parent.parent / "weights"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_file(url: str, destination: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if destination.exists():
            destination.unlink()
        return False


def download_weights(force: bool = False) -> dict:
    """
    Download all model weights from HuggingFace Hub.
    
    Args:
        force: If True, re-download even if files exist
        
    Returns:
        dict mapping filename to local path
    """
    weights_dir = get_weights_dir()
    downloaded = {}
    
    print(f"Weights directory: {weights_dir}")
    print(f"Downloading from: {HUGGINGFACE_REPO}")
    
    for filename in MODEL_FILES:
        local_path = weights_dir / filename
        
        if local_path.exists() and not force:
            print(f"✓ {filename} already exists")
            downloaded[filename] = local_path
            continue
            
        print(f"⬇ Downloading {filename}...")
        url = f"{HF_BASE_URL}/{filename}"
        
        if download_file(url, local_path):
            print(f"✓ {filename} downloaded successfully")
            downloaded[filename] = local_path
        else:
            print(f"✗ Failed to download {filename}")
    
    return downloaded


def ensure_weights() -> Path:
    """
    Ensure all weights are available, downloading if necessary.
    Returns the weights directory path.
    """
    weights_dir = get_weights_dir()
    
    # Check if all files exist
    missing = []
    for filename in MODEL_FILES:
        if not (weights_dir / filename).exists():
            missing.append(filename)
    
    if missing:
        print(f"Missing weights: {missing}")
        download_weights()
    
    return weights_dir


if __name__ == "__main__":
    # Test download
    download_weights(force=False)
