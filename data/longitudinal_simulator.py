"""
Longitudinal Data Simulator
Simulates longitudinal wound healing data from CO2Wounds-V2 dataset

This module:
1. Extracts visual features from wound images
2. Clusters similar-looking wounds (pseudo-patients)
3. Estimates healing stage for each image
4. Creates synthetic healing trajectories
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import DATA_DIR, DEVICE, WEIGHTS_DIR, OUTPUT_DIR
from data.co2wounds_dataset import CO2WoundsDataset


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class WoundFeatures:
    """Features extracted from a single wound image"""
    filename: str
    image_id: int
    
    # CNN features (1280-dim from EfficientNet)
    cnn_features: np.ndarray
    
    # Wound-specific features
    wound_area_ratio: float  # Wound area / total image area
    
    # Color features within wound region
    mean_red: float
    mean_green: float
    mean_blue: float
    
    # Tissue ratio estimates
    granulation_ratio: float  # Red tissue
    slough_ratio: float       # Yellow tissue  
    necrotic_ratio: float     # Dark tissue
    epithelium_ratio: float   # Pink/skin-tone tissue
    
    # Healing stage estimate (0-4)
    healing_stage: int = 0


@dataclass
class SyntheticTrajectory:
    """A synthetic healing trajectory"""
    trajectory_id: str
    pseudo_patient_id: str
    frames: List[Dict] = field(default_factory=list)
    
    def to_dict(self):
        return {
            'trajectory_id': self.trajectory_id,
            'pseudo_patient_id': self.pseudo_patient_id,
            'num_frames': len(self.frames),
            'frames': self.frames
        }


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class WoundFeatureExtractor:
    """
    Extracts visual features from wound images using:
    1. Pre-trained CNN (EfficientNet-B0) for global features
    2. Color analysis for tissue-specific features
    """
    
    def __init__(self, device=None):
        self.device = device or DEVICE
        self.model = None
        self.transform = None
        self._setup_model()
    
    def _setup_model(self):
        """Setup EfficientNet for feature extraction"""
        # Load pre-trained EfficientNet-B0
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Remove classification head to get features
        self.model.classifier = nn.Identity()
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Transform for EfficientNet
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"Feature extractor initialized on {self.device}")
    
    def extract_cnn_features(self, image: np.ndarray) -> np.ndarray:
        """Extract CNN features from image"""
        # Transform
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(img_tensor)
        
        return features.cpu().numpy().flatten()
    
    def extract_color_features(
        self, 
        image: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Extract color-based features from wound region
        
        Args:
            image: RGB image
            mask: Binary wound mask (optional)
        
        Returns:
            Dictionary with color features
        """
        # If no mask, use simple color thresholding to find wound
        if mask is None:
            mask = self._create_simple_mask(image)
        
        # Ensure mask is boolean
        mask_bool = mask > 0
        
        # Get wound pixels
        if mask_bool.sum() == 0:
            # No wound detected, use center region
            h, w = image.shape[:2]
            mask_bool = np.zeros((h, w), dtype=bool)
            mask_bool[h//4:3*h//4, w//4:3*w//4] = True
        
        wound_pixels = image[mask_bool]
        
        # Calculate wound area ratio
        total_pixels = image.shape[0] * image.shape[1]
        wound_area_ratio = mask_bool.sum() / total_pixels
        
        # Mean colors
        mean_red = wound_pixels[:, 0].mean() / 255.0
        mean_green = wound_pixels[:, 1].mean() / 255.0
        mean_blue = wound_pixels[:, 2].mean() / 255.0
        
        # Tissue type estimates based on color ranges
        tissue_ratios = self._estimate_tissue_ratios(wound_pixels)
        
        return {
            'wound_area_ratio': wound_area_ratio,
            'mean_red': mean_red,
            'mean_green': mean_green,
            'mean_blue': mean_blue,
            **tissue_ratios
        }
    
    def _create_simple_mask(self, image: np.ndarray) -> np.ndarray:
        """Create simple wound mask using color thresholding"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Wound regions tend to have:
        # - Reddish (granulation): H=0-20, high S
        # - Yellowish (slough): H=20-40, high S
        # - Dark (necrotic): low V
        
        # Create mask for wound-like colors
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Red/pink regions
        red_mask = ((hsv[:,:,0] < 20) | (hsv[:,:,0] > 160)) & (hsv[:,:,1] > 50)
        mask = mask | red_mask.astype(np.uint8)
        
        # Yellow regions
        yellow_mask = (hsv[:,:,0] >= 15) & (hsv[:,:,0] <= 45) & (hsv[:,:,1] > 50)
        mask = mask | yellow_mask.astype(np.uint8)
        
        # Dark regions (potential necrosis)
        dark_mask = hsv[:,:,2] < 60
        mask = mask | dark_mask.astype(np.uint8)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _estimate_tissue_ratios(self, wound_pixels: np.ndarray) -> Dict[str, float]:
        """
        Estimate tissue type ratios based on color
        
        Tissue types:
        - Granulation (Red/pink): healthy healing tissue
        - Slough (Yellow/cream): dead tissue needing debridement
        - Necrotic (Black/brown): dead/dying tissue
        - Epithelium (Pink/skin-tone): new skin forming
        """
        if len(wound_pixels) == 0:
            return {
                'granulation_ratio': 0.0,
                'slough_ratio': 0.0,
                'necrotic_ratio': 0.0,
                'epithelium_ratio': 0.0
            }
        
        # Convert to HSV for better color analysis
        # Need to reshape for cv2
        wound_pixels_reshaped = wound_pixels.reshape(1, -1, 3).astype(np.uint8)
        hsv_pixels = cv2.cvtColor(wound_pixels_reshaped, cv2.COLOR_RGB2HSV)
        hsv_pixels = hsv_pixels.reshape(-1, 3)
        
        h, s, v = hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2]
        
        total = len(wound_pixels)
        
        # Granulation: Red/pink, saturated (H < 20 or H > 160, S > 80)
        granulation_mask = ((h < 20) | (h > 160)) & (s > 80) & (v > 80)
        granulation_ratio = granulation_mask.sum() / total
        
        # Slough: Yellow/cream (H 15-45, moderate S)
        slough_mask = (h >= 15) & (h <= 45) & (s > 40) & (s < 180)
        slough_ratio = slough_mask.sum() / total
        
        # Necrotic: Dark regions (V < 80)
        necrotic_mask = v < 80
        necrotic_ratio = necrotic_mask.sum() / total
        
        # Epithelium: Pink/skin-tone (H 0-25, low-moderate S, high V)
        epithelium_mask = (h < 25) & (s < 80) & (v > 150)
        epithelium_ratio = epithelium_mask.sum() / total
        
        # Normalize to sum to 1 (approximately)
        total_ratio = granulation_ratio + slough_ratio + necrotic_ratio + epithelium_ratio
        if total_ratio > 0:
            granulation_ratio /= total_ratio
            slough_ratio /= total_ratio
            necrotic_ratio /= total_ratio
            epithelium_ratio /= total_ratio
        
        return {
            'granulation_ratio': float(granulation_ratio),
            'slough_ratio': float(slough_ratio),
            'necrotic_ratio': float(necrotic_ratio),
            'epithelium_ratio': float(epithelium_ratio)
        }
    
    def extract_features(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray],
        filename: str,
        image_id: int
    ) -> WoundFeatures:
        """Extract all features from a single image"""
        # CNN features
        cnn_features = self.extract_cnn_features(image)
        
        # Color features
        color_features = self.extract_color_features(image, mask)
        
        return WoundFeatures(
            filename=filename,
            image_id=image_id,
            cnn_features=cnn_features,
            wound_area_ratio=color_features['wound_area_ratio'],
            mean_red=color_features['mean_red'],
            mean_green=color_features['mean_green'],
            mean_blue=color_features['mean_blue'],
            granulation_ratio=color_features['granulation_ratio'],
            slough_ratio=color_features['slough_ratio'],
            necrotic_ratio=color_features['necrotic_ratio'],
            epithelium_ratio=color_features['epithelium_ratio']
        )


# ============================================================================
# HEALING STAGE ESTIMATION
# ============================================================================

class HealingStageEstimator:
    """
    Estimates healing stage (0-4) based on wound appearance
    
    Stages:
    0 - Near healed (mostly epithelium, minimal wound)
    1 - Good healing (mostly granulation)
    2 - Moderate (mixed tissue, some slough)
    3 - Poor (significant slough/necrotic)
    4 - Critical (large necrotic areas)
    """
    
    def estimate_stage(self, features: WoundFeatures) -> int:
        """
        Estimate healing stage from wound features
        
        Uses tissue ratios and wound size to determine stage
        """
        necrotic = features.necrotic_ratio
        slough = features.slough_ratio
        granulation = features.granulation_ratio
        epithelium = features.epithelium_ratio
        area = features.wound_area_ratio
        
        # Rule-based classification
        if necrotic > 0.35:
            return 4  # Critical - significant necrosis
        elif necrotic > 0.2 or slough > 0.45:
            return 3  # Poor - notable necrosis or heavy slough
        elif slough > 0.25 or (necrotic > 0.1 and slough > 0.15):
            return 2  # Moderate - some slough/necrosis
        elif granulation > 0.35 or (granulation > 0.2 and epithelium < 0.3):
            return 1  # Good - healing with granulation
        elif epithelium > 0.4 or area < 0.05:
            return 0  # Near healed
        else:
            # Default to moderate if unclear
            return 2
    
    def estimate_all(self, features_list: List[WoundFeatures]) -> List[WoundFeatures]:
        """Estimate healing stage for all features"""
        for features in features_list:
            features.healing_stage = self.estimate_stage(features)
        return features_list


# ============================================================================
# WOUND CLUSTERING
# ============================================================================

class WoundClusterer:
    """
    Clusters wounds by visual similarity to create pseudo-patient groups
    """
    
    def __init__(
        self,
        n_clusters: int = 40,
        min_cluster_size: int = 5,
        pca_components: int = 50
    ):
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size
        self.pca_components = pca_components
        
        self.pca = None
        self.scaler = None
        self.clustering = None
    
    def _prepare_features(self, features_list: List[WoundFeatures]) -> np.ndarray:
        """Prepare feature matrix for clustering"""
        feature_matrix = []
        
        for f in features_list:
            # Combine CNN features with wound-specific features
            wound_features = np.array([
                f.wound_area_ratio,
                f.mean_red,
                f.mean_green,
                f.mean_blue,
                f.granulation_ratio,
                f.slough_ratio,
                f.necrotic_ratio,
                f.epithelium_ratio
            ])
            
            # Weight wound-specific features more
            wound_features_weighted = wound_features * 5.0
            
            # Combine
            combined = np.concatenate([f.cnn_features, wound_features_weighted])
            feature_matrix.append(combined)
        
        return np.array(feature_matrix)
    
    def cluster(self, features_list: List[WoundFeatures]) -> Dict[int, List[int]]:
        """
        Cluster wounds into pseudo-patient groups
        
        Args:
            features_list: List of WoundFeatures
            
        Returns:
            Dictionary mapping cluster_id to list of feature indices
        """
        print(f"Clustering {len(features_list)} wound images...")
        
        # Prepare features
        feature_matrix = self._prepare_features(features_list)
        
        # Scale features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(feature_matrix)
        
        # Reduce dimensionality
        n_components = min(self.pca_components, len(features_list) - 1, scaled_features.shape[1])
        self.pca = PCA(n_components=n_components)
        reduced_features = self.pca.fit_transform(scaled_features)
        
        print(f"  Reduced features to {n_components} dimensions")
        print(f"  Explained variance: {self.pca.explained_variance_ratio_.sum():.2%}")
        
        # Adjust number of clusters if needed
        actual_n_clusters = min(self.n_clusters, len(features_list) // self.min_cluster_size)
        actual_n_clusters = max(actual_n_clusters, 5)  # At least 5 clusters
        
        # Hierarchical clustering
        self.clustering = AgglomerativeClustering(
            n_clusters=actual_n_clusters,
            linkage='ward'
        )
        cluster_labels = self.clustering.fit_predict(reduced_features)
        
        # Group by cluster
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)
        
        # Report
        sizes = [len(v) for v in clusters.values()]
        print(f"  Created {len(clusters)} clusters")
        print(f"  Cluster sizes: min={min(sizes)}, max={max(sizes)}, avg={np.mean(sizes):.1f}")
        
        return clusters


# ============================================================================
# TRAJECTORY BUILDER
# ============================================================================

class TrajectoryBuilder:
    """
    Builds synthetic healing trajectories from clustered wound images
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or (OUTPUT_DIR / "synthetic_longitudinal")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def build_trajectories(
        self,
        clusters: Dict[int, List[int]],
        features_list: List[WoundFeatures],
        images: List[np.ndarray],
        filenames: List[str],
        min_trajectory_length: int = 3
    ) -> List[SyntheticTrajectory]:
        """
        Build synthetic trajectories from clustered images
        
        Within each cluster, sort by healing stage to create
        a progression from worst (stage 4) to best (stage 0)
        
        Args:
            clusters: Cluster ID -> list of image indices
            features_list: List of WoundFeatures for all images
            images: List of image arrays
            filenames: List of filenames
            min_trajectory_length: Minimum frames per trajectory
            
        Returns:
            List of SyntheticTrajectory objects
        """
        trajectories = []
        
        for cluster_id, indices in clusters.items():
            # Skip small clusters
            if len(indices) < min_trajectory_length:
                continue
            
            # Get features for this cluster
            cluster_features = [(idx, features_list[idx]) for idx in indices]
            
            # Sort by healing stage (worst to best for healing progression)
            cluster_features.sort(key=lambda x: x[1].healing_stage, reverse=True)
            
            # Check if cluster has variety of stages (good trajectory)
            stages = [f.healing_stage for _, f in cluster_features]
            stage_variety = len(set(stages))
            
            if stage_variety < 2:
                # All same stage - not a good trajectory, but still use it
                pass
            
            # Create trajectory
            trajectory = SyntheticTrajectory(
                trajectory_id=f"traj_{cluster_id:03d}",
                pseudo_patient_id=f"pseudo_patient_{cluster_id:03d}"
            )
            
            # Build frames
            base_date = datetime(2024, 1, 1)
            
            for frame_idx, (img_idx, features) in enumerate(cluster_features):
                # Simulate timestamps (3-7 days between visits)
                days_elapsed = frame_idx * np.random.randint(3, 8)
                timestamp = base_date + timedelta(days=days_elapsed)
                
                frame = {
                    'frame_idx': frame_idx,
                    'original_filename': filenames[img_idx],
                    'image_id': features.image_id,
                    'healing_stage': features.healing_stage,
                    'severity_level': 4 - features.healing_stage,  # For diffusion (0=healed, 4=critical)
                    'day': days_elapsed,
                    'timestamp': timestamp.isoformat(),
                    'tissue_ratios': {
                        'granulation': features.granulation_ratio,
                        'slough': features.slough_ratio,
                        'necrotic': features.necrotic_ratio,
                        'epithelium': features.epithelium_ratio
                    },
                    'wound_area_ratio': features.wound_area_ratio
                }
                
                trajectory.frames.append(frame)
            
            trajectories.append(trajectory)
        
        print(f"Created {len(trajectories)} synthetic trajectories")
        
        # Summary
        lengths = [len(t.frames) for t in trajectories]
        print(f"  Trajectory lengths: min={min(lengths)}, max={max(lengths)}, avg={np.mean(lengths):.1f}")
        
        return trajectories
    
    def save_trajectories(
        self,
        trajectories: List[SyntheticTrajectory],
        images: List[np.ndarray],
        features_list: List[WoundFeatures],
        filenames: List[str]
    ):
        """
        Save trajectories to disk with images and metadata
        """
        print(f"Saving trajectories to {self.output_dir}...")
        
        # Create directories
        trajectories_dir = self.output_dir / "trajectories"
        trajectories_dir.mkdir(exist_ok=True)
        
        # Create filename to index lookup
        filename_to_idx = {fn: idx for idx, fn in enumerate(filenames)}
        
        # Save each trajectory
        for traj in tqdm(trajectories, desc="Saving trajectories"):
            traj_dir = trajectories_dir / traj.trajectory_id
            traj_dir.mkdir(exist_ok=True)
            
            # Save frames
            for frame in traj.frames:
                # Get original image
                orig_filename = frame['original_filename']
                if orig_filename in filename_to_idx:
                    img_idx = filename_to_idx[orig_filename]
                    image = images[img_idx]
                    
                    # Save with day-based naming
                    output_name = f"day_{frame['day']:03d}.jpg"
                    output_path = traj_dir / output_name
                    
                    # Convert RGB to BGR for cv2
                    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # Save trajectory metadata
            metadata_path = traj_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(traj.to_dict(), f, indent=2)
        
        # Save global metadata
        all_trajectories_meta = {
            'num_trajectories': len(trajectories),
            'created_at': datetime.now().isoformat(),
            'source_dataset': 'CO2Wounds-V2',
            'trajectories': [t.to_dict() for t in trajectories]
        }
        
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(all_trajectories_meta, f, indent=2)
        
        # Create training pairs (before -> after)
        self._create_training_pairs(trajectories, images, filenames)
        
        print(f"Saved {len(trajectories)} trajectories to {self.output_dir}")
    
    def _create_training_pairs(
        self,
        trajectories: List[SyntheticTrajectory],
        images: List[np.ndarray],
        filenames: List[str]
    ):
        """Create training pairs for diffusion model"""
        
        pairs_dir = self.output_dir / "training_pairs"
        pairs_dir.mkdir(exist_ok=True)
        
        filename_to_idx = {fn: idx for idx, fn in enumerate(filenames)}
        
        pairs = []
        pair_id = 0
        
        for traj in trajectories:
            frames = traj.frames
            
            # Create pairs from consecutive frames and skip connections
            for i in range(len(frames)):
                for j in range(i + 1, min(i + 4, len(frames))):  # Up to 3 frames ahead
                    before_frame = frames[i]
                    after_frame = frames[j]
                    
                    before_file = before_frame['original_filename']
                    after_file = after_frame['original_filename']
                    
                    if before_file in filename_to_idx and after_file in filename_to_idx:
                        pair = {
                            'pair_id': pair_id,
                            'trajectory_id': traj.trajectory_id,
                            'before_file': before_file,
                            'after_file': after_file,
                            'before_severity': before_frame['severity_level'],
                            'after_severity': after_frame['severity_level'],
                            'severity_delta': before_frame['severity_level'] - after_frame['severity_level'],
                            'days_elapsed': after_frame['day'] - before_frame['day']
                        }
                        pairs.append(pair)
                        pair_id += 1
        
        # Save pairs metadata
        pairs_meta = {
            'num_pairs': len(pairs),
            'pairs': pairs
        }
        
        pairs_path = pairs_dir / "pairs.json"
        with open(pairs_path, 'w') as f:
            json.dump(pairs_meta, f, indent=2)
        
        print(f"  Created {len(pairs)} training pairs")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class LongitudinalSimulator:
    """
    Main class that orchestrates the longitudinal data simulation pipeline
    """
    
    def __init__(
        self,
        dataset_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        n_clusters: int = 40,
        device=None
    ):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir or (OUTPUT_DIR / "synthetic_longitudinal")
        self.n_clusters = n_clusters
        self.device = device or DEVICE
        
        # Components
        self.feature_extractor = None
        self.stage_estimator = HealingStageEstimator()
        self.clusterer = WoundClusterer(n_clusters=n_clusters)
        self.trajectory_builder = TrajectoryBuilder(output_dir=self.output_dir)
        
        # Data
        self.dataset = None
        self.images = None
        self.masks = None
        self.filenames = None
        self.features_list = None
        self.clusters = None
        self.trajectories = None
    
    def run(self, save_features: bool = True) -> List[SyntheticTrajectory]:
        """
        Run the complete longitudinal simulation pipeline
        
        Returns:
            List of synthetic trajectories
        """
        print("=" * 60)
        print("LONGITUDINAL DATA SIMULATION PIPELINE")
        print("=" * 60)
        
        # Step 1: Load dataset
        print("\n[1/5] Loading CO2Wounds-V2 dataset...")
        self.dataset = CO2WoundsDataset(root_dir=self.dataset_dir)
        self.images, self.masks, self.filenames = self.dataset.get_all_images_and_masks()
        print(f"  Loaded {len(self.images)} images")
        
        # Step 2: Extract features
        print("\n[2/5] Extracting features...")
        self.feature_extractor = WoundFeatureExtractor(device=self.device)
        self.features_list = []
        
        for idx in tqdm(range(len(self.images)), desc="Extracting features"):
            features = self.feature_extractor.extract_features(
                image=self.images[idx],
                mask=self.masks[idx],
                filename=self.filenames[idx],
                image_id=idx
            )
            self.features_list.append(features)
        
        # Save features if requested
        if save_features:
            self._save_features()
        
        # Step 3: Estimate healing stages
        print("\n[3/5] Estimating healing stages...")
        self.features_list = self.stage_estimator.estimate_all(self.features_list)
        
        # Report stage distribution
        stages = [f.healing_stage for f in self.features_list]
        for stage in range(5):
            count = stages.count(stage)
            print(f"  Stage {stage}: {count} images ({count/len(stages)*100:.1f}%)")
        
        # Step 4: Cluster wounds
        print("\n[4/5] Clustering wounds...")
        self.clusters = self.clusterer.cluster(self.features_list)
        
        # Step 5: Build trajectories
        print("\n[5/5] Building trajectories...")
        self.trajectories = self.trajectory_builder.build_trajectories(
            clusters=self.clusters,
            features_list=self.features_list,
            images=self.images,
            filenames=self.filenames
        )
        
        # Save trajectories
        self.trajectory_builder.save_trajectories(
            trajectories=self.trajectories,
            images=self.images,
            features_list=self.features_list,
            filenames=self.filenames
        )
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print(f"Output saved to: {self.output_dir}")
        print("=" * 60)
        
        return self.trajectories
    
    def _save_features(self):
        """Save extracted features to disk"""
        features_path = self.output_dir / "features.pkl"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to serializable format
        features_data = []
        for f in self.features_list:
            data = {
                'filename': f.filename,
                'image_id': f.image_id,
                'cnn_features': f.cnn_features,
                'wound_area_ratio': f.wound_area_ratio,
                'mean_red': f.mean_red,
                'mean_green': f.mean_green,
                'mean_blue': f.mean_blue,
                'granulation_ratio': f.granulation_ratio,
                'slough_ratio': f.slough_ratio,
                'necrotic_ratio': f.necrotic_ratio,
                'epithelium_ratio': f.epithelium_ratio,
                'healing_stage': f.healing_stage
            }
            features_data.append(data)
        
        with open(features_path, 'wb') as f:
            pickle.dump(features_data, f)
        
        print(f"  Features saved to {features_path}")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Simulate longitudinal wound data from CO2Wounds-V2"
    )
    parser.add_argument(
        "--n-clusters", "-n",
        type=int,
        default=40,
        help="Number of clusters (pseudo-patients)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    simulator = LongitudinalSimulator(
        n_clusters=args.n_clusters,
        output_dir=Path(args.output) if args.output else None
    )
    
    trajectories = simulator.run()
    
    print(f"\nGenerated {len(trajectories)} synthetic trajectories")
