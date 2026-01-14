"""
Trajectory Generation Module
Generates multi-step healing trajectories and animations
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import diffusion_config, DEVICE
from .diffusion_model import HealingDiffusion, GeneratedImage


@dataclass
class TrajectoryFrame:
    """Single frame in healing trajectory"""
    image: np.ndarray
    day: int
    severity_level: int
    severity_name: str
    tissue_change: Dict[str, float]  # Predicted tissue composition
    volume_change: float


class TrajectoryGenerator:
    """
    Generates and visualizes healing trajectories
    """
    
    def __init__(self,
                 weights_path: Optional[Path] = None,
                 device = None):
        self.diffusion = HealingDiffusion(weights_path=weights_path, device=device)
        
        # Default healing rates (days to improve one severity level)
        self.severity_healing_rates = {
            "critical": 14,    # Days to go from critical to severe
            "severe": 10,
            "moderate": 7,
            "mild": 5
        }
        
        # Tissue composition templates for each severity
        self.tissue_templates = {
            4: {"granulation": 0.1, "slough": 0.3, "necrotic": 0.5, "epithelium": 0.1},
            3: {"granulation": 0.2, "slough": 0.4, "necrotic": 0.3, "epithelium": 0.1},
            2: {"granulation": 0.4, "slough": 0.3, "necrotic": 0.1, "epithelium": 0.2},
            1: {"granulation": 0.5, "slough": 0.1, "necrotic": 0.0, "epithelium": 0.4},
            0: {"granulation": 0.1, "slough": 0.0, "necrotic": 0.0, "epithelium": 0.9}
        }
    
    def load(self) -> bool:
        """Load the diffusion model"""
        return self.diffusion.load()
    
    def generate_trajectory(self,
                             start_image: np.ndarray,
                             start_severity: int,
                             treatment_scenario: str = "optimal",
                             num_days: int = 30,
                             frames_per_day: float = 0.5) -> List[TrajectoryFrame]:
        """
        Generate a complete healing trajectory
        
        Args:
            start_image: Initial wound image
            start_severity: Starting severity (0-4)
            treatment_scenario: "optimal", "standard", or "suboptimal"
            num_days: Total days to simulate
            frames_per_day: Generate this many frames per day
            
        Returns:
            List of TrajectoryFrame objects
        """
        # Adjust healing rates based on treatment scenario
        rate_multiplier = {
            "optimal": 0.7,
            "standard": 1.0,
            "suboptimal": 1.5
        }.get(treatment_scenario, 1.0)
        
        adjusted_rates = {
            k: int(v * rate_multiplier)
            for k, v in self.severity_healing_rates.items()
        }
        
        # Generate trajectory
        frames = []
        current_severity = start_severity
        current_image = start_image.copy()
        day = 0
        
        # First frame
        frames.append(TrajectoryFrame(
            image=current_image,
            day=0,
            severity_level=current_severity,
            severity_name=self.diffusion.severity_names[current_severity],
            tissue_change=self.tissue_templates[current_severity],
            volume_change=0.0
        ))
        
        while day < num_days and current_severity > 0:
            # Calculate days until next severity level
            severity_name = self.diffusion.severity_names[current_severity]
            days_to_improve = adjusted_rates.get(severity_name, 7)
            
            # Generate intermediate frames
            num_intermediate = int(days_to_improve * frames_per_day)
            intermediate_days = np.linspace(day, day + days_to_improve, num_intermediate + 1)[1:]
            
            for inter_day in intermediate_days:
                if inter_day > num_days:
                    break
                
                # Interpolate tissue composition
                progress = (inter_day - day) / days_to_improve
                current_tissue = self._interpolate_tissue(
                    self.tissue_templates[current_severity],
                    self.tissue_templates[max(0, current_severity - 1)],
                    progress
                )
                
                # Estimate volume change
                volume_change = -progress * 0.1 * (current_severity / 4)
                
                frames.append(TrajectoryFrame(
                    image=current_image,  # Will be updated with diffusion
                    day=int(inter_day),
                    severity_level=current_severity,
                    severity_name=severity_name,
                    tissue_change=current_tissue,
                    volume_change=volume_change
                ))
            
            # Update for next iteration
            day += days_to_improve
            current_severity = max(0, current_severity - 1)
            
            # Generate image for new severity level using diffusion
            if current_severity >= 0:
                generated = self.diffusion.generate_for_severity(current_severity, 1)
                if generated:
                    current_image = generated[0].image
        
        return frames
    
    def generate_quick_trajectory(self,
                                    start_image: np.ndarray,
                                    start_severity: int,
                                    end_severity: int = 0,
                                    num_frames: int = 5) -> List[GeneratedImage]:
        """
        Generate a quick trajectory with specified number of frames
        
        This uses the diffusion model directly for faster generation
        """
        return self.diffusion.generate_trajectory(
            start_image=start_image,
            start_severity=start_severity,
            end_severity=end_severity,
            num_steps=num_frames
        )
    
    def _interpolate_tissue(self,
                             tissue_a: Dict[str, float],
                             tissue_b: Dict[str, float],
                             progress: float) -> Dict[str, float]:
        """Interpolate between two tissue compositions"""
        result = {}
        for tissue in tissue_a:
            a = tissue_a.get(tissue, 0)
            b = tissue_b.get(tissue, 0)
            result[tissue] = a + (b - a) * progress
        return result
    
    def create_animation(self,
                          frames: List[TrajectoryFrame],
                          output_path: Path,
                          fps: int = 2,
                          add_labels: bool = True) -> Path:
        """
        Create video animation from trajectory frames
        
        Args:
            frames: List of trajectory frames
            output_path: Output video path
            fps: Frames per second
            add_labels: Add text labels to frames
            
        Returns:
            Path to created video
        """
        if not frames:
            return None
        
        # Get frame size
        h, w = frames[0].image.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        
        for frame in frames:
            img = frame.image.copy()
            
            if add_labels:
                # Add day label
                cv2.putText(img, f"Day {frame.day}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add severity label
                color = self._get_severity_color(frame.severity_level)
                cv2.putText(img, frame.severity_name.upper(), (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            writer.write(img)
        
        writer.release()
        print(f"Animation saved to {output_path}")
        return output_path
    
    def _get_severity_color(self, severity: int) -> Tuple[int, int, int]:
        """Get color for severity level (BGR)"""
        colors = {
            0: (0, 255, 0),    # Green - healed
            1: (255, 255, 0),  # Cyan - mild
            2: (0, 255, 255),  # Yellow - moderate
            3: (0, 165, 255),  # Orange - severe
            4: (0, 0, 255)     # Red - critical
        }
        return colors.get(severity, (255, 255, 255))
    
    def create_comparison_grid(self,
                                frames: List[TrajectoryFrame],
                                num_cols: int = 5) -> np.ndarray:
        """
        Create a grid image showing trajectory progression
        """
        if not frames:
            return None
        
        # Select evenly spaced frames
        if len(frames) > num_cols:
            indices = np.linspace(0, len(frames) - 1, num_cols).astype(int)
            selected = [frames[i] for i in indices]
        else:
            selected = frames
        
        # Get frame size
        h, w = selected[0].image.shape[:2]
        target_w = 200
        scale = target_w / w
        target_h = int(h * scale)
        
        # Create grid
        num_frames = len(selected)
        grid = np.zeros((target_h + 60, target_w * num_frames, 3), dtype=np.uint8)
        
        for i, frame in enumerate(selected):
            x_start = i * target_w
            
            # Resize and place image
            resized = cv2.resize(frame.image, (target_w, target_h))
            grid[:target_h, x_start:x_start + target_w] = resized
            
            # Add label
            cv2.putText(grid, f"Day {frame.day}", (x_start + 10, target_h + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(grid, frame.severity_name, (x_start + 10, target_h + 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, 
                       self._get_severity_color(frame.severity_level), 1)
        
        return grid
    
    def save_trajectory(self,
                         frames: List[TrajectoryFrame],
                         output_dir: Path):
        """Save trajectory frames and metadata"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = []
        
        for i, frame in enumerate(frames):
            # Save image
            img_path = output_dir / f"frame_{i:03d}.jpg"
            cv2.imwrite(str(img_path), frame.image)
            
            metadata.append({
                "frame": i,
                "image": str(img_path),
                "day": frame.day,
                "severity_level": frame.severity_level,
                "severity_name": frame.severity_name,
                "tissue_change": frame.tissue_change,
                "volume_change": frame.volume_change
            })
        
        # Save metadata
        with open(output_dir / "trajectory.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Create comparison grid
        grid = self.create_comparison_grid(frames)
        if grid is not None:
            cv2.imwrite(str(output_dir / "trajectory_grid.jpg"), grid)
        
        print(f"Saved {len(frames)} frames to {output_dir}")


if __name__ == "__main__":
    # Test trajectory generation
    generator = TrajectoryGenerator()
    
    # Create dummy starting image
    start_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Generate trajectory (without loading model for test)
    print("Testing trajectory frame generation...")
    
    # Use simulated trajectory based on templates
    frames = []
    for day in range(0, 30, 5):
        severity = max(0, 4 - day // 7)
        frames.append(TrajectoryFrame(
            image=start_image,
            day=day,
            severity_level=severity,
            severity_name=["healed", "mild", "moderate", "severe", "critical"][severity],
            tissue_change=generator.tissue_templates[severity],
            volume_change=-day * 0.02
        ))
    
    print(f"Generated {len(frames)} frames")
    
    # Create grid
    grid = generator.create_comparison_grid(frames)
    if grid is not None:
        print(f"Comparison grid shape: {grid.shape}")
