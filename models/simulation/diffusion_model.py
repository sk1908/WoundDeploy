"""
Conditional Diffusion Model for Healing Trajectory Simulation
Generates future wound states based on severity conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import math

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import diffusion_config, DEVICE


@dataclass
class GeneratedImage:
    """Generated wound image result"""
    image: np.ndarray
    severity_level: int
    severity_name: str
    confidence: float


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion models"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ConvBlock(nn.Module):
    """Convolutional block with normalization and activation"""
    
    def __init__(self, in_channels: int, out_channels: int, time_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # Add time embedding
        time_emb = self.time_mlp(t)[:, :, None, None]
        h = h + time_emb
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block"""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        x_flat = x_norm.view(b, c, -1).permute(0, 2, 1)  # (B, H*W, C)
        
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        attn_out = attn_out.permute(0, 2, 1).view(b, c, h, w)
        
        return x + attn_out


class UNet(nn.Module):
    """
    U-Net architecture for diffusion denoising
    Includes severity conditioning
    """
    
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 channels: int = 64,
                 num_severity_levels: int = 5,
                 time_dim: int = 256):
        super().__init__()
        
        self.time_dim = time_dim
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Severity embedding
        self.severity_embedding = nn.Embedding(num_severity_levels, time_dim)
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, channels, time_dim)
        self.enc2 = ConvBlock(channels, channels * 2, time_dim)
        self.enc3 = ConvBlock(channels * 2, channels * 4, time_dim)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(channels * 4, channels * 8, time_dim)
        self.attn = AttentionBlock(channels * 8)
        
        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.dec3 = ConvBlock(channels * 8 + channels * 4, channels * 4, time_dim)
        self.dec2 = ConvBlock(channels * 4 + channels * 2, channels * 2, time_dim)
        self.dec1 = ConvBlock(channels * 2 + channels, channels, time_dim)
        
        # Output
        self.out_conv = nn.Conv2d(channels, out_channels, 1)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, severity: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Noisy image (B, C, H, W)
            t: Timestep (B,)
            severity: Severity level (B,)
        """
        # Embeddings
        t_emb = self.time_embedding(t)
        s_emb = self.severity_embedding(severity)
        cond_emb = t_emb + s_emb
        
        # Encoder
        e1 = self.enc1(x, cond_emb)
        e2 = self.enc2(self.pool(e1), cond_emb)
        e3 = self.enc3(self.pool(e2), cond_emb)
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3), cond_emb)
        b = self.attn(b)
        
        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1), cond_emb)
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1), cond_emb)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1), cond_emb)
        
        return self.out_conv(d1)


class DiffusionModel:
    """
    Denoising Diffusion Probabilistic Model for wound healing simulation
    """
    
    def __init__(self,
                 num_steps: int = 1000,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 num_severity_levels: int = None,
                 device: Optional[torch.device] = None):
        self.num_steps = num_steps
        self.device = device or DEVICE
        self.num_severity_levels = num_severity_levels or diffusion_config.num_severity_levels
        
        # Noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)
        
        # Precompute values
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        
        # Model
        self.model = UNet(num_severity_levels=self.num_severity_levels).to(self.device)
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add noise to clean image (forward process)"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t][:, None, None, None]
        sqrt_one_minus_t = self.sqrt_one_minus_alpha_cumprod[t][:, None, None, None]
        
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_t * noise
    
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int, severity: torch.Tensor) -> torch.Tensor:
        """Remove noise from noisy image (one reverse step)"""
        batch_size = x.shape[0]
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
        
        # Predict noise
        predicted_noise = self.model(x, t_tensor, severity)
        
        # Compute mean
        alpha = self.alphas[t]
        alpha_cumprod = self.alpha_cumprod[t]
        beta = self.betas[t]
        
        mean = self.sqrt_recip_alpha[t] * (x - beta / self.sqrt_one_minus_alpha_cumprod[t] * predicted_noise)
        
        # Add noise (except for last step)
        if t > 0:
            noise = torch.randn_like(x)
            variance = torch.sqrt(self.posterior_variance[t])
            return mean + variance * noise
        else:
            return mean
    
    @torch.no_grad()
    def sample(self, 
               shape: Tuple[int, ...],
               severity: torch.Tensor,
               num_steps: Optional[int] = None,
               start_image: Optional[torch.Tensor] = None,
               start_step: Optional[int] = None) -> torch.Tensor:
        """
        Generate image from noise
        
        Args:
            shape: Output shape (B, C, H, W)
            severity: Target severity level
            num_steps: Number of denoising steps
            start_image: Optional starting image for interpolation
            start_step: Step to start from if using start_image
        """
        num_steps = num_steps or diffusion_config.num_inference_steps
        step_size = self.num_steps // num_steps
        
        if start_image is not None and start_step is not None:
            # Start from partially noised image
            t_start = torch.tensor([start_step], device=self.device)
            x = self.q_sample(start_image, t_start)
            timesteps = list(range(start_step, 0, -step_size))
        else:
            # Start from pure noise
            x = torch.randn(shape, device=self.device)
            timesteps = list(range(self.num_steps - 1, 0, -step_size))
        
        self.model.eval()
        
        for t in timesteps:
            x = self.p_sample(x, t, severity)
        
        return x
    
    def training_step(self, 
                      images: torch.Tensor,
                      severity: torch.Tensor) -> torch.Tensor:
        """
        Single training step
        
        Args:
            images: Clean images (B, C, H, W) normalized to [-1, 1]
            severity: Severity labels (B,)
            
        Returns:
            Loss value
        """
        batch_size = images.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.num_steps, (batch_size,), device=self.device)
        
        # Add noise
        noise = torch.randn_like(images)
        noisy_images = self.q_sample(images, t, noise)
        
        # Predict noise
        predicted_noise = self.model(noisy_images, t, severity)
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    def save(self, path: Path):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: Path):
        """Load model weights"""
        if path.exists():
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Loaded diffusion model from {path}")


class HealingDiffusion:
    """
    High-level interface for healing trajectory generation
    """
    
    def __init__(self,
                 weights_path: Optional[Path] = None,
                 device: Optional[torch.device] = None):
        self.weights_path = weights_path or diffusion_config.weights_path
        self.device = device or DEVICE
        self.img_size = diffusion_config.img_size
        
        self.diffusion = None
        self._is_loaded = False
        
        self.severity_names = ["healed", "mild", "moderate", "severe", "critical"]
    
    def load(self) -> bool:
        """Load model"""
        try:
            self.diffusion = DiffusionModel(device=self.device)
            
            if self.weights_path.exists():
                self.diffusion.load(self.weights_path)
            else:
                print("No trained diffusion model, using random initialization")
            
            self._is_loaded = True
            return True
        except Exception as e:
            print(f"Error loading diffusion model: {e}")
            return False
    
    def generate_for_severity(self,
                               target_severity: int,
                               num_images: int = 1) -> List[GeneratedImage]:
        """
        Generate wound images for a specific severity level
        
        Args:
            target_severity: 0 (healed) to 4 (critical)
            num_images: Number of images to generate
        """
        if not self._is_loaded:
            self.load()
        
        shape = (num_images, 3, self.img_size[0], self.img_size[1])
        severity = torch.full((num_images,), target_severity, device=self.device, dtype=torch.long)
        
        generated = self.diffusion.sample(shape, severity)
        
        results = []
        for i in range(num_images):
            # Convert to image
            img = generated[i].cpu().numpy()
            img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
            img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
            
            results.append(GeneratedImage(
                image=img,
                severity_level=target_severity,
                severity_name=self.severity_names[target_severity],
                confidence=0.8  # Placeholder
            ))
        
        return results
    
    def generate_trajectory(self,
                             start_image: np.ndarray,
                             start_severity: int,
                             end_severity: int,
                             num_steps: int = 5) -> List[GeneratedImage]:
        """
        Generate healing trajectory from start to end severity
        
        Args:
            start_image: Starting wound image
            start_severity: Current severity level
            end_severity: Target severity level
            num_steps: Number of intermediate steps
            
        Returns:
            List of generated images showing progression
        """
        if not self._is_loaded:
            self.load()
        
        results = []
        
        # Preprocess start image
        img = cv2.resize(start_image, self.img_size)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        img_tensor = (img_tensor / 127.5) - 1  # Normalize to [-1, 1]
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Generate intermediate severity levels
        severity_range = np.linspace(start_severity, end_severity, num_steps).astype(int)
        
        current_img = img_tensor
        
        for severity in severity_range:
            severity_tensor = torch.tensor([severity], device=self.device, dtype=torch.long)
            
            # Generate with mild conditioning
            start_step = self.diffusion.num_steps // 2  # Start halfway
            generated = self.diffusion.sample(
                current_img.shape,
                severity_tensor,
                start_image=current_img,
                start_step=start_step
            )
            
            # Convert to image
            img_np = generated[0].cpu().numpy()
            img_np = ((img_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
            img_np = np.transpose(img_np, (1, 2, 0))
            
            results.append(GeneratedImage(
                image=img_np,
                severity_level=int(severity),
                severity_name=self.severity_names[int(severity)],
                confidence=0.7
            ))
            
            current_img = generated
        
        return results
    
    def warmup(self, iterations: int = 1):
        """Warmup model"""
        if not self._is_loaded:
            self.load()
        
        dummy = np.zeros((512, 512, 3), dtype=np.uint8)
        for _ in range(iterations):
            self.generate_trajectory(dummy, 3, 2, num_steps=2)
        
        print("Diffusion model warmup complete")


if __name__ == "__main__":
    # Test diffusion model
    model = DiffusionModel()
    print("Diffusion model created")
    
    # Test forward pass
    x = torch.randn(1, 3, 64, 64).to(DEVICE)
    t = torch.tensor([100]).to(DEVICE)
    severity = torch.tensor([2]).to(DEVICE)
    
    noise_pred = model.model(x, t, severity)
    print(f"Predicted noise shape: {noise_pred.shape}")
    
    # Test sampling
    print("Testing sampling...")
    samples = model.sample((1, 3, 64, 64), severity, num_steps=10)
    print(f"Generated sample shape: {samples.shape}")
