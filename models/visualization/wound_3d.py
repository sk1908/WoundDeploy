"""
3D Wound Visualization Module
Generates interactive 3D mesh from depth maps for wound visualization
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import plotly.graph_objects as go


@dataclass
class Mesh3DResult:
    """Result of 3D mesh generation"""
    vertices: np.ndarray  # (N, 3) array of vertex positions
    faces: np.ndarray  # (M, 3) array of triangle indices
    colors: np.ndarray  # (N, 3) array of RGB colors
    plotly_figure: go.Figure  # Interactive Plotly figure
    stats: Dict  # Mesh statistics


class Wound3DVisualizer:
    """
    Converts depth maps to interactive 3D visualizations
    """
    
    def __init__(self, 
                 depth_scale: float = 5.0,
                 mesh_resolution: int = 128,
                 invert_depth: bool = True):
        """
        Args:
            depth_scale: Scale factor for depth values (higher = more exaggerated depth)
            mesh_resolution: Resolution of the mesh grid (lower = faster, higher = more detail)
            invert_depth: If True, treats higher depth values as deeper (cavity)
        """
        self.depth_scale = depth_scale
        self.mesh_resolution = mesh_resolution
        self.invert_depth = invert_depth
    
    def generate_mesh(self, 
                      depth_map: np.ndarray, 
                      rgb_image: np.ndarray,
                      wound_mask: Optional[np.ndarray] = None) -> Mesh3DResult:
        """
        Generate 3D mesh from depth map
        
        Args:
            depth_map: 2D array of depth values (H, W) normalized 0-1
            rgb_image: RGB image for texture (H, W, 3)
            wound_mask: Optional binary mask for wound region
            
        Returns:
            Mesh3DResult with vertices, faces, colors, and Plotly figure
        """
        # Resize to mesh resolution
        h, w = self.mesh_resolution, self.mesh_resolution
        depth_resized = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb_resized = cv2.resize(rgb_image, (w, h), interpolation=cv2.INTER_LINEAR)
        
        if wound_mask is not None:
            mask_resized = cv2.resize(wound_mask.astype(np.float32), (w, h), 
                                      interpolation=cv2.INTER_NEAREST) > 0.5
        else:
            mask_resized = np.ones((h, w), dtype=bool)
        
        # Normalize depth
        depth_norm = depth_resized.astype(np.float32)
        if depth_norm.max() > 0:
            depth_norm = (depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min() + 1e-8)
        
        # Invert if needed (so wounds appear as cavities)
        if self.invert_depth:
            depth_norm = 1.0 - depth_norm
        
        # Create grid coordinates
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        X, Y = np.meshgrid(x, y)
        
        # Apply depth scaling
        Z = depth_norm * self.depth_scale
        
        # Apply mask - set non-wound areas to base level
        if wound_mask is not None:
            Z[~mask_resized] = Z.max()  # Flatten non-wound to top surface
        
        # Generate vertices
        vertices = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        # Generate faces (triangles)
        faces = []
        for i in range(h - 1):
            for j in range(w - 1):
                idx = i * w + j
                # Two triangles per quad
                faces.append([idx, idx + 1, idx + w])
                faces.append([idx + 1, idx + w + 1, idx + w])
        faces = np.array(faces)
        
        # Get colors from RGB image
        colors = rgb_resized.reshape(-1, 3) / 255.0
        
        # Create Plotly figure
        fig = self._create_plotly_figure(X, Y, Z, rgb_resized, mask_resized)
        
        # Compute statistics
        stats = {
            "max_depth": float(depth_norm.max() * self.depth_scale),
            "mean_depth": float(depth_norm.mean() * self.depth_scale),
            "wound_area_ratio": float(mask_resized.sum() / mask_resized.size) if wound_mask is not None else 1.0,
            "mesh_vertices": len(vertices),
            "mesh_faces": len(faces)
        }
        
        return Mesh3DResult(
            vertices=vertices,
            faces=faces,
            colors=colors,
            plotly_figure=fig,
            stats=stats
        )
    
    def _create_plotly_figure(self, 
                               X: np.ndarray, 
                               Y: np.ndarray, 
                               Z: np.ndarray,
                               rgb: np.ndarray,
                               mask: np.ndarray) -> go.Figure:
        """Create interactive Plotly 3D surface"""
        
        # Create color array for surface
        h, w = rgb.shape[:2]
        surfacecolor = np.zeros((h, w))
        
        # Use grayscale intensity for surface shading
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        surfacecolor = gray / 255.0
        
        # Create custom colorscale from wound colors
        # Map the RGB image to the surface
        colors_flat = []
        for i in range(h):
            for j in range(w):
                r, g, b = rgb[i, j]
                colors_flat.append(f'rgb({r},{g},{b})')
        
        # Create surface plot
        fig = go.Figure()
        
        # Add wound surface
        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=surfacecolor,
            colorscale=[
                [0, 'rgb(50,50,50)'],
                [0.2, 'rgb(139,69,19)'],  # Brown (necrotic-like)
                [0.4, 'rgb(178,34,34)'],   # Red (granulation)
                [0.6, 'rgb(255,182,193)'], # Pink (healing)
                [0.8, 'rgb(255,218,185)'], # Peach (skin)
                [1, 'rgb(255,228,196)']    # Light skin
            ],
            showscale=False,
            opacity=0.95,
            lighting=dict(
                ambient=0.6,
                diffuse=0.8,
                specular=0.3,
                roughness=0.5
            ),
            lightposition=dict(x=0, y=0, z=1000),
            hovertemplate="Depth: %{z:.2f}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>"
        ))
        
        # Update layout for better 3D viewing
        fig.update_layout(
            title=dict(
                text="🔬 3D Wound Surface Visualization",
                font=dict(size=20, color='white'),
                x=0.5
            ),
            scene=dict(
                xaxis=dict(
                    title="Width",
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.2)',
                    showbackground=True,
                    backgroundcolor='rgb(20,20,30)'
                ),
                yaxis=dict(
                    title="Height",
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.2)',
                    showbackground=True,
                    backgroundcolor='rgb(20,20,30)'
                ),
                zaxis=dict(
                    title="Depth",
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.2)',
                    showbackground=True,
                    backgroundcolor='rgb(20,20,30)'
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            paper_bgcolor='rgb(15,23,42)',
            plot_bgcolor='rgb(15,23,42)',
            margin=dict(l=0, r=0, t=50, b=0),
            height=500
        )
        
        return fig
    
    def generate_depth_colored_mesh(self,
                                     depth_map: np.ndarray,
                                     wound_mask: Optional[np.ndarray] = None) -> go.Figure:
        """
        Generate 3D mesh colored by depth instead of texture
        
        Args:
            depth_map: 2D array of depth values
            wound_mask: Optional wound mask
            
        Returns:
            Plotly figure with depth-colored surface
        """
        h, w = self.mesh_resolution, self.mesh_resolution
        depth_resized = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
        
        if wound_mask is not None:
            mask_resized = cv2.resize(wound_mask.astype(np.float32), (w, h),
                                      interpolation=cv2.INTER_NEAREST) > 0.5
        else:
            mask_resized = np.ones((h, w), dtype=bool)
        
        # Normalize
        depth_norm = depth_resized.astype(np.float32)
        if depth_norm.max() > 0:
            depth_norm = (depth_norm - depth_norm.min()) / (depth_norm.max() - depth_norm.min() + 1e-8)
        
        if self.invert_depth:
            depth_norm = 1.0 - depth_norm
        
        # Create grid
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        X, Y = np.meshgrid(x, y)
        Z = depth_norm * self.depth_scale
        
        # Apply mask
        if wound_mask is not None:
            Z[~mask_resized] = Z.max()
        
        # Create figure with depth coloring
        fig = go.Figure(data=[go.Surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=Z,
            colorscale='RdYlBu_r',  # Red (deep) to Blue (shallow)
            colorbar=dict(
                title="Depth",
                titlefont=dict(color='white'),
                tickfont=dict(color='white'),
                len=0.5
            ),
            opacity=0.95,
            lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2),
            hovertemplate="Depth: %{z:.3f}<extra></extra>"
        )])
        
        fig.update_layout(
            title=dict(
                text="🔬 3D Wound Depth Map",
                font=dict(size=20, color='white'),
                x=0.5
            ),
            scene=dict(
                xaxis=dict(title="", showticklabels=False, showgrid=False, showbackground=False),
                yaxis=dict(title="", showticklabels=False, showgrid=False, showbackground=False),
                zaxis=dict(title="Depth", gridcolor='rgba(255,255,255,0.2)', showbackground=False),
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.0)),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.4)
            ),
            paper_bgcolor='rgb(15,23,42)',
            margin=dict(l=0, r=0, t=50, b=0),
            height=450
        )
        
        return fig


def create_3d_visualization(depth_map: np.ndarray,
                            rgb_image: np.ndarray,
                            wound_mask: Optional[np.ndarray] = None,
                            depth_scale: float = 5.0,
                            resolution: int = 100) -> Tuple[go.Figure, go.Figure, Dict]:
    """
    Convenience function to create both texture and depth-colored 3D visualizations
    
    Args:
        depth_map: Depth map from Depth Anything
        rgb_image: Original wound RGB image
        wound_mask: Optional segmentation mask
        depth_scale: Exaggeration factor for depth
        resolution: Mesh resolution
        
    Returns:
        Tuple of (textured_figure, depth_colored_figure, stats)
    """
    visualizer = Wound3DVisualizer(
        depth_scale=depth_scale,
        mesh_resolution=resolution,
        invert_depth=True
    )
    
    # Generate textured mesh
    result = visualizer.generate_mesh(depth_map, rgb_image, wound_mask)
    
    # Generate depth-colored mesh
    depth_fig = visualizer.generate_depth_colored_mesh(depth_map, wound_mask)
    
    return result.plotly_figure, depth_fig, result.stats


if __name__ == "__main__":
    # Test with synthetic data
    import matplotlib.pyplot as plt
    
    # Create synthetic depth map (circular wound)
    h, w = 256, 256
    y, x = np.ogrid[:h, :w]
    center = (h // 2, w // 2)
    distance = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    
    # Wound is deeper in center
    radius = 80
    depth_map = np.zeros((h, w))
    mask = distance < radius
    depth_map[mask] = 1 - (distance[mask] / radius)  # Deeper toward center
    
    # Create synthetic RGB (red in center, pink at edges)
    rgb = np.ones((h, w, 3), dtype=np.uint8) * 255
    rgb[mask, 0] = 200  # Red channel
    rgb[mask, 1] = 100 + (distance[mask] / radius * 155).astype(np.uint8)
    rgb[mask, 2] = 100 + (distance[mask] / radius * 155).astype(np.uint8)
    
    # Generate visualization
    textured, depth_colored, stats = create_3d_visualization(
        depth_map, rgb, mask, depth_scale=3.0, resolution=64
    )
    
    print("3D Visualization Stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Show figures
    textured.show()
    depth_colored.show()
