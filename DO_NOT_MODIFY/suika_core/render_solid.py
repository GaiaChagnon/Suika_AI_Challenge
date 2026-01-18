"""
Solid Renderer
==============

Fast numpy-based renderer that draws fruits as solid-color circles.
No pygame dependency required.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
import math
import numpy as np

from DO_NOT_MODIFY.suika_core.config_loader import GameConfig, get_config


class SolidRenderer:
    """
    Renders the game board as solid-color circles.
    
    Uses numpy for fast CPU-based rendering without pygame.
    """
    
    def __init__(self, config: Optional[GameConfig] = None):
        """
        Initialize renderer.
        
        Args:
            config: Game configuration. Uses default if None.
        """
        if config is None:
            config = get_config()
        
        self._config = config
        
        # Background color (dark gray)
        self._bg_color = np.array([30, 30, 40], dtype=np.uint8)
        
        # Wall color
        self._wall_color = np.array([60, 60, 70], dtype=np.uint8)
        
        # Lose line color (red, semi-transparent effect)
        self._lose_line_color = np.array([200, 50, 50], dtype=np.uint8)
    
    def render(
        self,
        render_data: Dict[str, Any],
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Render the game state to an RGB array.
        
        Args:
            render_data: Data from CoreGame.get_render_data().
            width: Output image width.
            height: Output image height.
            
        Returns:
            (height, width, 3) uint8 array.
        """
        # Create image
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = self._bg_color
        
        # Calculate scale
        board_width = render_data["board_width"]
        board_height = render_data["board_height"]
        scale_x = width / board_width
        scale_y = height / board_height
        scale = min(scale_x, scale_y)
        
        # Offset to center
        offset_x = (width - board_width * scale) / 2
        offset_y = (height - board_height * scale) / 2
        
        # Draw walls
        wall_thickness = int(5 * scale)
        # Left wall
        img[:, :wall_thickness] = self._wall_color
        # Right wall
        img[:, -wall_thickness:] = self._wall_color
        # Bottom wall
        img[-wall_thickness:, :] = self._wall_color
        
        # Draw lose line
        lose_y = render_data["lose_line_y"]
        # Convert to image coordinates (flip Y)
        lose_y_img = int(height - (lose_y * scale + offset_y))
        if 0 <= lose_y_img < height:
            # Dashed line effect
            for x in range(0, width, 10):
                end_x = min(x + 5, width)
                img[lose_y_img:lose_y_img+2, x:end_x] = self._lose_line_color
        
        # Draw fruits (sorted by Y so lower fruits are drawn first)
        fruits = sorted(render_data["fruits"], key=lambda f: f["y"])
        
        for fruit in fruits:
            self._draw_fruit(img, fruit, scale, offset_x, offset_y, height)
        
        return img
    
    def _draw_fruit(
        self,
        img: np.ndarray,
        fruit: Dict[str, Any],
        scale: float,
        offset_x: float,
        offset_y: float,
        img_height: int
    ) -> None:
        """Draw a single fruit as a filled circle."""
        height, width = img.shape[:2]
        
        # Get fruit center in image coords
        world_x = fruit["x"]
        world_y = fruit["y"]
        
        # Convert to image coordinates (flip Y axis)
        cx = int(world_x * scale + offset_x)
        cy = int(img_height - (world_y * scale + offset_y))
        
        # Get visual radius scaled
        radius = int(fruit["visual_radius"] * scale)
        
        # Get color
        color = np.array(fruit["color_solid"], dtype=np.uint8)
        
        # Draw filled circle using vectorized numpy
        self._draw_circle(img, cx, cy, radius, color)
        
        # Draw outline (slightly darker)
        outline_color = (color * 0.7).astype(np.uint8)
        self._draw_circle_outline(img, cx, cy, radius, outline_color, 2)
    
    def _draw_circle(
        self,
        img: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
        color: np.ndarray
    ) -> None:
        """Draw a filled circle using numpy."""
        height, width = img.shape[:2]
        
        # Calculate bounding box
        y_min = max(0, cy - radius)
        y_max = min(height, cy + radius + 1)
        x_min = max(0, cx - radius)
        x_max = min(width, cx + radius + 1)
        
        if y_min >= y_max or x_min >= x_max:
            return
        
        # Create coordinate grids for the bounding box
        y_coords = np.arange(y_min, y_max)
        x_coords = np.arange(x_min, x_max)
        
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        # Calculate distance from center
        dist_sq = (xx - cx)**2 + (yy - cy)**2
        
        # Create mask for pixels inside circle
        mask = dist_sq <= radius**2
        
        # Apply color
        img[y_min:y_max, x_min:x_max][mask] = color
    
    def _draw_circle_outline(
        self,
        img: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
        color: np.ndarray,
        thickness: int = 1
    ) -> None:
        """Draw circle outline."""
        height, width = img.shape[:2]
        
        outer_r = radius
        inner_r = max(0, radius - thickness)
        
        y_min = max(0, cy - outer_r)
        y_max = min(height, cy + outer_r + 1)
        x_min = max(0, cx - outer_r)
        x_max = min(width, cx + outer_r + 1)
        
        if y_min >= y_max or x_min >= x_max:
            return
        
        y_coords = np.arange(y_min, y_max)
        x_coords = np.arange(x_min, x_max)
        
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        dist_sq = (xx - cx)**2 + (yy - cy)**2
        
        # Ring mask
        mask = (dist_sq <= outer_r**2) & (dist_sq >= inner_r**2)
        
        img[y_min:y_max, x_min:x_max][mask] = color
    
    def render_to_screen(self, render_data: Dict[str, Any]) -> None:
        """
        Render to screen (no-op for solid renderer).
        
        Use PygameRenderer for screen display.
        """
        pass
    
    def close(self) -> None:
        """Clean up resources (no-op for solid renderer)."""
        pass
