"""
Solid Renderer
==============

Fast numpy-based renderer that draws fruits as solid-color circles.
Shows collision hitboxes, score display, and fruit legend.
Uses OpenCV for text rendering when available.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, List
import math
import numpy as np

from DO_NOT_MODIFY.suika_core.config_loader import GameConfig, get_config

# Try to import cv2 for text rendering
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class SolidRenderer:
    """
    Renders the game board as solid-color circles.
    
    Features:
    - Draws actual collision hitboxes (not visual radius)
    - Shows score display
    - Shows fruit legend with colors
    
    Uses numpy for fast CPU-based rendering without pygame.
    """
    
    def __init__(self, config: Optional[GameConfig] = None, show_legend: bool = True):
        """
        Initialize renderer.
        
        Args:
            config: Game configuration. Uses default if None.
            show_legend: Whether to show the fruit legend.
        """
        if config is None:
            config = get_config()
        
        self._config = config
        self._show_legend = show_legend
        
        # Background color (dark gray)
        self._bg_color = np.array([30, 30, 40], dtype=np.uint8)
        
        # Wall color
        self._wall_color = np.array([60, 60, 70], dtype=np.uint8)
        
        # Lose line color (red, semi-transparent effect)
        self._lose_line_color = np.array([200, 50, 50], dtype=np.uint8)
        
        # UI colors
        self._text_color = (255, 255, 255)  # White
        self._text_shadow = (40, 40, 50)
        
        # Build fruit info for legend
        self._fruit_info = []
        for fruit_cfg in config.fruits:
            self._fruit_info.append({
                "id": fruit_cfg.id,
                "name": fruit_cfg.name,
                "color": fruit_cfg.color_solid,
            })
    
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
        # Reserve space for legend at bottom
        legend_height = 80 if self._show_legend else 0
        game_height = height - legend_height
        
        # Create image
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = self._bg_color
        
        # Calculate scale for game area
        board_width = render_data["board_width"]
        board_height = render_data["board_height"]
        scale_x = width / board_width
        scale_y = game_height / board_height
        scale = min(scale_x, scale_y)
        
        # Offset to center game in game area
        offset_x = (width - board_width * scale) / 2
        offset_y = (game_height - board_height * scale) / 2
        
        # Draw walls
        wall_thickness = int(5 * scale)
        game_left = int(offset_x)
        game_right = int(offset_x + board_width * scale)
        game_bottom = game_height
        
        # Left wall
        img[:game_height, game_left:game_left+wall_thickness] = self._wall_color
        # Right wall
        img[:game_height, game_right-wall_thickness:game_right] = self._wall_color
        # Bottom wall
        img[game_bottom-wall_thickness:game_bottom, game_left:game_right] = self._wall_color
        
        # Draw lose line
        lose_y = render_data["lose_line_y"]
        lose_y_img = int(game_height - (lose_y * scale + offset_y))
        if 0 <= lose_y_img < game_height:
            for x in range(game_left, game_right, 10):
                end_x = min(x + 5, game_right)
                img[lose_y_img:lose_y_img+2, x:end_x] = self._lose_line_color
        
        # Draw fruits (sorted by Y so lower fruits are drawn first)
        fruits = sorted(render_data["fruits"], key=lambda f: f["y"])
        
        for fruit in fruits:
            self._draw_fruit_hitboxes(img, fruit, scale, offset_x, offset_y, game_height)
        
        # Draw score
        score = render_data.get("score", 0)
        drops = render_data.get("drops_used", 0)
        self._draw_score(img, score, drops, width)
        
        # Draw legend
        if self._show_legend:
            self._draw_legend(img, width, height, legend_height)
        
        return img
    
    def _draw_fruit_hitboxes(
        self,
        img: np.ndarray,
        fruit: Dict[str, Any],
        scale: float,
        offset_x: float,
        offset_y: float,
        game_height: int
    ) -> None:
        """Draw a fruit using its actual collision circles (hitboxes)."""
        # Get fruit center and angle
        world_x = fruit["x"]
        world_y = fruit["y"]
        angle = fruit.get("angle", 0.0)
        
        # Get color
        color = np.array(fruit["color_solid"], dtype=np.uint8)
        outline_color = (color * 0.7).astype(np.uint8)
        
        # Get collision circles
        collision_circles = fruit.get("collision_circles", [])
        
        if not collision_circles:
            # Fallback: use visual radius as single circle
            cx = int(world_x * scale + offset_x)
            cy = int(game_height - (world_y * scale + offset_y))
            radius = int(fruit["visual_radius"] * scale)
            self._draw_circle(img, cx, cy, radius, color)
            self._draw_circle_outline(img, cx, cy, radius, outline_color, 2)
            return
        
        # Draw each collision circle, rotated by the fruit's angle
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        for circle_data in collision_circles:
            # circle_data is (radius, offset_x, offset_y)
            circle_radius, local_ox, local_oy = circle_data
            
            # Rotate offset by fruit angle
            rotated_ox = local_ox * cos_a - local_oy * sin_a
            rotated_oy = local_ox * sin_a + local_oy * cos_a
            
            # Calculate circle center in world coords
            circle_world_x = world_x + rotated_ox
            circle_world_y = world_y + rotated_oy
            
            # Convert to image coords
            cx = int(circle_world_x * scale + offset_x)
            cy = int(game_height - (circle_world_y * scale + offset_y))
            radius = int(circle_radius * scale)
            
            # Draw filled circle
            self._draw_circle(img, cx, cy, radius, color)
            # Draw outline
            self._draw_circle_outline(img, cx, cy, radius, outline_color, 2)
    
    def _draw_score(
        self,
        img: np.ndarray,
        score: int,
        drops: int,
        width: int
    ) -> None:
        """Draw score and drops count at top of image."""
        if not CV2_AVAILABLE:
            return
        
        # Score text
        score_text = f"Score: {score}"
        drops_text = f"Drops: {drops}"
        
        # Draw with shadow effect
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Score (top left)
        cv2.putText(img, score_text, (12, 27), font, font_scale, self._text_shadow, thickness + 1)
        cv2.putText(img, score_text, (10, 25), font, font_scale, self._text_color, thickness)
        
        # Drops (top right)
        text_size = cv2.getTextSize(drops_text, font, font_scale, thickness)[0]
        x = width - text_size[0] - 10
        cv2.putText(img, drops_text, (x + 2, 27), font, font_scale, self._text_shadow, thickness + 1)
        cv2.putText(img, drops_text, (x, 25), font, font_scale, self._text_color, thickness)
    
    def _draw_legend(
        self,
        img: np.ndarray,
        width: int,
        height: int,
        legend_height: int
    ) -> None:
        """Draw fruit legend at bottom of image showing ALL fruit types."""
        legend_y = height - legend_height
        
        # Dark background for legend
        img[legend_y:, :] = np.array([20, 20, 25], dtype=np.uint8)
        
        # Draw separator line
        img[legend_y:legend_y+2, :] = np.array([60, 60, 70], dtype=np.uint8)
        
        # Show ALL fruits
        num_fruits = len(self._fruit_info)
        
        # Adjust circle size based on number of fruits
        circle_radius = max(5, min(10, (width - 20) // (num_fruits * 3)))
        spacing = width // (num_fruits + 1)
        y_center = legend_y + legend_height // 2
        
        for i, fruit in enumerate(self._fruit_info):
            x_center = spacing * (i + 1)
            
            # Draw circle
            color = np.array(fruit["color"], dtype=np.uint8)
            self._draw_circle(img, x_center, y_center - 8, circle_radius, color)
            
            # Draw name if cv2 available
            if CV2_AVAILABLE:
                # Use first 4 chars for compactness
                name = fruit["name"][:4].capitalize()
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.3
                text_size = cv2.getTextSize(name, font, font_scale, 1)[0]
                text_x = x_center - text_size[0] // 2
                text_y = y_center + 18
                cv2.putText(img, name, (text_x, text_y), font, font_scale, (150, 150, 150), 1)
    
    def _draw_fruit(
        self,
        img: np.ndarray,
        fruit: Dict[str, Any],
        scale: float,
        offset_x: float,
        offset_y: float,
        img_height: int
    ) -> None:
        """Draw a single fruit as a filled circle (legacy method)."""
        self._draw_fruit_hitboxes(img, fruit, scale, offset_x, offset_y, img_height)
    
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
