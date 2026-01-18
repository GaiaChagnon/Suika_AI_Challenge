"""
Full Pygame Renderer
====================

Pretty renderer using pygame with sprite support for Suika Game aesthetics.
Supports both display mode (human play) and headless RGB output.
"""

from __future__ import annotations

import math
from typing import Dict, Any, Optional, Tuple, List

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

import numpy as np

from DO_NOT_MODIFY.suika_core.config_loader import GameConfig, get_config


class PygameRenderer:
    """
    Full-featured renderer using pygame.
    
    Supports:
    - Sprite-based fruit rendering with fallback gradients
    - Score/UI overlay with next fruit preview
    - Screen display for human mode
    - RGB array output for agents
    """
    
    def __init__(self, config: Optional[GameConfig] = None, use_sprites: bool = True):
        """
        Initialize renderer.
        
        Args:
            config: Game configuration.
            use_sprites: Whether to load and use sprite graphics.
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for PygameRenderer")
        
        if config is None:
            config = get_config()
        
        self._config = config
        self._use_sprites = use_sprites
        
        # Initialize pygame
        if not pygame.get_init():
            pygame.init()
        
        # Display surface (created on demand)
        self._screen: Optional[pygame.Surface] = None
        self._screen_size: Optional[Tuple[int, int]] = None
        
        # Fonts
        pygame.font.init()
        self._font = pygame.font.Font(None, 28)
        self._font_large = pygame.font.Font(None, 48)
        self._font_small = pygame.font.Font(None, 20)
        
        # Colors - warm Suika-style palette
        self._bg_color = (245, 235, 220)           # Warm cream background
        self._box_color = (255, 248, 240)          # Light cream for game area
        self._box_border_color = (220, 180, 140)   # Warm tan border
        self._box_shadow_color = (200, 160, 120)   # Shadow
        self._lose_line_color = (230, 100, 100)    # Soft red
        self._text_color = (80, 60, 40)            # Dark brown
        self._text_shadow = (140, 120, 100)        # Light brown shadow
        
        # Sprite loader (lazy init)
        self._sprite_loader: Optional[Any] = None
        
        # Pre-computed fruit gradient surfaces (fallback)
        self._gradient_cache: Dict[Tuple[int, int], pygame.Surface] = {}
    
    def _get_sprite_loader(self):
        """Lazy load the sprite loader."""
        if self._sprite_loader is None and self._use_sprites:
            try:
                from DO_NOT_MODIFY.suika_core.sprite_loader import get_sprite_loader
                self._sprite_loader = get_sprite_loader()
            except Exception:
                self._sprite_loader = False  # Mark as unavailable
        return self._sprite_loader if self._sprite_loader else None
    
    def render(
        self,
        render_data: Dict[str, Any],
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Render to RGB array (for agent observation).
        
        Args:
            render_data: Data from CoreGame.get_render_data().
            width: Output image width.
            height: Output image height.
            
        Returns:
            (height, width, 3) uint8 array.
        """
        surface = pygame.Surface((width, height))
        self._render_to_surface(surface, render_data)
        array = pygame.surfarray.array3d(surface)
        return np.transpose(array, (1, 0, 2))
    
    def render_to_screen(
        self,
        render_data: Dict[str, Any],
        window_width: int = 600,
        window_height: int = 800
    ) -> None:
        """
        Render to pygame window.
        
        Args:
            render_data: Data from CoreGame.get_render_data().
            window_width: Window width.
            window_height: Window height.
        """
        if self._screen is None or self._screen_size != (window_width, window_height):
            self._screen = pygame.display.set_mode((window_width, window_height))
            self._screen_size = (window_width, window_height)
            pygame.display.set_caption("Suika Game")
        
        self._render_to_surface(self._screen, render_data)
    
    def _render_to_surface(
        self,
        surface: pygame.Surface,
        render_data: Dict[str, Any]
    ) -> None:
        """Render game state to a pygame surface."""
        width, height = surface.get_size()
        
        # Calculate layout
        board_width = render_data["board_width"]
        board_height = render_data["board_height"]
        
        # Reserve space for UI at top
        ui_height = 100
        game_area_height = height - ui_height
        
        # Scale game board to fit
        scale_x = (width - 40) / board_width  # 20px margin on each side
        scale_y = (game_area_height - 40) / board_height
        scale = min(scale_x, scale_y)
        
        # Center the board horizontally
        board_render_width = board_width * scale
        board_render_height = board_height * scale
        offset_x = (width - board_render_width) / 2
        offset_y = ui_height + (game_area_height - board_render_height) / 2
        
        # Clear background with warm color
        surface.fill(self._bg_color)
        
        # Draw UI area
        self._draw_ui(surface, render_data, width, ui_height, scale)
        
        # Draw game box with shadow effect
        box_rect = pygame.Rect(
            int(offset_x) - 3,
            int(offset_y) - 3,
            int(board_render_width) + 6,
            int(board_render_height) + 6
        )
        
        # Shadow
        shadow_rect = box_rect.copy()
        shadow_rect.x += 4
        shadow_rect.y += 4
        pygame.draw.rect(surface, self._box_shadow_color, shadow_rect, border_radius=8)
        
        # Main box border
        pygame.draw.rect(surface, self._box_border_color, box_rect, border_radius=8)
        
        # Inner game area
        inner_rect = pygame.Rect(
            int(offset_x),
            int(offset_y),
            int(board_render_width),
            int(board_render_height)
        )
        pygame.draw.rect(surface, self._box_color, inner_rect, border_radius=4)
        
        # Draw lose line (dashed)
        lose_y = render_data["lose_line_y"]
        lose_y_screen = int(offset_y + (board_height - lose_y) * scale)
        if 0 <= lose_y_screen < height:
            for x in range(int(offset_x) + 5, int(offset_x + board_render_width) - 5, 12):
                pygame.draw.line(
                    surface,
                    (*self._lose_line_color, 150),
                    (x, lose_y_screen),
                    (min(x + 6, int(offset_x + board_render_width) - 5), lose_y_screen),
                    2
                )
        
        # Draw fruits (sorted by Y so lower fruits are drawn first)
        fruits = sorted(render_data["fruits"], key=lambda f: f["y"])
        for fruit in fruits:
            self._draw_fruit(surface, fruit, scale, offset_x, offset_y, board_height)
        
        # Draw spawner indicator if present
        if "spawner_screen_x" in render_data:
            self._draw_spawner(
                surface, render_data, scale, offset_x, offset_y, 
                board_width, board_height, width
            )
    
    def _draw_fruit(
        self,
        surface: pygame.Surface,
        fruit: Dict[str, Any],
        scale: float,
        offset_x: float,
        offset_y: float,
        board_height: float
    ) -> None:
        """Draw a single fruit with sprite or gradient."""
        world_x = fruit["x"]
        world_y = fruit["y"]
        angle = fruit.get("angle", 0)
        
        # Convert to screen coordinates (Y is flipped)
        cx = int(world_x * scale + offset_x)
        cy = int(offset_y + (board_height - world_y) * scale)
        
        radius = int(fruit["visual_radius"] * scale)
        if radius < 2:
            radius = 2
        
        # Get fruit name for sprite lookup
        fruit_names = [
            "cherry", "strawberry", "grape", "dekopon", "persimmon",
            "apple", "pear", "peach", "pineapple", "honeydew", "melon"
        ]
        type_id = fruit["type_id"]
        fruit_name = fruit_names[type_id] if type_id < len(fruit_names) else f"fruit_{type_id}"
        
        sprite_loader = self._get_sprite_loader()
        sprite_drawn = False
        
        if sprite_loader is not None:
            try:
                sprite = sprite_loader.get_sprite(fruit_name, radius * 2)
                # Rotate sprite
                if abs(angle) > 0.01:
                    rotated = pygame.transform.rotate(sprite, -math.degrees(angle))
                else:
                    rotated = sprite
                
                # Draw centered
                rect = rotated.get_rect(center=(cx, cy))
                surface.blit(rotated, rect)
                sprite_drawn = True
            except Exception:
                pass
        
        if not sprite_drawn:
            # Fallback: gradient circle
            self._draw_gradient_fruit(surface, cx, cy, radius, fruit["color_full"], angle)
    
    def _draw_gradient_fruit(
        self,
        surface: pygame.Surface,
        cx: int,
        cy: int,
        radius: int,
        base_color: Tuple[int, int, int],
        angle: float = 0
    ) -> None:
        """Draw a fruit as a gradient circle with face."""
        cache_key = (tuple(base_color), radius)
        
        if cache_key not in self._gradient_cache:
            size = radius * 2
            fruit_surface = pygame.Surface((size, size), pygame.SRCALPHA)
            center = radius
            
            # Draw gradient layers
            for r in range(radius, 0, -1):
                ratio = r / radius
                brightness = 0.7 + 0.3 * ratio
                c = tuple(int(min(255, c * brightness)) for c in base_color)
                pygame.draw.circle(fruit_surface, c, (center, center), r)
            
            # Highlight
            highlight_color = tuple(min(255, c + 80) for c in base_color)
            highlight_r = max(2, radius // 3)
            pygame.draw.circle(
                fruit_surface,
                (*highlight_color, 200),
                (center - radius // 4, center - radius // 4),
                highlight_r
            )
            
            # Face
            if radius > 12:
                self._draw_cute_face(fruit_surface, center, center, radius)
            
            # Outline
            outline_color = tuple(max(0, c - 50) for c in base_color)
            pygame.draw.circle(fruit_surface, outline_color, (center, center), radius, 2)
            
            self._gradient_cache[cache_key] = fruit_surface
        
        fruit_surface = self._gradient_cache[cache_key]
        
        # Rotate if needed
        if abs(angle) > 0.01:
            rotated = pygame.transform.rotate(fruit_surface, -math.degrees(angle))
        else:
            rotated = fruit_surface
        
        rect = rotated.get_rect(center=(cx, cy))
        surface.blit(rotated, rect)
    
    def _draw_cute_face(
        self,
        surface: pygame.Surface,
        cx: int,
        cy: int,
        radius: int
    ) -> None:
        """Draw a simple cute face on a fruit."""
        eye_color = (40, 40, 40)
        eye_size = max(2, radius // 8)
        eye_y = cy - radius // 6
        eye_spacing = radius // 3
        
        # Eyes
        pygame.draw.circle(surface, eye_color, (cx - eye_spacing, eye_y), eye_size)
        pygame.draw.circle(surface, eye_color, (cx + eye_spacing, eye_y), eye_size)
        
        # Eye highlights
        highlight_size = max(1, eye_size // 2)
        pygame.draw.circle(
            surface, (255, 255, 255),
            (cx - eye_spacing - 1, eye_y - 1), highlight_size
        )
        pygame.draw.circle(
            surface, (255, 255, 255),
            (cx + eye_spacing - 1, eye_y - 1), highlight_size
        )
        
        # Simple smile
        mouth_y = cy + radius // 6
        mouth_w = radius // 2
        mouth_rect = pygame.Rect(cx - mouth_w // 2, mouth_y - radius // 8, mouth_w, radius // 4)
        pygame.draw.arc(surface, eye_color, mouth_rect, 3.14, 0, max(1, radius // 12))
    
    def _draw_ui(
        self,
        surface: pygame.Surface,
        render_data: Dict[str, Any],
        width: int,
        ui_height: int,
        scale: float
    ) -> None:
        """Draw UI overlay with score and next fruit preview."""
        # Score panel (left side)
        score_text = f"Score: {render_data['score']}"
        score_surface = self._font_large.render(score_text, True, self._text_color)
        surface.blit(score_surface, (20, 20))
        
        # Drops counter
        drops_text = f"Drops: {render_data['drops_used']}"
        drops_surface = self._font.render(drops_text, True, self._text_shadow)
        surface.blit(drops_surface, (20, 60))
        
        # Next fruit preview (right side)
        next_id = render_data.get("next_fruit_id", 0)
        fruit_names = [
            "cherry", "strawberry", "grape", "dekopon", "persimmon",
            "apple", "pear", "peach", "pineapple", "honeydew", "melon"
        ]
        fruit_name = fruit_names[next_id] if next_id < len(fruit_names) else f"#{next_id}"
        
        # "NEXT" label
        next_label = self._font_small.render("NEXT", True, self._text_shadow)
        label_x = width - 80
        surface.blit(next_label, (label_x, 10))
        
        # Draw preview fruit
        fruit_config = self._config.fruits[next_id] if next_id < len(self._config.fruits) else None
        preview_size = 50
        preview_cx = label_x + 25
        preview_cy = 55
        
        if fruit_config is not None:
            sprite_loader = self._get_sprite_loader()
            sprite_drawn = False
            
            if sprite_loader is not None:
                try:
                    sprite = sprite_loader.get_sprite(fruit_name, preview_size)
                    rect = sprite.get_rect(center=(preview_cx, preview_cy))
                    surface.blit(sprite, rect)
                    sprite_drawn = True
                except Exception:
                    pass
            
            if not sprite_drawn:
                self._draw_gradient_fruit(
                    surface, preview_cx, preview_cy,
                    preview_size // 2, fruit_config.color_full
                )
    
    def _draw_spawner(
        self,
        surface: pygame.Surface,
        render_data: Dict[str, Any],
        scale: float,
        offset_x: float,
        offset_y: float,
        board_width: float,
        board_height: float,
        screen_width: int
    ) -> None:
        """Draw the spawner/dropper indicator and current fruit preview."""
        screen_x = render_data.get("spawner_screen_x", screen_width // 2)
        current_id = render_data.get("current_fruit_id", 0)
        
        fruit_config = self._config.fruits[current_id] if current_id < len(self._config.fruits) else None
        if fruit_config is None:
            return
        
        spawn_y = render_data.get("spawn_y", board_height * 0.95)
        
        # Convert spawn_y to screen coords
        spawner_screen_y = int(offset_y + (board_height - spawn_y) * scale)
        
        # Clamp screen_x to valid board range
        left_bound = int(offset_x + fruit_config.visual_radius * scale)
        right_bound = int(offset_x + board_width * scale - fruit_config.visual_radius * scale)
        screen_x = max(left_bound, min(right_bound, screen_x))
        
        radius = int(fruit_config.visual_radius * scale)
        
        # Draw vertical guide line
        line_color = (180, 160, 140, 100)
        pygame.draw.line(
            surface,
            line_color[:3],
            (screen_x, int(offset_y) - 30),
            (screen_x, spawner_screen_y - radius - 5),
            2
        )
        
        # Draw current fruit preview (semi-transparent)
        fruit_names = [
            "cherry", "strawberry", "grape", "dekopon", "persimmon",
            "apple", "pear", "peach", "pineapple", "honeydew", "melon"
        ]
        fruit_name = fruit_names[current_id] if current_id < len(fruit_names) else f"#{current_id}"
        
        sprite_loader = self._get_sprite_loader()
        sprite_drawn = False
        
        if sprite_loader is not None:
            try:
                sprite = sprite_loader.get_sprite(fruit_name, radius * 2)
                # Make semi-transparent
                sprite = sprite.copy()
                sprite.set_alpha(200)
                rect = sprite.get_rect(center=(screen_x, spawner_screen_y))
                surface.blit(sprite, rect)
                sprite_drawn = True
            except Exception:
                pass
        
        if not sprite_drawn:
            # Draw semi-transparent circle
            preview_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            color = fruit_config.color_full
            
            for r in range(radius, 0, -1):
                ratio = r / radius
                brightness = 0.7 + 0.3 * ratio
                c = tuple(int(min(255, c * brightness)) for c in color)
                pygame.draw.circle(preview_surface, (*c, 180), (radius, radius), r)
            
            surface.blit(preview_surface, (screen_x - radius, spawner_screen_y - radius))
        
        # Draw small indicator dot at drop point
        pygame.draw.circle(surface, self._text_color, (screen_x, spawner_screen_y), 3)
    
    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if quit requested."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True
    
    def close(self) -> None:
        """Clean up pygame resources."""
        self._gradient_cache.clear()
        if self._screen is not None:
            self._screen = None
