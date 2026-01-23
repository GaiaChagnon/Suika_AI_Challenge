"""
Human Play Mode
================

Play Suika interactively with mouse control and real-time physics animation.
Features beautiful Suika-style UI with fruit evolution wheel.

Controls:
    - Mouse: Move spawner position
    - Click/Space: Drop fruit
    - R: Restart game
    - ESC: Quit

Usage:
    python -m tools.play_human [--seed SEED] [--width WIDTH] [--height HEIGHT]
"""

from __future__ import annotations

import argparse
import sys
import time
import math
from typing import Optional, Tuple

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from DO_NOT_MODIFY.suika_core.config_loader import load_config, GameConfig
from DO_NOT_MODIFY.suika_core.game import CoreGame


class SuikaRenderer:
    """
    Beautiful Suika-style renderer for human play mode.
    Features fruit evolution wheel, gradient backgrounds, and polished UI.
    """
    
    def __init__(self, config: GameConfig, window_width: int, window_height: int):
        """Initialize renderer with Suika aesthetics."""
        self._config = config
        self._window_width = window_width
        self._window_height = window_height
        
        # Colors - warm Suika palette
        self._bg_gradient_top = (255, 230, 200)
        self._bg_gradient_bottom = (255, 210, 170)
        self._box_fill = (255, 252, 245)
        self._box_border = (200, 160, 120)
        self._box_shadow = (180, 140, 100)
        self._ui_panel = (255, 245, 230)
        self._text_dark = (80, 60, 40)
        self._text_light = (140, 110, 80)
        self._lose_line = (255, 100, 100, 150)
        
        # Fonts
        pygame.font.init()
        self._font_huge = pygame.font.Font(None, 56)
        self._font_large = pygame.font.Font(None, 42)
        self._font_medium = pygame.font.Font(None, 28)
        self._font_small = pygame.font.Font(None, 20)
        
        # Sprite loader
        self._sprite_loader = None
        try:
            from DO_NOT_MODIFY.suika_core.sprite_loader import get_sprite_loader
            self._sprite_loader = get_sprite_loader()
        except Exception:
            pass
        
        # Pre-render background
        self._bg_surface = self._create_gradient_background()
        
        # Layout calculations
        self._calculate_layout()
    
    def _create_gradient_background(self) -> pygame.Surface:
        """Create warm gradient background."""
        surface = pygame.Surface((self._window_width, self._window_height))
        for y in range(self._window_height):
            t = y / self._window_height
            r = int(self._bg_gradient_top[0] * (1-t) + self._bg_gradient_bottom[0] * t)
            g = int(self._bg_gradient_top[1] * (1-t) + self._bg_gradient_bottom[1] * t)
            b = int(self._bg_gradient_top[2] * (1-t) + self._bg_gradient_bottom[2] * t)
            pygame.draw.line(surface, (r, g, b), (0, y), (self._window_width, y))
        return surface
    
    def _calculate_layout(self) -> None:
        """Calculate layout for game elements."""
        # Side panel for evolution wheel (right side)
        self._panel_width = 160
        self._game_area_width = self._window_width - self._panel_width
        
        # Top area for score
        self._top_ui_height = 80
        
        # Bottom area for controls
        self._bottom_ui_height = 50
        
        # Game board area
        board_width = self._config.board.width
        board_height = self._config.board.height
        
        available_height = self._window_height - self._top_ui_height - self._bottom_ui_height - 40
        available_width = self._game_area_width - 40
        
        scale_x = available_width / board_width
        scale_y = available_height / board_height
        self._scale = min(scale_x, scale_y)
        
        self._board_render_width = int(board_width * self._scale)
        self._board_render_height = int(board_height * self._scale)
        
        self._board_x = (self._game_area_width - self._board_render_width) // 2
        self._board_y = self._top_ui_height + (available_height - self._board_render_height) // 2
    
    def render(
        self,
        screen: pygame.Surface,
        render_data: dict,
        spawner_screen_x: Optional[int] = None,
        game_over: bool = False,
        final_score: int = 0
    ) -> None:
        """Render the complete game scene."""
        # Background
        screen.blit(self._bg_surface, (0, 0))
        
        # Side panel with evolution wheel
        self._draw_evolution_panel(screen, render_data)
        
        # Game container
        self._draw_game_container(screen, render_data)
        
        # Fruits
        self._draw_fruits(screen, render_data)
        
        # Spawner indicator
        if spawner_screen_x is not None and not game_over:
            self._draw_spawner(screen, render_data, spawner_screen_x)
        
        # Top UI (score)
        self._draw_score_ui(screen, render_data)
        
        # Bottom UI (controls)
        self._draw_controls_ui(screen)
        
        # Game over overlay
        if game_over:
            self._draw_game_over(screen, final_score)
    
    def _draw_evolution_panel(self, screen: pygame.Surface, render_data: dict) -> None:
        """Draw the fruit evolution wheel on the right panel with uniform icon sizes."""
        panel_x = self._game_area_width
        panel_rect = pygame.Rect(panel_x, 0, self._panel_width, self._window_height)
        
        # Panel background
        pygame.draw.rect(screen, self._ui_panel, panel_rect)
        pygame.draw.line(screen, self._box_border, (panel_x, 0), (panel_x, self._window_height), 2)
        
        # Title
        title = self._font_medium.render("FRUITS", True, self._text_dark)
        screen.blit(title, (panel_x + (self._panel_width - title.get_width()) // 2, 15))
        
        # Draw evolution chain (including skull as final form)
        fruit_names = [
            "cherry", "strawberry", "grape", "dekopon", "persimmon",
            "apple", "pear", "peach", "pineapple", "banana", "watermelon", "skull"
        ]
        
        # Uniform icon size for clean look
        icon_size = 40
        
        # Calculate available space
        available_height = self._window_height - 60  # Space after title
        num_fruits = min(len(fruit_names), len(self._config.fruits))
        spacing = min(65, (available_height - 20) // num_fruits)
        start_y = 50
        
        current_id = render_data.get("current_fruit_id", 0)
        next_id = render_data.get("next_fruit_id", 0)
        
        for i, name in enumerate(fruit_names):
            if i >= len(self._config.fruits):
                break
            
            fruit_config = self._config.fruits[i]
            y = start_y + i * spacing
            cx = panel_x + 50
            cy = y + 25
            
            # Highlight current and next
            is_current = (i == current_id)
            is_next = (i == next_id)
            
            if is_current or is_next:
                highlight_color = (255, 220, 100) if is_current else (200, 230, 255)
                pygame.draw.circle(screen, highlight_color, (cx, cy), icon_size // 2 + 6)
            
            # Draw fruit sprite (uniform size)
            if self._sprite_loader and self._sprite_loader.has_sprite(name):
                sprite = self._sprite_loader.get_sprite(name, icon_size)
                rect = sprite.get_rect(center=(cx, cy))
                screen.blit(sprite, rect)
            else:
                # Fallback colored circle - use pinker color for peach
                if name == "peach":
                    color = (255, 130, 160)  # Pink peach
                else:
                    color = fruit_config.color_full
                radius = icon_size // 2
                pygame.draw.circle(screen, color, (cx, cy), radius)
                pygame.draw.circle(screen, tuple(min(255, c + 50) for c in color),
                                 (cx - radius // 3, cy - radius // 3), radius // 3)
            
            # Level number
            level_text = self._font_small.render(f"{i+1}", True, self._text_light)
            screen.blit(level_text, (panel_x + 85, y + 18))
            
            # Arrow pointing UP to next (except for last)
            if i < num_fruits - 1 and spacing >= 40:
                arrow_y = y + spacing - 8
                # Triangle pointing upward (shows progression to next fruit)
                pygame.draw.polygon(screen, self._text_light, [
                    (cx, arrow_y - 6),      # Top point
                    (cx - 5, arrow_y + 2),  # Bottom left
                    (cx + 5, arrow_y + 2)   # Bottom right
                ])
    
    def _draw_game_container(self, screen: pygame.Surface, render_data: dict) -> None:
        """Draw the game box container with visible floor."""
        # Shadow
        shadow_rect = pygame.Rect(
            self._board_x + 4, self._board_y + 4,
            self._board_render_width, self._board_render_height
        )
        pygame.draw.rect(screen, self._box_shadow, shadow_rect, border_radius=8)
        
        # Border
        border_rect = pygame.Rect(
            self._board_x - 4, self._board_y - 4,
            self._board_render_width + 8, self._board_render_height + 8
        )
        pygame.draw.rect(screen, self._box_border, border_rect, border_radius=10)
        
        # Fill (no border radius at bottom so floor is flush)
        fill_rect = pygame.Rect(
            self._board_x, self._board_y,
            self._board_render_width, self._board_render_height
        )
        pygame.draw.rect(screen, self._box_fill, fill_rect)
        
        # Round top corners only by drawing over with background
        corner_radius = 6
        # Top-left corner
        pygame.draw.circle(screen, self._box_fill, 
                          (self._board_x + corner_radius, self._board_y + corner_radius), corner_radius)
        # Top-right corner  
        pygame.draw.circle(screen, self._box_fill,
                          (self._board_x + self._board_render_width - corner_radius, self._board_y + corner_radius), corner_radius)
        
        # Floor line (solid, darker) - at the bottom of the container
        floor_y = self._board_y + self._board_render_height
        pygame.draw.line(
            screen, self._box_border,
            (self._board_x, floor_y - 2),
            (self._board_x + self._board_render_width, floor_y - 2),
            4
        )
        
        # Side walls (solid lines)
        pygame.draw.line(screen, self._box_border,
                        (self._board_x, self._board_y), 
                        (self._board_x, floor_y), 3)
        pygame.draw.line(screen, self._box_border,
                        (self._board_x + self._board_render_width, self._board_y),
                        (self._board_x + self._board_render_width, floor_y), 3)
        
        # Lose line (dashed)
        lose_y = render_data["lose_line_y"]
        lose_y_screen = self._world_to_screen_y(lose_y)
        
        if self._board_y <= lose_y_screen <= self._board_y + self._board_render_height:
            for x in range(self._board_x + 5, self._board_x + self._board_render_width - 5, 15):
                pygame.draw.line(
                    screen, self._lose_line[:3],
                    (x, lose_y_screen), (min(x + 8, self._board_x + self._board_render_width - 5), lose_y_screen),
                    2
                )
    
    def _draw_fruits(self, screen: pygame.Surface, render_data: dict) -> None:
        """Draw all fruits with sprites or fallback."""
        fruit_names = [
            "cherry", "strawberry", "grape", "dekopon", "persimmon",
            "apple", "pear", "peach", "pineapple", "banana", "watermelon", "skull"
        ]
        
        # Sort by Y for proper layering
        fruits = sorted(render_data["fruits"], key=lambda f: f["y"])
        
        for fruit in fruits:
            type_id = fruit["type_id"]
            name = fruit_names[type_id] if type_id < len(fruit_names) else f"fruit_{type_id}"
            
            cx = self._world_to_screen_x(fruit["x"])
            cy = self._world_to_screen_y(fruit["y"])
            radius = int(fruit["visual_radius"] * self._scale)
            angle = fruit.get("angle", 0)
            
            # Clip to game area
            if not (self._board_x - radius <= cx <= self._board_x + self._board_render_width + radius):
                continue
            if not (self._board_y - radius <= cy <= self._board_y + self._board_render_height + radius):
                continue
            
            if self._sprite_loader and self._sprite_loader.has_sprite(name):
                sprite = self._sprite_loader.get_sprite(name, radius * 2)
                if abs(angle) > 0.01:
                    sprite = pygame.transform.rotate(sprite, -math.degrees(angle))
                rect = sprite.get_rect(center=(cx, cy))
                screen.blit(sprite, rect)
            else:
                # Fallback gradient circle
                self._draw_gradient_fruit(screen, cx, cy, radius, fruit["color_full"])
    
    def _draw_gradient_fruit(
        self, screen: pygame.Surface, cx: int, cy: int, radius: int, color: Tuple[int, int, int]
    ) -> None:
        """Draw a gradient circle fruit."""
        surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        for r in range(radius, 0, -1):
            brightness = 0.7 + 0.3 * (r / radius)
            c = tuple(int(min(255, c * brightness)) for c in color)
            pygame.draw.circle(surf, c, (radius, radius), r)
        
        # Highlight
        pygame.draw.circle(surf, tuple(min(255, c + 60) for c in color),
                         (radius - radius // 4, radius - radius // 4), radius // 3)
        
        screen.blit(surf, (cx - radius, cy - radius))
    
    def _draw_spawner(self, screen: pygame.Surface, render_data: dict, screen_x: int) -> None:
        """Draw the spawner indicator and current fruit preview."""
        current_id = render_data.get("current_fruit_id", 0)
        if current_id >= len(self._config.fruits):
            return
        
        fruit_config = self._config.fruits[current_id]
        fruit_names = [
            "cherry", "strawberry", "grape", "dekopon", "persimmon",
            "apple", "pear", "peach", "pineapple", "honeydew", "melon"
        ]
        name = fruit_names[current_id] if current_id < len(fruit_names) else ""
        
        radius = int(fruit_config.visual_radius * self._scale)
        spawn_y_world = render_data.get("spawn_y", self._config.board.spawn_y)
        spawn_y_screen = self._world_to_screen_y(spawn_y_world)
        
        # Clamp X to valid range
        left = self._board_x + radius
        right = self._board_x + self._board_render_width - radius
        screen_x = max(left, min(right, screen_x))
        
        # Guide line
        pygame.draw.line(screen, (180, 160, 140),
                        (screen_x, self._board_y - 20), (screen_x, spawn_y_screen - radius - 5), 2)
        
        # Fruit preview (semi-transparent)
        if self._sprite_loader and self._sprite_loader.has_sprite(name):
            sprite = self._sprite_loader.get_sprite(name, radius * 2)
            sprite = sprite.copy()
            sprite.set_alpha(180)
            rect = sprite.get_rect(center=(screen_x, spawn_y_screen))
            screen.blit(sprite, rect)
        else:
            preview = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            color = fruit_config.color_full
            pygame.draw.circle(preview, (*color, 180), (radius, radius), radius)
            screen.blit(preview, (screen_x - radius, spawn_y_screen - radius))
        
        # Small drop point indicator
        pygame.draw.circle(screen, self._text_dark, (screen_x, spawn_y_screen), 4)
    
    def _draw_score_ui(self, screen: pygame.Surface, render_data: dict) -> None:
        """Draw score and next fruit at the top."""
        # Score
        score_text = f"SCORE"
        score_label = self._font_medium.render(score_text, True, self._text_light)
        screen.blit(score_label, (20, 15))
        
        score_value = self._font_huge.render(f"{render_data['score']:,}", True, self._text_dark)
        screen.blit(score_value, (20, 38))
        
        # Skull multiplier (if any skulls present)
        skull_count = render_data.get("skull_count", 0)
        if skull_count > 0:
            multiplier = render_data.get("skull_multiplier", 1.0)
            mult_text = f"x{multiplier:.1f} ({skull_count}💀)"
            mult_surface = self._font_small.render(mult_text, True, (100, 50, 150))
            screen.blit(mult_surface, (20, 78))
        
        # Next fruit box
        next_x = self._game_area_width - 110
        pygame.draw.rect(screen, self._box_fill, (next_x - 10, 10, 100, 65), border_radius=8)
        pygame.draw.rect(screen, self._box_border, (next_x - 10, 10, 100, 65), 2, border_radius=8)
        
        next_label = self._font_small.render("NEXT", True, self._text_light)
        screen.blit(next_label, (next_x + 25, 15))
        
        # Draw next fruit
        next_id = render_data.get("next_fruit_id", 0)
        if next_id < len(self._config.fruits):
            fruit_names = [
                "cherry", "strawberry", "grape", "dekopon", "persimmon",
                "apple", "pear", "peach", "pineapple", "banana", "watermelon", "skull"
            ]
            name = fruit_names[next_id] if next_id < len(fruit_names) else ""
            
            if self._sprite_loader and self._sprite_loader.has_sprite(name):
                sprite = self._sprite_loader.get_sprite(name, 36)
                rect = sprite.get_rect(center=(next_x + 40, 52))
                screen.blit(sprite, rect)
            else:
                color = self._config.fruits[next_id].color_full
                pygame.draw.circle(screen, color, (next_x + 40, 52), 16)
    
    def _draw_controls_ui(self, screen: pygame.Surface) -> None:
        """Draw controls at the bottom."""
        y = self._window_height - self._bottom_ui_height + 10
        
        controls = [
            ("Click/Space", "Drop"),
            ("R", "Restart"),
            ("ESC", "Quit")
        ]
        
        x = 20
        for key, action in controls:
            # Key box
            key_text = self._font_small.render(key, True, self._text_dark)
            box_width = key_text.get_width() + 12
            pygame.draw.rect(screen, self._box_fill, (x, y, box_width, 24), border_radius=4)
            pygame.draw.rect(screen, self._box_border, (x, y, box_width, 24), 1, border_radius=4)
            screen.blit(key_text, (x + 6, y + 4))
            
            # Action text
            action_text = self._font_small.render(action, True, self._text_light)
            screen.blit(action_text, (x + box_width + 8, y + 4))
            
            x += box_width + action_text.get_width() + 25
    
    def _draw_game_over(self, screen: pygame.Surface, score: int) -> None:
        """Draw game over overlay."""
        # Semi-transparent overlay
        overlay = pygame.Surface((self._window_width, self._window_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        screen.blit(overlay, (0, 0))
        
        # Game over box
        box_w, box_h = 300, 180
        box_x = (self._game_area_width - box_w) // 2
        box_y = (self._window_height - box_h) // 2
        
        pygame.draw.rect(screen, self._box_fill, (box_x, box_y, box_w, box_h), border_radius=16)
        pygame.draw.rect(screen, self._box_border, (box_x, box_y, box_w, box_h), 3, border_radius=16)
        
        # Text
        title = self._font_huge.render("GAME OVER", True, self._text_dark)
        screen.blit(title, (box_x + (box_w - title.get_width()) // 2, box_y + 25))
        
        score_text = self._font_large.render(f"Score: {score:,}", True, self._text_dark)
        screen.blit(score_text, (box_x + (box_w - score_text.get_width()) // 2, box_y + 85))
        
        hint = self._font_medium.render("Press R to restart", True, self._text_light)
        screen.blit(hint, (box_x + (box_w - hint.get_width()) // 2, box_y + 130))
    
    def _world_to_screen_x(self, world_x: float) -> int:
        """Convert world X to screen X."""
        return int(self._board_x + world_x * self._scale)
    
    def _world_to_screen_y(self, world_y: float) -> int:
        """Convert world Y to screen Y (Y is flipped)."""
        return int(self._board_y + (self._config.board.height - world_y) * self._scale)
    
    def screen_to_action(self, screen_x: int) -> float:
        """Convert screen X to action [-1, 1]."""
        relative_x = (screen_x - self._board_x) / self._board_render_width
        relative_x = max(0, min(1, relative_x))
        return relative_x * 2 - 1
    
    @property
    def game_area_width(self) -> int:
        return self._game_area_width


class HumanPlayer:
    """
    Human-playable Suika game with real-time physics animation
    and time acceleration for faster settling.
    """
    
    def __init__(
        self,
        config: Optional[GameConfig] = None,
        seed: Optional[int] = None,
        window_width: int = 700,
        window_height: int = 800,
        target_fps: int = 60
    ):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame required. Install: pip install pygame")
        
        if config is None:
            config = load_config()
        
        self._config = config
        self._seed = seed
        self._window_width = window_width
        self._window_height = window_height
        self._target_fps = target_fps
        
        # Initialize game
        self._game = CoreGame(config=config, seed=seed)
        self._game.reset(seed=seed)
        
        # Initialize pygame
        pygame.init()
        self._screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Suika Game")
        self._clock = pygame.time.Clock()
        
        # Initialize renderer
        self._renderer = SuikaRenderer(config, window_width, window_height)
        
        # State
        self._running = True
        self._game_over = False
        self._settling = False
        self._settle_start_time = 0.0
        self._settle_ticks = 0
        self._consecutive_settled = 0
        
        # Drop cooldown to prevent spawning fruits too close together
        self._last_drop_time = 0.0
        self._drop_cooldown = 0.4  # Minimum 150ms between drops
        
        # Physics timing
        self._physics_dt = config.physics.dt
        self._settle_ticks_needed = config.physics.settle_consecutive_ticks
        self._max_settle_time = config.physics.max_sim_seconds_per_drop
        
        # For smooth physics (no time acceleration - same as AI mode)
        self._physics_accumulator = 0.0
        self._last_time = time.time()
    
    def run(self) -> int:
        """Run the game loop. Returns final score."""
        print("=== Suika Game ===")
        print("Click or Space to drop fruit (you can drop while settling!)")
        print("R to restart, ESC to quit")
        print()
        
        while self._running:
            self._handle_events()
            
            # Always update physics if settling (continuous simulation)
            if self._settling and not self._game_over:
                self._update_physics()
            
            self._render()
            self._clock.tick(self._target_fps)
        
        pygame.quit()
        return self._game.score
    
    def _handle_events(self) -> None:
        """Process pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
                elif event.key == pygame.K_r:
                    self._restart()
                elif event.key == pygame.K_SPACE:
                    # Human mode: allow dropping even while settling
                    if not self._game_over:
                        self._drop_fruit()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Human mode: allow dropping even while settling
                if event.button == 1 and not self._game_over:
                    self._drop_fruit()
    
    def _drop_fruit(self) -> None:
        """
        Drop the current fruit at mouse position.
        Human mode allows dropping even while settling (unlike AI mode).
        Has a cooldown to prevent spawning fruits too close together.
        """
        # Check if we have fruits left to drop
        if self._game.is_over:
            return
        
        # Enforce cooldown to prevent rapid-fire drops
        current_time = time.time()
        if current_time - self._last_drop_time < self._drop_cooldown:
            return  # Too soon, ignore this drop
        
        mouse_x, _ = pygame.mouse.get_pos()
        action = self._renderer.screen_to_action(mouse_x)
        
        # Use the low-level spawn directly instead of full step
        fruit_id = self._game._spawn_queue.advance()
        fruit_type = self._game._catalog[fruit_id]
        spawn_x = self._game._rules.spawn.action_to_spawn_x(action, fruit_type)
        spawn_y = self._game._rules.spawn.spawn_y
        
        self._game._physics.spawn_fruit(fruit_type, spawn_x, spawn_y)
        self._game._drops_used += 1
        self._game._last_spawner_x = action
        
        # Record drop time
        self._last_drop_time = current_time
        
        # Start settling if not already settling
        if not self._settling:
            self._settling = True
            self._settle_start_time = current_time
            self._settle_ticks = 0
            self._consecutive_settled = 0
            self._physics_accumulator = 0.0
            self._last_time = current_time
    
    def _update_physics(self) -> None:
        """
        Update physics in real-time (human mode continuously simulates).
        Unlike AI mode which waits for full settle, human mode keeps simulating
        and allows new drops at any time.
        """
        current_time = time.time()
        frame_dt = current_time - self._last_time
        self._last_time = current_time
        
        settle_elapsed = current_time - self._settle_start_time
        
        # Accumulate time for physics
        self._physics_accumulator += frame_dt
        
        # Limit to prevent spiral
        if self._physics_accumulator > 0.2:
            self._physics_accumulator = 0.2
        
        # Run physics ticks
        while self._physics_accumulator >= self._physics_dt:
            self._physics_accumulator -= self._physics_dt
            
            is_settled, delta_score, merges = self._game.tick_physics()
            self._settle_ticks += 1
            
            if delta_score > 0:
                print(f"  +{delta_score} (Total: {self._game.score})")
            
            # Check settle - but don't stop physics in human mode
            # Just check for game over conditions
            if is_settled:
                self._consecutive_settled += 1
            else:
                self._consecutive_settled = 0
            
            # Check termination periodically (every 10 ticks)
            if self._settle_ticks % 10 == 0:
                terminated, truncated, reason = self._game.check_and_update_termination()
                if terminated or truncated:
                    self._game_over = True
                    self._settling = False
                    if terminated:
                        print(f"\nGAME OVER - Score: {self._game.score}")
                    else:
                        print(f"\nGAME COMPLETE - Score: {self._game.score}")
                    return
            
            # Safety timeout - force settle if taking too long
            if settle_elapsed >= self._max_settle_time * 2:
                self._game.force_settle()
                self._settle_ticks = 0
                self._consecutive_settled = 0
                self._settle_start_time = current_time
    
    def _restart(self) -> None:
        """Restart the game."""
        self._game.reset(seed=self._seed)
        self._game_over = False
        self._settling = False
        self._last_drop_time = 0.0  # Reset cooldown
        print("\n=== Game Restarted ===\n")
    
    def _render(self) -> None:
        """Render the game."""
        render_data = self._game.get_render_data()
        
        # Always show spawner unless game over (human mode allows drops anytime)
        spawner_x = None
        if not self._game_over:
            spawner_x, _ = pygame.mouse.get_pos()
        
        self._renderer.render(
            self._screen,
            render_data,
            spawner_screen_x=spawner_x,
            game_over=self._game_over,
            final_score=self._game.score
        )
        
        pygame.display.flip()


def main():
    parser = argparse.ArgumentParser(description="Play Suika game interactively")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--width", type=int, default=600, help="Window width (default: 600)")
    parser.add_argument("--height", type=int, default=700, help="Window height (default: 700)")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS")
    
    args = parser.parse_args()
    
    try:
        config = load_config()
        player = HumanPlayer(
            config=config,
            seed=args.seed,
            window_width=args.width,
            window_height=args.height,
            target_fps=args.fps
        )
        score = player.run()
        print(f"\nFinal Score: {score}")
        return 0
    except ImportError as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
