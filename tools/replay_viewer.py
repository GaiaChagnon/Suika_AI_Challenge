"""
Replay Viewer
=============

View recorded game replays with a visual timeline.

Usage:
    python tools/replay_viewer.py replay.json
    python -m tools.replay_viewer replay.json

Controls:
    SPACE       Play/Pause
    LEFT/RIGHT  Step backward/forward
    HOME/END    Jump to start/end
    Click       Seek on timeline
    Drag        Scrub timeline
    R           Restart
    +/-         Speed up/slow down
    ESC         Quit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Add project root to path for direct script execution
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from DO_NOT_MODIFY.suika_core.config_loader import load_config
from DO_NOT_MODIFY.suika_core.game import CoreGame
from DO_NOT_MODIFY.suika_core.render_full_pygame import PygameRenderer


# -----------------------------------------------------------------------------
# Timeline UI Component
# -----------------------------------------------------------------------------
class Timeline:
    """
    A visual timeline bar for replay navigation.
    Shows all moves, current position, scores, and allows click/drag seeking.
    """
    
    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        total_steps: int,
        scores: List[int]
    ):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.total_steps = max(1, total_steps)
        self.scores = scores if scores else [0] * total_steps
        
        # Normalize scores for visual height mapping
        max_score = max(self.scores) if self.scores else 1
        self.normalized_scores = [s / max_score for s in self.scores] if max_score > 0 else [0] * len(self.scores)
        
        # Interaction state
        self.dragging = False
        self.hover_idx = -1
        
        # Colors
        self.bg_color = (30, 30, 35)
        self.border_color = (60, 60, 70)
        self.bar_color = (50, 50, 60)
        self.score_color = (80, 120, 180)
        self.played_color = (100, 180, 100)
        self.cursor_color = (255, 200, 50)
        self.hover_color = (200, 200, 200)
        self.text_color = (200, 200, 200)
    
    def point_to_index(self, px: int, py: int) -> int:
        """Convert screen point to step index."""
        if px < self.x or px > self.x + self.width:
            return -1
        if py < self.y or py > self.y + self.height:
            return -1
        
        rel_x = px - self.x
        idx = int((rel_x / self.width) * self.total_steps)
        return max(0, min(self.total_steps - 1, idx))
    
    def handle_event(self, event, current_idx: int) -> Optional[int]:
        """
        Handle mouse events. Returns new index if seeking, None otherwise.
        """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            idx = self.point_to_index(*event.pos)
            if idx >= 0:
                self.dragging = True
                return idx
        
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False
        
        elif event.type == pygame.MOUSEMOTION:
            # Update hover
            self.hover_idx = self.point_to_index(*event.pos)
            
            # Dragging support
            if self.dragging:
                idx = self.point_to_index(*event.pos)
                if idx >= 0:
                    return idx
        
        return None
    
    def render(self, screen: pygame.Surface, font: pygame.font.Font, current_idx: int):
        """Render the timeline."""
        # Background
        bg_rect = pygame.Rect(self.x - 5, self.y - 5, self.width + 10, self.height + 10)
        pygame.draw.rect(screen, self.bg_color, bg_rect)
        pygame.draw.rect(screen, self.border_color, bg_rect, 1)
        
        # Draw score waveform
        if len(self.normalized_scores) > 1:
            bar_area_height = self.height - 20  # Leave space for labels
            bar_y_base = self.y + bar_area_height
            
            for i, norm_score in enumerate(self.normalized_scores):
                bar_x = self.x + int((i / self.total_steps) * self.width)
                bar_height = max(2, int(norm_score * bar_area_height * 0.8))
                
                # Color based on whether we've played past this point
                if i < current_idx:
                    color = self.played_color
                else:
                    color = self.score_color
                
                pygame.draw.line(
                    screen, color,
                    (bar_x, bar_y_base),
                    (bar_x, bar_y_base - bar_height),
                    1
                )
        
        # Draw cursor (current position)
        cursor_x = self.x + int((current_idx / self.total_steps) * self.width)
        pygame.draw.line(
            screen, self.cursor_color,
            (cursor_x, self.y),
            (cursor_x, self.y + self.height - 15),
            2
        )
        
        # Draw hover indicator
        if self.hover_idx >= 0 and self.hover_idx != current_idx:
            hover_x = self.x + int((self.hover_idx / self.total_steps) * self.width)
            pygame.draw.line(
                screen, self.hover_color,
                (hover_x, self.y),
                (hover_x, self.y + self.height - 15),
                1
            )
            
            # Show hover score
            if self.hover_idx < len(self.scores):
                hover_text = font.render(
                    f"Step {self.hover_idx}: {self.scores[self.hover_idx]}",
                    True, self.hover_color
                )
                text_x = min(hover_x + 5, self.x + self.width - hover_text.get_width())
                screen.blit(hover_text, (text_x, self.y - 20))
        
        # Draw labels
        label_y = self.y + self.height - 12
        
        # Start label
        start_text = font.render("0", True, self.text_color)
        screen.blit(start_text, (self.x, label_y))
        
        # End label
        end_text = font.render(str(self.total_steps), True, self.text_color)
        screen.blit(end_text, (self.x + self.width - end_text.get_width(), label_y))
        
        # Current position label (center)
        pos_text = font.render(f"{current_idx}/{self.total_steps}", True, self.cursor_color)
        screen.blit(pos_text, (self.x + self.width // 2 - pos_text.get_width() // 2, label_y))


# -----------------------------------------------------------------------------
# Replay Viewer
# -----------------------------------------------------------------------------
def rebuild_game_to_step(
    config,
    seed: int,
    actions: List[float],
    target_step: int
) -> CoreGame:
    """
    Rebuild game state by replaying actions up to target_step.
    """
    game = CoreGame(config=config, seed=seed)
    game.reset(seed=seed)
    
    for i in range(min(target_step, len(actions))):
        if game.is_over:
            break
        game.step(actions[i])
    
    return game


def view_replay(
    replay_path: str,
    window_width: int = 500,
    window_height: int = 700,
    speed: float = 1.0
) -> None:
    """
    View a recorded replay with visual timeline.
    
    Args:
        replay_path: Path to replay JSON file.
        window_width: Window width.
        window_height: Window height.
        speed: Playback speed multiplier.
    """
    if not PYGAME_AVAILABLE:
        print("Error: pygame is required for replay viewer.")
        print("Install with: pip install pygame")
        return
    
    # Load replay
    with open(replay_path, "r") as f:
        replay = json.load(f)
    
    seed = replay.get("seed")
    actions = replay.get("actions", [])
    scores = replay.get("scores", [])
    final_score = replay.get("final_score", 0)
    agent_name = replay.get("agent", "unknown")
    termination_reason = replay.get("termination_reason", "unknown")
    
    if not actions:
        print("Error: Replay contains no actions")
        return
    
    # Ensure scores list matches actions length
    if len(scores) < len(actions):
        # Pad with last known score or 0
        last_score = scores[-1] if scores else 0
        scores = scores + [last_score] * (len(actions) - len(scores))
    
    # Check config hash for compatibility
    config = load_config()
    current_hash = hashlib.md5(json.dumps({
        "board": {"width": config.board.width, "height": config.board.height},
        "fruits": [{"id": f.id, "visual_radius": f.visual_radius} for f in config.fruits],
    }, sort_keys=True).encode()).hexdigest()[:8]
    
    replay_hash = replay.get("config_hash", "unknown")
    config_mismatch = replay_hash != "unknown" and replay_hash != current_hash
    
    print(f"Replay: {replay_path}")
    print(f"Seed: {seed}")
    print(f"Agent: {agent_name}")
    print(f"Actions: {len(actions)}")
    print(f"Final score: {final_score}")
    print(f"Game ended: {termination_reason}")
    
    if config_mismatch:
        print()
        print("WARNING: Replay was recorded with different game config!")
        print(f"  Replay config hash: {replay_hash}")
        print(f"  Current config hash: {current_hash}")
        print("  Visual playback may not match recorded scores.")
    
    print()
    print("NOTE: Physics replays may show slight variations due to")
    print("      floating-point non-determinism. Scores shown are from")
    print("      the original recording.")
    print()
    print("Controls:")
    print("  SPACE       Play/Pause")
    print("  LEFT/RIGHT  Step backward/forward")
    print("  HOME/END    Jump to start/end")
    print("  Click       Seek on timeline")
    print("  +/-         Speed up/slow down")
    print("  R           Restart")
    print("  ESC         Quit")
    print()
    
    # Create game and renderer
    config = load_config()
    game = CoreGame(config=config, seed=seed)
    game.reset(seed=seed)
    renderer = PygameRenderer(config)
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(f"Replay: {Path(replay_path).name}")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 22)
    font_large = pygame.font.Font(None, 28)
    
    # Timeline dimensions
    timeline_height = 60
    timeline_margin = 20
    game_area_height = window_height - timeline_height - timeline_margin * 2
    
    # Create timeline
    timeline = Timeline(
        x=timeline_margin,
        y=window_height - timeline_height - timeline_margin // 2,
        width=window_width - timeline_margin * 2,
        height=timeline_height,
        total_steps=len(actions),
        scores=scores
    )
    
    # Playback state
    action_idx = 0
    paused = True
    step_delay = 0.3 / speed
    last_step_time = time.time()
    playback_speed = speed
    
    def seek_to(target_idx: int):
        """Seek to a specific step by rebuilding game state."""
        nonlocal game, action_idx
        target_idx = max(0, min(len(actions), target_idx))
        game = rebuild_game_to_step(config, seed, actions, target_idx)
        action_idx = target_idx
    
    running = True
    while running:
        current_time = time.time()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Timeline interaction
            new_idx = timeline.handle_event(event, action_idx)
            if new_idx is not None:
                seek_to(new_idx)
                paused = True
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                
                elif event.key == pygame.K_RIGHT:
                    # Step forward
                    if action_idx < len(actions) and not game.is_over:
                        game.step(actions[action_idx])
                        action_idx += 1
                    paused = True
                
                elif event.key == pygame.K_LEFT:
                    # Step backward (rebuild game state)
                    if action_idx > 0:
                        seek_to(action_idx - 1)
                    paused = True
                
                elif event.key == pygame.K_HOME:
                    # Jump to start
                    seek_to(0)
                    paused = True
                
                elif event.key == pygame.K_END:
                    # Jump to end
                    seek_to(len(actions))
                    paused = True
                
                elif event.key == pygame.K_r:
                    # Restart
                    seek_to(0)
                    paused = True
                
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    # Speed up
                    playback_speed = min(playback_speed * 1.5, 10.0)
                    step_delay = 0.3 / playback_speed
                
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    # Slow down
                    playback_speed = max(playback_speed / 1.5, 0.1)
                    step_delay = 0.3 / playback_speed
        
        # Auto-advance if not paused
        if not paused and action_idx < len(actions) and not game.is_over:
            if current_time - last_step_time >= step_delay:
                game.step(actions[action_idx])
                action_idx += 1
                last_step_time = current_time
        
        # Clear screen
        screen.fill((20, 20, 25))
        
        # Render game to a subsurface
        render_data = game.get_render_data()
        game_surface = pygame.Surface((window_width, game_area_height))
        renderer._render_to_surface(game_surface, render_data)
        screen.blit(game_surface, (0, 0))
        
        # Draw status bar
        status_y = game_area_height + 5
        
        # Status text - show termination reason when at end
        at_end = action_idx >= len(actions) or game.is_over
        if at_end:
            # Format termination reason for display
            reason_display = termination_reason.replace("_", " ").title() if termination_reason else "Unknown"
            status = f"GAME OVER: {reason_display}"
            status_color = (255, 100, 100)
        elif paused:
            status = "PAUSED"
            status_color = (255, 200, 100)
        else:
            status = "PLAYING"
            status_color = (100, 255, 100)
        
        status_text = font_large.render(status, True, status_color)
        screen.blit(status_text, (timeline_margin, status_y))
        
        # Speed indicator (only show when not at end, to make room for reason text)
        if not at_end:
            speed_text = font.render(f"Speed: {playback_speed:.1f}x", True, (150, 150, 150))
            screen.blit(speed_text, (window_width - speed_text.get_width() - timeline_margin, status_y))
        
        # Score display
        current_score = scores[min(action_idx, len(scores) - 1)] if scores else 0
        score_text = font_large.render(f"Score: {current_score}", True, (255, 255, 255))
        screen.blit(score_text, (window_width - score_text.get_width() - timeline_margin, status_y + 25))
        
        # Render timeline
        timeline.render(screen, font, action_idx)
        
        pygame.display.flip()
        clock.tick(60)
    
    renderer.close()
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="View a recorded Suika game replay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  SPACE       Play/Pause
  LEFT/RIGHT  Step backward/forward
  HOME/END    Jump to start/end
  Click       Seek on timeline
  +/-         Speed up/slow down
  R           Restart
  ESC         Quit
        """
    )
    parser.add_argument("replay", type=str, help="Path to replay JSON file")
    parser.add_argument("--width", type=int, default=500, help="Window width (default: 500)")
    parser.add_argument("--height", type=int, default=700, help="Window height (default: 700)")
    parser.add_argument("--speed", type=float, default=1.0, help="Initial playback speed")
    
    args = parser.parse_args()
    
    view_replay(
        replay_path=args.replay,
        window_width=args.width,
        window_height=args.height,
        speed=args.speed
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
