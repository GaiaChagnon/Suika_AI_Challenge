"""
Replay Viewer
=============

View recorded game replays from evaluation.

Usage:
    python tools/replay_viewer.py replay.json
    python -m tools.replay_viewer replay.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

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


def view_replay(
    replay_path: str,
    window_width: int = 400,
    window_height: int = 600,
    speed: float = 1.0
) -> None:
    """
    View a recorded replay.
    
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
    
    if not actions:
        print("Error: Replay contains no actions")
        return
    
    print(f"Replay: {replay_path}")
    print(f"Seed: {seed}")
    print(f"Actions: {len(actions)}")
    print(f"Final score: {replay.get('final_score', 'unknown')}")
    print()
    
    # Create game
    config = load_config()
    game = CoreGame(config=config, seed=seed)
    game.reset(seed=seed)
    
    # Create renderer
    renderer = PygameRenderer(config)
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption(f"Replay: {Path(replay_path).name}")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)
    
    # Playback state
    action_idx = 0
    paused = True
    step_delay = 0.5 / speed  # seconds between actions
    last_step_time = time.time()
    
    print("Controls: SPACE=play/pause, LEFT/RIGHT=step, ESC=quit")
    
    running = True
    while running:
        current_time = time.time()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                
                elif event.key == pygame.K_RIGHT:
                    # Step forward
                    if action_idx < len(actions) and not game.is_over:
                        game.step(actions[action_idx])
                        action_idx += 1
                
                elif event.key == pygame.K_LEFT:
                    # Restart replay
                    game.reset(seed=seed)
                    action_idx = 0
                
                elif event.key == pygame.K_r:
                    # Restart
                    game.reset(seed=seed)
                    action_idx = 0
        
        # Auto-advance if not paused
        if not paused and action_idx < len(actions) and not game.is_over:
            if current_time - last_step_time >= step_delay:
                game.step(actions[action_idx])
                action_idx += 1
                last_step_time = current_time
        
        # Render
        render_data = game.get_render_data()
        renderer.render_to_screen(render_data, window_width, window_height)
        
        # Draw overlay
        status = "PAUSED" if paused else "PLAYING"
        if game.is_over:
            status = "FINISHED"
        
        status_text = font.render(
            f"{status} | Action {action_idx}/{len(actions)}",
            True, (255, 255, 255)
        )
        screen.blit(status_text, (10, window_height - 25))
        
        pygame.display.flip()
        clock.tick(60)
    
    renderer.close()
    pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="View a recorded Suika game replay")
    parser.add_argument("replay", type=str, help="Path to replay JSON file")
    parser.add_argument("--width", type=int, default=400, help="Window width")
    parser.add_argument("--height", type=int, default=600, help="Window height")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed")
    
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
