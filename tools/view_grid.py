"""
Grid Viewer - Live Training Visualization
==========================================

Display multiple environments in a grid layout for monitoring training progress.
Can connect to external training processes via shared environments.

Usage:
    # Standalone with random agents
    python -m tools.view_grid [--envs N] [--cols C]
    
    # For live training: pass your vector_env directly
    from tools.view_grid import LiveTrainingViewer
    viewer = LiveTrainingViewer(your_vector_env)
    viewer.start_background()  # Non-blocking

Controls:
    - SPACE: Pause/resume auto-stepping
    - R: Reset all environments
    - ESC: Close viewer
    - +/-: Adjust auto-step speed
"""

from __future__ import annotations

import argparse
import sys
import time
import threading
import queue
from typing import Optional, List, Callable, Any
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from DO_NOT_MODIFY.suika_core.config_loader import load_config
from DO_NOT_MODIFY.suika_core.vector_env import SuikaVectorEnv
from DO_NOT_MODIFY.suika_core.render_solid import SolidRenderer


class LiveTrainingViewer:
    """
    Live viewer for monitoring training progress.
    
    Can run in background thread to visualize agent training without
    blocking the training loop.
    
    Example:
        vec_env = SuikaVectorEnv(num_envs=16)
        viewer = LiveTrainingViewer(vec_env, num_cols=4)
        viewer.start_background()
        
        # Your training loop
        for step in range(100000):
            actions = agent.get_actions(obs)
            obs, rewards, dones, truncs, infos = vec_env.step(actions)
            viewer.update()  # Updates viewer with latest state
            
        viewer.stop()
    """
    
    def __init__(
        self,
        vec_env: SuikaVectorEnv,
        num_cols: Optional[int] = None,
        cell_width: int = 180,
        cell_height: int = 270,
        max_fps: int = 30,
        title: str = "Training Viewer"
    ):
        """
        Initialize viewer.
        
        Args:
            vec_env: Vector environment to visualize.
            num_cols: Number of columns. Auto-calculated if None.
            cell_width: Width of each cell in pixels.
            cell_height: Height of each cell in pixels.
            max_fps: Maximum display frame rate.
            title: Window title.
        """
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame required for viewer: pip install pygame")
        
        self._vec_env = vec_env
        self._num_envs = vec_env.num_envs
        self._cell_width = cell_width
        self._cell_height = cell_height
        self._max_fps = max_fps
        self._title = title
        
        # Calculate grid layout
        if num_cols is None:
            num_cols = int(np.ceil(np.sqrt(self._num_envs)))
        self._num_cols = num_cols
        self._num_rows = int(np.ceil(self._num_envs / num_cols))
        
        # Window dimensions
        self._window_width = self._num_cols * cell_width
        self._window_height = self._num_rows * cell_height + 40  # Extra for stats
        
        # Thread control
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._update_queue: queue.Queue = queue.Queue(maxsize=1)
        
        # Stats tracking
        self._step_count = 0
        self._scores: List[int] = [0] * self._num_envs
        self._last_update_time = time.time()
        self._fps = 0.0
    
    def start_background(self) -> None:
        """Start viewer in background thread (non-blocking)."""
        if self._thread is not None:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """Stop the background viewer."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
    
    def update(self, step_count: Optional[int] = None) -> None:
        """
        Signal viewer to refresh display.
        
        Call this after each training step for live updates.
        Non-blocking: if viewer is busy, update is skipped.
        
        Args:
            step_count: Optional training step number to display.
        """
        if step_count is not None:
            self._step_count = step_count
        else:
            self._step_count += 1
        
        # Non-blocking update signal
        try:
            self._update_queue.put_nowait(True)
        except queue.Full:
            pass  # Viewer busy, skip this frame
    
    def _run_loop(self) -> None:
        """Main viewer loop (runs in background thread)."""
        pygame.init()
        screen = pygame.display.set_mode((self._window_width, self._window_height))
        pygame.display.set_caption(self._title)
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 20)
        font_large = pygame.font.Font(None, 28)
        
        config = load_config()
        renderer = SolidRenderer(config)
        
        paused = False
        
        while self._running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self._running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
            
            # Clear update queue
            try:
                while True:
                    self._update_queue.get_nowait()
            except queue.Empty:
                pass
            
            # Calculate FPS
            current_time = time.time()
            dt = current_time - self._last_update_time
            if dt > 0:
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
            self._last_update_time = current_time
            
            # Render grid
            screen.fill((25, 25, 35))
            
            total_score = 0
            for i in range(self._num_envs):
                row = i // self._num_cols
                col = i % self._num_cols
                
                x = col * self._cell_width
                y = row * self._cell_height
                
                # Get render data from environment
                try:
                    game = self._vec_env.get_game(i)
                    render_data = game.get_render_data()
                    img = renderer.render(render_data, self._cell_width, self._cell_height - 20)
                    self._scores[i] = render_data["score"]
                    total_score += self._scores[i]
                except Exception:
                    # Env might be resetting
                    img = np.zeros((self._cell_height - 20, self._cell_width, 3), dtype=np.uint8)
                
                # Convert to pygame surface
                surface = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
                screen.blit(surface, (x, y))
                
                # Draw score
                score_text = font.render(f"#{i}: {self._scores[i]}", True, (200, 200, 200))
                screen.blit(score_text, (x + 5, y + self._cell_height - 18))
            
            # Stats bar at bottom
            stats_y = self._num_rows * self._cell_height + 5
            
            # Step count
            step_text = font_large.render(f"Step: {self._step_count:,}", True, (255, 255, 255))
            screen.blit(step_text, (10, stats_y))
            
            # Average score
            avg_score = total_score / self._num_envs if self._num_envs > 0 else 0
            score_text = font_large.render(f"Avg Score: {avg_score:.1f}", True, (100, 255, 100))
            screen.blit(score_text, (200, stats_y))
            
            # FPS
            fps_text = font.render(f"FPS: {self._fps:.1f}", True, (150, 150, 150))
            screen.blit(fps_text, (self._window_width - 80, stats_y + 5))
            
            # Paused indicator
            if paused:
                pause_text = font_large.render("PAUSED", True, (255, 200, 100))
                screen.blit(pause_text, (self._window_width // 2 - 40, stats_y))
            
            pygame.display.flip()
            clock.tick(self._max_fps)
        
        pygame.quit()
    
    def run_standalone(
        self,
        agent_fn: Optional[Callable] = None,
        auto_step: bool = True
    ) -> None:
        """
        Run viewer in standalone mode with optional agent.
        
        Args:
            agent_fn: Optional function (obs) -> actions.
            auto_step: Whether to automatically step environments.
        """
        if not PYGAME_AVAILABLE:
            print("Error: pygame required for viewer")
            return
        
        pygame.init()
        screen = pygame.display.set_mode((self._window_width, self._window_height))
        pygame.display.set_caption(self._title)
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 20)
        font_large = pygame.font.Font(None, 28)
        
        config = load_config()
        renderer = SolidRenderer(config)
        
        obs, _ = self._vec_env.reset()
        running = True
        step_delay = 0.1  # Seconds between auto-steps
        last_step = time.time()
        
        print(f"Viewing {self._num_envs} environments in {self._num_rows}x{self._num_cols} grid")
        print("Controls: SPACE=pause, R=reset, +/-=speed, ESC=quit")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        auto_step = not auto_step
                        print(f"Auto-step: {'ON' if auto_step else 'OFF'}")
                    elif event.key == pygame.K_r:
                        obs, _ = self._vec_env.reset()
                        self._step_count = 0
                        print("Environments reset")
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        step_delay = max(0.01, step_delay / 1.5)
                        print(f"Step delay: {step_delay:.3f}s")
                    elif event.key == pygame.K_MINUS:
                        step_delay = min(1.0, step_delay * 1.5)
                        print(f"Step delay: {step_delay:.3f}s")
            
            # Auto-step
            current_time = time.time()
            if auto_step and current_time - last_step >= step_delay:
                if agent_fn is not None:
                    actions = agent_fn(obs)
                else:
                    actions = self._vec_env.sample_actions()
                
                obs, _, terminateds, truncateds, _ = self._vec_env.step(actions)
                self._step_count += 1
                last_step = current_time
                
                # Auto-reset
                done_indices = np.where(terminateds | truncateds)[0].tolist()
                if done_indices:
                    self._vec_env.reset(env_indices=done_indices)
            
            # Render
            screen.fill((25, 25, 35))
            
            total_score = 0
            for i in range(self._num_envs):
                row = i // self._num_cols
                col = i % self._num_cols
                x = col * self._cell_width
                y = row * self._cell_height
                
                render_data = self._vec_env.get_game(i).get_render_data()
                img = renderer.render(render_data, self._cell_width, self._cell_height - 20)
                self._scores[i] = render_data["score"]
                total_score += self._scores[i]
                
                surface = pygame.surfarray.make_surface(np.transpose(img, (1, 0, 2)))
                screen.blit(surface, (x, y))
                
                score_text = font.render(f"#{i}: {self._scores[i]}", True, (200, 200, 200))
                screen.blit(score_text, (x + 5, y + self._cell_height - 18))
            
            # Stats
            stats_y = self._num_rows * self._cell_height + 5
            step_text = font_large.render(f"Step: {self._step_count:,}", True, (255, 255, 255))
            screen.blit(step_text, (10, stats_y))
            
            avg_score = total_score / self._num_envs
            score_text = font_large.render(f"Avg: {avg_score:.1f}", True, (100, 255, 100))
            screen.blit(score_text, (200, stats_y))
            
            if not auto_step:
                pause_text = font_large.render("PAUSED", True, (255, 200, 100))
                screen.blit(pause_text, (self._window_width // 2 - 40, stats_y))
            
            fps_text = font.render(f"FPS: {clock.get_fps():.1f}", True, (150, 150, 150))
            screen.blit(fps_text, (self._window_width - 80, stats_y + 5))
            
            pygame.display.flip()
            clock.tick(self._max_fps)
        
        pygame.quit()


def view_grid(
    num_envs: int = 9,
    num_cols: Optional[int] = None,
    cell_width: int = 180,
    cell_height: int = 270,
    max_fps: int = 30,
    seed: Optional[int] = None,
    agent_fn: Optional[Callable] = None,
    render_style: str = "solid"
) -> None:
    """
    Display multiple environments in a grid.
    
    Args:
        num_envs: Number of environments to display.
        num_cols: Number of columns. Auto-calculated if None.
        cell_width: Width of each cell in pixels.
        cell_height: Height of each cell in pixels.
        max_fps: Maximum frame rate.
        seed: Random seed.
        agent_fn: Optional function (obs) -> actions for agents.
        render_style: "solid" or "full".
    """
    if not PYGAME_AVAILABLE:
        print("Error: pygame is required for grid viewer.")
        print("Install with: pip install pygame")
        return
    
    vec_env = SuikaVectorEnv(
        num_envs=num_envs,
        seed=seed,
        render_style=render_style
    )
    
    viewer = LiveTrainingViewer(
        vec_env=vec_env,
        num_cols=num_cols,
        cell_width=cell_width,
        cell_height=cell_height,
        max_fps=max_fps,
        title=f"Suika Grid Viewer - {num_envs} envs"
    )
    
    viewer.run_standalone(agent_fn=agent_fn)
    vec_env.close()


def main():
    parser = argparse.ArgumentParser(
        description="View multiple Suika environments in a grid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # View 9 environments with random actions
    python -m tools.view_grid --envs 9
    
    # View 16 environments in 4 columns
    python -m tools.view_grid --envs 16 --cols 4
    
    # Larger cells, slower FPS
    python -m tools.view_grid --envs 4 --cell-width 300 --cell-height 400 --fps 15

Controls:
    SPACE   Pause/resume auto-stepping
    R       Reset all environments
    +/-     Adjust step speed
    ESC     Close viewer
"""
    )
    parser.add_argument("--envs", type=int, default=9, help="Number of environments (default: 9)")
    parser.add_argument("--cols", type=int, default=None, help="Number of columns (default: auto)")
    parser.add_argument("--cell-width", type=int, default=180, help="Cell width in pixels (default: 180)")
    parser.add_argument("--cell-height", type=int, default=270, help="Cell height in pixels (default: 270)")
    parser.add_argument("--fps", type=int, default=30, help="Max FPS (default: 30)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--style", choices=["solid", "full"], default="solid", 
                       help="Render style: solid (fast) or full (sprites)")
    
    args = parser.parse_args()
    
    view_grid(
        num_envs=args.envs,
        num_cols=args.cols,
        cell_width=args.cell_width,
        cell_height=args.cell_height,
        max_fps=args.fps,
        seed=args.seed,
        render_style=args.style
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
