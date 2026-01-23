"""
Video Recorder
==============

Records gameplay to MP4 video files by capturing actual rendered frames.
Works in headless environments (Docker, training servers) using software rendering.

Unlike action-based replays, video recordings are deterministic - they capture
exactly what happened, regardless of domain randomization or physics variations.

Usage:
    from DO_NOT_MODIFY.suika_core import SuikaEnv, VideoRecorder
    
    env = SuikaEnv()
    recorder = VideoRecorder(fps=30, width=400, height=500)
    
    obs, info = env.reset(seed=42)
    recorder.start("my_episode.mp4", env)
    
    done = False
    while not done:
        action = agent(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        recorder.capture_frame(env)
        done = terminated or truncated
    
    recorder.stop()

For headless environments (Docker/training), set environment variable:
    SDL_VIDEODRIVER=dummy python your_script.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

# Set SDL to dummy driver for headless rendering if no display
if os.environ.get("DISPLAY") is None and sys.platform != "darwin":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from DO_NOT_MODIFY.suika_core.config_loader import GameConfig, load_config


def generate_video_filename(
    agent_name: str = "episode",
    seed: Optional[int] = None,
    directory: Optional[Union[str, Path]] = None
) -> Path:
    """
    Generate a timestamped video filename.
    
    Format: {agent_name}_{YYYYMMDD_HHMMSS}_{seed}.mp4
    
    Args:
        agent_name: Name of the agent/episode.
        seed: Random seed (optional).
        directory: Directory for the file.
        
    Returns:
        Path object for the video file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if seed is not None:
        filename = f"{agent_name}_{timestamp}_s{seed}.mp4"
    else:
        filename = f"{agent_name}_{timestamp}.mp4"
    
    if directory:
        return Path(directory) / filename
    return Path(filename)


class VideoRecorder:
    """
    Records gameplay to MP4 video by capturing rendered frames.
    
    Works in headless environments using software rendering.
    Uses OpenCV for video encoding (H.264 codec).
    
    Attributes:
        fps: Frames per second for the output video.
        width: Output video width in pixels.
        height: Output video height in pixels.
        recording: Whether currently recording.
    """
    
    def __init__(
        self,
        fps: int = 30,
        width: int = 400,
        height: int = 500,
        use_sprites: bool = False
    ):
        """
        Initialize video recorder.
        
        Args:
            fps: Frames per second (default: 30).
            width: Video width in pixels (default: 400).
            height: Video height in pixels (default: 500).
            use_sprites: Whether to use sprite graphics (may not work headless).
        """
        self.fps = fps
        self.width = width
        self.height = height
        self._use_sprites = use_sprites
        
        self._writer = None
        self._renderer = None
        self._recording = False
        self._frame_count = 0
        self._output_path: Optional[Path] = None
        self._config: Optional[GameConfig] = None
        
        # Try to import cv2
        try:
            import cv2
            self._cv2 = cv2
            self._cv2_available = True
        except ImportError:
            self._cv2 = None
            self._cv2_available = False
    
    @property
    def recording(self) -> bool:
        """Whether currently recording."""
        return self._recording
    
    @property
    def frame_count(self) -> int:
        """Number of frames captured so far."""
        return self._frame_count
    
    def _init_renderer(self) -> None:
        """Initialize the renderer for frame capture."""
        if self._renderer is not None:
            return
        
        if self._config is None:
            self._config = load_config()
        
        # Try pygame renderer first (better quality), fall back to solid
        if self._use_sprites:
            try:
                from DO_NOT_MODIFY.suika_core.render_full_pygame import PygameRenderer
                self._renderer = PygameRenderer(self._config, use_sprites=True)
                return
            except (ImportError, Exception):
                pass
        
        # Use solid renderer (works headless, no dependencies)
        from DO_NOT_MODIFY.suika_core.render_solid import SolidRenderer
        self._renderer = SolidRenderer(self._config)
    
    def start(
        self,
        output_path: Optional[Union[str, Path]] = None,
        env_or_game: Optional[Any] = None,
        agent_name: str = "episode",
        seed: Optional[int] = None,
        directory: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Start recording to a video file.
        
        Args:
            output_path: Path for the output video. Auto-generated if None.
            env_or_game: Optional environment/game to get config from.
            agent_name: Name for auto-generated filename.
            seed: Seed for auto-generated filename.
            directory: Directory for auto-generated filename.
            
        Returns:
            Path where video will be saved.
            
        Raises:
            ImportError: If OpenCV is not available.
            RuntimeError: If already recording.
        """
        if not self._cv2_available:
            raise ImportError(
                "OpenCV (cv2) is required for video recording.\n"
                "Install with: pip install opencv-python"
            )
        
        if self._recording:
            raise RuntimeError("Already recording. Call stop() first.")
        
        # Generate path if not provided
        if output_path is None:
            output_path = generate_video_filename(
                agent_name=agent_name,
                seed=seed,
                directory=directory
            )
        else:
            output_path = Path(output_path)
        
        # Create parent directories
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get config from environment if provided
        if env_or_game is not None:
            if hasattr(env_or_game, '_config'):
                self._config = env_or_game._config
            elif hasattr(env_or_game, 'config'):
                self._config = env_or_game.config
        
        if self._config is None:
            self._config = load_config()
        
        # Initialize renderer
        self._init_renderer()
        
        # Create video writer - try multiple codecs for cross-platform compatibility
        codecs_to_try = [
            ('avc1', '.mp4'),   # H.264 - best quality, works on most systems
            ('mp4v', '.mp4'),   # MPEG-4 - fallback
            ('XVID', '.avi'),   # XVID - very compatible fallback
            ('MJPG', '.avi'),   # Motion JPEG - always works
        ]
        
        self._writer = None
        actual_path = output_path
        
        for codec, ext in codecs_to_try:
            # Adjust file extension if needed
            if ext != output_path.suffix:
                actual_path = output_path.with_suffix(ext)
            
            try:
                fourcc = self._cv2.VideoWriter_fourcc(*codec)
                writer = self._cv2.VideoWriter(
                    str(actual_path),
                    fourcc,
                    self.fps,
                    (self.width, self.height)
                )
                
                if writer.isOpened():
                    self._writer = writer
                    output_path = actual_path
                    break
                else:
                    writer.release()
            except Exception:
                continue
        
        if self._writer is None or not self._writer.isOpened():
            raise RuntimeError(
                f"Failed to create video writer. Tried codecs: {[c[0] for c in codecs_to_try]}\n"
                f"Make sure opencv-python is installed: pip install opencv-python"
            )
        
        self._output_path = output_path
        self._recording = True
        self._frame_count = 0
        
        return output_path
    
    def capture_frame(self, env_or_game: Any) -> None:
        """
        Capture a single frame from the environment/game.
        
        Args:
            env_or_game: The environment (SuikaEnv) or game (CoreGame) to capture.
        """
        if not self._recording:
            return
        
        # Get render data from environment or game
        if hasattr(env_or_game, '_game'):
            # SuikaEnv wrapper
            render_data = env_or_game._game.get_render_data()
        elif hasattr(env_or_game, 'get_render_data'):
            # CoreGame directly
            render_data = env_or_game.get_render_data()
        else:
            raise ValueError("Cannot get render data from provided object")
        
        # Render to RGB array
        frame = self._renderer.render(render_data, self.width, self.height)
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR)
        
        # Write frame
        self._writer.write(frame_bgr)
        self._frame_count += 1
    
    def capture_frames_during_settle(
        self,
        env_or_game: Any,
        num_frames: int = 10,
        frame_interval: int = 4
    ) -> None:
        """
        Capture multiple frames during physics settling.
        
        For smooth video during the settle phase after each drop.
        Note: This requires ticking physics manually, so use with CoreGame.
        
        Args:
            env_or_game: CoreGame instance (not SuikaEnv).
            num_frames: Number of frames to capture.
            frame_interval: Physics ticks between frames.
        """
        if not self._recording:
            return
        
        if not hasattr(env_or_game, 'tick_physics'):
            # Can't tick physics on SuikaEnv, just capture one frame
            self.capture_frame(env_or_game)
            return
        
        game = env_or_game
        for _ in range(num_frames):
            self.capture_frame(game)
            # Tick physics for smooth animation
            for _ in range(frame_interval):
                game.tick_physics()
    
    def stop(self) -> Optional[Path]:
        """
        Stop recording and finalize the video file.
        
        Returns:
            Path to the saved video, or None if not recording.
        """
        if not self._recording:
            return None
        
        self._writer.release()
        self._writer = None
        self._recording = False
        
        output = self._output_path
        duration = self._frame_count / self.fps if self.fps > 0 else 0
        
        print(f"Video saved: {output}")
        print(f"  Frames: {self._frame_count}")
        print(f"  Duration: {duration:.1f}s @ {self.fps} fps")
        print(f"  Resolution: {self.width}x{self.height}")
        
        return output
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        if self._recording:
            self.stop()


def record_episode_video(
    env,
    agent_fn,
    seed: int,
    output_path: Optional[str] = None,
    agent_name: str = "episode",
    fps: int = 30,
    width: int = 400,
    height: int = 500,
    directory: Optional[str] = None
) -> Tuple[Dict[str, Any], Path]:
    """
    Convenience function to record a single episode to video.
    
    Args:
        env: The Gymnasium environment (SuikaEnv).
        agent_fn: Function that takes observation and returns action.
        seed: Random seed for the episode.
        output_path: Path for output video. Auto-generated if None.
        agent_name: Name for auto-generated filename.
        fps: Frames per second.
        width: Video width.
        height: Video height.
        directory: Directory for auto-generated filename.
        
    Returns:
        Tuple of (episode_info, video_path).
        
    Example:
        from DO_NOT_MODIFY.suika_core import SuikaEnv, record_episode_video
        
        env = SuikaEnv()
        
        def my_agent(obs):
            return 0.0  # Your logic here
        
        info, path = record_episode_video(
            env, my_agent, seed=42,
            agent_name="my_agent", directory="videos/"
        )
        print(f"Score: {info['score']}, Video: {path}")
    """
    recorder = VideoRecorder(fps=fps, width=width, height=height)
    
    obs, info = env.reset(seed=seed)
    video_path = recorder.start(
        output_path=output_path,
        env_or_game=env,
        agent_name=agent_name,
        seed=seed,
        directory=directory
    )
    
    # Capture initial frame
    recorder.capture_frame(env)
    
    done = False
    total_reward = 0.0
    steps = 0
    
    while not done:
        action = agent_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # Capture frame after each step
        recorder.capture_frame(env)
        
        done = terminated or truncated
    
    recorder.stop()
    
    episode_info = {
        "seed": seed,
        "score": info.get("score", 0),
        "drops": steps,
        "total_reward": total_reward,
        "terminated_reason": info.get("terminated_reason", "unknown"),
        "video_path": str(video_path),
    }
    
    return episode_info, video_path
