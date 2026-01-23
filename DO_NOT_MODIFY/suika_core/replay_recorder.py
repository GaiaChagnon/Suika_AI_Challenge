"""
Replay Recorder
===============

A simple wrapper to record Gymnasium environment episodes for replay.

Usage:
    from DO_NOT_MODIFY.suika_core import SuikaEnv, ReplayRecorder
    
    env = SuikaEnv()
    recorder = ReplayRecorder(env)
    
    obs, info = recorder.reset(seed=42)
    
    done = False
    while not done:
        action = your_agent(obs)
        obs, reward, terminated, truncated, info = recorder.step(action)
        done = terminated or truncated
    
    recorder.save("my_replay.json")

The saved replay can be viewed with:
    python tools/replay_viewer.py my_replay.json
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import gymnasium as gym

from DO_NOT_MODIFY.suika_core.config_loader import load_config


def generate_replay_filename(
    agent_name: str = "replay",
    seed: Optional[int] = None,
    directory: Optional[Union[str, Path]] = None
) -> Path:
    """
    Generate a timestamped replay filename.
    
    Format: {agent_name}_{YYYYMMDD_HHMMSS}_{seed}.json
    
    Args:
        agent_name: Name of the agent.
        seed: Random seed (optional, included if provided).
        directory: Directory for the file. Defaults to current directory.
        
    Returns:
        Path object for the replay file.
        
    Example:
        >>> generate_replay_filename("my_agent", seed=42)
        Path('my_agent_20260119_143052_s42.json')
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if seed is not None:
        filename = f"{agent_name}_{timestamp}_s{seed}.json"
    else:
        filename = f"{agent_name}_{timestamp}.json"
    
    if directory:
        return Path(directory) / filename
    return Path(filename)


def _compute_config_hash() -> str:
    """Compute a hash of the game config for replay validation."""
    config = load_config()
    # Hash all parameters that affect gameplay/physics
    hash_data = {
        "board": {
            "width": config.board.width,
            "height": config.board.height,
            "lose_line_y": config.board.lose_line_y,
            "spawn_y": config.board.spawn_y,
        },
        "physics": {
            "gravity": config.physics.gravity,
            "damping": config.physics.damping,
            "dt": config.physics.dt,
            "default_friction": config.physics.default_friction,
            "default_elasticity": config.physics.default_elasticity,
        },
        "caps": {
            "out_of_bounds_distance": config.caps.out_of_bounds_distance,
        },
        "domain_randomization": {
            "enabled": config.domain_randomization.enabled,
        },
        "fruits": [
            {
                "id": f.id,
                "visual_radius": f.visual_radius,
                "mass": f.mass,
                "friction": f.friction,
                "elasticity": f.elasticity,
            }
            for f in config.fruits
        ],
    }
    return hashlib.md5(json.dumps(hash_data, sort_keys=True).encode()).hexdigest()[:8]


class ReplayRecorder:
    """
    Wrapper that records environment interactions for replay.
    
    Wraps a SuikaEnv and records all actions, scores, and metadata
    in a format compatible with tools/replay_viewer.py.
    
    Attributes:
        env: The wrapped Gymnasium environment.
        recording: Whether currently recording.
    
    Example:
        env = SuikaEnv()
        recorder = ReplayRecorder(env, agent_name="my_agent")
        
        obs, info = recorder.reset(seed=123)
        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = recorder.step(action)
            done = terminated or truncated
        
        recorder.save("episode_123.json")
    """
    
    def __init__(
        self,
        env: gym.Env,
        agent_name: str = "unknown",
        auto_save_path: Optional[str] = None
    ):
        """
        Initialize the replay recorder.
        
        Args:
            env: The Gymnasium environment to wrap.
            agent_name: Name of the agent (stored in replay metadata).
            auto_save_path: If provided, automatically save replay on episode end.
        """
        self.env = env
        self.agent_name = agent_name
        self.auto_save_path = auto_save_path
        
        # Recording state
        self._recording = False
        self._seed: Optional[int] = None
        self._actions: List[float] = []
        self._scores: List[int] = []
        self._rewards: List[float] = []
        self._termination_reason: str = ""
        self._config_hash = _compute_config_hash()
    
    @property
    def recording(self) -> bool:
        """Whether currently recording."""
        return self._recording
    
    @property
    def observation_space(self):
        """Forward observation space from wrapped env."""
        return self.env.observation_space
    
    @property
    def action_space(self):
        """Forward action space from wrapped env."""
        return self.env.action_space
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[Any, Dict]:
        """
        Reset the environment and start recording.
        
        Args:
            seed: Random seed for the episode.
            options: Additional reset options.
            
        Returns:
            Initial observation and info dict.
        """
        # Clear previous recording
        self._actions = []
        self._scores = []
        self._rewards = []
        self._termination_reason = ""
        self._seed = seed
        self._recording = True
        
        # Reset wrapped environment
        obs, info = self.env.reset(seed=seed, options=options)
        
        return obs, info
    
    def step(self, action: Union[float, np.ndarray]) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Take a step and record it.
        
        Args:
            action: The action to take.
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # Convert action to float for storage
        if isinstance(action, np.ndarray):
            action_val = float(action.item() if action.size == 1 else action[0])
        else:
            action_val = float(action)
        
        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Record
        if self._recording:
            self._actions.append(action_val)
            self._scores.append(int(info.get("score", 0)))
            self._rewards.append(float(reward))
            
            # Capture termination reason on episode end
            if terminated or truncated:
                self._termination_reason = info.get("terminated_reason", "unknown")
        
        # Auto-save on episode end
        if (terminated or truncated) and self.auto_save_path:
            self.save(self.auto_save_path)
        
        return obs, reward, terminated, truncated, info
    
    def get_replay_data(self) -> Dict[str, Any]:
        """
        Get the current replay data as a dictionary.
        
        Returns:
            Dictionary containing all replay data.
        """
        return {
            "seed": self._seed,
            "agent": self.agent_name,
            "config_hash": self._config_hash,
            "actions": self._actions.copy(),
            "scores": self._scores.copy(),
            "rewards": self._rewards.copy(),
            "final_score": self._scores[-1] if self._scores else 0,
            "total_steps": len(self._actions),
            "total_reward": sum(self._rewards),
            "termination_reason": self._termination_reason,
        }
    
    def save(
        self,
        path: Optional[Union[str, Path]] = None,
        overwrite: bool = True,
        directory: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Save the replay to a JSON file.
        
        Args:
            path: Path to save the replay. If None, auto-generates a timestamped name.
            overwrite: If True, overwrite existing file.
            directory: Directory for auto-generated filename (only used if path is None).
            
        Returns:
            Path where the replay was saved.
            
        Example:
            # Auto-generated filename
            recorder.save()  # -> my_agent_20260119_143052_s42.json
            
            # Specific path
            recorder.save("my_replay.json")
            
            # Auto-generated in specific directory
            recorder.save(directory="replays/")
        """
        if path is None:
            path = generate_replay_filename(
                agent_name=self.agent_name,
                seed=self._seed,
                directory=directory
            )
        else:
            path = Path(path)
        
        if path.exists() and not overwrite:
            raise FileExistsError(f"Replay file already exists: {path}")
        
        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        replay_data = self.get_replay_data()
        
        with open(path, "w") as f:
            json.dump(replay_data, f, indent=2)
        
        print(f"Replay saved: {path}")
        print(f"  Seed: {self._seed}")
        print(f"  Steps: {len(self._actions)}")
        print(f"  Final score: {replay_data['final_score']}")
        
        return path
    
    def close(self) -> None:
        """Close the wrapped environment."""
        self.env.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def record_episode(
    env: gym.Env,
    agent_fn,
    seed: int,
    save_path: Optional[str] = None,
    agent_name: str = "unknown"
) -> Dict[str, Any]:
    """
    Convenience function to record a single episode.
    
    Args:
        env: The Gymnasium environment.
        agent_fn: Function that takes observation and returns action.
        seed: Random seed for the episode.
        save_path: If provided, save replay to this path.
        agent_name: Name of the agent.
        
    Returns:
        Replay data dictionary.
        
    Example:
        from DO_NOT_MODIFY.suika_core import SuikaEnv, record_episode
        
        env = SuikaEnv()
        
        def my_agent(obs):
            # Your logic here
            return 0.0
        
        replay = record_episode(env, my_agent, seed=42, save_path="my_game.json")
        print(f"Score: {replay['final_score']}")
    """
    recorder = ReplayRecorder(env, agent_name=agent_name)
    
    obs, info = recorder.reset(seed=seed)
    
    done = False
    while not done:
        action = agent_fn(obs)
        obs, reward, terminated, truncated, info = recorder.step(action)
        done = terminated or truncated
    
    replay_data = recorder.get_replay_data()
    
    if save_path:
        recorder.save(save_path)
    
    return replay_data
