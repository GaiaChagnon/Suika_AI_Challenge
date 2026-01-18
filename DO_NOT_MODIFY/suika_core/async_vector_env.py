"""
Async Vector Environment Factory
================================

Provides factory functions for creating truly parallel environments
using gymnasium.vector.AsyncVectorEnv.

Unlike the single-process SuikaVectorEnv, this spawns separate Python
processes that can utilize multiple CPU cores for physics simulation.

Usage:
    from DO_NOT_MODIFY.suika_core.async_vector_env import make_async_vec_env
    
    # Create 8 parallel environments (one per CPU core)
    vec_env = make_async_vec_env(num_envs=8, seed=42)
    
    # Use like any vectorized environment
    obs, infos = vec_env.reset()
    obs, rewards, terms, truncs, infos = vec_env.step(actions)
    vec_env.close()
"""

from __future__ import annotations

import multiprocessing
from typing import Callable, Optional, List
import numpy as np

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv


def make_env(
    rank: int,
    seed: int,
    config_path: Optional[str] = None,
    render_mode: Optional[str] = None,
    render_style: str = "solid",
    image_obs: bool = False,
) -> Callable[[], gym.Env]:
    """
    Create a factory function for a single environment.
    
    This pattern is required for multiprocessing - we return a function
    that will be called in the subprocess to create the environment.
    
    Args:
        rank: Index of this environment (0 to num_envs-1).
        seed: Base seed. Each env gets seed + rank.
        config_path: Path to game_config.yaml (None = default).
        render_mode: "human", "rgb_array", or None.
        render_style: "solid" or "full" for rendering.
        image_obs: Include board_rgb in observations.
        
    Returns:
        Factory function that creates the environment.
    """
    def _init() -> gym.Env:
        # Import here to avoid issues with multiprocessing spawn
        from DO_NOT_MODIFY.suika_core.env_gym import SuikaEnv
        
        env = SuikaEnv(
            config_path=config_path,
            render_mode=render_mode,
            render_style=render_style,
            image_obs=image_obs,
        )
        # Each environment gets a unique seed
        env.reset(seed=seed + rank)
        return env
    
    return _init


def make_async_vec_env(
    num_envs: Optional[int] = None,
    seed: int = 42,
    config_path: Optional[str] = None,
    render_style: str = "solid",
    image_obs: bool = False,
    use_sync: bool = False,
) -> gym.vector.VectorEnv:
    """
    Create a vectorized environment with true multiprocessing.
    
    This spawns separate Python processes, each running its own game
    simulation. This bypasses the GIL and allows parallel physics.
    
    Args:
        num_envs: Number of parallel environments.
                  Default: number of CPU cores.
        seed: Base random seed. Each env gets seed + i.
        config_path: Path to game_config.yaml (None = default).
        render_style: "solid" or "full" for rendering.
        image_obs: Include board_rgb in observations.
        use_sync: If True, use SyncVectorEnv (single-threaded, for debugging).
        
    Returns:
        AsyncVectorEnv or SyncVectorEnv instance.
        
    Example:
        vec_env = make_async_vec_env(num_envs=8, seed=42)
        obs, infos = vec_env.reset()
        
        for _ in range(1000):
            actions = np.random.uniform(-1, 1, size=8)
            obs, rewards, terms, truncs, infos = vec_env.step(actions)
            
            # Handle terminated environments
            for i, (term, trunc) in enumerate(zip(terms, truncs)):
                if term or trunc:
                    # Auto-reset happens, but you might want to log final scores
                    pass
                    
        vec_env.close()
        
    Performance Notes:
        - num_envs should roughly equal CPU core count.
        - Too many envs causes context-switching overhead.
        - Avoid image_obs=True if possible (bandwidth bottleneck).
    """
    if num_envs is None:
        num_envs = multiprocessing.cpu_count()
    
    env_fns = [
        make_env(
            rank=i,
            seed=seed,
            config_path=config_path,
            render_style=render_style,
            image_obs=image_obs,
        )
        for i in range(num_envs)
    ]
    
    if use_sync:
        # Single-threaded, good for debugging
        return SyncVectorEnv(env_fns)
    else:
        # True multiprocessing - spawns separate processes
        return AsyncVectorEnv(env_fns)


def get_recommended_num_envs() -> int:
    """
    Get the recommended number of parallel environments.
    
    Returns:
        Number of CPU cores, capped at 32.
    """
    return min(multiprocessing.cpu_count(), 32)


# Convenience exports
__all__ = [
    "make_env",
    "make_async_vec_env", 
    "get_recommended_num_envs",
]
