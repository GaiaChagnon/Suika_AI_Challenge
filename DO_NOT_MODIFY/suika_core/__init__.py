"""
Suika Core - The heart of the competition environment.

This module provides the core game simulation, Gymnasium environment wrapper,
and all supporting systems (physics, merging, scoring, RNG).

Main exports:
- SuikaEnv: Gymnasium environment for single-agent training
- SuikaVectorEnv: Vectorized environment for parallel training (single-process)
- make_async_vec_env: Create truly parallel environments (multiprocessing)
- CoreGame: Low-level game simulation (used internally)
- GameConfig: Configuration loaded from game_config.yaml
"""

from DO_NOT_MODIFY.suika_core.config_loader import GameConfig, load_config
from DO_NOT_MODIFY.suika_core.fruit_catalog import FruitType, FruitCatalog
from DO_NOT_MODIFY.suika_core.game import CoreGame
from DO_NOT_MODIFY.suika_core.env_gym import SuikaEnv
from DO_NOT_MODIFY.suika_core.vector_env import SuikaVectorEnv
from DO_NOT_MODIFY.suika_core.async_vector_env import (
    make_async_vec_env,
    make_env,
    get_recommended_num_envs,
)
from DO_NOT_MODIFY.suika_core.replay_recorder import (
    ReplayRecorder,
    record_episode,
    generate_replay_filename,
)

__all__ = [
    "GameConfig",
    "load_config",
    "FruitType",
    "FruitCatalog",
    "CoreGame",
    "SuikaEnv",
    "SuikaVectorEnv",
    "make_async_vec_env",
    "make_env",
    "get_recommended_num_envs",
    "ReplayRecorder",
    "record_episode",
    "generate_replay_filename",
]
