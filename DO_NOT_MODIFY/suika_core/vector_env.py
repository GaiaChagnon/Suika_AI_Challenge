"""
Vector Environment
==================

Single-process vectorized environment for parallel training.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from DO_NOT_MODIFY.suika_core.config_loader import GameConfig, load_config
from DO_NOT_MODIFY.suika_core.game import CoreGame
from DO_NOT_MODIFY.suika_core.state_snapshot import GameSnapshot


class SuikaVectorEnv:
    """
    Vectorized Suika environment for parallel training.
    
    Runs multiple game instances in a single process.
    Designed for 100+ simultaneous environments.
    """
    
    def __init__(
        self,
        num_envs: int,
        config_path: Optional[str] = None,
        seed: Optional[int] = None,
        render_style: str = "solid",
        image_obs: bool = False,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ):
        """
        Initialize vectorized environment.
        
        Args:
            num_envs: Number of parallel environments.
            config_path: Path to game_config.yaml.
            seed: Base random seed. Each env gets seed+i.
            render_style: "solid" or "full" for image rendering.
            image_obs: If True, include board_rgb in observations.
            image_width: Override observation image width.
            image_height: Override observation image height.
        """
        self._num_envs = num_envs
        self._config = load_config(config_path)
        self._base_seed = seed
        self._render_style = render_style
        self._image_obs = image_obs
        
        # Image dimensions
        self._img_width = image_width or self._config.observation.image_width
        self._img_height = image_height or self._config.observation.image_height
        
        # Create game instances
        self._games: List[CoreGame] = []
        for i in range(num_envs):
            env_seed = (seed + i) if seed is not None else None
            game = CoreGame(config=self._config, seed=env_seed)
            self._games.append(game)
        
        # Initialize renderer (lazy)
        self._renderer = None
        
        # Pre-allocate observation arrays
        self._max_obj = self._config.observation.max_objects
        self._init_obs_arrays()
        
        # Action space info
        self.single_action_space = {
            "low": -1.0,
            "high": 1.0,
            "shape": (),
            "dtype": np.float32
        }
        self.action_space = {
            "low": -1.0,
            "high": 1.0,
            "shape": (num_envs,),
            "dtype": np.float32
        }
    
    def _init_obs_arrays(self) -> None:
        """Pre-allocate observation arrays."""
        n = self._num_envs
        m = self._max_obj
        
        # Core state
        self._obs_spawner_x = np.zeros(n, dtype=np.float32)
        self._obs_current_fruit_id = np.zeros(n, dtype=np.int32)
        self._obs_next_fruit_id = np.zeros(n, dtype=np.int32)
        self._obs_score = np.zeros(n, dtype=np.int64)
        self._obs_drops_used = np.zeros(n, dtype=np.int32)
        self._obs_objects_count = np.zeros(n, dtype=np.int32)
        
        # Object arrays
        self._obs_obj_type_id = np.zeros((n, m), dtype=np.int16)
        self._obs_obj_x = np.zeros((n, m), dtype=np.float32)
        self._obs_obj_y = np.zeros((n, m), dtype=np.float32)
        self._obs_obj_vx = np.zeros((n, m), dtype=np.float32)
        self._obs_obj_vy = np.zeros((n, m), dtype=np.float32)
        self._obs_obj_ang = np.zeros((n, m), dtype=np.float32)
        self._obs_obj_ang_vel = np.zeros((n, m), dtype=np.float32)
        self._obs_obj_mask = np.zeros((n, m), dtype=bool)
        
        # Derived observations (HEIGHT_MAP_SLICES = 20)
        self._obs_height_map = np.zeros((n, 20), dtype=np.float32)
        self._obs_danger_level = np.zeros(n, dtype=np.float32)
        self._obs_highest_fruit_y = np.zeros(n, dtype=np.float32)
        self._obs_distance_to_lose_line = np.zeros(n, dtype=np.float32)
        self._obs_packing_efficiency = np.zeros(n, dtype=np.float32)
        self._obs_surface_roughness = np.zeros(n, dtype=np.float32)
        self._obs_island_count = np.zeros(n, dtype=np.int32)
        self._obs_buried_count = np.zeros(n, dtype=np.int32)
        self._obs_neighbor_discord = np.zeros(n, dtype=np.float32)
        self._obs_center_of_mass_x = np.zeros(n, dtype=np.float32)
        self._obs_center_of_mass_y = np.zeros(n, dtype=np.float32)
        self._obs_largest_fruit_type_id = np.zeros(n, dtype=np.int32)
        self._obs_largest_fruit_x = np.zeros(n, dtype=np.float32)
        self._obs_largest_fruit_y = np.zeros(n, dtype=np.float32)
        
        # Board constants
        self._obs_board_width = np.full(n, self._config.board.width, dtype=np.float32)
        self._obs_board_height = np.full(n, self._config.board.height, dtype=np.float32)
        self._obs_lose_line_y = np.full(n, self._config.board.lose_line_y, dtype=np.float32)
        
        if self._image_obs:
            self._obs_board_rgb = np.zeros(
                (n, self._img_height, self._img_width, 3),
                dtype=np.uint8
            )
        
        # Results arrays
        self._rewards = np.zeros(n, dtype=np.float32)
        self._terminateds = np.zeros(n, dtype=bool)
        self._truncateds = np.zeros(n, dtype=bool)
    
    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self._num_envs
    
    @property
    def config(self) -> GameConfig:
        """Game configuration."""
        return self._config
    
    def reset(
        self,
        seed: Optional[int] = None,
        env_indices: Optional[List[int]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset environments.
        
        Args:
            seed: Base random seed. Each env gets seed+i.
            env_indices: Indices of envs to reset. None = all.
            
        Returns:
            (observations, infos) tuple.
        """
        if env_indices is None:
            env_indices = list(range(self._num_envs))
        
        if seed is not None:
            self._base_seed = seed
        
        for i in env_indices:
            env_seed = (self._base_seed + i) if self._base_seed is not None else None
            self._games[i].reset(seed=env_seed)
        
        return self._collect_observations(env_indices)
    
    def step(
        self,
        actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Step all environments.
        
        Args:
            actions: (num_envs,) array of actions in [-1, 1].
            
        Returns:
            (observations, rewards, terminateds, truncateds, infos) tuple.
            Rewards are always 0.0.
        """
        if len(actions) != self._num_envs:
            raise ValueError(f"Expected {self._num_envs} actions, got {len(actions)}")
        
        # Reset results
        self._rewards.fill(0.0)
        self._terminateds.fill(False)
        self._truncateds.fill(False)
        
        delta_scores = np.zeros(self._num_envs, dtype=np.int32)
        
        # Step each environment
        for i, action in enumerate(actions):
            result = self._games[i].step(float(action))
            self._terminateds[i] = result.terminated
            self._truncateds[i] = result.truncated
            delta_scores[i] = result.delta_score
        
        # Collect observations
        obs, infos = self._collect_observations(range(self._num_envs))
        infos["delta_score"] = delta_scores
        
        return (
            obs,
            self._rewards.copy(),
            self._terminateds.copy(),
            self._truncateds.copy(),
            infos
        )
    
    def _collect_observations(
        self,
        env_indices: List[int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Collect observations from specified environments."""
        for i in env_indices:
            game = self._games[i]
            snapshot = game._build_snapshot()
            
            # Core state
            self._obs_spawner_x[i] = snapshot.spawner_x
            self._obs_current_fruit_id[i] = snapshot.current_fruit_id
            self._obs_next_fruit_id[i] = snapshot.next_fruit_id
            self._obs_score[i] = snapshot.score
            self._obs_drops_used[i] = snapshot.drops_used
            self._obs_objects_count[i] = snapshot.objects_count
            
            # Object arrays
            self._obs_obj_type_id[i] = snapshot.obj_type_id
            self._obs_obj_x[i] = snapshot.obj_x
            self._obs_obj_y[i] = snapshot.obj_y
            self._obs_obj_vx[i] = snapshot.obj_vx
            self._obs_obj_vy[i] = snapshot.obj_vy
            self._obs_obj_ang[i] = snapshot.obj_ang
            self._obs_obj_ang_vel[i] = snapshot.obj_ang_vel
            self._obs_obj_mask[i] = snapshot.obj_mask
            
            # Derived observations
            self._obs_height_map[i] = snapshot.height_map
            self._obs_danger_level[i] = snapshot.danger_level
            self._obs_highest_fruit_y[i] = snapshot.highest_fruit_y
            self._obs_distance_to_lose_line[i] = snapshot.distance_to_lose_line
            self._obs_packing_efficiency[i] = snapshot.packing_efficiency
            self._obs_surface_roughness[i] = snapshot.surface_roughness
            self._obs_island_count[i] = snapshot.island_count
            self._obs_buried_count[i] = snapshot.buried_count
            self._obs_neighbor_discord[i] = snapshot.neighbor_discord
            self._obs_center_of_mass_x[i] = snapshot.center_of_mass_x
            self._obs_center_of_mass_y[i] = snapshot.center_of_mass_y
            self._obs_largest_fruit_type_id[i] = snapshot.largest_fruit_type_id
            self._obs_largest_fruit_x[i] = snapshot.largest_fruit_x
            self._obs_largest_fruit_y[i] = snapshot.largest_fruit_y
            
            if self._image_obs:
                self._obs_board_rgb[i] = self._render_env(i)
        
        obs = {
            # Core state
            "spawner_x": self._obs_spawner_x.copy(),
            "current_fruit_id": self._obs_current_fruit_id.copy(),
            "next_fruit_id": self._obs_next_fruit_id.copy(),
            "score": self._obs_score.copy(),
            "drops_used": self._obs_drops_used.copy(),
            "objects_count": self._obs_objects_count.copy(),
            
            # Object arrays
            "obj_type_id": self._obs_obj_type_id.copy(),
            "obj_x": self._obs_obj_x.copy(),
            "obj_y": self._obs_obj_y.copy(),
            "obj_vx": self._obs_obj_vx.copy(),
            "obj_vy": self._obs_obj_vy.copy(),
            "obj_ang": self._obs_obj_ang.copy(),
            "obj_ang_vel": self._obs_obj_ang_vel.copy(),
            "obj_mask": self._obs_obj_mask.copy(),
            
            # Derived observations (LIDAR, danger, quality metrics)
            "height_map": self._obs_height_map.copy(),
            "danger_level": self._obs_danger_level.copy(),
            "highest_fruit_y": self._obs_highest_fruit_y.copy(),
            "distance_to_lose_line": self._obs_distance_to_lose_line.copy(),
            "packing_efficiency": self._obs_packing_efficiency.copy(),
            "surface_roughness": self._obs_surface_roughness.copy(),
            "island_count": self._obs_island_count.copy(),
            "buried_count": self._obs_buried_count.copy(),
            "neighbor_discord": self._obs_neighbor_discord.copy(),
            "center_of_mass_x": self._obs_center_of_mass_x.copy(),
            "center_of_mass_y": self._obs_center_of_mass_y.copy(),
            "largest_fruit_type_id": self._obs_largest_fruit_type_id.copy(),
            "largest_fruit_x": self._obs_largest_fruit_x.copy(),
            "largest_fruit_y": self._obs_largest_fruit_y.copy(),
            
            # Board constants
            "board_width": self._obs_board_width.copy(),
            "board_height": self._obs_board_height.copy(),
            "lose_line_y": self._obs_lose_line_y.copy(),
        }
        
        if self._image_obs:
            obs["board_rgb"] = self._obs_board_rgb.copy()
        
        # Build infos
        infos = {
            "score": self._obs_score.copy(),
            "drops_used": self._obs_drops_used.copy(),
            "fruit_count": self._obs_objects_count.copy(),
        }
        
        return obs, infos
    
    def _render_env(self, env_idx: int) -> np.ndarray:
        """Render a single environment to RGB array."""
        if self._renderer is None:
            self._init_renderer()
        
        render_data = self._games[env_idx].get_render_data()
        return self._renderer.render(
            render_data,
            self._img_width,
            self._img_height
        )
    
    def _init_renderer(self) -> None:
        """Initialize renderer."""
        if self._render_style == "full":
            try:
                from DO_NOT_MODIFY.suika_core.render_full_pygame import PygameRenderer
                self._renderer = PygameRenderer(self._config)
            except ImportError:
                from DO_NOT_MODIFY.suika_core.render_solid import SolidRenderer
                self._renderer = SolidRenderer(self._config)
        else:
            from DO_NOT_MODIFY.suika_core.render_solid import SolidRenderer
            self._renderer = SolidRenderer(self._config)
    
    def render(self, env_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Render specified environments.
        
        Args:
            env_indices: Indices to render. None = all.
            
        Returns:
            (len(env_indices), height, width, 3) array.
        """
        if env_indices is None:
            env_indices = list(range(self._num_envs))
        
        images = []
        for i in env_indices:
            images.append(self._render_env(i))
        
        return np.stack(images, axis=0)
    
    def get_game(self, env_idx: int) -> CoreGame:
        """Get the underlying game instance for an environment."""
        return self._games[env_idx]
    
    def close(self) -> None:
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
    
    def sample_actions(self) -> np.ndarray:
        """Sample random actions for all environments."""
        return np.random.uniform(-1.0, 1.0, size=self._num_envs).astype(np.float32)
