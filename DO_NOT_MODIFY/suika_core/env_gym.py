"""
Gymnasium Environment Wrapper
=============================

Provides a standard Gymnasium interface to the Suika game.
Reward is always 0.0 - teams must compute their own from info.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from DO_NOT_MODIFY.suika_core.config_loader import GameConfig, load_config
from DO_NOT_MODIFY.suika_core.game import CoreGame
from DO_NOT_MODIFY.suika_core.state_snapshot import GameSnapshot


class SuikaEnv(gym.Env):
    """
    Suika fruit-merging game as a Gymnasium environment.
    
    Action Space:
        Box(low=-1.0, high=1.0, shape=(), dtype=float32)
        Represents spawner X position from left wall (-1) to right wall (+1).
        
    Observation Space:
        Dict containing structured game state and optional RGB image.
        
    Reward:
        Always 0.0. Teams must compute their own reward from the info dict.
        
    Info:
        Contains score, delta_score, drops_used, terminated_reason, etc.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        render_mode: Optional[str] = None,
        render_style: str = "solid",
        image_obs: bool = False,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        debug: bool = False,
    ):
        """
        Initialize Suika environment.
        
        Args:
            config_path: Path to game_config.yaml. Uses default if None.
            render_mode: "human" for window, "rgb_array" for numpy, None for headless.
            render_style: "solid" for blob rendering, "full" for sprites.
            image_obs: If True, include board_rgb in observations.
            image_width: Override observation image width.
            image_height: Override observation image height.
            debug: If True, enables verbose debug output for agent development.
        """
        super().__init__()
        
        # Load config
        self._config = load_config(config_path)
        
        # Store render settings
        self.render_mode = render_mode
        self._render_style = render_style
        self._image_obs = image_obs
        self._debug = debug
        
        # Image dimensions
        self._img_width = image_width or self._config.observation.image_width
        self._img_height = image_height or self._config.observation.image_height
        
        # Initialize game
        self._game = CoreGame(config=self._config)
        
        # Initialize renderer (lazy)
        self._renderer = None
        
        # Define action space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(),
            dtype=np.float32
        )
        
        # Define observation space
        self.observation_space = self._build_observation_space()
        
        if self._debug:
            print(f"[DEBUG] SuikaEnv initialized")
            print(f"[DEBUG]   Board: {self._config.board.width}x{self._config.board.height}")
            print(f"[DEBUG]   Lose line Y: {self._config.board.lose_line_y}")
            print(f"[DEBUG]   Max objects: {self._config.observation.max_objects}")
    
    def _build_observation_space(self) -> spaces.Dict:
        """Build the observation space definition."""
        from DO_NOT_MODIFY.suika_core.state_snapshot import HEIGHT_MAP_SLICES
        
        max_obj = self._config.observation.max_objects
        board = self._config.board
        
        obs_dict = {
            # Core state
            "spawner_x": spaces.Box(low=-1, high=1, shape=(), dtype=np.float32),
            "current_fruit_id": spaces.Discrete(self._config.num_fruit_types),
            "next_fruit_id": spaces.Discrete(self._config.num_fruit_types),
            "score": spaces.Box(low=0, high=np.iinfo(np.int64).max, shape=(), dtype=np.int64),
            "drops_used": spaces.Box(low=0, high=self._config.caps.max_drops, shape=(), dtype=np.int32),
            "objects_count": spaces.Box(low=0, high=max_obj, shape=(), dtype=np.int32),
            
            # Board info
            "board_width": spaces.Box(low=0, high=1000, shape=(), dtype=np.float32),
            "board_height": spaces.Box(low=0, high=1000, shape=(), dtype=np.float32),
            "lose_line_y": spaces.Box(low=0, high=1000, shape=(), dtype=np.float32),
            
            # Basic derived features
            "largest_fruit_type_id": spaces.Box(low=-1, high=self._config.num_fruit_types, shape=(), dtype=np.int32),
            "largest_fruit_x": spaces.Box(low=0, high=board.width, shape=(), dtype=np.float32),
            "largest_fruit_y": spaces.Box(low=0, high=board.height, shape=(), dtype=np.float32),
            "highest_fruit_y": spaces.Box(low=0, high=board.height, shape=(), dtype=np.float32),
            "danger_level": spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
            "distance_to_lose_line": spaces.Box(low=0, high=board.height, shape=(), dtype=np.float32),
            
            # Advanced derived features
            "center_of_mass_x": spaces.Box(low=0, high=board.width, shape=(), dtype=np.float32),
            "center_of_mass_y": spaces.Box(low=0, high=board.height, shape=(), dtype=np.float32),
            "packing_efficiency": spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
            "surface_roughness": spaces.Box(low=0, high=board.height, shape=(), dtype=np.float32),
            "island_count": spaces.Box(low=0, high=max_obj, shape=(), dtype=np.int32),
            "buried_count": spaces.Box(low=0, high=max_obj, shape=(), dtype=np.int32),
            "neighbor_discord": spaces.Box(low=0, high=10, shape=(), dtype=np.float32),
            
            # Height map (1D lidar)
            "height_map": spaces.Box(low=0, high=board.height, shape=(HEIGHT_MAP_SLICES,), dtype=np.float32),
            
            # Object arrays
            "obj_type_id": spaces.Box(low=-1, high=self._config.num_fruit_types, shape=(max_obj,), dtype=np.int16),
            "obj_x": spaces.Box(low=-np.inf, high=np.inf, shape=(max_obj,), dtype=np.float32),
            "obj_y": spaces.Box(low=-np.inf, high=np.inf, shape=(max_obj,), dtype=np.float32),
            "obj_vx": spaces.Box(low=-np.inf, high=np.inf, shape=(max_obj,), dtype=np.float32),
            "obj_vy": spaces.Box(low=-np.inf, high=np.inf, shape=(max_obj,), dtype=np.float32),
            "obj_ang": spaces.Box(low=-np.inf, high=np.inf, shape=(max_obj,), dtype=np.float32),
            "obj_ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(max_obj,), dtype=np.float32),
            "obj_radius": spaces.Box(low=0, high=200, shape=(max_obj,), dtype=np.float32),
            "obj_mask": spaces.MultiBinary(max_obj),
        }
        
        if self._image_obs:
            obs_dict["board_rgb"] = spaces.Box(
                low=0,
                high=255,
                shape=(self._img_height, self._img_width, 3),
                dtype=np.uint8
            )
        
        return spaces.Dict(obs_dict)
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment.
        
        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).
            
        Returns:
            (observation, info) tuple.
        """
        super().reset(seed=seed)
        
        # Reset game
        snapshot = self._game.reset(seed=seed)
        
        # Build observation
        obs = self._snapshot_to_obs(snapshot)
        info = self._game.get_info()
        info["delta_score"] = 0
        
        return obs, info
    
    def step(
        self,
        action: Union[float, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one step.
        
        Args:
            action: Spawner X position in [-1, 1].
            
        Returns:
            (observation, reward, terminated, truncated, info) tuple.
            Reward is always 0.0.
        """
        # Convert action to scalar
        if isinstance(action, np.ndarray):
            action = float(action.item() if action.ndim == 0 else action[0])
        
        # Step game
        result = self._game.step(action)
        
        # Build observation
        obs = self._snapshot_to_obs(result.snapshot)
        
        # Reward is always 0.0 - teams compute their own
        reward = 0.0
        
        # Build info
        info = self._game.get_info()
        info["delta_score"] = result.delta_score
        info["sim_time"] = result.sim_time
        info["merges"] = len(result.merges)  # Number of merges this step
        
        # Debug output
        if self._debug:
            print(f"[DEBUG] Step: action={action:.3f}, delta_score={result.delta_score}, "
                  f"objects={obs['objects_count']}, danger={obs['danger_level']:.2f}")
            if result.terminated:
                print(f"[DEBUG] TERMINATED: {info.get('terminated_reason', 'unknown')}")
        
        return obs, reward, result.terminated, result.truncated, info
    
    def _snapshot_to_obs(self, snapshot: GameSnapshot) -> Dict[str, np.ndarray]:
        """Convert snapshot to observation dict."""
        obs = snapshot.to_obs_dict()
        
        # Add image if requested
        if self._image_obs:
            obs["board_rgb"] = self._render_to_array()
        
        return obs
    
    def _render_to_array(self) -> np.ndarray:
        """Render board to RGB array."""
        if self._renderer is None:
            self._init_renderer()
        
        render_data = self._game.get_render_data()
        return self._renderer.render(
            render_data,
            self._img_width,
            self._img_height
        )
    
    def _init_renderer(self) -> None:
        """Initialize renderer based on style."""
        if self._render_style == "full":
            try:
                from DO_NOT_MODIFY.suika_core.render_full_pygame import PygameRenderer
                self._renderer = PygameRenderer(self._config)
            except ImportError:
                # Fall back to solid if pygame not available
                from DO_NOT_MODIFY.suika_core.render_solid import SolidRenderer
                self._renderer = SolidRenderer(self._config)
        else:
            from DO_NOT_MODIFY.suika_core.render_solid import SolidRenderer
            self._renderer = SolidRenderer(self._config)
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the current game state.
        
        Returns:
            RGB array if render_mode is "rgb_array", None otherwise.
        """
        if self.render_mode == "rgb_array":
            return self._render_to_array()
        
        if self.render_mode == "human":
            if self._renderer is None:
                self._init_renderer()
            
            render_data = self._game.get_render_data()
            self._renderer.render_to_screen(render_data)
            return None
        
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
    
    @property
    def game(self) -> CoreGame:
        """Access to underlying game (for debugging/tools)."""
        return self._game
    
    @property
    def config(self) -> GameConfig:
        """Game configuration."""
        return self._config
