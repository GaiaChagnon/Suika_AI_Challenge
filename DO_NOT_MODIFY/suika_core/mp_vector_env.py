"""
Multiprocessing Vector Environment
==================================

Distributes environments across multiple processes for maximum throughput.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from DO_NOT_MODIFY.suika_core.config_loader import GameConfig, load_config
from DO_NOT_MODIFY.suika_core.game import CoreGame


def _worker_process(
    pipe: Connection,
    config_path: Optional[str],
    env_indices: List[int],
    base_seed: Optional[int]
) -> None:
    """
    Worker process function.
    
    Manages a subset of game instances and responds to commands.
    """
    config = load_config(config_path)
    
    # Create game instances for this worker
    games: Dict[int, CoreGame] = {}
    for idx in env_indices:
        seed = (base_seed + idx) if base_seed is not None else None
        games[idx] = CoreGame(config=config, seed=seed)
    
    while True:
        try:
            cmd, data = pipe.recv()
        except EOFError:
            break
        
        if cmd == "reset":
            seed = data.get("seed")
            indices = data.get("indices", list(games.keys()))
            
            results = {}
            for idx in indices:
                if idx in games:
                    env_seed = (seed + idx) if seed is not None else None
                    snapshot = games[idx].reset(seed=env_seed)
                    results[idx] = {
                        "snapshot": snapshot,
                        "info": games[idx].get_info()
                    }
            
            pipe.send(results)
        
        elif cmd == "step":
            actions = data["actions"]  # {idx: action}
            
            results = {}
            for idx, action in actions.items():
                if idx in games:
                    result = games[idx].step(action)
                    results[idx] = {
                        "snapshot": result.snapshot,
                        "terminated": result.terminated,
                        "truncated": result.truncated,
                        "delta_score": result.delta_score,
                        "info": games[idx].get_info()
                    }
            
            pipe.send(results)
        
        elif cmd == "render":
            indices = data.get("indices", list(games.keys()))
            
            results = {}
            for idx in indices:
                if idx in games:
                    results[idx] = games[idx].get_render_data()
            
            pipe.send(results)
        
        elif cmd == "close":
            break
    
    pipe.close()


class SuikaMPVectorEnv:
    """
    Multiprocessing vectorized environment.
    
    Distributes game instances across worker processes for parallel execution.
    Useful when Python overhead becomes the bottleneck (100+ envs).
    """
    
    def __init__(
        self,
        num_envs: int,
        num_workers: Optional[int] = None,
        config_path: Optional[str] = None,
        seed: Optional[int] = None,
        render_style: str = "solid",
        image_obs: bool = False,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
    ):
        """
        Initialize multiprocessing vector environment.
        
        Args:
            num_envs: Total number of environments.
            num_workers: Number of worker processes. Defaults to CPU count.
            config_path: Path to game_config.yaml.
            seed: Base random seed.
            render_style: "solid" or "full" for rendering.
            image_obs: If True, include board_rgb in observations.
            image_width: Observation image width.
            image_height: Observation image height.
        """
        self._num_envs = num_envs
        self._config_path = config_path
        self._base_seed = seed
        self._render_style = render_style
        self._image_obs = image_obs
        
        # Load config
        self._config = load_config(config_path)
        self._img_width = image_width or self._config.observation.image_width
        self._img_height = image_height or self._config.observation.image_height
        self._max_obj = self._config.observation.max_objects
        
        # Determine number of workers
        if num_workers is None:
            num_workers = min(mp.cpu_count(), num_envs)
        self._num_workers = num_workers
        
        # Distribute environments across workers
        self._env_to_worker: Dict[int, int] = {}
        self._worker_envs: List[List[int]] = [[] for _ in range(num_workers)]
        
        for i in range(num_envs):
            worker_idx = i % num_workers
            self._env_to_worker[i] = worker_idx
            self._worker_envs[worker_idx].append(i)
        
        # Create worker processes
        self._workers: List[Process] = []
        self._pipes: List[Connection] = []
        
        for worker_idx in range(num_workers):
            parent_conn, child_conn = Pipe()
            
            p = Process(
                target=_worker_process,
                args=(child_conn, config_path, self._worker_envs[worker_idx], seed)
            )
            p.start()
            
            self._workers.append(p)
            self._pipes.append(parent_conn)
        
        # Initialize renderer (lazy)
        self._renderer = None
        
        # Pre-allocate arrays
        self._init_obs_arrays()
    
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
        
        self._rewards = np.zeros(n, dtype=np.float32)
        self._terminateds = np.zeros(n, dtype=bool)
        self._truncateds = np.zeros(n, dtype=bool)
    
    @property
    def num_envs(self) -> int:
        return self._num_envs
    
    @property
    def num_workers(self) -> int:
        return self._num_workers
    
    @property
    def config(self) -> GameConfig:
        return self._config
    
    def reset(
        self,
        seed: Optional[int] = None,
        env_indices: Optional[List[int]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environments."""
        if seed is not None:
            self._base_seed = seed
        
        if env_indices is None:
            env_indices = list(range(self._num_envs))
        
        # Group by worker
        worker_indices: Dict[int, List[int]] = {i: [] for i in range(self._num_workers)}
        for idx in env_indices:
            worker_idx = self._env_to_worker[idx]
            worker_indices[worker_idx].append(idx)
        
        # Send reset commands
        for worker_idx, indices in worker_indices.items():
            if indices:
                self._pipes[worker_idx].send(("reset", {"seed": self._base_seed, "indices": indices}))
        
        # Collect results
        for worker_idx, indices in worker_indices.items():
            if indices:
                results = self._pipes[worker_idx].recv()
                self._update_obs_from_results(results)
        
        return self._build_obs_dict(), self._build_info_dict()
    
    def step(
        self,
        actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Step all environments."""
        # Group actions by worker
        worker_actions: Dict[int, Dict[int, float]] = {i: {} for i in range(self._num_workers)}
        for idx, action in enumerate(actions):
            worker_idx = self._env_to_worker[idx]
            worker_actions[worker_idx][idx] = float(action)
        
        # Send step commands
        for worker_idx, acts in worker_actions.items():
            if acts:
                self._pipes[worker_idx].send(("step", {"actions": acts}))
        
        # Reset result arrays
        self._rewards.fill(0.0)
        self._terminateds.fill(False)
        self._truncateds.fill(False)
        delta_scores = np.zeros(self._num_envs, dtype=np.int32)
        
        # Collect results
        for worker_idx, acts in worker_actions.items():
            if acts:
                results = self._pipes[worker_idx].recv()
                for idx, result in results.items():
                    self._terminateds[idx] = result["terminated"]
                    self._truncateds[idx] = result["truncated"]
                    delta_scores[idx] = result["delta_score"]
                self._update_obs_from_results(results)
        
        obs = self._build_obs_dict()
        infos = self._build_info_dict()
        infos["delta_score"] = delta_scores
        
        return obs, self._rewards.copy(), self._terminateds.copy(), self._truncateds.copy(), infos
    
    def _update_obs_from_results(self, results: Dict[int, Dict]) -> None:
        """Update observation arrays from worker results."""
        for idx, result in results.items():
            snapshot = result["snapshot"]
            
            # Core state
            self._obs_spawner_x[idx] = snapshot.spawner_x
            self._obs_current_fruit_id[idx] = snapshot.current_fruit_id
            self._obs_next_fruit_id[idx] = snapshot.next_fruit_id
            self._obs_score[idx] = snapshot.score
            self._obs_drops_used[idx] = snapshot.drops_used
            self._obs_objects_count[idx] = snapshot.objects_count
            
            # Object arrays
            self._obs_obj_type_id[idx] = snapshot.obj_type_id
            self._obs_obj_x[idx] = snapshot.obj_x
            self._obs_obj_y[idx] = snapshot.obj_y
            self._obs_obj_vx[idx] = snapshot.obj_vx
            self._obs_obj_vy[idx] = snapshot.obj_vy
            self._obs_obj_ang[idx] = snapshot.obj_ang
            self._obs_obj_ang_vel[idx] = snapshot.obj_ang_vel
            self._obs_obj_mask[idx] = snapshot.obj_mask
            
            # Derived observations
            self._obs_height_map[idx] = snapshot.height_map
            self._obs_danger_level[idx] = snapshot.danger_level
            self._obs_highest_fruit_y[idx] = snapshot.highest_fruit_y
            self._obs_distance_to_lose_line[idx] = snapshot.distance_to_lose_line
            self._obs_packing_efficiency[idx] = snapshot.packing_efficiency
            self._obs_surface_roughness[idx] = snapshot.surface_roughness
            self._obs_island_count[idx] = snapshot.island_count
            self._obs_buried_count[idx] = snapshot.buried_count
            self._obs_neighbor_discord[idx] = snapshot.neighbor_discord
            self._obs_center_of_mass_x[idx] = snapshot.center_of_mass_x
            self._obs_center_of_mass_y[idx] = snapshot.center_of_mass_y
            self._obs_largest_fruit_type_id[idx] = snapshot.largest_fruit_type_id
            self._obs_largest_fruit_x[idx] = snapshot.largest_fruit_x
            self._obs_largest_fruit_y[idx] = snapshot.largest_fruit_y
    
    def _build_obs_dict(self) -> Dict[str, np.ndarray]:
        """Build observation dictionary."""
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
        
        return obs
    
    def _build_info_dict(self) -> Dict[str, Any]:
        """Build info dictionary."""
        return {
            "score": self._obs_score.copy(),
            "drops_used": self._obs_drops_used.copy(),
            "fruit_count": self._obs_objects_count.copy(),
        }
    
    def close(self) -> None:
        """Shut down worker processes."""
        for pipe in self._pipes:
            try:
                pipe.send(("close", {}))
            except:
                pass
        
        for worker in self._workers:
            worker.join(timeout=1.0)
            if worker.is_alive():
                worker.terminate()
        
        self._workers.clear()
        self._pipes.clear()
    
    def __del__(self):
        self.close()
