"""
State Snapshot
==============

Packs game state into fixed-size numpy arrays for Gymnasium observations.
Includes advanced derived features for agent decision-making.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import math

from DO_NOT_MODIFY.suika_core.config_loader import GameConfig, get_config

if TYPE_CHECKING:
    from DO_NOT_MODIFY.suika_core.physics_world import PhysicsWorld, FruitBody

# Number of slices for height map
HEIGHT_MAP_SLICES = 20


@dataclass
class GameSnapshot:
    """
    Complete game state snapshot with advanced derived features.
    
    All arrays are fixed-size with masking for variable object counts.
    """
    # Core state
    spawner_x: float
    current_fruit_id: int
    next_fruit_id: int
    score: int
    drops_used: int
    objects_count: int
    
    # Board info (for normalization)
    board_width: float
    board_height: float
    lose_line_y: float
    
    # Basic derived features
    largest_fruit_type_id: int
    largest_fruit_x: float
    largest_fruit_y: float
    highest_fruit_y: float
    danger_level: float
    distance_to_lose_line: float
    
    # Advanced derived features
    center_of_mass_x: float           # COM X coordinate
    center_of_mass_y: float           # COM Y coordinate
    packing_efficiency: float         # Ratio of fruit area to convex hull (0-1)
    surface_roughness: float          # Std dev of surface heights
    island_count: int                 # Number of separate fruit clusters
    buried_count: int                 # Fruits trapped under others
    neighbor_discord: float           # Average type difference between neighbors
    
    # Height map (1D lidar)
    height_map: np.ndarray            # (HEIGHT_MAP_SLICES,) float32
    
    # Object arrays (fixed size, padded)
    obj_type_id: np.ndarray           # (MAX_OBJ,) int16
    obj_x: np.ndarray                 # (MAX_OBJ,) float32
    obj_y: np.ndarray                 # (MAX_OBJ,) float32
    obj_vx: np.ndarray                # (MAX_OBJ,) float32
    obj_vy: np.ndarray                # (MAX_OBJ,) float32
    obj_ang: np.ndarray               # (MAX_OBJ,) float32
    obj_ang_vel: np.ndarray           # (MAX_OBJ,) float32
    obj_radius: np.ndarray            # (MAX_OBJ,) float32
    obj_mask: np.ndarray              # (MAX_OBJ,) bool
    
    # Optional image
    board_rgb: Optional[np.ndarray] = None
    
    def to_obs_dict(self) -> Dict[str, np.ndarray]:
        """Convert to Gymnasium observation dictionary."""
        obs = {
            # Core state
            "spawner_x": np.array(self.spawner_x, dtype=np.float32),
            "current_fruit_id": np.array(self.current_fruit_id, dtype=np.int32),
            "next_fruit_id": np.array(self.next_fruit_id, dtype=np.int32),
            "score": np.array(self.score, dtype=np.int64),
            "drops_used": np.array(self.drops_used, dtype=np.int32),
            "objects_count": np.array(self.objects_count, dtype=np.int32),
            
            # Board info
            "board_width": np.array(self.board_width, dtype=np.float32),
            "board_height": np.array(self.board_height, dtype=np.float32),
            "lose_line_y": np.array(self.lose_line_y, dtype=np.float32),
            
            # Basic derived
            "largest_fruit_type_id": np.array(self.largest_fruit_type_id, dtype=np.int32),
            "largest_fruit_x": np.array(self.largest_fruit_x, dtype=np.float32),
            "largest_fruit_y": np.array(self.largest_fruit_y, dtype=np.float32),
            "highest_fruit_y": np.array(self.highest_fruit_y, dtype=np.float32),
            "danger_level": np.array(self.danger_level, dtype=np.float32),
            "distance_to_lose_line": np.array(self.distance_to_lose_line, dtype=np.float32),
            
            # Advanced derived
            "center_of_mass_x": np.array(self.center_of_mass_x, dtype=np.float32),
            "center_of_mass_y": np.array(self.center_of_mass_y, dtype=np.float32),
            "packing_efficiency": np.array(self.packing_efficiency, dtype=np.float32),
            "surface_roughness": np.array(self.surface_roughness, dtype=np.float32),
            "island_count": np.array(self.island_count, dtype=np.int32),
            "buried_count": np.array(self.buried_count, dtype=np.int32),
            "neighbor_discord": np.array(self.neighbor_discord, dtype=np.float32),
            
            # Height map
            "height_map": self.height_map,
            
            # Object arrays
            "obj_type_id": self.obj_type_id,
            "obj_x": self.obj_x,
            "obj_y": self.obj_y,
            "obj_vx": self.obj_vx,
            "obj_vy": self.obj_vy,
            "obj_ang": self.obj_ang,
            "obj_ang_vel": self.obj_ang_vel,
            "obj_radius": self.obj_radius,
            "obj_mask": self.obj_mask,
        }
        
        if self.board_rgb is not None:
            obs["board_rgb"] = self.board_rgb
        
        return obs


class SnapshotBuilder:
    """Builds game state snapshots with pre-allocated arrays."""
    
    def __init__(self, config: Optional[GameConfig] = None):
        if config is None:
            config = get_config()
        
        self._config = config
        self._max_objects = config.observation.max_objects
        self._include_angular = config.observation.include_angular
        
        # Board info
        self._board_width = config.board.width
        self._board_height = config.board.height
        self._lose_line_y = config.board.lose_line_y
        
        # Pre-allocate arrays
        self._obj_type_id = np.zeros(self._max_objects, dtype=np.int16)
        self._obj_x = np.zeros(self._max_objects, dtype=np.float32)
        self._obj_y = np.zeros(self._max_objects, dtype=np.float32)
        self._obj_vx = np.zeros(self._max_objects, dtype=np.float32)
        self._obj_vy = np.zeros(self._max_objects, dtype=np.float32)
        self._obj_ang = np.zeros(self._max_objects, dtype=np.float32)
        self._obj_ang_vel = np.zeros(self._max_objects, dtype=np.float32)
        self._obj_radius = np.zeros(self._max_objects, dtype=np.float32)
        self._obj_mask = np.zeros(self._max_objects, dtype=bool)
        self._height_map = np.zeros(HEIGHT_MAP_SLICES, dtype=np.float32)
    
    @property
    def max_objects(self) -> int:
        return self._max_objects
    
    def build(
        self,
        physics: "PhysicsWorld",
        spawner_x: float,
        current_fruit_id: int,
        next_fruit_id: int,
        score: int,
        drops_used: int,
        board_rgb: Optional[np.ndarray] = None
    ) -> GameSnapshot:
        """Build a snapshot from current game state."""
        # Reset arrays
        self._obj_type_id.fill(-1)
        self._obj_x.fill(0)
        self._obj_y.fill(0)
        self._obj_vx.fill(0)
        self._obj_vy.fill(0)
        self._obj_ang.fill(0)
        self._obj_ang_vel.fill(0)
        self._obj_radius.fill(0)
        self._obj_mask.fill(False)
        self._height_map.fill(0)
        
        # Pack fruit data
        fruits = list(physics.fruits.values())
        count = min(len(fruits), self._max_objects)
        
        # Basic tracking
        largest_type_id = -1
        largest_x = 0.0
        largest_y = 0.0
        highest_y = 0.0
        
        # For advanced metrics
        total_mass = 0.0
        mass_x_sum = 0.0
        mass_y_sum = 0.0
        total_area = 0.0
        
        positions: List[Tuple[float, float, float, int]] = []  # (x, y, radius, type_id)
        
        for i in range(count):
            fruit = fruits[i]
            x, y = fruit.position
            radius = fruit.fruit_type.visual_radius
            type_id = fruit.type_id
            mass = fruit.fruit_type.mass
            
            self._obj_type_id[i] = type_id
            self._obj_x[i] = x
            self._obj_y[i] = y
            self._obj_vx[i] = fruit.velocity[0]
            self._obj_vy[i] = fruit.velocity[1]
            self._obj_radius[i] = radius
            if self._include_angular:
                self._obj_ang[i] = fruit.angle
                self._obj_ang_vel[i] = fruit.angular_velocity
            self._obj_mask[i] = True
            
            # Track largest fruit
            if type_id > largest_type_id:
                largest_type_id = type_id
                largest_x = x
                largest_y = y
            
            # Track highest fruit
            top_y = fruit.top_y
            if top_y > highest_y:
                highest_y = top_y
            
            # Accumulate for COM
            total_mass += mass
            mass_x_sum += mass * x
            mass_y_sum += mass * y
            
            # Area for packing
            total_area += math.pi * radius * radius
            
            positions.append((x, y, radius, type_id))
        
        # Calculate derived metrics
        if count > 0:
            danger_level = max(0.0, min(1.0, highest_y / self._lose_line_y))
            distance_to_lose_line = max(0.0, self._lose_line_y - highest_y)
            center_of_mass_x = mass_x_sum / total_mass
            center_of_mass_y = mass_y_sum / total_mass
        else:
            danger_level = 0.0
            distance_to_lose_line = self._lose_line_y
            center_of_mass_x = self._board_width / 2
            center_of_mass_y = 0.0
        
        # Compute advanced features
        height_map = self._compute_height_map(positions)
        surface_roughness = self._compute_surface_roughness(height_map)
        packing_efficiency = self._compute_packing_efficiency(positions, total_area)
        island_count = self._compute_island_count(positions)
        buried_count = self._compute_buried_count(positions)
        neighbor_discord = self._compute_neighbor_discord(positions)
        
        return GameSnapshot(
            spawner_x=spawner_x,
            current_fruit_id=current_fruit_id,
            next_fruit_id=next_fruit_id,
            score=score,
            drops_used=drops_used,
            objects_count=count,
            board_width=self._board_width,
            board_height=self._board_height,
            lose_line_y=self._lose_line_y,
            largest_fruit_type_id=largest_type_id,
            largest_fruit_x=largest_x,
            largest_fruit_y=largest_y,
            highest_fruit_y=highest_y,
            danger_level=danger_level,
            distance_to_lose_line=distance_to_lose_line,
            center_of_mass_x=center_of_mass_x,
            center_of_mass_y=center_of_mass_y,
            packing_efficiency=packing_efficiency,
            surface_roughness=surface_roughness,
            island_count=island_count,
            buried_count=buried_count,
            neighbor_discord=neighbor_discord,
            height_map=height_map.copy(),
            obj_type_id=self._obj_type_id.copy(),
            obj_x=self._obj_x.copy(),
            obj_y=self._obj_y.copy(),
            obj_vx=self._obj_vx.copy(),
            obj_vy=self._obj_vy.copy(),
            obj_ang=self._obj_ang.copy(),
            obj_ang_vel=self._obj_ang_vel.copy(),
            obj_radius=self._obj_radius.copy(),
            obj_mask=self._obj_mask.copy(),
            board_rgb=board_rgb
        )
    
    def _compute_height_map(
        self, positions: List[Tuple[float, float, float, int]]
    ) -> np.ndarray:
        """
        Compute 1D height map (lidar-style) across board width.
        
        Returns array of shape (HEIGHT_MAP_SLICES,) with max Y at each slice.
        """
        height_map = np.zeros(HEIGHT_MAP_SLICES, dtype=np.float32)
        slice_width = self._board_width / HEIGHT_MAP_SLICES
        
        for x, y, radius, _ in positions:
            # Find which slices this fruit covers
            left_slice = max(0, int((x - radius) / slice_width))
            right_slice = min(HEIGHT_MAP_SLICES - 1, int((x + radius) / slice_width))
            
            top_y = y + radius
            for s in range(left_slice, right_slice + 1):
                if top_y > height_map[s]:
                    height_map[s] = top_y
        
        return height_map
    
    def _compute_surface_roughness(self, height_map: np.ndarray) -> float:
        """Compute standard deviation of non-zero heights."""
        non_zero = height_map[height_map > 0]
        if len(non_zero) < 2:
            return 0.0
        return float(np.std(non_zero))
    
    def _compute_packing_efficiency(
        self, positions: List[Tuple[float, float, float, int]], total_area: float
    ) -> float:
        """
        Compute ratio of total fruit area to convex hull area.
        
        Returns 0-1 where 1 = perfectly packed.
        """
        if len(positions) < 3 or total_area <= 0:
            return 1.0  # Trivially packed
        
        # Get points for convex hull (use fruit edges)
        points = []
        for x, y, r, _ in positions:
            # Add 4 edge points per fruit
            points.extend([
                (x - r, y), (x + r, y), (x, y - r), (x, y + r)
            ])
        
        if len(points) < 3:
            return 1.0
        
        # Simple convex hull area using shoelace formula
        hull_area = self._convex_hull_area(points)
        
        if hull_area <= 0:
            return 1.0
        
        efficiency = min(1.0, total_area / hull_area)
        return float(efficiency)
    
    def _convex_hull_area(self, points: List[Tuple[float, float]]) -> float:
        """Compute convex hull area using gift-wrapping + shoelace."""
        if len(points) < 3:
            return 0.0
        
        # Simple convex hull using gift wrapping
        points = list(set(points))  # Remove duplicates
        if len(points) < 3:
            return 0.0
        
        # Find leftmost point
        start = min(points, key=lambda p: (p[0], p[1]))
        hull = [start]
        current = start
        
        while True:
            candidate = points[0]
            for p in points[1:]:
                if p == current:
                    continue
                cross = (candidate[0] - current[0]) * (p[1] - current[1]) - \
                        (candidate[1] - current[1]) * (p[0] - current[0])
                if candidate == current or cross < 0:
                    candidate = p
            
            if candidate == start:
                break
            hull.append(candidate)
            current = candidate
            
            if len(hull) > len(points):  # Safety
                break
        
        # Shoelace formula
        n = len(hull)
        if n < 3:
            return 0.0
        
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += hull[i][0] * hull[j][1]
            area -= hull[j][0] * hull[i][1]
        
        return abs(area) / 2.0
    
    def _compute_island_count(
        self, positions: List[Tuple[float, float, float, int]]
    ) -> int:
        """Count connected components (fruit clusters) using Union-Find."""
        n = len(positions)
        if n == 0:
            return 0
        if n == 1:
            return 1
        
        # Union-Find
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Connect touching fruits (distance < sum of radii + tolerance)
        tolerance = 5.0  # pixels
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, r1, _ = positions[i]
                x2, y2, r2, _ = positions[j]
                dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if dist < r1 + r2 + tolerance:
                    union(i, j)
        
        # Count unique roots
        roots = set(find(i) for i in range(n))
        return len(roots)
    
    def _compute_buried_count(
        self, positions: List[Tuple[float, float, float, int]]
    ) -> int:
        """
        Count fruits that are "buried" (blocked from above by larger fruits).
        
        A fruit is buried if it has neighbors on left, right, AND top.
        """
        if len(positions) < 2:
            return 0
        
        buried = 0
        wall_margin = 30  # Consider near-wall as blocked
        
        for i, (x, y, r, type_id) in enumerate(positions):
            has_left = x - r < wall_margin  # Near left wall
            has_right = x + r > self._board_width - wall_margin  # Near right wall
            has_top = False
            
            for j, (ox, oy, or_, _) in enumerate(positions):
                if i == j:
                    continue
                
                # Check if blocking from left
                if not has_left:
                    if ox < x and abs(oy - y) < r + or_:
                        has_left = True
                
                # Check if blocking from right
                if not has_right:
                    if ox > x and abs(oy - y) < r + or_:
                        has_right = True
                
                # Check if blocking from above
                if not has_top:
                    if oy > y and abs(ox - x) < r + or_:
                        has_top = True
            
            if has_left and has_right and has_top:
                buried += 1
        
        return buried
    
    def _compute_neighbor_discord(
        self, positions: List[Tuple[float, float, float, int]]
    ) -> float:
        """
        Compute average type difference between touching fruits.
        
        Low = well-sorted (similar types together), High = chaotic.
        """
        if len(positions) < 2:
            return 0.0
        
        total_diff = 0.0
        contact_count = 0
        tolerance = 5.0
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                x1, y1, r1, t1 = positions[i]
                x2, y2, r2, t2 = positions[j]
                dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                
                if dist < r1 + r2 + tolerance:
                    total_diff += abs(t1 - t2)
                    contact_count += 1
        
        if contact_count == 0:
            return 0.0
        
        return total_diff / contact_count
    
    def empty_snapshot(self) -> GameSnapshot:
        """Create an empty snapshot."""
        return GameSnapshot(
            spawner_x=0.0,
            current_fruit_id=0,
            next_fruit_id=0,
            score=0,
            drops_used=0,
            objects_count=0,
            board_width=self._board_width,
            board_height=self._board_height,
            lose_line_y=self._lose_line_y,
            largest_fruit_type_id=-1,
            largest_fruit_x=0.0,
            largest_fruit_y=0.0,
            highest_fruit_y=0.0,
            danger_level=0.0,
            distance_to_lose_line=self._lose_line_y,
            center_of_mass_x=self._board_width / 2,
            center_of_mass_y=0.0,
            packing_efficiency=1.0,
            surface_roughness=0.0,
            island_count=0,
            buried_count=0,
            neighbor_discord=0.0,
            height_map=np.zeros(HEIGHT_MAP_SLICES, dtype=np.float32),
            obj_type_id=np.zeros(self._max_objects, dtype=np.int16),
            obj_x=np.zeros(self._max_objects, dtype=np.float32),
            obj_y=np.zeros(self._max_objects, dtype=np.float32),
            obj_vx=np.zeros(self._max_objects, dtype=np.float32),
            obj_vy=np.zeros(self._max_objects, dtype=np.float32),
            obj_ang=np.zeros(self._max_objects, dtype=np.float32),
            obj_ang_vel=np.zeros(self._max_objects, dtype=np.float32),
            obj_radius=np.zeros(self._max_objects, dtype=np.float32),
            obj_mask=np.zeros(self._max_objects, dtype=bool),
            board_rgb=None
        )
