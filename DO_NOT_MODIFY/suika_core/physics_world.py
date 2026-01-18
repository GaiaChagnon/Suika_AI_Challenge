"""
Physics World
=============

Manages the pymunk Space, static walls, and fruit body creation/removal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Set
import math

import pymunk

from DO_NOT_MODIFY.suika_core.config_loader import GameConfig, get_config
from DO_NOT_MODIFY.suika_core.fruit_catalog import FruitType, FruitCatalog, get_catalog


# Collision types for pymunk
COLLISION_TYPE_FRUIT = 1
COLLISION_TYPE_WALL = 2


@dataclass
class FruitBody:
    """
    Represents a fruit instance in the physics world.
    
    Wraps pymunk Body and shapes with game-specific metadata.
    """
    uid: int
    fruit_type: FruitType
    body: pymunk.Body
    shapes: Tuple[pymunk.Circle, ...]
    
    @property
    def type_id(self) -> int:
        return self.fruit_type.id
    
    @property
    def position(self) -> Tuple[float, float]:
        return self.body.position.x, self.body.position.y
    
    @property
    def velocity(self) -> Tuple[float, float]:
        return self.body.velocity.x, self.body.velocity.y
    
    @property
    def angle(self) -> float:
        return self.body.angle
    
    @property
    def angular_velocity(self) -> float:
        return self.body.angular_velocity
    
    @property
    def is_sleeping(self) -> bool:
        return self.body.is_sleeping
    
    @property
    def top_y(self) -> float:
        """Highest Y coordinate of this fruit (for lose line check)."""
        max_y = self.body.position.y
        for shape in self.shapes:
            offset = shape.offset.rotated(self.body.angle)
            y = self.body.position.y + offset.y + shape.radius
            max_y = max(max_y, y)
        return max_y
    
    @property
    def speed(self) -> float:
        """Linear speed magnitude."""
        vx, vy = self.velocity
        return math.sqrt(vx*vx + vy*vy)


class PhysicsWorld:
    """
    Manages the pymunk physics simulation.
    
    Handles:
    - Space creation and configuration
    - Static wall segments
    - Fruit body creation and removal
    - Physics stepping
    - Collision callback registration
    - Domain randomization for robust training
    """
    
    def __init__(self, config: Optional[GameConfig] = None):
        """
        Initialize physics world.
        
        Args:
            config: Game configuration. Uses default if None.
        """
        if config is None:
            config = get_config()
        
        self._config = config
        self._catalog = get_catalog(config)
        
        # Domain randomization factors (1.0 = no change)
        self._friction_factor = 1.0
        self._elasticity_factor = 1.0
        self._mass_factors: Dict[int, float] = {}  # per fruit type
        self._gravity_factor = 1.0
        
        # Create space with gravity
        self._space = pymunk.Space()
        self._space.gravity = config.physics.gravity
        self._space.damping = config.physics.damping
        
        # Enable sleeping for performance
        self._space.sleep_time_threshold = 0.5
        self._space.idle_speed_threshold = config.physics.settle_velocity_threshold
        
        # Track fruit bodies
        self._fruits: Dict[int, FruitBody] = {}
        self._next_uid = 0
        
        # Create walls
        self._wall_shapes: List[pymunk.Segment] = []
        self._create_walls()
        
        # Collision handler placeholder (set by merge_system)
        self._collision_handler: Optional[Callable] = None
    
    def apply_domain_randomization(self, seed: Optional[int] = None) -> None:
        """
        Apply domain randomization for robust training.
        
        Varies friction, elasticity, mass, and gravity slightly.
        Same seed produces same randomization for fair evaluation.
        
        Args:
            seed: Random seed for reproducibility.
        """
        import numpy as np
        
        dr = self._config.domain_randomization
        if not dr.enabled:
            # Reset to defaults
            self._friction_factor = 1.0
            self._elasticity_factor = 1.0
            self._mass_factors = {}
            self._gravity_factor = 1.0
            return
        
        # Create local RNG seeded by episode seed
        rng = np.random.default_rng(seed)
        
        # Sample random factors (uniform distribution around 1.0)
        self._friction_factor = 1.0 + rng.uniform(-dr.friction_variance, dr.friction_variance)
        self._elasticity_factor = 1.0 + rng.uniform(-dr.elasticity_variance, dr.elasticity_variance)
        self._gravity_factor = 1.0 + rng.uniform(-dr.gravity_variance, dr.gravity_variance)
        
        # Per-fruit mass factors
        self._mass_factors = {}
        for fruit in self._config.fruits:
            self._mass_factors[fruit.id] = 1.0 + rng.uniform(-dr.mass_variance, dr.mass_variance)
        
        # Apply gravity randomization
        gx, gy = self._config.physics.gravity
        self._space.gravity = (gx * self._gravity_factor, gy * self._gravity_factor)
        
        # Update wall friction/elasticity
        for wall in self._wall_shapes:
            wall.friction = self._config.physics.default_friction * self._friction_factor
            wall.elasticity = self._config.physics.default_elasticity * self._elasticity_factor
    
    def _create_walls(self) -> None:
        """Create static wall segments."""
        board = self._config.board
        
        # Wall thickness
        thickness = 10.0
        
        # Create static body for walls
        static_body = self._space.static_body
        
        # Bottom wall
        bottom = pymunk.Segment(
            static_body,
            (0, 0),
            (board.width, 0),
            thickness
        )
        bottom.friction = self._config.physics.default_friction
        bottom.elasticity = self._config.physics.default_elasticity
        bottom.collision_type = COLLISION_TYPE_WALL
        self._wall_shapes.append(bottom)
        
        # Left wall
        left = pymunk.Segment(
            static_body,
            (0, 0),
            (0, board.height),
            thickness
        )
        left.friction = self._config.physics.default_friction
        left.elasticity = self._config.physics.default_elasticity
        left.collision_type = COLLISION_TYPE_WALL
        self._wall_shapes.append(left)
        
        # Right wall
        right = pymunk.Segment(
            static_body,
            (board.width, 0),
            (board.width, board.height),
            thickness
        )
        right.friction = self._config.physics.default_friction
        right.elasticity = self._config.physics.default_elasticity
        right.collision_type = COLLISION_TYPE_WALL
        self._wall_shapes.append(right)
        
        self._space.add(*self._wall_shapes)
    
    @property
    def space(self) -> pymunk.Space:
        """The pymunk Space instance."""
        return self._space
    
    @property
    def fruits(self) -> Dict[int, FruitBody]:
        """Dictionary of all fruit bodies by UID."""
        return self._fruits
    
    @property
    def fruit_count(self) -> int:
        """Number of fruits currently in world."""
        return len(self._fruits)
    
    @property
    def board_width(self) -> int:
        """Width of the game board."""
        return self._config.board.width
    
    @property
    def board_height(self) -> int:
        """Height of the game board."""
        return self._config.board.height
    
    def set_collision_handler(
        self,
        handler: Callable[[pymunk.Arbiter, pymunk.Space, any], None]
    ) -> None:
        """
        Set the fruit-fruit collision handler.
        
        Args:
            handler: Collision callback function (arbiter, space, data).
        """
        # pymunk 7.x uses on_collision() instead of add_collision_handler()
        self._space.on_collision(begin=handler)
        self._collision_handler = handler
    
    def spawn_fruit(
        self,
        fruit_type: FruitType,
        x: float,
        y: float,
        velocity: Tuple[float, float] = (0, 0)
    ) -> FruitBody:
        """
        Spawn a new fruit at the specified position.
        
        Args:
            fruit_type: Type of fruit to spawn.
            x: X coordinate.
            y: Y coordinate.
            velocity: Initial velocity (default stationary).
            
        Returns:
            The created FruitBody.
        """
        # Apply mass randomization
        mass_factor = self._mass_factors.get(fruit_type.id, 1.0)
        effective_mass = fruit_type.mass * mass_factor
        
        # Calculate moment of inertia for composite shape
        moment = 0.0
        for circle in fruit_type.collision_circles:
            m = pymunk.moment_for_circle(
                effective_mass / len(fruit_type.collision_circles),
                0,
                circle.radius,
                (circle.offset_x, circle.offset_y)
            )
            moment += m
        
        # Create body
        body = pymunk.Body(effective_mass, moment)
        body.position = (x, y)
        body.velocity = velocity
        
        # Apply friction/elasticity randomization
        effective_friction = fruit_type.friction * self._friction_factor
        effective_elasticity = fruit_type.elasticity * self._elasticity_factor
        
        # Create shapes for each collision circle
        shapes = []
        for circle in fruit_type.collision_circles:
            shape = pymunk.Circle(
                body,
                circle.radius,
                (circle.offset_x, circle.offset_y)
            )
            shape.friction = effective_friction
            shape.elasticity = effective_elasticity
            shape.collision_type = COLLISION_TYPE_FRUIT
            shapes.append(shape)
        
        shapes_tuple = tuple(shapes)
        
        # Assign UID
        uid = self._next_uid
        self._next_uid += 1
        
        # Store UID in body for collision lookup
        body.fruit_uid = uid
        
        # Create FruitBody wrapper
        fruit_body = FruitBody(
            uid=uid,
            fruit_type=fruit_type,
            body=body,
            shapes=shapes_tuple
        )
        
        # Add to space and tracking
        self._space.add(body, *shapes_tuple)
        self._fruits[uid] = fruit_body
        
        return fruit_body
    
    def remove_fruit(self, uid: int) -> Optional[FruitBody]:
        """
        Remove a fruit from the world.
        
        Args:
            uid: Unique ID of fruit to remove.
            
        Returns:
            The removed FruitBody, or None if not found.
        """
        fruit = self._fruits.pop(uid, None)
        if fruit is not None:
            self._space.remove(fruit.body, *fruit.shapes)
        return fruit
    
    def get_fruit(self, uid: int) -> Optional[FruitBody]:
        """Get a fruit by UID."""
        return self._fruits.get(uid)
    
    def get_fruit_by_body(self, body: pymunk.Body) -> Optional[FruitBody]:
        """Get a fruit by its pymunk Body."""
        uid = getattr(body, 'fruit_uid', None)
        if uid is not None:
            return self._fruits.get(uid)
        return None
    
    def step(self, dt: Optional[float] = None) -> None:
        """
        Advance physics simulation by one timestep.
        
        Args:
            dt: Timestep duration. Uses config default if None.
        """
        if dt is None:
            dt = self._config.physics.dt
        
        for _ in range(self._config.physics.substeps):
            self._space.step(dt / self._config.physics.substeps)
    
    def check_settled(self) -> bool:
        """
        Check if all fruits are settled (below velocity thresholds).
        
        Returns:
            True if all fruits are settled.
        """
        vel_threshold = self._config.physics.settle_velocity_threshold
        ang_threshold = self._config.physics.settle_angular_threshold
        
        for fruit in self._fruits.values():
            if fruit.speed > vel_threshold:
                return False
            if abs(fruit.angular_velocity) > ang_threshold:
                return False
        
        return True
    
    def force_freeze(self) -> None:
        """Force all fruits to zero velocity (emergency settle)."""
        for fruit in self._fruits.values():
            fruit.body.velocity = (0, 0)
            fruit.body.angular_velocity = 0
            fruit.body.force = (0, 0)
            fruit.body.torque = 0
    
    def get_fruits_above_line(self, y: float) -> List[FruitBody]:
        """Get all fruits with top edge above the specified Y coordinate."""
        return [f for f in self._fruits.values() if f.top_y > y]
    
    def apply_impulse(
        self,
        uid: int,
        impulse: Tuple[float, float],
        point: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Apply an impulse to a fruit.
        
        Args:
            uid: Fruit UID.
            impulse: Impulse vector (px, py).
            point: Application point (world coords). Uses center if None.
        """
        fruit = self._fruits.get(uid)
        if fruit is not None:
            if point is None:
                point = fruit.body.position
            fruit.body.apply_impulse_at_world_point(impulse, point)
    
    def clear(self) -> None:
        """Remove all fruits from the world."""
        for uid in list(self._fruits.keys()):
            self.remove_fruit(uid)
        self._next_uid = 0
    
    def get_all_positions(self) -> List[Tuple[int, float, float]]:
        """Get (type_id, x, y) for all fruits."""
        return [
            (f.type_id, f.position[0], f.position[1])
            for f in self._fruits.values()
        ]
