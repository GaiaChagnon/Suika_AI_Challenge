"""
Configuration Loader
====================

Loads and validates game_config.yaml, providing typed access to all parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

import yaml


@dataclass(frozen=True)
class BoardConfig:
    """Board geometry and lose line settings."""
    width: int                   # Fixed board width in pixels
    height: int                  # Total arena height in pixels
    lose_line_y: float           # Y coordinate of lose line
    lose_line_grace_time: float  # Seconds before game over triggers
    spawn_y: float               # Y coordinate where fruits spawn
    spawn_margin: int            # Extra margin from walls (pixels)


@dataclass(frozen=True)
class PhysicsConfig:
    """Physics simulation parameters."""
    gravity_x: float
    gravity_y: float
    damping: float
    dt: float
    substeps: int
    settle_velocity_threshold: float
    settle_angular_threshold: float
    settle_consecutive_ticks: int
    max_sim_seconds_per_drop: float
    merge_resets_timer: bool
    default_friction: float
    default_elasticity: float
    
    @property
    def gravity(self) -> Tuple[float, float]:
        return (self.gravity_x, self.gravity_y)


@dataclass(frozen=True)
class CollisionCircle:
    """A single circle in a composite collision shape."""
    radius: float
    offset_x: float
    offset_y: float


@dataclass(frozen=True)
class FruitConfig:
    """Configuration for a single fruit type."""
    id: int
    name: str
    mass: float
    visual_radius: float
    collision_circles: Tuple[CollisionCircle, ...]
    merge_score: int
    friction: float
    elasticity: float
    color_solid: Tuple[int, int, int]
    color_full: Tuple[int, int, int]
    is_final: bool = False  # If True, cannot merge with anything (e.g., skull)


@dataclass(frozen=True)
class MergeConfig:
    """Merge behavior parameters."""
    impulse_upward: float
    impulse_variance: float
    radial_impulse: float


@dataclass(frozen=True)
class RngConfig:
    """RNG and spawn queue parameters."""
    spawnable_count: int
    bag_size: int
    weights: Tuple[int, ...]


@dataclass(frozen=True)
class ScoringConfig:
    """Scoring parameters."""
    melon_melon_bonus: int


@dataclass(frozen=True)
class CapsConfig:
    """Game limits."""
    max_drops: int
    max_objects: int
    out_of_bounds_distance: float  # Pixels beyond board edge before termination


@dataclass(frozen=True)
class ObservationConfig:
    """Observation space parameters."""
    max_objects: int
    include_angular: bool
    image_enabled: bool
    image_width: int
    image_height: int
    render_style: str


@dataclass(frozen=True)
class DomainRandomizationConfig:
    """Domain randomization parameters for robust training."""
    enabled: bool
    friction_variance: float   # ±% variation (e.g., 0.02 = ±2%)
    elasticity_variance: float
    mass_variance: float
    gravity_variance: float


@dataclass(frozen=True)
class GameConfig:
    """
    Complete game configuration loaded from YAML.
    
    All values are immutable to prevent accidental modification during runtime.
    """
    board: BoardConfig
    physics: PhysicsConfig
    fruits: Tuple[FruitConfig, ...]
    merge: MergeConfig
    rng: RngConfig
    scoring: ScoringConfig
    caps: CapsConfig
    observation: ObservationConfig
    domain_randomization: DomainRandomizationConfig
    
    @property
    def melon_type(self) -> FruitConfig:
        """The largest fruit type (melon)."""
        return self.fruits[-1]
    
    @property
    def melon_radius(self) -> float:
        """Visual radius of the melon."""
        return self.melon_type.visual_radius
    
    @property
    def num_fruit_types(self) -> int:
        """Total number of fruit types in the ladder."""
        return len(self.fruits)
    
    def get_fruit(self, fruit_id: int) -> FruitConfig:
        """Get fruit config by ID."""
        if 0 <= fruit_id < len(self.fruits):
            return self.fruits[fruit_id]
        raise ValueError(f"Invalid fruit ID: {fruit_id}")


def _parse_collision_circles(circles_data: List) -> Tuple[CollisionCircle, ...]:
    """Parse collision circle data from YAML."""
    circles = []
    for circle in circles_data:
        if len(circle) != 3:
            raise ValueError(f"Collision circle must have 3 values [radius, offset_x, offset_y], got {circle}")
        circles.append(CollisionCircle(
            radius=float(circle[0]),
            offset_x=float(circle[1]),
            offset_y=float(circle[2])
        ))
    return tuple(circles)


def _parse_color(color_data: List) -> Tuple[int, int, int]:
    """Parse RGB color from YAML."""
    if len(color_data) != 3:
        raise ValueError(f"Color must have 3 values [R, G, B], got {color_data}")
    return (int(color_data[0]), int(color_data[1]), int(color_data[2]))


def _parse_fruit(fruit_data: dict, default_friction: float, default_elasticity: float) -> FruitConfig:
    """Parse a single fruit configuration from YAML."""
    return FruitConfig(
        id=int(fruit_data["id"]),
        name=str(fruit_data["name"]),
        mass=float(fruit_data["mass"]),
        visual_radius=float(fruit_data["visual_radius"]),
        collision_circles=_parse_collision_circles(fruit_data["collision_circles"]),
        merge_score=int(fruit_data["merge_score"]),
        friction=float(fruit_data.get("friction", default_friction)),
        elasticity=float(fruit_data.get("elasticity", default_elasticity)),
        color_solid=_parse_color(fruit_data["color_solid"]),
        color_full=_parse_color(fruit_data.get("color_full", fruit_data["color_solid"])),
        is_final=bool(fruit_data.get("is_final", False))
    )


def _validate_config(config: GameConfig) -> None:
    """Validate configuration consistency."""
    # Validate fruit IDs are sequential
    for i, fruit in enumerate(config.fruits):
        if fruit.id != i:
            raise ValueError(f"Fruit ID mismatch: expected {i}, got {fruit.id}")
    
    # Validate RNG weights match spawnable count
    if len(config.rng.weights) != config.rng.spawnable_count:
        raise ValueError(
            f"RNG weights length ({len(config.rng.weights)}) must match "
            f"spawnable_count ({config.rng.spawnable_count})"
        )
    
    # Validate spawnable count doesn't exceed fruit count
    if config.rng.spawnable_count > len(config.fruits):
        raise ValueError(
            f"spawnable_count ({config.rng.spawnable_count}) exceeds "
            f"fruit count ({len(config.fruits)})"
        )
    
    # Validate observation max_objects matches caps
    if config.observation.max_objects != config.caps.max_objects:
        raise ValueError(
            f"observation.max_objects ({config.observation.max_objects}) must match "
            f"caps.max_objects ({config.caps.max_objects})"
        )
    
    # Validate render style
    if config.observation.render_style not in ("solid", "full"):
        raise ValueError(f"render_style must be 'solid' or 'full', got '{config.observation.render_style}'")


def load_config(config_path: Optional[str] = None) -> GameConfig:
    """
    Load and validate game configuration from YAML.
    
    Args:
        config_path: Path to game_config.yaml. If None, uses default location.
        
    Returns:
        Validated GameConfig instance.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config validation fails.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "game_config.yaml"
        )
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    
    # Parse physics first to get defaults
    physics_data = raw["physics"]
    physics = PhysicsConfig(
        gravity_x=float(physics_data["gravity_x"]),
        gravity_y=float(physics_data["gravity_y"]),
        damping=float(physics_data["damping"]),
        dt=float(physics_data["dt"]),
        substeps=int(physics_data.get("substeps", 1)),
        settle_velocity_threshold=float(physics_data["settle_velocity_threshold"]),
        settle_angular_threshold=float(physics_data["settle_angular_threshold"]),
        settle_consecutive_ticks=int(physics_data["settle_consecutive_ticks"]),
        max_sim_seconds_per_drop=float(physics_data["max_sim_seconds_per_drop"]),
        merge_resets_timer=bool(physics_data.get("merge_resets_timer", True)),
        default_friction=float(physics_data["default_friction"]),
        default_elasticity=float(physics_data["default_elasticity"])
    )
    
    # Parse fruits
    fruits = tuple(
        _parse_fruit(f, physics.default_friction, physics.default_elasticity)
        for f in raw["fruits"]
    )
    
    # Parse board config
    board_data = raw["board"]
    board = BoardConfig(
        width=int(board_data["width"]),
        height=int(board_data["height"]),
        lose_line_y=float(board_data["lose_line_y"]),
        lose_line_grace_time=float(board_data["lose_line_grace_time"]),
        spawn_y=float(board_data["spawn_y"]),
        spawn_margin=int(board_data.get("spawn_margin", 5))
    )
    
    # Parse remaining sections
    merge_data = raw["merge"]
    merge = MergeConfig(
        impulse_upward=float(merge_data["impulse_upward"]),
        impulse_variance=float(merge_data.get("impulse_variance", 0)),
        radial_impulse=float(merge_data.get("radial_impulse", 150.0))
    )
    
    rng_data = raw["rng"]
    rng = RngConfig(
        spawnable_count=int(rng_data["spawnable_count"]),
        bag_size=int(rng_data["bag_size"]),
        weights=tuple(int(w) for w in rng_data["weights"])
    )
    
    scoring_data = raw["scoring"]
    scoring = ScoringConfig(
        melon_melon_bonus=int(scoring_data["melon_melon_bonus"])
    )
    
    caps_data = raw["caps"]
    caps = CapsConfig(
        max_drops=int(caps_data["max_drops"]),
        max_objects=int(caps_data.get("max_objects", 200)),
        out_of_bounds_distance=float(caps_data.get("out_of_bounds_distance", 300.0))
    )
    
    obs_data = raw["observation"]
    observation = ObservationConfig(
        max_objects=int(obs_data["max_objects"]),
        include_angular=bool(obs_data.get("include_angular", True)),
        image_enabled=bool(obs_data.get("image_enabled", False)),
        image_width=int(obs_data.get("image_width", 270)),
        image_height=int(obs_data.get("image_height", 400)),
        render_style=str(obs_data.get("render_style", "solid"))
    )
    
    # Parse domain randomization (optional section)
    dr_data = raw.get("domain_randomization", {})
    domain_randomization = DomainRandomizationConfig(
        enabled=bool(dr_data.get("enabled", True)),
        friction_variance=float(dr_data.get("friction_variance", 0.02)),
        elasticity_variance=float(dr_data.get("elasticity_variance", 0.02)),
        mass_variance=float(dr_data.get("mass_variance", 0.04)),
        gravity_variance=float(dr_data.get("gravity_variance", 0.01))
    )
    
    config = GameConfig(
        board=board,
        physics=physics,
        fruits=fruits,
        merge=merge,
        rng=rng,
        scoring=scoring,
        caps=caps,
        observation=observation,
        domain_randomization=domain_randomization
    )
    
    _validate_config(config)
    return config


# Module-level singleton for convenience
_cached_config: Optional[GameConfig] = None


def get_config() -> GameConfig:
    """Get the cached game configuration, loading if necessary."""
    global _cached_config
    if _cached_config is None:
        _cached_config = load_config()
    return _cached_config


def reload_config(config_path: Optional[str] = None) -> GameConfig:
    """Reload the configuration (useful for testing)."""
    global _cached_config
    _cached_config = load_config(config_path)
    return _cached_config
