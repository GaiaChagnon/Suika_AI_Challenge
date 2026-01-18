"""
Fruit Catalog
=============

Provides convenient access to fruit type definitions loaded from config.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

from DO_NOT_MODIFY.suika_core.config_loader import (
    GameConfig,
    FruitConfig,
    CollisionCircle,
    get_config
)


@dataclass
class FruitType:
    """
    Runtime representation of a fruit type.
    
    Wraps FruitConfig with additional computed properties and convenience methods.
    """
    config: FruitConfig
    
    @property
    def id(self) -> int:
        return self.config.id
    
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def mass(self) -> float:
        return self.config.mass
    
    @property
    def visual_radius(self) -> float:
        return self.config.visual_radius
    
    @property
    def collision_circles(self) -> Tuple[CollisionCircle, ...]:
        return self.config.collision_circles
    
    @property
    def merge_score(self) -> int:
        return self.config.merge_score
    
    @property
    def friction(self) -> float:
        return self.config.friction
    
    @property
    def elasticity(self) -> float:
        return self.config.elasticity
    
    @property
    def color_solid(self) -> Tuple[int, int, int]:
        return self.config.color_solid
    
    @property
    def color_full(self) -> Tuple[int, int, int]:
        return self.config.color_full
    
    @property
    def is_final(self) -> bool:
        """True if this fruit cannot merge (e.g., skull)."""
        return self.config.is_final
    
    @property
    def is_composite(self) -> bool:
        """True if this fruit has multiple collision circles."""
        return len(self.collision_circles) > 1
    
    @property
    def bounding_radius(self) -> float:
        """
        Maximum distance from center to any point on collision shape.
        
        Used for broad-phase calculations and spawn positioning.
        """
        max_dist = 0.0
        for circle in self.collision_circles:
            dist = circle.radius + (circle.offset_x**2 + circle.offset_y**2)**0.5
            max_dist = max(max_dist, dist)
        return max_dist
    
    @property
    def total_collision_area(self) -> float:
        """Approximate collision area (sum of circle areas, may overlap)."""
        import math
        return sum(math.pi * c.radius**2 for c in self.collision_circles)
    
    def __repr__(self) -> str:
        return f"FruitType({self.id}: {self.name})"


class FruitCatalog:
    """
    Collection of all fruit types in the game ladder.
    
    Provides indexed access and helper methods for the fruit progression.
    """
    
    def __init__(self, config: Optional[GameConfig] = None):
        """
        Initialize catalog from game config.
        
        Args:
            config: GameConfig instance. If None, loads from default location.
        """
        if config is None:
            config = get_config()
        
        self._config = config
        self._types: Tuple[FruitType, ...] = tuple(
            FruitType(fruit_config) for fruit_config in config.fruits
        )
        self._spawnable_count = config.rng.spawnable_count
    
    def __len__(self) -> int:
        """Total number of fruit types."""
        return len(self._types)
    
    def __getitem__(self, fruit_id: int) -> FruitType:
        """Get fruit type by ID."""
        if 0 <= fruit_id < len(self._types):
            return self._types[fruit_id]
        raise IndexError(f"Fruit ID {fruit_id} out of range [0, {len(self._types)})")
    
    def __iter__(self):
        """Iterate over all fruit types."""
        return iter(self._types)
    
    @property
    def all_types(self) -> Tuple[FruitType, ...]:
        """All fruit types in order."""
        return self._types
    
    @property
    def spawnable_types(self) -> Tuple[FruitType, ...]:
        """Fruit types that can be spawned directly (first N types)."""
        return self._types[:self._spawnable_count]
    
    @property
    def spawnable_count(self) -> int:
        """Number of fruit types that can be spawned."""
        return self._spawnable_count
    
    @property
    def melon(self) -> FruitType:
        """The melon fruit type (ID 9)."""
        return self._types[9] if len(self._types) > 9 else self._types[-1]
    
    @property
    def melon_id(self) -> int:
        """ID of the melon type (9)."""
        return 9
    
    @property
    def watermelon(self) -> FruitType:
        """The watermelon fruit type (ID 10, largest mergeable)."""
        return self._types[10] if len(self._types) > 10 else self._types[-1]
    
    @property
    def watermelon_id(self) -> int:
        """ID of the watermelon type (10)."""
        return 10
    
    @property
    def skull(self) -> Optional[FruitType]:
        """The skull fruit type (ID 11, final form)."""
        return self._types[11] if len(self._types) > 11 else None
    
    @property
    def skull_id(self) -> int:
        """ID of the skull type (11)."""
        return 11
    
    def get_next_type(self, fruit_id: int) -> Optional[FruitType]:
        """
        Get the fruit type produced by merging two fruits of given type.
        
        Args:
            fruit_id: ID of fruits being merged.
            
        Returns:
            Next fruit type, or None if this fruit cannot merge.
        """
        # Final fruits (like skull) cannot merge
        if self._types[fruit_id].is_final:
            return None
        next_id = fruit_id + 1
        if next_id >= len(self._types):
            return None
        return self._types[next_id]
    
    def is_melon(self, fruit_id: int) -> bool:
        """Check if a fruit ID is the melon type."""
        return fruit_id == self.melon_id
    
    def is_watermelon(self, fruit_id: int) -> bool:
        """Check if a fruit ID is the watermelon type."""
        return fruit_id == self.watermelon_id
    
    def is_skull(self, fruit_id: int) -> bool:
        """Check if a fruit ID is the skull type."""
        return fruit_id == self.skull_id
    
    def is_final_fruit(self, fruit_id: int) -> bool:
        """Check if a fruit cannot merge (e.g., skull)."""
        if 0 <= fruit_id < len(self._types):
            return self._types[fruit_id].is_final
        return False
    
    def is_spawnable(self, fruit_id: int) -> bool:
        """Check if a fruit ID is in the spawnable set."""
        return 0 <= fruit_id < self._spawnable_count
    
    def get_by_name(self, name: str) -> Optional[FruitType]:
        """Get fruit type by name (case-insensitive)."""
        name_lower = name.lower()
        for fruit_type in self._types:
            if fruit_type.name.lower() == name_lower:
                return fruit_type
        return None


# Module-level singleton
_cached_catalog: Optional[FruitCatalog] = None


def get_catalog(config: Optional[GameConfig] = None) -> FruitCatalog:
    """
    Get the fruit catalog singleton.
    
    Args:
        config: Optional config to use. If None, uses cached or default config.
        
    Returns:
        FruitCatalog instance.
    """
    global _cached_catalog
    if _cached_catalog is None or config is not None:
        _cached_catalog = FruitCatalog(config)
    return _cached_catalog
