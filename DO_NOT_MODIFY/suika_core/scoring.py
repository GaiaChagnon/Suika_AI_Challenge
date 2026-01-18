"""
Scoring System
==============

Applies merge scores based on game configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from DO_NOT_MODIFY.suika_core.config_loader import GameConfig, get_config
from DO_NOT_MODIFY.suika_core.fruit_catalog import FruitCatalog, get_catalog


@dataclass
class ScoreEvent:
    """Record of a scoring event."""
    points: int
    fruit_type_id: int
    is_melon_bonus: bool
    is_skull_bonus: bool = False  # True when watermelons merge into skull
    skull_multiplier: float = 1.0  # Multiplier from skulls present
    
    def __repr__(self) -> str:
        if self.is_skull_bonus:
            return f"ScoreEvent(skull_bonus={self.points}, multiplier={self.skull_multiplier}x)"
        if self.is_melon_bonus:
            return f"ScoreEvent(melon_bonus={self.points})"
        return f"ScoreEvent(merge_to_{self.fruit_type_id}={self.points})"


class ScoreTracker:
    """
    Tracks game score and provides merge score calculations.
    
    Skulls provide a 50% point multiplier each:
    - 0 skulls: 1.0x
    - 1 skull: 1.5x
    - 2 skulls: 2.0x
    - n skulls: (1 + 0.5*n)x
    """
    
    # Multiplier per skull present on board
    SKULL_MULTIPLIER_PER = 0.5
    
    def __init__(self, config: Optional[GameConfig] = None):
        """
        Initialize score tracker.
        
        Args:
            config: Game configuration. Uses default if None.
        """
        if config is None:
            config = get_config()
        
        self._config = config
        self._catalog = get_catalog(config)
        self._score: int = 0
        self._merges: int = 0
        self._skull_count: int = 0
    
    @property
    def score(self) -> int:
        """Current total score."""
        return self._score
    
    @property
    def merges(self) -> int:
        """Total number of merges performed."""
        return self._merges
    
    @property
    def skull_count(self) -> int:
        """Number of skulls currently on board."""
        return self._skull_count
    
    @skull_count.setter
    def skull_count(self, value: int) -> None:
        """Set the current skull count (called by game)."""
        self._skull_count = max(0, value)
    
    def get_skull_multiplier(self) -> float:
        """Get current score multiplier from skulls (1.0 + 0.5 * skull_count)."""
        return 1.0 + self.SKULL_MULTIPLIER_PER * self._skull_count
    
    def get_merge_score(self, merged_from_type_id: int) -> int:
        """
        Get the base score for merging two fruits of a given type.
        
        Note: Does not include skull multiplier. Use get_multiplied_score() for that.
        
        Args:
            merged_from_type_id: Type ID of the fruits being merged.
            
        Returns:
            Base points awarded for this merge.
        """
        next_type = self._catalog.get_next_type(merged_from_type_id)
        if next_type is None:
            # Watermelon-watermelon merge -> skull
            if self._catalog.is_watermelon(merged_from_type_id):
                return self._catalog.skull.merge_score if self._catalog.skull else 100
            # Melon-melon merge (shouldn't happen now, but kept for compatibility)
            return self._config.scoring.melon_melon_bonus
        return next_type.merge_score
    
    def apply_merge(self, merged_from_type_id: int) -> ScoreEvent:
        """
        Apply score for a merge and return the event.
        
        Score is multiplied by skull count: (1 + 0.5 * skull_count).
        
        Args:
            merged_from_type_id: Type ID of the fruits being merged.
            
        Returns:
            ScoreEvent describing the points awarded.
        """
        multiplier = self.get_skull_multiplier()
        is_watermelon = self._catalog.is_watermelon(merged_from_type_id)
        is_melon = self._catalog.is_melon(merged_from_type_id)
        
        if is_watermelon:
            # Watermelon-watermelon merge -> creates skull
            skull_type = self._catalog.skull
            base_points = skull_type.merge_score if skull_type else 100
            points = int(base_points * multiplier)
            event = ScoreEvent(
                points=points,
                fruit_type_id=self._catalog.skull_id,
                is_melon_bonus=False,
                is_skull_bonus=True,
                skull_multiplier=multiplier
            )
        elif is_melon:
            base_points = self._config.scoring.melon_melon_bonus
            points = int(base_points * multiplier)
            event = ScoreEvent(
                points=points,
                fruit_type_id=merged_from_type_id,
                is_melon_bonus=True,
                skull_multiplier=multiplier
            )
        else:
            next_type = self._catalog.get_next_type(merged_from_type_id)
            base_points = next_type.merge_score
            points = int(base_points * multiplier)
            event = ScoreEvent(
                points=points,
                fruit_type_id=next_type.id,
                is_melon_bonus=False,
                skull_multiplier=multiplier
            )
        
        self._score += points
        self._merges += 1
        return event
    
    def reset(self) -> None:
        """Reset score to zero."""
        self._score = 0
        self._merges = 0
        self._skull_count = 0
    
    def add_bonus(self, points: int) -> None:
        """Add bonus points (for special events)."""
        self._score += points
