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
    
    def __repr__(self) -> str:
        if self.is_melon_bonus:
            return f"ScoreEvent(melon_bonus={self.points})"
        return f"ScoreEvent(merge_to_{self.fruit_type_id}={self.points})"


class ScoreTracker:
    """
    Tracks game score and provides merge score calculations.
    """
    
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
    
    @property
    def score(self) -> int:
        """Current total score."""
        return self._score
    
    @property
    def merges(self) -> int:
        """Total number of merges performed."""
        return self._merges
    
    def get_merge_score(self, merged_from_type_id: int) -> int:
        """
        Get the score for merging two fruits of a given type.
        
        Args:
            merged_from_type_id: Type ID of the fruits being merged.
            
        Returns:
            Points awarded for this merge.
        """
        next_type = self._catalog.get_next_type(merged_from_type_id)
        if next_type is None:
            # Melon-melon merge
            return self._config.scoring.melon_melon_bonus
        return next_type.merge_score
    
    def apply_merge(self, merged_from_type_id: int) -> ScoreEvent:
        """
        Apply score for a merge and return the event.
        
        Args:
            merged_from_type_id: Type ID of the fruits being merged.
            
        Returns:
            ScoreEvent describing the points awarded.
        """
        is_melon = self._catalog.is_melon(merged_from_type_id)
        
        if is_melon:
            points = self._config.scoring.melon_melon_bonus
            event = ScoreEvent(
                points=points,
                fruit_type_id=merged_from_type_id,
                is_melon_bonus=True
            )
        else:
            next_type = self._catalog.get_next_type(merged_from_type_id)
            points = next_type.merge_score
            event = ScoreEvent(
                points=points,
                fruit_type_id=next_type.id,
                is_melon_bonus=False
            )
        
        self._score += points
        self._merges += 1
        return event
    
    def reset(self) -> None:
        """Reset score to zero."""
        self._score = 0
        self._merges = 0
    
    def add_bonus(self, points: int) -> None:
        """Add bonus points (for special events)."""
        self._score += points
