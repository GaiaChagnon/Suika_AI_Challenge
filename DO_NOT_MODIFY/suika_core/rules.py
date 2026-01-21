"""
Game Rules
==========

Handles spawn positioning, termination conditions, and game caps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

from DO_NOT_MODIFY.suika_core.config_loader import GameConfig, get_config
from DO_NOT_MODIFY.suika_core.fruit_catalog import FruitType, FruitCatalog, get_catalog


@dataclass
class TerminationResult:
    """Result of termination check."""
    terminated: bool
    truncated: bool
    reason: str
    
    @staticmethod
    def none() -> "TerminationResult":
        return TerminationResult(False, False, "")
    
    @staticmethod
    def game_over(reason: str) -> "TerminationResult":
        return TerminationResult(True, False, reason)
    
    @staticmethod
    def truncation(reason: str) -> "TerminationResult":
        return TerminationResult(False, True, reason)


class SpawnRules:
    """
    Handles spawn position calculation.
    
    Maps normalized action [-1, 1] to world X coordinate,
    ensuring fruits don't spawn inside walls.
    """
    
    def __init__(self, config: Optional[GameConfig] = None):
        """
        Initialize spawn rules.
        
        Args:
            config: Game configuration. Uses default if None.
        """
        if config is None:
            config = get_config()
        
        self._config = config
        self._catalog = get_catalog(config)
        
        # Base spawn Y
        self._spawn_y = config.board.spawn_y
        self._spawn_margin = config.board.spawn_margin
        self._board_width = config.board.width
    
    def get_spawn_x_range(self, fruit_type: FruitType) -> Tuple[float, float]:
        """
        Get valid spawn X range for a fruit type.
        
        Args:
            fruit_type: The fruit type to spawn.
            
        Returns:
            (min_x, max_x) tuple.
        """
        # Use bounding radius to ensure no wall clipping
        radius = fruit_type.bounding_radius
        min_x = radius + self._spawn_margin
        max_x = self._board_width - radius - self._spawn_margin
        return (min_x, max_x)
    
    def action_to_spawn_x(
        self,
        action: float,
        fruit_type: FruitType
    ) -> float:
        """
        Convert normalized action [-1, 1] to world X coordinate.
        
        Args:
            action: Normalized X position in [-1, 1].
            fruit_type: The fruit type being spawned.
            
        Returns:
            World X coordinate.
        """
        # Clamp action to valid range
        action = max(-1.0, min(1.0, action))
        
        min_x, max_x = self.get_spawn_x_range(fruit_type)
        
        # Map [-1, 1] to [min_x, max_x]
        t = (action + 1.0) / 2.0  # [0, 1]
        return min_x + t * (max_x - min_x)
    
    def spawn_x_to_action(
        self,
        x: float,
        fruit_type: FruitType
    ) -> float:
        """
        Convert world X coordinate to normalized action.
        
        Args:
            x: World X coordinate.
            fruit_type: The fruit type.
            
        Returns:
            Normalized action in [-1, 1].
        """
        min_x, max_x = self.get_spawn_x_range(fruit_type)
        
        # Map [min_x, max_x] to [-1, 1]
        t = (x - min_x) / (max_x - min_x)  # [0, 1]
        action = t * 2.0 - 1.0  # [-1, 1]
        return max(-1.0, min(1.0, action))
    
    @property
    def spawn_y(self) -> float:
        """Y coordinate for spawning."""
        return self._spawn_y


class TerminationRules:
    """
    Handles game termination conditions.
    
    - Lose line: Fruits above line for > grace time
    - Drop cap: Maximum drops per episode
    - Out of bounds: Fruit ejected beyond safe distance from board
    """
    
    def __init__(self, config: Optional[GameConfig] = None):
        """
        Initialize termination rules.
        
        Args:
            config: Game configuration. Uses default if None.
        """
        if config is None:
            config = get_config()
        
        self._config = config
        self._lose_line_y = config.board.lose_line_y
        self._grace_time = config.board.lose_line_grace_time
        self._max_drops = config.caps.max_drops
        self._max_objects = config.caps.max_objects
        self._out_of_bounds_distance = config.caps.out_of_bounds_distance
        
        # Track time above lose line per fruit
        self._above_line_time: float = 0.0
        self._was_above_line: bool = False
    
    @property
    def lose_line_y(self) -> float:
        """Y coordinate of the lose line."""
        return self._lose_line_y
    
    @property
    def max_drops(self) -> int:
        """Maximum drops per episode."""
        return self._max_drops
    
    @property
    def out_of_bounds_distance(self) -> float:
        """Distance beyond board edge that triggers out-of-bounds termination."""
        return self._out_of_bounds_distance
    
    def reset(self) -> None:
        """Reset termination state."""
        self._above_line_time = 0.0
        self._was_above_line = False
    
    def update_lose_line_timer(
        self,
        any_above_line: bool,
        dt: float
    ) -> None:
        """
        Update the lose line timer.
        
        Args:
            any_above_line: True if any fruit is above the lose line.
            dt: Time delta since last update.
        """
        if any_above_line:
            self._above_line_time += dt
            self._was_above_line = True
        else:
            # Reset timer when no fruits above line
            self._above_line_time = 0.0
            self._was_above_line = False
    
    def check_termination(
        self,
        drops_used: int,
        fruit_count: int,
        any_above_line: bool,
        any_out_of_bounds: bool = False
    ) -> TerminationResult:
        """
        Check all termination conditions.
        
        Args:
            drops_used: Number of fruits dropped so far.
            fruit_count: Current number of fruits on board.
            any_above_line: True if any fruit is above lose line.
            any_out_of_bounds: True if any fruit is beyond the safe boundary.
            
        Returns:
            TerminationResult indicating game state.
        """
        # Check out of bounds (immediate termination for physics glitches)
        if any_out_of_bounds:
            return TerminationResult.game_over("out_of_bounds")
        
        # Check lose line (1-second rule)
        if any_above_line and self._above_line_time >= self._grace_time:
            return TerminationResult.game_over("lose_line")
        
        # Check drop cap
        if drops_used >= self._max_drops:
            return TerminationResult.truncation("drop_cap")
        
        # Check object cap (safety)
        if fruit_count >= self._max_objects:
            return TerminationResult.truncation("object_cap")
        
        return TerminationResult.none()
    
    @property
    def above_line_time(self) -> float:
        """Current accumulated time with fruits above lose line."""
        return self._above_line_time


class GameRules:
    """
    Combined interface for all game rules.
    """
    
    def __init__(self, config: Optional[GameConfig] = None):
        """
        Initialize game rules.
        
        Args:
            config: Game configuration. Uses default if None.
        """
        if config is None:
            config = get_config()
        
        self._config = config
        self.spawn = SpawnRules(config)
        self.termination = TerminationRules(config)
    
    def reset(self) -> None:
        """Reset all rule state."""
        self.termination.reset()
