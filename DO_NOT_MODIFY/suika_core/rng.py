"""
RNG - Weighted Shuffle-Bag Queue
================================

Provides deterministic fruit spawning with reduced variance through
a weighted shuffle-bag mechanism.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from DO_NOT_MODIFY.suika_core.config_loader import GameConfig, get_config


class SpawnQueue:
    """
    Weighted shuffle-bag queue for fruit spawning.
    
    The bag contains a weighted distribution of spawnable fruit IDs.
    When the bag is exhausted, it refills and reshuffles.
    
    This reduces variance compared to pure random while maintaining variability.
    """
    
    def __init__(
        self,
        config: Optional[GameConfig] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize spawn queue.
        
        Args:
            config: Game configuration. Uses default if None.
            seed: Random seed for reproducibility. Random if None.
        """
        if config is None:
            config = get_config()
        
        self._config = config
        self._rng = random.Random(seed)
        
        # Build bag template from weights
        self._bag_template: List[int] = []
        for fruit_id, weight in enumerate(config.rng.weights):
            self._bag_template.extend([fruit_id] * weight)
        
        # Adjust to target bag size
        target_size = config.rng.bag_size
        if len(self._bag_template) < target_size:
            # Pad with weighted random selections
            while len(self._bag_template) < target_size:
                fruit_id = self._weighted_choice()
                self._bag_template.append(fruit_id)
        elif len(self._bag_template) > target_size:
            # Truncate (shouldn't happen with good config)
            self._bag_template = self._bag_template[:target_size]
        
        # Current bag and position
        self._bag: List[int] = []
        self._index: int = 0
        
        # Refill and shuffle initial bag
        self._refill_bag()
        
        # Cache for peeking
        self._current: int = self._bag[0]
        self._next: int = self._bag[1] if len(self._bag) > 1 else self._bag[0]
    
    def _weighted_choice(self) -> int:
        """Choose a random fruit ID weighted by config weights."""
        weights = self._config.rng.weights
        total = sum(weights)
        r = self._rng.random() * total
        cumulative = 0
        for fruit_id, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return fruit_id
        return len(weights) - 1
    
    def _refill_bag(self) -> None:
        """Refill and shuffle the bag."""
        self._bag = self._bag_template.copy()
        self._rng.shuffle(self._bag)
        self._index = 0
    
    @property
    def current_fruit_id(self) -> int:
        """ID of the fruit that will drop next."""
        return self._current
    
    @property
    def next_fruit_id(self) -> int:
        """ID of the fruit after the current one."""
        return self._next
    
    def advance(self) -> int:
        """
        Advance the queue and return the fruit ID that was dropped.
        
        Returns:
            The fruit ID that was current (now consumed).
        """
        consumed = self._current
        
        self._index += 1
        if self._index >= len(self._bag):
            self._refill_bag()
        
        # Update current and next
        self._current = self._bag[self._index]
        next_index = self._index + 1
        if next_index >= len(self._bag):
            # Peek into what the next bag's first item would be
            # For simplicity, just use weighted choice
            self._next = self._weighted_choice()
        else:
            self._next = self._bag[next_index]
        
        return consumed
    
    def peek(self, count: int = 2) -> List[int]:
        """
        Peek at upcoming fruit IDs without consuming.
        
        Args:
            count: Number of upcoming fruits to peek.
            
        Returns:
            List of upcoming fruit IDs.
        """
        result = []
        temp_index = self._index
        temp_bag = self._bag.copy()
        
        for _ in range(count):
            if temp_index >= len(temp_bag):
                # Would refill - use template shuffled
                temp_bag = self._bag_template.copy()
                self._rng.shuffle(temp_bag)
                temp_index = 0
            result.append(temp_bag[temp_index])
            temp_index += 1
        
        return result
    
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the queue with optional new seed.
        
        Args:
            seed: New random seed. Keeps current if None.
        """
        if seed is not None:
            self._rng = random.Random(seed)
        self._refill_bag()
        self._current = self._bag[0]
        self._next = self._bag[1] if len(self._bag) > 1 else self._bag[0]
    
    def get_state(self) -> Tuple[List[int], int, int]:
        """
        Get serializable state for replay/checkpointing.
        
        Returns:
            Tuple of (bag, index, rng_state_hash).
        """
        return (self._bag.copy(), self._index, hash(self._rng.getstate()))
    
    def set_state(self, bag: List[int], index: int) -> None:
        """
        Restore queue state.
        
        Args:
            bag: The bag contents.
            index: Current index in bag.
        """
        self._bag = bag.copy()
        self._index = index
        self._current = self._bag[self._index]
        next_index = self._index + 1
        if next_index < len(self._bag):
            self._next = self._bag[next_index]
        else:
            self._next = self._weighted_choice()
