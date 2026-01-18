"""
Team Template Agent
===================

Your agent must provide one of:
1. A `SuikaAgent` class with an `act(obs) -> action` method
2. A standalone `act(obs) -> action` function

Actions should be floats in [-1, 1] representing spawner X position.

See API_REFERENCE.md for full observation space documentation.
"""

from __future__ import annotations

from typing import Dict
import numpy as np


class SuikaAgent:
    """
    Your Suika agent implementation.
    
    Replace the strategy in `act()` with your own logic.
    """
    
    def __init__(self):
        """Initialize your agent. Load models, set up state, etc."""
        self.rng = np.random.default_rng()
    
    def act(self, obs: Dict[str, np.ndarray]) -> float:
        """
        Choose an action based on the observation.
        
        Args:
            obs: Dictionary containing game state.
                 See API_REFERENCE.md for full documentation.
                
        Returns:
            action: Float in [-1, 1] for spawner X position.
        """
        # TODO: Replace with your strategy
        return float(self.rng.uniform(-0.5, 0.5))
    
    def reset(self) -> None:
        """Called when a new episode starts (optional)."""
        pass


def act(obs: Dict[str, np.ndarray]) -> float:
    """Standalone act function (alternative to class-based agent)."""
    return float(np.random.uniform(-0.5, 0.5))
