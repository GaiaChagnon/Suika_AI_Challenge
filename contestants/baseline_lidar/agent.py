"""
Baseline LIDAR Agent - Drops fruits at the lowest column.

This is a simple heuristic agent that uses the height_map observation
(a 1D "LIDAR" scan of the board surface) to find the column with the
lowest height and drops fruits there.

This serves as:
1. A working example of how to read observations and return actions
2. A baseline benchmark for teams to compare against
3. A verification that the environment API works correctly

Strategy:
- Read the height_map (20 values showing stack height at each column)
- Check danger_level to see how risky the situation is
- Find the column with the minimum height
- Convert that column index to an action in [-1, 1]
- Add small random noise to avoid always hitting the exact same spot
"""

import numpy as np
from typing import Any, Dict, Optional


# Height map has 20 columns
NUM_COLUMNS = 20
# Board width is read from observation, not hardcoded


class SuikaAgent:
    """
    Simple baseline agent that drops fruits at the lowest column.
    
    Uses the height_map observation to find gaps and low spots,
    then places fruits there to build a stable base.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize the agent.
        
        Args:
            debug: If True, print decisions to stdout.
        """
        self.debug = debug
        self._rng = np.random.default_rng()
    
    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset agent state for a new episode.
        
        Args:
            seed: Optional random seed for reproducibility.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
    
    def act(self, observation: Dict[str, Any], debug: bool = False) -> float:
        """
        Choose where to drop the fruit based on height map and danger level.
        
        Strategy:
        1. Read the height_map (20 columns)
        2. Check danger_level to understand urgency
        3. Find the column with minimum height
        4. Convert to action in [-1, 1]
        5. Add small noise to avoid repetitive patterns
        
        Args:
            observation: Dict of numpy arrays from the environment.
            debug: If True, print debug info for this step.
            
        Returns:
            Action in [-1, 1] representing drop position.
        """
        # Read the pre-computed height map (1D "LIDAR" scan of the surface)
        height_map = observation["height_map"]  # Shape: (20,)
        
        # Read danger level (0.0 = safe, 1.0 = critical)
        danger_level = float(observation["danger_level"])
        
        # Find the column with the lowest height
        lowest_column = int(np.argmin(height_map))
        
        # Convert column index [0, 19] to action [-1, 1]
        action = (lowest_column / (NUM_COLUMNS - 1)) * 2.0 - 1.0
        
        # Add small random noise to avoid always hitting exact same spot
        # Use less noise when danger is high (be more precise)
        noise_scale = 0.05 * (1.0 - danger_level * 0.5)
        noise = self._rng.uniform(-noise_scale, noise_scale)
        action = np.clip(action + noise, -1.0, 1.0)
        
        # Debug output
        if debug or self.debug:
            current_fruit = int(observation["current_fruit_id"])
            min_height = float(height_map[lowest_column])
            max_height = float(height_map.max())
            
            print(f"[LIDAR Agent] Fruit={current_fruit}, "
                  f"Danger={danger_level:.2f}, "
                  f"Target col={lowest_column}/{NUM_COLUMNS-1}, "
                  f"Heights: min={min_height:.0f} max={max_height:.0f}, "
                  f"Action={action:.3f}")
        
        return float(action)


# Convenience function to create agent (used by evaluation harness)
def create_agent(**kwargs) -> SuikaAgent:
    """Factory function to create an agent instance."""
    return SuikaAgent(**kwargs)
