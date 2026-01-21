"""
Core Game
=========

Main game orchestrator combining physics, merging, scoring, and rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from DO_NOT_MODIFY.suika_core.config_loader import GameConfig, get_config
from DO_NOT_MODIFY.suika_core.fruit_catalog import FruitCatalog, get_catalog
from DO_NOT_MODIFY.suika_core.physics_world import PhysicsWorld
from DO_NOT_MODIFY.suika_core.merge_system import MergeSystem, MergeResult
from DO_NOT_MODIFY.suika_core.rng import SpawnQueue
from DO_NOT_MODIFY.suika_core.scoring import ScoreTracker
from DO_NOT_MODIFY.suika_core.rules import GameRules, TerminationResult
from DO_NOT_MODIFY.suika_core.state_snapshot import SnapshotBuilder, GameSnapshot


@dataclass
class StepResult:
    """Result of a single game step (drop + settle)."""
    snapshot: GameSnapshot
    terminated: bool
    truncated: bool
    termination_reason: str
    delta_score: int
    merges: List[MergeResult]
    sim_time: float


class CoreGame:
    """
    Main game simulation class.
    
    Orchestrates:
    - Physics world
    - Merge detection and resolution
    - Spawn queue (RNG)
    - Scoring
    - Termination rules
    - State snapshots
    
    One step = one drop, then simulate until settled.
    """
    
    def __init__(
        self,
        config: Optional[GameConfig] = None,
        seed: Optional[int] = None,
        render_callback: Optional[callable] = None
    ):
        """
        Initialize game.
        
        Args:
            config: Game configuration. Uses default if None.
            seed: Random seed for reproducibility.
            render_callback: Optional callback for rendering during settle.
        """
        if config is None:
            config = get_config()
        
        self._config = config
        self._seed = seed
        self._render_callback = render_callback
        
        # Initialize subsystems
        self._catalog = get_catalog(config)
        self._physics = PhysicsWorld(config)
        self._scorer = ScoreTracker(config)
        self._merger = MergeSystem(
            physics=self._physics,
            scorer=self._scorer,
            config=config,
            rng_seed=seed
        )
        self._spawn_queue = SpawnQueue(config, seed)
        self._rules = GameRules(config)
        self._snapshot_builder = SnapshotBuilder(config)
        
        # Game state
        self._drops_used: int = 0
        self._last_spawner_x: float = 0.0
        self._terminated: bool = False
        self._truncated: bool = False
        self._termination_reason: str = ""
    
    @property
    def config(self) -> GameConfig:
        """Game configuration."""
        return self._config
    
    @property
    def physics(self) -> PhysicsWorld:
        """Physics world instance."""
        return self._physics
    
    @property
    def catalog(self) -> FruitCatalog:
        """Fruit catalog."""
        return self._catalog
    
    @property
    def score(self) -> int:
        """Current score."""
        return self._scorer.score
    
    @property
    def drops_used(self) -> int:
        """Number of fruits dropped."""
        return self._drops_used
    
    @property
    def current_fruit_id(self) -> int:
        """ID of the next fruit to drop."""
        return self._spawn_queue.current_fruit_id
    
    @property
    def next_fruit_id(self) -> int:
        """ID of the fruit after the current one."""
        return self._spawn_queue.next_fruit_id
    
    @property
    def is_over(self) -> bool:
        """True if game has ended."""
        return self._terminated or self._truncated
    
    @property
    def termination_reason(self) -> str:
        """Reason for game end, or empty string."""
        return self._termination_reason
    
    @property
    def fruit_count(self) -> int:
        """Number of fruits currently on board."""
        return self._physics.fruit_count
    
    @property
    def spawn_queue(self) -> SpawnQueue:
        """The spawn queue (for preview access)."""
        return self._spawn_queue
    
    @property
    def skull_count(self) -> int:
        """Number of skulls currently on board."""
        return self._count_skulls()
    
    @property
    def skull_multiplier(self) -> float:
        """Current score multiplier from skulls (1.0 + 0.5 * skull_count)."""
        return self._scorer.get_skull_multiplier()
    
    def _count_skulls(self) -> int:
        """Count number of skull fruits currently on the board."""
        skull_id = self._catalog.skull_id
        count = 0
        for fruit in self._physics.fruits.values():
            if fruit.type_id == skull_id:
                count += 1
        return count
    
    def _update_skull_count(self) -> None:
        """Update scorer's skull count from current physics state."""
        self._scorer.skull_count = self._count_skulls()
    
    def reset(self, seed: Optional[int] = None) -> GameSnapshot:
        """
        Reset game to initial state.
        
        Args:
            seed: New random seed. Uses previous if None.
            
        Returns:
            Initial game snapshot.
        """
        if seed is not None:
            self._seed = seed
        
        # Clear physics
        self._physics.clear()
        
        # Apply domain randomization (uses seed for reproducibility)
        self._physics.apply_domain_randomization(self._seed)
        
        # Reset subsystems
        self._scorer.reset()
        self._spawn_queue.reset(self._seed)
        self._rules.reset()
        
        # Reset state
        self._drops_used = 0
        self._last_spawner_x = 0.0
        self._terminated = False
        self._truncated = False
        self._termination_reason = ""
        
        # Return initial snapshot
        return self._build_snapshot()
    
    def step(self, action: float) -> StepResult:
        """
        Execute one game step: drop fruit, simulate until settled.
        
        Args:
            action: Spawner X position in [-1, 1].
            
        Returns:
            StepResult with new state and metadata.
        """
        if self.is_over:
            # Game already ended, return current state
            return StepResult(
                snapshot=self._build_snapshot(),
                terminated=self._terminated,
                truncated=self._truncated,
                termination_reason=self._termination_reason,
                delta_score=0,
                merges=[],
                sim_time=0.0
            )
        
        score_before = self._scorer.score
        
        # Get current fruit type
        fruit_id = self._spawn_queue.advance()
        fruit_type = self._catalog[fruit_id]
        
        # Convert action to world coordinates
        spawn_x = self._rules.spawn.action_to_spawn_x(action, fruit_type)
        spawn_y = self._rules.spawn.spawn_y
        
        # Store for snapshot
        self._last_spawner_x = action
        
        # Spawn fruit
        self._physics.spawn_fruit(fruit_type, spawn_x, spawn_y)
        self._drops_used += 1
        
        # Simulate until settled
        all_merges: List[MergeResult] = []
        sim_time = self._simulate_until_settled(all_merges)
        
        # Calculate delta score
        delta_score = self._scorer.score - score_before
        
        # Check termination
        term_result = self._check_termination()
        self._terminated = term_result.terminated
        self._truncated = term_result.truncated
        self._termination_reason = term_result.reason
        
        # Build snapshot
        snapshot = self._build_snapshot()
        
        return StepResult(
            snapshot=snapshot,
            terminated=self._terminated,
            truncated=self._truncated,
            termination_reason=self._termination_reason,
            delta_score=delta_score,
            merges=all_merges,
            sim_time=sim_time
        )
    
    def _simulate_until_settled(
        self,
        all_merges: List[MergeResult]
    ) -> float:
        """
        Simulate physics until all objects settle.
        
        Args:
            all_merges: List to append merge results to.
            
        Returns:
            Total simulation time.
        """
        dt = self._config.physics.dt
        max_time = self._config.physics.max_sim_seconds_per_drop
        settle_ticks_needed = self._config.physics.settle_consecutive_ticks
        merge_resets_timer = self._config.physics.merge_resets_timer
        
        sim_time = 0.0
        time_since_last_merge = 0.0  # Track time since last merge
        consecutive_settled = 0
        
        while True:
            # Check timeout - use time since last merge if resets are enabled
            effective_time = time_since_last_merge if merge_resets_timer else sim_time
            if effective_time >= max_time:
                break
            
            # Step physics
            self._physics.step(dt)
            sim_time += dt
            time_since_last_merge += dt
            
            # Resolve merges
            merges = self._merger.resolve_merges()
            if merges:
                all_merges.extend(merges)
                # Update skull count for scoring multiplier
                self._update_skull_count()
                # Reset timer on merge - more time for chain reactions
                if merge_resets_timer:
                    time_since_last_merge = 0.0
                consecutive_settled = 0  # Reset settle counter too
            
            # Update lose line timer
            fruits_above = self._physics.get_fruits_above_line(
                self._rules.termination.lose_line_y
            )
            self._rules.termination.update_lose_line_timer(
                any_above_line=len(fruits_above) > 0,
                dt=dt
            )
            
            # Check if settled
            if self._physics.check_settled():
                consecutive_settled += 1
                if consecutive_settled >= settle_ticks_needed:
                    break
            else:
                consecutive_settled = 0
            
            # Optional render callback
            if self._render_callback is not None:
                self._render_callback()
        
        # Force freeze if we hit the time cap
        if time_since_last_merge >= max_time or (not merge_resets_timer and sim_time >= max_time):
            self._physics.force_freeze()
        
        return sim_time
    
    def _check_termination(self) -> TerminationResult:
        """Check all termination conditions."""
        fruits_above = self._physics.get_fruits_above_line(
            self._rules.termination.lose_line_y
        )
        
        fruits_out_of_bounds = self._physics.get_fruits_out_of_bounds(
            self._rules.termination.out_of_bounds_distance
        )
        
        return self._rules.termination.check_termination(
            drops_used=self._drops_used,
            fruit_count=self._physics.fruit_count,
            any_above_line=len(fruits_above) > 0,
            any_out_of_bounds=len(fruits_out_of_bounds) > 0
        )
    
    def _build_snapshot(
        self,
        board_rgb: Optional[np.ndarray] = None
    ) -> GameSnapshot:
        """Build current game state snapshot."""
        return self._snapshot_builder.build(
            physics=self._physics,
            spawner_x=self._last_spawner_x,
            current_fruit_id=self.current_fruit_id,
            next_fruit_id=self.next_fruit_id,
            score=self._scorer.score,
            drops_used=self._drops_used,
            board_rgb=board_rgb
        )
    
    def get_info(self) -> Dict[str, Any]:
        """Get additional info dict for Gymnasium."""
        return {
            "score": self._scorer.score,
            "drops_used": self._drops_used,
            "fruit_count": self._physics.fruit_count,
            "merges": self._scorer.merges,
            "terminated_reason": self._termination_reason,
            "above_line_time": self._rules.termination.above_line_time,
        }
    
    def get_render_data(self) -> Dict[str, Any]:
        """
        Get data needed for rendering.
        
        Returns:
            Dict with positions, types, and board info.
        """
        fruits_data = []
        for fruit in self._physics.fruits.values():
            fruits_data.append({
                "uid": fruit.uid,
                "type_id": fruit.type_id,
                "x": fruit.position[0],
                "y": fruit.position[1],
                "angle": fruit.angle,
                "visual_radius": fruit.fruit_type.visual_radius,
                "color_solid": fruit.fruit_type.color_solid,
                "color_full": fruit.fruit_type.color_full,
                "collision_circles": [
                    (c.radius, c.offset_x, c.offset_y)
                    for c in fruit.fruit_type.collision_circles
                ]
            })
        
        return {
            "board_width": self._physics.board_width,
            "board_height": self._physics.board_height,
            "lose_line_y": self._rules.termination.lose_line_y,
            "spawn_y": self._rules.spawn.spawn_y,
            "fruits": fruits_data,
            "score": self._scorer.score,
            "drops_used": self._drops_used,
            "current_fruit_id": self.current_fruit_id,
            "next_fruit_id": self.next_fruit_id,
            "spawner_x_norm": self._last_spawner_x,
            "skull_count": self.skull_count,
            "skull_multiplier": self.skull_multiplier,
        }
    
    def spawn_current_fruit(self, action: float) -> bool:
        """
        Spawn the current fruit at the given position (for animated mode).
        
        This only spawns - caller must call tick_physics() repeatedly until settled.
        
        Args:
            action: Spawner X position in [-1, 1].
            
        Returns:
            True if fruit was spawned, False if game is over.
        """
        if self.is_over:
            return False
        
        fruit_id = self._spawn_queue.advance()
        fruit_type = self._catalog[fruit_id]
        
        spawn_x = self._rules.spawn.action_to_spawn_x(action, fruit_type)
        spawn_y = self._rules.spawn.spawn_y
        
        self._last_spawner_x = action
        self._physics.spawn_fruit(fruit_type, spawn_x, spawn_y)
        self._drops_used += 1
        
        return True
    
    def tick_physics(self) -> Tuple[bool, int, List[MergeResult]]:
        """
        Advance physics by one timestep (for animated mode).
        
        Returns:
            Tuple of (is_settled, delta_score, merges_this_tick).
        """
        dt = self._config.physics.dt
        score_before = self._scorer.score
        
        # Step physics
        self._physics.step(dt)
        
        # Resolve merges
        merges = self._merger.resolve_merges()
        
        # Update lose line timer
        fruits_above = self._physics.get_fruits_above_line(
            self._rules.termination.lose_line_y
        )
        self._rules.termination.update_lose_line_timer(
            any_above_line=len(fruits_above) > 0,
            dt=dt
        )
        
        delta_score = self._scorer.score - score_before
        is_settled = self._physics.check_settled()
        
        return is_settled, delta_score, merges
    
    def check_and_update_termination(self) -> Tuple[bool, bool, str]:
        """
        Check termination conditions (for animated mode).
        
        Returns:
            Tuple of (terminated, truncated, reason).
        """
        term_result = self._check_termination()
        self._terminated = term_result.terminated
        self._truncated = term_result.truncated
        self._termination_reason = term_result.reason
        return self._terminated, self._truncated, self._termination_reason
    
    def force_settle(self) -> None:
        """Force all objects to settle immediately."""
        self._physics.force_freeze()
