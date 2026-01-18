"""
Merge System
============

Handles fruit collision detection, merge queue management, and merge resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import random

import pymunk

from DO_NOT_MODIFY.suika_core.config_loader import GameConfig, get_config
from DO_NOT_MODIFY.suika_core.fruit_catalog import FruitCatalog, get_catalog
from DO_NOT_MODIFY.suika_core.scoring import ScoreTracker, ScoreEvent

if TYPE_CHECKING:
    from DO_NOT_MODIFY.suika_core.physics_world import PhysicsWorld, FruitBody


@dataclass
class MergeCandidate:
    """A pair of fruits that may merge."""
    uid_a: int
    uid_b: int
    contact_point: Tuple[float, float]
    collision_normal: Tuple[float, float]  # Normal direction of collision
    relative_velocity: Tuple[float, float]  # Velocity difference at collision
    
    @property
    def sorted_pair(self) -> Tuple[int, int]:
        """Canonical ordering for deduplication."""
        return (min(self.uid_a, self.uid_b), max(self.uid_a, self.uid_b))


@dataclass
class MergeResult:
    """Result of a single merge operation."""
    removed_uids: Tuple[int, int]
    created_uid: Optional[int]  # None for melon-melon
    new_type_id: Optional[int]  # None for melon-melon
    position: Tuple[float, float]
    score_event: ScoreEvent


class MergeSystem:
    """
    Manages collision-based fruit merging.
    
    Collision callbacks enqueue merge candidates. After physics steps,
    the queue is resolved with deduplication and deterministic ordering.
    """
    
    def __init__(
        self,
        physics: "PhysicsWorld",
        scorer: ScoreTracker,
        config: Optional[GameConfig] = None,
        rng_seed: Optional[int] = None
    ):
        """
        Initialize merge system.
        
        Args:
            physics: The physics world instance.
            scorer: Score tracker instance.
            config: Game configuration. Uses default if None.
            rng_seed: Seed for merge impulse variance.
        """
        if config is None:
            config = get_config()
        
        self._config = config
        self._physics = physics
        self._scorer = scorer
        self._catalog = get_catalog(config)
        self._rng = random.Random(rng_seed)
        
        # Merge queue
        self._queue: List[MergeCandidate] = []
        
        # Track which fruits merged this tick (for dedup)
        self._merged_this_tick: Set[int] = set()
        
        # Register collision handler
        physics.set_collision_handler(self._on_collision)
    
    def _on_collision(
        self,
        arbiter: pymunk.Arbiter,
        space: pymunk.Space,
        data: any
    ) -> None:
        """
        Pymunk 7.x collision callback for fruit-fruit collisions.
        
        Enqueues merge candidates if fruits are same type.
        
        Args:
            arbiter: Collision arbiter.
            space: Physics space.
            data: User data (unused).
        """
        # Get bodies from arbiter shapes
        shape_a, shape_b = arbiter.shapes
        body_a = shape_a.body
        body_b = shape_b.body
        
        # Get fruit bodies
        fruit_a = self._physics.get_fruit_by_body(body_a)
        fruit_b = self._physics.get_fruit_by_body(body_b)
        
        if fruit_a is None or fruit_b is None:
            return
        
        # Check if same type (can merge)
        if fruit_a.type_id != fruit_b.type_id:
            return
        
        # Final fruits (like skull) cannot merge
        if self._catalog.is_final_fruit(fruit_a.type_id):
            return
        
        # Get contact point and collision normal
        contact_set = arbiter.contact_point_set
        if len(contact_set.points) > 0:
            point = contact_set.points[0].point_a
            contact_point = (point.x, point.y)
            # Get collision normal (from a to b)
            normal = contact_set.normal
            collision_normal = (normal.x, normal.y)
        else:
            # Fallback to midpoint and computed normal
            pos_a = fruit_a.position
            pos_b = fruit_b.position
            contact_point = ((pos_a[0] + pos_b[0]) / 2, (pos_a[1] + pos_b[1]) / 2)
            # Compute normal from a to b
            dx = pos_b[0] - pos_a[0]
            dy = pos_b[1] - pos_a[1]
            dist = max(0.001, (dx*dx + dy*dy) ** 0.5)
            collision_normal = (dx / dist, dy / dist)
        
        # Get relative velocity at collision
        vel_a = fruit_a.velocity
        vel_b = fruit_b.velocity
        relative_velocity = (vel_b[0] - vel_a[0], vel_b[1] - vel_a[1])
        
        # Enqueue merge candidate
        candidate = MergeCandidate(
            uid_a=fruit_a.uid,
            uid_b=fruit_b.uid,
            contact_point=contact_point,
            collision_normal=collision_normal,
            relative_velocity=relative_velocity
        )
        self._queue.append(candidate)
    
    def resolve_merges(self) -> List[MergeResult]:
        """
        Process all queued merge candidates.
        
        Deduplicates, ensures each fruit merges at most once,
        and uses deterministic ordering.
        
        Returns:
            List of merge results.
        """
        if not self._queue:
            return []
        
        # Deduplicate by sorted pair
        seen_pairs: Set[Tuple[int, int]] = set()
        unique_candidates: List[MergeCandidate] = []
        
        for candidate in self._queue:
            pair = candidate.sorted_pair
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_candidates.append(candidate)
        
        # Clear queue
        self._queue.clear()
        
        # Sort by first UID for deterministic ordering
        unique_candidates.sort(key=lambda c: c.sorted_pair)
        
        # Track fruits already merged
        self._merged_this_tick.clear()
        results: List[MergeResult] = []
        
        for candidate in unique_candidates:
            # Skip if either fruit already merged
            if candidate.uid_a in self._merged_this_tick:
                continue
            if candidate.uid_b in self._merged_this_tick:
                continue
            
            # Verify both fruits still exist and are same type
            fruit_a = self._physics.get_fruit(candidate.uid_a)
            fruit_b = self._physics.get_fruit(candidate.uid_b)
            
            if fruit_a is None or fruit_b is None:
                continue
            
            if fruit_a.type_id != fruit_b.type_id:
                continue
            
            # Perform merge with collision info
            result = self._do_merge(
                fruit_a, fruit_b, candidate.contact_point,
                candidate.collision_normal, candidate.relative_velocity
            )
            if result is not None:
                results.append(result)
                self._merged_this_tick.add(candidate.uid_a)
                self._merged_this_tick.add(candidate.uid_b)
        
        return results
    
    def _do_merge(
        self,
        fruit_a: "FruitBody",
        fruit_b: "FruitBody",
        contact_point: Tuple[float, float],
        collision_normal: Tuple[float, float],
        relative_velocity: Tuple[float, float]
    ) -> Optional[MergeResult]:
        """
        Execute a single merge operation.
        
        The merge impulse direction is influenced by the collision:
        - Faster collisions produce bigger jumps
        - Side collisions add horizontal momentum
        - Upward collisions boost vertical impulse
        
        Args:
            fruit_a: First fruit.
            fruit_b: Second fruit.
            contact_point: Where fruits collided.
            collision_normal: Normal vector of collision (unit vector).
            relative_velocity: Relative velocity at collision.
            
        Returns:
            MergeResult if merge succeeded.
        """
        import math
        
        type_id = fruit_a.type_id
        pos_a = fruit_a.position
        pos_b = fruit_b.position
        
        # Compute merge position (average of centers)
        merge_x = (pos_a[0] + pos_b[0]) / 2
        merge_y = (pos_a[1] + pos_b[1]) / 2
        
        # Remove both fruits
        self._physics.remove_fruit(fruit_a.uid)
        self._physics.remove_fruit(fruit_b.uid)
        
        # Apply score (will be multiplied by skull count if applicable)
        score_event = self._scorer.apply_merge(type_id)
        
        # Watermelons merge into a skull (the final form)
        if self._catalog.is_watermelon(type_id):
            # Create a skull instead of removing both
            skull_type = self._catalog.skull
            if skull_type is not None:
                new_uid = self._physics.spawn_fruit(
                    skull_type,
                    merge_x,
                    merge_y,
                    velocity=(0.0, 50.0)  # Small upward pop
                )
                return MergeResult(
                    removed_uids=(fruit_a.uid, fruit_b.uid),
                    created_uid=new_uid,
                    new_type_id=skull_type.id,
                    position=(merge_x, merge_y),
                    score_event=score_event
                )
            else:
                # No skull defined, just remove both
                return MergeResult(
                    removed_uids=(fruit_a.uid, fruit_b.uid),
                    created_uid=None,
                    new_type_id=None,
                    position=(merge_x, merge_y),
                    score_event=score_event
                )
        
        # Spawn next fruit type
        next_type = self._catalog.get_next_type(type_id)
        if next_type is None:
            return None
        
        # Calculate collision-influenced impulse
        # The merged fruit moves OPPOSITE to where it was hit
        impulse_base = self._config.merge.impulse_upward
        impulse_var = self._config.merge.impulse_variance
        
        # Get collision normal - points from A to B at contact
        nx, ny = collision_normal
        
        # Determine which fruit was moving faster (the "striker")
        vel_a = fruit_a.velocity
        vel_b = fruit_b.velocity
        speed_a = math.sqrt(vel_a[0]**2 + vel_a[1]**2)
        speed_b = math.sqrt(vel_b[0]**2 + vel_b[1]**2)
        
        # Calculate collision speed (how fast they hit each other)
        collision_speed = math.sqrt(
            relative_velocity[0]**2 + relative_velocity[1]**2
        )
        
        # The impulse direction is based on which fruit was the "striker"
        # If fruit A was faster, push in the direction of the normal (A->B)
        # If fruit B was faster, push opposite to the normal (B->A)
        if speed_a > speed_b:
            # A was the striker, push in normal direction (away from A)
            push_x = nx
            push_y = ny
        else:
            # B was the striker, push opposite to normal (away from B)
            push_x = -nx
            push_y = -ny
        
        # Scale impulse by collision energy - more speed = more dramatic effect
        # Increased multipliers for more noticeable movement
        speed_factor = min(collision_speed * 0.5, impulse_base * 1.5)
        
        # Base upward impulse with variance
        base_up = impulse_base + self._rng.uniform(-impulse_var, impulse_var)
        
        # Strong directional impulse based on collision
        # Horizontal: mostly from impact direction
        impulse_x = push_x * speed_factor * 1.2
        
        # Vertical: combine upward base + impact direction
        # Always add some upward component for the "pop" effect
        impulse_y = base_up * 0.6 + push_y * speed_factor * 0.8
        
        # Ensure some minimum upward motion
        impulse_y = max(impulse_y, impulse_base * 0.3)
        
        # Spawn new fruit
        new_fruit = self._physics.spawn_fruit(
            fruit_type=next_type,
            x=merge_x,
            y=merge_y,
            velocity=(0, 0)
        )
        
        # Apply collision-influenced impulse
        self._physics.apply_impulse(
            new_fruit.uid,
            (impulse_x * next_type.mass, impulse_y * next_type.mass)
        )
        
        # Apply radial impulse to nearby fruits for "pop" effect
        self._apply_radial_impulse(merge_x, merge_y, next_type.visual_radius * 2.5, new_fruit.uid)
        
        return MergeResult(
            removed_uids=(fruit_a.uid, fruit_b.uid),
            created_uid=new_fruit.uid,
            new_type_id=next_type.id,
            position=(merge_x, merge_y),
            score_event=score_event
        )
    
    def _apply_radial_impulse(
        self,
        center_x: float,
        center_y: float,
        radius: float,
        exclude_uid: int
    ) -> None:
        """
        Apply outward radial impulse to fruits near a merge point.
        
        Makes nearby fruits "pop" away from the merge, creating a
        more dynamic and interactive feel.
        
        Args:
            center_x: X coordinate of merge center.
            center_y: Y coordinate of merge center.
            radius: Radius within which to affect fruits.
            exclude_uid: UID of newly created fruit to exclude.
        """
        import math
        
        radial_impulse = getattr(self._config.merge, 'radial_impulse', 150.0)
        
        for uid, fruit in self._physics.fruits.items():
            if uid == exclude_uid:
                continue
            
            fx, fy = fruit.position
            dx = fx - center_x
            dy = fy - center_y
            dist = math.sqrt(dx * dx + dy * dy)
            
            if dist < 0.1:  # Avoid division by zero
                continue
            
            if dist < radius:
                # Strength falls off with distance
                strength = radial_impulse * (1.0 - dist / radius)
                # Lighter fruits get pushed more (inverse mass scaling)
                # With mass range 0.5 (cherry) to 85 (melon), scale appropriately
                mass_factor = 5.0 / max(0.5, fruit.fruit_type.mass)
                mass_factor = min(mass_factor, 8.0)  # Cap for tiny fruits
                
                # Normalize direction and apply impulse
                impulse_x = (dx / dist) * strength * mass_factor
                impulse_y = (dy / dist) * strength * mass_factor + strength * 0.3  # Slight upward bias
                
                self._physics.apply_impulse(uid, (impulse_x, impulse_y))
    
    def clear_queue(self) -> None:
        """Clear pending merge candidates."""
        self._queue.clear()
        self._merged_this_tick.clear()
    
    @property
    def pending_count(self) -> int:
        """Number of pending merge candidates."""
        return len(self._queue)
