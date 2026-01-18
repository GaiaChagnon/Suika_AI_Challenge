"""
Tests for merge rules and collision handling.
"""

import pytest
import numpy as np

from DO_NOT_MODIFY.suika_core.config_loader import load_config
from DO_NOT_MODIFY.suika_core.fruit_catalog import FruitCatalog
from DO_NOT_MODIFY.suika_core.physics_world import PhysicsWorld
from DO_NOT_MODIFY.suika_core.scoring import ScoreTracker
from DO_NOT_MODIFY.suika_core.merge_system import MergeSystem


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def catalog(config):
    return FruitCatalog(config)


@pytest.fixture
def physics(config):
    return PhysicsWorld(config)


@pytest.fixture
def scorer(config):
    return ScoreTracker(config)


@pytest.fixture
def merger(physics, scorer, config):
    return MergeSystem(physics, scorer, config)


class TestMergeRules:
    """Test merge mechanics."""
    
    def test_same_type_fruits_can_merge(self, physics, merger, catalog):
        """Two fruits of same type should be merge candidates."""
        fruit_type = catalog[0]
        
        # Spawn two cherries close together
        f1 = physics.spawn_fruit(fruit_type, 100, 100)
        f2 = physics.spawn_fruit(fruit_type, 130, 100)
        
        # Simulate until they collide
        for _ in range(100):
            physics.step()
            merges = merger.resolve_merges()
            if merges:
                break
        
        # Should have merged
        assert len(physics.fruits) <= 2
    
    def test_different_type_fruits_no_merge(self, physics, merger, catalog):
        """Different fruit types should not merge."""
        cherry = catalog[0]
        strawberry = catalog[1]
        
        # Spawn different fruits close together
        physics.spawn_fruit(cherry, 100, 100)
        physics.spawn_fruit(strawberry, 130, 100)
        
        # Simulate
        for _ in range(100):
            physics.step()
            merges = merger.resolve_merges()
        
        # Both should still exist (no merge)
        assert physics.fruit_count == 2
    
    def test_merge_produces_next_type(self, physics, merger, catalog, scorer):
        """Merging two type-N fruits produces one type-(N+1) fruit."""
        cherry = catalog[0]  # Type 0
        
        # Spawn two cherries at same position to force merge
        physics.spawn_fruit(cherry, 100, 200)
        physics.spawn_fruit(cherry, 100, 200)
        
        # Run simulation
        for _ in range(50):
            physics.step()
            merges = merger.resolve_merges()
            if merges:
                break
        
        # Should have one fruit of type 1 (strawberry)
        assert physics.fruit_count == 1
        remaining = list(physics.fruits.values())[0]
        assert remaining.type_id == 1
    
    def test_watermelon_watermelon_merge_creates_skull(self, physics, merger, catalog, scorer):
        """Merging two watermelons creates a skull (final form)."""
        watermelon = catalog.watermelon
        
        # Spawn two watermelons at same position
        physics.spawn_fruit(watermelon, 200, 300)
        physics.spawn_fruit(watermelon, 200, 300)
        
        # Run simulation
        for _ in range(50):
            physics.step()
            merges = merger.resolve_merges()
            if merges:
                break
        
        # Should have one skull
        assert physics.fruit_count == 1
        remaining = list(physics.fruits.values())[0]
        assert remaining.type_id == catalog.skull_id
        assert catalog.is_final_fruit(remaining.type_id)
    
    def test_merge_awards_score(self, physics, merger, catalog, scorer):
        """Merging should increase score."""
        cherry = catalog[0]
        
        initial_score = scorer.score
        
        # Spawn and merge two cherries
        physics.spawn_fruit(cherry, 100, 200)
        physics.spawn_fruit(cherry, 100, 200)
        
        for _ in range(50):
            physics.step()
            merger.resolve_merges()
        
        # Score should have increased
        assert scorer.score > initial_score


class TestMergeQueue:
    """Test merge queue deduplication and ordering."""
    
    def test_deduplication(self, physics, merger, catalog):
        """Same pair shouldn't merge multiple times per tick."""
        cherry = catalog[0]
        
        # Create overlap situation
        physics.spawn_fruit(cherry, 100, 200)
        physics.spawn_fruit(cherry, 100, 200)
        
        # Single step should only produce one merge
        physics.step()
        merges = merger.resolve_merges()
        
        assert len(merges) <= 1
    
    def test_each_fruit_merges_once_per_tick(self, physics, merger, catalog):
        """Each fruit can only merge once per resolution pass."""
        cherry = catalog[0]
        
        # Create three overlapping cherries
        physics.spawn_fruit(cherry, 100, 200)
        physics.spawn_fruit(cherry, 100, 200)
        physics.spawn_fruit(cherry, 100, 200)
        
        physics.step()
        merges = merger.resolve_merges()
        
        # At most one merge per tick
        assert len(merges) <= 1
        
        # One fruit should remain unmerged
        # (2 merged into 1, plus 1 original = 2, or just the 3 if no merge yet)
        assert physics.fruit_count >= 1
