"""
Tests for spawn queue RNG.
"""

import pytest
from collections import Counter

from DO_NOT_MODIFY.suika_core.config_loader import load_config
from DO_NOT_MODIFY.suika_core.rng import SpawnQueue


@pytest.fixture
def config():
    return load_config()


class TestSpawnQueue:
    """Test weighted shuffle-bag queue."""
    
    def test_deterministic_with_seed(self, config):
        """Same seed should produce same sequence."""
        q1 = SpawnQueue(config, seed=42)
        q2 = SpawnQueue(config, seed=42)
        
        seq1 = [q1.advance() for _ in range(50)]
        seq2 = [q2.advance() for _ in range(50)]
        
        assert seq1 == seq2
    
    def test_different_seeds_differ(self, config):
        """Different seeds should produce different sequences."""
        q1 = SpawnQueue(config, seed=42)
        q2 = SpawnQueue(config, seed=123)
        
        seq1 = [q1.advance() for _ in range(50)]
        seq2 = [q2.advance() for _ in range(50)]
        
        assert seq1 != seq2
    
    def test_only_spawnable_fruits(self, config):
        """Queue should only produce spawnable fruit IDs."""
        queue = SpawnQueue(config, seed=42)
        max_spawnable = config.rng.spawnable_count
        
        for _ in range(200):
            fruit_id = queue.advance()
            assert 0 <= fruit_id < max_spawnable
    
    def test_weighted_distribution(self, config):
        """Fruits should appear roughly according to weights."""
        queue = SpawnQueue(config, seed=42)
        
        # Sample many fruits
        counts = Counter()
        for _ in range(1000):
            counts[queue.advance()] += 1
        
        # Check that all spawnable fruits appear
        for i in range(config.rng.spawnable_count):
            assert counts[i] > 0
        
        # Check relative ordering (lower IDs should be more common)
        # This is approximate due to randomness
        assert counts[0] > counts[config.rng.spawnable_count - 1]
    
    def test_current_and_next_peek(self, config):
        """Current and next fruit IDs should be predictable."""
        queue = SpawnQueue(config, seed=42)
        
        for _ in range(20):
            current = queue.current_fruit_id
            next_fruit = queue.next_fruit_id
            
            # Advance should return current
            advanced = queue.advance()
            assert advanced == current
            
            # Next should become current
            assert queue.current_fruit_id == next_fruit
    
    def test_reset_restores_sequence(self, config):
        """Reset with same seed should restore sequence."""
        queue = SpawnQueue(config, seed=42)
        
        # Get initial sequence
        initial = [queue.advance() for _ in range(10)]
        
        # Reset
        queue.reset(seed=42)
        
        # Should get same sequence
        after_reset = [queue.advance() for _ in range(10)]
        
        assert initial == after_reset
    
    def test_bag_exhaustion_refill(self, config):
        """Queue should refill bag when exhausted."""
        queue = SpawnQueue(config, seed=42)
        bag_size = config.rng.bag_size
        
        # Draw more than one bag
        for _ in range(bag_size * 3):
            fruit_id = queue.advance()
            assert 0 <= fruit_id < config.rng.spawnable_count
