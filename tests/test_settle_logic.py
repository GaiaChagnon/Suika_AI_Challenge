"""
Tests for physics settle logic.
"""

import pytest

from DO_NOT_MODIFY.suika_core.config_loader import load_config
from DO_NOT_MODIFY.suika_core.fruit_catalog import FruitCatalog
from DO_NOT_MODIFY.suika_core.physics_world import PhysicsWorld
from DO_NOT_MODIFY.suika_core.game import CoreGame


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
def game(config):
    return CoreGame(config=config, seed=42)


class TestSettleDetection:
    """Test physics settle detection."""
    
    def test_initial_state_is_settled(self, physics):
        """Empty world should be settled."""
        assert physics.check_settled()
    
    def test_moving_fruit_not_settled(self, physics, catalog):
        """Fruit with velocity should not be settled."""
        fruit_type = catalog[0]
        physics.spawn_fruit(fruit_type, 100, 400, velocity=(0, -100))
        
        assert not physics.check_settled()
    
    def test_stationary_fruit_is_settled(self, physics, catalog):
        """Fruit at rest should be settled."""
        fruit_type = catalog[0]
        fruit = physics.spawn_fruit(fruit_type, 100, 50)
        
        # Let it settle on the ground
        for _ in range(500):
            physics.step()
        
        assert physics.check_settled()
    
    def test_force_freeze(self, physics, catalog):
        """Force freeze should stop all motion."""
        fruit_type = catalog[0]
        physics.spawn_fruit(fruit_type, 100, 400, velocity=(100, -200))
        physics.spawn_fruit(fruit_type, 200, 400, velocity=(-50, -150))
        
        # Still moving
        physics.step()
        assert not physics.check_settled()
        
        # Force freeze
        physics.force_freeze()
        
        # Now settled
        assert physics.check_settled()
        
        # All velocities should be zero
        for fruit in physics.fruits.values():
            vx, vy = fruit.velocity
            assert vx == 0
            assert vy == 0
            assert fruit.angular_velocity == 0


class TestSettleInGame:
    """Test settle behavior in CoreGame."""
    
    def test_step_waits_for_settle(self, game):
        """Game step should wait for physics to settle."""
        game.reset(seed=42)
        
        # Drop a fruit
        result = game.step(0.0)
        
        # After step, physics should be settled
        assert game.physics.check_settled()
    
    def test_settle_has_time_limit(self, config):
        """Settle should not run forever."""
        game = CoreGame(config=config, seed=42)
        game.reset()
        
        # Drop fruit - should complete within reasonable time
        import time
        start = time.time()
        result = game.step(0.0)
        elapsed = time.time() - start
        
        # Should complete well under max_sim_seconds_per_drop
        assert elapsed < config.physics.max_sim_seconds_per_drop + 1
    
    def test_multiple_drops_all_settle(self, game):
        """Multiple consecutive drops should all settle properly."""
        game.reset(seed=42)
        
        for i in range(10):
            # Drop at different positions
            action = (i / 5) - 1  # Range from -1 to ~1
            result = game.step(action)
            
            # Should always be settled after step
            assert game.physics.check_settled()
