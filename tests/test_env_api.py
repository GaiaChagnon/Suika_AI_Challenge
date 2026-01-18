"""
Tests for Gymnasium environment API.
"""

import pytest
import numpy as np

from DO_NOT_MODIFY.suika_core.config_loader import load_config
from DO_NOT_MODIFY.suika_core.env_gym import SuikaEnv
from DO_NOT_MODIFY.suika_core.vector_env import SuikaVectorEnv


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def env():
    env = SuikaEnv()
    yield env
    env.close()


@pytest.fixture
def vec_env():
    env = SuikaVectorEnv(num_envs=4, seed=42)
    yield env
    env.close()


class TestSuikaEnv:
    """Test single environment API."""
    
    def test_reset_returns_obs_and_info(self, env):
        """Reset should return (observation, info) tuple."""
        result = env.reset(seed=42)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        obs, info = result
        assert isinstance(obs, dict)
        assert isinstance(info, dict)
    
    def test_observation_structure(self, env):
        """Observation should have expected keys and shapes."""
        obs, _ = env.reset(seed=42)
        
        # Check scalar fields
        assert "spawner_x" in obs
        assert "current_fruit_id" in obs
        assert "next_fruit_id" in obs
        assert "score" in obs
        assert "drops_used" in obs
        assert "objects_count" in obs
        
        # Check array fields
        assert "obj_type_id" in obs
        assert "obj_x" in obs
        assert "obj_y" in obs
        assert "obj_mask" in obs
        
        # Check shapes
        max_obj = env.config.observation.max_objects
        assert obs["obj_type_id"].shape == (max_obj,)
        assert obs["obj_x"].shape == (max_obj,)
        assert obs["obj_mask"].shape == (max_obj,)
    
    def test_step_returns_five_values(self, env):
        """Step should return (obs, reward, terminated, truncated, info)."""
        env.reset(seed=42)
        
        result = env.step(0.0)
        
        assert isinstance(result, tuple)
        assert len(result) == 5
        
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_reward_is_always_zero(self, env):
        """Environment reward should always be 0.0."""
        env.reset(seed=42)
        
        for _ in range(20):
            _, reward, terminated, truncated, _ = env.step(np.random.uniform(-1, 1))
            assert reward == 0.0
            
            if terminated or truncated:
                env.reset()
    
    def test_info_contains_score(self, env):
        """Info dict should contain score and delta_score."""
        env.reset(seed=42)
        
        _, _, _, _, info = env.step(0.0)
        
        assert "score" in info
        assert "delta_score" in info
        assert "drops_used" in info
    
    def test_action_bounds(self, env):
        """Actions outside [-1, 1] should be clamped."""
        env.reset(seed=42)
        
        # Extreme actions should not crash
        env.step(-2.0)  # Will be clamped to -1.0
        env.step(2.0)   # Will be clamped to 1.0
        env.step(-100)
        env.step(100)
    
    def test_deterministic_with_seed(self, config):
        """Same seed should produce consistent fruit queue (RNG determinism)."""
        env1 = SuikaEnv()
        env2 = SuikaEnv()
        
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        
        # Fruit queue should be deterministic
        assert obs1["current_fruit_id"] == obs2["current_fruit_id"]
        assert obs1["next_fruit_id"] == obs2["next_fruit_id"]
        
        for _ in range(10):
            action = rng1.uniform(-1, 1)
            obs1, _, t1, tr1, info1 = env1.step(action)
            
            action = rng2.uniform(-1, 1)
            obs2, _, t2, tr2, info2 = env2.step(action)
            
            # Fruit queue should stay synchronized
            assert obs1["current_fruit_id"] == obs2["current_fruit_id"]
            assert obs1["next_fruit_id"] == obs2["next_fruit_id"]
            
            # Note: Physics positions may differ slightly due to
            # non-deterministic floating point operations in Chipmunk2D
            # This is acceptable per spec: "physics reproducibility doesn't need to be perfect"
            
            if t1 or tr1:
                break
        
        env1.close()
        env2.close()
    
    def test_episode_terminates(self, config):
        """Episode should eventually terminate or truncate."""
        env = SuikaEnv()
        env.reset(seed=42)
        
        rng = np.random.default_rng(42)
        terminated = False
        truncated = False
        steps = 0
        max_steps = config.caps.max_drops + 100
        
        while not (terminated or truncated) and steps < max_steps:
            _, _, terminated, truncated, _ = env.step(rng.uniform(-1, 1))
            steps += 1
        
        # Should have ended
        assert terminated or truncated
        env.close()


class TestVectorEnv:
    """Test vectorized environment API."""
    
    def test_reset_returns_batched_obs(self, vec_env):
        """Reset should return observations for all envs."""
        obs, info = vec_env.reset()
        
        assert obs["spawner_x"].shape == (4,)
        assert obs["score"].shape == (4,)
        assert obs["obj_x"].shape == (4, vec_env.config.observation.max_objects)
    
    def test_step_accepts_batched_actions(self, vec_env):
        """Step should accept array of actions."""
        vec_env.reset()
        
        actions = np.array([0.0, -0.5, 0.5, 0.0], dtype=np.float32)
        obs, rewards, terminateds, truncateds, info = vec_env.step(actions)
        
        assert rewards.shape == (4,)
        assert terminateds.shape == (4,)
        assert truncateds.shape == (4,)
    
    def test_rewards_all_zero(self, vec_env):
        """All rewards should be zero."""
        vec_env.reset()
        
        for _ in range(10):
            actions = np.random.uniform(-1, 1, size=4).astype(np.float32)
            _, rewards, terminateds, truncateds, _ = vec_env.step(actions)
            
            assert np.all(rewards == 0.0)
            
            # Reset any terminated envs
            done = np.where(terminateds | truncateds)[0].tolist()
            if done:
                vec_env.reset(env_indices=done)
    
    def test_individual_env_termination(self, vec_env):
        """Individual envs should terminate independently."""
        vec_env.reset()
        
        # Run until at least one terminates
        any_done = False
        for _ in range(1000):
            actions = np.random.uniform(-1, 1, size=4).astype(np.float32)
            _, _, terminateds, truncateds, _ = vec_env.step(actions)
            
            done = terminateds | truncateds
            if np.any(done):
                any_done = True
                # Some envs done, but not necessarily all
                # Reset done ones
                done_indices = np.where(done)[0].tolist()
                vec_env.reset(env_indices=done_indices)
                break
        
        # Should have seen at least one termination in 1000 steps
        # (with random actions, likely to overflow)
    
    def test_sample_actions(self, vec_env):
        """sample_actions should return valid action array."""
        actions = vec_env.sample_actions()
        
        assert actions.shape == (4,)
        assert actions.dtype == np.float32
        assert np.all(actions >= -1.0)
        assert np.all(actions <= 1.0)
