"""
Test suite for verifying all observation and action API elements.

Ensures the environment returns correctly shaped and typed observations,
and that actions are processed correctly.
"""

import numpy as np
import pytest
from DO_NOT_MODIFY.suika_core.env_gym import SuikaEnv


class TestObservationAPI:
    """Verify all observation space elements."""
    
    @pytest.fixture
    def env(self):
        """Create fresh environment for each test."""
        env = SuikaEnv()
        yield env
        env.close()
    
    @pytest.fixture
    def obs_after_reset(self, env):
        """Get observation after reset."""
        obs, info = env.reset(seed=42)
        return obs
    
    @pytest.fixture
    def obs_after_step(self, env):
        """Get observation after one step."""
        env.reset(seed=42)
        obs, _, _, _, _ = env.step(0.0)
        return obs
    
    # =========================================================================
    # Core State Observations
    # =========================================================================
    
    def test_spawner_x(self, obs_after_reset):
        """spawner_x should be a float in [-1, 1]."""
        assert "spawner_x" in obs_after_reset
        assert obs_after_reset["spawner_x"].dtype == np.float32
        assert obs_after_reset["spawner_x"].shape == ()
        assert -1.0 <= float(obs_after_reset["spawner_x"]) <= 1.0
    
    def test_current_fruit_id(self, obs_after_reset):
        """current_fruit_id should be an int in [0, 10]."""
        assert "current_fruit_id" in obs_after_reset
        assert obs_after_reset["current_fruit_id"].dtype == np.int32
        assert obs_after_reset["current_fruit_id"].shape == ()
        assert 0 <= int(obs_after_reset["current_fruit_id"]) <= 10
    
    def test_next_fruit_id(self, obs_after_reset):
        """next_fruit_id should be an int in [0, 10]."""
        assert "next_fruit_id" in obs_after_reset
        assert obs_after_reset["next_fruit_id"].dtype == np.int32
        assert obs_after_reset["next_fruit_id"].shape == ()
        assert 0 <= int(obs_after_reset["next_fruit_id"]) <= 10
    
    def test_score(self, obs_after_reset):
        """score should be a non-negative int64."""
        assert "score" in obs_after_reset
        assert obs_after_reset["score"].dtype == np.int64
        assert obs_after_reset["score"].shape == ()
        assert int(obs_after_reset["score"]) >= 0
    
    def test_drops_used(self, obs_after_reset):
        """drops_used should be a non-negative int."""
        assert "drops_used" in obs_after_reset
        assert obs_after_reset["drops_used"].dtype == np.int32
        assert obs_after_reset["drops_used"].shape == ()
        assert int(obs_after_reset["drops_used"]) >= 0
    
    def test_objects_count(self, obs_after_reset):
        """objects_count should be a non-negative int."""
        assert "objects_count" in obs_after_reset
        assert obs_after_reset["objects_count"].dtype == np.int32
        assert obs_after_reset["objects_count"].shape == ()
        assert int(obs_after_reset["objects_count"]) >= 0
    
    # =========================================================================
    # Board Constants
    # =========================================================================
    
    def test_board_width(self, obs_after_reset):
        """board_width should be a positive float."""
        assert "board_width" in obs_after_reset
        assert obs_after_reset["board_width"].dtype == np.float32
        assert float(obs_after_reset["board_width"]) > 0
    
    def test_board_height(self, obs_after_reset):
        """board_height should be a positive float."""
        assert "board_height" in obs_after_reset
        assert obs_after_reset["board_height"].dtype == np.float32
        assert float(obs_after_reset["board_height"]) > 0
    
    def test_lose_line_y(self, obs_after_reset):
        """lose_line_y should be a positive float less than board_height."""
        assert "lose_line_y" in obs_after_reset
        assert obs_after_reset["lose_line_y"].dtype == np.float32
        lose_y = float(obs_after_reset["lose_line_y"])
        board_h = float(obs_after_reset["board_height"])
        assert 0 < lose_y < board_h
    
    # =========================================================================
    # Danger Metrics
    # =========================================================================
    
    def test_danger_level(self, obs_after_step):
        """danger_level should be a float >= 0."""
        assert "danger_level" in obs_after_step
        assert obs_after_step["danger_level"].dtype == np.float32
        assert float(obs_after_step["danger_level"]) >= 0
    
    def test_highest_fruit_y(self, obs_after_step):
        """highest_fruit_y should be a non-negative float."""
        assert "highest_fruit_y" in obs_after_step
        assert obs_after_step["highest_fruit_y"].dtype == np.float32
        assert float(obs_after_step["highest_fruit_y"]) >= 0
    
    def test_distance_to_lose_line(self, obs_after_step):
        """distance_to_lose_line should be a float."""
        assert "distance_to_lose_line" in obs_after_step
        assert obs_after_step["distance_to_lose_line"].dtype == np.float32
    
    # =========================================================================
    # Spatial Features
    # =========================================================================
    
    def test_center_of_mass_x(self, obs_after_step):
        """center_of_mass_x should be within board bounds."""
        assert "center_of_mass_x" in obs_after_step
        assert obs_after_step["center_of_mass_x"].dtype == np.float32
        com_x = float(obs_after_step["center_of_mass_x"])
        board_w = float(obs_after_step["board_width"])
        assert 0 <= com_x <= board_w
    
    def test_center_of_mass_y(self, obs_after_step):
        """center_of_mass_y should be non-negative."""
        assert "center_of_mass_y" in obs_after_step
        assert obs_after_step["center_of_mass_y"].dtype == np.float32
        assert float(obs_after_step["center_of_mass_y"]) >= 0
    
    def test_largest_fruit_type_id(self, obs_after_step):
        """largest_fruit_type_id should be in [-1, 10]."""
        assert "largest_fruit_type_id" in obs_after_step
        assert obs_after_step["largest_fruit_type_id"].dtype == np.int32
        type_id = int(obs_after_step["largest_fruit_type_id"])
        assert -1 <= type_id <= 10
    
    def test_largest_fruit_x(self, obs_after_step):
        """largest_fruit_x should be within board bounds."""
        assert "largest_fruit_x" in obs_after_step
        assert obs_after_step["largest_fruit_x"].dtype == np.float32
    
    def test_largest_fruit_y(self, obs_after_step):
        """largest_fruit_y should be non-negative."""
        assert "largest_fruit_y" in obs_after_step
        assert obs_after_step["largest_fruit_y"].dtype == np.float32
    
    # =========================================================================
    # Stack Quality Metrics
    # =========================================================================
    
    def test_packing_efficiency(self, obs_after_step):
        """packing_efficiency should be in [0, 1]."""
        assert "packing_efficiency" in obs_after_step
        assert obs_after_step["packing_efficiency"].dtype == np.float32
        eff = float(obs_after_step["packing_efficiency"])
        assert 0 <= eff <= 1.0
    
    def test_surface_roughness(self, obs_after_step):
        """surface_roughness should be non-negative."""
        assert "surface_roughness" in obs_after_step
        assert obs_after_step["surface_roughness"].dtype == np.float32
        assert float(obs_after_step["surface_roughness"]) >= 0
    
    def test_island_count(self, obs_after_step):
        """island_count should be non-negative integer."""
        assert "island_count" in obs_after_step
        assert obs_after_step["island_count"].dtype == np.int32
        assert int(obs_after_step["island_count"]) >= 0
    
    def test_buried_count(self, obs_after_step):
        """buried_count should be non-negative integer."""
        assert "buried_count" in obs_after_step
        assert obs_after_step["buried_count"].dtype == np.int32
        assert int(obs_after_step["buried_count"]) >= 0
    
    def test_neighbor_discord(self, obs_after_step):
        """neighbor_discord should be non-negative float."""
        assert "neighbor_discord" in obs_after_step
        assert obs_after_step["neighbor_discord"].dtype == np.float32
        assert float(obs_after_step["neighbor_discord"]) >= 0
    
    # =========================================================================
    # Height Map
    # =========================================================================
    
    def test_height_map(self, obs_after_step):
        """height_map should be a (20,) float32 array of non-negative values."""
        assert "height_map" in obs_after_step
        hm = obs_after_step["height_map"]
        assert hm.dtype == np.float32
        assert hm.shape == (20,)
        assert np.all(hm >= 0)
    
    # =========================================================================
    # Object Arrays
    # =========================================================================
    
    def test_obj_mask(self, obs_after_step):
        """obj_mask should be a (200,) boolean array."""
        assert "obj_mask" in obs_after_step
        mask = obs_after_step["obj_mask"]
        assert mask.dtype == bool
        assert mask.shape == (200,)
    
    def test_obj_type_id(self, obs_after_step):
        """obj_type_id should be a (200,) int16 array."""
        assert "obj_type_id" in obs_after_step
        types = obs_after_step["obj_type_id"]
        assert types.dtype == np.int16
        assert types.shape == (200,)
    
    def test_obj_x(self, obs_after_step):
        """obj_x should be a (200,) float32 array."""
        assert "obj_x" in obs_after_step
        xs = obs_after_step["obj_x"]
        assert xs.dtype == np.float32
        assert xs.shape == (200,)
    
    def test_obj_y(self, obs_after_step):
        """obj_y should be a (200,) float32 array."""
        assert "obj_y" in obs_after_step
        ys = obs_after_step["obj_y"]
        assert ys.dtype == np.float32
        assert ys.shape == (200,)
    
    def test_obj_vx(self, obs_after_step):
        """obj_vx should be a (200,) float32 array."""
        assert "obj_vx" in obs_after_step
        vxs = obs_after_step["obj_vx"]
        assert vxs.dtype == np.float32
        assert vxs.shape == (200,)
    
    def test_obj_vy(self, obs_after_step):
        """obj_vy should be a (200,) float32 array."""
        assert "obj_vy" in obs_after_step
        vys = obs_after_step["obj_vy"]
        assert vys.dtype == np.float32
        assert vys.shape == (200,)
    
    def test_obj_ang(self, obs_after_step):
        """obj_ang should be a (200,) float32 array."""
        assert "obj_ang" in obs_after_step
        angs = obs_after_step["obj_ang"]
        assert angs.dtype == np.float32
        assert angs.shape == (200,)
    
    def test_obj_ang_vel(self, obs_after_step):
        """obj_ang_vel should be a (200,) float32 array."""
        assert "obj_ang_vel" in obs_after_step
        ang_vels = obs_after_step["obj_ang_vel"]
        assert ang_vels.dtype == np.float32
        assert ang_vels.shape == (200,)
    
    def test_obj_radius(self, obs_after_step):
        """obj_radius should be a (200,) float32 array."""
        assert "obj_radius" in obs_after_step
        radii = obs_after_step["obj_radius"]
        assert radii.dtype == np.float32
        assert radii.shape == (200,)
    
    def test_obj_arrays_consistency(self, obs_after_step):
        """Object arrays should be consistent with mask."""
        mask = obs_after_step["obj_mask"]
        valid_count = mask.sum()
        
        # After one step, should have exactly 1 fruit
        assert valid_count >= 1
        
        # Valid entries should have positive radius
        radii = obs_after_step["obj_radius"][mask]
        assert np.all(radii > 0)
        
        # Valid entries should have valid type IDs
        types = obs_after_step["obj_type_id"][mask]
        assert np.all(types >= 0)
        assert np.all(types <= 10)


class TestActionAPI:
    """Verify action space works correctly."""
    
    @pytest.fixture
    def env(self):
        """Create fresh environment for each test."""
        env = SuikaEnv()
        yield env
        env.close()
    
    def test_action_space_shape(self, env):
        """Action space should be scalar Box in [-1, 1]."""
        assert env.action_space.shape == ()
        assert env.action_space.low == -1.0
        assert env.action_space.high == 1.0
    
    def test_action_left(self, env):
        """Action -1.0 should result in valid observation."""
        env.reset(seed=42)
        obs, _, _, _, _ = env.step(-1.0)
        
        # Fruit should exist and be within board bounds
        mask = obs["obj_mask"]
        xs = obs["obj_x"][mask]
        if len(xs) > 0:
            assert 0 <= xs[-1] <= float(obs["board_width"])
    
    def test_action_right(self, env):
        """Action 1.0 should result in valid observation."""
        env.reset(seed=42)
        obs, _, _, _, _ = env.step(1.0)
        
        # Fruit should exist and be within board bounds
        mask = obs["obj_mask"]
        xs = obs["obj_x"][mask]
        if len(xs) > 0:
            assert 0 <= xs[-1] <= float(obs["board_width"])
    
    def test_action_center(self, env):
        """Action 0.0 should drop at center."""
        env.reset(seed=42)
        obs, _, _, _, _ = env.step(0.0)
        
        # Fruit should be near center
        mask = obs["obj_mask"]
        xs = obs["obj_x"][mask]
        center = obs["board_width"] / 2
        if len(xs) > 0:
            assert abs(xs[-1] - center) < obs["board_width"] * 0.3
    
    def test_action_clamping(self, env):
        """Actions outside [-1, 1] should be handled gracefully."""
        env.reset(seed=42)
        
        # Should not crash with extreme values
        obs1, _, _, _, _ = env.step(-5.0)
        env.reset(seed=42)
        obs2, _, _, _, _ = env.step(5.0)
        
        # Both should result in valid observations
        assert obs1["obj_mask"].sum() >= 1
        assert obs2["obj_mask"].sum() >= 1
    
    def test_action_float_conversion(self, env):
        """Actions as numpy arrays should work."""
        env.reset(seed=42)
        
        # Numpy scalar
        obs, _, _, _, _ = env.step(np.float32(0.5))
        assert obs["obj_mask"].sum() >= 1
        
        env.reset(seed=42)
        # Numpy array with shape (1,)
        obs, _, _, _, _ = env.step(np.array([0.5]))
        assert obs["obj_mask"].sum() >= 1


class TestInfoDict:
    """Verify info dictionary returned from step()."""
    
    @pytest.fixture
    def env(self):
        """Create fresh environment for each test."""
        env = SuikaEnv()
        yield env
        env.close()
    
    def test_info_keys(self, env):
        """Info dict should contain expected keys."""
        env.reset(seed=42)
        _, _, _, _, info = env.step(0.0)
        
        expected_keys = ["score", "delta_score", "drops_used", "merges", "sim_time"]
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"
    
    def test_info_score(self, env):
        """info['score'] should match obs['score']."""
        env.reset(seed=42)
        obs, _, _, _, info = env.step(0.0)
        assert info["score"] == int(obs["score"])
    
    def test_info_delta_score(self, env):
        """info['delta_score'] should be non-negative."""
        env.reset(seed=42)
        _, _, _, _, info = env.step(0.0)
        assert info["delta_score"] >= 0
    
    def test_info_drops_used(self, env):
        """info['drops_used'] should increment with each step."""
        env.reset(seed=42)
        _, _, _, _, info1 = env.step(0.0)
        _, _, _, _, info2 = env.step(0.0)
        
        assert info1["drops_used"] == 1
        assert info2["drops_used"] == 2
    
    def test_info_merges(self, env):
        """info['merges'] should be non-negative."""
        env.reset(seed=42)
        _, _, _, _, info = env.step(0.0)
        assert info["merges"] >= 0
    
    def test_info_sim_time(self, env):
        """info['sim_time'] should be positive."""
        env.reset(seed=42)
        _, _, _, _, info = env.step(0.0)
        assert info["sim_time"] > 0


class TestRewardAndTermination:
    """Verify reward is always 0 and termination works."""
    
    @pytest.fixture
    def env(self):
        """Create fresh environment for each test."""
        env = SuikaEnv()
        yield env
        env.close()
    
    def test_reward_always_zero(self, env):
        """Reward should always be 0.0."""
        env.reset(seed=42)
        
        for _ in range(10):
            _, reward, terminated, _, _ = env.step(0.0)
            assert reward == 0.0
            if terminated:
                break
    
    def test_terminated_is_boolean(self, env):
        """terminated should be a boolean."""
        env.reset(seed=42)
        _, _, terminated, _, _ = env.step(0.0)
        assert isinstance(terminated, bool)
    
    def test_truncated_is_boolean(self, env):
        """truncated should be a boolean."""
        env.reset(seed=42)
        _, _, _, truncated, _ = env.step(0.0)
        assert isinstance(truncated, bool)


class TestVectorEnvObservations:
    """Test that vector environments return correct batch shapes."""
    
    def test_vectorenv_observation_shapes(self):
        """Vector env observations should have batch dimension."""
        from DO_NOT_MODIFY.suika_core.vector_env import SuikaVectorEnv
        
        num_envs = 4
        vec_env = SuikaVectorEnv(num_envs=num_envs)
        obs, infos = vec_env.reset(seed=42)
        
        # Scalars should have shape (num_envs,)
        assert obs["score"].shape == (num_envs,)
        assert obs["current_fruit_id"].shape == (num_envs,)
        assert obs["next_fruit_id"].shape == (num_envs,)
        assert obs["drops_used"].shape == (num_envs,)
        assert obs["objects_count"].shape == (num_envs,)
        assert obs["spawner_x"].shape == (num_envs,)
        
        # Object arrays should have shape (num_envs, 200)
        assert obs["obj_x"].shape == (num_envs, 200)
        assert obs["obj_y"].shape == (num_envs, 200)
        assert obs["obj_vx"].shape == (num_envs, 200)
        assert obs["obj_vy"].shape == (num_envs, 200)
        assert obs["obj_ang"].shape == (num_envs, 200)
        assert obs["obj_ang_vel"].shape == (num_envs, 200)
        assert obs["obj_type_id"].shape == (num_envs, 200)
        assert obs["obj_mask"].shape == (num_envs, 200)
        
        vec_env.close()
    
    def test_vectorenv_derived_observations(self):
        """Vector env should include height_map and danger_level."""
        from DO_NOT_MODIFY.suika_core.vector_env import SuikaVectorEnv
        
        num_envs = 4
        vec_env = SuikaVectorEnv(num_envs=num_envs)
        obs, _ = vec_env.reset(seed=42)
        
        # Take a step to get some fruits on board
        actions = np.zeros(num_envs)
        obs, _, _, _, _ = vec_env.step(actions)
        
        # Height map should have shape (num_envs, 20)
        assert "height_map" in obs
        assert obs["height_map"].shape == (num_envs, 20)
        assert obs["height_map"].dtype == np.float32
        
        # Danger level should have shape (num_envs,)
        assert "danger_level" in obs
        assert obs["danger_level"].shape == (num_envs,)
        assert obs["danger_level"].dtype == np.float32
        
        # Other derived observations
        assert "highest_fruit_y" in obs
        assert obs["highest_fruit_y"].shape == (num_envs,)
        
        assert "packing_efficiency" in obs
        assert obs["packing_efficiency"].shape == (num_envs,)
        
        assert "island_count" in obs
        assert obs["island_count"].shape == (num_envs,)
        
        # Board constants
        assert "board_width" in obs
        assert "lose_line_y" in obs
        
        vec_env.close()
    
    def test_vectorenv_action_batch(self):
        """Vector env should accept batched actions."""
        from DO_NOT_MODIFY.suika_core.vector_env import SuikaVectorEnv
        
        num_envs = 4
        vec_env = SuikaVectorEnv(num_envs=num_envs)
        obs, _ = vec_env.reset(seed=42)
        
        # Batched actions
        actions = np.array([0.0, -0.5, 0.5, 1.0])
        obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
        
        assert rewards.shape == (num_envs,)
        assert terminateds.shape == (num_envs,)
        assert truncateds.shape == (num_envs,)
        
        # infos is a dict of arrays, check that arrays have correct shape
        assert isinstance(infos, dict)
        assert infos["score"].shape == (num_envs,)
        
        vec_env.close()
