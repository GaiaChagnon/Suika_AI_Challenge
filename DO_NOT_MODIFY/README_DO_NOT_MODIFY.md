# DO NOT MODIFY

This directory contains the core game logic for the Suika Competition.

## Why This Is Locked

The files in this directory control:
- **Fruit RNG**: The sequence of fruits available to drop
- **Scoring**: Point values for merges
- **Merge Rules**: When and how fruits combine
- **Termination**: Game over conditions
- **Physics Tuning**: Gravity, friction, elasticity

Modifying any of these would give unfair advantages and invalidate evaluation results.

## Contents

### game_config.yaml
Single YAML file containing ALL tunable parameters:
- Board dimensions and lose line
- Physics constants
- All 11 fruit types with collision shapes, masses, scores
- RNG bag weights
- Game caps

### suika_core/
Core simulation modules:
- `config_loader.py` - YAML parsing and validation
- `rng.py` - Weighted shuffle-bag fruit queue
- `fruit_catalog.py` - Fruit type definitions
- `physics_world.py` - pymunk Space setup
- `merge_system.py` - Collision handling and merge queue
- `scoring.py` - Score calculation
- `rules.py` - Termination and spawn rules
- `state_snapshot.py` - Observation packing
- `game.py` - CoreGame orchestrator
- `env_gym.py` - Gymnasium environment wrapper
- `vector_env.py` - Single-process vectorized env
- `mp_vector_env.py` - Multiprocessing vectorized env
- `render_solid.py` - Fast blob renderer
- `render_full_pygame.py` - Pretty renderer with sprites

### evaluation/
- `seed_bank.json` - Fixed seeds for evaluation
- `run_eval.py` - Submission evaluation harness

## For Contestants

Your agent code goes in `contestants/your_team_name/`.
You only need to implement `agent.py` with an `act(obs) -> action` function.

See `contestants/team_template/` for an example.
