# Suika Competition Environment

A Suika-style fruit-merging game designed as a competition environment for AI agents.

## Overview

Drop fruits from the top of the arena. When two fruits of the same type collide, they merge into a larger fruit. Score points by creating merges, with higher-level fruits worth more. The game ends if any fruit crosses the lose line for more than 1 second.

## Installation

```bash
# Basic installation (headless training)
pip install -e .

# With rendering support (human play, visualization)
pip install -e ".[render]"

# Development (includes testing)
pip install -e ".[all]"
```

## Quick Start

### Human Play
```bash
python -m tools.play_human
python -m tools.play_human --seed 42  # Reproducible game
```

### Training (Example)
```python
from DO_NOT_MODIFY.suika_core import SuikaEnv

env = SuikaEnv(render_mode=None)  # Headless
obs, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()  # Your agent here
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

### Debug Mode (for agent development)
```python
env = SuikaEnv(debug=True)  # Enables verbose output
obs, info = env.reset()
# [DEBUG] SuikaEnv initialized
# [DEBUG]   Board: 450x600
# [DEBUG]   Lose line Y: 540
# [DEBUG]   Max objects: 200
```

### Vectorized Training (Multiprocessing)
```python
from DO_NOT_MODIFY.suika_core import make_async_vec_env, get_recommended_num_envs

# True multiprocessing - one process per CPU core
num_envs = get_recommended_num_envs()  # Auto-detects CPU cores
vec_env = make_async_vec_env(num_envs=num_envs, seed=42)

obs, infos = vec_env.reset()
actions = np.random.uniform(-1, 1, size=num_envs).astype(np.float32)
obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)

vec_env.close()  # Important: close to cleanup processes
```

### Evaluation
```bash
python -m DO_NOT_MODIFY.evaluation.run_eval --agent contestants/team_template
```

---

## Architecture Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SUIKA GAME ARCHITECTURE                           │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────┐
                    │   YOUR AGENT CODE    │
                    │  contestants/team/   │
                    │     agent.py         │
                    └──────────┬───────────┘
                               │ action ∈ [-1, 1]
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         GYMNASIUM INTERFACE                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  SuikaEnv (env_gym.py)                                                  │ │
│  │  • action_space: Box(-1, 1)  →  spawner X position                      │ │
│  │  • observation_space: Dict   ←  game state + derived features           │ │
│  │  • step(action) → (obs, reward=0.0, terminated, truncated, info)        │ │
│  │  • reset(seed) → (obs, info)                                            │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           CORE GAME (game.py)                                │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  │
│  │ PhysicsWorld  │  │  MergeSystem  │  │   ShuffleBag  │  │  ScoreTracker │  │
│  │  (pymunk)     │  │  (collisions) │  │   (RNG)       │  │  (points)     │  │
│  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘  │
│                                                                              │
│  step(action):                                                               │
│    1. Map action [-1,1] → world X coordinate                                 │
│    2. Spawn fruit at (X, spawn_y)                                            │
│    3. Simulate physics until settled                                         │
│    4. Process merges, update score                                           │
│    5. Check termination (lose line, drop cap)                                │
│    6. Return snapshot                                                        │
└──────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        CONFIGURATION (game_config.yaml)                      │
│  • Board dimensions, lose line position                                      │
│  • 11 fruit types with mass, radius, collision shapes                        │
│  • Physics: gravity, friction, elasticity                                    │
│  • Scoring: points per merge                                                 │
│  • RNG: spawn weights                                                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
suika_competition/
├── DO_NOT_MODIFY/              # Core rules, physics, scoring (LOCKED)
│   ├── game_config.yaml        # All tunable parameters
│   ├── suika_core/             # Game simulation
│   │   ├── env_gym.py          # Gymnasium environment wrapper
│   │   ├── game.py             # Core game logic
│   │   ├── physics_world.py    # Pymunk physics
│   │   ├── merge_system.py     # Collision-based merging
│   │   ├── state_snapshot.py   # Observation packing
│   │   ├── scoring.py          # Score calculation
│   │   ├── rng.py              # Shuffle-bag RNG
│   │   ├── rules.py            # Termination conditions
│   │   ├── vector_env.py       # Vectorized environments
│   │   └── render_*.py         # Renderers
│   └── evaluation/             # Scoring harness
│       ├── run_eval.py         # Evaluation script
│       └── seed_bank.json      # Fixed evaluation seeds
│
├── contestants/                # Team submissions
│   └── team_template/          # Copy this to start
│       ├── agent.py            # Your agent implementation
│       └── README.md           # Instructions
│
├── tools/                      # Utilities
│   ├── play_human.py           # Interactive play
│   ├── view_grid.py            # Multi-env viewer
│   ├── benchmark_speed.py      # Performance testing
│   └── replay_viewer.py        # Replay analysis
│
├── assets/sprites/             # Game graphics
└── tests/                      # Test suite
```

---

## Coordinate System

```
                 spawn_y (570)
                     ▲
    ┌────────────────┬────────────────┐  ◄─ lose_line_y (540)
    │                │                │     Game over if fruit stays
    │                │                │     above this for >1 second
    │    GAME        │     AREA       │
    │                │                │
    │       Y        │                │
    │       ▲        │                │
    │       │        │                │
    │       │        │                │
    │       └──────► X                │
    │    (0,0)                        │
    └─────────────────────────────────┘
   X=0                             X=board_width (450)

   • Origin (0, 0) is BOTTOM-LEFT
   • Y increases UPWARD
   • Gravity pulls DOWN (negative Y direction)
   • spawn_y is near the top
   • lose_line_y is the danger threshold
```

### Action Mapping
```
action = -1.0  →  X = left_wall + fruit_radius   (leftmost valid position)
action =  0.0  →  X = board_width / 2            (center)
action = +1.0  →  X = right_wall - fruit_radius  (rightmost valid position)
```

---

## Observation Space

The environment returns a dictionary observation with these fields.

**See `contestants/team_template/API_REFERENCE.md` for complete documentation.**

### Core State
| Key | Shape | Description |
|-----|-------|-------------|
| `spawner_x` | `()` | Current spawner position in [-1, 1] |
| `current_fruit_id` | `()` | Type ID of fruit to drop next (0-10) |
| `next_fruit_id` | `()` | Type ID of fruit after that |
| `score` | `()` | Current game score |
| `drops_used` | `()` | Number of fruits dropped |
| `objects_count` | `()` | Number of fruits on board |

### Basic Derived Features
| Key | Shape | Description |
|-----|-------|-------------|
| `largest_fruit_type_id` | `()` | Type ID of largest fruit on board (-1 if empty) |
| `largest_fruit_x` | `()` | X position of largest fruit |
| `largest_fruit_y` | `()` | Y position of largest fruit |
| `highest_fruit_y` | `()` | Y of topmost fruit (closest to lose line) |
| `danger_level` | `()` | 0-1 indicating proximity to lose line |
| `distance_to_lose_line` | `()` | Distance from highest fruit to lose line (px) |

### Advanced Derived Features
| Key | Shape | Description |
|-----|-------|-------------|
| `center_of_mass_x` | `()` | X coordinate of center of mass |
| `center_of_mass_y` | `()` | Y coordinate of center of mass |
| `packing_efficiency` | `()` | 0-1 ratio of fruit area to convex hull |
| `surface_roughness` | `()` | Std dev of surface heights (flat=0) |
| `island_count` | `()` | Number of separate fruit clusters |
| `buried_count` | `()` | Fruits trapped under others |
| `neighbor_discord` | `()` | Avg type difference between neighbors |
| `height_map` | `(20,)` | 1D "lidar" surface profile |

### Board Info (For Normalization)
| Key | Shape | Description |
|-----|-------|-------------|
| `board_width` | `()` | Width of game board in pixels (450) |
| `board_height` | `()` | Height of game board in pixels (600) |
| `lose_line_y` | `()` | Y coordinate of the lose line (540) |

### Object Arrays
| Key | Shape | Description |
|-----|-------|-------------|
| `obj_type_id` | `(MAX_OBJ,)` | Type IDs of all fruits |
| `obj_x` | `(MAX_OBJ,)` | X positions (world coords) |
| `obj_y` | `(MAX_OBJ,)` | Y positions (world coords) |
| `obj_vx` | `(MAX_OBJ,)` | X velocities |
| `obj_vy` | `(MAX_OBJ,)` | Y velocities |
| `obj_ang` | `(MAX_OBJ,)` | Angular positions (radians) |
| `obj_ang_vel` | `(MAX_OBJ,)` | Angular velocities |
| `obj_radius` | `(MAX_OBJ,)` | Visual radius of each fruit |
| `obj_mask` | `(MAX_OBJ,)` | Boolean mask (True = valid object) |
| `board_rgb` | `(H, W, 3)` | RGB image (if `image_obs=True`) |

---

## Action Space

- **Type**: `Box(low=-1.0, high=1.0, shape=(), dtype=float32)`
- **Meaning**: Spawner X position where -1.0 = left wall, 1.0 = right wall

---

## Reward

**The environment always returns `reward = 0.0`.**

This is intentional: teams must design their own reward function. The `info` dict provides everything needed to compute custom rewards:

```python
info = {
    "score": 1234,              # Current total score
    "delta_score": 18,          # Points gained THIS step (use for reward shaping)
    "drops_used": 50,           # Fruits dropped so far
    "merges": 2,                # Number of merges this step
    "terminated_reason": "",    # "lose_line" or "drop_cap" if ended
    "sim_time": 0.42,           # Physics simulation time (seconds)
}

# Example reward functions:
reward = info["delta_score"]                    # Direct score reward
reward = info["delta_score"] / 100.0            # Scaled
reward = 1.0 if info["delta_score"] > 0 else 0  # Binary merge reward
reward = -100 if terminated else info["delta_score"]  # Penalize death
```

---

## Fruit Types

| ID | Name | Radius | Mass | Merge Score |
|----|------|--------|------|-------------|
| 0 | Cherry | 15 | 1.0 | 2 |
| 1 | Strawberry | 22 | 2.0 | 4 |
| 2 | Grape | 30 | 4.0 | 7 |
| 3 | Dekopon | 42 | 8.0 | 11 |
| 4 | Persimmon | 52 | 12.0 | 18 |
| 5 | Apple | 62 | 18.0 | 30 |
| 6 | Pear | 75 | 28.0 | 50 |
| 7 | Peach | 88 | 40.0 | 80 |
| 8 | Pineapple | 100 | 55.0 | 130 |
| 9 | Honeydew | 115 | 75.0 | 210 |
| 10 | Melon | 135 | 100.0 | 340 (+2000 bonus for melon-melon) |

---

## CLI Reference

### Human Play
```bash
python -m tools.play_human [OPTIONS]

Options:
  --seed SEED      Random seed for reproducibility
  --width WIDTH    Window width (default: 700)
  --height HEIGHT  Window height (default: 800)
  --fps FPS        Target FPS (default: 60)

Controls:
  Mouse            Move spawner
  Click/Space      Drop fruit
  R                Restart game
  ESC              Quit
```

### Grid Viewer (Training Monitor)
```bash
python -m tools.view_grid [OPTIONS]

Options:
  --envs N         Number of environments (default: 9)
  --cols C         Number of columns (default: auto)
  --cell-width W   Cell width in pixels (default: 180)
  --cell-height H  Cell height in pixels (default: 270)
  --fps FPS        Max FPS (default: 30)
  --seed SEED      Random seed
  --style STYLE    "solid" (fast) or "full" (sprites)

Controls:
  SPACE            Pause/resume auto-stepping
  R                Reset all environments
  +/-              Adjust step speed
  ESC              Close viewer
```

### Benchmark
```bash
python -m tools.benchmark_speed [OPTIONS]

Options:
  --envs N         Number of parallel environments (default: 100)
  --steps S        Steps to run (default: 1000)
  --warmup W       Warmup steps (default: 100)
```

### Evaluation
```bash
python -m DO_NOT_MODIFY.evaluation.run_eval [OPTIONS]

Options:
  --agent PATH     Path to agent directory (required)
  --seeds N        Number of seeds to run (default: all in seed_bank.json)
  --save-replays   Save replay data for analysis
  --verbose        Show per-seed scores
```

---

## Live Training Visualization

Use `LiveTrainingViewer` to monitor training in real-time:

```python
from DO_NOT_MODIFY.suika_core import SuikaVectorEnv
from tools.view_grid import LiveTrainingViewer

# Create environments
vec_env = SuikaVectorEnv(num_envs=16)

# Create viewer (runs in background thread)
viewer = LiveTrainingViewer(vec_env, num_cols=4)
viewer.start_background()

# Your training loop
obs, _ = vec_env.reset()
for step in range(100000):
    actions = your_agent.act(obs)
    obs, rewards, dones, truncs, infos = vec_env.step(actions)
    
    viewer.update(step_count=step)  # Non-blocking update
    
    # Handle resets
    done_mask = dones | truncs
    if done_mask.any():
        vec_env.reset(env_indices=np.where(done_mask)[0].tolist())

viewer.stop()
```

---

## Competition Rules

1. **Do not modify anything in `DO_NOT_MODIFY/`**
2. Your agent must implement `act(obs) -> action`
3. Evaluation uses fixed seeds from `seed_bank.json`
4. Final score = mean score across all evaluation seeds

---

## Domain Randomization

**Physics varies slightly each episode** for robust training:

| Parameter | Variance | Description |
|-----------|----------|-------------|
| Friction | ±2% | How quickly fruits stop sliding |
| Elasticity | ±2% | How much fruits bounce |
| Mass (per fruit) | ±4% | Weight of each fruit type |
| Gravity | ±1% | Strength of gravity |

- Same seed = same randomization (deterministic)
- AI agents **cannot observe** these values
- Forces agents to generalize, not overfit

---

## Evaluation Seeds

### Public Seeds (for development)

The `seed_bank.json` contains seeds you can test against:
```bash
python -m DO_NOT_MODIFY.evaluation.run_eval --agent contestants/my_team
```

### ⚠️ Hidden Evaluation Seeds

**The final competition uses a SEPARATE set of hidden seeds.**

- You cannot see or test against these seeds before evaluation
- They are chosen to be diverse and challenging
- Your agent must **generalize**, not memorize specific seeds
- Domain randomization helps prepare for unknown conditions

---

## Tips for Building Agents

1. **Start simple**: Random agent, then rule-based, then ML
2. **Use `danger_level`**: High value means you're about to lose
3. **Use `height_map`**: 20-slice surface profile for easy NN input
4. **Track `packing_efficiency`**: Low values = unstable stack
5. **Watch `buried_count`**: Small fruits trapped = wasted merges
6. **Multiprocessing**: Use `make_async_vec_env` with num_envs = CPU cores
7. **Custom rewards**: Score delta is a good starting point
8. **Debug mode**: Enable `debug=True` to see what's happening
