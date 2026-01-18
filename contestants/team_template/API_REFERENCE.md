# Suika Competition API Reference

Complete guide to building intelligent agents for the Suika game competition.

---

## Table of Contents

### Part I: Observation Space Overview
- [Why These Observations?](#why-these-observations)
- [State & Metadata](#category-state--metadata)
- [Spatial Awareness](#category-spatial-awareness)
- [Danger & Survival](#category-danger--survival)
- [Stack Quality](#category-stack-quality)
- [Per-Fruit Details](#category-per-fruit-details)
- [Strategic Insights](#category-strategic-insights)

### Part II: Technical Reference
- [Environment Setup](#environment-setup)
- [Action Space](#action-space)
- [Observation Space (Detailed)](#observation-space-detailed)
- [Info Dictionary](#info-dictionary)
- [Domain Randomization](#domain-randomization)
- [Evaluation](#evaluation)
- [Coordinate System](#coordinate-system)

---

# Part I: Observation Space Overview

## Why These Observations?

The observation space is designed to give your agent **complete information** about the game state without requiring complex image processing. Each observation serves a specific strategic purpose.

The observations are split into **six categories** based on their function:

1. **State & Metadata** - Basic game information
2. **Spatial Awareness** - Where things are
3. **Danger & Survival** - How close to losing
4. **Stack Quality** - How well-organized the board is
5. **Per-Fruit Details** - Detailed physics data
6. **Strategic Insights** - High-level derived metrics

---

## Category: State & Metadata

**Purpose:** Track game progress and know what fruit you're working with.

| Observation | Type | What It Tells You |
|-------------|------|-------------------|
| `current_fruit_id` | int | What fruit you're about to drop (0=cherry, 10=melon) |
| `next_fruit_id` | int | What's coming next (plan ahead!) |
| `score` | int | Your current score |
| `drops_used` | int | How many fruits you've dropped |
| `objects_count` | int | How many fruits are on the board |

**Why useful for ML:**
- **Sequential planning:** Knowing the next fruit lets models plan 2 steps ahead
- **Game phase detection:** Early game (few drops) vs late game (high score) need different strategies
- **Fruit size awareness:** Large fruits (high IDs) need more careful placement

**Neural network usage:**
- Embed `current_fruit_id` and `next_fruit_id` into feature vectors
- Normalize `score` and `drops_used` to track episode progress
- Use `objects_count` as a complexity signal

---

## Category: Spatial Awareness

**Purpose:** Understand where fruits are and how the board is shaped.

| Observation | Shape | What It Tells You |
|-------------|-------|-------------------|
| `height_map` | (20,) | "1D lidar" - height of stack at 20 horizontal positions |
| `center_of_mass_x` | scalar | Where the weight is distributed horizontally |
| `center_of_mass_y` | scalar | How tall/heavy the stack is |
| `obj_x`, `obj_y` | (200,) each | Exact position of every fruit |
| `obj_radius` | (200,) | Size of each fruit |

**Why useful for ML:**
- **Height map is CNN-friendly:** Treat it as a 1D "image" of the surface
- **Center of mass:** Quick proxy for stability (high COM = unstable)
- **Raw positions:** Full state for policy networks or transformers

**Neural network usage:**
- **CNNs:** Process `height_map` with 1D convolutions to detect patterns
- **MLPs:** Concatenate COM coordinates with other features
- **Attention:** Use `obj_x`, `obj_y` as spatial keys for transformer architectures
- **Graph networks:** Build connectivity graph from positions

---

## Category: Danger & Survival

**Purpose:** Know when you're about to lose and react accordingly.

| Observation | Range | What It Tells You |
|-------------|-------|-------------------|
| `danger_level` | 0.0-1.0 | How close to the lose line (0=safe, 1=critical) |
| `highest_fruit_y` | float | Y position of topmost fruit |
| `distance_to_lose_line` | float | Pixels between highest fruit and lose line |

**Why useful for ML:**
- **Explicit risk signal:** Models can learn "safe mode" vs "aggressive mode"
- **Reward shaping:** Penalize danger_level approaching 1.0
- **Policy switching:** Train separate policies for safe/risky situations

**Neural network usage:**
- Use `danger_level` as a gating signal for different network branches
- Multiply action logicits by `(1 - danger_level)` to be more conservative when risky
- Include in value network to predict expected returns based on danger

---

## Category: Stack Quality

**Purpose:** Understand how well-organized the board is.

| Observation | Type | What It Measures |
|-------------|------|------------------|
| `packing_efficiency` | 0.0-1.0 | Ratio of fruit area to convex hull area* (1.0 = tight, 0.5 = loose) |

_*Convex hull = smallest shape enclosing all fruits (like a rubber band wrapped around them)_
| `surface_roughness` | float | Standard deviation of surface heights (0 = flat) |
| `island_count` | int | Number of disconnected fruit clusters (1 = good) |
| `buried_count` | int | Fruits trapped under others (can't merge) |
| `neighbor_discord` | float | Avg type difference between touching fruits (0 = sorted) |

**Why useful for ML:**
- **Quality metrics:** Learn to optimize for tight packing and low discord
- **Board state features:** Compress complex geometry into simple numbers
- **Auxiliary losses:** Train model to predict these as auxiliary tasks

**Neural network usage:**
- Include in critic network to estimate long-term value
- Use as auxiliary prediction targets (multi-task learning improves representations)
- Concatenate with other features in policy network
- Weight different strategies based on these values (e.g., be careful if `island_count > 1`)

---

## Category: Per-Fruit Details

**Purpose:** Full physics state for advanced reasoning.

| Observation | Shape | What It Contains |
|-------------|-------|------------------|
| `obj_type_id` | (200,) | Fruit type for each slot (-1 = empty) |
| `obj_x`, `obj_y` | (200,) each | Position of each fruit |
| `obj_vx`, `obj_vy` | (200,) each | Velocity of each fruit |
| `obj_ang`, `obj_ang_vel` | (200,) each | Rotation and spin |
| `obj_mask` | (200,) | Boolean mask for valid fruits |

**Why useful for ML:**
- **Full observability:** Everything the physics engine knows
- **Velocity awareness:** Predict where fruits will land
- **Flexible architectures:** Set-based models, transformers, graph networks

**Neural network usage:**
- **Transformers:** Treat each fruit as a token with position/velocity/type features
- **Graph Neural Networks:** Build collision graphs, pass messages between nearby fruits
- **RNNs/LSTMs:** Process fruits sequentially (sorted by Y or type)
- **PointNet-style:** Permutation-invariant networks over the set of fruits

---

## Category: Strategic Insights

**Purpose:** Pre-computed features to accelerate learning.

| Observation | Type | What It Tells You |
|-------------|------|-------------------|
| `largest_fruit_type_id` | int | ID of biggest fruit on board |
| `largest_fruit_x`, `largest_fruit_y` | float | Where the biggest fruit is |
| `board_width`, `board_height` | float | Board dimensions (constants) |
| `lose_line_y` | float | Y coordinate of the lose line (constant) |

**Why useful for ML:**
- **Simplify heuristics:** Easy to implement "drop near largest" strategy
- **Normalization:** Use board dimensions to normalize coordinates
- **Curriculum learning:** Start with simple policies using these hints

**Neural network usage:**
- Quick baseline: Simple MLP with just these + height_map can work surprisingly well
- Attention mechanisms: Use largest fruit position as a query
- Hierarchical policies: High-level policy chooses "merge largest" vs "find safe spot", low-level executes

---

# Part II: Technical Reference

## Environment Setup

### Creating an Environment

```python
from DO_NOT_MODIFY.suika_core import SuikaEnv

# Basic headless environment (fastest)
env = SuikaEnv()

# With debug output (prints merges, scores)
env = SuikaEnv(debug=True)

# With RGB image observations (adds 'rgb' key to obs dict)
env = SuikaEnv(image_obs=True, image_width=270, image_height=400)

# Human-viewable window (for watching your agent)
env = SuikaEnv(render_mode="human")

# Fully specified
env = SuikaEnv(
    config_path=None,           # Use default config
    render_mode="rgb_array",    # Return images but no window
    render_style="full",        # Use sprites (vs "solid" colored circles)
    image_obs=True,             # Include image in observation dict
    image_width=270,            # Image width in pixels
    image_height=400,           # Image height in pixels
    debug=False                 # Silent operation
)
```

### Environment Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_path` | str | None | Path to `game_config.yaml` (uses default if None) |
| `render_mode` | str | None | `None` (no rendering), `"human"` (window), `"rgb_array"` (images) |
| `render_style` | str | `"solid"` | `"solid"` (colored circles) or `"full"` (sprites) |
| `image_obs` | bool | False | If True, adds `"rgb"` key to observation dict |
| `image_width` | int | 270 | Width of RGB observation images |
| `image_height` | int | 400 | Height of RGB observation images |
| `debug` | bool | False | If True, prints game events (merges, scores) |

### Standard Gymnasium API

```python
# Reset to start episode
obs, info = env.reset(seed=42)  # Seed is optional but it should be changed for each episode for the ai to be robust

# Take action
action = 0.0  # Float in [-1, 1]
obs, reward, terminated, truncated, info = env.step(action)

# reward is ALWAYS 0.0 - compute your own!
# terminated = True if game over (lose condition)
# truncated = True if drop cap reached

# Check episode end
done = terminated or truncated

# Environment properties
env.action_space  # Box(low=-1.0, high=1.0, shape=(), dtype=float32)
env.observation_space  # Dict of spaces (see below)
```

---

## Action Space

**Type:** `gym.spaces.Box(low=-1.0, high=1.0, shape=(), dtype=np.float32)`

**Your agent returns a single float** representing the spawner X position.

| Action Value | Physical Meaning |
|--------------|------------------|
| `-1.0` | Drop at leftmost valid position |
| `0.0` | Drop at center of board |
| `+1.0` | Drop at rightmost valid position |

**Important:** The action is automatically clamped to prevent spawning inside walls. The actual spawn X accounts for the current fruit's radius.

**Formula:**
```
spawn_x = (action + 1) / 2 * (board_width - 2 * fruit_radius) + fruit_radius
```

**Example:**
```python
# Drop at center
action = 0.0

# Drop 30% to the right of center
action = 0.3

# Convert world X back to action
def world_x_to_action(x, board_width=450):
    return (x / board_width) * 2 - 1
```

---

## Observation Space (Detailed)

All observations are returned as a **dictionary of NumPy arrays**.

```python
obs, info = env.reset()
# obs is a dict: {"score": np.int64(0), "obj_x": np.array([...]), ...}
```

### Scalar Observations

| Key | Type | Shape | Range | Description |
|-----|------|-------|-------|-------------|
| `spawner_x` | float32 | () | [-1, 1] | Current spawner action (where next fruit will drop) |
| `current_fruit_id` | int32 | () | [0, 10] | Type ID of fruit about to be dropped |
| `next_fruit_id` | int32 | () | [0, 10] | Type ID of fruit coming after current |
| `score` | int64 | () | [0, ∞) | Current game score |
| `drops_used` | int32 | () | [0, max_drops] | Number of fruits dropped so far |
| `objects_count` | int32 | () | [0, 200] | Number of fruits currently on board |

**Fruit ID mapping:**
```
0: Cherry      4: Persimmon   8: Pineapple
1: Strawberry  5: Apple       9: Honeydew
2: Grape       6: Pear       10: Melon (largest)
3: Dekopon     7: Peach
```

**Usage:**
```python
current = int(obs["current_fruit_id"])
next_fruit = int(obs["next_fruit_id"])

# Plan ahead
if current <= 2 and next_fruit <= 2:
    # Two small fruits in a row - can afford to stack
    pass
```

---

### Board Constants

| Key | Type | Shape | Value | Description |
|-----|------|-------|-------|-------------|
| `board_width` | float32 | () | 450.0 | Board width in pixels |
| `board_height` | float32 | () | 600.0 | Board height in pixels |
| `lose_line_y` | float32 | () | 540.0 | Y coordinate of lose line |

**Why included:** Makes it easy to normalize coordinates without hardcoding values.

**Usage:**
```python
# Normalize X position to [0, 1]
normalized_x = obs["obj_x"] / obs["board_width"]

# Normalize Y position relative to lose line
danger_ratio = obs["obj_y"] / obs["lose_line_y"]
```

---

### Danger Metrics

| Key | Type | Shape | Range | Description |
|-----|------|-------|-------|-------------|
| `danger_level` | float32 | () | [0.0, 1.0+] | Highest fruit Y / lose_line_y |
| `highest_fruit_y` | float32 | () | [0.0, ∞) | Y position of topmost fruit |
| `distance_to_lose_line` | float32 | () | (-∞, ∞) | `lose_line_y - highest_fruit_y` (negative = above line) |

**Interpretation:**

`danger_level`:
- `0.0`: Board is empty
- `0.5`: Stack is halfway to lose line (safe)
- `0.8`: Getting risky
- `1.0`: Fruit is AT the lose line
- `>1.0`: Fruit is ABOVE lose line (will lose soon if stays there)

`distance_to_lose_line`:
- Positive: Safe (e.g., 100 = 100px below line)
- Zero: Touching the line
- Negative: Above the line (danger!)

**Usage:**
```python
if obs["danger_level"] > 0.85:
    # Emergency mode - find absolute lowest spot
    height_map = obs["height_map"]
    safest_idx = np.argmin(height_map)
    action = (safest_idx / 19) * 2 - 1
```

---

### Spatial Features

| Key | Type | Shape | Range | Description |
|-----|------|-------|-------|-------------|
| `center_of_mass_x` | float32 | () | [0, board_width] | X coordinate of center of mass |
| `center_of_mass_y` | float32 | () | [0, ∞) | Y coordinate of center of mass |
| `largest_fruit_type_id` | int32 | () | [-1, 10] | Type ID of largest fruit (-1 if empty) |
| `largest_fruit_x` | float32 | () | [0, board_width] | X position of largest fruit (0 if empty) |
| `largest_fruit_y` | float32 | () | [0, ∞) | Y position of largest fruit (0 if empty) |

**Why center of mass matters:**
- High Y = tall stack (unstable)
- X far from center = lopsided stack (may topple)

**Usage:**
```python
com_x = obs["center_of_mass_x"]
com_y = obs["center_of_mass_y"]

# Check if stack is lopsided
if com_x < obs["board_width"] * 0.3:
    # Leaning left - balance by dropping right
    action = 0.5
elif com_x > obs["board_width"] * 0.7:
    # Leaning right - balance by dropping left
    action = -0.5
```

---

### Stack Quality Metrics

| Key | Type | Shape | Range | Description |
|-----|------|-------|-------|-------------|
| `packing_efficiency` | float32 | () | [0.0, 1.0] | Fruit area / convex hull area |
| `surface_roughness` | float32 | () | [0.0, ∞) | Std dev of surface heights |
| `island_count` | int32 | () | [0, ∞) | Number of disconnected clusters |
| `buried_count` | int32 | () | [0, ∞) | Fruits with neighbors on 3+ sides |
| `neighbor_discord` | float32 | () | [0.0, ∞) | Avg |type_diff| between touching fruits |

**Detailed explanations:**

**`packing_efficiency`:**
- Computed as: `total_fruit_area / convex_hull_area`
- 1.0 = perfect packing (no gaps)
- 0.5 = 50% air gaps (loose, unstable)
- Lower values mean fruits may shift unpredictably

**What is a convex hull?**

The convex hull is the smallest convex shape that encloses all the fruits. Imagine wrapping a rubber band around all the fruit centers - the area inside that rubber band is the convex hull area.

```
Example 1: Tight packing (high efficiency ~0.85)
    
    Convex hull boundary (rubber band)
    ╭──────────────────────╮
    │  ●●●●●●●●●●●         │
    │ ●●●●●●●●●●●●         │
    │ ●●●●●●●●●●●          │
    │  ●●●●●●●●            │
    ╰──────────────────────╯
    
    Most of the hull area is filled with fruits!

Example 2: Loose packing (low efficiency ~0.45)
    
    Convex hull boundary
    ╭──────────────────────╮
    │ ●●                   │
    │   ●      ●●          │
    │                      │
    │          ●●          │
    │               ●●     │
    ╰──────────────────────╯
    
    Lots of empty space inside the hull!

Example 3: Tower (medium efficiency ~0.60)
    
    Convex hull
    ╭──────╮
    │  ●●  │
    │  ●●  │
    │ ●●●  │
    │ ●●●● │
    │ ●●●● │
    ╰──────╯
    
    Some wasted space at the edges
```

**Why it matters:**
- **High efficiency (0.8+):** Fruits are tightly packed, stable, predictable
- **Low efficiency (< 0.6):** Lots of gaps, fruits can shift and cause chain reactions
- Helps detect "loose" stacks that might collapse unpredictably

**`surface_roughness`:**
- Standard deviation of the Y positions of "surface" fruits
- 0.0 = perfectly flat top
- High values = jagged surface with peaks/valleys
- Flat surfaces are easier to stack on

**`island_count`:**
- Uses Union-Find to count connected components
- 1 = all fruits touching (good)
- 2+ = separate piles (hard to merge across gaps)

**`buried_count`:**
- Counts fruits with obstacles on left, right, AND top
- These fruits can't be easily merged
- Especially bad if small fruits are buried (wasted potential)

**`neighbor_discord`:**
- For each pair of touching fruits, compute |typeA - typeB|
- Average over all contacts
- 0.0 = identical types touching (ideal for merging)
- 1.0 = adjacent types (good)
- 5.0+ = chaotic board (e.g., cherry touching melon)

**Usage:**
```python
# Assess board quality
if obs["packing_efficiency"] < 0.6 and obs["neighbor_discord"] > 3.0:
    # Messy board - play conservatively
    action = 0.0  # Drop center
```

---

### Height Map

| Key | Type | Shape | Range | Description |
|-----|------|-------|-------|-------------|
| `height_map` | float32 | (20,) | [0.0, ∞) | Max Y at each horizontal slice |

**What it is:**
- Board is divided into 20 vertical slices (columns)
- Each value is the maximum Y position of any fruit in that slice
- Think of it as a 1D "lidar scan" of the surface

**Slice mapping:**
```
slice_idx = 0  → X in [0, 22.5]
slice_idx = 1  → X in [22.5, 45]
...
slice_idx = 19 → X in [427.5, 450]
```

**Usage:**
```python
height_map = obs["height_map"]  # Shape: (20,)

# Find the lowest column
lowest_idx = np.argmin(height_map)  # 0-19
action = (lowest_idx / 19) * 2 - 1  # Convert to [-1, 1]

# Find the highest column
highest_idx = np.argmax(height_map)

# Detect flat regions
diffs = np.abs(np.diff(height_map))
flatness = 1.0 / (diffs.mean() + 1e-6)

# Visual debug
def visualize_height_map(hm):
    max_h = hm.max()
    normalized = (hm / (max_h + 1e-6) * 8).astype(int)
    chars = " ▁▂▃▄▅▆▇█"
    return "".join(chars[min(h, 8)] for h in normalized)

print(visualize_height_map(obs["height_map"]))
# Output: "▁▁▂▃▅▇██▇▆▅▄▃▂▂▁▁▁"
```

---

### Object Arrays

**All object arrays have shape `(200,)` and use a mask for valid entries.**

| Key | Type | Shape | Range | Description |
|-----|------|-------|-------|-------------|
| `obj_mask` | bool | (200,) | {False, True} | True = valid fruit at this index |
| `obj_type_id` | int16 | (200,) | [-1, 10] | Fruit type (-1 for invalid) |
| `obj_x` | float32 | (200,) | [0, board_width] | X position in world coords |
| `obj_y` | float32 | (200,) | [0, ∞) | Y position in world coords |
| `obj_vx` | float32 | (200,) | (-∞, ∞) | X velocity (pixels/sec) |
| `obj_vy` | float32 | (200,) | (-∞, ∞) | Y velocity (pixels/sec) |
| `obj_ang` | float32 | (200,) | [0, 2π) | Rotation angle in radians |
| `obj_ang_vel` | float32 | (200,) | (-∞, ∞) | Angular velocity (rad/sec) |
| `obj_radius` | float32 | (200,) | [0, ∞) | Visual radius in pixels |

**CRITICAL:** Always filter with `obj_mask` before using these arrays!

**Usage patterns:**

```python
# Pattern 1: Filter to valid objects
mask = obs["obj_mask"]
valid_count = mask.sum()

types = obs["obj_type_id"][mask]      # Shape: (valid_count,)
xs = obs["obj_x"][mask]
ys = obs["obj_y"][mask]
vxs = obs["obj_vx"][mask]
vys = obs["obj_vy"][mask]

# Pattern 2: Find specific fruit types
cherry_mask = (obs["obj_type_id"] == 0) & obs["obj_mask"]
cherry_positions = obs["obj_x"][cherry_mask], obs["obj_y"][cherry_mask]

# Pattern 3: Find nearest fruit to a position
target_x = 225.0
if mask.any():
    distances = np.abs(obs["obj_x"][mask] - target_x)
    nearest_idx = np.argmin(distances)
    nearest_x = obs["obj_x"][mask][nearest_idx]

# Pattern 4: Check if fruits are moving
moving_mask = mask & ((np.abs(obs["obj_vx"]) > 1.0) | (np.abs(obs["obj_vy"]) > 1.0))
num_moving = moving_mask.sum()

# Pattern 5: Neural network input
# Option A: Pad/mask
features = np.stack([
    obs["obj_x"],
    obs["obj_y"],
    obs["obj_type_id"],
    obs["obj_mask"].astype(np.float32)
], axis=1)  # Shape: (200, 4)

# Option B: Variable length (list of valid fruits)
valid_indices = np.where(mask)[0]
fruit_list = [{
    "type": obs["obj_type_id"][i],
    "x": obs["obj_x"][i],
    "y": obs["obj_y"][i],
    "vx": obs["obj_vx"][i],
    "vy": obs["obj_vy"][i],
} for i in valid_indices]
```

---

## Info Dictionary

Returned alongside observations from `env.step()`.

```python
obs, reward, terminated, truncated, info = env.step(action)
```

| Key | Type | Description |
|-----|------|-------------|
| `score` | int | Current total score (same as `obs["score"]`) |
| `delta_score` | int | Points earned THIS step |
| `drops_used` | int | Total fruits dropped |
| `merges` | int | Number of merges that occurred this step |
| `sim_time` | float | Physics simulation time (seconds) |
| `termination_reason` | str | `""` (not done), `"lose_line"` (game over), or `"drop_cap"` (max drops) |

**Usage for custom rewards:**

```python
# Simple score-based
reward = info["delta_score"] / 10.0

# Survival bonus
reward = 0.01 if not terminated else -1.0

# Merge-focused
reward = info["merges"] * 0.5

# Combined
reward = info["delta_score"] / 10.0 + info["merges"] * 0.1
if terminated:
    reward -= 5.0

# Time penalty (encourage faster play)
reward = info["delta_score"] - info["sim_time"] * 0.1
```

---

## Domain Randomization

**To prevent overfitting, physics parameters are slightly randomized each episode.**

| Parameter | Variation | Affects |
|-----------|-----------|---------|
| Friction | ±2% | How quickly fruits stop sliding |
| Elasticity | ±2% | How much fruits bounce on collision |
| Mass (per fruit) | ±4% | Weight and collision response |
| Gravity | ±1% | Falling speed |

**Key points:**
- Same seed → same randomization (deterministic)
- AI agents **cannot observe** these values
- Forces robust policies that generalize
- Mimics real-world physics uncertainty

**Example:** With default gravity=980, an episode might use 990.8 (+1.1%) or 970.2 (-1%).

**Implementation:**
```python
# Randomization is automatic on reset
obs, info = env.reset(seed=42)
# Physics params are now varied based on seed 42

# Same seed gives same physics
obs1, _ = env.reset(seed=123)
obs2, _ = env.reset(seed=123)
# obs1 and obs2 will have identical physics parameters
```

---

## Evaluation

### Local Development

Test against public seeds:
```bash
python -m DO_NOT_MODIFY.evaluation.run_eval --agent contestants/my_team
```

Public seeds are in `DO_NOT_MODIFY/evaluation/seed_bank.json`.

### Competition Scoring

**The final competition uses HIDDEN evaluation seeds.**

- You will not see these seeds until after submission
- They are chosen to be diverse and challenging
- Your agent must generalize, not memorize
- Domain randomization prepares you for this

**Final score:**
```
Score = Mean(agent_score(seed_1), agent_score(seed_2), ..., agent_score(seed_N))
```

Higher average score wins!

---

## Coordinate System

```
                     spawn_y (570)
                         ▲
┌────────────────────────┼────────────────────────┐
│                        │                        │
│                   lose_line_y (540)             │ ◄─ Lose if fruit stays
│ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ - │    above for >1 sec
│                        │                        │
│                        │                        │
│                        │                        │
│         GAME           │          AREA          │
│                        │                        │
│                        │                        │
│          Y             │                        │
│          ▲             │                        │
│          │             │                        │
│          │             │                        │
│          └─────────────┼──────────► X           │
│        (0,0)           │                        │
│                        │                        │
└────────────────────────┼────────────────────────┘
X=0                      │                    X=board_width (450)
```

**Key facts:**
- Origin `(0, 0)` is **bottom-left**
- Y increases **upward**
- Gravity pulls **down** (negative Y direction)
- `spawn_y` (570) is where fruits appear
- `lose_line_y` (540) is the danger threshold
- Board width = 450px
- Board height = 600px

**Normalization helpers:**
```python
def normalize_x(world_x, obs):
    """World X to [0, 1]"""
    return world_x / obs["board_width"]

def normalize_y(world_y, obs):
    """World Y to [0, 1] relative to lose line"""
    return world_y / obs["lose_line_y"]

def world_to_action(world_x, obs):
    """World X to action [-1, 1]"""
    return (world_x / obs["board_width"]) * 2 - 1

def action_to_world(action, obs):
    """Action [-1, 1] to world X"""
    return ((action + 1) / 2) * obs["board_width"]
```

---

## Quick Reference Card

```python
# Create environment
env = SuikaEnv()

# Reset
obs, info = env.reset(seed=42)

# Essential observations
current_fruit = int(obs["current_fruit_id"])  # 0-10
danger = float(obs["danger_level"])           # 0.0-1.0
height_map = obs["height_map"]                # (20,) surface profile

# Get valid fruits
mask = obs["obj_mask"]
xs = obs["obj_x"][mask]
ys = obs["obj_y"][mask]
types = obs["obj_type_id"][mask]

# Take action
action = 0.0  # Float in [-1, 1]
obs, reward, terminated, truncated, info = env.step(action)

# Custom reward (reward is always 0.0 by default)
my_reward = info["delta_score"] / 10.0

# Check episode end
done = terminated or truncated
```

---

## Advanced Tips

### For Dense Neural Networks (MLPs)

Recommended input features:
```python
features = np.concatenate([
    [obs["current_fruit_id"] / 10.0],        # Normalize
    [obs["next_fruit_id"] / 10.0],
    [obs["danger_level"]],
    [obs["packing_efficiency"]],
    [obs["surface_roughness"] / 100.0],
    [obs["neighbor_discord"] / 5.0],
    obs["height_map"] / 600.0,               # 20 values
    [obs["center_of_mass_x"] / 450.0],
    [obs["center_of_mass_y"] / 600.0],
    # Total: ~26 features
])
```

### For Convolutional Networks

```python
# Treat height_map as a 1D "image"
height_map = obs["height_map"].reshape(1, 20, 1)  # (batch, width, channels)

# Or create a 2D occupancy grid
grid = np.zeros((60, 45))  # 10px bins
for i in range(len(obs["obj_mask"])):
    if obs["obj_mask"][i]:
        x_bin = int(obs["obj_x"][i] / 10)
        y_bin = int(obs["obj_y"][i] / 10)
        grid[y_bin, x_bin] = obs["obj_type_id"][i] + 1
```

### For Transformers

```python
# Each fruit is a token
mask = obs["obj_mask"]
tokens = []
for i in range(200):
    if mask[i]:
        tokens.append([
            obs["obj_type_id"][i] / 10.0,
            obs["obj_x"][i] / 450.0,
            obs["obj_y"][i] / 600.0,
            obs["obj_vx"][i] / 100.0,
            obs["obj_vy"][i] / 100.0,
        ])

# Pad to fixed length
while len(tokens) < 50:
    tokens.append([0, 0, 0, 0, 0])
tokens = np.array(tokens[:50])  # Shape: (50, 5)
```

### For Recurrent Networks (RNNs/LSTMs)

```python
# Process fruits sequentially (e.g., sorted by Y)
mask = obs["obj_mask"]
indices = np.where(mask)[0]
sorted_indices = indices[np.argsort(obs["obj_y"][indices])]  # Bottom to top

sequence = []
for i in sorted_indices:
    sequence.append([
        obs["obj_type_id"][i],
        obs["obj_x"][i],
        obs["obj_y"][i],
    ])
```

---

**End of API Reference**