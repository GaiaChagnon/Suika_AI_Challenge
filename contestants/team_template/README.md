# Team Template - Getting Started Guide

This template provides everything you need to build and evaluate an AI agent for the Suika Competition.

**See `API_REFERENCE.md` for complete observation space documentation.**

---

## Quick Start

1. **Copy this folder** to create your team's submission:
   ```bash
   cp -r contestants/team_template contestants/my_team
   ```

2. **Edit `agent.py`** with your strategy

3. **Test locally**:
   ```bash
   python -c "
   from DO_NOT_MODIFY.suika_core import SuikaEnv
   from contestants.my_team.agent import SuikaAgent
   
   env = SuikaEnv(debug=True)
   agent = SuikaAgent()
   
   obs, info = env.reset(seed=42)
   for _ in range(100):
       action = agent.act(obs)
       obs, reward, terminated, truncated, info = env.step(action)
       if terminated or truncated:
           break
   print(f'Final score: {info[\"score\"]}')
   "
   ```

4. **Evaluate on public seeds**:
   ```bash
   python -m DO_NOT_MODIFY.evaluation.run_eval --agent contestants/my_team --verbose
   ```

---

## Files in This Template

| File | Purpose |
|------|---------|
| `agent.py` | Your agent implementation (edit this!) |
| `API_REFERENCE.md` | Complete observation/action space docs |
| `README.md` | This guide |

---

## The Game Loop

```
┌─────────────────────────────────────────────────────────┐
│                    GAME LOOP                            │
│                                                         │
│  1. Agent receives observation (game state)             │
│  2. Agent returns action in [-1, 1] (spawn position)    │
│  3. Environment spawns fruit, simulates physics         │
│  4. Fruits merge when same-type collide                 │
│  5. Environment returns new observation + info          │
│  6. Repeat until game over                              │
│                                                         │
│  Game ends when:                                        │
│  - A fruit stays above lose_line for >1 second          │
│  - Maximum drops reached (10,000)                       │
└─────────────────────────────────────────────────────────┘
```

---

## Key Observations (Quick Reference)

See `API_REFERENCE.md` for complete documentation.

```python
def act(self, obs):
    # Most important fields
    current_fruit = int(obs["current_fruit_id"])  # 0-10
    danger = float(obs["danger_level"])           # 0=safe, 1=dying
    height_map = obs["height_map"]                # (20,) surface heights
    
    # Get valid fruit positions
    mask = obs["obj_mask"].astype(bool)
    if mask.any():
        xs = obs["obj_x"][mask]
        ys = obs["obj_y"][mask]
        types = obs["obj_type_id"][mask]
    
    # Your decision here
    return action  # float in [-1, 1]
```

---

## Training with Vectorized Environments

### Option 1: AsyncVectorEnv (Recommended - True Multiprocessing)

Uses multiple CPU cores for parallel physics simulation:

```python
from DO_NOT_MODIFY.suika_core import make_async_vec_env, get_recommended_num_envs
import numpy as np

# Defaults to CPU core count
num_envs = get_recommended_num_envs()  # e.g., 8 or 12
vec_env = make_async_vec_env(num_envs=num_envs, seed=42)

obs, infos = vec_env.reset()

for step in range(100000):
    actions = your_agent.batch_act(obs)  # Shape: (num_envs,)
    obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
    
    # Auto-resets terminated envs. Check final_info for episode stats.
    for i, (term, trunc) in enumerate(zip(terminateds, truncateds)):
        if term or trunc:
            final_score = infos["final_info"][i].get("score", 0)
            print(f"Episode {i} ended with score {final_score}")

vec_env.close()
```

### Option 2: SuikaVectorEnv (Single-Process)

Simpler but runs on a single thread:

```python
from DO_NOT_MODIFY.suika_core import SuikaVectorEnv

vec_env = SuikaVectorEnv(num_envs=128, seed=42)
obs, _ = vec_env.reset()

for step in range(100000):
    actions = your_agent.batch_act(obs)
    obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
    
    done_mask = terminateds | truncateds
    if done_mask.any():
        done_indices = np.where(done_mask)[0].tolist()
        vec_env.reset(env_indices=done_indices)
```

**Performance Tip:** Use `make_async_vec_env` for training. Use `SuikaVectorEnv` for debugging (simpler).

---

## Custom Rewards

The environment returns `reward=0.0` always. Design your own:

```python
# Examples
reward = info["delta_score"] / 100.0              # Scaled score
reward = -10.0 if terminated else info["delta_score"] / 100.0  # Death penalty
reward = info["merges"] * 1.0                     # Merge count
```

---

## Domain Randomization

**Physics varies slightly each episode** to encourage robust agents:

| Parameter | Variance |
|-----------|----------|
| Friction | ±2% |
| Elasticity | ±2% |
| Mass (per fruit) | ±4% |
| Gravity | ±1% |

- Same seed = same randomization
- AI agents **cannot see** these values
- Your agent should generalize, not overfit

---

## Evaluation

### Public Seeds (for development)

The `seed_bank.json` contains seeds you can test against:

```bash
python -m DO_NOT_MODIFY.evaluation.run_eval --agent contestants/my_team
```

### ⚠️ Hidden Evaluation Seeds

**The final competition uses a SEPARATE set of hidden seeds.**

- You cannot see or test against these seeds
- They are chosen to be diverse and challenging
- Your agent must **generalize**, not memorize
- Domain randomization helps prepare for this

### Scoring

```
Final Score = Mean score across all evaluation seeds
```

---

