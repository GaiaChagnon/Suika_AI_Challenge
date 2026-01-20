# Baseline LIDAR Agent

A simple heuristic agent that serves as a benchmark for the Suika competition.

## Strategy

The agent uses a straightforward approach:

1. **Read the height map** - Use the pre-computed `height_map` observation (20-column LIDAR scan of the surface)
2. **Check danger level** - The `danger_level` observation indicates how close to losing (0.0 = safe, 1.0 = critical)
3. **Find the lowest column** - Use `np.argmin(height_map)` to find where the stack is shortest
4. **Drop there** - Convert the column index to an action with small noise (less noise when danger is high)

This is a simple but effective baseline that focuses on building a stable, low stack.

## Files

| File | Description |
|------|-------------|
| `agent.py` | The agent implementation - **study this as an example!** |
| `benchmark.py` | Performance benchmarking and statistics collection |
| `replay.json` | Saved replay of one episode (created by benchmark) |
| `README.md` | This file |

## Usage

### Run the Benchmark

```bash
# Default: 100 envs × 5 runs = 500 episodes
python -m contestants.baseline_lidar.benchmark

# Custom configuration
python -m contestants.baseline_lidar.benchmark --num-envs 50 --num-runs 3

# With verbose output (shows each episode)
python -m contestants.baseline_lidar.benchmark --verbose

# Skip replay saving (faster)
python -m contestants.baseline_lidar.benchmark --no-replay
```

### View the Saved Replay

After running the benchmark, a replay of one episode is saved. View it with:

```bash
python tools/replay_viewer.py contestants/baseline_lidar/replay.json
```

### Use the Agent Directly

```python
from contestants.baseline_lidar import SuikaAgent
from DO_NOT_MODIFY.suika_core import SuikaEnv

# Create environment and agent
env = SuikaEnv()
agent = SuikaAgent(debug=True)

# Run one episode
obs, info = env.reset(seed=42)
agent.reset(seed=42)

done = False
while not done:
    action = agent.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"Final score: {info['score']}")
```

## Benchmark Output

The benchmark provides comprehensive statistics:

```
SCORE STATISTICS
----------------------------------------
  Count:       500
  Min:         234
  Max:        1847
  Mean:        756
  Median:      712
  Std:         298
  P25:         523
  P75:         945
  P90:        1156
  P95:        1298

PERFORMANCE METRICS
----------------------------------------
  Total benchmark time:   45.23 seconds
  Total episodes:         500
  Avg steps/second:       892.4
  Peak memory:            156.3 MB
  Avg CPU usage:          78.2%
```

## Performance Notes

The benchmark now uses **AsyncVectorEnv** with true multiprocessing - each environment runs in its own Python process on separate CPU cores. This bypasses the GIL.

Expected throughput: **40-120 steps/second** (varies by hardware and stack complexity).
- Early game: ~100+ steps/sec (simple physics with few fruits)
- Late game: ~40-60 steps/sec (complex physics with many fruits)

Default: `num_envs = CPU core count` (optimal for your machine).

For debugging (single-threaded):
```bash
python -m contestants.baseline_lidar.benchmark --sync
```

## Expected Performance

Typical results for the baseline agent:

| Metric | Value |
|--------|-------|
| Mean Score | ~1700-1800 |
| Median Score | ~1500-2000 |
| Best Score | ~2000-3000 |
| Worst Score | ~600-700 |

Your agent should aim to beat these numbers!

## Learning from This Example

Study `agent.py` to see:

1. **How to read observations** - The agent uses `observation["height_map"]`
2. **How to return actions** - Returns a float in `[-1, 1]`
3. **How to structure an agent** - Clean `__init__`, `reset`, and `act` methods
4. **How to add debug output** - Optional printing for development

The agent is intentionally simple so you can understand it quickly and start building something better!

## Improving on the Baseline

Ideas to beat this baseline:

1. **Consider fruit types** - Different fruits have different sizes
2. **Plan ahead** - Look at `next_fruit_id` to prepare
3. **Watch for merges** - Drop matching fruits near each other
4. **Monitor danger** - Use `danger_level` to switch strategies when risky
5. **Use machine learning** - Train a neural network policy

Good luck!
