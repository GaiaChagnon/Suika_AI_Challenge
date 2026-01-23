#!/usr/bin/env python3
"""
Benchmark script for the Baseline LIDAR Agent.

Runs the agent across multiple parallel environments and collects
comprehensive statistics about performance, timing, and resource usage.

Also saves a replay of one episode for visualization.

Usage:
    python -m contestants.baseline_lidar.benchmark [options]
    
Options:
    --num-envs      Number of parallel environments (default: CPU cores)
    --num-runs      Number of benchmark runs (default: 5)
    --seed          Base random seed (default: 42)
    --verbose       Print per-episode details
    --no-replay     Don't save replay
    --sync          Use single-threaded mode (for debugging)

Example:
    python -m contestants.baseline_lidar.benchmark --num-envs 8 --num-runs 5

To view the saved replay:
    python tools/replay_viewer.py contestants/baseline_lidar/replay.json
"""

import argparse
import json
import multiprocessing
import time
import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# Optional dependency for resource monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import the agent
from .agent import SuikaAgent

# Import environment - add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from DO_NOT_MODIFY.suika_core import SuikaEnv
from DO_NOT_MODIFY.suika_core.async_vector_env import make_async_vec_env, get_recommended_num_envs


# Videos directory (auto-created)
VIDEOS_DIR = Path(__file__).parent / "videos"


@dataclass
class RunMetrics:
    """Metrics collected from a single benchmark run."""
    scores: List[int] = field(default_factory=list)
    drops_per_episode: List[int] = field(default_factory=list)
    episode_times: List[float] = field(default_factory=list)
    steps_per_second: float = 0.0
    total_steps: int = 0
    total_time: float = 0.0
    memory_peak_mb: float = 0.0


@dataclass
class BenchmarkResults:
    """Aggregated results from all benchmark runs."""
    all_scores: List[int] = field(default_factory=list)
    run_metrics: List[RunMetrics] = field(default_factory=list)
    total_time: float = 0.0
    avg_steps_per_second: float = 0.0
    peak_memory_mb: float = 0.0


def save_video(seed: int = 42, fps: int = 30) -> Tuple[Dict[str, Any], Path]:
    """
    Play one episode and save it as an MP4 video.
    
    Records actual rendered frames - deterministic regardless of domain randomization.
    Saves to the videos/ subdirectory with format: baseline_lidar_{timestamp}_s{seed}.mp4
    
    Args:
        seed: Random seed for the episode.
        fps: Frames per second for the video.
        
    Returns:
        Tuple of (episode_info, saved_path).
    """
    from DO_NOT_MODIFY.suika_core.video_recorder import VideoRecorder
    
    # Ensure videos directory exists
    VIDEOS_DIR.mkdir(exist_ok=True)
    
    env = SuikaEnv()
    recorder = VideoRecorder(fps=fps, width=440, height=550)
    agent = SuikaAgent()
    
    obs, info = env.reset(seed=seed)
    agent.reset(seed=seed)
    
    # Start recording
    saved_path = recorder.start(
        env_or_game=env,
        agent_name="baseline_lidar",
        seed=seed,
        directory=VIDEOS_DIR
    )
    
    # Capture initial frame
    recorder.capture_frame(env)
    
    done = False
    steps = 0
    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        
        # Capture frame after each step
        recorder.capture_frame(env)
    
    recorder.stop()
    env.close()
    
    episode_info = {
        "seed": seed,
        "final_score": info.get("score", 0),
        "drops": steps,
        "terminated_reason": info.get("terminated_reason", "unknown"),
    }
    
    return episode_info, saved_path


def run_single_benchmark(
    num_envs: int,
    seed: int,
    verbose: bool = False,
    use_sync: bool = False
) -> RunMetrics:
    """
    Run a single benchmark with the specified number of environments.
    
    Uses AsyncVectorEnv for true multiprocessing across CPU cores.
    """
    metrics = RunMetrics()
    process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None
    
    # Create async vector environment (true multiprocessing)
    vec_env = make_async_vec_env(
        num_envs=num_envs,
        seed=seed,
        use_sync=use_sync,
    )
    
    # Create one agent per environment
    agents = [SuikaAgent() for _ in range(num_envs)]
    for i, agent in enumerate(agents):
        agent.reset(seed=seed + i)
    
    # Reset all environments
    obs, infos = vec_env.reset()
    
    # Track per-environment stats
    env_scores = np.zeros(num_envs, dtype=np.int64)
    env_drops = np.zeros(num_envs, dtype=np.int32)
    env_start_times = np.full(num_envs, time.time())
    
    total_steps = 0
    start_time = time.time()
    memory_samples = []
    
    # Run until all environments complete
    active = np.ones(num_envs, dtype=bool)
    last_print_time = time.time()
    
    while active.any():
        # Collect actions from all agents
        actions = np.zeros(num_envs, dtype=np.float32)
        for i in range(num_envs):
            if active[i]:
                # Extract observation for this environment
                env_obs = {key: val[i] for key, val in obs.items()}
                actions[i] = agents[i].act(env_obs)
        
        # Step all environments in parallel (across multiple processes)
        obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)
        total_steps += active.sum()
        
        # Update stats for active environments
        # Note: AsyncVectorEnv infos is a dict of lists
        for i in range(num_envs):
            if active[i]:
                env_drops[i] += 1
                # Get score from final_info if terminated, else from infos
                if terminateds[i] or truncateds[i]:
                    if "final_info" in infos and infos["final_info"][i] is not None:
                        env_scores[i] = infos["final_info"][i].get("score", 0)
                    elif "score" in infos:
                        env_scores[i] = infos["score"][i] if hasattr(infos["score"], "__getitem__") else infos["score"]
        
        # Check for completed environments
        dones = terminateds | truncateds
        for i in range(num_envs):
            if active[i] and dones[i]:
                active[i] = False
                episode_time = time.time() - env_start_times[i]
                
                metrics.scores.append(int(env_scores[i]))
                metrics.drops_per_episode.append(int(env_drops[i]))
                metrics.episode_times.append(episode_time)
                
                if verbose:
                    print(f"  Env {i}: Score={env_scores[i]}, "
                          f"Drops={env_drops[i]}, "
                          f"Time={episode_time:.2f}s")
        
        # Progress update every 2 seconds
        current_time = time.time()
        if current_time - last_print_time > 2.0:
            completed = num_envs - active.sum()
            elapsed = current_time - start_time
            print(f"    Progress: {completed}/{num_envs} episodes, "
                  f"{total_steps} steps, "
                  f"{total_steps/elapsed:.0f} steps/sec")
            last_print_time = current_time
        
        # Sample resource usage periodically
        if total_steps % 500 == 0 and process is not None:
            memory_samples.append(process.memory_info().rss / 1024 / 1024)
    
    end_time = time.time()
    
    # Compute run metrics
    metrics.total_time = end_time - start_time
    metrics.total_steps = total_steps
    metrics.steps_per_second = total_steps / metrics.total_time if metrics.total_time > 0 else 0
    metrics.memory_peak_mb = max(memory_samples) if memory_samples else 0
    
    vec_env.close()
    return metrics


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute comprehensive statistics for a list of values."""
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0, "median": 0, "std": 0}
    
    arr = np.array(values)
    return {
        "count": len(arr),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def run_benchmark(
    num_envs: Optional[int] = None,
    num_runs: int = 5,
    seed: int = 42,
    verbose: bool = False,
    save_replay_flag: bool = True,
    use_sync: bool = False
) -> BenchmarkResults:
    """Run the full benchmark suite."""
    results = BenchmarkResults()
    
    # Default to CPU count
    if num_envs is None:
        num_envs = get_recommended_num_envs()
    
    mode = "Sync (single-threaded)" if use_sync else f"Async (multiprocessing, {multiprocessing.cpu_count()} cores)"
    
    print("=" * 70)
    print("BASELINE LIDAR AGENT BENCHMARK")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Parallel environments: {num_envs}")
    print(f"  Number of runs:        {num_runs}")
    print(f"  Total episodes:        {num_envs * num_runs}")
    print(f"  Base seed:             {seed}")
    print(f"  Mode:                  {mode}")
    print("=" * 70)
    print()
    
    total_start = time.time()
    
    for run_idx in range(num_runs):
        run_seed = seed + run_idx * 1000
        print(f"Run {run_idx + 1}/{num_runs} (seed={run_seed})...")
        
        run_metrics = run_single_benchmark(
            num_envs=num_envs,
            seed=run_seed,
            verbose=verbose,
            use_sync=use_sync
        )
        
        results.run_metrics.append(run_metrics)
        results.all_scores.extend(run_metrics.scores)
        
        # Print run summary
        run_stats = compute_statistics(run_metrics.scores)
        print(f"  Completed: {len(run_metrics.scores)} episodes")
        print(f"  Scores: min={run_stats['min']:.0f}, "
              f"max={run_stats['max']:.0f}, "
              f"mean={run_stats['mean']:.0f}")
        print(f"  Speed: {run_metrics.steps_per_second:.1f} steps/sec")
        print(f"  Time: {run_metrics.total_time:.2f}s")
        print()
    
    total_end = time.time()
    results.total_time = total_end - total_start
    
    # Compute aggregate metrics
    all_steps_per_sec = [m.steps_per_second for m in results.run_metrics]
    results.avg_steps_per_second = np.mean(all_steps_per_sec)
    results.peak_memory_mb = max((m.memory_peak_mb for m in results.run_metrics), default=0)
    
    # Save video of one episode AFTER benchmark completes
    if save_replay_flag:
        print("Recording video episode...")
        episode_info, video_path = save_video(seed=seed)
        print(f"  Episode score: {episode_info['final_score']}")
        print(f"  Episode drops: {episode_info['drops']}")
        print(f"  Termination: {episode_info['terminated_reason']}")
        print()
        print("  To view video, open:")
        print(f"    {video_path}")
        print()
    
    return results


def print_results(results: BenchmarkResults) -> None:
    """Print comprehensive benchmark results."""
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print()
    
    # Score statistics
    print("SCORE STATISTICS")
    print("-" * 40)
    score_stats = compute_statistics(results.all_scores)
    for key, val in score_stats.items():
        print(f"  {key.capitalize():<10} {val:>10.1f}")
    print()
    
    # Drops per episode
    all_drops = []
    for m in results.run_metrics:
        all_drops.extend(m.drops_per_episode)
    print("DROPS PER EPISODE")
    print("-" * 40)
    drop_stats = compute_statistics(all_drops)
    for key, val in drop_stats.items():
        print(f"  {key.capitalize():<10} {val:>10.1f}")
    print()
    
    # Performance metrics
    print("PERFORMANCE METRICS")
    print("-" * 40)
    print(f"  Total benchmark time:   {results.total_time:.2f} seconds")
    print(f"  Total episodes:         {len(results.all_scores)}")
    print(f"  Avg steps/second:       {results.avg_steps_per_second:.1f}")
    if results.peak_memory_mb > 0:
        print(f"  Peak memory:            {results.peak_memory_mb:.1f} MB")
    print()
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Agent: Baseline LIDAR (drop at lowest column)")
    print(f"  Mean Score: {score_stats['mean']:.1f} ± {score_stats['std']:.1f}")
    print(f"  Median Score: {score_stats['median']:.1f}")
    print(f"  Best Score: {score_stats['max']:.0f}")
    print(f"  Worst Score: {score_stats['min']:.0f}")
    print()
    print(f"  Videos saved to: {VIDEOS_DIR}/")
    print()
    print("  Use this baseline to compare your agent's performance!")
    print("=" * 70)


def main():
    """Main entry point."""
    default_envs = get_recommended_num_envs()
    
    parser = argparse.ArgumentParser(
        description="Benchmark the Baseline LIDAR Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python -m contestants.baseline_lidar.benchmark
  python -m contestants.baseline_lidar.benchmark --num-envs 8 --num-runs 3
  python -m contestants.baseline_lidar.benchmark --sync  # Single-threaded for debugging
  
Performance Notes:
  - Uses multiprocessing (AsyncVectorEnv) for parallel physics simulation
  - Default num-envs = CPU cores ({default_envs} on this machine)
  - Don't use more envs than CPU cores (causes context-switching overhead)
  
Videos are saved to:
  contestants/baseline_lidar/videos/<filename>.mp4
        """
    )
    parser.add_argument(
        "--num-envs", type=int, default=None,
        help=f"Number of parallel environments (default: CPU cores = {default_envs})"
    )
    parser.add_argument(
        "--num-runs", type=int, default=5,
        help="Number of benchmark runs (default: 5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-episode details"
    )
    parser.add_argument(
        "--no-video", action="store_true",
        help="Don't record a video"
    )
    parser.add_argument(
        "--sync", action="store_true",
        help="Use single-threaded mode (for debugging)"
    )
    
    args = parser.parse_args()
    
    results = run_benchmark(
        num_envs=args.num_envs,
        num_runs=args.num_runs,
        seed=args.seed,
        verbose=args.verbose,
        save_replay_flag=not args.no_video,
        use_sync=args.sync
    )
    
    print_results(results)


if __name__ == "__main__":
    main()
