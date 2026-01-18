"""
Performance Benchmark
=====================

Measures environment step throughput for performance tuning.

Usage:
    python -m tools.benchmark_speed [--envs N] [--steps S]
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional
import numpy as np

from DO_NOT_MODIFY.suika_core.config_loader import load_config
from DO_NOT_MODIFY.suika_core.game import CoreGame
from DO_NOT_MODIFY.suika_core.env_gym import SuikaEnv
from DO_NOT_MODIFY.suika_core.vector_env import SuikaVectorEnv


def benchmark_single_env(
    num_steps: int = 1000,
    seed: int = 42
) -> dict:
    """
    Benchmark single environment performance.
    
    Args:
        num_steps: Number of steps to run.
        seed: Random seed.
        
    Returns:
        Dict with timing results.
    """
    env = SuikaEnv()
    rng = np.random.default_rng(seed)
    
    # Warmup
    obs, _ = env.reset(seed=seed)
    for _ in range(10):
        action = rng.uniform(-1, 1)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    
    # Benchmark
    obs, _ = env.reset(seed=seed)
    start = time.perf_counter()
    
    for _ in range(num_steps):
        action = rng.uniform(-1, 1)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    
    elapsed = time.perf_counter() - start
    env.close()
    
    return {
        "mode": "single",
        "num_envs": 1,
        "num_steps": num_steps,
        "elapsed_seconds": elapsed,
        "steps_per_second": num_steps / elapsed,
        "ms_per_step": (elapsed * 1000) / num_steps
    }


def benchmark_vector_env(
    num_envs: int = 16,
    num_steps: int = 1000,
    seed: int = 42
) -> dict:
    """
    Benchmark vectorized environment performance.
    
    Args:
        num_envs: Number of parallel environments.
        num_steps: Number of steps per environment.
        seed: Random seed.
        
    Returns:
        Dict with timing results.
    """
    vec_env = SuikaVectorEnv(num_envs=num_envs, seed=seed)
    rng = np.random.default_rng(seed)
    
    # Warmup
    vec_env.reset()
    for _ in range(10):
        actions = rng.uniform(-1, 1, size=num_envs).astype(np.float32)
        obs, _, terminateds, truncateds, _ = vec_env.step(actions)
        done = np.where(terminateds | truncateds)[0].tolist()
        if done:
            vec_env.reset(env_indices=done)
    
    # Benchmark
    vec_env.reset()
    start = time.perf_counter()
    
    total_steps = 0
    for _ in range(num_steps):
        actions = rng.uniform(-1, 1, size=num_envs).astype(np.float32)
        obs, _, terminateds, truncateds, _ = vec_env.step(actions)
        total_steps += num_envs
        
        done = np.where(terminateds | truncateds)[0].tolist()
        if done:
            vec_env.reset(env_indices=done)
    
    elapsed = time.perf_counter() - start
    vec_env.close()
    
    return {
        "mode": "vector",
        "num_envs": num_envs,
        "num_steps": num_steps,
        "total_env_steps": total_steps,
        "elapsed_seconds": elapsed,
        "steps_per_second": total_steps / elapsed,
        "batch_steps_per_second": num_steps / elapsed,
        "ms_per_batch": (elapsed * 1000) / num_steps
    }


def benchmark_core_game(
    num_steps: int = 1000,
    seed: int = 42
) -> dict:
    """
    Benchmark raw CoreGame without Gym overhead.
    
    Args:
        num_steps: Number of steps.
        seed: Random seed.
        
    Returns:
        Dict with timing results.
    """
    config = load_config()
    game = CoreGame(config=config, seed=seed)
    rng = np.random.default_rng(seed)
    
    # Warmup
    game.reset(seed=seed)
    for _ in range(10):
        action = rng.uniform(-1, 1)
        result = game.step(action)
        if result.terminated or result.truncated:
            game.reset()
    
    # Benchmark
    game.reset(seed=seed)
    start = time.perf_counter()
    
    for _ in range(num_steps):
        action = rng.uniform(-1, 1)
        result = game.step(action)
        if result.terminated or result.truncated:
            game.reset()
    
    elapsed = time.perf_counter() - start
    
    return {
        "mode": "core_game",
        "num_envs": 1,
        "num_steps": num_steps,
        "elapsed_seconds": elapsed,
        "steps_per_second": num_steps / elapsed,
        "ms_per_step": (elapsed * 1000) / num_steps
    }


def run_all_benchmarks(
    vector_env_sizes: list = [1, 4, 16, 64, 128],
    steps: int = 500
) -> list:
    """Run comprehensive benchmarks."""
    results = []
    
    print("=" * 60)
    print("SUIKA ENVIRONMENT PERFORMANCE BENCHMARK")
    print("=" * 60)
    print()
    
    # Core game (no Gym overhead)
    print("Benchmarking CoreGame (raw)...")
    result = benchmark_core_game(num_steps=steps)
    results.append(result)
    print(f"  Steps/sec: {result['steps_per_second']:.1f}")
    print(f"  ms/step:   {result['ms_per_step']:.3f}")
    print()
    
    # Single Gym env
    print("Benchmarking SuikaEnv (single)...")
    result = benchmark_single_env(num_steps=steps)
    results.append(result)
    print(f"  Steps/sec: {result['steps_per_second']:.1f}")
    print(f"  ms/step:   {result['ms_per_step']:.3f}")
    print()
    
    # Vector envs
    for num_envs in vector_env_sizes:
        print(f"Benchmarking SuikaVectorEnv (n={num_envs})...")
        result = benchmark_vector_env(num_envs=num_envs, num_steps=steps)
        results.append(result)
        print(f"  Env steps/sec: {result['steps_per_second']:.1f}")
        print(f"  Batch steps/sec: {result['batch_steps_per_second']:.1f}")
        print(f"  ms/batch:   {result['ms_per_batch']:.3f}")
        print()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print(f"{'Mode':<20} {'Envs':>6} {'Steps/s':>12} {'ms/step':>10}")
    print("-" * 50)
    
    for r in results:
        mode = r["mode"]
        envs = r["num_envs"]
        sps = r["steps_per_second"]
        
        if "ms_per_step" in r:
            ms = r["ms_per_step"]
        else:
            ms = r["ms_per_batch"]
        
        print(f"{mode:<20} {envs:>6} {sps:>12.1f} {ms:>10.3f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Suika environment performance")
    parser.add_argument("--steps", type=int, default=500, help="Steps per benchmark")
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 4, 16, 64, 128],
                        help="Vector env sizes to test")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (fewer steps)")
    
    args = parser.parse_args()
    
    steps = 100 if args.quick else args.steps
    
    run_all_benchmarks(
        vector_env_sizes=args.envs,
        steps=steps
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
