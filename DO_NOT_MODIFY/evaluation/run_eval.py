"""
Evaluation Harness
==================

Runs agent submissions against the fixed seed bank and computes scores.

Usage:
    python -m DO_NOT_MODIFY.evaluation.run_eval --agent contestants/team_name
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from DO_NOT_MODIFY.suika_core.config_loader import load_config
from DO_NOT_MODIFY.suika_core.env_gym import SuikaEnv


@dataclass
class EvalResult:
    """Result for a single seed."""
    seed: int
    final_score: int
    drops_used: int
    termination_reason: str
    elapsed_time: float
    actions: Optional[List[float]] = None


@dataclass
class EvalSummary:
    """Summary of evaluation across all seeds."""
    mean_score: float
    std_score: float
    min_score: int
    max_score: int
    median_score: float
    total_time: float
    results: List[EvalResult]


def load_seed_bank(path: Optional[str] = None) -> List[int]:
    """
    Load the evaluation seed bank.
    
    Args:
        path: Path to seed_bank.json. Uses default if None.
        
    Returns:
        List of seeds.
    """
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "seed_bank.json")
    
    with open(path, "r") as f:
        data = json.load(f)
    
    return data["seeds"]


def load_agent(agent_path: str) -> Callable:
    """
    Load an agent from a path.
    
    Args:
        agent_path: Path to agent directory or agent.py file.
        
    Returns:
        Agent's act function.
    """
    agent_path = Path(agent_path)
    
    if agent_path.is_dir():
        # Look for agent.py in directory
        agent_file = agent_path / "agent.py"
    else:
        agent_file = agent_path
    
    if not agent_file.exists():
        raise FileNotFoundError(f"Agent file not found: {agent_file}")
    
    # Load the module
    spec = importlib.util.spec_from_file_location("agent_module", agent_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load agent module from {agent_file}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_module"] = module
    spec.loader.exec_module(module)
    
    # Look for SuikaAgent class or act function
    if hasattr(module, "SuikaAgent"):
        agent_class = getattr(module, "SuikaAgent")
        agent_instance = agent_class()
        if hasattr(agent_instance, "act"):
            return agent_instance.act
        raise AttributeError("SuikaAgent class must have an 'act' method")
    
    elif hasattr(module, "act"):
        return getattr(module, "act")
    
    else:
        raise AttributeError(
            f"Agent module must have either 'SuikaAgent' class with 'act' method "
            f"or standalone 'act' function"
        )


def evaluate_single_seed(
    agent_fn: Callable,
    seed: int,
    record_actions: bool = False,
    verbose: bool = False
) -> EvalResult:
    """
    Evaluate agent on a single seed.
    
    Args:
        agent_fn: Agent's act function (obs) -> action.
        seed: Random seed.
        record_actions: If True, record all actions for replay.
        verbose: If True, print progress.
        
    Returns:
        EvalResult for this seed.
    """
    env = SuikaEnv()
    
    obs, info = env.reset(seed=seed)
    
    actions = [] if record_actions else None
    start_time = time.time()
    
    done = False
    while not done:
        action = agent_fn(obs)
        
        if record_actions:
            actions.append(float(action))
        
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    elapsed = time.time() - start_time
    
    result = EvalResult(
        seed=seed,
        final_score=info["score"],
        drops_used=info["drops_used"],
        termination_reason=info["terminated_reason"],
        elapsed_time=elapsed,
        actions=actions
    )
    
    env.close()
    
    if verbose:
        print(f"  Seed {seed}: score={result.final_score}, "
              f"drops={result.drops_used}, time={elapsed:.2f}s")
    
    return result


def evaluate_agent(
    agent_fn: Callable,
    seeds: Optional[List[int]] = None,
    record_actions: bool = False,
    verbose: bool = True
) -> EvalSummary:
    """
    Evaluate agent on all seeds in the seed bank.
    
    Args:
        agent_fn: Agent's act function (obs) -> action.
        seeds: List of seeds. Uses seed_bank.json if None.
        record_actions: If True, record actions for replay.
        verbose: If True, print progress.
        
    Returns:
        EvalSummary with aggregate statistics.
    """
    if seeds is None:
        seeds = load_seed_bank()
    
    if verbose:
        print(f"Evaluating on {len(seeds)} seeds...")
    
    results: List[EvalResult] = []
    total_start = time.time()
    
    for i, seed in enumerate(seeds):
        if verbose:
            print(f"[{i+1}/{len(seeds)}] Running seed {seed}...")
        
        result = evaluate_single_seed(
            agent_fn,
            seed,
            record_actions=record_actions,
            verbose=verbose
        )
        results.append(result)
    
    total_time = time.time() - total_start
    
    scores = [r.final_score for r in results]
    
    summary = EvalSummary(
        mean_score=float(np.mean(scores)),
        std_score=float(np.std(scores)),
        min_score=int(min(scores)),
        max_score=int(max(scores)),
        median_score=float(np.median(scores)),
        total_time=total_time,
        results=results
    )
    
    if verbose:
        print()
        print("=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Seeds evaluated: {len(seeds)}")
        print(f"Mean score:      {summary.mean_score:.2f}")
        print(f"Std deviation:   {summary.std_score:.2f}")
        print(f"Min score:       {summary.min_score}")
        print(f"Max score:       {summary.max_score}")
        print(f"Median score:    {summary.median_score:.2f}")
        print(f"Total time:      {total_time:.2f}s")
        print("=" * 50)
    
    return summary


def save_results(
    summary: EvalSummary,
    agent_name: str,
    output_path: str
) -> None:
    """Save evaluation results to JSON."""
    data = {
        "agent": agent_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mean_score": summary.mean_score,
        "std_score": summary.std_score,
        "min_score": summary.min_score,
        "max_score": summary.max_score,
        "median_score": summary.median_score,
        "total_time": summary.total_time,
        "results": [
            {
                "seed": r.seed,
                "final_score": r.final_score,
                "drops_used": r.drops_used,
                "termination_reason": r.termination_reason,
                "elapsed_time": r.elapsed_time
            }
            for r in summary.results
        ]
    }
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Suika agent")
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        help="Path to agent directory or agent.py file"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Path to seed bank JSON (uses default if not specified)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record actions for replay"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Load agent
    print(f"Loading agent from {args.agent}...")
    try:
        agent_fn = load_agent(args.agent)
    except Exception as e:
        print(f"Error loading agent: {e}")
        return 1
    
    # Load seeds
    seeds = None
    if args.seeds:
        seeds = load_seed_bank(args.seeds)
    
    # Run evaluation
    summary = evaluate_agent(
        agent_fn,
        seeds=seeds,
        record_actions=args.record,
        verbose=not args.quiet
    )
    
    # Save results
    if args.output:
        agent_name = Path(args.agent).name
        save_results(summary, agent_name, args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
