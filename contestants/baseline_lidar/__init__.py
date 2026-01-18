"""
Baseline LIDAR Agent Package

A simple heuristic agent that drops fruits at the lowest column
using the height_map observation. Serves as a benchmark and example.
"""

from .agent import SuikaAgent, create_agent

__all__ = ["SuikaAgent", "create_agent"]
