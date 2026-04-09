"""
Entry point for `python -m graph`.

Runs the full graph population pipeline via graph.runner.run_graph_population().

Usage:
    cd backend && python -m graph
"""

from __future__ import annotations

import logging
import sys

from graph.runner import run_graph_population

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if __name__ == "__main__":
    result = run_graph_population()
    print(f"\nGraph population complete: {result}")
