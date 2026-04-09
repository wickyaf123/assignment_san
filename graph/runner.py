"""
CLI runner for the ViddhiAI graph population pipeline.

Orchestrates the full pipeline in strict order:
    1. Schema setup — 14 constraints + 4 indexes (idempotent)
    2. Load Act — Companies Act 2013 nodes and structural relationships
    3. Load Amendment Act — AmendmentAct node + amendment relationships
    4. Load RuleSet — Rule/Form nodes and structural relationships
    5. Link cross-references — REFERS_TO, SUBJECT_TO, NOTWITHSTANDING,
       IMPOSES_PENALTY, PRESCRIBES_FOR edges between nodes

All steps use MERGE — the pipeline is fully idempotent (safe to run twice).

Usage:
    python -m graph                    (via graph/__main__.py)
    python graph/runner.py             (direct)
    from graph.runner import run_graph_population  (import)

Security (T-02-07):
    FileNotFoundError messages list available parsed files — these are public
    legislation filenames only, no credentials or sensitive paths exposed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from graph.connection import close_driver, get_driver
from graph.linker import link_cross_references, link_derived_rules, link_ruleset_cross_references
from graph.loader import load_act, load_amendment_act, load_ruleset
from graph.schema import ensure_schema
from ingestion.models import Act, AmendmentAct, RuleSet

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants — all derived from __file__, never user-configurable
# ---------------------------------------------------------------------------

# backend/data/parsed/ — final parsed JSON files from ingestion pipeline
PARSED_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "parsed"


# ---------------------------------------------------------------------------
# JSON loader helper
# ---------------------------------------------------------------------------


def _load_parsed_json(filename: str, model_class: type) -> Any:
    """Load and deserialize a parsed JSON file from PARSED_DIR.

    Args:
        filename: Filename within PARSED_DIR (e.g., "companies_act_2013_parsed.json").
        model_class: Pydantic model class to deserialize into.

    Returns:
        Validated Pydantic model instance.

    Raises:
        FileNotFoundError: If the file does not exist, with a message listing
            available files in PARSED_DIR (T-02-07: public filenames only).
    """
    path = PARSED_DIR / filename
    if not path.exists():
        available = sorted(p.name for p in PARSED_DIR.iterdir() if p.suffix == ".json") if PARSED_DIR.exists() else []
        raise FileNotFoundError(
            f"Parsed JSON not found: {path}\n"
            f"Run the ingestion pipeline first (`python -m ingestion`).\n"
            f"Available files in {PARSED_DIR}: {available}"
        )

    logger.info("Loading %s from %s", model_class.__name__, filename)
    text = path.read_text(encoding="utf-8")
    return model_class.model_validate_json(text)


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------


def run_graph_population() -> dict[str, Any]:
    """Orchestrate the full graph population pipeline.

    Pipeline steps (in strict order):
        1. get_driver() — connect to Neo4j
        2. ensure_schema() — create constraints/indexes
        3. load_act() — Companies Act 2013
        4. load_amendment_act() — Amendment Act 2026
        5. load_ruleset() — Companies Rules 2014
        6. link_cross_references() — cross-reference edges
        7. link_derived_rules() — DERIVED_RULE edges (Rule -> Section)
        8. close_driver() — always called via finally

    Returns:
        Summary dict:
        {
            "act_nodes": dict[str, int],
            "amendment_nodes": dict[str, int],
            "rules_nodes": dict[str, int],
            "cross_ref_edges": dict[str, int],
            "derived_rule_edges": dict[str, int],
            "total_time_seconds": float,
        }
    """
    pipeline_start = time.monotonic()
    logger.info("Graph population pipeline starting...")

    # Initialize result variables before try block so they are always bound
    act_result: dict[str, int] = {}
    amendment_result: dict[str, int] = {}
    rules_result: dict[str, int] = {}
    cross_ref_result: dict[str, int] = {}
    derived_rule_result: dict[str, int] = {}

    driver = get_driver()

    try:
        # ------------------------------------------------------------------
        # Step 1: Schema setup
        # ------------------------------------------------------------------
        logger.info("Step 1/7: Setting up Neo4j schema...")
        step_start = time.monotonic()
        ensure_schema(driver)
        logger.info("Schema setup complete in %.2fs", time.monotonic() - step_start)

        # ------------------------------------------------------------------
        # Step 2: Load Companies Act 2013
        # ------------------------------------------------------------------
        logger.info("Step 2/7: Loading Companies Act 2013...")
        step_start = time.monotonic()
        act: Act = _load_parsed_json("companies_act_2013_parsed.json", Act)
        act_result = load_act(driver, act)
        logger.info(
            "Act loaded in %.2fs: %s",
            time.monotonic() - step_start,
            act_result,
        )

        # ------------------------------------------------------------------
        # Step 3: Load Amendment Act 2026
        # ------------------------------------------------------------------
        logger.info("Step 3/7: Loading Amendment Act 2026...")
        step_start = time.monotonic()
        amendment_act: AmendmentAct = _load_parsed_json(
            "amendment_act_2026_parsed.json", AmendmentAct
        )
        amendment_result = load_amendment_act(driver, amendment_act)
        logger.info(
            "Amendment Act loaded in %.2fs: %s",
            time.monotonic() - step_start,
            amendment_result,
        )

        # ------------------------------------------------------------------
        # Step 4: Load Companies Rules 2014
        # ------------------------------------------------------------------
        logger.info("Step 4/7: Loading Companies Rules 2014...")
        step_start = time.monotonic()
        ruleset: RuleSet = _load_parsed_json(
            "companies_rules_2014_parsed.json", RuleSet
        )
        rules_result = load_ruleset(driver, ruleset)
        logger.info(
            "Rules loaded in %.2fs: %s",
            time.monotonic() - step_start,
            rules_result,
        )

        # ------------------------------------------------------------------
        # Step 5: Link cross-references (Act tree)
        # ------------------------------------------------------------------
        logger.info("Step 5/7: Linking Act cross-references...")
        step_start = time.monotonic()
        cross_ref_result = link_cross_references(driver, act)
        logger.info(
            "Act cross-references linked in %.2fs: %s",
            time.monotonic() - step_start,
            cross_ref_result,
        )

        # ------------------------------------------------------------------
        # Step 6: Link RuleSet cross-references (Rule → Section edges)
        # ------------------------------------------------------------------
        logger.info("Step 6/7: Linking RuleSet cross-references...")
        step_start = time.monotonic()
        rules_xref_result = link_ruleset_cross_references(driver, ruleset)
        logger.info(
            "RuleSet cross-references linked in %.2fs: %s",
            time.monotonic() - step_start,
            rules_xref_result,
        )

        # ------------------------------------------------------------------
        # Step 7: Link DERIVED_RULE edges (Rule -> Section)
        # ------------------------------------------------------------------
        logger.info("Step 7/7: Linking DERIVED_RULE edges...")
        step_start = time.monotonic()
        derived_rule_result = link_derived_rules(driver, ruleset)
        logger.info(
            "DERIVED_RULE edges linked in %.2fs: %s",
            time.monotonic() - step_start,
            derived_rule_result,
        )

    finally:
        close_driver()

    total_time = time.monotonic() - pipeline_start
    logger.info("Graph population complete in %.2fs", total_time)

    return {
        "act_nodes": act_result,
        "amendment_nodes": amendment_result,
        "rules_nodes": rules_result,
        "cross_ref_edges": cross_ref_result,
        "rules_cross_ref_edges": rules_xref_result,
        "derived_rule_edges": derived_rule_result,
        "total_time_seconds": round(total_time, 2),
    }


# ---------------------------------------------------------------------------
# Entry point (direct script execution)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    result = run_graph_population()
    print(f"\nGraph population complete: {result}")
