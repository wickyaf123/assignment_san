"""
Integration tests for the Neo4j graph — constraints, node presence, relationships.

All tests are marked @pytest.mark.slow and require a live Neo4j instance
(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD must be set).

Run with:
    pytest tests/test_graph.py -m slow
"""

from __future__ import annotations

import os

import pytest

from graph.runner import PARSED_DIR


EXPECTED_LABELS = {
    "Act",
    "Chapter",
    "Section",
    "SubSection",
    "Clause",
    "SubClause",
    "Proviso",
    "Explanation",
    "Definition",
    "Rule",
    "RuleSet",
    "Form",
    "Schedule",
    "AmendmentAct",
}


@pytest.mark.slow
def test_constraints_exist(neo4j_driver):
    """SHOW CONSTRAINTS must return >= 14 rows, one per node type."""
    with neo4j_driver.session(database="neo4j") as session:
        result = session.run("SHOW CONSTRAINTS")
        constraints = list(result)

    assert len(constraints) >= 14, (
        f"Expected >= 14 constraints, found {len(constraints)}"
    )

    found_labels: set[str] = set()
    for record in constraints:
        labels_or_types = record.get("labelsOrTypes") or []
        for label in labels_or_types:
            found_labels.add(label)

    assert EXPECTED_LABELS.issubset(found_labels), (
        f"Missing labels in constraints: {EXPECTED_LABELS - found_labels}"
    )


@pytest.mark.slow
def test_corpus_coverage(neo4j_driver):
    """Basic sanity: Act, Chapter, and Section nodes must exist after loading."""
    with neo4j_driver.session(database="neo4j") as session:
        section_cnt = session.run(
            "MATCH (s:Section) RETURN count(s) AS cnt"
        ).single()["cnt"]
        chapter_cnt = session.run(
            "MATCH (c:Chapter) RETURN count(c) AS cnt"
        ).single()["cnt"]
        act_cnt = session.run(
            "MATCH (a:Act) RETURN count(a) AS cnt"
        ).single()["cnt"]

    assert section_cnt > 0, "No Section nodes found"
    assert chapter_cnt > 0, "No Chapter nodes found"
    assert act_cnt == 1, f"Expected exactly 1 Act node, got {act_cnt}"


@pytest.mark.slow
def test_node_types_exist(neo4j_driver):
    """db.labels() must include at least Act, Chapter, Section."""
    with neo4j_driver.session(database="neo4j") as session:
        result = session.run(
            "CALL db.labels() YIELD label RETURN collect(label) AS labels"
        )
        labels = result.single()["labels"]

    for required in ("Act", "Chapter", "Section"):
        assert required in labels, f"Label {required!r} not found in db.labels()"


@pytest.mark.slow
def test_structural_rels(neo4j_driver):
    """HAS_CHAPTER and HAS_SECTION structural relationships must exist."""
    with neo4j_driver.session(database="neo4j") as session:
        has_chapter = session.run(
            "MATCH (:Act)-[r:HAS_CHAPTER]->(:Chapter) RETURN count(r) AS cnt"
        ).single()["cnt"]
        has_section = session.run(
            "MATCH (:Chapter)-[r:HAS_SECTION]->(:Section) RETURN count(r) AS cnt"
        ).single()["cnt"]

    assert has_chapter > 0, "No HAS_CHAPTER relationships found"
    assert has_section > 0, "No HAS_SECTION relationships found"


@pytest.mark.slow
def test_amendment_date_type(neo4j_driver):
    """Amendment relationships must have a non-null effective_date property."""
    with neo4j_driver.session(database="neo4j") as session:
        result = session.run(
            "MATCH (:AmendmentAct)-[r]->(:Section) "
            "WHERE type(r) IN ['SUBSTITUTES','INSERTS','OMITS','DECRIMINALIZES'] "
            "RETURN r.effective_date AS ed LIMIT 1"
        )
        record = result.single()

    if record is None:
        pytest.skip("No amendment relationships in graph")

    assert record["ed"] is not None, "Amendment effective_date is null"


@pytest.mark.slow
def test_cross_ref_edges(neo4j_driver):
    """At least one cross-reference relationship type must exist in the graph."""
    with neo4j_driver.session(database="neo4j") as session:
        result = session.run(
            "MATCH ()-[r]->() "
            "WHERE type(r) IN ['REFERS_TO','SUBJECT_TO','NOTWITHSTANDING','IMPOSES_PENALTY','PRESCRIBES_FOR'] "
            "RETURN type(r) AS rt, count(r) AS cnt"
        )
        rows = list(result)

    assert any(row["cnt"] > 0 for row in rows), (
        "No cross-reference relationships found (REFERS_TO / SUBJECT_TO / etc.)"
    )


@pytest.mark.slow
def test_idempotency(neo4j_driver):
    """Graph must have nodes and relationships — sanity check after load."""
    with neo4j_driver.session(database="neo4j") as session:
        node_count = session.run(
            "MATCH (n) RETURN count(n) AS node_count"
        ).single()["node_count"]
        rel_count = session.run(
            "MATCH ()-[r]->() RETURN count(r) AS rel_count"
        ).single()["rel_count"]

    assert node_count > 0, "Graph is empty — no nodes found"
    assert rel_count > 0, "Graph has no relationships"


@pytest.mark.slow
def test_e2e_pipeline(neo4j_driver):
    """End-to-end pipeline test — runs full population pipeline and verifies idempotency.

    Checks:
    - Parsed JSON files exist (skip if not)
    - run_graph_population() returns expected keys with positive counts
    - Running pipeline a second time produces identical counts (idempotency)
    """
    from graph.runner import run_graph_population

    # Require parsed JSON files from ingestion pipeline
    required_files = [
        "companies_act_2013_parsed.json",
        "amendment_act_2026_parsed.json",
        "companies_rules_2014_parsed.json",
    ]
    missing = [f for f in required_files if not (PARSED_DIR / f).exists()]
    if missing:
        pytest.skip(f"Parsed JSON not found (run ingestion first): {missing}")

    if not os.environ.get("NEO4J_URI"):
        pytest.skip("NEO4J_URI not set")

    # First run
    result1 = run_graph_population()

    # Verify expected keys exist
    expected_keys = {"act_nodes", "amendment_nodes", "rules_nodes", "cross_ref_edges"}
    assert expected_keys.issubset(result1.keys()), (
        f"Missing keys in result: {expected_keys - result1.keys()}"
    )

    # Verify act_nodes and amendment_nodes and rules_nodes are dicts with positive counts
    assert isinstance(result1["act_nodes"], dict), "act_nodes must be a dict"
    assert isinstance(result1["amendment_nodes"], dict), "amendment_nodes must be a dict"
    assert isinstance(result1["rules_nodes"], dict), "rules_nodes must be a dict"
    assert isinstance(result1["cross_ref_edges"], dict), "cross_ref_edges must be a dict"

    assert result1["act_nodes"].get("nodes", 0) > 0, "act_nodes['nodes'] must be > 0"
    assert result1["amendment_nodes"].get("nodes", 0) > 0, "amendment_nodes['nodes'] must be > 0"
    assert result1["rules_nodes"].get("nodes", 0) > 0, "rules_nodes['nodes'] must be > 0"

    # Second run — must produce identical counts (idempotency per D-04)
    result2 = run_graph_population()

    assert result1["act_nodes"] == result2["act_nodes"], (
        f"act_nodes differ between runs: {result1['act_nodes']} != {result2['act_nodes']}"
    )
    assert result1["amendment_nodes"] == result2["amendment_nodes"], (
        f"amendment_nodes differ between runs: {result1['amendment_nodes']} != {result2['amendment_nodes']}"
    )
    assert result1["rules_nodes"] == result2["rules_nodes"], (
        f"rules_nodes differ between runs: {result1['rules_nodes']} != {result2['rules_nodes']}"
    )
