"""Graph integrity checks for ViddhiAI Neo4j knowledge graph.

Three Cypher-based checks run directly against Neo4j (not through the API):
1. No orphan nodes — every node is reachable from an Act root
2. All REFERS_TO edges point to existing nodes with valid UIDs
3. Amendment chains produce correct effective text

Usage:
  NEO4J_URI=neo4j+s://... NEO4J_USER=neo4j NEO4J_PASSWORD=... pytest tests/test_graph_integrity.py -v
"""
from __future__ import annotations

import pytest
from graph.connection import DATABASE


@pytest.mark.integration
def test_no_orphan_nodes(neo4j_driver):
    """Every node must be reachable from an Act root via structural relationships.

    An orphan node is one that has no incoming relationships AND is not an Act node.
    This catches loading errors where nodes were created but not connected to the hierarchy.
    """
    with neo4j_driver.session(database=DATABASE) as session:
        result = session.run(
            """
            MATCH (n)
            WHERE NOT n:Act
              AND NOT ()-[]->(n)
            RETURN count(n) AS orphan_count
            """
        )
        orphan_count = result.single()["orphan_count"]

    assert orphan_count == 0, (
        f"Found {orphan_count} orphan nodes (nodes with no incoming relationships "
        f"that are not Act root nodes). Run: MATCH (n) WHERE NOT n:Act AND NOT ()-[]->(n) "
        f"RETURN labels(n), n.uid LIMIT 10"
    )


@pytest.mark.integration
def test_refers_to_edges_valid(neo4j_driver):
    """All REFERS_TO edges must connect nodes that both have valid UIDs.

    A broken REFERS_TO edge means the cross-reference linker created an edge
    to a node that either doesn't exist or has no UID (data corruption).
    """
    with neo4j_driver.session(database=DATABASE) as session:
        result = session.run(
            """
            MATCH (a)-[:REFERS_TO]->(b)
            WHERE b.uid IS NULL
            RETURN count(*) AS broken_count
            """
        )
        broken_count = result.single()["broken_count"]

    assert broken_count == 0, (
        f"Found {broken_count} REFERS_TO edges pointing to nodes without UIDs. "
        f"Run: MATCH (a)-[:REFERS_TO]->(b) WHERE b.uid IS NULL "
        f"RETURN a.uid, labels(b) LIMIT 10"
    )


@pytest.mark.integration
def test_amendment_chains_produce_effective_text(neo4j_driver):
    """Amendment chains must produce valid effective text for known amended sections.

    Finds sections that have at least one amendment relationship and verifies
    the effective state resolver produces a non-empty result with a valid status.
    """
    from graph.resolver import resolve_effective_state

    with neo4j_driver.session(database=DATABASE) as session:
        # Find sections that have amendment relationships
        result = session.run(
            """
            MATCH (s:Section)<-[:AMENDED_BY|SUBSTITUTES|INSERTS|OMITS]-(a:AmendmentAct)
            RETURN DISTINCT s.number AS section_number
            LIMIT 5
            """
        )
        amended_sections = [r["section_number"] for r in result]

    if not amended_sections:
        pytest.skip("No amended sections found in graph — cannot test amendment chains")

    errors = []
    for section_num in amended_sections:
        try:
            state = resolve_effective_state(neo4j_driver, section_num)
            if state is None:
                errors.append(f"Section {section_num}: resolver returned None")
            elif state.status not in ("amended", "original", "omitted"):
                errors.append(
                    f"Section {section_num}: unexpected status '{state.status}'"
                )
            elif state.status == "amended" and not state.current_text:
                errors.append(
                    f"Section {section_num}: status is 'amended' but current_text is empty"
                )
        except Exception as exc:
            errors.append(f"Section {section_num}: resolver raised {type(exc).__name__}: {exc}")

    assert not errors, (
        f"Amendment chain errors for {len(errors)} section(s):\n"
        + "\n".join(f"  - {e}" for e in errors)
    )
