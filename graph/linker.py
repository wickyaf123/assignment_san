"""
Cross-reference linker for ViddhiAI.

Resolves target_pattern strings from CrossReference objects to existing node
UIDs, then creates semantic relationship edges in Neo4j. Also detects
IMPOSES_PENALTY and PRESCRIBES_FOR patterns from node text (partial GRAPH-04
coverage).

Relationship types supported (ALLOWED_REF_TYPES):
    REFERS_TO       — general section reference
    SUBJECT_TO      — "subject to section N"
    NOTWITHSTANDING — "notwithstanding section N"
    IMPOSES_PENALTY — "penalty/punishable under section N" (new in this module)
    PRESCRIBES_FOR  — "as prescribed in rule/section N" (new in this module)

Deferred (GRAPH-04 partial coverage):
    DERIVED_FROM, APPLIES_TO, SETS_THRESHOLD — require additional detection
    patterns in cross_ref_detector.py beyond Phase 2 scope.

Security (T-02-05, T-02-06):
    - resolve_target_uid extracts digits only via regex; result is passed to
      section_uid(int(N)) — no string interpolation into Cypher.
    - ref_type is validated against ALLOWED_REF_TYPES whitelist before being
      used as a Cypher relationship type identifier.

Public API:
    link_cross_references(driver, act) -> {"edges_attempted": N, "edges_created": M}
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any

import neo4j

from graph.connection import DATABASE
from graph.loader import BATCH_SIZE
from ingestion.models import Act, RuleSet
from ingestion.uid_generator import section_uid

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Whitelist for Cypher relationship types — prevents injection (T-02-06).
ALLOWED_REF_TYPES: frozenset[str] = frozenset({
    "REFERS_TO",
    "SUBJECT_TO",
    "NOTWITHSTANDING",
    "IMPOSES_PENALTY",
    "PRESCRIBES_FOR",
    "DERIVED_RULE",
})

# Pattern to identify a simple "section N" reference (digits only, no alpha suffix).
# The $ anchor ensures trailing content (e.g., "section 149A") does not match.
_SECTION_TARGET_RE = re.compile(r"section\s+(\d+)$", re.IGNORECASE)

# Pattern to extract "under section N" or "in pursuance of section N" from Rule text/title
_DERIVED_RULE_RE = re.compile(
    r"(?:under|in\s+pursuance\s+of|pursuant\s+to)\s+section\s+(\d+)", re.IGNORECASE
)

# Additional detection patterns for GRAPH-04 partial coverage.
# Each tuple: (compiled_pattern, ref_type_string)
# IMPOSES_PENALTY: "penalty under section N" or "punishable under section N"
# PRESCRIBES_FOR: "as prescribed in rule/section N" or "in the prescribed manner"
ADDITIONAL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"(?:penalty|punishable)\s+under\s+section\s+(\d+)", re.IGNORECASE),
        "IMPOSES_PENALTY",
    ),
    (
        re.compile(r"as\s+prescribed\s+in\s+(?:rule|section)\s+(\d+)", re.IGNORECASE),
        "PRESCRIBES_FOR",
    ),
    (
        re.compile(
            r"in\s+the\s+(?:manner\s+)?prescribed\s+(?:in|under)\s+(?:rule|section)\s+(\d+)",
            re.IGNORECASE,
        ),
        "PRESCRIBES_FOR",
    ),
]


# ---------------------------------------------------------------------------
# Core resolver
# ---------------------------------------------------------------------------


def resolve_target_uid(target_pattern: str) -> str | None:
    """Resolve a target_pattern string to a Neo4j node UID.

    Currently supports section references only — rules and sub-section refs
    cannot be resolved (resolver returns None).

    Args:
        target_pattern: Normalized pattern string from CrossReference.target_pattern,
            e.g. "section 135", "section 149A", "rule 3".

    Returns:
        Node UID string (e.g. "sec-135") if resolvable, None otherwise.
    """
    stripped = target_pattern.strip()
    if not stripped:
        return None

    match = _SECTION_TARGET_RE.search(stripped)
    if match:
        return section_uid(int(match.group(1)))

    return None


# ---------------------------------------------------------------------------
# Additional pattern detector
# ---------------------------------------------------------------------------


def _detect_additional_refs(node_uid: str, text: str) -> list[dict[str, Any]]:
    """Detect IMPOSES_PENALTY and PRESCRIBES_FOR patterns in node text.

    Applies ADDITIONAL_PATTERNS to the node's text content and builds edge
    dicts for each match whose target resolves to a known node UID.

    Args:
        node_uid: UID of the source node (e.g. "sec-135").
        text: Full text content of the node.

    Returns:
        Deduplicated list of edge dicts: {"source_uid", "target_uid", "ref_type"}.
        Edges whose target_uid cannot be resolved are skipped with a WARNING.
    """
    if not text:
        return []

    seen: set[tuple[str, str, str]] = set()
    edges: list[dict[str, Any]] = []

    for pattern, ref_type in ADDITIONAL_PATTERNS:
        for match in pattern.finditer(text):
            group_num = match.group(1)
            target_uid = resolve_target_uid(f"section {group_num}")

            if target_uid is None:
                logger.warning(
                    "Unresolvable additional ref: source=%s, pattern=%s, matched_group=%s",
                    node_uid,
                    ref_type,
                    match.group(0),
                )
                continue

            key = (node_uid, target_uid, ref_type)
            if key in seen:
                continue
            seen.add(key)

            edges.append({
                "source_uid": node_uid,
                "target_uid": target_uid,
                "ref_type": ref_type,
            })

    return edges


# ---------------------------------------------------------------------------
# Act tree walker
# ---------------------------------------------------------------------------


def _iter_act_nodes(act: Act):
    """Iterate over every node in the Act tree that may have cross_references.

    Yields node objects in tree order (Act -> Chapter -> Section -> ...).
    """
    yield act

    for chapter in act.chapters:
        yield chapter

        for section in chapter.sections:
            yield section

            for proviso in section.provisos:
                yield proviso
            for explanation in section.explanations:
                yield explanation
            for definition in section.definitions:
                yield definition

            for clause in section.clauses:
                yield clause
                for proviso in clause.provisos:
                    yield proviso
                for explanation in clause.explanations:
                    yield explanation
                for subclause in clause.subclauses:
                    yield subclause
                    for proviso in subclause.provisos:
                        yield proviso
                    for explanation in subclause.explanations:
                        yield explanation

            for subsection in section.subsections:
                yield subsection
                for proviso in subsection.provisos:
                    yield proviso
                for explanation in subsection.explanations:
                    yield explanation
                for definition in subsection.definitions:
                    yield definition

                for clause in subsection.clauses:
                    yield clause
                    for proviso in clause.provisos:
                        yield proviso
                    for explanation in clause.explanations:
                        yield explanation
                    for subclause in clause.subclauses:
                        yield subclause
                        for proviso in subclause.provisos:
                            yield proviso
                        for explanation in subclause.explanations:
                            yield explanation

    # Schedules (if present on Act)
    if hasattr(act, "schedules"):
        for schedule in act.schedules:  # type: ignore[attr-defined]
            yield schedule


def _collect_cross_ref_edges(act: Act) -> list[dict[str, Any]]:
    """Walk the Act tree and collect all cross-reference edges.

    Processes:
    1. CrossReference objects on each node (from cross_ref_detector.py output).
    2. Additional IMPOSES_PENALTY / PRESCRIBES_FOR patterns detected from node text.

    Unresolvable target_patterns are logged as WARNING and skipped — no stub
    nodes are created (per D-10, D-11, D-12).
    ref_type values not in ALLOWED_REF_TYPES are logged as WARNING and skipped
    (per T-02-06).

    Args:
        act: Fully populated Act object (from ingestion pipeline).

    Returns:
        Deduplicated list of edge dicts: {"source_uid", "target_uid", "ref_type"}.
    """
    seen: set[tuple[str, str, str]] = set()
    edges: list[dict[str, Any]] = []

    for node in _iter_act_nodes(act):
        # Process declared cross-references
        for ref in getattr(node, "cross_references", []):
            # Validate ref_type against whitelist (T-02-06)
            if ref.ref_type not in ALLOWED_REF_TYPES:
                logger.warning(
                    "Invalid ref_type rejected: source=%s, ref_type=%r (not in ALLOWED_REF_TYPES)",
                    node.uid,
                    ref.ref_type,
                )
                continue

            target_uid = resolve_target_uid(ref.target_pattern)
            if target_uid is None:
                logger.warning(
                    "Unresolvable cross-reference: source=%s, target_pattern=%r",
                    node.uid,
                    ref.target_pattern,
                )
                continue

            key = (node.uid, target_uid, ref.ref_type)
            if key in seen:
                continue
            seen.add(key)

            edges.append({
                "source_uid": node.uid,
                "target_uid": target_uid,
                "ref_type": ref.ref_type,
            })

        # Process additional pattern detection on node text
        text = getattr(node, "text", "")
        if text:
            for extra_edge in _detect_additional_refs(node.uid, text):
                key = (
                    extra_edge["source_uid"],
                    extra_edge["target_uid"],
                    extra_edge["ref_type"],
                )
                if key in seen:
                    continue
                seen.add(key)
                edges.append(extra_edge)

    return edges


# ---------------------------------------------------------------------------
# Neo4j writer
# ---------------------------------------------------------------------------


def _merge_cross_ref_edges(session: neo4j.Session, edges: list[dict[str, Any]]) -> int:
    """MERGE cross-reference edges into Neo4j, grouped by ref_type.

    Uses UNWIND + MERGE for idempotency. Processes in BATCH_SIZE chunks.
    ref_type is validated against ALLOWED_REF_TYPES before building Cypher (T-02-06).

    Args:
        session: Active Neo4j session.
        edges: List of edge dicts from _collect_cross_ref_edges().

    Returns:
        Total count of edges processed.
    """
    by_type: dict[str, list[dict]] = defaultdict(list)
    for edge in edges:
        by_type[edge["ref_type"]].append(edge)

    total = 0

    for ref_type, group in by_type.items():
        # Security check: validate ref_type before string-interpolating into Cypher (T-02-06)
        if ref_type not in ALLOWED_REF_TYPES:
            logger.warning(
                "Skipping edges with invalid ref_type=%r — not in ALLOWED_REF_TYPES",
                ref_type,
            )
            continue

        cypher = (
            f"UNWIND $batch AS row "
            f"MATCH (src {{uid: row.source_uid}}) "
            f"MATCH (tgt {{uid: row.target_uid}}) "
            f"MERGE (src)-[:{ref_type}]->(tgt)"
        )

        for i in range(0, len(group), BATCH_SIZE):
            chunk = group[i : i + BATCH_SIZE]

            def _write_tx(tx: neo4j.ManagedTransaction, chunk: list[dict] = chunk) -> None:
                tx.run(cypher, batch=chunk)

            session.execute_write(_write_tx)
            total += len(chunk)
            logger.info("Merged %d %s cross-reference edges", len(chunk), ref_type)

    return total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def link_cross_references(driver: neo4j.Driver, act: Act) -> dict[str, int]:
    """Create cross-reference edges in Neo4j for the given Act.

    Orchestrates:
    1. _collect_cross_ref_edges(act) — pure tree walk, no DB
    2. _merge_cross_ref_edges(session, edges) — MERGE into Neo4j

    Args:
        driver: Active Neo4j driver.
        act: Fully populated Act object (nodes must already be loaded).

    Returns:
        {"edges_attempted": N, "edges_created": M}
    """
    edges = _collect_cross_ref_edges(act)
    logger.info(
        "link_cross_references: %d unique edges collected from Act tree",
        len(edges),
    )

    created = 0
    with driver.session(database=DATABASE) as session:
        created = _merge_cross_ref_edges(session, edges)

    logger.info(
        "link_cross_references complete: %d edges attempted, %d edges merged",
        len(edges),
        created,
    )
    return {"edges_attempted": len(edges), "edges_created": created}


# ---------------------------------------------------------------------------
# DERIVED_RULE linking (Rule -> Section)
# ---------------------------------------------------------------------------


def link_derived_rules(driver: neo4j.Driver, ruleset: RuleSet) -> dict[str, int]:
    """Create DERIVED_RULE edges from Rule nodes to their parent Section.

    Each Rule in subordinate legislation is typically framed "under section N".
    This function extracts the primary section reference and creates a
    (Rule)-[:DERIVED_RULE]->(Section) edge.

    Args:
        driver: Active Neo4j driver.
        ruleset: Fully populated RuleSet object (nodes must already be loaded).

    Returns:
        {"edges_attempted": N, "edges_created": M}
    """
    edges: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for rule in ruleset.rules:
        # Search title first, then text for the primary section reference
        for source_text in [rule.title, rule.text]:
            if not source_text:
                continue
            match = _DERIVED_RULE_RE.search(source_text)
            if match:
                target_uid = section_uid(int(match.group(1)))
                key = (rule.uid, target_uid)
                if key not in seen:
                    seen.add(key)
                    edges.append({"source_uid": rule.uid, "target_uid": target_uid})
                break  # Use first match only (primary section reference)

    logger.info(
        "link_derived_rules: %d edges collected from %d rules",
        len(edges),
        len(ruleset.rules),
    )

    created = 0
    if edges:
        cypher = (
            "UNWIND $batch AS row "
            "MATCH (src {uid: row.source_uid}) "
            "MATCH (tgt {uid: row.target_uid}) "
            "MERGE (src)-[:DERIVED_RULE]->(tgt)"
        )
        with driver.session(database=DATABASE) as session:
            for i in range(0, len(edges), BATCH_SIZE):
                chunk = edges[i : i + BATCH_SIZE]

                def _write_tx(tx: neo4j.ManagedTransaction, chunk: list[dict] = chunk) -> None:
                    tx.run(cypher, batch=chunk)

                session.execute_write(_write_tx)
                created += len(chunk)
                logger.info("Merged %d DERIVED_RULE edges", len(chunk))

    return {"edges_attempted": len(edges), "edges_created": created}


# ---------------------------------------------------------------------------
# RuleSet cross-reference linking
# ---------------------------------------------------------------------------


def _iter_ruleset_nodes(ruleset: RuleSet):
    """Iterate over every node in the RuleSet tree that may have cross_references."""
    yield ruleset
    for rule in ruleset.rules:
        yield rule
        for form in rule.forms:
            yield form


def _collect_ruleset_cross_ref_edges(ruleset: RuleSet) -> list[dict[str, Any]]:
    """Walk the RuleSet tree and collect cross-reference edges to Section nodes.

    Processes both declared CrossReference objects and text-based pattern
    detection (IMPOSES_PENALTY, PRESCRIBES_FOR). All targets resolve to
    Section UIDs only (same constraint as Act linking).
    """
    seen: set[tuple[str, str, str]] = set()
    edges: list[dict[str, Any]] = []

    for node in _iter_ruleset_nodes(ruleset):
        for ref in getattr(node, "cross_references", []):
            if ref.ref_type not in ALLOWED_REF_TYPES:
                logger.warning(
                    "Invalid ref_type rejected (RuleSet): source=%s, ref_type=%r",
                    node.uid, ref.ref_type,
                )
                continue

            target_uid = resolve_target_uid(ref.target_pattern)
            if target_uid is None:
                continue

            key = (node.uid, target_uid, ref.ref_type)
            if key in seen:
                continue
            seen.add(key)
            edges.append({
                "source_uid": node.uid,
                "target_uid": target_uid,
                "ref_type": ref.ref_type,
            })

        text = getattr(node, "text", "")
        if text:
            for extra_edge in _detect_additional_refs(node.uid, text):
                key = (
                    extra_edge["source_uid"],
                    extra_edge["target_uid"],
                    extra_edge["ref_type"],
                )
                if key in seen:
                    continue
                seen.add(key)
                edges.append(extra_edge)

    return edges


def link_ruleset_cross_references(
    driver: neo4j.Driver, ruleset: RuleSet
) -> dict[str, int]:
    """Create cross-reference edges from RuleSet/Rule/Form nodes to Section nodes.

    Mirrors link_cross_references() but walks the RuleSet tree instead of the
    Act tree. This creates the Rule→Section edges that enable cross-document
    queries (e.g. "which rules prescribe for Section 135?").

    Args:
        driver: Active Neo4j driver.
        ruleset: Fully populated RuleSet object (nodes must already be loaded).

    Returns:
        {"edges_attempted": N, "edges_created": M}
    """
    edges = _collect_ruleset_cross_ref_edges(ruleset)
    logger.info(
        "link_ruleset_cross_references: %d unique edges collected from RuleSet tree",
        len(edges),
    )

    created = 0
    with driver.session(database=DATABASE) as session:
        created = _merge_cross_ref_edges(session, edges)

    logger.info(
        "link_ruleset_cross_references complete: %d edges attempted, %d edges merged",
        len(edges), created,
    )
    return {"edges_attempted": len(edges), "edges_created": created}
