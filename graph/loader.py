"""
Recursive JSON-to-Neo4j loader for ViddhiAI.

Walks the parsed legal document tree (Act -> Chapter -> Section -> ...) and
MERGE-loads all node types with structural and amendment relationships into Neo4j.

Design decisions:
- Pure collection functions (_collect_*) prepare data without touching Neo4j.
  This makes them unit-testable without a live database.
- UNWIND-based batching (BATCH_SIZE=500) reduces round-trips to Neo4j.
- MERGE + ON CREATE/MATCH SET ensures idempotent loads — safe to run twice.
- AmendmentEntry.removed_text is mapped to old_text on Neo4j relationship
  properties so it aligns with the RESOLVE_QUERY Cypher in resolver.py.
- Any amendment_type value (including DECRIMINALIZES or unknown types) is
  passed through as-is to become the Neo4j relationship type.

Public API:
    load_act(driver, act) -> {"nodes": N, "relationships": M}
    load_amendment_act(driver, amendment_act) -> {"nodes": N, "relationships": M}
    load_ruleset(driver, ruleset) -> {"nodes": N, "relationships": M}
"""

from __future__ import annotations

import logging
import re
from typing import Any

import neo4j

from graph.connection import DATABASE

from ingestion.models import (
    Act,
    AmendmentAct,
    Chapter,
    Clause,
    Explanation,
    Proviso,
    RuleSet,
    Section,
    SubClause,
    SubSection,
)
from ingestion.preprocessor import LLMFormatter, QualityScorer
from ingestion.uid_generator import section_uid

logger = logging.getLogger(__name__)

BATCH_SIZE = 500

# Whitelist of valid amendment relationship types for Cypher interpolation.
# Matches the relationship types declared in schema.py and queried in resolver.py.
ALLOWED_AMENDMENT_TYPES: frozenset[str] = frozenset({
    "SUBSTITUTES",
    "INSERTS",
    "OMITS",
    "DECRIMINALIZES",
})

# Whitelist of valid structural relationship types for Cypher interpolation.
# Mirrors the security pattern from linker.py's ALLOWED_REF_TYPES.
ALLOWED_STRUCTURAL_REL_TYPES: frozenset[str] = frozenset({
    "HAS_CHAPTER",
    "HAS_SECTION",
    "HAS_SUBSECTION",
    "HAS_CLAUSE",
    "HAS_SUBCLAUSE",
    "HAS_PROVISO",
    "HAS_EXPLANATION",
    "HAS_DEFINITION",
    "HAS_SCHEDULE",
    "HAS_RULE",
    "HAS_FORM",
})

# Hardcoded Act UID — there is only one Act in the corpus.
_DEFAULT_ACT_UID = "act-companies-2013"


# ---------------------------------------------------------------------------
# Pure collection helpers
# ---------------------------------------------------------------------------


def _collect_act_nodes(act: Act) -> dict[str, list[dict[str, Any]]]:
    """Recursively collect all nodes from an Act tree, grouped by Neo4j label.

    Returns a dict keyed by label: {"Act": [...], "Chapter": [...], ...}
    Each entry is a property dict suitable for use in a MERGE/SET Cypher.
    Includes quality_score and llm_text on content nodes.
    """
    scorer = QualityScorer()
    formatter = LLMFormatter()

    nodes: dict[str, list[dict[str, Any]]] = {
        "Act": [],
        "Chapter": [],
        "Section": [],
        "SubSection": [],
        "Clause": [],
        "SubClause": [],
        "Proviso": [],
        "Explanation": [],
        "Definition": [],
        "Schedule": [],
    }

    nodes["Act"].append({
        "uid": act.uid,
        "text": act.text,
        "page_number": act.page_number,
        "source_pdf": act.source_pdf,
        "title": act.title,
        "year": act.year,
    })

    for chapter in act.chapters:
        nodes["Chapter"].append({
            "uid": chapter.uid,
            "text": chapter.text,
            "page_number": chapter.page_number,
            "source_pdf": chapter.source_pdf,
            "number": chapter.number,
            "title": chapter.title,
        })

        for section in chapter.sections:
            _collect_section_nodes(section, nodes, scorer, formatter, chapter)

    # Schedules (if present on Act)
    if hasattr(act, "schedules"):
        for schedule in act.schedules:  # type: ignore[attr-defined]
            nodes["Schedule"].append({
                "uid": schedule.uid,
                "text": schedule.text,
                "page_number": schedule.page_number,
                "source_pdf": schedule.source_pdf,
                "number": schedule.number,
                "title": schedule.title,
            })

    # Remove empty label lists to keep things tidy
    return {label: batch for label, batch in nodes.items() if batch}


def _collect_section_nodes(
    section: Section,
    nodes: dict[str, list[dict[str, Any]]],
    scorer: QualityScorer | None = None,
    formatter: LLMFormatter | None = None,
    chapter: Chapter | None = None,
) -> None:
    """Recursively collect Section and all descendants into `nodes`."""
    section_dict: dict[str, Any] = {
        "uid": section.uid,
        "text": section.text,
        "page_number": section.page_number,
        "source_pdf": section.source_pdf,
        "number": section.number,
        "title": section.title,
        "language": section.language,
    }
    if scorer:
        quality = scorer.score(section.text, "Section")
        section_dict["quality_score"] = round(quality.score, 1)
    if formatter:
        section_dict["llm_text"] = formatter.format_section(section, chapter)
    nodes["Section"].append(section_dict)

    # Provisos at section level
    for idx, proviso in enumerate(section.provisos, start=1):
        nodes["Proviso"].append(_proviso_dict(proviso))

    # Explanations at section level
    for idx, explanation in enumerate(section.explanations, start=1):
        nodes["Explanation"].append(_explanation_dict(explanation))

    # Definitions at section level
    for definition in section.definitions:
        nodes["Definition"].append({
            "uid": definition.uid,
            "text": definition.text,
            "page_number": definition.page_number,
            "source_pdf": definition.source_pdf,
            "term": definition.term,
            "language": definition.language,
        })

    # Direct clauses at section level (some sections skip sub-sections)
    for clause in section.clauses:
        _collect_clause_nodes(clause, nodes)

    # Sub-sections
    for subsection in section.subsections:
        _collect_subsection_nodes(subsection, nodes)


def _collect_subsection_nodes(
    subsection: SubSection,
    nodes: dict[str, list[dict[str, Any]]],
) -> None:
    """Recursively collect SubSection and all descendants."""
    scorer = QualityScorer()
    quality = scorer.score(subsection.text, "SubSection")
    nodes["SubSection"].append({
        "uid": subsection.uid,
        "text": subsection.text,
        "page_number": subsection.page_number,
        "source_pdf": subsection.source_pdf,
        "number": subsection.number,
        "quality_score": round(quality.score, 1),
        "language": subsection.language,
    })

    for proviso in subsection.provisos:
        nodes["Proviso"].append(_proviso_dict(proviso))

    for explanation in subsection.explanations:
        nodes["Explanation"].append(_explanation_dict(explanation))

    for definition in subsection.definitions:
        nodes["Definition"].append({
            "uid": definition.uid,
            "text": definition.text,
            "page_number": definition.page_number,
            "source_pdf": definition.source_pdf,
            "term": definition.term,
            "language": definition.language,
        })

    for clause in subsection.clauses:
        _collect_clause_nodes(clause, nodes)


def _collect_clause_nodes(
    clause: Clause,
    nodes: dict[str, list[dict[str, Any]]],
) -> None:
    """Recursively collect Clause and all descendants."""
    nodes["Clause"].append({
        "uid": clause.uid,
        "text": clause.text,
        "page_number": clause.page_number,
        "source_pdf": clause.source_pdf,
        "label": clause.label,
    })

    for proviso in clause.provisos:
        nodes["Proviso"].append(_proviso_dict(proviso))

    for explanation in clause.explanations:
        nodes["Explanation"].append(_explanation_dict(explanation))

    for subclause in clause.subclauses:
        _collect_subclause_nodes(subclause, nodes)


def _collect_subclause_nodes(
    subclause: SubClause,
    nodes: dict[str, list[dict[str, Any]]],
) -> None:
    """Collect SubClause and its provisos/explanations."""
    nodes["SubClause"].append({
        "uid": subclause.uid,
        "text": subclause.text,
        "page_number": subclause.page_number,
        "source_pdf": subclause.source_pdf,
        "label": subclause.label,
    })

    for proviso in subclause.provisos:
        nodes["Proviso"].append(_proviso_dict(proviso))

    for explanation in subclause.explanations:
        nodes["Explanation"].append(_explanation_dict(explanation))


def _proviso_dict(proviso: Proviso) -> dict[str, Any]:
    return {
        "uid": proviso.uid,
        "text": proviso.text,
        "page_number": proviso.page_number,
        "source_pdf": proviso.source_pdf,
        "proviso_type": proviso.proviso_type,
    }


def _explanation_dict(explanation: Explanation) -> dict[str, Any]:
    return {
        "uid": explanation.uid,
        "text": explanation.text,
        "page_number": explanation.page_number,
        "source_pdf": explanation.source_pdf,
        "label": explanation.label,
    }


def _collect_structural_rels(act: Act) -> list[dict[str, str]]:
    """Collect all structural relationships from an Act tree.

    Returns a list of dicts: {"parent_uid": ..., "child_uid": ..., "rel_type": ...}
    """
    rels: list[dict[str, str]] = []

    for chapter in act.chapters:
        rels.append({"parent_uid": act.uid, "child_uid": chapter.uid, "rel_type": "HAS_CHAPTER"})
        for section in chapter.sections:
            rels.append({"parent_uid": chapter.uid, "child_uid": section.uid, "rel_type": "HAS_SECTION"})
            _collect_section_rels(section, rels)

    # Schedule relationships (if present)
    if hasattr(act, "schedules"):
        for schedule in act.schedules:  # type: ignore[attr-defined]
            rels.append({"parent_uid": act.uid, "child_uid": schedule.uid, "rel_type": "HAS_SCHEDULE"})

    return rels


def _collect_section_rels(section: Section, rels: list[dict[str, str]]) -> None:
    """Collect structural rels for a Section and all descendants."""
    for proviso in section.provisos:
        rels.append({"parent_uid": section.uid, "child_uid": proviso.uid, "rel_type": "HAS_PROVISO"})

    for explanation in section.explanations:
        rels.append({"parent_uid": section.uid, "child_uid": explanation.uid, "rel_type": "HAS_EXPLANATION"})

    for definition in section.definitions:
        rels.append({"parent_uid": section.uid, "child_uid": definition.uid, "rel_type": "HAS_DEFINITION"})

    for clause in section.clauses:
        rels.append({"parent_uid": section.uid, "child_uid": clause.uid, "rel_type": "HAS_CLAUSE"})
        _collect_clause_rels(clause, rels)

    for subsection in section.subsections:
        rels.append({"parent_uid": section.uid, "child_uid": subsection.uid, "rel_type": "HAS_SUBSECTION"})
        _collect_subsection_rels(subsection, rels)


def _collect_subsection_rels(subsection: SubSection, rels: list[dict[str, str]]) -> None:
    """Collect structural rels for a SubSection and all descendants."""
    for proviso in subsection.provisos:
        rels.append({"parent_uid": subsection.uid, "child_uid": proviso.uid, "rel_type": "HAS_PROVISO"})

    for explanation in subsection.explanations:
        rels.append({"parent_uid": subsection.uid, "child_uid": explanation.uid, "rel_type": "HAS_EXPLANATION"})

    for definition in subsection.definitions:
        rels.append({"parent_uid": subsection.uid, "child_uid": definition.uid, "rel_type": "HAS_DEFINITION"})

    for clause in subsection.clauses:
        rels.append({"parent_uid": subsection.uid, "child_uid": clause.uid, "rel_type": "HAS_CLAUSE"})
        _collect_clause_rels(clause, rels)


def _collect_clause_rels(clause: Clause, rels: list[dict[str, str]]) -> None:
    """Collect structural rels for a Clause and all descendants."""
    for proviso in clause.provisos:
        rels.append({"parent_uid": clause.uid, "child_uid": proviso.uid, "rel_type": "HAS_PROVISO"})

    for explanation in clause.explanations:
        rels.append({"parent_uid": clause.uid, "child_uid": explanation.uid, "rel_type": "HAS_EXPLANATION"})

    for subclause in clause.subclauses:
        rels.append({"parent_uid": clause.uid, "child_uid": subclause.uid, "rel_type": "HAS_SUBCLAUSE"})
        for proviso in subclause.provisos:
            rels.append({"parent_uid": subclause.uid, "child_uid": proviso.uid, "rel_type": "HAS_PROVISO"})
        for explanation in subclause.explanations:
            rels.append({"parent_uid": subclause.uid, "child_uid": explanation.uid, "rel_type": "HAS_EXPLANATION"})


_SECTION_PATTERN = re.compile(r"section\s+(\d+)", re.IGNORECASE)


def _collect_amendment_rels(
    amendment_act: AmendmentAct,
    year: int | None = None,
) -> list[dict[str, Any]]:
    """Collect amendment relationship data from an AmendmentAct.

    Maps AmendmentEntry.removed_text -> old_text on the returned dict.
    Handles any amendment_type value (SUBSTITUTES, INSERTS, OMITS, DECRIMINALIZES, etc.).
    Logs WARNING and skips entries whose target_section cannot be parsed.

    Returns a list of dicts ready for _merge_amendment_rels().
    """
    effective_year = year or amendment_act.year
    rels: list[dict[str, Any]] = []

    for entry in amendment_act.entries:
        match = _SECTION_PATTERN.search(entry.target_section)
        if not match:
            logger.warning(
                "Cannot resolve amendment target: %s", entry.target_section
            )
            continue

        target = section_uid(int(match.group(1)))
        effective_date = f"{effective_year}-01-01"

        rels.append({
            "amend_uid": amendment_act.uid,
            "target_uid": target,
            "amendment_type": entry.amendment_type,
            "new_text": entry.new_text,
            "old_text": entry.removed_text,   # NOTE: removed_text -> old_text
            "effective_date": effective_date,
        })

    return rels


def _collect_ruleset_nodes(ruleset: RuleSet) -> dict[str, list[dict[str, Any]]]:
    """Collect all nodes from a RuleSet, grouped by Neo4j label."""
    scorer = QualityScorer()
    formatter = LLMFormatter()

    nodes: dict[str, list[dict[str, Any]]] = {
        "RuleSet": [],
        "Rule": [],
        "Form": [],
    }

    nodes["RuleSet"].append({
        "uid": ruleset.uid,
        "text": ruleset.text,
        "page_number": ruleset.page_number,
        "source_pdf": ruleset.source_pdf,
        "title": ruleset.title,
        "year": ruleset.year,
    })

    for rule in ruleset.rules:
        quality = scorer.score(rule.text, "Rule")
        nodes["Rule"].append({
            "uid": rule.uid,
            "text": rule.text,
            "page_number": rule.page_number,
            "source_pdf": rule.source_pdf,
            "number": rule.number,
            "title": rule.title,
            "quality_score": round(quality.score, 1),
            "llm_text": formatter.format_rule(rule, ruleset.title),
            "language": rule.language,
        })
        for form in rule.forms:
            nodes["Form"].append({
                "uid": form.uid,
                "text": form.text,
                "page_number": form.page_number,
                "source_pdf": form.source_pdf,
                "form_number": form.form_number,
                "title": form.title,
                "associated_rule": form.associated_rule,
            })

    return {label: batch for label, batch in nodes.items() if batch}


def _collect_ruleset_rels(ruleset: RuleSet) -> list[dict[str, str]]:
    """Collect structural rels for a RuleSet."""
    rels: list[dict[str, str]] = []
    for rule in ruleset.rules:
        rels.append({"parent_uid": ruleset.uid, "child_uid": rule.uid, "rel_type": "HAS_RULE"})
        for form in rule.forms:
            rels.append({"parent_uid": rule.uid, "child_uid": form.uid, "rel_type": "HAS_FORM"})
    return rels


# ---------------------------------------------------------------------------
# Neo4j loading helpers
# ---------------------------------------------------------------------------


def _merge_nodes(session: neo4j.Session, label: str, batch: list[dict]) -> int:
    """MERGE a batch of nodes with the given label into Neo4j.

    Uses UNWIND + ON CREATE/MATCH SET for idempotency.
    Processes in chunks of BATCH_SIZE to bound memory per transaction.

    Returns total count of nodes processed.
    """
    total = 0
    cypher = (
        f"UNWIND $batch AS row "
        f"MERGE (n:{label} {{uid: row.uid}}) "
        f"ON CREATE SET n += row "
        f"ON MATCH SET n += row"
    )
    for i in range(0, len(batch), BATCH_SIZE):
        chunk = batch[i : i + BATCH_SIZE]

        def _write_tx(tx: neo4j.ManagedTransaction, chunk: list[dict] = chunk) -> None:
            tx.run(cypher, batch=chunk)

        session.execute_write(_write_tx)
        total += len(chunk)
        logger.info("Merged %d %s nodes", len(chunk), label)

    return total


def _merge_structural_rels(session: neo4j.Session, rels: list[dict]) -> int:
    """MERGE structural relationships grouped by rel_type.

    Returns total count of relationships processed.
    """
    from collections import defaultdict

    by_type: dict[str, list[dict]] = defaultdict(list)
    for rel in rels:
        by_type[rel["rel_type"]].append(rel)

    total = 0
    for rel_type, group in by_type.items():
        if rel_type not in ALLOWED_STRUCTURAL_REL_TYPES:
            logger.warning("Skipping rels with unknown rel_type=%r", rel_type)
            continue
        cypher = (
            f"UNWIND $batch AS row "
            f"MATCH (parent {{uid: row.parent_uid}}) "
            f"MATCH (child {{uid: row.child_uid}}) "
            f"MERGE (parent)-[:{rel_type}]->(child)"
        )
        for i in range(0, len(group), BATCH_SIZE):
            chunk = group[i : i + BATCH_SIZE]

            def _write_tx(tx: neo4j.ManagedTransaction, chunk: list[dict] = chunk) -> None:
                tx.run(cypher, batch=chunk)

            session.execute_write(_write_tx)
            total += len(chunk)
            logger.info("Merged %d %s relationships", len(chunk), rel_type)

    return total


def _merge_amendment_rels(session: neo4j.Session, rels: list[dict]) -> int:
    """MERGE amendment relationships grouped by amendment_type.

    Uses date() Cypher function so effective_date is stored as a Date type.
    Stores old_text (NOT removed_text) per the data contract.

    Returns total count of relationships processed.
    """
    from collections import defaultdict

    by_type: dict[str, list[dict]] = defaultdict(list)
    for rel in rels:
        by_type[rel["amendment_type"]].append(rel)

    total = 0
    for amendment_type, group in by_type.items():
        if amendment_type not in ALLOWED_AMENDMENT_TYPES:
            logger.warning(
                "Skipping amendment relationships with invalid amendment_type=%r",
                amendment_type,
            )
            continue
        cypher = (
            f"UNWIND $batch AS row "
            f"MATCH (a:AmendmentAct {{uid: row.amend_uid}}) "
            f"MATCH (t:Section {{uid: row.target_uid}}) "
            f"MERGE (a)-[r:{amendment_type}]->(t) "
            f"ON CREATE SET r.new_text = row.new_text, r.old_text = row.old_text, r.effective_date = date(row.effective_date) "
            f"ON MATCH SET r.new_text = row.new_text, r.old_text = row.old_text, r.effective_date = date(row.effective_date)"
        )
        for i in range(0, len(group), BATCH_SIZE):
            chunk = group[i : i + BATCH_SIZE]

            def _write_tx(tx: neo4j.ManagedTransaction, chunk: list[dict] = chunk) -> None:
                tx.run(cypher, batch=chunk)

            session.execute_write(_write_tx)
            total += len(chunk)
            logger.info("Merged %d %s amendment relationships", len(chunk), amendment_type)

    return total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_act(driver: neo4j.Driver, act: Act) -> dict[str, int]:
    """Load an Act and its full hierarchy into Neo4j.

    Calls ensure_schema first, then MERGE-loads all nodes and structural rels.

    Returns:
        {"nodes": total_nodes_processed, "relationships": total_rels_processed}
    """
    from graph.schema import ensure_schema

    ensure_schema(driver)

    node_batches = _collect_act_nodes(act)
    rels = _collect_structural_rels(act)

    total_nodes = 0
    total_rels = 0

    with driver.session(database=DATABASE) as session:
        for label, batch in node_batches.items():
            total_nodes += _merge_nodes(session, label, batch)
        total_rels += _merge_structural_rels(session, rels)

    logger.info("load_act complete: %d nodes, %d relationships", total_nodes, total_rels)
    return {"nodes": total_nodes, "relationships": total_rels}


def load_amendment_act(
    driver: neo4j.Driver,
    amendment_act: AmendmentAct,
) -> dict[str, int]:
    """Load an AmendmentAct and its amendment relationships into Neo4j.

    Creates the AmendmentAct node, SUBSTITUTES/INSERTS/OMITS/DECRIMINALIZES
    relationships to targeted Sections, and an AMENDED_BY relationship from
    the AmendmentAct to the parent Act.

    Returns:
        {"nodes": total_nodes_processed, "relationships": total_rels_processed}
    """
    from graph.schema import ensure_schema

    ensure_schema(driver)

    amend_node = {
        "uid": amendment_act.uid,
        "text": amendment_act.text,
        "page_number": amendment_act.page_number,
        "source_pdf": amendment_act.source_pdf,
        "title": amendment_act.title,
        "year": amendment_act.year,
    }
    rels = _collect_amendment_rels(amendment_act)

    total_nodes = 0
    total_rels = 0

    with driver.session(database=DATABASE) as session:
        total_nodes += _merge_nodes(session, "AmendmentAct", [amend_node])
        total_rels += _merge_amendment_rels(session, rels)

        # AMENDED_BY relationship from AmendmentAct to the parent Act
        amended_by_cypher = (
            "MATCH (a:AmendmentAct {uid: $amend_uid}) "
            "MATCH (act:Act {uid: $act_uid}) "
            "MERGE (a)-[:AMENDED_BY]->(act)"
        )

        def _write_amended_by(tx: neo4j.ManagedTransaction) -> None:
            tx.run(amended_by_cypher, amend_uid=amendment_act.uid, act_uid=_DEFAULT_ACT_UID)

        session.execute_write(_write_amended_by)
        total_rels += 1

    logger.info(
        "load_amendment_act complete: %d nodes, %d relationships",
        total_nodes,
        total_rels,
    )
    return {"nodes": total_nodes, "relationships": total_rels}


def load_ruleset(driver: neo4j.Driver, ruleset: RuleSet) -> dict[str, int]:
    """Load a RuleSet and its Rules/Forms into Neo4j.

    Also creates a PRESCRIBES_FOR relationship from the RuleSet to the parent Act.

    Returns:
        {"nodes": total_nodes_processed, "relationships": total_rels_processed}
    """
    from graph.schema import ensure_schema

    ensure_schema(driver)

    node_batches = _collect_ruleset_nodes(ruleset)
    rels = _collect_ruleset_rels(ruleset)

    total_nodes = 0
    total_rels = 0

    with driver.session(database=DATABASE) as session:
        for label, batch in node_batches.items():
            total_nodes += _merge_nodes(session, label, batch)
        total_rels += _merge_structural_rels(session, rels)

        # PRESCRIBES_FOR relationship from RuleSet to the parent Act
        prescribes_cypher = (
            "MATCH (rs:RuleSet {uid: $ruleset_uid}) "
            "MATCH (act:Act {uid: $act_uid}) "
            "MERGE (rs)-[:PRESCRIBES_FOR]->(act)"
        )

        def _write_prescribes(tx: neo4j.ManagedTransaction) -> None:
            tx.run(prescribes_cypher, ruleset_uid=ruleset.uid, act_uid=_DEFAULT_ACT_UID)

        session.execute_write(_write_prescribes)
        total_rels += 1

    logger.info(
        "load_ruleset complete: %d nodes, %d relationships",
        total_nodes,
        total_rels,
    )
    return {"nodes": total_nodes, "relationships": total_rels}
