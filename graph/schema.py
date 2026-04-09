"""
Neo4j schema setup for ViddhiAI — unique constraints and range indexes.

14 unique constraints (one per node type) ensure MERGE idempotency.
4 range indexes on commonly queried properties speed up Cypher lookups.

All statements use IF NOT EXISTS so they are safe to run repeatedly.
"""

from __future__ import annotations

import logging

import neo4j

from graph.connection import DATABASE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 14 unique constraints — one per node type.
# SubClause is the 14th type (not Penalty — that was a PRD error).
# ---------------------------------------------------------------------------
CONSTRAINT_STATEMENTS: list[str] = [
    "CREATE CONSTRAINT act_uid IF NOT EXISTS FOR (n:Act) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT chapter_uid IF NOT EXISTS FOR (n:Chapter) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT section_uid IF NOT EXISTS FOR (n:Section) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT subsection_uid IF NOT EXISTS FOR (n:SubSection) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT clause_uid IF NOT EXISTS FOR (n:Clause) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT subclause_uid IF NOT EXISTS FOR (n:SubClause) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT proviso_uid IF NOT EXISTS FOR (n:Proviso) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT explanation_uid IF NOT EXISTS FOR (n:Explanation) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT definition_uid IF NOT EXISTS FOR (n:Definition) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT rule_uid IF NOT EXISTS FOR (n:Rule) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT ruleset_uid IF NOT EXISTS FOR (n:RuleSet) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT form_uid IF NOT EXISTS FOR (n:Form) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT schedule_uid IF NOT EXISTS FOR (n:Schedule) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT amendmentact_uid IF NOT EXISTS FOR (n:AmendmentAct) REQUIRE n.uid IS UNIQUE",
]

# ---------------------------------------------------------------------------
# 4 range indexes on commonly queried scalar properties.
# ---------------------------------------------------------------------------
INDEX_STATEMENTS: list[str] = [
    "CREATE INDEX section_number IF NOT EXISTS FOR (n:Section) ON (n.number)",
    "CREATE INDEX definition_term IF NOT EXISTS FOR (n:Definition) ON (n.term)",
    "CREATE INDEX rule_number IF NOT EXISTS FOR (n:Rule) ON (n.number)",
    "CREATE INDEX chapter_number IF NOT EXISTS FOR (n:Chapter) ON (n.number)",
]

# ---------------------------------------------------------------------------
# 1 fulltext index for BM25 fallback search across all text-bearing node types.
# Covers text, title, and term properties across all content nodes so the
# fallback path can find SubSections, Clauses, Provisos, etc. — not just
# Sections, Definitions, and Rules.
# ---------------------------------------------------------------------------
FULLTEXT_INDEX_STATEMENTS: list[str] = [
    "CREATE FULLTEXT INDEX legal_text_search IF NOT EXISTS "
    "FOR (n:Section|SubSection|Clause|SubClause|Proviso|Explanation|Definition|Rule|Schedule|Form) "
    "ON EACH [n.text, n.title, n.term]",
]


def ensure_schema(driver: neo4j.Driver) -> None:
    """Create all constraints, indexes, and fulltext indexes in Neo4j if they do not already exist.

    Safe to call multiple times — all statements use IF NOT EXISTS.

    Args:
        driver: Active Neo4j driver instance.
    """
    with driver.session(database=DATABASE) as session:
        for stmt in CONSTRAINT_STATEMENTS:
            session.run(stmt)
            logger.info("Created constraint: %s...", stmt[:60])

        for stmt in INDEX_STATEMENTS:
            session.run(stmt)
            logger.info("Created index: %s...", stmt[:60])

        for stmt in FULLTEXT_INDEX_STATEMENTS:
            session.run(stmt)
            logger.info("Created fulltext index: %s...", stmt[:60])

    logger.info(
        "Schema setup complete: %d constraints, %d indexes, %d fulltext indexes",
        len(CONSTRAINT_STATEMENTS),
        len(INDEX_STATEMENTS),
        len(FULLTEXT_INDEX_STATEMENTS),
    )
