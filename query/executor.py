"""
Neo4j Cypher executor with retry state management and BM25 fulltext fallback.

Responsibilities:
- Execute read-only Cypher queries against Neo4j, capturing errors rather than
  re-raising (so the LangGraph flow can route to retry or fallback).
- Provide BM25 fulltext fallback via the legal_text_search fulltext index when
  structured Cypher returns no results after max retries.

Security notes:
- T-03-10: execute_cypher receives Cypher that has ALREADY passed validate_cypher().
  The flow enforces: validate_node -> execute_node. This module does NOT re-validate.
- T-03-11: execute_fulltext_fallback uses parameterized Cypher ($query_text,
  $index_name, $threshold, $limit) — no string interpolation, preventing Cypher
  injection in the fulltext path.
- T-03-12: BM25_RESULT_LIMIT = 15 caps fulltext results to prevent DoS.
"""

from __future__ import annotations

import logging
import re

import neo4j

from graph.connection import DATABASE, get_driver

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FULLTEXT_INDEX_NAME = "legal_text_search"
BM25_SCORE_THRESHOLD = 0.3
BM25_RESULT_LIMIT = 15

# Parameterized fulltext query — all user input is passed as Neo4j parameters
# so there is no Cypher injection path (T-03-11).
FULLTEXT_QUERY = """
CALL db.index.fulltext.queryNodes($index_name, $query_text)
YIELD node, score
WHERE score > $threshold
RETURN node { .uid, .text, .title, .term, .number } AS node_props,
       score,
       labels(node)[0] AS node_type
ORDER BY score DESC
LIMIT $limit
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def execute_cypher(cypher: str) -> dict:
    """Execute a read-only Cypher query against Neo4j.

    Errors are captured and returned in the result dict rather than re-raised,
    so the LangGraph flow can route the state to a retry or fallback node.

    Args:
        cypher: A read-only Cypher query (already validated by validator.py).

    Returns:
        dict with keys:
            - "results": list[dict] — Neo4j records as dicts (empty on error)
            - "error": str | None — error message if execution failed, else None
    """
    driver = get_driver()
    logger.info("Executing Cypher: %s", cypher[:100])

    try:
        with driver.session(database=DATABASE) as session:
            result = session.run(cypher)
            records = [dict(record) for record in result]

        logger.info("Cypher returned %d records", len(records))
        return {"results": records, "error": None}

    except neo4j.exceptions.CypherSyntaxError as exc:
        logger.warning("Cypher syntax error: %s", exc)
        return {"results": [], "error": str(exc)}

    except neo4j.exceptions.ClientError as exc:
        logger.warning("Neo4j ClientError during Cypher execution: %s", exc)
        return {"results": [], "error": str(exc)}


_STOPWORDS: frozenset[str] = frozenset({
    "what", "which", "who", "how", "is", "are", "was", "were", "the", "a", "an",
    "of", "in", "for", "to", "and", "or", "by", "on", "at", "from", "with",
    "does", "do", "did", "has", "have", "had", "be", "been", "being", "it",
    "its", "that", "this", "those", "these", "there", "their", "they", "them",
    "about", "between", "under", "after", "before", "if", "when", "where",
    "all", "any", "no", "not", "can", "shall", "may", "will", "would",
    "should", "could", "also", "me", "my", "we", "our", "you", "your",
    "list", "show", "find", "tell", "trace", "full", "chain", "consequences",
    "give", "get", "make", "let", "know",
})

_SECTION_RE = re.compile(r"section\s+(\d+)", re.IGNORECASE)
_RULE_RE = re.compile(r"rule\s+(\d+)", re.IGNORECASE)


def _extract_search_terms(query_text: str) -> str:
    """Extract meaningful legal keywords from a natural language query for BM25.

    Pulls out section/rule references and non-stopword terms, then joins them
    into a Lucene-friendly query string that ranks better than raw sentences.
    """
    terms: list[str] = []

    for m in _SECTION_RE.finditer(query_text):
        terms.append(f"section {m.group(1)}")
    for m in _RULE_RE.finditer(query_text):
        terms.append(f"rule {m.group(1)}")

    words = re.findall(r"[a-zA-Z]+", query_text.lower())
    keywords = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    terms.extend(keywords)

    if not terms:
        return query_text

    return " ".join(terms)


def execute_fulltext_fallback(query_text: str) -> list[dict]:
    """Execute a BM25 fulltext search as a fallback when structured Cypher yields no results.

    Uses the legal_text_search fulltext index covering Section.text, Section.title,
    Definition.term, Definition.text, and Rule.text.

    Extracts legal keywords from the query before searching to improve BM25
    relevance (raw NL sentences score poorly against legal text).

    All parameters are passed as Neo4j driver parameters — no string interpolation
    anywhere in the query (T-03-11).

    Args:
        query_text: The natural language query text from the user.

    Returns:
        List of result dicts from the fulltext index (empty list if no matches or error).
    """
    driver = get_driver()
    search_text = _extract_search_terms(query_text)
    logger.info("BM25 fulltext fallback — original: %s | search: %s", query_text[:60], search_text[:80])

    try:
        with driver.session(database=DATABASE) as session:
            result = session.run(
                FULLTEXT_QUERY,
                index_name=FULLTEXT_INDEX_NAME,
                query_text=search_text,
                threshold=BM25_SCORE_THRESHOLD,
                limit=BM25_RESULT_LIMIT,
            )
            records = [dict(record) for record in result]

        logger.info("BM25 fallback returned %d results", len(records))
        return records

    except neo4j.exceptions.ClientError as exc:
        logger.warning(
            "Neo4j ClientError during fulltext fallback (index may not exist): %s", exc
        )
        return []
