"""
Deterministic Cypher write-safety validator for ViddhiAI.

Rejects any Cypher containing write keywords using word-boundary regex.
This is the first line of defence before any LLM-generated query reaches Neo4j.

Security note (T-03-01): Uses word-boundary \\b regex to avoid false positives
on property names that contain keyword substrings (e.g., s.number contains 'SET'
as a substring of 'number' when uppercased — word boundary prevents this match).

CALL is intentionally NOT in the blocklist:
  The fulltext fallback uses CALL db.index.fulltext.queryNodes(...) internally
  within executor.py. Blocking CALL would break BM25 fallback searches.
  Phase 4 (POST /api/cypher user-submitted endpoint) will add CALL to the
  blocklist for external/user-submitted queries only.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Write keyword blocklist — any Cypher containing these (as whole words) is
# rejected before execution. Frozen for O(1) membership checks.
# ---------------------------------------------------------------------------
WRITE_KEYWORDS: frozenset[str] = frozenset(
    {
        "CREATE",
        "SET",
        "DELETE",
        "REMOVE",
        "MERGE",
        "DROP",
        "DETACH",
        "FOREACH",
        "LOAD",
    }
)


def validate_cypher(cypher: str) -> tuple[bool, str | None]:
    """Validate that a Cypher query is read-only.

    Uses word-boundary regex to prevent false positives on property/variable names
    that happen to contain write-keyword substrings.

    Args:
        cypher: The Cypher query string to validate.

    Returns:
        (True, None) if the query is safe to execute.
        (False, rejection_reason) if a write keyword was detected.

    Examples:
        >>> validate_cypher("MATCH (s:Section) RETURN s")
        (True, None)
        >>> validate_cypher("CREATE (n:Section {uid: 'x'})")
        (False, "Rejected: Cypher contains write keyword 'CREATE'")
        >>> validate_cypher("")
        (True, None)
    """
    upper = cypher.upper()
    for kw in WRITE_KEYWORDS:
        if re.search(rf"\b{kw}\b", upper):
            return False, f"Rejected: Cypher contains write keyword '{kw}'"
    return True, None
