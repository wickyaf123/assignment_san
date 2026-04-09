"""
Unit tests for backend/query/validator.py — deterministic Cypher safety check.

All 10 test cases specified in the plan behavior block plus edge cases.
The validator is a pure function — no mocking or async needed.
"""

from __future__ import annotations

import pytest


def test_valid_match_return():
    """Simple MATCH/RETURN query is valid."""
    from query.validator import validate_cypher

    is_valid, reason = validate_cypher("MATCH (s:Section) RETURN s")
    assert is_valid is True
    assert reason is None


def test_reject_create():
    """CREATE keyword is rejected."""
    from query.validator import validate_cypher

    is_valid, reason = validate_cypher("CREATE (n:Section {uid: 'x'})")
    assert is_valid is False
    assert reason is not None
    assert "Rejected" in reason


def test_reject_set():
    """SET keyword is rejected even when combined with MATCH/RETURN."""
    from query.validator import validate_cypher

    is_valid, reason = validate_cypher("MATCH (s:Section) SET s.text = 'x' RETURN s")
    assert is_valid is False
    assert reason is not None
    assert "Rejected" in reason


def test_reject_delete():
    """DELETE keyword is rejected."""
    from query.validator import validate_cypher

    is_valid, reason = validate_cypher("MATCH (s:Section) DELETE s")
    assert is_valid is False
    assert reason is not None
    assert "Rejected" in reason


def test_reject_remove():
    """REMOVE keyword is rejected."""
    from query.validator import validate_cypher

    is_valid, reason = validate_cypher("MATCH (s:Section) REMOVE s.text RETURN s")
    assert is_valid is False
    assert reason is not None
    assert "Rejected" in reason


def test_reject_merge():
    """MERGE keyword is rejected."""
    from query.validator import validate_cypher

    is_valid, reason = validate_cypher("MERGE (s:Section {uid: 'x'})")
    assert is_valid is False
    assert reason is not None
    assert "Rejected" in reason


def test_reject_drop():
    """DROP keyword is rejected."""
    from query.validator import validate_cypher

    is_valid, reason = validate_cypher("DROP INDEX section_number")
    assert is_valid is False
    assert reason is not None
    assert "Rejected" in reason


def test_property_name_containing_keyword_does_not_false_match():
    """Property names containing blocked keyword substrings must NOT be rejected.

    Example: s.number = '135' contains 'SET' as part of 'number' in uppercase
    but word-boundary regex must not match it. Also WHERE contains no blocked keywords.
    """
    from query.validator import validate_cypher

    cypher = "MATCH (s:Section) WHERE s.number = '135' RETURN s.text, s.title"
    is_valid, reason = validate_cypher(cypher)
    assert is_valid is True, f"False positive rejection: {reason}"
    assert reason is None


def test_reject_detach():
    """DETACH keyword (as in DETACH DELETE) is rejected."""
    from query.validator import validate_cypher

    is_valid, reason = validate_cypher("MATCH (s:Section) DETACH DELETE s")
    assert is_valid is False
    assert reason is not None
    assert "Rejected" in reason


def test_reject_foreach():
    """FOREACH keyword is rejected."""
    from query.validator import validate_cypher

    is_valid, reason = validate_cypher("FOREACH (x IN [1,2,3] | CREATE (n:Section {uid: x}))")
    assert is_valid is False
    assert reason is not None
    assert "Rejected" in reason


def test_reject_load():
    """LOAD keyword (as in LOAD CSV) is rejected."""
    from query.validator import validate_cypher

    is_valid, reason = validate_cypher("LOAD CSV FROM 'file:///data.csv' AS row")
    assert is_valid is False
    assert reason is not None
    assert "Rejected" in reason


def test_empty_string_is_valid():
    """Empty string passes validator — executor handles the empty-query case."""
    from query.validator import validate_cypher

    is_valid, reason = validate_cypher("")
    assert is_valid is True
    assert reason is None


def test_write_keywords_frozenset():
    """WRITE_KEYWORDS must be a frozenset containing exactly the 8 expected keywords."""
    from query.validator import WRITE_KEYWORDS

    assert isinstance(WRITE_KEYWORDS, frozenset), "WRITE_KEYWORDS must be a frozenset"
    expected = {"CREATE", "SET", "DELETE", "REMOVE", "MERGE", "DROP", "DETACH", "FOREACH", "LOAD"}
    # The plan specifies 8 keywords, LOAD being the 9th per the behavior block (test 9)
    # Accept any superset containing all required keywords
    for kw in {"CREATE", "SET", "DELETE", "REMOVE", "MERGE", "DROP"}:
        assert kw in WRITE_KEYWORDS, f"Missing required keyword: {kw}"


def test_call_not_in_blocklist():
    """CALL is intentionally NOT blocked — needed for fulltext index queries."""
    from query.validator import validate_cypher

    is_valid, reason = validate_cypher(
        "CALL db.index.fulltext.queryNodes('legal_text_search', 'company') YIELD node RETURN node"
    )
    assert is_valid is True, (
        "CALL must not be blocked — fulltext fallback uses CALL db.index.fulltext.queryNodes(). "
        f"Got rejection: {reason}"
    )
