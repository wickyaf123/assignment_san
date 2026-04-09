"""
Unit tests for backend/query/executor.py.

All tests use mocked Neo4j driver — no real database required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import neo4j
import pytest


# ---------------------------------------------------------------------------
# Test 1: execute_cypher with valid MATCH query returns results
# ---------------------------------------------------------------------------


def test_execute_cypher_success():
    """execute_cypher returns dict with results list and error=None on success."""
    mock_record = MagicMock()
    mock_record.data.return_value = {"uid": "section-135", "text": "CSR provisions"}

    mock_result = MagicMock()
    mock_result.__iter__ = MagicMock(return_value=iter([mock_record]))

    mock_session = MagicMock()
    mock_session.run.return_value = mock_result
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    mock_drv = MagicMock(spec=neo4j.Driver)
    mock_drv.session.return_value = mock_session

    with patch("query.executor.get_driver", return_value=mock_drv):
        from query.executor import execute_cypher

        result = execute_cypher("MATCH (s:Section {number: '135'}) RETURN s.uid AS uid")

    assert result["error"] is None
    assert isinstance(result["results"], list)


# ---------------------------------------------------------------------------
# Test 2: execute_cypher with syntax error returns error string
# ---------------------------------------------------------------------------


def test_execute_cypher_syntax_error():
    """execute_cypher returns dict with error string and empty results on CypherSyntaxError."""
    mock_session = MagicMock()
    mock_session.run.side_effect = neo4j.exceptions.CypherSyntaxError(
        "Invalid input 'METCH'", "Neo.ClientError.Statement.SyntaxError"
    )
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    mock_drv = MagicMock(spec=neo4j.Driver)
    mock_drv.session.return_value = mock_session

    with patch("query.executor.get_driver", return_value=mock_drv):
        from query.executor import execute_cypher

        result = execute_cypher("METCH (s:Section) RETURN s")

    assert result["results"] == []
    assert result["error"] is not None
    assert isinstance(result["error"], str)
    assert len(result["error"]) > 0


# ---------------------------------------------------------------------------
# Test 3: execute_cypher with valid Cypher that returns empty results
# ---------------------------------------------------------------------------


def test_execute_cypher_empty_results():
    """execute_cypher returns dict with empty results list and error=None when no rows."""
    mock_result = MagicMock()
    mock_result.__iter__ = MagicMock(return_value=iter([]))

    mock_session = MagicMock()
    mock_session.run.return_value = mock_result
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    mock_drv = MagicMock(spec=neo4j.Driver)
    mock_drv.session.return_value = mock_session

    with patch("query.executor.get_driver", return_value=mock_drv):
        from query.executor import execute_cypher

        result = execute_cypher("MATCH (s:Section {number: '999'}) RETURN s")

    assert result["results"] == []
    assert result["error"] is None


# ---------------------------------------------------------------------------
# Test 4: execute_fulltext_fallback returns results above threshold
# ---------------------------------------------------------------------------


def test_execute_fulltext_fallback_returns_results():
    """execute_fulltext_fallback returns list of result dicts when matches above threshold exist."""
    mock_record = {
        "node_props": {"uid": "section-135", "text": "CSR", "title": None, "term": None, "number": "135"},
        "score": 0.85,
        "node_type": "Section",
    }

    mock_result = MagicMock()
    mock_result.__iter__ = MagicMock(return_value=iter([mock_record]))

    mock_session = MagicMock()
    mock_session.run.return_value = mock_result
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    mock_drv = MagicMock(spec=neo4j.Driver)
    mock_drv.session.return_value = mock_session

    with patch("query.executor.get_driver", return_value=mock_drv):
        from query.executor import execute_fulltext_fallback

        results = execute_fulltext_fallback("corporate social responsibility")

    assert isinstance(results, list)
    assert len(results) > 0


# ---------------------------------------------------------------------------
# Test 5: execute_fulltext_fallback returns empty list when no matches
# ---------------------------------------------------------------------------


def test_execute_fulltext_fallback_no_matches():
    """execute_fulltext_fallback returns empty list when fulltext query returns no results."""
    mock_result = MagicMock()
    mock_result.__iter__ = MagicMock(return_value=iter([]))

    mock_session = MagicMock()
    mock_session.run.return_value = mock_result
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    mock_drv = MagicMock(spec=neo4j.Driver)
    mock_drv.session.return_value = mock_session

    with patch("query.executor.get_driver", return_value=mock_drv):
        from query.executor import execute_fulltext_fallback

        results = execute_fulltext_fallback("xyzzy nonexistent query")

    assert results == []


# ---------------------------------------------------------------------------
# Test 6: execute_cypher wraps ClientError into error string (not re-raised)
# ---------------------------------------------------------------------------


def test_execute_cypher_client_error_not_reraised():
    """execute_cypher catches ClientError and returns it as error string, not raises."""
    mock_session = MagicMock()
    mock_session.run.side_effect = neo4j.exceptions.ClientError(
        "Authentication failure", "Neo.ClientError.Security.Unauthorized"
    )
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    mock_drv = MagicMock(spec=neo4j.Driver)
    mock_drv.session.return_value = mock_session

    with patch("query.executor.get_driver", return_value=mock_drv):
        from query.executor import execute_cypher

        # Must not raise — exception is captured as error string
        result = execute_cypher("MATCH (s:Section) RETURN s")

    assert result["results"] == []
    assert result["error"] is not None
    assert isinstance(result["error"], str)


# ---------------------------------------------------------------------------
# Test 7: execute_fulltext_fallback with ClientError (index not found) returns empty list
# ---------------------------------------------------------------------------


def test_execute_fulltext_fallback_client_error_returns_empty():
    """execute_fulltext_fallback returns empty list on ClientError (e.g. index not found)."""
    mock_session = MagicMock()
    mock_session.run.side_effect = neo4j.exceptions.ClientError(
        "There is no such fulltext schema index: legal_text_search",
        "Neo.ClientError.Procedure.ProcedureCallFailed",
    )
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    mock_drv = MagicMock(spec=neo4j.Driver)
    mock_drv.session.return_value = mock_session

    with patch("query.executor.get_driver", return_value=mock_drv):
        from query.executor import execute_fulltext_fallback

        results = execute_fulltext_fallback("some query")

    assert results == []


# ---------------------------------------------------------------------------
# Test 8: FULLTEXT_INDEX_NAME constant is correct
# ---------------------------------------------------------------------------


def test_fulltext_index_name_constant():
    """FULLTEXT_INDEX_NAME constant equals 'legal_text_search'."""
    from query.executor import FULLTEXT_INDEX_NAME

    assert FULLTEXT_INDEX_NAME == "legal_text_search"


# ---------------------------------------------------------------------------
# Test 9: BM25_SCORE_THRESHOLD constant is 0.5
# ---------------------------------------------------------------------------


def test_bm25_score_threshold_constant():
    """BM25_SCORE_THRESHOLD constant equals 0.3."""
    from query.executor import BM25_SCORE_THRESHOLD

    assert BM25_SCORE_THRESHOLD == 0.3
