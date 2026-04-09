"""
TDD tests for Task 1: dependency installation, fulltext index, and schema coverage.

These tests verify:
1. New packages (langgraph, langchain_google_genai, langsmith) are importable
2. schema.FULLTEXT_INDEX_STATEMENTS contains exactly one entry with 'legal_text_search'
3. ensure_schema() iterates FULLTEXT_INDEX_STATEMENTS (via mock driver)
"""

from __future__ import annotations

from unittest.mock import MagicMock, call


def test_langgraph_importable():
    """langgraph must be installed and importable."""
    import langgraph  # noqa: F401


def test_langchain_google_genai_importable():
    """langchain_google_genai must be installed and importable."""
    import langchain_google_genai  # noqa: F401


def test_langsmith_importable():
    """langsmith must be installed and importable."""
    import langsmith  # noqa: F401


def test_fulltext_index_statements_exists():
    """FULLTEXT_INDEX_STATEMENTS list must exist in graph.schema."""
    from graph.schema import FULLTEXT_INDEX_STATEMENTS

    assert isinstance(FULLTEXT_INDEX_STATEMENTS, list), "FULLTEXT_INDEX_STATEMENTS must be a list"


def test_fulltext_index_statements_length():
    """FULLTEXT_INDEX_STATEMENTS must contain exactly one entry."""
    from graph.schema import FULLTEXT_INDEX_STATEMENTS

    assert len(FULLTEXT_INDEX_STATEMENTS) == 1, (
        f"Expected 1 fulltext index statement, got {len(FULLTEXT_INDEX_STATEMENTS)}"
    )


def test_fulltext_index_contains_legal_text_search():
    """The fulltext index statement must reference 'legal_text_search'."""
    from graph.schema import FULLTEXT_INDEX_STATEMENTS

    assert "legal_text_search" in FULLTEXT_INDEX_STATEMENTS[0], (
        "Fulltext index must be named 'legal_text_search'"
    )


def test_ensure_schema_runs_fulltext_indexes(mock_driver):
    """ensure_schema() must iterate FULLTEXT_INDEX_STATEMENTS after INDEX_STATEMENTS."""
    from graph.schema import FULLTEXT_INDEX_STATEMENTS, ensure_schema

    mock_drv, mock_session = mock_driver
    ensure_schema(mock_drv)

    # Extract all the cypher strings run against the session
    run_calls = [c.args[0] for c in mock_session.run.call_args_list]

    # The fulltext index statement must have been called
    for stmt in FULLTEXT_INDEX_STATEMENTS:
        assert stmt in run_calls, (
            f"ensure_schema() did not run fulltext index: {stmt!r}"
        )
