"""
Unit tests for query.classifier module.

Tests verify:
- classify_intent() returns a string via keyword/regex matching
- Amendment, penalty, cross-reference, and general_info detection
- All 5 intent categories are handled correctly
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Test 1: classify_intent returns a string
# ---------------------------------------------------------------------------


def test_classify_intent_returns_string():
    """classify_intent must return the intent string."""
    from query.classifier import classify_intent

    result = classify_intent("What does Section 135 say?")
    assert isinstance(result, str)
    assert result == "structured_lookup"


# ---------------------------------------------------------------------------
# Test 2: Amendment intent detection via keyword matching
# ---------------------------------------------------------------------------


def test_classify_intent_amendment_detection():
    """classify_intent must detect amendment-related queries."""
    from query.classifier import classify_intent

    assert classify_intent("How was Section 135 amended?") == "amendment_query"
    assert classify_intent("What changes were made to Section 149?") == "amendment_query"
    assert classify_intent("Was Section 185 modified by any amendment?") == "amendment_query"


# ---------------------------------------------------------------------------
# Test 3: Penalty intent detection (requires section reference)
# ---------------------------------------------------------------------------


def test_classify_intent_penalty_detection():
    """classify_intent must detect penalty queries with section references."""
    from query.classifier import classify_intent

    assert classify_intent("What is the penalty under Section 447?") == "penalty_query"
    assert classify_intent("What is the fine for failure to hold AGM under Section 99?") == "penalty_query"
    # Vague penalty query without section reference -> NOT penalty_query
    assert classify_intent("What is the penalty?") != "penalty_query"


# ---------------------------------------------------------------------------
# Tests 4-8: All 5 intent categories are handled correctly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected_intent",
    [
        ("What does Section 135 say?", "structured_lookup"),
        ("How was Section 135 amended?", "amendment_query"),
        ("Which sections refer to Section 135?", "cross_reference"),
        ("What is the penalty under Section 447?", "penalty_query"),
        ("How many chapters are in the Companies Act?", "general_info"),
    ],
)
def test_classify_intent_all_categories(query: str, expected_intent: str):
    """classify_intent must correctly classify queries into all 5 intent categories."""
    from query.classifier import classify_intent

    result = classify_intent(query)
    assert result == expected_intent, f"Expected '{expected_intent}', got '{result}' for query: {query}"
    assert isinstance(result, str)
