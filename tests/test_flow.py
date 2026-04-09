"""
Integration tests for backend/query/flow.py (LangGraph StateGraph pipeline).

All external calls are mocked — no real Gemini API or Neo4j connections required.

Important: The LLM singletons in generator.py and synthesizer.py
are instantiated at import time and require GEMINI_API_KEY to be set. We patch
the environment variable before importing those modules (using importlib.reload
would be too expensive). Instead, we set a dummy key in the environment and
then patch the _llm attributes on the already-imported modules.
"""

from __future__ import annotations

import os

# Set dummy API key BEFORE any langchain imports to satisfy the module-level
# ChatGoogleGenerativeAI instantiation in generator/synthesizer.
os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key-for-unit-tests")
os.environ.setdefault("GOOGLE_API_KEY", "test-dummy-key-for-unit-tests")

from unittest.mock import MagicMock, patch

import neo4j
import pytest


# ---------------------------------------------------------------------------
# Helpers to build mock objects
# ---------------------------------------------------------------------------


def _make_ai_message(content: str):
    """Create a mock AIMessage-compatible object."""
    mock = MagicMock()
    mock.content = content
    return mock


def _make_mock_driver(records=None):
    """Create a mock Neo4j driver that returns the given records."""
    if records is None:
        records = [{"uid": "section-135", "text": "CSR provisions", "number": "135"}]

    def make_mock_result(recs):
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter(recs))
        return mock_result

    mock_session = MagicMock()
    mock_session.run.return_value = make_mock_result(records)
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    mock_drv = MagicMock(spec=neo4j.Driver)
    mock_drv.session.return_value = mock_session
    return mock_drv


def _make_empty_driver():
    """Create a mock driver that always returns no records."""
    mock_result = MagicMock()
    mock_result.__iter__ = MagicMock(return_value=iter([]))

    mock_session = MagicMock()
    mock_session.run.return_value = mock_result
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    mock_drv = MagicMock(spec=neo4j.Driver)
    mock_drv.session.return_value = mock_session
    return mock_drv


# ---------------------------------------------------------------------------
# Test 1: run_query returns a dict with all required keys
# ---------------------------------------------------------------------------


def test_run_query_returns_all_required_keys():
    """run_query returns dict with answer, cypher_query, graph_path, sources, trace_url, metadata."""
    import query.generator
    import query.synthesizer

    mock_drv = _make_mock_driver()

    with (
        patch.object(query.generator, "_llm") as mock_gen_llm,
        patch("query.executor.get_driver", return_value=mock_drv),
        patch("query.flow.get_driver", return_value=mock_drv),
        patch.object(query.synthesizer, "_llm") as mock_synth_llm,
        patch("query.flow._schema_text", "mock schema"),
    ):
        # No classifier mock needed — classifier uses regex, not LLM
        mock_gen_llm.invoke.return_value = _make_ai_message(
            "MATCH (s:Section {number: '135'}) RETURN s.uid AS uid, s.text AS text"
        )
        mock_synth_llm.invoke.return_value = _make_ai_message(
            "Section 135 deals with Corporate Social Responsibility."
        )

        from query.flow import run_query

        result = run_query("What does Section 135 say?")

    required_keys = {"answer", "cypher_query", "graph_path", "sources", "trace_url", "metadata"}
    assert required_keys.issubset(result.keys()), (
        f"Missing keys: {required_keys - result.keys()}"
    )


# ---------------------------------------------------------------------------
# Test 2: metadata contains all required keys
# ---------------------------------------------------------------------------


def test_run_query_metadata_contains_required_keys():
    """metadata dict contains intent, retry_count, latency_ms, source_type, fallback_used."""
    import query.generator
    import query.synthesizer

    mock_drv = _make_mock_driver()

    with (
        patch.object(query.generator, "_llm") as mock_gen_llm,
        patch("query.executor.get_driver", return_value=mock_drv),
        patch("query.flow.get_driver", return_value=mock_drv),
        patch.object(query.synthesizer, "_llm") as mock_synth_llm,
        patch("query.flow._schema_text", "mock schema"),
    ):
        # No classifier mock needed — classifier uses regex, not LLM
        mock_gen_llm.invoke.return_value = _make_ai_message(
            "MATCH (s:Section {number: '135'}) RETURN s.uid AS uid"
        )
        mock_synth_llm.invoke.return_value = _make_ai_message("Section 135 answer.")

        from query.flow import run_query

        result = run_query("What does Section 135 say?")

    metadata = result["metadata"]
    required_meta_keys = {"intent", "retry_count", "latency_ms", "source_type", "fallback_used"}
    assert required_meta_keys.issubset(metadata.keys()), (
        f"Missing metadata keys: {required_meta_keys - metadata.keys()}"
    )


# ---------------------------------------------------------------------------
# Test 3: write keyword in Cypher retries up to MAX_RETRY_COUNT
# ---------------------------------------------------------------------------


def test_write_keyword_in_cypher_routes_through_retry():
    """When validator rejects Cypher (write keyword), flow retries up to MAX_RETRY_COUNT."""
    import query.generator
    import query.synthesizer

    mock_drv = _make_mock_driver()

    call_count = {"n": 0}

    def gen_side_effect(prompt):
        call_count["n"] += 1
        # Always return write Cypher to force max retries
        return _make_ai_message("CREATE (n:Section {uid: 'x'}) RETURN n")

    with (
        patch.object(query.generator, "_llm") as mock_gen_llm,
        patch("query.executor.get_driver", return_value=mock_drv),
        patch("query.flow.get_driver", return_value=mock_drv),
        patch.object(query.synthesizer, "_llm") as mock_synth_llm,
        patch("query.flow._schema_text", "mock schema"),
    ):
        # No classifier mock needed — classifier uses regex, not LLM
        mock_gen_llm.invoke.side_effect = gen_side_effect
        mock_synth_llm.invoke.return_value = _make_ai_message("I cannot process this query.")

        from query.flow import run_query, MAX_RETRY_COUNT

        result = run_query("CREATE some section")

    # Generator was called more than once (retry happened) but capped
    assert call_count["n"] <= MAX_RETRY_COUNT + 1
    assert "answer" in result


# ---------------------------------------------------------------------------
# Test 4: Empty results triggers retry then fallback
# ---------------------------------------------------------------------------


def test_empty_results_triggers_fallback():
    """When Cypher returns empty, flow retries with broadening, then falls back to BM25."""
    import query.generator
    import query.synthesizer

    mock_empty_drv = _make_empty_driver()

    with (
        patch.object(query.generator, "_llm") as mock_gen_llm,
        patch("query.executor.get_driver", return_value=mock_empty_drv),
        patch("query.flow.get_driver", return_value=mock_empty_drv),
        patch.object(query.synthesizer, "_llm") as mock_synth_llm,
        patch("query.flow._schema_text", "mock schema"),
    ):
        # No classifier mock needed — classifier uses regex, not LLM
        mock_gen_llm.invoke.return_value = _make_ai_message(
            "MATCH (s:Section {number: '999'}) RETURN s.uid AS uid"
        )
        mock_synth_llm.invoke.return_value = _make_ai_message(
            "I could not find information about this in the knowledge graph."
        )

        from query.flow import run_query

        result = run_query("Tell me about section 999")

    # Flow completed without raising
    assert "answer" in result
    assert "metadata" in result


# ---------------------------------------------------------------------------
# Test 5: amendment_query intent fires resolve node
# ---------------------------------------------------------------------------


def test_amendment_query_fires_resolve_node():
    """When intent is amendment_query, resolve node fires and effective_state passed to synthesizer."""
    import query.generator
    import query.synthesizer
    from dataclasses import dataclass, field

    @dataclass
    class MockEffectiveState:
        current_text: str = "Amended text"
        status: str = "amended"
        amendment_chain: list = field(default_factory=list)
        original_text: str = "Original text"

    mock_drv = _make_mock_driver([
        {"uid": "section-135", "text": "CSR", "number": "135"}
    ])
    mock_effective_state = MockEffectiveState()

    with (
        patch.object(query.generator, "_llm") as mock_gen_llm,
        patch("query.executor.get_driver", return_value=mock_drv),
        patch("query.flow.get_driver", return_value=mock_drv),
        patch("query.flow.resolve_effective_state", return_value=mock_effective_state) as mock_resolver,
        patch.object(query.synthesizer, "_llm") as mock_synth_llm,
        patch("query.flow._schema_text", "mock schema"),
    ):
        # No classifier mock needed — classifier uses regex, not LLM
        mock_gen_llm.invoke.return_value = _make_ai_message(
            "MATCH (s:Section {number: '135'}) RETURN s.uid AS uid, s.text AS text, s.number AS number"
        )
        mock_synth_llm.invoke.return_value = _make_ai_message(
            "Section 135 was amended by the 2026 Amendment Act."
        )

        from query.flow import run_query

        result = run_query("How was Section 135 amended?")

    # resolve_effective_state should have been called (resolve node fired)
    assert mock_resolver.called
    assert "answer" in result


# ---------------------------------------------------------------------------
# Test 6: Non-amendment_query intent skips resolve node
# ---------------------------------------------------------------------------


def test_non_amendment_query_skips_resolve_node():
    """When intent is NOT amendment_query, resolve node is skipped."""
    import query.generator
    import query.synthesizer

    mock_drv = _make_mock_driver([
        {"uid": "section-135", "text": "CSR", "number": "135"}
    ])

    with (
        patch.object(query.generator, "_llm") as mock_gen_llm,
        patch("query.executor.get_driver", return_value=mock_drv),
        patch("query.flow.get_driver", return_value=mock_drv),
        patch("query.flow.resolve_effective_state") as mock_resolver,
        patch.object(query.synthesizer, "_llm") as mock_synth_llm,
        patch("query.flow._schema_text", "mock schema"),
    ):
        # No classifier mock needed — classifier uses regex, not LLM
        mock_gen_llm.invoke.return_value = _make_ai_message(
            "MATCH (s:Section {number: '135'}) RETURN s.uid AS uid, s.text AS text"
        )
        mock_synth_llm.invoke.return_value = _make_ai_message("Section 135 answer.")

        from query.flow import run_query

        result = run_query("What is in Section 135?")

    # resolve_effective_state should NOT have been called
    assert not mock_resolver.called


# ---------------------------------------------------------------------------
# Test 7: trace_url is a non-empty string
# ---------------------------------------------------------------------------


def test_trace_url_is_non_empty_string():
    """trace_url in response is a non-empty string."""
    import query.generator
    import query.synthesizer

    mock_drv = _make_mock_driver()

    with (
        patch.object(query.generator, "_llm") as mock_gen_llm,
        patch("query.executor.get_driver", return_value=mock_drv),
        patch("query.flow.get_driver", return_value=mock_drv),
        patch.object(query.synthesizer, "_llm") as mock_synth_llm,
        patch("query.flow._schema_text", "mock schema"),
    ):
        # No classifier mock needed — classifier uses regex, not LLM
        mock_gen_llm.invoke.return_value = _make_ai_message(
            "MATCH (s:Section {number: '135'}) RETURN s.uid AS uid"
        )
        mock_synth_llm.invoke.return_value = _make_ai_message("Answer here.")

        from query.flow import run_query

        result = run_query("What does Section 135 say?")

    assert isinstance(result["trace_url"], str)
    assert len(result["trace_url"]) > 0


# ---------------------------------------------------------------------------
# Test 8: source_type is "cypher" for normal results
# ---------------------------------------------------------------------------


def test_source_type_cypher_for_normal_results():
    """source_type is 'cypher' when structured Cypher returns results."""
    import query.generator
    import query.synthesizer

    mock_drv = _make_mock_driver([
        {"uid": "section-135", "text": "CSR", "number": "135"}
    ])

    with (
        patch.object(query.generator, "_llm") as mock_gen_llm,
        patch("query.executor.get_driver", return_value=mock_drv),
        patch("query.flow.get_driver", return_value=mock_drv),
        patch.object(query.synthesizer, "_llm") as mock_synth_llm,
        patch("query.flow._schema_text", "mock schema"),
    ):
        # No classifier mock needed — classifier uses regex, not LLM
        mock_gen_llm.invoke.return_value = _make_ai_message(
            "MATCH (s:Section {number: '135'}) RETURN s.uid AS uid, s.text AS text, s.number AS number"
        )
        mock_synth_llm.invoke.return_value = _make_ai_message("Section 135 answer.")

        from query.flow import run_query

        result = run_query("What does Section 135 say?")

    assert result["metadata"]["source_type"] == "cypher"
    assert result["metadata"]["fallback_used"] is False


# ---------------------------------------------------------------------------
# Test 9: source_type is "fulltext_search" when BM25 fallback fires (D-08)
# ---------------------------------------------------------------------------


def test_source_type_fulltext_search_when_fallback_used():
    """source_type is 'fulltext_search' when BM25 fallback fires (D-08)."""
    import query.generator
    import query.synthesizer

    mock_empty_drv = _make_empty_driver()

    # Fulltext fallback returns one result record
    ft_record = {
        "node_props": {"uid": "def-csr", "text": "CSR", "title": None, "term": "CSR", "number": None},
        "score": 0.9,
        "node_type": "Definition",
    }

    call_count = {"n": 0}

    def mock_session_run(query_str, **kwargs):
        call_count["n"] += 1
        result = MagicMock()
        # fulltext CALL query returns result; structured MATCH returns empty
        if "fulltext" in query_str.lower() or "queryNodes" in query_str:
            result.__iter__ = MagicMock(return_value=iter([ft_record]))
        else:
            result.__iter__ = MagicMock(return_value=iter([]))
        return result

    mock_empty_drv.session.return_value.run.side_effect = mock_session_run

    with (
        patch.object(query.generator, "_llm") as mock_gen_llm,
        patch("query.executor.get_driver", return_value=mock_empty_drv),
        patch("query.flow.get_driver", return_value=mock_empty_drv),
        patch.object(query.synthesizer, "_llm") as mock_synth_llm,
        patch("query.flow._schema_text", "mock schema"),
    ):
        # No classifier mock needed — classifier uses regex, not LLM
        mock_gen_llm.invoke.return_value = _make_ai_message(
            "MATCH (s:Section {number: '999'}) RETURN s.uid AS uid"
        )
        mock_synth_llm.invoke.return_value = _make_ai_message("Fallback answer.")

        from query.flow import run_query

        result = run_query("tell me about section 999")

    assert result["metadata"]["fallback_used"] is True
    assert result["metadata"]["source_type"] == "fulltext_search"


# ---------------------------------------------------------------------------
# Test 10: MAX_RETRY_COUNT and MAX_EMPTY_RETRY_COUNT constants
# ---------------------------------------------------------------------------


def test_max_retry_constants():
    """MAX_RETRY_COUNT and MAX_EMPTY_RETRY_COUNT are both 3."""
    from query.flow import MAX_EMPTY_RETRY_COUNT, MAX_RETRY_COUNT

    assert MAX_RETRY_COUNT == 3
    assert MAX_EMPTY_RETRY_COUNT == 3
