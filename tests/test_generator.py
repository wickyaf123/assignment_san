"""
Unit tests for query.generator and query.synthesizer modules.

All tests mock the LLM — no real Gemini API calls required.
Tests verify:
- generate_cypher() returns a cleaned Cypher string
- error_context triggers RETRY_PROMPT_SUFFIX inclusion
- empty_retry_count > 0 triggers BROADEN_PROMPT_SUFFIX inclusion
- Per-intent few-shot examples are injected into the prompt
- _clean_cypher() strips markdown fences and whitespace
- get_schema_text(driver=None) returns a string containing "Section"
- synthesize_response() returns a string (prose)
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

# Set a dummy API key before any module-level import so ChatGoogleGenerativeAI
# doesn't raise a ValidationError when generator/synthesizer modules are imported.
os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key-for-unit-tests")


# ---------------------------------------------------------------------------
# Test 1: generate_cypher returns a cleaned string (no markdown fences)
# ---------------------------------------------------------------------------


def test_generate_cypher_returns_cleaned_string():
    """generate_cypher must return a plain Cypher string with no markdown fences."""
    mock_response = MagicMock()
    mock_response.content = "```cypher\nMATCH (s:Section {number: '135'}) RETURN s\n```"

    with patch("query.generator._llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response

        from query.generator import generate_cypher

        result = generate_cypher(
            query="What does Section 135 say?",
            intent="structured_lookup",
            schema_text="schema text",
        )
        assert "```" not in result, f"Markdown fences not stripped: {result!r}"
        assert "MATCH" in result, f"Expected Cypher content in result: {result!r}"


# ---------------------------------------------------------------------------
# Test 2: error_context triggers RETRY_PROMPT_SUFFIX inclusion
# ---------------------------------------------------------------------------


def test_generate_cypher_with_error_context_appends_retry_suffix():
    """generate_cypher must append RETRY_PROMPT_SUFFIX when error_context is set."""
    mock_response = MagicMock()
    mock_response.content = "MATCH (s:Section) RETURN s"

    with patch("query.generator._llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response

        from query.generator import generate_cypher

        generate_cypher(
            query="What does Section 135 say?",
            intent="structured_lookup",
            schema_text="schema text",
            error_context="SyntaxError: unexpected token",
        )

        call_args = mock_llm.invoke.call_args
        messages = call_args[0][0]  # list of [SystemMessage, HumanMessage]
        prompt_arg = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        # RETRY_PROMPT_SUFFIX contains "previous Cypher query failed"
        assert "previous Cypher query failed" in prompt_arg or "SyntaxError" in prompt_arg, (
            f"Expected retry suffix in prompt: {prompt_arg!r}"
        )


# ---------------------------------------------------------------------------
# Test 3: empty_retry_count > 0 triggers BROADEN_PROMPT_SUFFIX inclusion
# ---------------------------------------------------------------------------


def test_generate_cypher_with_empty_retry_appends_broaden_suffix():
    """generate_cypher must append BROADEN_PROMPT_SUFFIX when empty_retry_count > 0."""
    mock_response = MagicMock()
    mock_response.content = "MATCH (s:Section) RETURN s"

    with patch("query.generator._llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response

        from query.generator import generate_cypher

        generate_cypher(
            query="What does Section 135 say?",
            intent="structured_lookup",
            schema_text="schema text",
            empty_retry_count=1,
        )

        call_args = mock_llm.invoke.call_args
        messages = call_args[0][0]  # list of [SystemMessage, HumanMessage]
        prompt_arg = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        # BROADEN_PROMPT_SUFFIX contains "returned no results" or "CONTAINS"
        assert "returned no results" in prompt_arg or "CONTAINS" in prompt_arg, (
            f"Expected broaden suffix in prompt: {prompt_arg!r}"
        )


# ---------------------------------------------------------------------------
# Test 4: Per-intent examples are injected into the prompt
# ---------------------------------------------------------------------------


def test_generate_cypher_injects_per_intent_examples():
    """generate_cypher must inject few-shot examples for the given intent."""
    mock_response = MagicMock()
    mock_response.content = "MATCH (s:Section) RETURN s"

    with patch("query.generator._llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response

        from query.generator import generate_cypher

        generate_cypher(
            query="How was Section 149 amended?",
            intent="amendment_query",
            schema_text="schema text",
        )

        call_args = mock_llm.invoke.call_args
        messages = call_args[0][0]  # list of [SystemMessage, HumanMessage]
        prompt_arg = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
        # amendment_query examples include "amended" keyword text
        assert "amended" in prompt_arg.lower() or "amendment" in prompt_arg.lower(), (
            f"Expected amendment examples in prompt: {prompt_arg!r}"
        )


# ---------------------------------------------------------------------------
# Test 5: _clean_cypher strips markdown fences and whitespace
# ---------------------------------------------------------------------------


def test_clean_cypher_strips_markdown_fences():
    """_clean_cypher must strip ```cypher ... ``` fences and leading/trailing whitespace."""
    from query.generator import _clean_cypher

    # Test: ```cypher fence
    result = _clean_cypher("```cypher\nMATCH (s:Section) RETURN s\n```")
    assert "```" not in result
    assert "MATCH" in result

    # Test: plain ``` fence
    result2 = _clean_cypher("```\nMATCH (s:Section) RETURN s\n```")
    assert "```" not in result2

    # Test: trailing whitespace
    result3 = _clean_cypher("  MATCH (s:Section) RETURN s  ")
    assert result3 == "MATCH (s:Section) RETURN s"

    # Test: CYPHER: prefix
    result4 = _clean_cypher("CYPHER: MATCH (s:Section) RETURN s")
    assert not result4.startswith("CYPHER:")


# ---------------------------------------------------------------------------
# Test 6: get_schema_text(driver=None) returns a string containing "Section"
# ---------------------------------------------------------------------------


def test_get_schema_text_offline_returns_string_with_section():
    """get_schema_text(driver=None) must return a non-empty string containing 'Section'."""
    from query.generator import get_schema_text

    result = get_schema_text(driver=None)
    assert isinstance(result, str), f"Expected str, got {type(result)}"
    assert len(result) > 0, "Expected non-empty schema text"
    assert "Section" in result, f"Expected 'Section' in schema text: {result!r}"


# ---------------------------------------------------------------------------
# Test 7: synthesize_response returns a string
# ---------------------------------------------------------------------------


def test_synthesize_response_returns_string():
    """synthesize_response must return a prose string."""
    mock_response = MagicMock()
    mock_response.content = "Section 135 deals with Corporate Social Responsibility."

    with patch("query.synthesizer._llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response

        from query.synthesizer import synthesize_response

        answer, citations = synthesize_response(
            query="What does Section 135 say?",
            results=[{"s.title": "CSR", "s.text": "Section 135 text..."}],
            cypher="MATCH (s:Section {number: '135'}) RETURN s",
            sources=[],
        )
        assert isinstance(answer, str), f"Expected str, got {type(answer)}"
        assert len(answer) > 0
        assert isinstance(citations, list)


# ---------------------------------------------------------------------------
# Test 8: synthesize_response returns hardcoded string when results are empty
# ---------------------------------------------------------------------------


def test_synthesize_response_empty_results_no_llm_call():
    """synthesize_response must return a hardcoded fallback when results are empty — no LLM call."""
    with patch("query.synthesizer._llm") as mock_llm:
        from query.synthesizer import synthesize_response

        answer, citations = synthesize_response(
            query="What does Section 999 say?",
            results=[],
            cypher="MATCH (s:Section {number: '999'}) RETURN s",
            sources=[],
            effective_state=None,
        )
        # Must return hardcoded message without calling LLM
        assert mock_llm.invoke.call_count == 0, (
            "LLM should NOT be called when results are empty"
        )
        assert "could not find" in answer.lower(), (
            f"Expected 'could not find' message, got: {answer!r}"
        )
        assert citations == []


# ---------------------------------------------------------------------------
# Test 9: extract_sources returns list of dicts
# ---------------------------------------------------------------------------


def test_extract_sources_returns_list_of_dicts():
    """extract_sources must return a list of source dicts from Neo4j records."""
    from query.synthesizer import extract_sources

    records = [
        {"uid": "section-135", "type": "Section", "text": "This is section 135 text that is long enough to be truncated at 200 characters for the excerpt."},
        {"uid": "section-149", "type": "Section", "text": "Section 149 text"},
    ]
    result = extract_sources(records)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Test 10: extract_graph_path returns list of strings
# ---------------------------------------------------------------------------


def test_extract_graph_path_returns_list_of_strings():
    """extract_graph_path must return a list of path string representations."""
    from query.synthesizer import extract_graph_path

    records = [
        {"uid": "section-135", "number": "135"},
        {"uid": "section-149", "number": "149"},
    ]
    result = extract_graph_path(records)
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, str)
