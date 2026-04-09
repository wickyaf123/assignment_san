"""
Unit tests for query.synthesizer module (QUERY-06).

All tests mock the LLM — no real Gemini API calls required.

Tests verify:
- synthesize_response() returns non-empty prose for valid graph results
- Anti-hallucination guard returns "no results" message when results empty
  and effective_state is None — without calling the LLM
- extract_sources() returns list of dicts with uid/type/text_excerpt keys
- extract_graph_path() returns list of strings in "(Label:identifier)" format
- synthesize_response() incorporates effective_state enrichment (amendment queries)
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

# Set a dummy API key before any module-level import so ChatGoogleGenerativeAI
# doesn't raise a ValidationError when the synthesizer module is first imported.
os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key-for-unit-tests")
os.environ.setdefault("GOOGLE_API_KEY", "test-dummy-key-for-unit-tests")


# ---------------------------------------------------------------------------
# Test 1: synthesize_response returns non-empty prose for valid graph results
# ---------------------------------------------------------------------------


def test_synthesize_response_returns_non_empty_answer_for_valid_results():
    """synthesize_response must call the LLM and return a non-empty prose answer
    when results are non-empty."""
    mock_response = MagicMock()
    mock_response.content = "Section 135 mandates Corporate Social Responsibility obligations."

    with patch("query.synthesizer._llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response

        from query.synthesizer import synthesize_response

        answer, citations = synthesize_response(
            query="What does Section 135 say?",
            results=[{"uid": "section-135", "text": "CSR provisions text", "number": "135"}],
            cypher="MATCH (s:Section {number: '135'}) RETURN s.uid AS uid, s.text AS text",
            sources=[{"uid": "section-135", "type": "Section", "text_excerpt": "CSR"}],
        )

    assert isinstance(answer, str), f"Expected str, got {type(answer)}"
    assert len(answer) > 0, "Expected non-empty prose answer"
    assert isinstance(citations, list), f"Expected list, got {type(citations)}"
    assert mock_llm.invoke.call_count == 1, "Expected exactly one LLM call for valid results"


# ---------------------------------------------------------------------------
# Test 2: Anti-hallucination guard: no LLM call and "no results" when empty
# ---------------------------------------------------------------------------


def test_synthesize_response_anti_hallucination_guard_no_llm_call():
    """When results is empty AND effective_state is None, synthesize_response must
    return the hardcoded 'could not find' message without calling the LLM at all."""
    with patch("query.synthesizer._llm") as mock_llm:
        from query.synthesizer import synthesize_response

        answer, citations = synthesize_response(
            query="What does Section 999 say?",
            results=[],
            cypher="MATCH (s:Section {number: '999'}) RETURN s",
            sources=[],
            effective_state=None,
        )

    assert mock_llm.invoke.call_count == 0, (
        "LLM must NOT be called when results are empty and effective_state is None"
    )
    assert isinstance(answer, str), f"Expected str, got {type(answer)}"
    assert "could not find" in answer.lower(), (
        f"Expected 'could not find' in fallback message, got: {answer!r}"
    )
    assert citations == [], "Expected empty citations for no-results fallback"


# ---------------------------------------------------------------------------
# Test 3: extract_sources returns list of dicts with uid, type, text_excerpt keys
# ---------------------------------------------------------------------------


def test_extract_sources_returns_dicts_with_required_keys():
    """extract_sources must return a list of dicts each containing uid, type,
    and text_excerpt keys (TRACE-03)."""
    from query.synthesizer import extract_sources

    records = [
        {
            "uid": "section-135",
            "type": "Section",
            "text": "Corporate Social Responsibility text for section one three five.",
        },
        {
            "uid": "section-149",
            "type": "Section",
            "text": "Independent directors text for section one four nine.",
        },
    ]

    result = extract_sources(records)

    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) >= 1, "Expected at least one source extracted"

    for source in result:
        assert isinstance(source, dict), f"Each source must be a dict, got {type(source)}"
        assert "uid" in source, f"Source missing 'uid' key: {source!r}"
        assert "type" in source, f"Source missing 'type' key: {source!r}"
        assert "text_excerpt" in source, f"Source missing 'text_excerpt' key: {source!r}"
        # All values must be strings
        assert isinstance(source["uid"], str), f"uid must be str: {source['uid']!r}"
        assert isinstance(source["type"], str), f"type must be str: {source['type']!r}"
        assert isinstance(source["text_excerpt"], str), (
            f"text_excerpt must be str: {source['text_excerpt']!r}"
        )


def test_extract_sources_deduplicates_by_uid():
    """extract_sources must not return duplicate sources for the same uid."""
    from query.synthesizer import extract_sources

    records = [
        {"uid": "section-135", "type": "Section", "text": "First occurrence."},
        {"uid": "section-135", "type": "Section", "text": "Duplicate occurrence."},
        {"uid": "section-149", "type": "Section", "text": "Different section."},
    ]

    result = extract_sources(records)
    uids = [s["uid"] for s in result]
    assert len(uids) == len(set(uids)), f"Duplicate uids found in sources: {uids}"


def test_extract_sources_returns_empty_list_for_records_without_uid():
    """extract_sources must return empty list when no records have a uid field."""
    from query.synthesizer import extract_sources

    records = [
        {"score": 0.9, "node_type": "Section"},
        {"count": 42},
    ]

    result = extract_sources(records)
    assert isinstance(result, list)
    assert len(result) == 0, f"Expected empty list for uid-less records, got: {result!r}"


# ---------------------------------------------------------------------------
# Test 4: extract_graph_path returns list of strings in "(Label:identifier)" format
# ---------------------------------------------------------------------------


def test_extract_graph_path_returns_list_of_strings():
    """extract_graph_path must return a list of strings in '(Label:identifier)' format
    (TRACE-02)."""
    from query.synthesizer import extract_graph_path

    records = [
        {"uid": "section-135", "number": "135", "type": "Section"},
        {"uid": "amendment-2026", "number": "2026", "type": "AmendmentAct"},
    ]

    result = extract_graph_path(records)

    assert isinstance(result, list), f"Expected list, got {type(result)}"
    for item in result:
        assert isinstance(item, str), f"Each path item must be a str, got {type(item)}"
        # Path strings must follow "(Label:identifier)" format
        assert item.startswith("("), f"Path string must start with '(': {item!r}"
        assert item.endswith(")"), f"Path string must end with ')': {item!r}"
        assert ":" in item, f"Path string must contain ':' separator: {item!r}"


def test_extract_graph_path_deduplicates_identical_paths():
    """extract_graph_path must not produce duplicate path entries."""
    from query.synthesizer import extract_graph_path

    records = [
        {"uid": "section-135", "number": "135", "type": "Section"},
        {"uid": "section-135", "number": "135", "type": "Section"},
    ]

    result = extract_graph_path(records)
    assert len(result) == len(set(result)), f"Duplicate path entries found: {result}"


# ---------------------------------------------------------------------------
# Test 5: synthesize_response incorporates effective_state for amendment queries
# ---------------------------------------------------------------------------


def test_synthesize_response_with_effective_state_calls_llm():
    """When effective_state is provided (amendment query), synthesize_response must
    call the LLM — the guard must NOT fire for non-None effective_state even when
    results is empty."""
    mock_response = MagicMock()
    mock_response.content = "Section 135 was amended by the Corporate Laws Amendment Act 2026."

    with patch("query.synthesizer._llm") as mock_llm:
        mock_llm.invoke.return_value = mock_response

        from query.synthesizer import synthesize_response

        effective_state = {
            "current_text": "Amended text of section 135.",
            "status": "amended",
            "amendment_chain": [{"act": "Corporate Laws Amendment Act 2026"}],
            "original_text": "Original text of section 135.",
        }

        answer, citations = synthesize_response(
            query="How was Section 135 amended?",
            results=[],  # Empty results — but effective_state is provided
            cypher="MATCH (s:Section {number: '135'}) RETURN s",
            sources=[],
            effective_state=effective_state,
        )

    # LLM must have been called (effective_state bypasses the empty-result guard)
    assert mock_llm.invoke.call_count == 1, (
        "LLM must be called when effective_state is provided, even if results are empty"
    )
    assert isinstance(answer, str), f"Expected str, got {type(answer)}"
    assert len(answer) > 0, "Expected non-empty answer from LLM"


def test_synthesize_response_effective_state_included_in_prompt():
    """When effective_state has current_text and status, synthesize_response must
    include that information in the prompt sent to the LLM."""
    captured_prompt = {}

    def capture_invoke(prompt):
        captured_prompt["value"] = prompt
        mock_response = MagicMock()
        mock_response.content = "Amendment answer."
        return mock_response

    with patch("query.synthesizer._llm") as mock_llm:
        mock_llm.invoke.side_effect = capture_invoke

        from query.synthesizer import synthesize_response

        effective_state = {
            "current_text": "Amended CSR text with new threshold.",
            "status": "amended",
            "amendment_chain": [],
            "original_text": "Original CSR text.",
        }

        synthesize_response(
            query="How was Section 135 amended?",
            results=[{"uid": "section-135", "text": "CSR text", "number": "135"}],
            cypher="MATCH (s:Section {number: '135'}) RETURN s",
            sources=[],
            effective_state=effective_state,
        )

    prompt_used = captured_prompt.get("value", "")
    assert isinstance(prompt_used, str), "Expected prompt to be a string"
    # The effective state's current_text or status must appear somewhere in the prompt
    assert "amended" in prompt_used.lower() or "Amended CSR text" in prompt_used, (
        f"Expected effective state content in prompt, but got: {prompt_used!r}"
    )


# ---------------------------------------------------------------------------
# Test 6: extract_sources with dotted uid keys (s.uid, r.uid, d.uid)
# ---------------------------------------------------------------------------


def test_extract_sources_with_dotted_uid_keys():
    """extract_sources must recognize dotted uid keys like s.uid, r.uid, d.uid."""
    from query.synthesizer import extract_sources

    records = [
        {"s.uid": "sec-135", "s.text": "CSR provisions", "s.title": "CSR"},
        {"r.uid": "rule-8", "r.text": "Rule 8 text"},
        {"d.uid": "def-company", "d.term": "company", "d.text": "A company means..."},
    ]
    result = extract_sources(records)
    uids = [s["uid"] for s in result]
    assert "sec-135" in uids, f"Expected sec-135 in uids: {uids}"
    assert "rule-8" in uids, f"Expected rule-8 in uids: {uids}"
    assert "def-company" in uids, f"Expected def-company in uids: {uids}"


# ---------------------------------------------------------------------------
# Test 7: extract_sources with multiple uids per record
# ---------------------------------------------------------------------------


def test_extract_sources_multiple_uids_per_record():
    """extract_sources must extract BOTH s.uid and a.uid from a single record."""
    from query.synthesizer import extract_sources

    records = [
        {
            "s.uid": "sec-135",
            "a.uid": "amend-corporate-laws-amendment-2026",
            "s.title": "CSR",
            "a.title": "Corporate Laws Amendment Act 2026",
        },
    ]
    result = extract_sources(records)
    uids = [s["uid"] for s in result]
    assert len(uids) >= 2, f"Expected at least 2 sources from multi-uid record, got {len(uids)}: {uids}"
    assert "sec-135" in uids
    assert "amend-corporate-laws-amendment-2026" in uids


# ---------------------------------------------------------------------------
# Test 8: extract_sources fallback uid from number
# ---------------------------------------------------------------------------


def test_extract_sources_fallback_uid_from_number():
    """extract_sources must construct uid from s.number when no uid column exists."""
    from query.synthesizer import extract_sources

    records = [
        {"s.number": 135, "s.text": "CSR provisions", "s.title": "CSR"},
    ]
    result = extract_sources(records)
    assert len(result) >= 1, f"Expected at least 1 source from fallback, got {len(result)}"
    assert result[0]["uid"] == "sec-135", f"Expected sec-135, got {result[0]['uid']}"


# ---------------------------------------------------------------------------
# Test 9: extract_sources type inference from uid pattern
# ---------------------------------------------------------------------------


def test_extract_sources_type_inference_from_uid():
    """extract_sources must infer node type from uid pattern when no type column exists."""
    from query.synthesizer import extract_sources

    records = [
        {"s.uid": "sec-135", "s.text": "text"},
        {"ss.uid": "sec-135-ss-1", "ss.text": "subsection text"},
        {"d.uid": "def-company", "d.term": "company"},
    ]
    result = extract_sources(records)
    type_map = {s["uid"]: s["type"] for s in result}
    assert type_map.get("sec-135") == "Section", f"Expected Section, got {type_map.get('sec-135')}"
    assert type_map.get("sec-135-ss-1") == "SubSection", f"Expected SubSection, got {type_map.get('sec-135-ss-1')}"
    assert type_map.get("def-company") == "Definition", f"Expected Definition, got {type_map.get('def-company')}"
