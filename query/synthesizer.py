"""
Response synthesizer for ViddhiAI query engine.

Formats raw Neo4j result records into traceable prose using Gemini.
Also provides helper functions to extract source nodes and graph paths
for the traceability layer (TRACE-02, TRACE-03).

Design choices:
- D-04: Synthesis prompt forbids fact generation — LLM only formats graph data
- T-03-07: Returns hardcoded "could not find" string when results are empty,
  preventing LLM hallucination on empty data
- D-06: Module-level LLM singleton (instantiated once at import time)
- T-03-09: GEMINI_API_KEY read from env var, never logged
"""

from __future__ import annotations

import logging
import os
import re

from langchain_google_genai import ChatGoogleGenerativeAI

from query.prompts import FALLBACK_SYNTHESIS_PROMPT, SYNTHESIS_PROMPT, SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level LLM singleton (D-06, T-03-09)
# temperature=0.2 for slightly more natural prose while keeping it grounded
# ---------------------------------------------------------------------------

_llm = ChatGoogleGenerativeAI(
    model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    temperature=0.2,
    api_key=os.environ.get("GEMINI_API_KEY"),
)

# ---------------------------------------------------------------------------
# Fallback message when graph returns no results (T-03-07)
# Anti-hallucination: never call LLM on empty data
# ---------------------------------------------------------------------------

_NO_RESULTS_MESSAGE = (
    "I could not find information about this in the knowledge graph."
)


# ---------------------------------------------------------------------------
# Helper: format Neo4j records for LLM prompt
# ---------------------------------------------------------------------------


def _format_results(records: list[dict]) -> str:
    """Format Neo4j result records as readable text for the synthesis prompt.

    Args:
        records: List of Neo4j result dicts.

    Returns:
        Human-readable multi-line string of the graph results.
    """
    if not records:
        return "(no results)"

    parts = []
    for i, record in enumerate(records, 1):
        record_parts = [f"Record {i}:"]
        for key, value in record.items():
            if value is not None:
                # Truncate very long text fields to 500 chars
                val_str = str(value)
                if len(val_str) > 500:
                    val_str = val_str[:500] + "..."
                record_parts.append(f"  {key}: {val_str}")
        parts.append("\n".join(record_parts))

    return "\n\n".join(parts)


def _format_sources_for_prompt(sources: list[dict]) -> str:
    """Format source node list into a compact string for the synthesis prompt.

    Each source is rendered as ``uid (Type)`` so the LLM knows which UIDs are
    available for citation markers.

    Args:
        sources: List of source dicts with at least ``uid`` and ``type`` keys.

    Returns:
        Comma-separated string, e.g. ``sec-135 (Section), rule-8 (Rule)``.
        Returns ``(none)`` when sources is empty.
    """
    if not sources:
        return "(none)"
    return ", ".join(f"{s['uid']} ({s.get('type', 'unknown')})" for s in sources)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def synthesize_response(
    query: str,
    results: list[dict],
    cypher: str,
    sources: list[dict],
    effective_state: dict | None = None,
    fallback_used: bool = False,
) -> tuple[str, list[dict]]:
    """Format graph query results into traceable prose for the user.

    Anti-hallucination rule (T-03-07): if results are empty AND effective_state
    is None, returns a hardcoded fallback without calling the LLM.

    Args:
        query: Original natural language question from the user.
        results: Raw Neo4j result records.
        cypher: The Cypher query that produced the results (for transparency).
        sources: Extracted source node excerpts (for traceability).
        effective_state: Optional effective state resolution dict (amendment chain).
        fallback_used: Whether BM25 fulltext fallback was used instead of Cypher.

    Returns:
        Tuple of (prose answer string with inline citations, citations list).
        Citations list contains dicts: [{marker, uid, type, text_excerpt, index}].
    """
    # Anti-hallucination guard (T-03-07): never call LLM on empty data
    if not results and effective_state is None:
        logger.info("Empty results and no effective_state — returning fallback message")
        return _NO_RESULTS_MESSAGE, []

    # Format results text for the prompt
    results_text = _format_results(results)

    # Format sources for citation marker instructions
    sources_text = _format_sources_for_prompt(sources)

    # Append effective state information if available (amendment chain)
    if effective_state is not None:
        state_parts = ["", "Effective state resolution:"]
        if effective_state.get("current_text"):
            state_parts.append(f"  Current text: {effective_state['current_text'][:300]}")
        if effective_state.get("status"):
            state_parts.append(f"  Status: {effective_state['status']}")
        if effective_state.get("amendment_chain"):
            chain = effective_state["amendment_chain"]
            state_parts.append(f"  Amendment chain ({len(chain)} entries): {chain}")
        results_text += "\n".join(state_parts)

    # Use fallback prompt for BM25 results, standard prompt for Cypher results
    if fallback_used:
        prompt = FALLBACK_SYNTHESIS_PROMPT.format(
            query=query,
            results=results_text,
            sources=sources_text,
        )
    else:
        prompt = SYNTHESIS_PROMPT.format(
            query=query,
            results=results_text,
            cypher=cypher,
            sources=sources_text,
        )

    # Call LLM — extract prose from AIMessage
    response = _llm.invoke(prompt)
    answer = response.content

    # Extract citation markers from the answer and map to source nodes
    citations = _extract_citations(answer, sources)

    logger.info(
        "Synthesized response for query '%s...': %d chars, %d citations",
        query[:60],
        len(answer),
        len(citations),
    )
    return answer, citations


def _extract_citations(answer_text: str, sources: list[dict]) -> list[dict]:
    """Extract citation markers from answer text and map to source nodes.

    Finds all ``[uid]`` markers in the answer, maps each to its source entry.
    Returns a de-duplicated list of citation dicts ordered by first appearance.

    Args:
        answer_text: The LLM-generated prose answer containing ``[uid]`` markers.
        sources: List of source dicts with ``uid``, ``type``, and ``text_excerpt``.

    Returns:
        List of citation dicts: [{marker, uid, type, text_excerpt, index}].
    """
    # Build uid -> source lookup
    source_map = {s["uid"]: s for s in sources}

    # Find all [uid] patterns — uid format is like sec-135, rule-8, def-company, sec-135-ss-7
    marker_pattern = re.compile(r"\[([a-z][\w-]*)\]")
    found_markers = marker_pattern.findall(answer_text)

    citations: list[dict] = []
    seen: set[str] = set()
    index = 1
    for marker in found_markers:
        if marker in seen:
            continue
        seen.add(marker)
        source = source_map.get(marker)
        if source:
            citations.append(
                {
                    "marker": marker,
                    "uid": source["uid"],
                    "type": source.get("type", "unknown"),
                    "text_excerpt": source.get("text_excerpt", ""),
                    "index": index,
                }
            )
            index += 1

    return citations


async def synthesize_response_stream(
    query: str,
    results: list[dict],
    cypher: str,
    sources: list[dict],
    effective_state: dict | None = None,
    fallback_used: bool = False,
):
    """Async generator that yields answer token chunks from Gemini streaming.

    Same logic as synthesize_response() but uses astream() for token-by-token
    output.  Yields string chunks as they arrive from Gemini.

    Anti-hallucination rule (T-03-07): if results are empty AND effective_state
    is None, yields a hardcoded fallback without calling the LLM.

    Args:
        query: Original natural language question from the user.
        results: Raw Neo4j result records.
        cypher: The Cypher query that produced the results (for transparency).
        sources: Extracted source node excerpts (for traceability).
        effective_state: Optional effective state resolution dict (amendment chain).
        fallback_used: Whether BM25 fulltext fallback was used instead of Cypher.

    Yields:
        String chunks of the synthesised prose answer.
    """
    # Anti-hallucination guard (T-03-07): never call LLM on empty data
    if not results and effective_state is None:
        logger.info("Stream: empty results and no effective_state — returning fallback message")
        yield _NO_RESULTS_MESSAGE
        return

    # Format results text for the prompt
    results_text = _format_results(results)

    # Format sources for citation marker instructions
    sources_text = _format_sources_for_prompt(sources)

    # Append effective state information if available (amendment chain)
    if effective_state is not None:
        state_parts = ["", "Effective state resolution:"]
        if effective_state.get("current_text"):
            state_parts.append(f"  Current text: {effective_state['current_text'][:300]}")
        if effective_state.get("status"):
            state_parts.append(f"  Status: {effective_state['status']}")
        if effective_state.get("amendment_chain"):
            chain = effective_state["amendment_chain"]
            state_parts.append(f"  Amendment chain ({len(chain)} entries): {chain}")
        results_text += "\n".join(state_parts)

    # Use fallback prompt for BM25 results, standard prompt for Cypher results
    if fallback_used:
        prompt = FALLBACK_SYNTHESIS_PROMPT.format(
            query=query,
            results=results_text,
            sources=sources_text,
        )
    else:
        prompt = SYNTHESIS_PROMPT.format(
            query=query,
            results=results_text,
            cypher=cypher,
            sources=sources_text,
        )

    # Stream tokens from Gemini via LangChain astream()
    async for chunk in _llm.astream(prompt):
        if chunk.content:
            yield chunk.content


def _infer_type_from_uid(uid: str) -> str | None:
    """Infer the Neo4j node type from a uid string pattern.

    UID patterns (from ingestion/uid_generator.py):
      sec-{n}           -> Section
      sec-{n}-ss-{n}    -> SubSection
      sec-*-cl-*        -> Clause
      sec-*-sc-*        -> SubClause
      sec-*-proviso-*   -> Proviso
      sec-*-expl-*      -> Explanation
      def-{slug}        -> Definition
      rule-{n}          -> Rule
      ruleset-{slug}    -> RuleSet
      ch-{roman}        -> Chapter
      amend-{slug}      -> AmendmentAct
      act-{slug}        -> Act
      form-{slug}       -> Form
      sch-{n}           -> Schedule
    """
    if uid.startswith("sec-"):
        if "-ss-" in uid:
            return "SubSection"
        if "-cl-" in uid:
            if "-sc-" in uid:
                return "SubClause"
            return "Clause"
        if "-proviso-" in uid:
            return "Proviso"
        if "-expl-" in uid:
            return "Explanation"
        return "Section"
    if uid.startswith("def-"):
        return "Definition"
    if uid.startswith("rule-"):
        return "Rule"
    if uid.startswith("ruleset-"):
        return "RuleSet"
    if uid.startswith("ch-"):
        return "Chapter"
    if uid.startswith("amend-"):
        return "AmendmentAct"
    if uid.startswith("act-"):
        return "Act"
    if uid.startswith("form-"):
        return "Form"
    if uid.startswith("sch-"):
        return "Schedule"
    return None


def extract_sources(records: list[dict]) -> list[dict]:
    """Extract source node excerpts from Neo4j result records (TRACE-03).

    Handles multiple uid columns per record (e.g. s.uid + a.uid).
    Falls back to constructing UIDs from node identifiers when no uid
    column is present.  Infers node type from the uid pattern when no
    explicit type/label column exists.

    Args:
        records: List of Neo4j result dicts.

    Returns:
        List of source dicts: [{uid, type, text_excerpt}].
    """
    sources: list[dict] = []
    seen_uids: set[str] = set()

    for record in records:
        # ---- Collect ALL uid-bearing keys from this record ----
        uid_entries: list[tuple[str, str]] = []
        for key, value in record.items():
            if value and (key == "uid" or key.endswith(".uid")):
                uid_str = str(value)
                if uid_str not in seen_uids:
                    uid_entries.append((key, uid_str))

        # ---- Fallback: construct uid from identifiers ----
        if not uid_entries:
            number = (
                record.get("number") or record.get("s.number")
                or record.get("r.number") or record.get("ss.number")
            )
            term = record.get("term") or record.get("d.term")

            constructed_uid: str | None = None
            if number is not None:
                num_str = str(number)
                try:
                    int(num_str)
                    constructed_uid = f"sec-{num_str}"
                except (ValueError, TypeError):
                    # Non-integer numbers are rules (Rule.number is STRING)
                    constructed_uid = f"rule-{num_str}"
            elif term:
                slug = re.sub(r"[^a-z0-9-]", "", str(term).lower().replace(" ", "-")).strip("-")
                constructed_uid = f"def-{slug}"

            if constructed_uid and constructed_uid not in seen_uids:
                uid_entries.append(("uid", constructed_uid))

        # ---- Process each uid found in this record ----
        for uid_key, uid_str in uid_entries:
            if uid_str in seen_uids:
                continue
            seen_uids.add(uid_str)

            # Infer node type from explicit columns first
            node_type = (
                record.get("type") or record.get("labels")
                or record.get("label") or record.get("source_type")
            )

            # Fallback: infer type from uid pattern
            if not node_type:
                node_type = _infer_type_from_uid(uid_str)

            # Extract text: try prefix-specific keys first
            prefix = uid_key.split(".")[0] if "." in uid_key else ""
            text = ""
            if prefix:
                text = (
                    record.get(f"{prefix}.text")
                    or record.get(f"{prefix}.title")
                    or record.get(f"{prefix}.term")
                    or ""
                )
            if not text:
                text = (
                    record.get("text") or record.get("s.text")
                    or record.get("r.text") or record.get("d.text")
                    or record.get("d.term") or record.get("ss.text")
                    or record.get("s.title") or record.get("r.title")
                    or record.get("c.title") or record.get("sc.text")
                    or ""
                )

            text_excerpt = str(text)[:200] if text else ""
            sources.append({
                "uid": uid_str,
                "type": str(node_type) if node_type else "unknown",
                "text_excerpt": text_excerpt,
            })

    return sources


def extract_graph_path(records: list[dict]) -> list[str]:
    """Build a human-readable graph path representation from result records (TRACE-02).

    For each record, constructs a path string in the format "(Label:identifier)".
    The identifier uses uid, number, or title — whichever is available.

    Args:
        records: List of Neo4j result dicts.

    Returns:
        List of path strings, e.g. ["(Section:135)", "(AmendmentAct:2026)"].
    """
    path_items = []
    seen: set[str] = set()

    for record in records:
        # Try to determine the node label/type
        label = (
            record.get("type")
            or record.get("label")
            or record.get("labels")
            or "Node"
        )

        # Try to determine the identifier
        identifier = (
            record.get("number")
            or record.get("s.number")
            or record.get("uid")
            or record.get("s.uid")
            or record.get("title")
            or record.get("s.title")
            or "?"
        )

        path_str = f"({label}:{identifier})"
        if path_str not in seen:
            seen.add(path_str)
            path_items.append(path_str)

    return path_items
