"""
LangGraph StateGraph orchestrating the full ViddhiAI query pipeline.

Pipeline: classify -> generate -> validate -> execute -> (retry | fallback | resolve) -> synthesize

This module wires all individual query engine modules into a single coherent
flow. Phase 4 wraps run_query() with FastAPI routes.

Design:
- D-09: Dedicated resolve node for amendment chain traversal
- D-10: resolve node fires ONLY when intent == "amendment_query"
- D-11: resolve_node reuses graph.resolver.resolve_effective_state
- TRACE-04, TRACE-05: run_id UUID pre-generated before graph.invoke();
  trace URL constructed via LangSmith Client after invoke() completes.
- OBS-02, OBS-03: LangSmith traces every node transition and LLM call via
  the run_id/run_name passed in config.

Security:
- T-03-12: MAX_RETRY_COUNT = 2, MAX_EMPTY_RETRY_COUNT = 2 cap total LLM calls.
"""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from dataclasses import asdict

from langgraph.graph import END, START, StateGraph

from graph.connection import get_driver
from graph.resolver import resolve_effective_state
from query.classifier import classify_intent
from query.executor import execute_cypher, execute_fulltext_fallback
from query.preprocessor import preprocess_query
from query.generator import generate_cypher, get_schema_text
from query.models import QueryState
from query.synthesizer import extract_graph_path, extract_sources, synthesize_response
from query.validator import validate_cypher

# ---------------------------------------------------------------------------
# Module-level logger and constants
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

MAX_RETRY_COUNT = 3
MAX_EMPTY_RETRY_COUNT = 3

# ---------------------------------------------------------------------------
# Schema text — loaded once at module level (cached across requests)
# ---------------------------------------------------------------------------

_schema_text: str | None = None


def _get_schema_text() -> str:
    """Return the Neo4j schema text, loading it once on first call.

    Falls back to offline mode schema if the Neo4j driver is unavailable
    (e.g., during tests or startup before Neo4j is reachable).
    """
    global _schema_text
    if _schema_text is None:
        try:
            driver = get_driver()
            _schema_text = get_schema_text(driver)
        except Exception:
            _schema_text = get_schema_text(None)
    return _schema_text


# ---------------------------------------------------------------------------
# Node functions — each takes the full QueryState and returns a partial update dict
# ---------------------------------------------------------------------------


def classify_node(state: QueryState) -> dict:
    """Preprocess query and classify the user's intent using keyword matching."""
    preprocessed = preprocess_query(state["query"])
    intent = classify_intent(preprocessed)
    logger.info("Classified intent: %s", intent)
    return {"query": preprocessed, "intent": intent}


def generate_node(state: QueryState) -> dict:
    """Generate a Cypher query from the user's question and intent."""
    schema_text = _get_schema_text()
    cypher = generate_cypher(
        state["query"],
        state["intent"],
        schema_text,
        error_context=state.get("error_message"),
        empty_retry_count=state.get("empty_retry_count", 0),
    )
    new_retry_count = state.get("retry_count", 0) + 1
    logger.info("Generated Cypher (attempt %d): %s", new_retry_count, cypher[:80])
    return {"cypher": cypher, "retry_count": new_retry_count}


def validate_node(state: QueryState) -> dict:
    """Validate the generated Cypher for write-safety."""
    is_valid, reason = validate_cypher(state["cypher"])
    logger.info("Cypher validation: valid=%s, reason=%s", is_valid, reason)
    return {"cypher_valid": is_valid, "error_message": reason}


def execute_node(state: QueryState) -> dict:
    """Execute the validated Cypher against Neo4j."""
    result = execute_cypher(state["cypher"])
    records = result["results"]
    error = result["error"]

    if error:
        logger.warning("Cypher execution error: %s", error)
        return {
            "results": [],
            "error": error,
            "error_message": error,
            "sources": [],
            "graph_path": [],
        }

    if not records:
        new_empty_count = state.get("empty_retry_count", 0) + 1
        logger.info("Cypher returned empty results (empty_retry_count -> %d)", new_empty_count)
        return {
            "results": [],
            "error": None,
            "empty_retry_count": new_empty_count,
            "sources": [],
            "graph_path": [],
        }

    sources = extract_sources(records)
    graph_path = extract_graph_path(records)
    logger.info("Cypher returned %d records, %d sources", len(records), len(sources))
    return {
        "results": records,
        "error": None,
        "sources": sources,
        "graph_path": graph_path,
    }


def resolve_node(state: QueryState) -> dict:
    """Resolve the effective state of amended sections (D-09, D-10, D-11).

    Only fires for amendment_query intent. Extracts the section number from
    results or the Cypher query, then calls resolve_effective_state().
    """
    effective_state_dict: dict | None = None

    try:
        # Extract section number from results or Cypher
        section_number = _extract_section_number(state)
        if section_number is not None:
            driver = get_driver()
            es = resolve_effective_state(driver, section_number)
            if es is not None:
                effective_state_dict = asdict(es)
                logger.info("Resolved effective state for section %d: status=%s", section_number, es.status)
            else:
                logger.info("resolve_effective_state returned None for section %d", section_number)
        else:
            logger.info("Could not extract section number from results — skipping resolve")
    except Exception as exc:
        logger.warning("resolve_node failed: %s — continuing without effective state", exc)

    return {"effective_state": effective_state_dict}


def fallback_node(state: QueryState) -> dict:
    """Execute BM25 fulltext fallback when Cypher returns no results after retries (QUERY-05)."""
    ft_results = execute_fulltext_fallback(state["query"])
    sources = _extract_fulltext_sources(ft_results)
    logger.info("BM25 fallback returned %d results", len(ft_results))
    return {
        "results": ft_results,
        "fallback_used": True,
        "sources": sources,
        "graph_path": [],
    }


def synthesize_node(state: QueryState) -> dict:
    """Synthesize a prose answer from the query results (QUERY-06)."""
    answer, citations = synthesize_response(
        state["query"],
        state.get("results", []),
        state.get("cypher", ""),
        state.get("sources", []),
        state.get("effective_state"),
        fallback_used=state.get("fallback_used", False),
        intent=state.get("intent", ""),
    )
    logger.info("Synthesized answer (%d chars, %d citations)", len(answer), len(citations))
    return {"answer": answer, "citations": citations}


# ---------------------------------------------------------------------------
# Router functions — determine conditional edge targets
# ---------------------------------------------------------------------------


def route_after_validate(state: QueryState) -> str:
    """Route after validate_node based on Cypher validity and retry count."""
    if state.get("cypher_valid"):
        return "execute"
    if state.get("retry_count", 0) < MAX_RETRY_COUNT:
        return "regenerate"
    # Exhausted retries — hard reject, still synthesize an error message
    logger.warning("Cypher validation failed after %d retries — rejecting", MAX_RETRY_COUNT)
    return "reject"


def route_after_execute(state: QueryState) -> str:
    """Route after execute_node based on results, errors, and intent."""
    # Execution error with retries remaining -> retry generation
    if state.get("error") and state.get("retry_count", 0) < MAX_RETRY_COUNT:
        logger.info("Retrying after execution error (retry_count=%d)", state.get("retry_count", 0))
        return "retry_generate"

    # Empty results with retries remaining -> broaden query
    if not state.get("results") and state.get("empty_retry_count", 0) < MAX_EMPTY_RETRY_COUNT:
        logger.info("Retrying after empty results (empty_retry_count=%d)", state.get("empty_retry_count", 0))
        return "retry_generate"

    # Empty results + exhausted retries -> BM25 fallback
    if not state.get("results"):
        logger.info("Empty results after max retries — using BM25 fallback")
        return "fallback"

    # Results obtained — check if we need amendment resolution (D-10)
    if state.get("intent") == "amendment_query":
        return "resolve"

    return "synthesize"


# ---------------------------------------------------------------------------
# StateGraph construction
# ---------------------------------------------------------------------------

builder = StateGraph(QueryState)

builder.add_node("classify", classify_node)
builder.add_node("generate", generate_node)
builder.add_node("validate", validate_node)
builder.add_node("execute", execute_node)
builder.add_node("resolve", resolve_node)
builder.add_node("fallback", fallback_node)
builder.add_node("synthesize", synthesize_node)

builder.add_edge(START, "classify")
builder.add_edge("classify", "generate")
builder.add_edge("generate", "validate")

builder.add_conditional_edges(
    "validate",
    route_after_validate,
    {
        "execute": "execute",
        "regenerate": "generate",
        "reject": "synthesize",  # rejected queries still get a "cannot process" answer
    },
)

builder.add_conditional_edges(
    "execute",
    route_after_execute,
    {
        "resolve": "resolve",
        "synthesize": "synthesize",
        "retry_generate": "generate",
        "fallback": "fallback",
    },
)

builder.add_edge("resolve", "synthesize")
builder.add_edge("fallback", "synthesize")
builder.add_edge("synthesize", END)

graph = builder.compile()


# ---------------------------------------------------------------------------
# Response cache — legal data is stable; invalidate on ingestion only
# ---------------------------------------------------------------------------

_query_cache: dict[str, dict] = {}
_CACHE_MAX_SIZE = 128


def clear_query_cache() -> None:
    """Clear the query response cache (call after ingestion)."""
    _query_cache.clear()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_query(question: str) -> dict:
    """Execute the full NL-to-answer query pipeline.

    This is the main entry point for the query engine.
    Phase 4 wraps this with FastAPI routes.

    Args:
        question: Natural language legal query.

    Returns:
        Dict with keys: answer, cypher_query, graph_path, sources,
        trace_url, metadata.
    """
    # Check cache first
    cache_key = question.strip().lower()
    if cache_key in _query_cache:
        logger.info("Cache hit for query: %s", question[:80])
        return dict(_query_cache[cache_key])

    start_time = time.monotonic()
    run_id = uuid.uuid4()

    config = {"run_id": run_id, "run_name": "viddhiai_query"}

    initial_state: QueryState = {
        "query": question,
        "intent": "",
        "cypher": "",
        "cypher_valid": False,
        "retry_count": 0,
        "error": None,
        "error_message": None,
        "empty_retry_count": 0,
        "results": [],
        "fallback_used": False,
        "effective_state": None,
        "answer": "",
        "citations": [],
        "sources": [],
        "graph_path": [],
        "trace_url": "",
        "latency_ms": 0.0,
    }

    result = graph.invoke(initial_state, config=config)

    latency_ms = (time.monotonic() - start_time) * 1000

    # Construct LangSmith trace URL locally (no API call — saves ~0.5-1s)
    trace_url = f"https://smith.langchain.com/public/{run_id}/r"

    source_type = "fulltext_search" if result.get("fallback_used") else "cypher"

    response = {
        "answer": result.get("answer", ""),
        "cypher_query": result.get("cypher", ""),
        "graph_path": result.get("graph_path", []),
        "sources": result.get("sources", []),
        "citations": result.get("citations", []),
        "trace_url": trace_url,
        "metadata": {
            "intent": result.get("intent", ""),
            "retry_count": result.get("retry_count", 0),
            "latency_ms": round(latency_ms, 2),
            "source_type": source_type,
            "fallback_used": result.get("fallback_used", False),
        },
    }

    # Store in cache (LRU eviction when full)
    if len(_query_cache) >= _CACHE_MAX_SIZE:
        _query_cache.pop(next(iter(_query_cache)))
    _query_cache[cache_key] = response

    return response


# ---------------------------------------------------------------------------
# Streaming API — runs pipeline step-by-step, yields SSE event dicts
# ---------------------------------------------------------------------------


async def run_query_stream(question: str):
    """Async generator that yields SSE event dicts as the pipeline progresses.

    Runs the same classify -> generate -> validate -> execute -> resolve ->
    synthesize pipeline as run_query(), but yields structured events between
    each step so the caller can forward them as Server-Sent Events.

    Each yielded dict has: {"event": str, "data": dict}

    Event types:
        status  -- Pipeline progress updates (step name + optional detail)
        cypher  -- Generated Cypher query text
        sources -- Source nodes and graph path (sent before synthesis)
        token   -- Single answer text chunk (streamed from Gemini)
        done    -- Final response with full answer, citations, and metadata
        error   -- Error message on failure

    Args:
        question: Natural language legal query.

    Yields:
        Dicts of the form {"event": <event_type>, "data": <payload_dict>}.
    """
    import asyncio

    start_time = time.monotonic()
    run_id = uuid.uuid4()

    try:
        # ----- Step 1: Classify -----
        state: QueryState = {
            "query": question,
            "intent": "",
            "cypher": "",
            "cypher_valid": False,
            "retry_count": 0,
            "error": None,
            "error_message": None,
            "empty_retry_count": 0,
            "results": [],
            "fallback_used": False,
            "effective_state": None,
            "answer": "",
            "citations": [],
            "sources": [],
            "graph_path": [],
            "trace_url": "",
            "latency_ms": 0.0,
        }

        classify_result = await asyncio.to_thread(classify_node, state)
        state.update(classify_result)
        yield {"event": "status", "data": {"step": "classified", "intent": state["intent"]}}

        # ----- Step 2: Generate + Validate + Execute (with retries) -----
        for attempt in range(MAX_RETRY_COUNT):
            # Generate Cypher
            gen_result = await asyncio.to_thread(generate_node, state)
            state.update(gen_result)
            yield {"event": "cypher", "data": {"query": state["cypher"]}}

            # Validate Cypher
            val_result = await asyncio.to_thread(validate_node, state)
            state.update(val_result)

            if not state.get("cypher_valid"):
                yield {
                    "event": "status",
                    "data": {
                        "step": "validation_failed",
                        "message": state.get("error_message", ""),
                    },
                }
                if attempt < MAX_RETRY_COUNT - 1:
                    continue
                else:
                    break

            # Execute Cypher
            yield {"event": "status", "data": {"step": "executing"}}
            exec_result = await asyncio.to_thread(execute_node, state)
            state.update(exec_result)

            if state.get("error"):
                yield {
                    "event": "status",
                    "data": {
                        "step": "execution_error",
                        "message": str(state["error"]),
                    },
                }
                if attempt < MAX_RETRY_COUNT - 1:
                    continue
                else:
                    break

            if state.get("results"):
                break

            # Empty results -- retry with broadening if retries remain
            if state.get("empty_retry_count", 0) < MAX_EMPTY_RETRY_COUNT:
                yield {
                    "event": "status",
                    "data": {"step": "empty_results", "message": "Broadening query..."},
                }
                continue
            else:
                break

        # ----- Step 3: Fallback if needed -----
        if not state.get("results"):
            yield {
                "event": "status",
                "data": {"step": "fallback", "message": "Using fulltext search..."},
            }
            fb_result = await asyncio.to_thread(fallback_node, state)
            state.update(fb_result)

        # ----- Step 4: Resolve if amendment query -----
        if state.get("intent") == "amendment_query" and state.get("results"):
            yield {"event": "status", "data": {"step": "resolving"}}
            resolve_result = await asyncio.to_thread(resolve_node, state)
            state.update(resolve_result)

        # ----- Step 5: Send sources before synthesis -----
        yield {
            "event": "sources",
            "data": {
                "sources": state.get("sources", []),
                "graph_path": state.get("graph_path", []),
            },
        }

        # ----- Step 6: Stream synthesis token-by-token -----
        yield {"event": "status", "data": {"step": "synthesizing"}}

        from query.synthesizer import _extract_citations, synthesize_response_stream

        full_answer = ""
        async for chunk in synthesize_response_stream(
            state["query"],
            state.get("results", []),
            state.get("cypher", ""),
            state.get("sources", []),
            state.get("effective_state"),
            fallback_used=state.get("fallback_used", False),
            intent=state.get("intent", ""),
        ):
            full_answer += chunk
            yield {"event": "token", "data": {"text": chunk}}

        # ----- Step 7: Extract citations and emit final done event -----
        citations = _extract_citations(full_answer, state.get("sources", []))

        latency_ms = (time.monotonic() - start_time) * 1000
        trace_url = f"https://smith.langchain.com/public/{run_id}/r"
        source_type = "fulltext_search" if state.get("fallback_used") else "cypher"

        yield {
            "event": "done",
            "data": {
                "answer": full_answer,
                "cypher_query": state.get("cypher", ""),
                "citations": citations,
                "metadata": {
                    "intent": state.get("intent", ""),
                    "retry_count": state.get("retry_count", 0),
                    "latency_ms": round(latency_ms, 2),
                    "source_type": source_type,
                    "fallback_used": state.get("fallback_used", False),
                },
                "trace_url": trace_url,
            },
        }

    except Exception as exc:
        logger.exception("Stream query failed: %s", exc)
        yield {"event": "error", "data": {"message": str(exc)}}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_section_number(state: QueryState) -> int | None:
    """Extract a section number from results or the Cypher query for resolve_node.

    Tries results records first (looking for 'number' or 's.number' fields),
    then falls back to regex extraction from the Cypher string.

    Returns:
        Integer section number if found, else None.
    """
    # Try to extract from result records
    for record in state.get("results", []):
        for key in ("number", "s.number", "s_number"):
            val = record.get(key)
            if val is not None:
                try:
                    return int(str(val))
                except (ValueError, TypeError):
                    continue

    # Fall back to regex extraction from the Cypher string
    cypher = state.get("cypher", "")
    if cypher:
        match = re.search(r"number['\"]?\s*[:=]\s*['\"]?(\d+)['\"]?", cypher, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, TypeError):
                pass

    return None


def _extract_fulltext_sources(ft_results: list[dict]) -> list[dict]:
    """Extract source node excerpts from BM25 fulltext fallback results.

    Fulltext results have a nested structure: {node_props: {...}, score, node_type}.
    This flattens them into the standard sources format.

    Args:
        ft_results: List of result dicts from execute_fulltext_fallback.

    Returns:
        List of source dicts: [{uid, type, text_excerpt}].
    """
    sources = []
    for record in ft_results:
        node_props = record.get("node_props", {}) or {}
        # node_props may be a dict or a neo4j Node object
        if hasattr(node_props, "items"):
            uid = node_props.get("uid", "")
            text = node_props.get("text") or node_props.get("title") or node_props.get("term") or ""
        else:
            uid = getattr(node_props, "get", lambda k, d=None: d)("uid", "")
            text = ""

        node_type = record.get("node_type", "unknown")
        text_excerpt = str(text)[:200] if text else ""

        if uid:
            sources.append(
                {
                    "uid": str(uid),
                    "type": str(node_type),
                    "text_excerpt": text_excerpt,
                }
            )
    return sources
