"""
Type contracts for the ViddhiAI query engine.

QueryState: LangGraph state dict passed between pipeline nodes.
QueryResponse: Pydantic v2 model for FastAPI response serialization.
"""

from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel


class QueryState(TypedDict):
    """State dict threaded through every node in the LangGraph query pipeline.

    All fields are set to their zero-value before the graph starts and updated
    by individual nodes. TypedDict gives type-safety without runtime overhead.
    """

    # Input
    query: str                      # Original natural-language query from user

    # Classification
    intent: str                     # One of: structured_lookup | amendment_query |
                                    #         cross_reference | penalty_query | general_info

    # Cypher generation
    cypher: str                     # LLM-generated Cypher query
    cypher_valid: bool              # True if validator passed; False triggers retry

    # Retry tracking
    retry_count: int                # Total LLM call retries across all error types
    error: str | None               # Cypher execution error (Neo4j driver exception text)
    error_message: str | None       # Last error message injected into retry prompt
    empty_retry_count: int          # Separate counter for empty-result retries (broadening)

    # Execution results
    results: list[dict]             # Raw Neo4j result records as dicts
    fallback_used: bool             # True if BM25 fulltext fallback fired

    # Effective state resolution
    effective_state: dict | None    # EffectiveState fields if resolved (amendment chain)

    # Response synthesis
    answer: str                     # Final prose response to user
    citations: list[dict]           # [{marker, uid, type, text_excerpt, index}]
    sources: list[dict]             # [{uid, type, text_excerpt}] — traceable source nodes
    graph_path: list[str]           # Node chain representation (e.g. Act->Chapter->Section)
    trace_url: str                  # LangSmith trace URL for this query run
    latency_ms: float               # End-to-end query time in milliseconds


class QueryResponse(BaseModel):
    """FastAPI response model for the /api/query endpoint.

    Every field traces back to graph data — the LLM never invents content.
    Serialized as JSON by FastAPI using Pydantic v2.
    """

    answer: str
    """Final prose answer formatted from graph results."""

    cypher_query: str
    """The Cypher query that produced the results — full traceability."""

    graph_path: list[str]
    """Node chain from Act root to answer node (e.g. ['Act:companies-act-2013',
    'Chapter:IX', 'Section:135'])."""

    sources: list[dict]
    """List of source node excerpts: [{uid, type, text_excerpt}]."""

    citations: list[dict] = []
    """Citation markers mapped to source nodes: [{marker, uid, type, text_excerpt, index}]."""

    trace_url: str
    """LangSmith trace URL — click to inspect every LLM call and graph traversal."""

    metadata: dict
    """Execution metadata: intent, retry_count, latency_ms, source_type, fallback_used."""
