"""
ViddhiAI FastAPI application.

Exposes 5 endpoints:
  POST /api/query   — NL-to-Cypher query with 60s timeout (API-01)
  POST /api/cypher  — User-submitted read-only Cypher (API-02)
  GET  /api/health  — Neo4j connectivity check (API-03)
  GET  /api/schema  — Live graph schema introspection (API-04)
  POST /api/ingest  — Trigger full ingestion pipeline (API-05)

Design decisions applied:
  D-04: Single flat file, all 5 routes
  D-09/D-10: Uniform error schema {error, detail, status}
  D-11: 60-second timeout on /api/query via asyncio.wait_for
  D-14: CORS from ALLOWED_ORIGINS env var, no wildcard
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from graph.connection import DATABASE, close_driver, get_driver
from query.executor import execute_cypher
from query.flow import run_query, run_query_stream
from query.models import QueryResponse
from query.validator import validate_cypher

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — warm Neo4j connection on startup, close on shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_driver()  # Warm connection on startup
    yield
    close_driver()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="ViddhiAI API", version="0.1.0", lifespan=lifespan)

# ---------------------------------------------------------------------------
# CORS middleware (D-14, RESEARCH.md Pattern 2)
# Strip whitespace from each origin to avoid config errors (Pitfall 3)
# ---------------------------------------------------------------------------

_allowed_origins = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str


class CypherRequest(BaseModel):
    query: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/api/query", response_model=QueryResponse)
async def query_api(request: QueryRequest):
    """Natural-language query endpoint.

    Delegates to run_query() (LangGraph pipeline) with a 90-second timeout.
    Returns a QueryResponse with full traceability: answer, Cypher, graph path,
    sources, LangSmith trace URL, and execution metadata.
    """
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(run_query, request.question),
            timeout=90.0,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail={"error": "gemini_timeout", "detail": "Query timed out after 90 seconds", "status": 504},
        )
    except Exception as exc:
        logger.exception("Query failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "detail": str(exc), "status": 500},
        )
    return QueryResponse(**result)


@app.post("/api/query/stream")
async def query_stream_api(request: QueryRequest):
    """Streaming SSE endpoint for the NL-to-answer query pipeline.

    Returns a Server-Sent Events stream with events:
    - status: Pipeline progress updates
    - cypher: Generated Cypher query
    - sources: Source nodes and graph path
    - token: Answer text chunks (streamed from Gemini)
    - done: Final response with metadata
    - error: Error message on failure
    """

    async def event_generator():
        async for event in run_query_stream(request.question):
            event_type = event["event"]
            event_data = json.dumps(event["data"])
            yield f"event: {event_type}\ndata: {event_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/cypher")
async def cypher_api(request: CypherRequest):
    """User-submitted Cypher endpoint.

    Validates the query for write-safety (blocklist) and additionally blocks
    CALL procedures (not allowed in user-submitted queries; internal LangGraph
    pipeline uses CALL for BM25 fulltext search, so CALL is not in the shared
    validator blocklist).
    """
    is_valid, reason = validate_cypher(request.query)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail={"error": "cypher_rejected", "detail": reason, "status": 400},
        )
    if re.search(r"\bCALL\b", request.query, re.IGNORECASE):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "cypher_rejected",
                "detail": "CALL procedures not allowed in user-submitted queries",
                "status": 400,
            },
        )
    result = await asyncio.to_thread(execute_cypher, request.query)
    if result["error"]:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_query", "detail": result["error"], "status": 400},
        )
    return {"results": result["results"]}


@app.get("/api/health")
async def health_api():
    """Neo4j connectivity health check.

    Returns 200 {"status": "ok", "neo4j": "connected"} when reachable.
    Returns 502 {"error": "neo4j_unavailable"} on any driver or connectivity error.
    """
    try:
        driver = get_driver()
        driver.verify_connectivity()
        return {"status": "ok", "neo4j": "connected"}
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={"error": "neo4j_unavailable", "detail": str(exc), "status": 502},
        )


@app.get("/api/schema")
async def schema_api():
    """Live graph schema introspection.

    Uses native Cypher (no APOC required) to count nodes by label and
    relationships by type. Returns {"node_labels": {...}, "relationship_types": {...}}.
    """
    try:
        driver = get_driver()
        with driver.session(database=DATABASE) as session:
            label_result = session.run(
                "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count ORDER BY label"
            )
            node_counts = {r["label"]: r["count"] for r in label_result}
            rel_result = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(r) AS count ORDER BY count DESC"
            )
            rel_counts = {r["rel_type"]: r["count"] for r in rel_result}
        return {"node_labels": node_counts, "relationship_types": rel_counts}
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={"error": "neo4j_unavailable", "detail": str(exc), "status": 502},
        )


@app.post("/api/ingest")
async def ingest_api():
    """Trigger full ingestion pipeline (Phase 1 + Phase 2).

    Chains run_ingestion() (PDF extraction + parsing) and run_graph_population()
    (Neo4j graph loading). Both are long-running (~5-15 min). Returns a summary
    dict on success.

    Note: Imports are lazy (inside handler) because ingestion.runner and
    graph.runner are heavy modules that start the JVM on import. This is the
    ONE intentional exception to the top-level-imports convention.
    """
    try:
        from ingestion.runner import run_ingestion
        from graph.runner import run_graph_population

        await asyncio.to_thread(run_ingestion)
        summary = await asyncio.to_thread(run_graph_population)
        return {"status": "ok", "summary": summary}
    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={"error": "ingest_failed", "detail": str(exc), "status": 500},
        )
