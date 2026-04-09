"""Tests for the ViddhiAI FastAPI API (all 5 endpoints).

Uses FastAPI TestClient with unittest.mock.patch to isolate all external
dependencies (Neo4j, Gemini, ingestion pipeline). No live services required.

Covers:
  - POST /api/query  — success, timeout (504), missing field (422)
  - POST /api/cypher — valid read, write rejected (400), CALL rejected (400)
  - GET  /api/health — healthy (200), Neo4j down (502)
  - GET  /api/schema — shape check (200 with node_labels + relationship_types)
  - POST /api/ingest — success (200), failure (500)
  - CORS headers — Origin header reflects access-control-allow-origin
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Shared mock data
# ---------------------------------------------------------------------------

MOCK_QUERY_RESULT = {
    "answer": "Section 135 mandates CSR.",
    "cypher_query": "MATCH (s:Section {number: 135}) RETURN s",
    "graph_path": ["Act:companies-act-2013", "Section:135"],
    "sources": [{"uid": "sec-135", "type": "Section", "text_excerpt": "CSR..."}],
    "trace_url": "https://smith.langchain.com/runs/abc123",
    "metadata": {
        "intent": "structured_lookup",
        "retry_count": 1,
        "latency_ms": 450.0,
        "source_type": "cypher",
        "fallback_used": False,
    },
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """TestClient with mocked Neo4j driver (avoids real connections in lifespan)."""
    with patch("main.get_driver") as mock_gd, patch("main.close_driver"):
        mock_driver = MagicMock()
        mock_gd.return_value = mock_driver
        from main import app

        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# POST /api/query tests
# ---------------------------------------------------------------------------


def test_query_endpoint(client):
    """Successful NL query returns 200 with all QueryResponse fields."""
    with patch("main.run_query", return_value=MOCK_QUERY_RESULT):
        resp = client.post("/api/query", json={"question": "What is CSR?"})

    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert "cypher_query" in body
    assert "graph_path" in body
    assert "sources" in body
    assert "trace_url" in body
    assert "metadata" in body
    assert body["answer"] == "Section 135 mandates CSR."


def test_query_timeout(client):
    """Slow run_query triggers asyncio.wait_for timeout — returns 504 gemini_timeout."""

    def slow_query(question):
        # Sleep longer than the 60s timeout so asyncio.wait_for triggers.
        # TestClient runs with a very short timeout override via monkeypatch below.
        time.sleep(200)
        return {}

    # Patch the timeout to 0.1s so the test doesn't actually wait 60 seconds.
    with patch("main.run_query", side_effect=slow_query), \
         patch("main.asyncio.wait_for", side_effect=asyncio.TimeoutError):
        resp = client.post("/api/query", json={"question": "timeout test"})

    assert resp.status_code == 504
    body = resp.json()
    assert body["detail"]["error"] == "gemini_timeout"


def test_query_missing_question(client):
    """POST /api/query with missing 'question' field returns 422 Unprocessable Entity."""
    resp = client.post("/api/query", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/cypher tests
# ---------------------------------------------------------------------------


def test_cypher_valid_read(client):
    """Valid read-only MATCH query returns 200 with results list."""
    mock_result = {"results": [{"n": "test"}], "error": None}
    with patch("main.execute_cypher", return_value=mock_result):
        resp = client.post(
            "/api/cypher",
            json={"query": "MATCH (n:Section) RETURN n LIMIT 5"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body


def test_cypher_write_rejected(client):
    """CREATE keyword in user-submitted Cypher returns 400 cypher_rejected."""
    resp = client.post(
        "/api/cypher",
        json={"query": "CREATE (n:Section {uid: 'test'})"},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"]["error"] == "cypher_rejected"


def test_cypher_call_rejected(client):
    """CALL keyword in user-submitted Cypher returns 400 cypher_rejected."""
    resp = client.post(
        "/api/cypher",
        json={"query": "CALL db.schema.visualization()"},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["detail"]["error"] == "cypher_rejected"


# ---------------------------------------------------------------------------
# GET /api/health tests
# ---------------------------------------------------------------------------


def test_health_ok(client):
    """Healthy Neo4j driver returns 200 with status ok and neo4j connected."""
    with patch("main.get_driver") as mock_gd:
        mock_drv = MagicMock()
        mock_drv.verify_connectivity.return_value = None  # no exception = healthy
        mock_gd.return_value = mock_drv
        resp = client.get("/api/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["neo4j"] == "connected"


def test_health_neo4j_down(client):
    """Driver raising exception on verify_connectivity returns 502 neo4j_unavailable."""
    with patch("main.get_driver") as mock_gd:
        mock_drv = MagicMock()
        mock_drv.verify_connectivity.side_effect = Exception("Connection refused")
        mock_gd.return_value = mock_drv
        resp = client.get("/api/health")

    assert resp.status_code == 502
    body = resp.json()
    assert body["detail"]["error"] == "neo4j_unavailable"


# ---------------------------------------------------------------------------
# GET /api/schema tests
# ---------------------------------------------------------------------------


def test_schema_shape(client):
    """Schema endpoint returns 200 with node_labels and relationship_types dicts."""
    # Build mock records for label query and rel query
    label_record = MagicMock()
    label_record.__getitem__ = lambda self, k: {"label": "Section", "count": 488}[k]

    rel_record = MagicMock()
    rel_record.__getitem__ = lambda self, k: {"rel_type": "HAS_SECTION", "count": 200}[k]

    mock_label_result = [label_record]
    mock_rel_result = [rel_record]

    mock_session = MagicMock()
    # session.run() returns different iterables depending on the call order
    mock_session.run.side_effect = [mock_label_result, mock_rel_result]

    mock_drv = MagicMock()
    mock_drv.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_drv.session.return_value.__exit__ = MagicMock(return_value=False)

    with patch("main.get_driver", return_value=mock_drv):
        resp = client.get("/api/schema")

    assert resp.status_code == 200
    body = resp.json()
    assert "node_labels" in body
    assert "relationship_types" in body
    assert isinstance(body["node_labels"], dict)
    assert isinstance(body["relationship_types"], dict)


# ---------------------------------------------------------------------------
# POST /api/ingest tests
# ---------------------------------------------------------------------------


def test_ingest_success(client):
    """Successful ingestion returns 200 with status ok and summary dict."""
    import sys
    import types

    mock_summary = {"nodes_created": 100, "relationships_created": 50}

    # The ingest endpoint uses lazy imports. Ensure stub modules exist in
    # sys.modules so that patch() can find them by dotted name.
    ingestion_mod = sys.modules.get("ingestion") or types.ModuleType("ingestion")
    runner_mod = types.ModuleType("ingestion.runner")
    runner_mod.run_ingestion = lambda: None
    sys.modules.setdefault("ingestion", ingestion_mod)
    sys.modules["ingestion.runner"] = runner_mod

    graph_mod = sys.modules.get("graph") or types.ModuleType("graph")
    graph_runner_mod = types.ModuleType("graph.runner")
    graph_runner_mod.run_graph_population = lambda: {}
    sys.modules.setdefault("graph", graph_mod)
    sys.modules["graph.runner"] = graph_runner_mod

    with patch("ingestion.runner.run_ingestion", return_value=None), \
         patch("graph.runner.run_graph_population", return_value=mock_summary):
        resp = client.post("/api/ingest")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "summary" in body


def test_ingest_failure(client):
    """Exception during ingestion returns 500 with error ingest_failed."""
    import sys
    import types

    # Ensure ingestion.runner exists in sys.modules for patching.
    ingestion_mod = sys.modules.get("ingestion") or types.ModuleType("ingestion")
    runner_mod = types.ModuleType("ingestion.runner")
    runner_mod.run_ingestion = lambda: None
    sys.modules.setdefault("ingestion", ingestion_mod)
    sys.modules["ingestion.runner"] = runner_mod

    with patch("ingestion.runner.run_ingestion", side_effect=RuntimeError("PDF not found")):
        resp = client.post("/api/ingest")

    assert resp.status_code == 500
    body = resp.json()
    assert body["detail"]["error"] == "ingest_failed"


# ---------------------------------------------------------------------------
# CORS tests
# ---------------------------------------------------------------------------


def test_cors_headers(client):
    """POST /api/query with allowed Origin returns Access-Control-Allow-Origin header."""
    with patch("main.run_query", return_value=MOCK_QUERY_RESULT):
        resp = client.post(
            "/api/query",
            json={"question": "CORS test"},
            headers={"Origin": "http://localhost:3000"},
        )

    assert resp.status_code == 200
    assert "access-control-allow-origin" in resp.headers
    assert resp.headers["access-control-allow-origin"] == "http://localhost:3000"
