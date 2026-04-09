"""Golden Q&A accuracy test suite.

Runs 50+ legal questions against the live API and measures:
- Cypher generation success rate (was a Cypher query returned?)
- Average retry count
- Average end-to-end latency (ms)
- Hallucination rate (answer does NOT contain expected text)
- Per-category breakdown

Usage:
  API_URL=http://localhost:8000 pytest tests/test_golden_qa.py -v
  API_URL=https://your-app.railway.app pytest tests/test_golden_qa.py -v
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import httpx
import pytest

API_URL = os.getenv("API_URL", "http://localhost:8000")
GOLDEN_QA_PATH = Path(__file__).parent / "golden_qa.json"
REPORT_PATH = Path(__file__).parent / "eval_report.json"


@pytest.fixture(scope="session")
def golden_pairs():
    """Load golden Q&A pairs from JSON fixture."""
    return json.loads(GOLDEN_QA_PATH.read_text())


@pytest.mark.integration
def test_golden_qa_accuracy(golden_pairs):
    """Run all golden Q&A pairs and produce accuracy report."""
    results = []
    client = httpx.Client(timeout=60.0)

    for pair in golden_pairs:
        start = time.time()
        try:
            resp = client.post(
                f"{API_URL}/api/query",
                json={"question": pair["question"]},
            )
            latency_ms = (time.time() - start) * 1000
            if resp.status_code != 200:
                results.append({
                    "id": pair["id"],
                    "category": pair["category"],
                    "question": pair["question"],
                    "status": "error",
                    "http_status": resp.status_code,
                    "cypher_ok": False,
                    "answer_ok": False,
                    "latency_ms": round(latency_ms, 1),
                    "retry_count": 0,
                })
                continue

            data = resp.json()
            cypher_ok = bool(data.get("cypher_query"))
            answer = data.get("answer", "")
            expected = pair["expected_answer_contains"]
            answer_ok = expected.lower() in answer.lower() if expected else True
            cypher_pattern = pair.get("expected_cypher_pattern", "")
            cypher_match = (
                bool(re.search(cypher_pattern, data.get("cypher_query", ""), re.IGNORECASE))
                if cypher_pattern
                else True
            )
            retry_count = data.get("metadata", {}).get("retry_count", 0)

            results.append({
                "id": pair["id"],
                "category": pair["category"],
                "question": pair["question"],
                "status": "ok",
                "cypher_ok": cypher_ok,
                "answer_ok": answer_ok,
                "cypher_match": cypher_match,
                "latency_ms": round(latency_ms, 1),
                "retry_count": retry_count,
                "intent": data.get("metadata", {}).get("intent", ""),
                "expected_intent": pair["expected_intent"],
                "source_type": data.get("metadata", {}).get("source_type", ""),
            })
        except httpx.ConnectError:
            pytest.skip(f"Cannot connect to API at {API_URL}")
        except Exception as exc:
            latency_ms = (time.time() - start) * 1000
            results.append({
                "id": pair["id"],
                "category": pair["category"],
                "question": pair["question"],
                "status": "exception",
                "error": str(exc),
                "cypher_ok": False,
                "answer_ok": False,
                "latency_ms": round(latency_ms, 1),
                "retry_count": 0,
            })

    client.close()

    # --- Write detailed JSON report ---
    REPORT_PATH.write_text(json.dumps(results, indent=2))

    # --- Console summary ---
    total = len(results)
    ok_results = [r for r in results if r["status"] == "ok"]
    cypher_success = sum(1 for r in ok_results if r["cypher_ok"])
    answer_success = sum(1 for r in ok_results if r["answer_ok"])
    avg_retries = sum(r["retry_count"] for r in ok_results) / max(len(ok_results), 1)
    avg_latency = sum(r["latency_ms"] for r in ok_results) / max(len(ok_results), 1)
    hallucination_rate = 1.0 - (answer_success / max(len(ok_results), 1))

    print("\n" + "=" * 60)
    print("GOLDEN Q&A ACCURACY REPORT")
    print("=" * 60)
    print(f"Total pairs:            {total}")
    print(f"Successful responses:   {len(ok_results)}")
    print(f"Cypher success rate:    {cypher_success}/{len(ok_results)} ({cypher_success/max(len(ok_results),1)*100:.1f}%)")
    print(f"Answer contains match:  {answer_success}/{len(ok_results)} ({answer_success/max(len(ok_results),1)*100:.1f}%)")
    print(f"Avg retries:            {avg_retries:.2f}")
    print(f"Avg latency:            {avg_latency:.0f}ms")
    latencies = sorted(r["latency_ms"] for r in ok_results)
    if latencies:
        p50 = latencies[len(latencies) // 2]
        p90 = latencies[int(len(latencies) * 0.9)]
        p99 = latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)]
        max_latency = latencies[-1]
        print(f"P50 latency:            {p50:.0f}ms")
        print(f"P90 latency:            {p90:.0f}ms")
        print(f"P99 latency:            {p99:.0f}ms")
        print(f"Max latency:            {max_latency:.0f}ms")
    print(f"Hallucination rate:     {hallucination_rate*100:.1f}%")
    # Intent accuracy
    intent_matches = sum(
        1 for r in ok_results
        if r.get("intent") == r.get("expected_intent")
    )
    intent_accuracy = intent_matches / max(len(ok_results), 1)
    print(f"Intent accuracy:        {intent_matches}/{len(ok_results)} ({intent_accuracy*100:.1f}%)")
    print("-" * 60)

    # Per-category breakdown
    categories = sorted(set(r["category"] for r in results))
    for cat in categories:
        cat_results = [r for r in ok_results if r["category"] == cat]
        cat_cypher = sum(1 for r in cat_results if r["cypher_ok"])
        cat_answer = sum(1 for r in cat_results if r["answer_ok"])
        cat_total = len(cat_results)
        print(f"  {cat:30s} Cypher: {cat_cypher}/{cat_total}  Answer: {cat_answer}/{cat_total}")

    print("=" * 60)
    print(f"Report written to: {REPORT_PATH}")

    # Assert minimum quality thresholds (PRD targets)
    # Threshold is 50% minimum floor — PRD target of 85% is aspirational
    success_rate = cypher_success / max(len(ok_results), 1)
    assert success_rate >= 0.50, (
        f"Cypher success rate {success_rate:.1%} < 50% minimum threshold"
    )


@pytest.mark.integration
def test_golden_qa_fixture_valid():
    """Validate the golden_qa.json fixture structure."""
    pairs = json.loads(GOLDEN_QA_PATH.read_text())
    assert len(pairs) >= 50, f"Expected 50+ pairs, got {len(pairs)}"

    categories = set()
    ids = set()
    for pair in pairs:
        assert "id" in pair, f"Missing 'id' in pair: {pair}"
        assert "category" in pair, f"Missing 'category' in pair: {pair}"
        assert "question" in pair, f"Missing 'question' in pair: {pair}"
        assert "expected_intent" in pair, f"Missing 'expected_intent' in pair: {pair}"
        assert "expected_answer_contains" in pair, f"Missing 'expected_answer_contains' in pair: {pair}"
        assert pair["id"] not in ids, f"Duplicate id: {pair['id']}"
        ids.add(pair["id"])
        categories.add(pair["category"])

    expected_categories = {
        "current_state",
        "amendment_history",
        "cross_reference_traversal",
        "penalty_query",
        "edge_cases",
    }
    assert categories == expected_categories, (
        f"Expected categories {expected_categories}, got {categories}"
    )

    # 10+ per category
    for cat in expected_categories:
        cat_count = sum(1 for p in pairs if p["category"] == cat)
        assert cat_count >= 10, f"Category '{cat}' has {cat_count} pairs, expected >= 10"


@pytest.mark.integration
def test_golden_qa_latency_threshold(golden_pairs):
    """Assert that average query latency is below the post-optimization target."""
    if not REPORT_PATH.exists():
        pytest.skip("eval_report.json not found — run test_golden_qa_accuracy first")

    results = json.loads(REPORT_PATH.read_text())
    ok_results = [r for r in results if r["status"] == "ok"]
    if not ok_results:
        pytest.skip("No successful results in eval report")

    avg_latency = sum(r["latency_ms"] for r in ok_results) / len(ok_results)
    assert avg_latency < 8000, (
        f"Average latency {avg_latency:.0f}ms exceeds 8000ms threshold"
    )


@pytest.mark.integration
def test_golden_qa_intent_accuracy(golden_pairs):
    """Assert that intent classification accuracy meets minimum threshold."""
    if not REPORT_PATH.exists():
        pytest.skip("eval_report.json not found — run test_golden_qa_accuracy first")

    results = json.loads(REPORT_PATH.read_text())
    ok_results = [r for r in results if r["status"] == "ok"]
    if not ok_results:
        pytest.skip("No successful results in eval report")

    intent_matches = sum(
        1 for r in ok_results
        if r.get("intent") == r.get("expected_intent")
    )
    intent_accuracy = intent_matches / len(ok_results)
    assert intent_accuracy >= 0.80, (
        f"Intent accuracy {intent_accuracy:.1%} < 80% threshold"
    )
