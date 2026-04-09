"""
Tests for ingestion.extractor — OpenDataLoader wrapper and element flattener.

Fast (non-slow) tests use only in-memory fixtures.
Integration tests marked @pytest.mark.slow require actual PDF files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Generator

import pytest

from ingestion.extractor import LEAF_TYPES, flatten_elements, load_raw_json


# ---------------------------------------------------------------------------
# flatten_elements tests
# ---------------------------------------------------------------------------


def test_flatten_elements_yields_leaf_types(sample_nested_kids: list[dict]) -> None:
    """All yielded elements must have a type in LEAF_TYPES; footer must be absent."""
    results = list(flatten_elements(sample_nested_kids))
    assert len(results) > 0, "Expected at least one leaf element"

    for elem in results:
        assert elem["type"] in LEAF_TYPES, (
            f"Unexpected element type {elem['type']!r} — only LEAF_TYPES should be yielded"
        )

    # Verify footer was skipped
    content_texts = [e.get("content", "") for e in results]
    assert not any("Page 1 of" in t for t in content_texts), (
        "Footer content leaked into yielded elements"
    )

    # Expected: 1 heading + 1 paragraph + 2 paragraphs inside list = 4
    assert len(results) == 4, f"Expected 4 leaf elements, got {len(results)}: {results}"


def test_flatten_elements_handles_empty_kids() -> None:
    """Empty input list must produce no output."""
    results = list(flatten_elements([]))
    assert results == []


def test_flatten_elements_recurses_into_list_items() -> None:
    """List element with 3 paragraph kids must yield exactly 3 paragraphs."""
    list_element = {
        "type": "list",
        "id": 1,
        "numbering style": "ordered",
        "number of list items": 3,
        "list items": [
            {
                "type": "list item",
                "id": 2,
                "kids": [
                    {
                        "type": "paragraph",
                        "id": 3,
                        "page number": 1,
                        "content": "Item one",
                        "font": "Arial",
                        "font size": 10.0,
                        "bounding box": [],
                    }
                ],
            },
            {
                "type": "list item",
                "id": 4,
                "kids": [
                    {
                        "type": "paragraph",
                        "id": 5,
                        "page number": 1,
                        "content": "Item two",
                        "font": "Arial",
                        "font size": 10.0,
                        "bounding box": [],
                    }
                ],
            },
            {
                "type": "list item",
                "id": 6,
                "kids": [
                    {
                        "type": "paragraph",
                        "id": 7,
                        "page number": 1,
                        "content": "Item three",
                        "font": "Arial",
                        "font size": 10.0,
                        "bounding box": [],
                    }
                ],
            },
        ],
    }
    results = list(flatten_elements([list_element]))
    assert len(results) == 3, f"Expected 3 paragraphs, got {len(results)}"
    assert all(r["type"] == "paragraph" for r in results)
    contents = [r["content"] for r in results]
    assert "Item one" in contents
    assert "Item two" in contents
    assert "Item three" in contents


def test_flatten_elements_skips_header() -> None:
    """Header elements must be skipped just like footer elements."""
    kids = [
        {
            "type": "header",
            "id": 1,
            "page number": 2,
            "content": "THE COMPANIES ACT, 2013",
            "bounding box": [72.0, 770.0, 540.0, 790.0],
        },
        {
            "type": "paragraph",
            "id": 2,
            "page number": 2,
            "content": "Some content here.",
            "font": "Arial",
            "font size": 10.0,
            "bounding box": [72.0, 700.0, 540.0, 730.0],
        },
    ]
    results = list(flatten_elements(kids))
    assert len(results) == 1
    assert results[0]["type"] == "paragraph"


# ---------------------------------------------------------------------------
# load_raw_json tests
# ---------------------------------------------------------------------------


def test_load_raw_json_validates_kids_key(tmp_path: Path) -> None:
    """load_raw_json must raise ValueError when 'kids' key is missing."""
    bad_file = tmp_path / "bad.json"
    bad_file.write_text(json.dumps({"no_kids": True}))
    with pytest.raises(ValueError, match="missing required 'kids' key"):
        load_raw_json(bad_file)


def test_load_raw_json_valid(tmp_path: Path) -> None:
    """load_raw_json must return a dict with 'kids' key for valid input."""
    valid_file = tmp_path / "valid.json"
    payload = {"kids": [{"type": "paragraph", "content": "test"}]}
    valid_file.write_text(json.dumps(payload))
    result = load_raw_json(valid_file)
    assert "kids" in result
    assert result["kids"][0]["type"] == "paragraph"


# ---------------------------------------------------------------------------
# Integration test (real PDFs, slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_extract_pdfs_integration(pdf_paths: list[Path], raw_dir: Path) -> None:
    """Real PDF extraction: 3 JSONs produced, each valid with 'kids' key.

    This is the primary validation for requirement INGEST-01.
    Marked slow — requires actual PDF files and Java 11+.
    """
    from ingestion.extractor import extract_pdfs

    result = extract_pdfs(pdf_paths, raw_dir)

    # Expect one JSON per input PDF
    assert len(result) == 3, f"Expected 3 JSON outputs, got {len(result)}: {list(result.keys())}"

    for stem, json_path in result.items():
        assert json_path.exists(), f"Output JSON not found: {json_path}"
        assert json_path.suffix == ".json"
        data = load_raw_json(json_path)
        assert "kids" in data, f"'kids' key missing in {json_path}"
        assert len(data["kids"]) > 0, f"Empty 'kids' in {json_path}"
