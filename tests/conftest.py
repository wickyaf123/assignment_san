"""
Shared pytest fixtures for ViddhiAI backend tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# OpenDataLoader element fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_heading_element() -> dict:
    """A typical heading element as produced by OpenDataLoader."""
    return {
        "type": "heading",
        "id": 1,
        "page number": 5,
        "heading level": 1,
        "content": "CHAPTER II",
        "font": "Arial-Bold",
        "font size": 14.0,
        "bounding box": [72.0, 700.0, 540.0, 730.0],
    }


@pytest.fixture
def sample_paragraph_element() -> dict:
    """A typical paragraph element as produced by OpenDataLoader."""
    return {
        "type": "paragraph",
        "id": 2,
        "page number": 5,
        "content": "3. Formation of company.— (1) A company may be formed...",
        "font": "Arial",
        "font size": 10.0,
        "bounding box": [72.0, 650.0, 540.0, 695.0],
    }


@pytest.fixture
def sample_proviso_element() -> dict:
    """A typical proviso paragraph element as produced by OpenDataLoader."""
    return {
        "type": "paragraph",
        "id": 3,
        "page number": 5,
        "content": "Provided that nothing in this sub-section shall apply to...",
        "font": "Arial",
        "font size": 10.0,
        "bounding box": [72.0, 600.0, 540.0, 640.0],
    }


@pytest.fixture
def sample_nested_kids() -> list[dict]:
    """A nested kids structure for testing flatten_elements recursion and footer skipping.

    Structure:
        - heading (leaf)
        - paragraph (leaf)
        - list containing 2 list items each with a paragraph kid
        - footer (should be skipped)

    Expected leaf count: 1 heading + 1 paragraph + 2 paragraphs in list = 4
    """
    return [
        {
            "type": "heading",
            "id": 10,
            "page number": 3,
            "heading level": 2,
            "content": "CHAPTER I — PRELIMINARY",
            "font": "Arial-Bold",
            "font size": 12.0,
            "bounding box": [72.0, 750.0, 540.0, 770.0],
        },
        {
            "type": "paragraph",
            "id": 11,
            "page number": 3,
            "content": "1. Short title, extent, commencement and application.",
            "font": "Arial",
            "font size": 10.0,
            "bounding box": [72.0, 720.0, 540.0, 745.0],
        },
        {
            "type": "list",
            "id": 12,
            "numbering style": "ordered",
            "number of list items": 2,
            "list items": [
                {
                    "type": "list item",
                    "id": 13,
                    "kids": [
                        {
                            "type": "paragraph",
                            "id": 14,
                            "page number": 3,
                            "content": "(a) This Act may be called the Companies Act, 2013.",
                            "font": "Arial",
                            "font size": 10.0,
                            "bounding box": [90.0, 690.0, 540.0, 710.0],
                        }
                    ],
                },
                {
                    "type": "list item",
                    "id": 15,
                    "kids": [
                        {
                            "type": "paragraph",
                            "id": 16,
                            "page number": 3,
                            "content": "(b) It extends to the whole of India.",
                            "font": "Arial",
                            "font size": 10.0,
                            "bounding box": [90.0, 665.0, 540.0, 685.0],
                        }
                    ],
                },
            ],
        },
        {
            "type": "footer",
            "id": 17,
            "page number": 3,
            "content": "Page 1 of 484",
            "bounding box": [72.0, 30.0, 540.0, 50.0],
        },
    ]


# ---------------------------------------------------------------------------
# File/path fixtures
# ---------------------------------------------------------------------------

# Derive project root relative to this file: tests/ -> backend/ -> project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PDF_NAMES = [
    "Companies Act, 2013.pdf",
    "Corporate Laws (Amendment) Act, 2026.pdf",
    "Companies Rules, 2014.pdf",
]


@pytest.fixture
def pdf_paths() -> list[Path]:
    """Paths to the three source PDFs. Skip test if any file is not present."""
    paths = [_PROJECT_ROOT / name for name in _PDF_NAMES]
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"PDFs not found (integration test requires real files): {missing}")
    return paths


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Temporary directory for OpenDataLoader JSON output."""
    return tmp_path / "raw"


# ---------------------------------------------------------------------------
# Neo4j fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def neo4j_driver():
    """Real Neo4j driver for integration tests.

    Skips the test if NEO4J_URI is not set in the environment.
    Tears down by closing the driver after the test.
    """
    import os

    if not os.environ.get("NEO4J_URI"):
        pytest.skip("NEO4J_URI not set — skipping integration test")

    from graph.connection import close_driver, get_driver

    driver = get_driver()
    yield driver
    close_driver()


@pytest.fixture
def mock_driver():
    """Mock Neo4j driver and session for unit tests that don't need a real database.

    Returns:
        tuple[MagicMock, MagicMock]: (mock_driver, mock_session) where
        mock_session is the context-manager session returned by mock_driver.session().
    """
    import neo4j
    from unittest.mock import MagicMock

    mock_drv = MagicMock(spec=neo4j.Driver)
    mock_session = MagicMock()
    mock_drv.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_drv.session.return_value.__exit__ = MagicMock(return_value=False)
    return mock_drv, mock_session
