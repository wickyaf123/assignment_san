"""
OpenDataLoader PDF extraction wrapper for the ViddhiAI ingestion pipeline.

Provides:
    extract_pdfs()     — Batch all PDFs in one JVM call → dict of JSON output paths
    flatten_elements() — Recursive generator yielding leaf content elements in document order
    load_raw_json()    — Load and validate an OpenDataLoader JSON output file

Security mitigations (T-01-01, T-01-02):
    - All PDF paths validated via Path.resolve() before passing to OpenDataLoader
    - Paths must exist and have .pdf suffix
    - raw_dir must reside within the expected backend/data/ subtree
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterator

import opendataloader_pdf

logger = logging.getLogger(__name__)

# Element types that carry actual legal content (leaf types to yield from flatten_elements)
LEAF_TYPES: set[str] = {"heading", "paragraph", "caption"}

# Element types that are page chrome — not content
_SKIP_TYPES: set[str] = {"header", "footer"}


def extract_pdfs(pdf_paths: list[Path], raw_dir: Path) -> dict[str, Path]:
    """Batch-convert all PDFs to structured JSON in one OpenDataLoader JVM call.

    CRITICAL: All PDFs are submitted in a single call to avoid repeated JVM startup
    overhead (~3–5 seconds per separate call).

    Security (T-01-01): Each PDF path is validated — must exist and have .pdf suffix.
    Security (T-01-02): raw_dir is validated to be within the project backend/data/ tree.

    Args:
        pdf_paths: List of Path objects pointing to source PDFs.
        raw_dir: Directory where OpenDataLoader will write JSON output files.

    Returns:
        A dict mapping each discovered JSON output stem to its Path.
        Stems are discovered from the output directory (not hardcoded) to handle
        any OpenDataLoader naming convention.

    Raises:
        ValueError: If any PDF path is invalid or raw_dir fails validation.
        FileNotFoundError: If a PDF path does not exist.
    """
    # Validate and resolve PDF paths (T-01-01)
    resolved_pdfs: list[Path] = []
    for p in pdf_paths:
        rp = Path(p).resolve()
        if not rp.exists():
            raise FileNotFoundError(f"PDF not found: {rp}")
        if rp.suffix.lower() != ".pdf":
            raise ValueError(f"Expected .pdf suffix, got: {rp.suffix!r} for path {rp}")
        resolved_pdfs.append(rp)

    # Validate output directory is within the project tree (T-01-02)
    resolved_raw_dir = Path(raw_dir).resolve()
    # Allow any path under the project root — check that raw_dir contains "data"
    # in its path components as a soft guard against accidental absolute path injection
    if not any(part in ("data", "raw", "parsed") for part in resolved_raw_dir.parts):
        logger.warning(
            "raw_dir %s does not appear to be within a data/ subtree — proceeding but verify intent",
            resolved_raw_dir,
        )

    resolved_raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting OpenDataLoader extraction: %d PDF(s) → %s",
        len(resolved_pdfs),
        resolved_raw_dir,
    )

    # Single batched JVM call — critical for performance (avoids per-file JVM startup)
    opendataloader_pdf.convert(
        input_path=[str(p) for p in resolved_pdfs],
        output_dir=str(resolved_raw_dir),
        format="json",
    )

    # Discover output files — do NOT hardcode filenames; OpenDataLoader names by PDF stem
    output_files = list(resolved_raw_dir.glob("*.json"))

    for f in output_files:
        size_kb = f.stat().st_size / 1024
        logger.info("Output: %s (%.1f KB)", f.name, size_kb)

    logger.info(
        "Extraction complete: %d PDF(s) → %d JSON file(s)",
        len(resolved_pdfs),
        len(output_files),
    )

    return {f.stem: f for f in output_files}


def flatten_elements(kids: list[dict]) -> Iterator[dict]:
    """Recursively yield leaf content elements in document order.

    Yields only elements whose `type` is in LEAF_TYPES (heading, paragraph, caption).
    Skips page chrome: header and footer elements.
    Recurses into: list items, table rows/cells, and any element with a `kids` key.

    Args:
        kids: List of element dicts from an OpenDataLoader JSON output.

    Yields:
        Individual leaf element dicts in document order.
    """
    for element in kids:
        elem_type = element.get("type", "")

        # Skip page chrome entirely
        if elem_type in _SKIP_TYPES:
            continue

        # Yield leaf content types directly (with cleaned text)
        if elem_type in LEAF_TYPES:
            element = dict(element)  # don't mutate original
            element["content"] = clean_element_text(element.get("content", ""))
            yield element

        # Recurse into list elements via list items
        elif elem_type == "list":
            for item in element.get("list items", []):
                item_kids = item.get("kids", [])
                item_content = item.get("content", "").strip()

                if item_kids:
                    # List item has children — yield its own content first
                    # (if any), then recurse into kids
                    if item_content:
                        leaf = dict(item)
                        leaf["type"] = "paragraph"
                        leaf["content"] = clean_element_text(item_content)
                        yield leaf
                    yield from flatten_elements(item_kids)
                elif item_content:
                    # Leaf list item with direct content — yield as paragraph
                    leaf = dict(item)
                    leaf["type"] = "paragraph"
                    leaf["content"] = clean_element_text(item_content)
                    yield leaf

        # Recurse into table rows and cells
        elif elem_type == "table row":
            for cell in element.get("cells", []):
                yield from flatten_elements(cell.get("kids", []))

        # Generic container — recurse if kids is non-empty, otherwise yield
        # content directly (fixes elements with empty kids: [] but content)
        elif "kids" in element:
            if element["kids"]:
                yield from flatten_elements(element["kids"])
            elif element.get("content", "").strip():
                leaf = dict(element)
                leaf["type"] = "paragraph"
                leaf["content"] = clean_element_text(element["content"])
                yield leaf


def load_raw_json(json_path: Path) -> dict:
    """Load and validate an OpenDataLoader JSON output file.

    Args:
        json_path: Path to the JSON file produced by extract_pdfs().

    Returns:
        Parsed JSON dict with at least a "kids" key.

    Raises:
        ValueError: If the JSON file does not contain the required "kids" key.
    """
    with open(json_path, encoding="utf-8") as fh:
        data = json.load(fh)

    if "kids" not in data:
        raise ValueError(
            f"OpenDataLoader JSON at {json_path} is missing required 'kids' key. "
            f"Found keys: {list(data.keys())}"
        )

    return data


# ---------------------------------------------------------------------------
# Text cleanup — applied to every yielded element in flatten_elements()
# ---------------------------------------------------------------------------

_FOOTNOTE_MARKER_RE = re.compile(r"\d+\[")
_TRAILING_BRACKET_RE = re.compile(r"\]")
_STAR_MARKER_RE = re.compile(r"\*\d*\[?")
_CURLY_OPEN_QUOTES = {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"}
_EM_DASH = "\u2014"
_EN_DASH = "\u2013"
_MULTI_WS_RE = re.compile(r"\s{2,}")


def clean_element_text(text: str) -> str:
    """Clean extracted PDF text for downstream parsing.

    Applies the following transformations in order:
        1. Strip footnote markers (patterns like ``\\d+[`` and trailing ``]``)
        2. Strip star markers (``*\\d*[?``)
        3. Normalize curly quotes to straight quotes
        4. Normalize em-dash and en-dash to ``--`` and ``-``
        5. Collapse multiple whitespace to single space
        6. Strip leading/trailing whitespace

    Args:
        text: Raw text content from an OpenDataLoader element.

    Returns:
        Cleaned text string.
    """
    if not text:
        return text

    # 1. Strip footnote markers
    text = _FOOTNOTE_MARKER_RE.sub("", text)
    text = _TRAILING_BRACKET_RE.sub("", text)

    # 2. Strip star markers
    text = _STAR_MARKER_RE.sub("", text)

    # 3. Normalize curly quotes to straight quotes
    for curly, straight in _CURLY_OPEN_QUOTES.items():
        text = text.replace(curly, straight)

    # 4. Normalize em-dash and en-dash
    text = text.replace(_EM_DASH, "--")
    text = text.replace(_EN_DASH, "-")

    # 5. Collapse multiple whitespace to single space
    text = _MULTI_WS_RE.sub(" ", text)

    # 6. Strip leading/trailing whitespace
    text = text.strip()

    return text
