"""
Cross-reference detection for Indian legal text.

Identifies statutory cross-references (section citations) from extracted text
and classifies them by relationship type:
    SUBJECT_TO      — "subject to section N"
    NOTWITHSTANDING — "notwithstanding anything in section N"
    REFERS_TO       — "referred to in section N", "under section N", etc.

Security (T-01-05): All patterns use anchored character classes and specific
keyword sequences. No greedy .* that could be exploited by crafted PDF content.
"""

from __future__ import annotations

import re

from ingestion.models import CrossReference

# ---------------------------------------------------------------------------
# Compiled regex patterns for cross-reference detection
# ---------------------------------------------------------------------------
# Each tuple: (compiled_pattern, ref_type_string)
# Patterns are applied in order; all matches are returned (no early exit).

CROSS_REF_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # --- SUBJECT_TO patterns ---
    (
        re.compile(
            r"subject\s+to\s+(?:the\s+provisions\s+of\s+)?section\s+(\d[\d\w]*)",
            re.IGNORECASE,
        ),
        "SUBJECT_TO",
    ),
    # --- NOTWITHSTANDING patterns ---
    (
        re.compile(
            r"notwithstanding\s+(?:anything\s+(?:contained\s+)?in\s+)?section\s+(\d[\d\w]*)",
            re.IGNORECASE,
        ),
        "NOTWITHSTANDING",
    ),
    (
        re.compile(
            r"save\s+as\s+(?:otherwise\s+)?provided\s+in\s+section\s+(\d[\d\w]*)",
            re.IGNORECASE,
        ),
        "NOTWITHSTANDING",
    ),
    (
        re.compile(
            r"except\s+(?:as\s+provided\s+in\s+)?section\s+(\d[\d\w]*)",
            re.IGNORECASE,
        ),
        "NOTWITHSTANDING",
    ),
    # --- REFERS_TO patterns (specific phrasings first, then broader) ---
    (
        re.compile(
            r"(?:as\s+)?(?:referred\s+to|mentioned|specified)\s+in\s+section\s+(\d[\d\w]*)",
            re.IGNORECASE,
        ),
        "REFERS_TO",
    ),
    (
        re.compile(
            r"under\s+section\s+(\d[\d\w]*)",
            re.IGNORECASE,
        ),
        "REFERS_TO",
    ),
    (
        re.compile(
            r"in\s+accordance\s+with\s+section\s+(\d[\d\w]*)",
            re.IGNORECASE,
        ),
        "REFERS_TO",
    ),
    (
        re.compile(
            r"as\s+provided\s+in\s+section\s+(\d[\d\w]*)",
            re.IGNORECASE,
        ),
        "REFERS_TO",
    ),
    (
        re.compile(
            r"in\s+terms\s+of\s+section\s+(\d[\d\w]*)",
            re.IGNORECASE,
        ),
        "REFERS_TO",
    ),
    (
        re.compile(
            r"read\s+with\s+section\s+(\d[\d\w]*)",
            re.IGNORECASE,
        ),
        "REFERS_TO",
    ),
    (
        re.compile(
            r"for\s+the\s+purposes\s+of\s+section\s+(\d[\d\w]*)",
            re.IGNORECASE,
        ),
        "REFERS_TO",
    ),
    (
        re.compile(
            r"in\s+relation\s+to\s+section\s+(\d[\d\w]*)",
            re.IGNORECASE,
        ),
        "REFERS_TO",
    ),
    (
        re.compile(
            r"as\s+required\s+(?:by|under)\s+section\s+(\d[\d\w]*)",
            re.IGNORECASE,
        ),
        "REFERS_TO",
    ),
    (
        re.compile(
            r"within\s+the\s+meaning\s+of\s+section\s+(\d[\d\w]*)",
            re.IGNORECASE,
        ),
        "REFERS_TO",
    ),
    (
        re.compile(
            r"section\s+(\d[\d\w]*)\s+shall\s+(?:mutatis\s+mutandis\s+)?apply",
            re.IGNORECASE,
        ),
        "REFERS_TO",
    ),
]


def extract_cross_references(text: str) -> list[CrossReference]:
    """Extract all cross-references from a legal text string.

    Scans text for section citation patterns and returns a list of
    CrossReference objects. A single text may contain multiple cross-references
    (e.g., "subject to section 149 and notwithstanding section 197").

    Args:
        text: Cleaned text content from a parsed legal element.

    Returns:
        List of CrossReference objects in match order. Empty list if none found.
    """
    if not text:
        return []

    results: list[CrossReference] = []
    seen_spans: set[tuple[int, int]] = set()

    for pattern, ref_type in CROSS_REF_PATTERNS:
        for match in pattern.finditer(text):
            span = match.span()
            # Avoid duplicate matches from overlapping patterns at same position
            if span in seen_spans:
                continue
            seen_spans.add(span)

            section_num = match.group(1)
            target_pattern = f"section {section_num}"
            results.append(
                CrossReference(
                    target_pattern=target_pattern,
                    ref_type=ref_type,
                    raw_text=match.group(0),
                )
            )

    # Sort by position in text for deterministic ordering
    # We can't sort by span directly since we discarded it, so just return in encounter order
    return results
