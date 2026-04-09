"""
Intent classifier for ViddhiAI query engine.

Uses deterministic regex/keyword matching to classify natural language
legal queries into one of 5 intent categories. Zero-latency, no LLM call.

Intent categories: structured_lookup, amendment_query, cross_reference,
penalty_query, general_info.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_AMENDMENT_PATTERNS = [
    re.compile(r"\bamend(ed|ment|ing|s)?\b", re.IGNORECASE),
    re.compile(r"\bchange[ds]?\b.*\b(section|provision|chapter)", re.IGNORECASE),
    re.compile(r"\bmodif(y|ied|ication)\b", re.IGNORECASE),
    re.compile(r"\bsubstitut(e[ds]?|ion)\b", re.IGNORECASE),
    re.compile(r"\binsert(ed|ion|s)?\b", re.IGNORECASE),
    re.compile(r"\bomit(ted|s)?\b", re.IGNORECASE),
    re.compile(r"\bdecriminal", re.IGNORECASE),
    re.compile(r"\brevis(e[ds]?|ion)\b", re.IGNORECASE),
    re.compile(r"\balter(ed|ation|ing)?\b", re.IGNORECASE),
]

_PENALTY_PATTERNS = [
    re.compile(r"\bpenalt(y|ies)\b", re.IGNORECASE),
    re.compile(r"\bfine[ds]?\b", re.IGNORECASE),
    re.compile(r"\bpunish(ment|able|ed)?\b", re.IGNORECASE),
    re.compile(r"\bimprison(ment|ed)?\b", re.IGNORECASE),
    re.compile(r"\bcontravention\b", re.IGNORECASE),
    re.compile(r"\bnon[- ]?compliance\b", re.IGNORECASE),
    re.compile(r"\boffence\b", re.IGNORECASE),
    re.compile(r"\bviolation\b", re.IGNORECASE),
    re.compile(r"\bliabilit(y|ies)\b", re.IGNORECASE),
    re.compile(r"\bpersonally\s+liable\b", re.IGNORECASE),
]

_CROSS_REF_PATTERNS = [
    re.compile(r"\brefer(s|ence|enced|ring)?\b.*\bto\b", re.IGNORECASE),
    re.compile(r"\bcross[- ]?refer", re.IGNORECASE),
    re.compile(r"\bsubject\s+to\b", re.IGNORECASE),
    re.compile(r"\bnotwithstanding\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+(sections?|provisions?|rules?)\s+(refer|cite|mention)", re.IGNORECASE),
    re.compile(r"\brules?\b.*\b(relate|reference|apply|prescribe)", re.IGNORECASE),
    re.compile(r"\bcorresponding\s+(rules?|sections?|provisions?)\b", re.IGNORECASE),
    re.compile(r"\b(have|has|with)\s+no\s+(corresponding|related|matching)\b", re.IGNORECASE),
    re.compile(r"\bchapters?\b.*\brules?\b", re.IGNORECASE),
    re.compile(r"\brules?\b.*\bchapters?\b", re.IGNORECASE),
    re.compile(r"\bwhich\s+(rules?|sections?)\s+(are|exist|apply|link|connect)", re.IGNORECASE),
]

_GENERAL_INFO_PATTERNS = [
    re.compile(r"\bhow\s+many\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(does|is)\s+(chapter|the\s+(companies\s+)?act)\b", re.IGNORECASE),
    re.compile(r"\boverview\b", re.IGNORECASE),
    re.compile(r"\bprovisions?\s+of\s+the\b", re.IGNORECASE),
]

_SECTION_REF_RE = re.compile(r"\b(section|s\.\s*\d|rule|chapter|provision)\b", re.IGNORECASE)


def classify_intent(query: str) -> str:
    """Classify a natural language legal query into one of 5 intent categories.

    Uses deterministic keyword/regex matching for zero-latency classification,
    replacing the previous LLM-based approach. Priority order: amendment >
    penalty (with section ref) > cross_reference > general_info > structured_lookup.

    Args:
        query: Natural language question from the user.

    Returns:
        One of: 'structured_lookup', 'amendment_query', 'cross_reference',
        'penalty_query', 'general_info'.
    """
    # Amendment patterns (highest priority — drives resolve_node routing)
    if any(p.search(query) for p in _AMENDMENT_PATTERNS):
        logger.info("Classified query as 'amendment_query': %s", query[:80])
        return "amendment_query"

    # Penalty patterns — require a section/provision reference to avoid
    # false matches on vague queries like "What is the penalty?"
    has_section_ref = bool(_SECTION_REF_RE.search(query))
    if has_section_ref and any(p.search(query) for p in _PENALTY_PATTERNS):
        logger.info("Classified query as 'penalty_query': %s", query[:80])
        return "penalty_query"

    # Cross-reference patterns
    if any(p.search(query) for p in _CROSS_REF_PATTERNS):
        logger.info("Classified query as 'cross_reference': %s", query[:80])
        return "cross_reference"

    # General info patterns
    if any(p.search(query) for p in _GENERAL_INFO_PATTERNS):
        logger.info("Classified query as 'general_info': %s", query[:80])
        return "general_info"

    # Default: structured_lookup (most common intent for direct section queries)
    logger.info("Classified query as 'structured_lookup' (default): %s", query[:80])
    return "structured_lookup"
