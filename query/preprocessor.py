"""
Query preprocessor for ViddhiAI query engine.

Normalizes abbreviations and cleans user queries before classification
and Cypher generation. This improves intent classification accuracy
and helps the LLM generate better Cypher by expanding domain-specific
abbreviations into their full legal terms.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Abbreviation map — pattern → replacement
# Ordered so longer/more specific patterns match first.
# ---------------------------------------------------------------------------

_ABBREVIATIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bCSR\b"), "Corporate Social Responsibility (Section 135)"),
    (re.compile(r"\bAGM\b"), "Annual General Meeting (Section 96)"),
    (re.compile(r"\bEGM\b"), "Extraordinary General Meeting"),
    (re.compile(r"\bKMP\b"), "Key Managerial Personnel"),
    (re.compile(r"\bMD\b"), "Managing Director"),
    (re.compile(r"\bROC\b"), "Registrar of Companies"),
    (re.compile(r"\bNCLT\b"), "National Company Law Tribunal"),
    (re.compile(r"\bNCLAT\b"), "National Company Law Appellate Tribunal"),
    (re.compile(r"\bsub[- ]?sec\.?\s+", re.IGNORECASE), "sub-section "),
    (re.compile(r"\bsec\.?\s+", re.IGNORECASE), "section "),
]

_HINDI_ABBREVIATIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bसीएसआर\b"), "कॉर्पोरेट सामाजिक उत्तरदायित्व (धारा 135)"),
    (re.compile(r"\bएजीएम\b"), "वार्षिक साधारण सभा (धारा 96)"),
    (re.compile(r"\bकेएमपी\b"), "प्रमुख प्रबंधकीय कार्मिक"),
    (re.compile(r"\bएनसीएलटी\b"), "राष्ट्रीय कंपनी विधि न्यायाधिकरण"),
]

_HINDI_ABBREV_TO_ENGLISH: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bसीएसआर\b"), "CSR"),
    (re.compile(r"\bएजीएम\b"), "AGM"),
    (re.compile(r"\bकेएमपी\b"), "KMP"),
    (re.compile(r"\bएनसीएलटी\b"), "NCLT"),
]

_HINDI_TO_ENGLISH: dict[str, str] = {
    "धारा": "section",
    "उपधारा": "sub-section",
    "कंपनी": "company",
    "निदेशक": "director",
    "अधिनियम": "act",
    "नियम": "rule",
    "परिभाषा": "definition",
    "अध्याय": "chapter",
    "दंड": "penalty",
    "संशोधन": "amendment",
    "शेयर": "share",
    "लाभांश": "dividend",
    "पूंजी": "capital",
    "सदस्य": "member",
    "लेखा": "accounts",
    "लेखापरीक्षा": "audit",
    "प्रतिभूति": "securities",
    "शक्ति": "power",
    "अनुसूची": "schedule",
    "प्रबंध": "management",
    "प्रशासन": "administration",
    "बैठक": "meeting",
    "प्रस्ताव": "resolution",
    "समापन": "winding up",
    "विलय": "merger",
    "संबंधित पक्ष": "related party",
    "कर्तव्य": "duties",
    "उत्तरदायित्व": "liability",
    "सामाजिक उत्तरदायित्व": "corporate social responsibility",
}

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def _detect_query_language(query: str) -> str:
    """Detect whether a query is in Hindi, English, or mixed."""
    non_ws = [ch for ch in query if not ch.isspace()]
    if not non_ws:
        return "en"
    dev_count = sum(1 for ch in non_ws if _DEVANAGARI_RE.match(ch))
    dev_ratio = dev_count / len(non_ws)
    if dev_ratio > 0.5:
        return "hi"
    elif dev_ratio > 0.1:
        return "mixed"
    return "en"


_HINDI_QUESTION_WORDS: dict[str, str] = {
    "क्या है": "what is",
    "क्या कहती है": "what does it say",
    "क्या कहता है": "what does it say",
    "क्या हैं": "what are",
    "कौन से": "which",
    "कौन सी": "which",
    "कितने": "how many",
    "कितनी": "how many",
    "कैसे": "how",
    "बताइए": "tell me about",
    "बताओ": "tell me about",
    "दिखाइए": "show me",
    "दिखाओ": "show me",
    "के बारे में": "about",
    "के तहत": "under",
    "में क्या": "what in",
    "का अर्थ": "meaning of",
    "की सूची": "list of",
}


def preprocess_query(query: str) -> str:
    """Normalize abbreviations and clean the user query.

    Supports both English and Hindi queries. Hindi queries are translated
    to English using a legal term dictionary so the LLM generates correct
    Cypher against the English-language graph data.

    Args:
        query: Raw natural language question from the user.

    Returns:
        Cleaned query with abbreviations expanded. Hindi queries include
        an English translation appended in brackets.
    """
    result = query.strip()
    lang = _detect_query_language(result)

    if lang in ("hi", "mixed"):
        english_version = result

        for pattern, replacement in _HINDI_ABBREV_TO_ENGLISH:
            english_version = pattern.sub(replacement, english_version)

        for hindi_term, english_term in sorted(
            _HINDI_TO_ENGLISH.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if hindi_term in english_version:
                english_version = english_version.replace(hindi_term, english_term)

        for hindi_phrase, english_phrase in sorted(
            _HINDI_QUESTION_WORDS.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if hindi_phrase in english_version:
                english_version = english_version.replace(hindi_phrase, english_phrase)

        english_version = re.sub(r"[\u0900-\u097F?।]+", "", english_version).strip()
        english_version = re.sub(r"\s+", " ", english_version).strip()

        for pattern, replacement in _ABBREVIATIONS:
            english_version = pattern.sub(replacement, english_version)

        if english_version:
            result = f"{result} [English: {english_version}]"

    for pattern, replacement in _ABBREVIATIONS:
        result = pattern.sub(replacement, result)

    return result
