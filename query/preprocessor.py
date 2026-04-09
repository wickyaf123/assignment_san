"""
Query preprocessor for ViddhiAI query engine.

Normalizes abbreviations, corrects common spelling mistakes, and cleans
user queries before classification and Cypher generation. Spelling
correction uses Levenshtein distance against a domain-specific legal
dictionary — no external dependency or LLM call required.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Domain spell-correction — Levenshtein distance against legal term dictionary
# ---------------------------------------------------------------------------

_LEGAL_TERMS: frozenset[str] = frozenset({
    # Classifier-critical (regex-matched for intent routing)
    "amendment", "amended", "amending", "amendments",
    "penalty", "penalties", "fine", "fines", "punishment", "punishable",
    "imprisonment", "contravention", "compliance", "offence", "violation",
    "liability", "liabilities", "liable",
    "section", "subsection", "provision", "provisions", "chapter", "rule",
    "reference", "referenced", "refers", "cross",
    "substitute", "substituted", "substitution",
    "insert", "inserted", "insertion",
    "omit", "omitted", "modify", "modified", "modification",
    "decriminalized", "revised", "revision", "altered", "alteration",
    "notwithstanding",
    # Graph property values (used in Cypher WHERE clauses)
    "director", "directors", "managing", "independent", "nominee",
    "company", "companies", "private", "public", "foreign", "government",
    "listed", "small", "person",
    "dividend", "dividends", "share", "shares", "debenture", "debentures",
    "capital", "securities", "member", "members", "shareholder",
    "accounts", "audit", "auditor", "auditors",
    "resolution", "ordinary", "special",
    "meeting", "annual", "extraordinary", "general",
    "merger", "amalgamation", "winding",
    "board", "committee", "registrar", "tribunal",
    "corporate", "social", "responsibility",
    "schedule", "form", "prescribed",
    "definition", "definitions", "meaning",
    "power", "powers", "duties", "duty",
    "appointment", "removal", "resignation", "qualification",
    "prospectus", "allotment", "transfer", "transmission",
    "deposit", "deposits", "loan", "loans", "borrowing",
    "charge", "charges", "registered", "registration",
    "inspection", "inquiry", "investigation",
    "oppression", "mismanagement",
    "managerial", "personnel", "remuneration",
    "valuation", "valuer",
    "related", "party", "transactions",
    "quorum", "voting", "proxy", "ballot",
})

_MIN_WORD_LENGTH = 4
_MAX_EDIT_DISTANCE = 2


def _levenshtein(s: str, t: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s) < len(t):
        return _levenshtein(t, s)
    if not t:
        return len(s)

    prev = list(range(len(t) + 1))
    for i, sc in enumerate(s):
        curr = [i + 1]
        for j, tc in enumerate(t):
            cost = 0 if sc == tc else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def _correct_word(word: str) -> str:
    """Correct a single word against the legal term dictionary.

    Returns the original word if it's already correct, too short to safely
    correct, or no close match exists within the edit distance threshold.
    """
    lower = word.lower()
    if lower in _LEGAL_TERMS or len(lower) < _MIN_WORD_LENGTH:
        return word

    best_match: str | None = None
    best_dist = _MAX_EDIT_DISTANCE + 1

    for term in _LEGAL_TERMS:
        if abs(len(term) - len(lower)) > _MAX_EDIT_DISTANCE:
            continue
        dist = _levenshtein(lower, term)
        if dist < best_dist:
            best_dist = dist
            best_match = term

    if best_match is None or best_dist > _MAX_EDIT_DISTANCE:
        return word

    # Preserve original casing style
    if word[0].isupper():
        return best_match.capitalize()
    return best_match


def _correct_spelling(text: str) -> str:
    """Apply domain-aware spelling correction to all words in the text.

    Skips numbers, short words, and words already in the dictionary.
    Only corrects English alphabetic tokens.
    """
    def _replace(m: re.Match) -> str:
        return _correct_word(m.group(0))

    return re.sub(r"[a-zA-Z]+", _replace, text)

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
    """Normalize abbreviations, fix spelling, and clean the user query.

    Pipeline order: spell-correct → abbreviation expansion → Hindi translation.
    Spelling correction runs first so that downstream regex patterns in the
    classifier and abbreviation map match correctly.

    Args:
        query: Raw natural language question from the user.

    Returns:
        Cleaned query with spelling corrected and abbreviations expanded.
        Hindi queries include an English translation appended in brackets.
    """
    result = query.strip()
    lang = _detect_query_language(result)

    # Spell-correct English tokens before any pattern matching
    if lang == "en":
        result = _correct_spelling(result)

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

        # Spell-correct the extracted English portion
        english_version = _correct_spelling(english_version)

        for pattern, replacement in _ABBREVIATIONS:
            english_version = pattern.sub(replacement, english_version)

        if english_version:
            result = f"{result} [English: {english_version}]"

    for pattern, replacement in _ABBREVIATIONS:
        result = pattern.sub(replacement, result)

    return result
