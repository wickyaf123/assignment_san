"""
Data preprocessing pipeline for ViddhiAI.

Sits between the parsing stage and the graph loading stage to ensure
all text entering Neo4j and the LLM is clean, scored, and consistently
formatted.

Pipeline stages:
    1. TextNormalizer  — encoding fixes, noise removal, footnote separation
    2. QualityScorer   — scores each node 0-100, flags unusable data
    3. LLMFormatter    — adds structural context for LLM consumption

Usage:
    from ingestion.preprocessor import preprocess_act, preprocess_ruleset

    act = parse_companies_act(elements, source_pdf)
    act, act_report = preprocess_act(act)  # cleaned + scored

    ruleset = parse_companies_rules(elements, source_pdf)
    ruleset, rules_report = preprocess_ruleset(ruleset)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from ingestion.models import (
    Act,
    BaseNode,
    Chapter,
    Clause,
    Definition,
    Explanation,
    Proviso,
    Rule,
    RuleSet,
    Section,
    SubClause,
    SubSection,
)
from ingestion.uid_generator import definition_uid
from ingestion.cross_ref_detector import extract_cross_references

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TextNormalizer
# ---------------------------------------------------------------------------

_NULL_RE = re.compile(r"\x00+")
_CONTROL_RE = re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f]+")

_DOT_LEADER_RE = re.compile(r"\.{5,}")
_UNDERSCORE_LEADER_RE = re.compile(r"_{5,}")
_DASH_LEADER_RE = re.compile(r"-{5,}")

# Devanagari text cleanup patterns
_REPEATED_DEVANAGARI_RE = re.compile(r"([\u0900-\u097F]{2,})\1{2,}")
_DOUBLED_HINDI_WORD_RE = re.compile(r"\b([\u0900-\u097F]+)\s+\1\b")
_TRIPLE_PAREN_RE = re.compile(r"\(\s*\(\s*\(\s*\(?\s*")
_TRIPLE_PAREN_CLOSE_RE = re.compile(r"\)\s*\)\s*\)\s*\)?\s*")

_FORM_BLOCK_RE = re.compile(
    r"(?:"
    r"Pre-fill|"
    r"_{10,}|"
    r"\.{10,}|"
    r"(?:\.\s+){5,}\.|"
    r"□\s*□|"
    r"○\s*○|"
    r"\*{3,}"
    r")",
    re.IGNORECASE,
)

_REPEATED_PAREN_RE = re.compile(r"(\(\s*\)\s*){3,}")
_REPEATED_STAR_RE = re.compile(r"(\*\s*){4,}")

_FOOTNOTE_START_RE = re.compile(
    r"(?:^|\.\s+)(\d+\.\s+(?:Ins|Subs|Omitted|Added|Rep|Renumbered)\.\s+by\s+Act\b.*?)(?=\d+\.\s+(?:Ins|Subs|Omitted|Added|Rep|Renumbered)\.\s+by\s+Act\b|$)",
    re.IGNORECASE | re.DOTALL,
)
_NOTIFICATION_RE = re.compile(
    r"(?:vide\s+notification\s+No\.\s+S\.O\.|"
    r"see\s+Gazette\s+of\s+India|"
    r"notification\s+No\.\s+(?:S\.O\.|G\.S\.R\.))",
    re.IGNORECASE,
)
_COMMENCEMENT_RE = re.compile(
    r"\d{1,2}(?:st|nd|rd|th)\s+\w+,?\s+\d{4}\s*-\s*(?:S(?:ection)?s?\.?\s+\d+|Sch)",
    re.IGNORECASE,
)

_CURLY_QUOTES = {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"}
_EM_DASH = "\u2014"
_EN_DASH = "\u2013"
_MULTI_WS_RE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


class TextNormalizer:
    """Cleans text for consistent downstream consumption.
    Supports both English and Devanagari (Hindi) text."""

    @staticmethod
    def normalize(text: str) -> str:
        if not text:
            return text

        text = _NULL_RE.sub("", text)
        text = _CONTROL_RE.sub("", text)

        for curly, straight in _CURLY_QUOTES.items():
            text = text.replace(curly, straight)
        text = text.replace(_EM_DASH, " -- ")
        text = text.replace(_EN_DASH, "-")

        text = TextNormalizer.clean_devanagari(text)

        text = _MULTI_WS_RE.sub(" ", text)
        text = _MULTI_NEWLINE_RE.sub("\n\n", text)

        return text.strip()

    @staticmethod
    def clean_devanagari(text: str) -> str:
        """Clean Devanagari text artifacts from bilingual PDF extraction.

        Fixes common issues:
        - Doubled Hindi words from column-merge errors
        - Repeated Devanagari character sequences
        - Tripled parentheses from layout artifacts
        """
        if not text or not DEVANAGARI_RE.search(text):
            return text

        text = _REPEATED_DEVANAGARI_RE.sub(r"\1", text)
        text = _DOUBLED_HINDI_WORD_RE.sub(r"\1", text)
        text = _TRIPLE_PAREN_RE.sub("(", text)
        text = _TRIPLE_PAREN_CLOSE_RE.sub(")", text)

        return text

    @staticmethod
    def detect_language(text: str) -> str:
        """Detect the primary language of text: 'en', 'hi', or 'mixed'."""
        if not text:
            return "en"
        non_ws = [ch for ch in text if not ch.isspace()]
        if not non_ws:
            return "en"
        dev_count = sum(1 for ch in non_ws if DEVANAGARI_RE.match(ch))
        dev_ratio = dev_count / len(non_ws)
        if dev_ratio > 0.5:
            return "hi"
        elif dev_ratio > 0.1:
            return "mixed"
        return "en"

    @staticmethod
    def strip_form_templates(text: str) -> str:
        """Remove form template artifacts (dot leaders, boxes, pre-fill markers)."""
        if not text:
            return text

        text = _DOT_LEADER_RE.sub("", text)
        text = _UNDERSCORE_LEADER_RE.sub("", text)
        text = _DASH_LEADER_RE.sub("-", text)
        text = _REPEATED_PAREN_RE.sub("", text)
        text = _REPEATED_STAR_RE.sub("", text)

        text = _MULTI_WS_RE.sub(" ", text)
        return text.strip()

    @staticmethod
    def separate_footnotes(text: str) -> tuple[str, list[str]]:
        """Separate legislative footnotes from substantive text.

        Returns (main_text, list_of_footnotes).
        """
        if not text:
            return text, []

        footnotes: list[str] = []
        main_parts: list[str] = []

        sentences = re.split(r"(?<=\.)\s+", text)
        for sentence in sentences:
            if _NOTIFICATION_RE.search(sentence) or _COMMENCEMENT_RE.search(sentence):
                footnotes.append(sentence.strip())
            else:
                main_parts.append(sentence)

        main_text = " ".join(main_parts).strip()
        return main_text, footnotes

    @staticmethod
    def _is_content_char(ch: str) -> bool:
        """Check if a character is meaningful content (alphabetic or Devanagari block)."""
        return ch.isalpha() or bool(DEVANAGARI_RE.match(ch))

    @staticmethod
    def is_garbled(text: str, threshold: float = 0.4) -> bool:
        """Detect garbled/unusable text based on character composition.

        Correctly handles both English and Devanagari text — Devanagari
        vowel signs (matras) are counted as content characters.
        """
        if not text or len(text.strip()) < 5:
            return bool(text and text.strip())

        non_ws = [ch for ch in text if not ch.isspace()]
        if not non_ws:
            return True

        total = len(non_ws)
        content_count = sum(1 for ch in non_ws if TextNormalizer._is_content_char(ch))
        content_ratio = content_count / total

        noise_chars = set("○●□■◆★☆※†‡‖¶•·")
        noise_count = sum(1 for ch in non_ws if ch in noise_chars)
        noise_ratio = noise_count / total

        null_count = text.count("\x00")

        if content_ratio < 0.25 and len(text) > 30:
            return True
        if noise_ratio > 0.15:
            return True
        if null_count > 3:
            return True

        return False

    @staticmethod
    def has_devanagari_majority(text: str, threshold: float = 0.30) -> bool:
        if not text:
            return False
        non_ws = [ch for ch in text if not ch.isspace()]
        if not non_ws:
            return False
        dev_count = sum(1 for ch in non_ws if DEVANAGARI_RE.match(ch))
        return (dev_count / len(non_ws)) > threshold


# ---------------------------------------------------------------------------
# QualityScorer
# ---------------------------------------------------------------------------

_LEGAL_TERMS_RE = re.compile(
    r"\b(section|sub-section|clause|proviso|provided|shall|may|"
    r"company|director|board|tribunal|registrar|act|rule|regulation|"
    r"prescribed|notwithstanding|subject to|deemed|liable|penalty|"
    r"central government|state government|memorandum|articles)\b",
    re.IGNORECASE,
)

_HINDI_LEGAL_TERMS_RE = re.compile(
    r"("
    r"धारा|उपधारा|खंड|परंतुक|अधिनियम|नियम|विनियम|"
    r"कंपनी|कपनी|निदेशक|अध्यक्ष|न्यायाधिकरण|"
    r"रजिस्ट्रार|निगमन|प्रतिभूति|शेयर|पूंजी|"
    r"केंद्र\s*सरकार|राज्य\s*सरकार|"
    r"अनुसूची|प्ररूप|प्रपत्र|संकल्प|"
    r"निवासी|लेखा|परीक्षा|शुल्क"
    r")",
)


@dataclass
class NodeQuality:
    """Quality assessment for a single node."""
    score: float = 0.0
    is_usable: bool = True
    issues: list[str] = field(default_factory=list)
    legal_density: float = 0.0
    text_length: int = 0


class QualityScorer:
    """Scores nodes for data quality on a 0-100 scale."""

    USABILITY_THRESHOLD = 35.0

    @classmethod
    def score(cls, text: str, node_type: str = "") -> NodeQuality:
        if not text or not text.strip():
            return NodeQuality(
                score=0.0,
                is_usable=node_type in ("Act", "Chapter", "RuleSet"),
                issues=["EMPTY_TEXT"],
                text_length=0,
            )

        result = NodeQuality(text_length=len(text))
        issues = result.issues

        non_ws = [ch for ch in text if not ch.isspace()]
        total_chars = len(non_ws) if non_ws else 1

        content_count = sum(1 for ch in non_ws if TextNormalizer._is_content_char(ch))
        content_ratio = content_count / total_chars

        words = text.split()
        word_count = len(words) if words else 1

        en_legal = _LEGAL_TERMS_RE.findall(text)
        hi_legal = _HINDI_LEGAL_TERMS_RE.findall(text)
        result.legal_density = (len(en_legal) + len(hi_legal)) / word_count

        null_count = text.count("\x00")
        dev_count = sum(1 for ch in non_ws if DEVANAGARI_RE.match(ch))
        dev_ratio = dev_count / total_chars
        is_hindi = dev_ratio > 0.3

        if null_count > 0:
            issues.append("NULL_BYTES")
        if is_hindi:
            issues.append("HINDI_TEXT")
        if content_ratio < 0.25 and len(text) > 30:
            issues.append("LOW_ALPHA_RATIO")
        if len(text) < 20:
            issues.append("VERY_SHORT")
        if len(text) > 5000:
            issues.append("OVERSIZED")
        if result.legal_density == 0 and len(text) > 100:
            issues.append("NO_LEGAL_TERMS")

        form_hits = len(_FORM_BLOCK_RE.findall(text))
        if form_hits > 5:
            issues.append("FORM_TEMPLATE_HEAVY")

        encoding_penalty = min(null_count * 10, 40)
        length_score = min(len(text) / 50, 1.0) * 20
        legal_score = min(result.legal_density * 300, 30)
        alpha_score = content_ratio * 30
        form_penalty = min(form_hits * 2, 20)

        result.score = max(0.0, min(100.0,
            length_score + legal_score + alpha_score
            - encoding_penalty - form_penalty
        ))

        result.is_usable = (
            result.score >= cls.USABILITY_THRESHOLD
            and "NULL_BYTES" not in issues
            and "LOW_ALPHA_RATIO" not in issues
        )

        return result


# ---------------------------------------------------------------------------
# LLMFormatter
# ---------------------------------------------------------------------------

class LLMFormatter:
    """Formats node text with structural context for LLM consumption."""

    @staticmethod
    def format_section(section: Section, chapter: Chapter | None = None) -> str:
        """Format a section with full context header."""
        parts = []

        header = f"Section {section.number}"
        if section.title:
            header += f" ({section.title})"
        if chapter:
            header = f"Chapter {chapter.number} - {chapter.title} > {header}"
        parts.append(header)
        parts.append("")

        if section.text:
            main_text, _ = TextNormalizer.separate_footnotes(section.text)
            parts.append(main_text)

        return "\n".join(parts)

    @staticmethod
    def format_definition(term: str, text: str, section_number: int = 2) -> str:
        return f'Definition: "{term}" [Section {section_number}]\n\n{text}'

    @staticmethod
    def format_node_with_path(
        node_text: str,
        ancestors: list[str],
    ) -> str:
        """Format any node with its ancestry path."""
        path = " > ".join(ancestors)
        return f"{path}\n\n{node_text}"

    @staticmethod
    def format_rule(rule: Rule, ruleset_title: str = "") -> str:
        parts = []
        header = f"Rule {rule.number}"
        if rule.title:
            header += f" ({rule.title})"
        if ruleset_title:
            header = f"{ruleset_title} > {header}"
        if rule.language != "en":
            header += f" [lang:{rule.language}]"
        parts.append(header)
        parts.append("")
        if rule.text:
            parts.append(rule.text)
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Definition extractor (post-parse)
# ---------------------------------------------------------------------------

_DEFINITION_RE = re.compile(
    r'^\((\d+)\)\s*["\u201c\u2018]([^"\u201d\u2019]+)["\u201d\u2019]\s+'
    r"(?:means|shall\s+mean|includes|shall\s+include)\b",
    re.IGNORECASE,
)

_DEFINITION_ALT_RE = re.compile(
    r'^\((\d+)\)\s*"([^"]+)"\s+'
    r"(?:means|shall\s+mean|includes|shall\s+include)\b",
    re.IGNORECASE,
)

_DEFINITION_STRAIGHT_RE = re.compile(
    r'^"([^"]+)"\s+(?:means|shall\s+mean|includes|shall\s+include)\b',
    re.IGNORECASE,
)


def extract_definitions_from_section2(act: Act) -> int:
    """Post-parse extraction of definitions from Section 2 subsections.

    Section 2 of the Companies Act defines ~95 terms as subsections like:
        (20) "company" means a company incorporated...

    The parser may miss these because DEFINITION_TERM_RE requires specific
    quote characters. This function scans Section 2 subsections and creates
    Definition nodes from those that match definition patterns.

    Returns count of definitions extracted.
    """
    count = 0
    section_2: Section | None = None

    for chapter in act.chapters:
        for section in chapter.sections:
            if section.number == 2:
                section_2 = section
                break
        if section_2:
            break

    if not section_2:
        logger.warning("Section 2 not found in Act — cannot extract definitions")
        return 0

    for subsection in section_2.subsections:
        text = subsection.text.strip()
        if not text:
            continue

        term: str | None = None

        for pattern in [_DEFINITION_RE, _DEFINITION_ALT_RE]:
            m = pattern.match(text)
            if m:
                term = m.group(2).strip().lower()
                break

        if not term:
            m = _DEFINITION_STRAIGHT_RE.match(text)
            if m:
                term = m.group(1).strip().lower()

        if not term and '"' in text:
            quote_match = re.search(r'"([^"]+)"', text)
            if quote_match:
                candidate = quote_match.group(1).strip()
                after_quote = text[quote_match.end():].strip().lower()
                if after_quote.startswith(("means", "shall mean", "includes", "shall include")):
                    term = candidate.lower()

        if term and len(term) > 1 and len(term) < 80:
            existing_terms = {d.term.lower() for d in subsection.definitions}
            if term not in existing_terms:
                defn = Definition(
                    uid=definition_uid(term),
                    node_type="Definition",
                    page_number=subsection.page_number,
                    text=text,
                    source_pdf=subsection.source_pdf,
                    term=term,
                    cross_references=extract_cross_references(text),
                )
                subsection.definitions.append(defn)
                count += 1

    if count > 0:
        logger.info("Extracted %d definitions from Section 2 subsections", count)
    return count


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

@dataclass
class PreprocessReport:
    """Report from preprocessing a document."""
    nodes_processed: int = 0
    nodes_cleaned: int = 0
    nodes_flagged_unusable: int = 0
    definitions_extracted: int = 0
    footnotes_separated: int = 0
    garbled_nodes_cleaned: int = 0
    quality_scores: list[float] = field(default_factory=list)

    @property
    def avg_quality(self) -> float:
        return sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0.0


def _process_node(node: BaseNode, normalizer: TextNormalizer, scorer: QualityScorer) -> NodeQuality:
    """Normalize text, detect language, and score a single node."""
    if node.text:
        node.text = normalizer.normalize(node.text)
        node.text = normalizer.strip_form_templates(node.text)
        node.language = normalizer.detect_language(node.text)

    quality = scorer.score(node.text, node.node_type)
    return quality


def preprocess_act(act: Act) -> tuple[Act, PreprocessReport]:
    """Run the full preprocessing pipeline on a parsed Act.

    1. Normalize all node text (encoding, formatting)
    2. Separate footnotes from section text
    3. Extract definitions from Section 2
    4. Score all nodes for quality
    """
    normalizer = TextNormalizer()
    scorer = QualityScorer()
    report = PreprocessReport()

    report.definitions_extracted = extract_definitions_from_section2(act)

    for chapter in act.chapters:
        _process_node(chapter, normalizer, scorer)
        report.nodes_processed += 1

        for section in chapter.sections:
            quality = _process_node(section, normalizer, scorer)
            report.nodes_processed += 1
            report.quality_scores.append(quality.score)
            if not quality.is_usable:
                report.nodes_flagged_unusable += 1

            if section.text and len(section.text) > 2000:
                main_text, footnotes = normalizer.separate_footnotes(section.text)
                if footnotes:
                    section.text = main_text
                    report.footnotes_separated += len(footnotes)

            for ss in section.subsections:
                q = _process_node(ss, normalizer, scorer)
                report.nodes_processed += 1
                report.quality_scores.append(q.score)

                for cl in ss.clauses:
                    _process_node(cl, normalizer, scorer)
                    report.nodes_processed += 1
                    for sc in cl.subclauses:
                        _process_node(sc, normalizer, scorer)
                        report.nodes_processed += 1

            for cl in section.clauses:
                _process_node(cl, normalizer, scorer)
                report.nodes_processed += 1
                for sc in cl.subclauses:
                    _process_node(sc, normalizer, scorer)
                    report.nodes_processed += 1

    report.nodes_cleaned = report.nodes_processed
    logger.info(
        "Preprocessed Act: %d nodes, %d flagged unusable, %d definitions extracted, "
        "%d footnotes separated, avg quality %.1f",
        report.nodes_processed,
        report.nodes_flagged_unusable,
        report.definitions_extracted,
        report.footnotes_separated,
        report.avg_quality,
    )
    return act, report


def preprocess_ruleset(ruleset: RuleSet) -> tuple[RuleSet, PreprocessReport]:
    """Run the full preprocessing pipeline on a parsed RuleSet.

    1. Normalize all rule text
    2. Flag and clean garbled rules
    3. Score all nodes
    """
    normalizer = TextNormalizer()
    scorer = QualityScorer()
    report = PreprocessReport()

    clean_rules: list[Rule] = []

    for rule in ruleset.rules:
        quality = _process_node(rule, normalizer, scorer)
        report.nodes_processed += 1
        report.quality_scores.append(quality.score)

        if normalizer.is_garbled(rule.text):
            report.garbled_nodes_cleaned += 1
            report.nodes_flagged_unusable += 1
            logger.debug(
                "Garbled rule %s (score=%.1f): %s",
                rule.number, quality.score, rule.text[:60],
            )
            continue

        if not rule.text.strip():
            report.nodes_flagged_unusable += 1
            continue

        if not quality.is_usable:
            report.nodes_flagged_unusable += 1
            continue

        try:
            base_num = rule.number.split("-dup-")[0] if "-dup-" in rule.number else rule.number
            rule_num = int(re.sub(r"[^\d]", "", base_num))
            if rule_num > 500:
                logger.debug(
                    "Filtering rule with abnormal number %s (parsed as %d)",
                    rule.number, rule_num,
                )
                report.nodes_flagged_unusable += 1
                continue
        except (ValueError, TypeError):
            pass

        clean_rules.append(rule)

        for form in rule.forms:
            _process_node(form, normalizer, scorer)
            report.nodes_processed += 1

    ruleset.rules = clean_rules
    report.nodes_cleaned = len(clean_rules)

    logger.info(
        "Preprocessed RuleSet: %d nodes processed, %d rules kept (from %d), "
        "%d garbled, %d flagged unusable, avg quality %.1f",
        report.nodes_processed,
        len(clean_rules),
        report.nodes_processed,
        report.garbled_nodes_cleaned,
        report.nodes_flagged_unusable,
        report.avg_quality,
    )
    return ruleset, report
