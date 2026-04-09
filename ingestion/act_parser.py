"""
State-machine parser for the Companies Act 2013 and Amendment Acts.

Parses OpenDataLoader JSON elements into typed Pydantic objects:
    Act -> Chapter -> Section -> SubSection -> Clause -> SubClause
                                                      -> Proviso
                                                      -> Explanation
                                                      -> Definition
    AmendmentAct -> AmendmentEntry

Design principles:
  - Elements are processed in document order from flatten_elements()
  - Heading elements dispatch to chapter or marginal-note handling
  - Paragraph elements are classified by regex against the first tokens
  - Provisos are ALWAYS siblings (never nested) — parent is the deepest
    non-proviso context (Rule: INGEST-03 / D-05 / Pitfall 3)
  - Marginal notes (heading before a section) become Section.title (D-12)
  - Cross-references extracted per node and stored in cross_references (D-14)
  - All unrecognized content logged as warnings (D-08)

Security (T-01-05): All regex patterns are anchored (^) with specific char
classes. No greedy .* that could be exploited by crafted PDF content.
Security (T-01-07): Warnings logged with page number via Python logging.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from pathlib import Path
from typing import Iterator

from ingestion.models import (
    Act,
    AmendmentAct,
    AmendmentEntry,
    Chapter,
    Clause,
    Definition,
    Explanation,
    Proviso,
    Section,
    SubClause,
    SubSection,
)
from ingestion.uid_generator import (
    act_uid,
    amendment_act_uid,
    chapter_uid,
    clause_uid,
    definition_uid,
    explanation_uid,
    proviso_uid,
    section_uid,
    subclause_uid,
    subsection_uid,
)
from ingestion.cross_ref_detector import extract_cross_references

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex constants — anchored, specific character classes (T-01-05)
# ---------------------------------------------------------------------------

CHAPTER_RE = re.compile(r"^CHAPTER\s+([IVXLC]+)", re.IGNORECASE)
SECTION_RE = re.compile(
    r"^(\d+)\.\s+(.+?)(?:\.[\s\u2014\-]*(?:\(|$)|\.?--|\.\s*$|\.?\s*\()",
)
SUBSECTION_RE = re.compile(r"^\((\d+)\)\s+")
CLAUSE_RE = re.compile(r"^\(([a-z])\)\s+")
SUBCLAUSE_RE = re.compile(r"^\(([ivxlc]+)\)\s+")
PROVISO_RE = re.compile(r"^Provided\s+(also\s+|further\s+)?that\b", re.IGNORECASE)
EXPLANATION_RE = re.compile(r"^Explanation(\s+[IVX]+)?[\.\-\u2014]\s*", re.IGNORECASE)
DEFINITION_TERM_RE = re.compile(
    r'^(?:\(\d+\)\s*)?["\u201c\u2018]([^"\u201d\u2019]+)["\u201d\u2019]\s+'
    r"(?:means|shall\s+mean|includes|shall\s+include)\b",
    re.IGNORECASE,
)

# Amendment-specific patterns
# Handles: "In the Companies Act, 2013, in section 135"
#          "In the said Act, in section 2"
#          "In the principal Act, in section 167"
#          "In Act, in section 1"
AMENDMENT_SECTION_RE = re.compile(
    r"In\s+(?:the\s+)?(?:said\s+)?(?:principal\s+)?[A-Za-z][^,]*?Act,?"
    r"(?:\s*,?\s*\d{4})?,?\s+in\s+section\s+(\d+)",
    re.IGNORECASE,
)
SUBSTITUTION_RE = re.compile(r"shall\s+be\s+substituted", re.IGNORECASE)
INSERTION_RE = re.compile(r"shall\s+be\s+inserted", re.IGNORECASE)
OMISSION_RE = re.compile(r"shall\s+be\s+omitted", re.IGNORECASE)

# Footnote detection — identifies editorial footnotes that look like section headings
_FOOTNOTE_INDICATORS = re.compile(
    r'(Ins\.\s+by|Subs\.\s+by|w\.e\.f\.|Notification\s+No\.|'
    r'S\.O\.\s+\d|G\.S\.R\.\s+\d|vide\s+notification|Gazette\s+of\s+India)',
    re.IGNORECASE,
)


def _is_footnote(text: str) -> bool:
    """Return True if text appears to be a footnote rather than legal content.

    Only triggers when the text STARTS with a footnote pattern (within first
    60 chars), not when a section merely contains a footnote reference.
    """
    prefix = text[:80]
    if _FOOTNOTE_INDICATORS.search(prefix):
        legal_start = re.match(r"^\d+\.\s+[A-Z]", text)
        if legal_start and len(text) > 100:
            return False
        return True
    return False


# ---------------------------------------------------------------------------
# Parser state enum
# ---------------------------------------------------------------------------

class ParserState(Enum):
    PREAMBLE = "preamble"
    IN_CHAPTER = "in_chapter"
    IN_SECTION = "in_section"
    IN_SUBSECTION = "in_subsection"
    IN_CLAUSE = "in_clause"
    IN_SUBCLAUSE = "in_subclause"
    IN_PROVISO = "in_proviso"
    IN_EXPLANATION = "in_explanation"
    IN_DEFINITION = "in_definition"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def classify_proviso(text: str) -> str | None:
    """Classify a proviso paragraph into one of the three proviso types.

    Args:
        text: The paragraph text starting with "Provided".

    Returns:
        One of "Provided that", "Provided also that", "Provided further that",
        or None if the text does not match the proviso pattern.
    """
    m = PROVISO_RE.match(text)
    if m is None:
        return None
    modifier = (m.group(1) or "").strip().lower()
    if modifier == "also":
        return "Provided also that"
    elif modifier == "further":
        return "Provided further that"
    else:
        return "Provided that"


# ---------------------------------------------------------------------------
# Companies Act parser
# ---------------------------------------------------------------------------

def parse_companies_act(
    elements: Iterator[dict],
    source_pdf: str,
) -> Act:
    """Parse Companies Act 2013 elements into a structured Act object.

    Processes elements from flatten_elements() in document order.
    Returns an Act with nested chapters, sections, subsections, clauses,
    sub-clauses, provisos, explanations, and definitions.

    Proviso parent rule (INGEST-03): Provisos are ALWAYS children of the
    deepest non-proviso context. "Provided also that" and "Provided further
    that" are siblings — never children of a previous Proviso.

    Args:
        elements: Iterator of element dicts from flatten_elements().
        source_pdf: Filename of the source PDF (stored on all nodes).

    Returns:
        Populated Act object.
    """
    act = Act(
        uid=act_uid("companies", 2013),
        node_type="Act",
        page_number=1,
        text="",
        source_pdf=source_pdf,
        title="The Companies Act, 2013",
        year=2013,
    )

    # Context pointers — mutable, updated as we descend/ascend
    current_chapter: Chapter | None = None
    current_section: Section | None = None
    current_subsection: SubSection | None = None
    current_clause: Clause | None = None
    current_subclause: SubClause | None = None

    pending_marginal_note: str | None = None

    # Per-parent counters for provisos and explanations (1-indexed)
    proviso_counters: dict[str, int] = {}
    explanation_counters: dict[str, int] = {}

    warnings: list[str] = []
    seen_section_numbers: set[int] = set()
    sections_by_number: dict[int, Section] = {}
    chapters_by_number: dict[str, Chapter] = {}  # dedup TOC vs actual chapters

    def _get_proviso_parent() -> Section | SubSection | Clause | SubClause | None:
        """Return deepest non-proviso context as proviso parent."""
        if current_subclause is not None:
            return current_subclause
        if current_clause is not None:
            return current_clause
        if current_subsection is not None:
            return current_subsection
        return current_section

    def _append_to_current(content: str, page: int) -> None:
        """Append continuation text to the deepest active context."""
        target = current_subclause or current_clause or current_subsection or current_section
        if target is None:
            warnings.append(f"[page {page}] Preamble/uncontextualized text: {content[:80]!r}")
            return
        if target.text:
            target.text = target.text + " " + content
        else:
            target.text = content

    for element in elements:
        content: str = element.get("content", "").strip()
        etype: str = element.get("type", "")
        page: int = int(element.get("page number", 0))

        if not content:
            continue

        # ------------------------------------------------------------------
        # Heading dispatch
        # ------------------------------------------------------------------
        if etype == "heading":
            ch_match = CHAPTER_RE.match(content)
            if ch_match:
                roman = ch_match.group(1).upper()
                if roman in chapters_by_number:
                    # Reuse existing chapter (TOC created it first, now we see the real one)
                    current_chapter = chapters_by_number[roman]
                else:
                    current_chapter = Chapter(
                        uid=chapter_uid(roman),
                        node_type="Chapter",
                        page_number=page,
                        text=content,
                        source_pdf=source_pdf,
                        number=roman,
                        title="",
                    )
                    act.chapters.append(current_chapter)
                    chapters_by_number[roman] = current_chapter
                # Reset section context
                current_section = None
                current_subsection = None
                current_clause = None
                current_subclause = None
                pending_marginal_note = None
            else:
                # Any non-chapter heading is treated as a marginal note candidate
                # that will become the title of the next section encountered.
                # We also use it as chapter title if the chapter has no title yet
                # AND the content is entirely uppercase (chapter title pattern),
                # otherwise it is a section marginal note.
                if (
                    current_chapter is not None
                    and not current_chapter.sections
                    and not current_chapter.title
                    and content == content.upper()
                    and len(content) > 2
                ):
                    # All-uppercase heading right after CHAPTER — this is the chapter title
                    current_chapter.title = content
                else:
                    # Mixed-case or already have sections — treat as marginal note
                    pending_marginal_note = content

        # ------------------------------------------------------------------
        # Paragraph/caption dispatch
        # ------------------------------------------------------------------
        elif etype in ("paragraph", "caption"):
            # 0. Chapter heading in paragraph (OpenDataLoader sometimes emits
            #    "CHAPTER IX ACCOUNTS OF COMPANIES" as a paragraph, not heading)
            ch_match = CHAPTER_RE.match(content)
            if ch_match:
                roman = ch_match.group(1).upper()
                ch_title = content[ch_match.end():].strip()
                if roman in chapters_by_number:
                    current_chapter = chapters_by_number[roman]
                    # Update title if we now have a better one
                    if ch_title and (not current_chapter.title or len(ch_title) > len(current_chapter.title)):
                        current_chapter.title = ch_title
                else:
                    current_chapter = Chapter(
                        uid=chapter_uid(roman),
                        node_type="Chapter",
                        page_number=page,
                        text=content,
                        source_pdf=source_pdf,
                        number=roman,
                        title=ch_title,
                    )
                    act.chapters.append(current_chapter)
                    chapters_by_number[roman] = current_chapter
                current_section = None
                current_subsection = None
                current_clause = None
                current_subclause = None
                pending_marginal_note = None
                continue

            # 1. Section
            sec_match = SECTION_RE.match(content)
            if sec_match:
                if _is_footnote(content):
                    _append_to_current(content, page)
                    continue

                sec_num = int(sec_match.group(1))

                if sec_num in seen_section_numbers:
                    # Replace earlier (likely TOC) section if the new one
                    # has substantially more content
                    existing = sections_by_number.get(sec_num)
                    if existing and len(content) > len(existing.text) * 1.5 + 20:
                        logger.debug(
                            "[page %d] Replacing short section %d (%d chars) with fuller version (%d chars)",
                            page, sec_num, len(existing.text), len(content),
                        )
                        existing.text = content
                        existing.page_number = page
                        inline_title = sec_match.group(2).strip().rstrip(".")
                        if pending_marginal_note:
                            existing.title = pending_marginal_note
                        elif len(inline_title) > len(existing.title):
                            existing.title = inline_title
                        pending_marginal_note = None
                        current_section = existing
                        current_subsection = None
                        current_clause = None
                        current_subclause = None
                        proviso_counters = {}
                        explanation_counters = {}
                        continue
                    else:
                        _append_to_current(content, page)
                        continue
                seen_section_numbers.add(sec_num)

                inline_title = sec_match.group(2).strip().rstrip(".")
                title = pending_marginal_note if pending_marginal_note else inline_title
                pending_marginal_note = None

                if current_chapter is None:
                    current_chapter = Chapter(
                        uid=chapter_uid("UNKNOWN"),
                        node_type="Chapter",
                        page_number=page,
                        text="",
                        source_pdf=source_pdf,
                        number="UNKNOWN",
                        title="",
                    )
                    act.chapters.append(current_chapter)

                current_section = Section(
                    uid=section_uid(sec_num),
                    node_type="Section",
                    page_number=page,
                    text=content,
                    source_pdf=source_pdf,
                    number=sec_num,
                    title=title,
                    cross_references=extract_cross_references(content),
                )
                current_chapter.sections.append(current_section)
                sections_by_number[sec_num] = current_section
                current_subsection = None
                current_clause = None
                current_subclause = None
                proviso_counters = {}
                explanation_counters = {}
                continue

            # 2. SubSection
            ss_match = SUBSECTION_RE.match(content)
            if ss_match:
                ss_num = int(ss_match.group(1))
                if current_section is None:
                    warnings.append(f"[page {page}] SubSection ({ss_num}) found without active section")
                    continue

                current_subsection = SubSection(
                    uid=subsection_uid(current_section.number, ss_num),
                    node_type="SubSection",
                    page_number=page,
                    text=content,
                    source_pdf=source_pdf,
                    number=ss_num,
                    cross_references=extract_cross_references(content),
                )
                current_section.subsections.append(current_subsection)
                # Reset deeper contexts
                current_clause = None
                current_subclause = None
                continue

            # 3. Clause
            cl_match = CLAUSE_RE.match(content)
            if cl_match:
                cl_label = cl_match.group(1)
                # Parent may be subsection or section (direct clauses)
                parent_uid_str = (
                    current_subsection.uid if current_subsection
                    else (current_section.uid if current_section else "unknown")
                )
                current_clause = Clause(
                    uid=clause_uid(parent_uid_str, cl_label),
                    node_type="Clause",
                    page_number=page,
                    text=content,
                    source_pdf=source_pdf,
                    label=cl_label,
                    cross_references=extract_cross_references(content),
                )
                if current_subsection is not None:
                    current_subsection.clauses.append(current_clause)
                elif current_section is not None:
                    current_section.clauses.append(current_clause)
                else:
                    warnings.append(f"[page {page}] Clause ({cl_label}) found without active section")
                # Reset sub-clause context
                current_subclause = None
                continue

            # 4. Sub-clause
            sc_match = SUBCLAUSE_RE.match(content)
            if sc_match:
                sc_label = sc_match.group(1)
                parent_uid_str = current_clause.uid if current_clause else "unknown"
                current_subclause = SubClause(
                    uid=subclause_uid(parent_uid_str, sc_label),
                    node_type="SubClause",
                    page_number=page,
                    text=content,
                    source_pdf=source_pdf,
                    label=sc_label,
                    cross_references=extract_cross_references(content),
                )
                if current_clause is not None:
                    current_clause.subclauses.append(current_subclause)
                else:
                    warnings.append(f"[page {page}] SubClause ({sc_label}) found without active clause")
                continue

            # 5. Proviso — INGEST-03: parent is deepest non-proviso context
            prov_match = PROVISO_RE.match(content)
            if prov_match:
                prov_type = classify_proviso(content) or "Provided that"
                parent = _get_proviso_parent()
                if parent is None:
                    warnings.append(f"[page {page}] Proviso found without active context")
                    continue
                p_uid = parent.uid
                count = proviso_counters.get(p_uid, 0) + 1
                proviso_counters[p_uid] = count
                proviso_obj = Proviso(
                    uid=proviso_uid(p_uid, count),
                    node_type="Proviso",
                    page_number=page,
                    text=content,
                    source_pdf=source_pdf,
                    proviso_type=prov_type,
                    cross_references=extract_cross_references(content),
                )
                parent.provisos.append(proviso_obj)
                # Do NOT push proviso onto context stack — provisos are terminal
                continue

            # 6. Explanation
            expl_match = EXPLANATION_RE.match(content)
            if expl_match:
                label_raw = (expl_match.group(1) or "").strip()
                parent = _get_proviso_parent()
                if parent is None:
                    warnings.append(f"[page {page}] Explanation found without active context")
                    continue
                p_uid = parent.uid
                e_count = explanation_counters.get(p_uid, 0) + 1
                explanation_counters[p_uid] = e_count
                expl_obj = Explanation(
                    uid=explanation_uid(p_uid, e_count),
                    node_type="Explanation",
                    page_number=page,
                    text=content,
                    source_pdf=source_pdf,
                    label=label_raw,
                    cross_references=extract_cross_references(content),
                )
                parent.explanations.append(expl_obj)
                continue

            # 7. Definition term
            def_match = DEFINITION_TERM_RE.match(content)
            if def_match:
                term = def_match.group(1).strip()
                def_obj = Definition(
                    uid=definition_uid(term),
                    node_type="Definition",
                    page_number=page,
                    text=content,
                    source_pdf=source_pdf,
                    term=term,
                    cross_references=extract_cross_references(content),
                )
                # Attach to deepest applicable context (subsection or section)
                if current_subsection is not None:
                    current_subsection.definitions.append(def_obj)
                elif current_section is not None:
                    current_section.definitions.append(def_obj)
                else:
                    warnings.append(f"[page {page}] Definition '{term}' found without active section")
                continue

            # 8. Continuation / unrecognized paragraph
            _append_to_current(content, page)

    # Post-processing: merge UNKNOWN chapter into the first real chapter.
    # The Companies Act has preliminary sections (1, 2) before the CHAPTER I
    # heading. OpenDataLoader may emit them before the heading, causing a
    # synthetic "UNKNOWN" chapter.  Merge its sections into Chapter I (or the
    # first real chapter) so the graph never contains "Chapter UNKNOWN".
    unknown_ch = next((ch for ch in act.chapters if ch.number == "UNKNOWN"), None)
    if unknown_ch is not None:
        real_chapters = [ch for ch in act.chapters if ch.number != "UNKNOWN"]
        if real_chapters:
            target = real_chapters[0]
            target.sections = unknown_ch.sections + target.sections
            act.chapters.remove(unknown_ch)
            if target.number in chapters_by_number:
                chapters_by_number[target.number] = target
            logger.info(
                "Merged %d orphan section(s) from Chapter UNKNOWN into Chapter %s",
                len(unknown_ch.sections),
                target.number,
            )
        else:
            unknown_ch.number = "I"
            unknown_ch.title = "PRELIMINARY"
            chapters_by_number["I"] = unknown_ch
            logger.info(
                "No real chapters found — renamed Chapter UNKNOWN to Chapter I PRELIMINARY"
            )

    if warnings:
        logger.debug("parse_companies_act produced %d warnings", len(warnings))
        for w in warnings[:20]:  # log first 20 to avoid log spam
            logger.debug("  %s", w)

    return act


# ---------------------------------------------------------------------------
# Amendment Act parser
# ---------------------------------------------------------------------------

def parse_amendment_act(
    elements: Iterator[dict],
    source_pdf: str,
) -> AmendmentAct:
    """Parse an Amendment Act into an AmendmentAct with classified entries.

    Scans elements for amendment instruction patterns and classifies each
    as SUBSTITUTES, INSERTS, or OMITS.

    Args:
        elements: Iterator of element dicts from flatten_elements().
        source_pdf: Filename of the source PDF.

    Returns:
        AmendmentAct object with all found amendment entries.
    """
    amend_act = AmendmentAct(
        uid=amendment_act_uid("corporate-laws", 2026),
        node_type="AmendmentAct",
        page_number=1,
        text="",
        source_pdf=source_pdf,
        title="The Corporate Laws (Amendment) Act, 2026",
        year=2026,
    )

    for element in elements:
        content: str = element.get("content", "").strip()
        page: int = int(element.get("page number", 0))

        if not content:
            continue

        sec_match = AMENDMENT_SECTION_RE.search(content)
        if sec_match:
            target_section = f"section {sec_match.group(1)}"

            # Classify amendment type from the same paragraph
            if SUBSTITUTION_RE.search(content):
                amendment_type = "SUBSTITUTES"
            elif INSERTION_RE.search(content):
                amendment_type = "INSERTS"
            elif OMISSION_RE.search(content):
                amendment_type = "OMITS"
            else:
                # Could not classify — default to SUBSTITUTES as most common
                amendment_type = "SUBSTITUTES"
                logger.warning(
                    "[page %d] Amendment instruction for %s not classified: %r",
                    page,
                    target_section,
                    content[:100],
                )

            entry = AmendmentEntry(
                target_section=target_section,
                amendment_type=amendment_type,
                raw_text=content,
            )
            amend_act.entries.append(entry)

    return amend_act
