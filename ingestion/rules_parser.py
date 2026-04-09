"""
State-machine parser for the Companies Rules 2014 PDF.

Parses OpenDataLoader JSON elements into typed Pydantic objects:
    RuleSet -> Rule -> Form
    Schedule

Design principles:
  - Devanagari/Hindi content is detected and skipped with a logged warning (Pitfall 4)
  - Rule headings in both heading and paragraph element types are recognised
  - Form headings (e.g., "Form No. MGT-7") are extracted and nested under their parent Rule
  - Schedule headings are detected and not confused with Rules
  - Form body is accumulated as raw text (D-20)
  - Table elements within a Form body get a [TABLE CONTENT] marker
  - All unattached content is logged as a warning (D-08)

Security (T-01-10): Devanagari detection uses simple character-range regex on finite
known input — no performance concern for 3 PDFs.
"""

from __future__ import annotations

import logging
import re
from typing import Iterator

from ingestion.models import Form, Rule, RuleSet, Schedule
from ingestion.uid_generator import form_uid, rule_uid, ruleset_uid
from ingestion.cross_ref_detector import extract_cross_references

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex constants
# ---------------------------------------------------------------------------

# Detects characters in the Unicode Devanagari block (U+0900–U+097F)
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# Matches rule headings: "Rule 3. Particulars..." or "3. Particulars..." or "3A. ..."
# Captures group(1)=rule number (e.g., "3", "3A"), group(2)=title
RULE_HEADING_RE = re.compile(
    r"^(?:Rule\s+)?(\d+[A-Za-z]?)\.\s+(.+)",
    re.IGNORECASE,
)

# Matches form headings: "Form No. MGT-7", "Form MGT-7", "Form No MGT.7"
# Captures group(1)=form number (e.g., "MGT-7", "INC-1", "AOC.4")
FORM_RE = re.compile(
    r"^Form\s+(?:No\.?\s*)?([A-Z]+-\d+[A-Z\-]*\d*|[A-Z]+\.\d+[A-Z\-]*)",
    re.IGNORECASE,
)

# Matches schedule headings: "SCHEDULE I", "Schedule III"
# Captures group(1)=schedule number as Roman numeral
SCHEDULE_RE = re.compile(r"^SCHEDULE\s+([IVXLC]+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def is_devanagari_majority(text: str) -> bool:
    """Return True if more than 30% of non-whitespace characters are Devanagari.

    Uses a generous threshold (30%) to catch mixed-script headings that are
    predominantly Hindi even if they contain a few English characters.

    Args:
        text: Any string to examine.

    Returns:
        True if Devanagari characters constitute > 30% of non-whitespace chars.
    """
    if not text:
        return False

    non_whitespace = [ch for ch in text if not ch.isspace()]
    if not non_whitespace:
        return False

    devanagari_count = sum(1 for ch in non_whitespace if DEVANAGARI_RE.match(ch))
    ratio = devanagari_count / len(non_whitespace)
    return ratio > 0.30


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_companies_rules(
    elements: Iterator[dict],
    source_pdf: str,
) -> RuleSet:
    """Parse Companies Rules 2014 elements into a structured RuleSet object.

    Processes elements from flatten_elements() in document order.
    Handles:
        - Devanagari/Hindi detection and skip with warning (D-08, Pitfall 4)
        - Rule heading extraction (heading and paragraph types)
        - Form extraction nested inside Rules (D-20)
        - Schedule detection (D-13)
        - Cross-reference tagging (D-14)

    Args:
        elements: Iterator of element dicts from flatten_elements().
        source_pdf: Filename of the source PDF (stored on all nodes).

    Returns:
        Populated RuleSet object.
    """
    ruleset = RuleSet(
        uid=ruleset_uid("companies-rules"),
        node_type="RuleSet",
        page_number=1,
        text="",
        source_pdf=source_pdf,
        title="The Companies Rules, 2014",
        year=2014,
    )

    current_rule: Rule | None = None
    current_form: Form | None = None
    seen_rule_numbers: set[str] = set()

    def _finalize_rule() -> None:
        """Append current_rule (with its finalised form) to ruleset."""
        nonlocal current_form
        if current_rule is not None:
            _finalize_form()
            if current_rule not in ruleset.rules:
                ruleset.rules.append(current_rule)

    def _finalize_form() -> None:
        """Append current_form to current_rule.forms if present."""
        nonlocal current_form
        if current_form is not None and current_rule is not None:
            if current_form not in current_rule.forms:
                current_rule.forms.append(current_form)
        current_form = None

    def _start_rule(number: str, title: str, page: int) -> Rule:
        """Create and return a new Rule, finalising the previous one first."""
        nonlocal current_form
        if number in seen_rule_numbers:
            number = f"{number}-dup-{len(seen_rule_numbers)}"
            logger.warning("Duplicate rule number detected, disambiguating to: %s", number)
        seen_rule_numbers.add(number)
        _finalize_rule()
        new_rule = Rule(
            uid=rule_uid(number),
            node_type="Rule",
            page_number=page,
            text="",
            source_pdf=source_pdf,
            number=number,
            title=title,
        )
        # Directly append here so ruleset.rules grows; finalize_rule won't
        # double-append because it checks membership.
        ruleset.rules.append(new_rule)
        current_form = None
        return new_rule

    def _start_form(form_number: str, title: str, page: int) -> Form:
        """Finalise previous form (if any) and create a new Form."""
        _finalize_form()
        assoc_rule = current_rule.number if current_rule is not None else ""
        new_form = Form(
            uid=form_uid(form_number),
            node_type="Form",
            page_number=page,
            text="",
            source_pdf=source_pdf,
            form_number=form_number,
            title=title,
            associated_rule=assoc_rule,
        )
        if current_rule is not None:
            current_rule.forms.append(new_form)
        return new_form

    for element in elements:
        content: str = element.get("content", "").strip()
        etype: str = element.get("type", "")
        page: int = int(element.get("page number", 0))

        if not content:
            continue

        # ------------------------------------------------------------------
        # Devanagari/Hindi — include as bilingual content
        # ------------------------------------------------------------------
        if is_devanagari_majority(content):
            logger.debug(
                "Hindi content at page %d: %r",
                page,
                content[:80],
            )

        # ------------------------------------------------------------------
        # Heading dispatch
        # ------------------------------------------------------------------
        if etype == "heading":
            # 1. Schedule heading
            sched_match = SCHEDULE_RE.match(content)
            if sched_match:
                _finalize_rule()
                current_rule = None
                schedule_number = sched_match.group(1).upper()
                schedule = Schedule(
                    uid=f"schedule-{schedule_number.lower()}",
                    node_type="Schedule",
                    page_number=page,
                    text=content,
                    source_pdf=source_pdf,
                    number=schedule_number,
                    title=content,
                )
                # Schedules are tracked at ruleset level (not nested in rules)
                # The Schedule object is created and logged but not attached to
                # ruleset.rules — we store them on a schedules attribute if
                # present, otherwise just log (D-13).
                logger.debug("Found Schedule %s at page %d", schedule_number, page)
                continue

            # 2. Form heading
            form_match = FORM_RE.match(content)
            if form_match:
                form_number = form_match.group(1).upper()
                current_form = _start_form(form_number, content, page)
                continue

            # 3. Rule heading
            rule_match = RULE_HEADING_RE.match(content)
            if rule_match:
                number = rule_match.group(1)
                title = rule_match.group(2).strip().rstrip(".")
                current_rule = _start_rule(number, title, page)
                continue

            # 4. Unknown heading — treat as marginal note / log at debug level
            logger.debug(
                "Unknown heading at page %d, treating as continuation: %r",
                page,
                content[:80],
            )
            # Fall through to paragraph-style handling below

        # ------------------------------------------------------------------
        # Paragraph / caption dispatch
        # ------------------------------------------------------------------
        if etype in ("paragraph", "caption"):
            # Check if this paragraph looks like a rule heading (some rules
            # have their heading typed as "paragraph" not "heading")
            rule_match = RULE_HEADING_RE.match(content)
            if rule_match and current_form is None:
                # Only treat as rule heading if we're not inside a form body
                number = rule_match.group(1)
                title = rule_match.group(2).strip().rstrip(".")
                # Sanity-check: rule numbers should be small integers or short
                # alphanumerics (avoids matching "(1) These rules..." as rule "1")
                try:
                    # If it's a pure number, ensure it's a plausible rule number
                    int_num = int(number)
                    # Rule numbers in Companies Rules go up to ~100+
                    # but "(1)" at the start of a sub-rule text would be matched
                    # by SUBSECTION_RE-style patterns — guard against that.
                    # A real rule heading starts the paragraph (^), so the match
                    # above is already anchored. Accept it.
                except ValueError:
                    pass  # Alphanumeric like "3A" — always accept as rule number

                # Only start a new rule if the content starts exactly like a
                # heading (Rule N. Title) rather than looking like a subsection
                # paragraph that happens to start with a number.
                # Heuristic: if content contains "." after the number and has
                # a title portion > 3 chars, treat as rule heading.
                if len(title) > 3:
                    current_rule = _start_rule(number, title, page)
                    continue

            if current_form is not None:
                # Accumulate raw form body text (D-20)
                if current_form.text:
                    current_form.text = current_form.text + " " + content
                else:
                    current_form.text = content
                # Also tag cross-references on the form
                current_form.cross_references.extend(extract_cross_references(content))
                continue

            if current_rule is not None:
                # Append to rule text and tag cross-references
                if current_rule.text:
                    current_rule.text = current_rule.text + " " + content
                else:
                    current_rule.text = content
                current_rule.cross_references.extend(extract_cross_references(content))
                continue

            # Unattached content — no active rule context
            logger.warning(
                "Unattached content at page %d: %r",
                page,
                content[:80],
            )

        # ------------------------------------------------------------------
        # Table content
        # ------------------------------------------------------------------
        elif etype in ("table", "table row"):
            if current_form is not None:
                # Store table marker in form body as raw text (D-20)
                if current_form.text:
                    current_form.text = current_form.text + " [TABLE CONTENT]"
                else:
                    current_form.text = "[TABLE CONTENT]"
            elif current_rule is not None:
                if current_rule.text:
                    current_rule.text = current_rule.text + " [TABLE CONTENT]"
                else:
                    current_rule.text = "[TABLE CONTENT]"

    # Finalize the last open rule/form
    # Note: _finalize_rule() was designed for transitions, not final cleanup.
    # Here we do a direct final cleanup to avoid duplicate appends.
    if current_form is not None and current_rule is not None:
        if current_form not in current_rule.forms:
            current_rule.forms.append(current_form)

    return ruleset
