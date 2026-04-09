"""
LLM-based amendment parser using Gemini structured output.

Falls back to the regex-based parse_amendment_act() on LLM failure.
"""
from __future__ import annotations

import logging
import os
from typing import Iterator, Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from ingestion.models import AmendmentAct, AmendmentEntry
from ingestion.uid_generator import amendment_act_uid

logger = logging.getLogger(__name__)


class AmendmentEntrySchema(BaseModel):
    target_section: str  # e.g., "section 135"
    amendment_type: Literal["SUBSTITUTES", "INSERTS", "OMITS", "DECRIMINALIZES"]
    new_text: str = ""
    removed_text: str = ""
    raw_text: str = ""


class AmendmentExtraction(BaseModel):
    entries: list[AmendmentEntrySchema]


_EXTRACTION_PROMPT = """\
You are analyzing the Corporate Laws (Amendment) Act, 2026 of India.

Extract ALL amendment entries from the following text. For each amendment:
- target_section: The section number being amended (e.g., "section 135")
- amendment_type: One of SUBSTITUTES, INSERTS, OMITS, or DECRIMINALIZES
  Use DECRIMINALIZES when a criminal penalty (imprisonment/fine) is replaced \
with a civil penalty, compoundable offence, or "penalty not exceeding" language.
- new_text: The new text being inserted or substituted (if applicable)
- removed_text: The text being removed or replaced (if applicable)
- raw_text: The original amendment instruction text

Text to analyze:
{text}
"""


def parse_amendment_with_llm(
    elements: Iterator[dict],
    source_pdf: str,
) -> AmendmentAct | None:
    """Use Gemini structured output to extract amendment entries.

    Args:
        elements: Iterator of element dicts from flatten_elements().
        source_pdf: Filename of the source PDF.

    Returns:
        AmendmentAct with extracted entries, or None on LLM failure.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set -- skipping LLM amendment parser")
        return None

    # Collect all text from elements
    full_text_parts = []
    for element in elements:
        content = element.get("content", "").strip()
        if content:
            full_text_parts.append(content)
    full_text = "\n".join(full_text_parts)

    if not full_text:
        logger.warning("No text found in amendment PDF elements")
        return None

    # Chunk if needed (Gemini has large context, but be safe)
    # For a single amendment act, full text should fit in one call
    try:
        llm = ChatGoogleGenerativeAI(
            model=os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
            temperature=0.0,
            api_key=api_key,
        )
        structured_llm = llm.with_structured_output(AmendmentExtraction)

        prompt = _EXTRACTION_PROMPT.format(text=full_text[:50000])  # safety truncation
        result: AmendmentExtraction = structured_llm.invoke(prompt)

        amend_act = AmendmentAct(
            uid=amendment_act_uid("corporate-laws", 2026),
            node_type="AmendmentAct",
            page_number=1,
            text="",
            source_pdf=source_pdf,
            title="The Corporate Laws (Amendment) Act, 2026",
            year=2026,
        )

        for entry_schema in result.entries:
            entry = AmendmentEntry(
                target_section=entry_schema.target_section,
                amendment_type=entry_schema.amendment_type,
                new_text=entry_schema.new_text,
                removed_text=entry_schema.removed_text,
                raw_text=entry_schema.raw_text,
            )
            amend_act.entries.append(entry)

        logger.info("LLM amendment parser extracted %d entries", len(amend_act.entries))
        return amend_act

    except Exception as exc:
        logger.warning("LLM amendment parser failed: %s -- falling back to regex", exc)
        return None
