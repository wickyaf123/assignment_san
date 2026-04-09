"""
Effective-state resolution for amended sections.

Hybrid approach (D-06): Cypher fetches the amendment chain ordered by
effective_date, Python applies transformations in sequence.

Lives in backend/graph/resolver.py (D-07).
On-demand computation at query time (D-08).
Returns structured result per AMEND-04 (D-09).

DATA CONTRACT: RESOLVE_QUERY reads r.old_text from Neo4j relationships.
This matches the property name set by loader.py _merge_amendment_rels(),
which maps AmendmentEntry.removed_text to r.old_text.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import neo4j

logger = logging.getLogger(__name__)

from graph.connection import DATABASE


@dataclass
class EffectiveState:
    """Structured result for effective-state resolution (D-09, AMEND-04).

    Attributes:
        current_text: The effective text after all amendments applied.
        status: "original" if no amendments, "amended" if substituted/inserted,
                "omitted" if text was removed.
        amendment_chain: Ordered list of amendment dicts that were applied.
        original_text: The pre-amendment text from the Section node.
    """

    current_text: str
    status: str  # "original" | "amended" | "omitted"
    amendment_chain: list[dict] = field(default_factory=list)
    original_text: str = ""


# Cypher query: fetches section + amendment chain ordered by effective_date ASC.
# OPTIONAL MATCH ensures unamended sections still return.
# Date filter: r.effective_date <= date($as_of) excludes prospective amendments (AMEND-03).
# Reads r.old_text (standardized property name, mapped from AmendmentEntry.removed_text by loader.py).
RESOLVE_QUERY = """
MATCH (s:Section {number: $section_number})
OPTIONAL MATCH (a:AmendmentAct)-[r]->(s)
WHERE type(r) IN ['SUBSTITUTES', 'INSERTS', 'OMITS', 'DECRIMINALIZES']
  AND r.effective_date <= date($as_of)
WITH s, a, r, type(r) AS rel_type
ORDER BY r.effective_date ASC
RETURN
    s.uid          AS uid,
    s.text         AS original_text,
    collect(
        CASE WHEN a IS NOT NULL THEN {
            amendment_uid: a.uid,
            amendment_type: rel_type,
            new_text: r.new_text,
            old_text: r.old_text,
            effective_date: toString(r.effective_date)
        } ELSE null END
    ) AS amendments
"""


def _apply_amendments(original_text: str, amendments: list[dict]) -> EffectiveState:
    """Apply amendment transformations in chronological order (AMEND-01, AMEND-02).

    Args:
        original_text: The original section text before any amendments.
        amendments: List of amendment dicts ordered by effective_date ASC.
            Each dict has: amendment_uid, amendment_type, new_text, old_text, effective_date.

    Returns:
        EffectiveState with current_text after all transformations applied.

    Note: Unknown amendment_types (e.g., DECRIMINALIZES) are logged and skipped
    without error. This is intentional — the loader passes through all amendment
    types from the source data, and the resolver only applies text transformations
    for known types.
    """
    if not amendments:
        return EffectiveState(
            current_text=original_text,
            status="original",
            amendment_chain=[],
            original_text=original_text,
        )

    current_text = original_text
    status = "amended"
    applied_chain: list[dict] = []

    for amendment in amendments:
        amendment_type = amendment.get("amendment_type", "")
        old_text = amendment.get("old_text", "")
        new_text = amendment.get("new_text", "")

        if amendment_type == "SUBSTITUTES":
            if old_text and old_text in current_text:
                current_text = current_text.replace(old_text, new_text)
            elif old_text:
                logger.warning(
                    "SUBSTITUTES old_text not found in current_text: "
                    "amendment=%s, old_text='%s...'",
                    amendment.get("amendment_uid"),
                    old_text[:50],
                )
        elif amendment_type == "INSERTS":
            current_text = current_text + new_text
        elif amendment_type == "OMITS":
            if old_text and old_text in current_text:
                current_text = current_text.replace(old_text, "")
                status = "omitted"   # only mark omitted if removal actually succeeded
            elif old_text:
                logger.warning(
                    "OMITS old_text not found in current_text: "
                    "amendment=%s, old_text='%s...'",
                    amendment.get("amendment_uid"),
                    old_text[:50],
                )
            # If old_text is empty or not found, status is not updated here
        else:
            logger.warning(
                "Unknown amendment_type: %s (amendment=%s) — skipping",
                amendment_type,
                amendment.get("amendment_uid"),
            )
            continue

        applied_chain.append(amendment)

    return EffectiveState(
        current_text=current_text,
        status=status,
        amendment_chain=applied_chain,
        original_text=original_text,
    )


def resolve_effective_state(
    driver: neo4j.Driver,
    section_number: int,
    as_of_date: str | None = None,
) -> EffectiveState | None:
    """Resolve the effective state of a section by traversing its amendment chain.

    Hybrid approach (D-06): Cypher fetches the chain, Python applies transforms.
    Prospective amendments excluded by date filter (AMEND-03).

    Args:
        driver: Neo4j driver instance.
        section_number: The section number to resolve (e.g., 135).
        as_of_date: ISO date string for "current" date. Defaults to today.
            Amendments with effective_date > as_of_date are excluded.

    Returns:
        EffectiveState if section exists, None if section not found.
    """
    from datetime import date as date_type

    if as_of_date is None:
        as_of_date = date_type.today().isoformat()

    with driver.session(database=DATABASE) as session:
        result = session.run(
            RESOLVE_QUERY,
            section_number=section_number,
            as_of=as_of_date,
        )
        record = result.single()

    if record is None:
        logger.warning("Section %d not found in graph", section_number)
        return None

    original_text = record["original_text"] or ""
    amendments_raw = record["amendments"] or []

    # Filter out null entries from OPTIONAL MATCH (unamended sections)
    amendments = [a for a in amendments_raw if a is not None]

    return _apply_amendments(original_text, amendments)
