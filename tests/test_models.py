"""
Tests for ingestion.models — Pydantic v2 model creation and serialization.
"""

from __future__ import annotations

import pytest

from ingestion.models import (
    Act,
    AmendmentAct,
    AmendmentEntry,
    Chapter,
    Clause,
    CrossReference,
    Definition,
    Explanation,
    Form,
    ParseReport,
    Proviso,
    Rule,
    RuleSet,
    Schedule,
    Section,
    SubClause,
    SubSection,
)
from ingestion.uid_generator import (
    clause_uid,
    definition_uid,
    explanation_uid,
    proviso_uid,
    section_uid,
    subsection_uid,
)


# ---------------------------------------------------------------------------
# Section model creation with nested hierarchy
# ---------------------------------------------------------------------------


def test_section_model_creation() -> None:
    """Section with nested SubSection → Clause → Proviso must round-trip via model_dump()."""
    sec_uid = section_uid(3)
    ss_uid = subsection_uid(3, 1)
    cl_uid = clause_uid(ss_uid, "a")
    prov_uid = proviso_uid(cl_uid, 1)

    proviso = Proviso(
        uid=prov_uid,
        node_type="Proviso",
        page_number=12,
        text="Provided that nothing in this sub-section shall apply to...",
        source_pdf="Companies Act, 2013.pdf",
        proviso_type="Provided that",
    )
    clause = Clause(
        uid=cl_uid,
        node_type="Clause",
        page_number=12,
        text="(a) a public company;",
        source_pdf="Companies Act, 2013.pdf",
        label="a",
        provisos=[proviso],
    )
    subsection = SubSection(
        uid=ss_uid,
        node_type="SubSection",
        page_number=12,
        text="(1) A company may be formed for any lawful purpose by—",
        source_pdf="Companies Act, 2013.pdf",
        number=1,
        clauses=[clause],
    )
    section = Section(
        uid=sec_uid,
        node_type="Section",
        page_number=12,
        text="3. Formation of company.",
        source_pdf="Companies Act, 2013.pdf",
        number=3,
        title="Formation of company",
        subsections=[subsection],
    )

    dumped = section.model_dump()
    assert dumped["uid"] == "sec-3"
    assert dumped["number"] == 3
    assert dumped["title"] == "Formation of company"
    assert len(dumped["subsections"]) == 1
    assert dumped["subsections"][0]["uid"] == "sec-3-ss-1"
    assert len(dumped["subsections"][0]["clauses"]) == 1
    assert dumped["subsections"][0]["clauses"][0]["uid"] == "sec-3-ss-1-cl-a"
    assert len(dumped["subsections"][0]["clauses"][0]["provisos"]) == 1
    assert dumped["subsections"][0]["clauses"][0]["provisos"][0]["proviso_type"] == "Provided that"


# ---------------------------------------------------------------------------
# Definition model
# ---------------------------------------------------------------------------


def test_definition_has_term_field() -> None:
    """Definition model must carry the `term` field distinct from `text`."""
    uid = definition_uid("company")
    defn = Definition(
        uid=uid,
        node_type="Definition",
        page_number=1,
        text='"company" means a company incorporated under this Act...',
        source_pdf="Companies Act, 2013.pdf",
        term="company",
    )
    assert defn.term == "company"
    assert defn.uid == "def-company"
    assert defn.node_type == "Definition"


# ---------------------------------------------------------------------------
# CrossReference model
# ---------------------------------------------------------------------------


def test_cross_reference_model() -> None:
    """CrossReference must store all fields correctly."""
    ref = CrossReference(
        target_pattern="section 149",
        ref_type="SUBJECT_TO",
        raw_text="subject to section 149",
    )
    assert ref.target_pattern == "section 149"
    assert ref.ref_type == "SUBJECT_TO"
    assert ref.raw_text == "subject to section 149"


# ---------------------------------------------------------------------------
# AmendmentEntry types
# ---------------------------------------------------------------------------


def test_amendment_entry_types() -> None:
    """AmendmentEntry must support all three amendment_type values with round-trip."""
    entries = [
        AmendmentEntry(
            target_section="section 135",
            amendment_type="SUBSTITUTES",
            new_text="(1) Every company having net worth...",
            removed_text="(1) Every company having net worth of rupees five hundred crore...",
            raw_text="In section 135, sub-section (1) shall be substituted...",
        ),
        AmendmentEntry(
            target_section="section 149",
            amendment_type="INSERTS",
            new_text="(1A) Every listed company shall have...",
            raw_text="In section 149, after sub-section (1), the following sub-section shall be inserted...",
        ),
        AmendmentEntry(
            target_section="section 197",
            amendment_type="OMITS",
            removed_text="the proviso to sub-section (3)",
            raw_text="In section 197, the proviso to sub-section (3) shall be omitted.",
        ),
    ]

    for entry in entries:
        dumped = entry.model_dump()
        assert dumped["amendment_type"] in ("SUBSTITUTES", "INSERTS", "OMITS")
        # Verify round-trip reconstruction
        restored = AmendmentEntry(**dumped)
        assert restored.amendment_type == entry.amendment_type
        assert restored.target_section == entry.target_section


# ---------------------------------------------------------------------------
# ParseReport fields
# ---------------------------------------------------------------------------


def test_parse_report_fields() -> None:
    """ParseReport must have all required fields; coverage_percent must be float."""
    report = ParseReport(
        source_pdf="Companies Act, 2013.pdf",
        total_sections=470,
        total_subsections=1200,
        total_clauses=850,
        total_provisos=430,
        total_explanations=95,
        total_definitions=67,
        warnings=["Page 12: could not parse sub-section numbering"],
        coverage_percent=97.5,
    )
    assert isinstance(report.coverage_percent, float)
    assert report.coverage_percent == 97.5
    assert report.total_sections == 470
    assert report.source_pdf == "Companies Act, 2013.pdf"
    assert len(report.warnings) == 1


# ---------------------------------------------------------------------------
# Additional edge-case model tests
# ---------------------------------------------------------------------------


def test_section_with_no_subsections_has_direct_clauses() -> None:
    """Sections can have direct clauses (no subsections) — model must allow it."""
    sec = Section(
        uid="sec-2",
        node_type="Section",
        page_number=1,
        text="2. Definitions.",
        source_pdf="Companies Act, 2013.pdf",
        number=2,
        title="Definitions",
        clauses=[
            Clause(
                uid="sec-2-cl-a",
                node_type="Clause",
                page_number=1,
                text='(a) "abridged prospectus" means...',
                source_pdf="Companies Act, 2013.pdf",
                label="a",
            )
        ],
    )
    assert len(sec.subsections) == 0
    assert len(sec.clauses) == 1


def test_proviso_types() -> None:
    """Proviso must store the exact proviso_type string."""
    for ptype in ("Provided that", "Provided also that", "Provided further that"):
        p = Proviso(
            uid=f"sec-1-proviso-1",
            node_type="Proviso",
            page_number=1,
            text=f"{ptype} ...",
            source_pdf="test.pdf",
            proviso_type=ptype,
        )
        assert p.proviso_type == ptype


def test_amendment_act_serialization() -> None:
    """AmendmentAct with entries must serialize to JSON and back."""
    act = AmendmentAct(
        uid="amend-corporate-laws-amendment-2026",
        node_type="AmendmentAct",
        page_number=1,
        text="Corporate Laws (Amendment) Act, 2026",
        source_pdf="Corporate Laws (Amendment) Act, 2026.pdf",
        title="Corporate Laws (Amendment) Act, 2026",
        year=2026,
        entries=[
            AmendmentEntry(
                target_section="section 2",
                amendment_type="INSERTS",
                new_text="New clause text",
                raw_text="Raw amendment text",
            )
        ],
    )
    json_str = act.model_dump_json()
    assert "AmendmentAct" in json_str
    assert "INSERTS" in json_str
