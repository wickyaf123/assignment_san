"""
Tests for act_parser.py and cross_ref_detector.py.

Covers INGEST-02: structure parsing of Companies Act 2013 elements
and Amendment Act classification.
"""

from __future__ import annotations

import pytest

from ingestion.cross_ref_detector import extract_cross_references
from ingestion.act_parser import parse_companies_act, parse_amendment_act, ParserState


# ---------------------------------------------------------------------------
# Helper: build minimal OpenDataLoader-style element dicts
# ---------------------------------------------------------------------------

def _heading(content: str, page: int = 1) -> dict:
    return {"type": "heading", "content": content, "page number": page}


def _para(content: str, page: int = 1) -> dict:
    return {"type": "paragraph", "content": content, "page number": page}


# ---------------------------------------------------------------------------
# Cross-reference detection tests
# ---------------------------------------------------------------------------

class TestCrossReferenceDetection:
    def test_cross_ref_subject_to(self):
        refs = extract_cross_references("subject to section 149")
        assert len(refs) == 1
        assert refs[0].ref_type == "SUBJECT_TO"
        assert "section 149" in refs[0].target_pattern

    def test_cross_ref_notwithstanding(self):
        refs = extract_cross_references("notwithstanding anything in section 197")
        assert len(refs) == 1
        assert refs[0].ref_type == "NOTWITHSTANDING"

    def test_cross_ref_refers_to(self):
        refs = extract_cross_references("as referred to in section 2")
        assert len(refs) == 1
        assert refs[0].ref_type == "REFERS_TO"

    def test_cross_ref_none(self):
        refs = extract_cross_references("no cross references here")
        assert refs == []

    def test_cross_ref_multiple(self):
        refs = extract_cross_references("under section 135 and subject to section 149")
        assert len(refs) == 2
        types = {r.ref_type for r in refs}
        assert "REFERS_TO" in types
        assert "SUBJECT_TO" in types

    def test_cross_ref_empty_string(self):
        refs = extract_cross_references("")
        assert refs == []


# ---------------------------------------------------------------------------
# Act parsing tests
# ---------------------------------------------------------------------------

class TestParseCompaniesAct:
    def test_parse_section_from_elements(self):
        """Feed chapter + marginal note + section paragraph — expect correct Act structure."""
        elements = [
            _heading("CHAPTER I"),
            _heading("Preliminary"),
            _para("1. Short title, commencement and application.— (1) This Act may be called the Companies Act, 2013."),
        ]
        act = parse_companies_act(iter(elements), "test.pdf")
        assert len(act.chapters) == 1
        chapter = act.chapters[0]
        assert len(chapter.sections) == 1
        section = chapter.sections[0]
        assert section.number == 1
        # Title may come from marginal note heading OR inline text
        assert section.title is not None and len(section.title) > 0

    def test_parse_subsection_and_clause(self):
        """Feed section + subsection + clause — expect nested structure."""
        elements = [
            _heading("CHAPTER I"),
            _para("3. Formation of company.— (1) A company may be formed for any lawful purpose."),
            _para("(1) Such a company shall be—"),
            _para("(a) a company limited by shares; or"),
        ]
        act = parse_companies_act(iter(elements), "test.pdf")
        section = act.chapters[0].sections[0]
        assert section.number == 3
        assert len(section.subsections) >= 1
        ss = section.subsections[0]
        assert ss.number == 1
        assert len(ss.clauses) >= 1
        assert ss.clauses[0].label == "a"

    def test_parse_definition_extracts_term(self):
        """Definition in Section 2 must produce a Definition with term field."""
        elements = [
            _heading("CHAPTER I"),
            _heading("Definitions"),
            _para('2. Definitions.— (1) In this Act, unless the context otherwise requires—'),
            _para('"company" means a company incorporated under this Act or under any previous company law.'),
        ]
        act = parse_companies_act(iter(elements), "test.pdf")
        section = act.chapters[0].sections[0]
        # Definitions may be at section level
        all_defs = list(section.definitions)
        for ss in section.subsections:
            all_defs.extend(ss.definitions)
        assert len(all_defs) >= 1
        terms = [d.term for d in all_defs]
        assert "company" in terms

    def test_marginal_note_becomes_title(self):
        """Heading before section paragraph must become Section.title."""
        elements = [
            _heading("CHAPTER X"),
            _heading("Corporate Social Responsibility"),
            _para("135. Corporate social responsibility.— (1) Every company shall constitute a CSR committee."),
        ]
        act = parse_companies_act(iter(elements), "test.pdf")
        section = act.chapters[0].sections[0]
        assert section.number == 135
        assert "Corporate Social Responsibility" in section.title

    def test_section_number_is_int(self):
        """Section.number must be an integer, not a string."""
        elements = [
            _heading("CHAPTER I"),
            _para("10. Registered office of company.— Every company shall, on and from the fifteenth day of its incorporation..."),
        ]
        act = parse_companies_act(iter(elements), "test.pdf")
        section = act.chapters[0].sections[0]
        assert isinstance(section.number, int)
        assert section.number == 10

    def test_parse_proviso_attached_to_subsection(self):
        """Proviso after subsection must be child of subsection, not section."""
        elements = [
            _heading("CHAPTER I"),
            _para("5. Name of company.— A company shall have a name."),
            _para("(1) The name shall comply with requirements."),
            _para("Provided that nothing in this sub-section shall apply to a company formed before this Act."),
        ]
        act = parse_companies_act(iter(elements), "test.pdf")
        section = act.chapters[0].sections[0]
        ss = section.subsections[0]
        assert len(ss.provisos) == 1
        assert len(section.provisos) == 0

    def test_source_pdf_stored_on_act(self):
        """source_pdf must be passed through to the Act object."""
        elements = [
            _heading("CHAPTER I"),
            _para("1. Short title.— This Act may be called the Companies Act."),
        ]
        act = parse_companies_act(iter(elements), "Companies Act, 2013.pdf")
        assert act.source_pdf == "Companies Act, 2013.pdf"


# ---------------------------------------------------------------------------
# Amendment Act parsing tests
# ---------------------------------------------------------------------------

class TestParseAmendmentAct:
    def _make_amendment_elements(self, verb: str) -> list[dict]:
        return [
            _para(f"In the Companies Act, 2013, in section 135, the following {verb} shall be {verb}."),
        ]

    def test_parse_amendment_act_classifies_types(self):
        """Amendment instruction elements produce AmendmentAct with classified entries."""
        elements = [
            _para("In the Companies Act, 2013, in section 135, the words 'five crore rupees' shall be substituted."),
            _para("In the said Act, in section 2, a new clause (aa) shall be inserted."),
            _para("In the principal Act, in section 167, clause (a) shall be omitted."),
        ]
        amend = parse_amendment_act(iter(elements), "amend.pdf")
        assert len(amend.entries) >= 1

    def test_amendment_substitution(self):
        """'shall be substituted' text -> amendment_type == 'SUBSTITUTES'."""
        elements = [
            _para("In the Companies Act, 2013, in section 135, the following shall be substituted."),
        ]
        amend = parse_amendment_act(iter(elements), "amend.pdf")
        assert any(e.amendment_type == "SUBSTITUTES" for e in amend.entries)

    def test_amendment_insertion(self):
        """'shall be inserted' text -> amendment_type == 'INSERTS'."""
        elements = [
            _para("In the Companies Act, 2013, in section 2, the following shall be inserted."),
        ]
        amend = parse_amendment_act(iter(elements), "amend.pdf")
        assert any(e.amendment_type == "INSERTS" for e in amend.entries)

    def test_amendment_omission(self):
        """'shall be omitted' text -> amendment_type == 'OMITS'."""
        elements = [
            _para("In the principal Act, in section 167, clause (a) shall be omitted."),
        ]
        amend = parse_amendment_act(iter(elements), "amend.pdf")
        assert any(e.amendment_type == "OMITS" for e in amend.entries)

    def test_amendment_target_section_captured(self):
        """AmendmentEntry.target_section contains the section number from the instruction."""
        elements = [
            _para("In the Companies Act, 2013, in section 135, the following shall be substituted."),
        ]
        amend = parse_amendment_act(iter(elements), "amend.pdf")
        assert len(amend.entries) >= 1
        assert "135" in amend.entries[0].target_section

    def test_amendment_source_pdf(self):
        """source_pdf passed through to AmendmentAct."""
        elements = [
            _para("In the Companies Act, 2013, in section 1, the following shall be inserted."),
        ]
        amend = parse_amendment_act(iter(elements), "Amendment2026.pdf")
        assert amend.source_pdf == "Amendment2026.pdf"


# ---------------------------------------------------------------------------
# Integration test (slow — requires actual data files)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_parse_act_integration(companies_act_json_path):
    """Load actual Companies Act raw JSON, parse, assert structure."""
    import json
    from ingestion.extractor import flatten_elements

    with open(companies_act_json_path, encoding="utf-8") as fh:
        data = json.load(fh)

    elements = flatten_elements(data.get("kids", []))
    act = parse_companies_act(elements, "Companies Act, 2013.pdf")

    assert len(act.chapters) > 0
    all_sections = [s for ch in act.chapters for s in ch.sections]
    assert len(all_sections) > 0

    # Section 135 should exist and have CSR-related title
    sec_135 = next((s for s in all_sections if s.number == 135), None)
    assert sec_135 is not None
    assert "Corporate Social Responsibility" in sec_135.title or "Corporate" in sec_135.title
