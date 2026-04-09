"""
Tests for INGEST-03: Proviso parent attribution.

Verifies that the act parser correctly:
  - Assigns single provisos to the deepest non-proviso context
  - Treats chained provisos as siblings (not nested)
  - Assigns proviso-after-clause to the clause (not its parent subsection)
  - Generates sequential UIDs for provisos
  - Correctly classifies all three proviso type strings

See: .planning/phases/01-ingestion-pipeline/01-RESEARCH.md (Pitfall 3)
"""

from __future__ import annotations

import pytest

from ingestion.act_parser import parse_companies_act


# ---------------------------------------------------------------------------
# Helper: build minimal OpenDataLoader-style element dicts
# ---------------------------------------------------------------------------

def make_elements(*specs: tuple[str, str]) -> list[dict]:
    """Construct OpenDataLoader-style element dicts from shorthand specs.

    Each spec is a (element_type, content) tuple, e.g.:
        ("heading", "CHAPTER I")
        ("paragraph", "(1) First subsection...")
        ("paragraph", "Provided that nothing shall...")

    Returns:
        List of element dicts ready for parse_companies_act().
    """
    elements = []
    for i, (etype, content) in enumerate(specs, start=1):
        elements.append({
            "type": etype,
            "content": content,
            "page number": i,
        })
    return elements


# ---------------------------------------------------------------------------
# Proviso parent attribution tests
# ---------------------------------------------------------------------------

class TestProvisoParentAttribution:
    def test_single_proviso_parent_is_subsection(self):
        """Single Provided-that after a SubSection must parent to the SubSection."""
        elements = make_elements(
            ("heading", "CHAPTER I"),
            ("paragraph", "5. Name of company.— Provisions about names."),
            ("paragraph", "(1) The name shall comply with requirements of this section."),
            ("paragraph", "Provided that nothing in this sub-section shall apply to a company formed before this Act."),
        )
        act = parse_companies_act(iter(elements), "test.pdf")
        section = act.chapters[0].sections[0]
        ss = section.subsections[0]

        # Proviso must be on the subsection, not the section
        assert len(ss.provisos) == 1, f"Expected 1 proviso on subsection, got {len(ss.provisos)}"
        assert len(section.provisos) == 0, f"Expected 0 provisos on section, got {len(section.provisos)}"

    def test_chained_provisos_are_siblings(self):
        """Three chained provisos after SubSection must be 3 siblings on that subsection."""
        elements = make_elements(
            ("heading", "CHAPTER I"),
            ("paragraph", "10. Registered office.— Every company shall have a registered office."),
            ("paragraph", "(1) The registered office shall be notified within thirty days of incorporation."),
            ("paragraph", "Provided that a company may apply for extension of time."),
            ("paragraph", "Provided also that the Board may grant additional time."),
            ("paragraph", "Provided further that where no extension is granted the default fee shall apply."),
        )
        act = parse_companies_act(iter(elements), "test.pdf")
        section = act.chapters[0].sections[0]
        ss = section.subsections[0]

        assert len(ss.provisos) == 3, f"Expected 3 provisos, got {len(ss.provisos)}"
        assert ss.provisos[0].proviso_type == "Provided that"
        assert ss.provisos[1].proviso_type == "Provided also that"
        assert ss.provisos[2].proviso_type == "Provided further that"

        # All three must share the same parent UID prefix (subsection UID)
        for p in ss.provisos:
            assert p.uid.startswith(ss.uid), (
                f"Proviso UID {p.uid!r} does not start with subsection UID {ss.uid!r}"
            )

    def test_provisos_not_nested(self):
        """Chained provisos must NOT be nested — all belong to the same parent list."""
        elements = make_elements(
            ("heading", "CHAPTER I"),
            ("paragraph", "15. Alteration of articles.— A company may alter its articles."),
            ("paragraph", "(1) Subject to the provisions of this Act, articles may be altered."),
            ("paragraph", "Provided that no alteration shall have retrospective effect."),
            ("paragraph", "Provided also that the Tribunal may modify the alteration."),
        )
        act = parse_companies_act(iter(elements), "test.pdf")
        section = act.chapters[0].sections[0]
        ss = section.subsections[0]

        # Both provisos are siblings on the subsection
        assert len(ss.provisos) == 2

        # Verify structurally: Proviso model has no `provisos` field at all,
        # so nesting is impossible. But also check they both come from the same list.
        assert ss.provisos[0].uid != ss.provisos[1].uid, "Provisos must have distinct UIDs"

        # They share the same parent UID prefix (not nested under each other)
        assert ss.provisos[0].uid.startswith(ss.uid)
        assert ss.provisos[1].uid.startswith(ss.uid)

        # proviso[1] must NOT start with proviso[0].uid (would indicate nesting)
        assert not ss.provisos[1].uid.startswith(ss.provisos[0].uid), (
            "Proviso 2 UID should not start with Proviso 1 UID — that would indicate nesting"
        )

    def test_proviso_after_clause_parents_to_clause(self):
        """Proviso after a Clause must parent to that Clause, not the SubSection."""
        elements = make_elements(
            ("heading", "CHAPTER I"),
            ("paragraph", "20. Forms of business.— Forms allowed for company formation."),
            ("paragraph", "(1) A company may be incorporated in the following forms—"),
            ("paragraph", "(a) a company limited by shares; or"),
            ("paragraph", "Provided that the Registrar may impose additional requirements."),
        )
        act = parse_companies_act(iter(elements), "test.pdf")
        section = act.chapters[0].sections[0]
        ss = section.subsections[0]
        clause = ss.clauses[0]

        # Proviso must be on the clause
        assert len(clause.provisos) == 1, f"Expected 1 proviso on clause, got {len(clause.provisos)}"
        assert len(ss.provisos) == 0, f"Expected 0 provisos on subsection, got {len(ss.provisos)}"
        assert len(section.provisos) == 0, f"Expected 0 provisos on section, got {len(section.provisos)}"

        # Proviso UID must contain the clause UID as prefix
        assert clause.provisos[0].uid.startswith(clause.uid), (
            f"Proviso UID {clause.provisos[0].uid!r} does not start with clause UID {clause.uid!r}"
        )

    def test_proviso_uids_are_sequential(self):
        """Proviso UIDs must follow the pattern '{parent_uid}-proviso-1', '-proviso-2', etc."""
        elements = make_elements(
            ("heading", "CHAPTER I"),
            ("paragraph", "25. Annual return.— Every company shall file an annual return."),
            ("paragraph", "(1) The annual return shall contain the particulars specified."),
            ("paragraph", "Provided that a small company may file an abridged return."),
            ("paragraph", "Provided also that a dormant company shall be exempt."),
        )
        act = parse_companies_act(iter(elements), "test.pdf")
        section = act.chapters[0].sections[0]
        ss = section.subsections[0]

        assert len(ss.provisos) == 2
        assert ss.provisos[0].uid.endswith("-proviso-1"), (
            f"First proviso UID should end with '-proviso-1', got: {ss.provisos[0].uid!r}"
        )
        assert ss.provisos[1].uid.endswith("-proviso-2"), (
            f"Second proviso UID should end with '-proviso-2', got: {ss.provisos[1].uid!r}"
        )

    def test_proviso_type_classification(self):
        """All three proviso type strings must be classified correctly."""
        # "Provided that" -> "Provided that"
        elements_1 = make_elements(
            ("heading", "CHAPTER I"),
            ("paragraph", "30. Share capital.— Every company shall have share capital."),
            ("paragraph", "Provided that nothing applies where the company is unlimited."),
        )
        act1 = parse_companies_act(iter(elements_1), "test.pdf")
        section1 = act1.chapters[0].sections[0]
        assert len(section1.provisos) == 1
        assert section1.provisos[0].proviso_type == "Provided that"

        # "Provided also that" -> "Provided also that"
        elements_2 = make_elements(
            ("heading", "CHAPTER I"),
            ("paragraph", "31. Alteration of share capital.— A company may alter its share capital."),
            ("paragraph", "Provided also that the board shall pass a resolution."),
        )
        act2 = parse_companies_act(iter(elements_2), "test.pdf")
        section2 = act2.chapters[0].sections[0]
        assert len(section2.provisos) == 1
        assert section2.provisos[0].proviso_type == "Provided also that"

        # "Provided further that" -> "Provided further that"
        elements_3 = make_elements(
            ("heading", "CHAPTER I"),
            ("paragraph", "32. Prohibition on issue.— No company shall issue shares at a discount."),
            ("paragraph", "Provided further that where shares are issued at a premium the excess shall be transferred."),
        )
        act3 = parse_companies_act(iter(elements_3), "test.pdf")
        section3 = act3.chapters[0].sections[0]
        assert len(section3.provisos) == 1
        assert section3.provisos[0].proviso_type == "Provided further that"
