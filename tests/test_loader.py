"""
Unit tests for loader.py pure collection functions.

These tests do NOT require a live Neo4j connection — they test the
data-preparation functions (_collect_act_nodes, _collect_structural_rels,
_collect_amendment_rels, _collect_ruleset_nodes) in isolation.

RED phase: these tests will fail with ImportError until Task 3 implements
backend/graph/loader.py.
"""

from __future__ import annotations

import logging

import pytest

from ingestion.models import (
    Act,
    AmendmentAct,
    AmendmentEntry,
    Chapter,
    Clause,
    Definition,
    Explanation,
    Form,
    Proviso,
    Rule,
    RuleSet,
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
    form_uid,
    proviso_uid,
    rule_uid,
    ruleset_uid,
    section_uid,
    subclause_uid,
    subsection_uid,
)

# Import the functions under test — will raise ImportError until Task 3.
from graph.loader import (
    _collect_act_nodes,
    _collect_amendment_rels,
    _collect_ruleset_nodes,
    _collect_structural_rels,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_act() -> Act:
    """Act with 1 Chapter, 1 Section, 1 SubSection — minimal hierarchy."""
    subsection = SubSection(
        uid=subsection_uid(135, 1),
        node_type="SubSection",
        page_number=10,
        text="(1) Every company having...",
        source_pdf="Companies Act, 2013.pdf",
        number=1,
    )
    section = Section(
        uid=section_uid(135),
        node_type="Section",
        page_number=10,
        text="135. Corporate Social Responsibility.",
        source_pdf="Companies Act, 2013.pdf",
        number=135,
        title="Corporate Social Responsibility",
        subsections=[subsection],
    )
    chapter = Chapter(
        uid=chapter_uid("IX"),
        node_type="Chapter",
        page_number=9,
        text="CHAPTER IX",
        source_pdf="Companies Act, 2013.pdf",
        number="IX",
        title="ACCOUNTS OF COMPANIES",
        sections=[section],
    )
    return Act(
        uid=act_uid("Companies", 2013),
        node_type="Act",
        page_number=1,
        text="The Companies Act, 2013.",
        source_pdf="Companies Act, 2013.pdf",
        title="The Companies Act, 2013",
        year=2013,
        chapters=[chapter],
    )


@pytest.fixture
def nested_act() -> Act:
    """Act with a Section containing Clause -> SubClause, Proviso, Explanation, Definition."""
    subclause = SubClause(
        uid=subclause_uid(clause_uid(subsection_uid(2, 1), "a"), "i"),
        node_type="SubClause",
        page_number=2,
        text="(i) any body corporate;",
        source_pdf="Companies Act, 2013.pdf",
        label="i",
    )
    clause = Clause(
        uid=clause_uid(subsection_uid(2, 1), "a"),
        node_type="Clause",
        page_number=2,
        text="(a) a body corporate;",
        source_pdf="Companies Act, 2013.pdf",
        label="a",
        subclauses=[subclause],
    )
    proviso = Proviso(
        uid=proviso_uid(section_uid(2), 1),
        node_type="Proviso",
        page_number=2,
        text="Provided that nothing in this section...",
        source_pdf="Companies Act, 2013.pdf",
        proviso_type="Provided that",
    )
    explanation = Explanation(
        uid=explanation_uid(section_uid(2), 1),
        node_type="Explanation",
        page_number=2,
        text="Explanation.— For the purposes of this section...",
        source_pdf="Companies Act, 2013.pdf",
        label="",
    )
    definition = Definition(
        uid=definition_uid("associate company"),
        node_type="Definition",
        page_number=2,
        text='"associate company" means...',
        source_pdf="Companies Act, 2013.pdf",
        term="associate company",
    )
    subsection = SubSection(
        uid=subsection_uid(2, 1),
        node_type="SubSection",
        page_number=2,
        text="(1) In this Act...",
        source_pdf="Companies Act, 2013.pdf",
        number=1,
        clauses=[clause],
    )
    section = Section(
        uid=section_uid(2),
        node_type="Section",
        page_number=2,
        text="2. Definitions.",
        source_pdf="Companies Act, 2013.pdf",
        number=2,
        title="Definitions",
        subsections=[subsection],
        provisos=[proviso],
        explanations=[explanation],
        definitions=[definition],
    )
    chapter = Chapter(
        uid=chapter_uid("I"),
        node_type="Chapter",
        page_number=1,
        text="CHAPTER I",
        source_pdf="Companies Act, 2013.pdf",
        number="I",
        title="PRELIMINARY",
        sections=[section],
    )
    return Act(
        uid=act_uid("Companies", 2013),
        node_type="Act",
        page_number=1,
        text="The Companies Act, 2013.",
        source_pdf="Companies Act, 2013.pdf",
        title="The Companies Act, 2013",
        year=2013,
        chapters=[chapter],
    )


@pytest.fixture
def simple_amendment_act() -> AmendmentAct:
    """AmendmentAct with one SUBSTITUTES entry targeting section 135."""
    return AmendmentAct(
        uid=amendment_act_uid("Corporate Laws Amendment", 2026),
        node_type="AmendmentAct",
        page_number=1,
        text="Corporate Laws (Amendment) Act, 2026.",
        source_pdf="Corporate Laws (Amendment) Act, 2026.pdf",
        title="Corporate Laws (Amendment) Act, 2026",
        year=2026,
        entries=[
            AmendmentEntry(
                target_section="section 135",
                amendment_type="SUBSTITUTES",
                new_text="new text here",
                removed_text="old text here",
                raw_text="In section 135, for the words...",
            )
        ],
    )


@pytest.fixture
def simple_ruleset() -> RuleSet:
    """RuleSet with 1 Rule and 1 Form."""
    form = Form(
        uid=form_uid("INC-1"),
        node_type="Form",
        page_number=5,
        text="Form INC-1",
        source_pdf="Companies Rules, 2014.pdf",
        form_number="INC-1",
        title="Application for reservation of name",
        associated_rule="9",
    )
    rule = Rule(
        uid=rule_uid("9"),
        node_type="Rule",
        page_number=5,
        text="9. Reservation of name.",
        source_pdf="Companies Rules, 2014.pdf",
        number="9",
        title="Reservation of name",
        forms=[form],
    )
    return RuleSet(
        uid=ruleset_uid("Companies Incorporation Rules 2014"),
        node_type="RuleSet",
        page_number=1,
        text="Companies (Incorporation) Rules, 2014.",
        source_pdf="Companies Rules, 2014.pdf",
        title="Companies (Incorporation) Rules, 2014",
        year=2014,
        rules=[rule],
    )


# ---------------------------------------------------------------------------
# Tests for _collect_act_nodes
# ---------------------------------------------------------------------------


def test_collect_act_nodes_basic(minimal_act):
    """_collect_act_nodes returns dict with Act, Chapter, Section, SubSection keys."""
    result = _collect_act_nodes(minimal_act)

    assert "Act" in result, "Missing 'Act' key"
    assert "Chapter" in result, "Missing 'Chapter' key"
    assert "Section" in result, "Missing 'Section' key"
    assert "SubSection" in result, "Missing 'SubSection' key"

    assert len(result["Act"]) == 1
    assert len(result["Chapter"]) == 1
    assert len(result["Section"]) == 1
    assert len(result["SubSection"]) == 1

    # Each dict must contain the required base properties
    for node_type, nodes in result.items():
        for node in nodes:
            assert "uid" in node, f"{node_type} node missing 'uid'"
            assert "text" in node, f"{node_type} node missing 'text'"
            assert "page_number" in node, f"{node_type} node missing 'page_number'"
            assert "source_pdf" in node, f"{node_type} node missing 'source_pdf'"


def test_collect_act_nodes_nested(nested_act):
    """_collect_act_nodes collects Clause, SubClause, Proviso, Explanation, Definition."""
    result = _collect_act_nodes(nested_act)

    assert "Clause" in result, "Missing 'Clause' key"
    assert "SubClause" in result, "Missing 'SubClause' key"
    assert "Proviso" in result, "Missing 'Proviso' key"
    assert "Explanation" in result, "Missing 'Explanation' key"
    assert "Definition" in result, "Missing 'Definition' key"

    assert len(result["Clause"]) >= 1
    assert len(result["SubClause"]) >= 1
    assert len(result["Proviso"]) >= 1
    assert len(result["Explanation"]) >= 1
    assert len(result["Definition"]) >= 1


# ---------------------------------------------------------------------------
# Tests for _collect_structural_rels
# ---------------------------------------------------------------------------


def test_collect_structural_rels(minimal_act):
    """_collect_structural_rels returns rels with parent_uid, child_uid, rel_type."""
    result = _collect_structural_rels(minimal_act)

    assert isinstance(result, list)
    assert len(result) > 0

    for rel in result:
        assert "parent_uid" in rel, "Rel missing 'parent_uid'"
        assert "child_uid" in rel, "Rel missing 'child_uid'"
        assert "rel_type" in rel, "Rel missing 'rel_type'"

    rel_types = {r["rel_type"] for r in result}
    assert "HAS_CHAPTER" in rel_types, "Missing HAS_CHAPTER relationship"
    assert "HAS_SECTION" in rel_types, "Missing HAS_SECTION relationship"

    # Verify Act->Chapter direction
    has_chapter = [r for r in result if r["rel_type"] == "HAS_CHAPTER"]
    assert has_chapter[0]["parent_uid"] == minimal_act.uid
    assert has_chapter[0]["child_uid"] == minimal_act.chapters[0].uid


# ---------------------------------------------------------------------------
# Tests for _collect_amendment_rels
# ---------------------------------------------------------------------------


def test_collect_amendment_rels_basic(simple_amendment_act):
    """_collect_amendment_rels returns correct dict with old_text mapped from removed_text."""
    result = _collect_amendment_rels(simple_amendment_act)

    assert len(result) == 1
    rel = result[0]

    assert "amend_uid" in rel, "Rel missing 'amend_uid'"
    assert "target_uid" in rel, "Rel missing 'target_uid'"
    assert "amendment_type" in rel, "Rel missing 'amendment_type'"
    assert "new_text" in rel, "Rel missing 'new_text'"
    assert "old_text" in rel, "Rel missing 'old_text' (should be mapped from removed_text)"
    assert "effective_date" in rel, "Rel missing 'effective_date'"

    # Verify correct mapping from removed_text -> old_text
    assert rel["old_text"] == "old text here", (
        f"old_text should be 'old text here', got {rel['old_text']!r}"
    )
    assert "removed_text" not in rel, "Rel should NOT have 'removed_text' key (use 'old_text')"

    # Verify target_uid is section_uid(135)
    assert rel["target_uid"] == "sec-135", (
        f"target_uid should be 'sec-135', got {rel['target_uid']!r}"
    )

    assert rel["amendment_type"] == "SUBSTITUTES"


def test_collect_amendment_rels_decriminalizes():
    """_collect_amendment_rels handles DECRIMINALIZES type without raising."""
    amend_act = AmendmentAct(
        uid=amendment_act_uid("Corporate Laws Amendment", 2026),
        node_type="AmendmentAct",
        page_number=1,
        text="Corporate Laws (Amendment) Act, 2026.",
        source_pdf="Corporate Laws (Amendment) Act, 2026.pdf",
        title="Corporate Laws (Amendment) Act, 2026",
        year=2026,
        entries=[
            AmendmentEntry(
                target_section="section 447",
                amendment_type="DECRIMINALIZES",
                new_text="",
                removed_text="",
                raw_text="Section 447 is decriminalized.",
            )
        ],
    )

    result = _collect_amendment_rels(amend_act)

    assert len(result) == 1, "DECRIMINALIZES entry should be included, not skipped"
    assert result[0]["amendment_type"] == "DECRIMINALIZES"
    assert result[0]["target_uid"] == "sec-447"


def test_collect_amendment_rels_bad_target(caplog):
    """_collect_amendment_rels logs WARNING and skips entries with unparseable target_section."""
    amend_act = AmendmentAct(
        uid=amendment_act_uid("Corporate Laws Amendment", 2026),
        node_type="AmendmentAct",
        page_number=1,
        text="Amendment act.",
        source_pdf="Corporate Laws (Amendment) Act, 2026.pdf",
        title="Corporate Laws (Amendment) Act, 2026",
        year=2026,
        entries=[
            AmendmentEntry(
                target_section="something invalid",
                amendment_type="SUBSTITUTES",
                new_text="x",
                removed_text="y",
                raw_text="invalid target.",
            )
        ],
    )

    with caplog.at_level(logging.WARNING, logger="graph.loader"):
        result = _collect_amendment_rels(amend_act)

    assert result == [], "Entry with bad target_section should be skipped"
    assert any("Cannot resolve" in record.message for record in caplog.records), (
        "Expected a WARNING log about unresolvable amendment target"
    )


# ---------------------------------------------------------------------------
# Tests for _collect_ruleset_nodes
# ---------------------------------------------------------------------------


def test_collect_ruleset_nodes(simple_ruleset):
    """_collect_ruleset_nodes returns dict with RuleSet, Rule, Form keys."""
    result = _collect_ruleset_nodes(simple_ruleset)

    assert "RuleSet" in result, "Missing 'RuleSet' key"
    assert "Rule" in result, "Missing 'Rule' key"
    assert "Form" in result, "Missing 'Form' key"

    assert len(result["RuleSet"]) == 1
    assert len(result["Rule"]) == 1
    assert len(result["Form"]) == 1

    # Verify base properties present
    for label, nodes in result.items():
        for node in nodes:
            assert "uid" in node, f"{label} node missing 'uid'"
            assert "source_pdf" in node, f"{label} node missing 'source_pdf'"
