"""
Unit tests for graph.linker — cross-reference resolution and edge creation.

Tests cover:
- resolve_target_uid: maps "section N" patterns to node UIDs
- _collect_cross_ref_edges: traverses Act tree, resolves refs, deduplicates
- _detect_additional_refs: detects IMPOSES_PENALTY and PRESCRIBES_FOR patterns

No Neo4j connection required — all functions under test are pure.
"""

from __future__ import annotations

import logging

import pytest

from ingestion.models import Act, Chapter, CrossReference, Section
from ingestion.uid_generator import (
    act_uid,
    chapter_uid,
    section_uid,
)


# ---------------------------------------------------------------------------
# Helpers to build minimal in-memory Act trees
# ---------------------------------------------------------------------------


def _make_section(number: int, text: str = "", cross_references: list | None = None) -> Section:
    """Build a minimal Section for testing."""
    return Section(
        uid=section_uid(number),
        node_type="Section",
        page_number=1,
        text=text,
        source_pdf="test.pdf",
        number=number,
        title=f"Section {number}",
        cross_references=cross_references or [],
    )


def _make_act(sections: list[Section]) -> Act:
    """Wrap sections in a minimal Act with one chapter."""
    chapter = Chapter(
        uid=chapter_uid("I"),
        node_type="Chapter",
        page_number=1,
        text="",
        source_pdf="test.pdf",
        number="I",
        title="Test Chapter",
        sections=sections,
    )
    return Act(
        uid=act_uid("Companies", 2013),
        node_type="Act",
        page_number=1,
        text="",
        source_pdf="test.pdf",
        title="The Companies Act, 2013",
        year=2013,
        chapters=[chapter],
    )


# ---------------------------------------------------------------------------
# Tests for resolve_target_uid
# ---------------------------------------------------------------------------


class TestResolveTargetUid:
    """Tests for resolve_target_uid() — maps target_pattern strings to node UIDs."""

    def test_plain_section_number(self):
        """'section 135' -> 'sec-135'"""
        from graph.linker import resolve_target_uid

        assert resolve_target_uid("section 135") == "sec-135"

    def test_small_section_number(self):
        """'section 2' -> 'sec-2'"""
        from graph.linker import resolve_target_uid

        assert resolve_target_uid("section 2") == "sec-2"

    def test_alpha_suffix_returns_none(self):
        """'section 149A' has non-numeric suffix — resolver cannot parse, return None."""
        from graph.linker import resolve_target_uid

        assert resolve_target_uid("section 149A") is None

    def test_rule_returns_none(self):
        """'rule 3' does not match section pattern — return None."""
        from graph.linker import resolve_target_uid

        assert resolve_target_uid("rule 3") is None

    def test_empty_string_returns_none(self):
        """Empty string -> None."""
        from graph.linker import resolve_target_uid

        assert resolve_target_uid("") is None

    def test_random_text_returns_none(self):
        """Unstructured text -> None."""
        from graph.linker import resolve_target_uid

        assert resolve_target_uid("something random") is None

    def test_section_with_extra_suffix_returns_none(self):
        """'section 135(1)' — trailing non-numeric chars -> None (not a simple section ref)."""
        from graph.linker import resolve_target_uid

        # The pattern uses $ anchor so trailing content disqualifies it
        assert resolve_target_uid("section 135(1)") is None


# ---------------------------------------------------------------------------
# Tests for _collect_cross_ref_edges
# ---------------------------------------------------------------------------


class TestCollectCrossRefEdges:
    """Tests for _collect_cross_ref_edges() — Act-wide edge collection."""

    def test_returns_edge_for_resolvable_ref(self):
        """Section with a resolvable cross-reference produces an edge dict."""
        from graph.linker import _collect_cross_ref_edges

        section = _make_section(
            135,
            cross_references=[
                CrossReference(
                    target_pattern="section 149",
                    ref_type="REFERS_TO",
                    raw_text="under section 149",
                ),
            ],
        )
        act = _make_act([section])

        edges = _collect_cross_ref_edges(act)

        assert len(edges) >= 1
        assert {"source_uid": "sec-135", "target_uid": "sec-149", "ref_type": "REFERS_TO"} in edges

    def test_unresolvable_ref_is_skipped(self, caplog):
        """Unresolvable target_pattern produces no edge and logs WARNING."""
        from graph.linker import _collect_cross_ref_edges

        section = _make_section(
            100,
            cross_references=[
                CrossReference(
                    target_pattern="section 999A",
                    ref_type="REFERS_TO",
                    raw_text="under section 999A",
                ),
            ],
        )
        act = _make_act([section])

        with caplog.at_level(logging.WARNING, logger="graph.linker"):
            edges = _collect_cross_ref_edges(act)

        # No edges from unresolvable reference
        assert not any(e["source_uid"] == "sec-100" for e in edges)
        # Warning logged
        assert any("Unresolvable" in msg or "unresolvable" in msg or "sec-100" in msg for msg in caplog.messages)

    def test_deduplication_same_source_target_ref_type(self):
        """Two CrossReference objects with the same source/target/ref_type produce only one edge."""
        from graph.linker import _collect_cross_ref_edges

        section = _make_section(
            135,
            cross_references=[
                CrossReference(
                    target_pattern="section 149",
                    ref_type="REFERS_TO",
                    raw_text="under section 149",
                ),
                CrossReference(
                    target_pattern="section 149",
                    ref_type="REFERS_TO",
                    raw_text="referred to in section 149",  # different raw_text, same edge
                ),
            ],
        )
        act = _make_act([section])

        edges = _collect_cross_ref_edges(act)

        matching = [
            e for e in edges
            if e["source_uid"] == "sec-135" and e["target_uid"] == "sec-149" and e["ref_type"] == "REFERS_TO"
        ]
        assert len(matching) == 1, f"Expected 1 deduplicated edge, got {len(matching)}"

    def test_invalid_ref_type_is_skipped(self, caplog):
        """ref_type not in ALLOWED_REF_TYPES is rejected with WARNING."""
        from graph.linker import _collect_cross_ref_edges

        section = _make_section(
            10,
            cross_references=[
                CrossReference(
                    target_pattern="section 20",
                    ref_type="UNKNOWN_TYPE",
                    raw_text="something under section 20",
                ),
            ],
        )
        act = _make_act([section])

        with caplog.at_level(logging.WARNING, logger="graph.linker"):
            edges = _collect_cross_ref_edges(act)

        assert not any(e.get("ref_type") == "UNKNOWN_TYPE" for e in edges)


# ---------------------------------------------------------------------------
# Tests for _detect_additional_refs
# ---------------------------------------------------------------------------


class TestDetectAdditionalRefs:
    """Tests for _detect_additional_refs() — IMPOSES_PENALTY and PRESCRIBES_FOR detection."""

    def test_penalty_under_section(self):
        """'penalty under section 135' -> IMPOSES_PENALTY edge to sec-135."""
        from graph.linker import _detect_additional_refs

        edges = _detect_additional_refs("sec-10", "shall be liable to penalty under section 135")

        assert any(
            e["ref_type"] == "IMPOSES_PENALTY" and e["target_uid"] == "sec-135"
            for e in edges
        ), f"Expected IMPOSES_PENALTY->sec-135, got {edges}"

    def test_punishable_under_section(self):
        """'punishable under section 447' -> IMPOSES_PENALTY edge to sec-447."""
        from graph.linker import _detect_additional_refs

        edges = _detect_additional_refs("sec-10", "shall be punishable under section 447")

        assert any(
            e["ref_type"] == "IMPOSES_PENALTY" and e["target_uid"] == "sec-447"
            for e in edges
        ), f"Expected IMPOSES_PENALTY->sec-447, got {edges}"

    def test_prescribed_in_rule_skipped(self, caplog):
        """'as prescribed in rule 3' -> PRESCRIBES_FOR but rule target not resolvable -> edge skipped."""
        from graph.linker import _detect_additional_refs

        with caplog.at_level(logging.WARNING, logger="graph.linker"):
            edges = _detect_additional_refs("sec-10", "in the manner as prescribed in rule 3")

        # Edge should be skipped since rule resolution returns None
        prescribes_edges = [e for e in edges if e.get("ref_type") == "PRESCRIBES_FOR"]
        # target_uid would be None -> edge must be skipped
        assert all(e.get("target_uid") is not None for e in prescribes_edges), (
            "Expected rule-based PRESCRIBES_FOR edge to be skipped (target_uid None)"
        )

    def test_no_additional_refs_plain_text(self):
        """Plain text with no patterns -> empty list."""
        from graph.linker import _detect_additional_refs

        edges = _detect_additional_refs("sec-10", "The company shall file returns annually.")

        assert edges == []

    def test_imposes_penalty_deduplication(self):
        """Same pattern match appearing twice still produces one edge."""
        from graph.linker import _detect_additional_refs

        # Text with the same section reference in two matching patterns
        edges = _detect_additional_refs(
            "sec-10",
            "penalty under section 135 and penalty under section 135",
        )

        imposes = [e for e in edges if e["target_uid"] == "sec-135" and e["ref_type"] == "IMPOSES_PENALTY"]
        assert len(imposes) == 1, f"Expected 1 deduplicated edge, got {imposes}"


# ---------------------------------------------------------------------------
# Tests for link_derived_rules
# ---------------------------------------------------------------------------


class TestLinkDerivedRules:
    """Tests for link_derived_rules() — DERIVED_RULE edge collection logic."""

    def test_extracts_derived_rule_from_title(self):
        """Rule with 'under section N' in title produces a DERIVED_RULE edge."""
        from graph.linker import _DERIVED_RULE_RE

        match = _DERIVED_RULE_RE.search("Companies (CSR Policy) Rules under section 135")
        assert match is not None
        assert match.group(1) == "135"

    def test_extracts_pursuant_to(self):
        """'pursuant to section N' is also matched."""
        from graph.linker import _DERIVED_RULE_RE

        match = _DERIVED_RULE_RE.search("pursuant to section 469 of the Companies Act")
        assert match is not None
        assert match.group(1) == "469"

    def test_extracts_in_pursuance_of(self):
        """'in pursuance of section N' is matched."""
        from graph.linker import _DERIVED_RULE_RE

        match = _DERIVED_RULE_RE.search("in pursuance of section 135")
        assert match is not None
        assert match.group(1) == "135"

    def test_no_match_for_plain_text(self):
        """Text without section reference patterns returns no match."""
        from graph.linker import _DERIVED_RULE_RE

        match = _DERIVED_RULE_RE.search("The company shall file annual returns")
        assert match is None
