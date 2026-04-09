"""
Tests for the Companies Rules 2014 parser.

Covers:
    - Devanagari/Hindi detection
    - Hindi content being skipped with warning
    - Rule heading extraction
    - Form extraction nested inside Rules
    - Form body stored as raw text
    - Schedule detection
    - Unattached content warnings
"""

from __future__ import annotations

import logging

import pytest

from ingestion.rules_parser import is_devanagari_majority, parse_companies_rules


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_element(etype: str, content: str, page: int = 1) -> dict:
    """Construct a minimal OpenDataLoader element dict for testing."""
    return {"type": etype, "content": content, "page number": page}


# ---------------------------------------------------------------------------
# is_devanagari_majority
# ---------------------------------------------------------------------------

class TestDevanagariDetection:
    def test_pure_hindi_is_majority(self):
        assert is_devanagari_majority("कंपनी अधिनियम") is True

    def test_english_is_not_majority(self):
        assert is_devanagari_majority("Companies Act") is False

    def test_empty_string_is_not_majority(self):
        assert is_devanagari_majority("") is False

    def test_whitespace_only_is_not_majority(self):
        assert is_devanagari_majority("   ") is False

    def test_mixed_mostly_hindi_is_majority(self):
        # Mix of Hindi and English where Hindi chars > 30%
        assert is_devanagari_majority("कंपनी act") is True

    def test_mixed_mostly_english_is_not_majority(self):
        # Single Devanagari char in a long English sentence — well under 30%
        assert is_devanagari_majority("Companies Act section क details") is False


# ---------------------------------------------------------------------------
# parse_companies_rules — Hindi skipping
# ---------------------------------------------------------------------------

class TestHindiContentSkipped:
    def test_hindi_content_skipped(self, caplog):
        elements = [
            make_element("heading", "1. Short title and commencement"),
            make_element("paragraph", "कंपनी अधिनियम"),   # should be skipped
            make_element("paragraph", "(1) These rules may be called the Companies Rules."),
        ]
        with caplog.at_level(logging.WARNING, logger="ingestion.rules_parser"):
            ruleset = parse_companies_rules(iter(elements), source_pdf="test.pdf")

        # The Hindi paragraph should not appear in the rule text
        assert len(ruleset.rules) == 1
        rule_text = ruleset.rules[0].text
        assert "कंपनी" not in rule_text
        assert "These rules" in rule_text

    def test_hindi_warning_logged(self, caplog):
        elements = [
            make_element("paragraph", "कंपनी अधिनियम"),
        ]
        with caplog.at_level(logging.WARNING, logger="ingestion.rules_parser"):
            parse_companies_rules(iter(elements), source_pdf="test.pdf")

        # A warning should mention Devanagari
        assert any(
            "Devanagari" in record.message
            for record in caplog.records
        ), f"Expected Devanagari warning but got: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# parse_companies_rules — rule parsing
# ---------------------------------------------------------------------------

class TestRuleParsing:
    def test_basic_rule_heading_and_number(self):
        elements = [
            make_element("heading", "1. Short title and commencement"),
            make_element("paragraph", "(1) These rules may be called the Companies Rules."),
        ]
        ruleset = parse_companies_rules(iter(elements), source_pdf="test.pdf")
        assert len(ruleset.rules) == 1
        assert ruleset.rules[0].number == "1"
        assert ruleset.rules[0].uid == "rule-1"

    def test_rule_title_extracted(self):
        elements = [
            make_element("heading", "3. Filing of forms"),
        ]
        ruleset = parse_companies_rules(iter(elements), source_pdf="test.pdf")
        assert len(ruleset.rules) == 1
        assert ruleset.rules[0].title == "Filing of forms"

    def test_multiple_rules_extracted(self):
        elements = [
            make_element("heading", "1. Short title"),
            make_element("paragraph", "Text for rule 1."),
            make_element("heading", "2. Definitions"),
            make_element("paragraph", "Text for rule 2."),
            make_element("heading", "3. Application"),
            make_element("paragraph", "Text for rule 3."),
        ]
        ruleset = parse_companies_rules(iter(elements), source_pdf="test.pdf")
        assert len(ruleset.rules) == 3
        numbers = [r.number for r in ruleset.rules]
        assert numbers == ["1", "2", "3"]

    def test_ruleset_metadata(self):
        ruleset = parse_companies_rules(iter([]), source_pdf="test.pdf")
        assert ruleset.year == 2014
        assert "Rules" in ruleset.title

    def test_rule_with_alpha_suffix(self):
        elements = [
            make_element("heading", "3A. Special provisions"),
        ]
        ruleset = parse_companies_rules(iter(elements), source_pdf="test.pdf")
        assert len(ruleset.rules) == 1
        assert ruleset.rules[0].number == "3A"
        assert ruleset.rules[0].uid == "rule-3A"


# ---------------------------------------------------------------------------
# parse_companies_rules — Form extraction
# ---------------------------------------------------------------------------

class TestFormExtraction:
    def test_form_extracted_and_nested_in_rule(self):
        elements = [
            make_element("heading", "3. Filing of forms"),
            make_element("heading", "Form No. MGT-7"),
            make_element("paragraph", "Annual return form content here."),
        ]
        ruleset = parse_companies_rules(iter(elements), source_pdf="test.pdf")
        assert len(ruleset.rules) == 1
        rule = ruleset.rules[0]
        assert len(rule.forms) == 1
        assert rule.forms[0].form_number == "MGT-7"

    def test_form_associated_rule_set_correctly(self):
        elements = [
            make_element("heading", "3. Filing of forms"),
            make_element("heading", "Form No. MGT-7"),
            make_element("paragraph", "Form body."),
        ]
        ruleset = parse_companies_rules(iter(elements), source_pdf="test.pdf")
        assert ruleset.rules[0].forms[0].associated_rule == "3"

    def test_form_body_is_raw_text(self):
        elements = [
            make_element("heading", "3. Filing of forms"),
            make_element("heading", "Form No. MGT-7"),
            make_element("paragraph", "First paragraph of form."),
            make_element("paragraph", "Second paragraph of form."),
        ]
        ruleset = parse_companies_rules(iter(elements), source_pdf="test.pdf")
        form = ruleset.rules[0].forms[0]
        assert "First paragraph" in form.text
        assert "Second paragraph" in form.text

    def test_form_body_includes_table_marker(self):
        elements = [
            make_element("heading", "3. Filing of forms"),
            make_element("heading", "Form No. MGT-7"),
            make_element("paragraph", "Intro text."),
            make_element("table row", "Table row content"),
        ]
        ruleset = parse_companies_rules(iter(elements), source_pdf="test.pdf")
        form = ruleset.rules[0].forms[0]
        assert "[TABLE CONTENT]" in form.text

    def test_multiple_forms_in_one_rule(self):
        elements = [
            make_element("heading", "3. Filing of forms"),
            make_element("heading", "Form No. MGT-7"),
            make_element("paragraph", "Form MGT-7 content."),
            make_element("heading", "Form No. MGT-8"),
            make_element("paragraph", "Form MGT-8 content."),
        ]
        ruleset = parse_companies_rules(iter(elements), source_pdf="test.pdf")
        rule = ruleset.rules[0]
        assert len(rule.forms) == 2
        assert rule.forms[0].form_number == "MGT-7"
        assert rule.forms[1].form_number == "MGT-8"

    def test_form_uid_generated(self):
        elements = [
            make_element("heading", "3. Filing of forms"),
            make_element("heading", "Form No. MGT-7"),
        ]
        ruleset = parse_companies_rules(iter(elements), source_pdf="test.pdf")
        form = ruleset.rules[0].forms[0]
        assert form.uid == "form-mgt-7"


# ---------------------------------------------------------------------------
# parse_companies_rules — Schedule detection
# ---------------------------------------------------------------------------

class TestScheduleDetection:
    def test_schedule_heading_not_treated_as_rule(self):
        elements = [
            make_element("heading", "1. Short title"),
            make_element("paragraph", "Rule text."),
            make_element("heading", "SCHEDULE I"),
            make_element("paragraph", "Schedule content."),
        ]
        ruleset = parse_companies_rules(iter(elements), source_pdf="test.pdf")
        # After SCHEDULE I heading, a new rule should NOT be created
        # Schedules are separate — only the 1 explicit rule should exist
        rule_numbers = [r.number for r in ruleset.rules]
        assert "1" in rule_numbers
        # SCHEDULE I should not produce a rule numbered "I" or similar
        # (it may produce a Schedule object internally)
        assert not any(r.number.upper() in ("I", "II", "III") for r in ruleset.rules)


# ---------------------------------------------------------------------------
# parse_companies_rules — Unattached content warnings
# ---------------------------------------------------------------------------

class TestUnattachedContentWarning:
    def test_unattached_paragraph_generates_warning(self, caplog):
        elements = [
            make_element("paragraph", "This text has no preceding rule heading."),
        ]
        with caplog.at_level(logging.WARNING, logger="ingestion.rules_parser"):
            parse_companies_rules(iter(elements), source_pdf="test.pdf")

        assert any(
            "Unattached" in record.message
            for record in caplog.records
        ), f"Expected 'Unattached' warning but got: {[r.message for r in caplog.records]}"


# ---------------------------------------------------------------------------
# Integration test (slow — skipped in normal runs)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_parse_rules_integration():
    """Load actual Rules raw JSON and parse — verifies end-to-end parsing on real data."""
    from pathlib import Path
    from ingestion.extractor import load_raw_json, flatten_elements

    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    json_files = list(raw_dir.glob("*.json"))

    rules_json = None
    for jf in json_files:
        if "rules" in jf.stem.lower():
            rules_json = jf
            break

    if rules_json is None:
        pytest.skip("Rules raw JSON not found — run extraction first")

    data = load_raw_json(rules_json)
    elements = flatten_elements(data["kids"])
    ruleset = parse_companies_rules(elements, source_pdf="Companies Rules, 2014.pdf")
    assert len(ruleset.rules) > 0, "Expected at least one rule to be parsed"
