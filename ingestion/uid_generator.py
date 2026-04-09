"""
Deterministic UID generation for legal document nodes.

UIDs are generated from structural paths in the legislation hierarchy.
Examples:
    section_uid(135)         -> "sec-135"
    subsection_uid(135, 1)   -> "sec-135-ss-1"
    definition_uid("associate company") -> "def-associate-company"
"""

from __future__ import annotations

import re


def slugify(text: str) -> str:
    """Convert text to a URL-safe slug: lowercase, hyphens replace non-alphanumeric, strip trailing hyphens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text


def act_uid(short_name: str, year: int) -> str:
    """Return UID for an Act node. e.g., act_uid('Companies', 2013) -> 'act-companies-2013'"""
    return f"act-{slugify(short_name)}-{year}"


def chapter_uid(roman: str) -> str:
    """Return UID for a Chapter node. e.g., chapter_uid('II') -> 'ch-II'"""
    return f"ch-{roman.upper()}"


def section_uid(number: int) -> str:
    """Return UID for a Section node. e.g., section_uid(135) -> 'sec-135'"""
    return f"sec-{number}"


def subsection_uid(section_number: int, ss_number: int) -> str:
    """Return UID for a SubSection node. e.g., subsection_uid(135, 1) -> 'sec-135-ss-1'"""
    return f"sec-{section_number}-ss-{ss_number}"


def clause_uid(parent_uid: str, label: str) -> str:
    """Return UID for a Clause node. Works for section-level or subsection-level clauses.
    e.g., clause_uid('sec-135-ss-1', 'a') -> 'sec-135-ss-1-cl-a'"""
    return f"{parent_uid}-cl-{label}"


def subclause_uid(parent_uid: str, label: str) -> str:
    """Return UID for a SubClause node.
    e.g., subclause_uid('sec-135-ss-1-cl-a', 'i') -> 'sec-135-ss-1-cl-a-sc-i'"""
    return f"{parent_uid}-sc-{label}"


def proviso_uid(parent_uid: str, index: int) -> str:
    """Return UID for a Proviso node (1-indexed).
    e.g., proviso_uid('sec-135-ss-1', 1) -> 'sec-135-ss-1-proviso-1'"""
    return f"{parent_uid}-proviso-{index}"


def explanation_uid(parent_uid: str, index: int) -> str:
    """Return UID for an Explanation node (1-indexed).
    e.g., explanation_uid('sec-135', 1) -> 'sec-135-expl-1'"""
    return f"{parent_uid}-expl-{index}"


def definition_uid(term: str) -> str:
    """Return UID for a Definition node.
    e.g., definition_uid('associate company') -> 'def-associate-company'"""
    return f"def-{slugify(term)}"


def rule_uid(rule_number: str) -> str:
    """Return UID for a Rule node.
    e.g., rule_uid('3') -> 'rule-3'"""
    return f"rule-{rule_number}"


def ruleset_uid(name: str) -> str:
    """Return UID for a RuleSet node.
    e.g., ruleset_uid('Companies (Incorporation) Rules, 2014') -> 'ruleset-companies-incorporation-rules-2014'"""
    return f"ruleset-{slugify(name)}"


def form_uid(form_number: str) -> str:
    """Return UID for a Form node.
    e.g., form_uid('INC-1') -> 'form-inc-1'"""
    return f"form-{slugify(form_number)}"


def amendment_act_uid(short_name: str, year: int) -> str:
    """Return UID for an AmendmentAct node.
    e.g., amendment_act_uid('Corporate Laws Amendment', 2026) -> 'amend-corporate-laws-amendment-2026'"""
    return f"amend-{slugify(short_name)}-{year}"
