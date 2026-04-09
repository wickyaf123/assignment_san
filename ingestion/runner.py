"""
CLI entry point for the ViddhiAI ingestion pipeline.

Orchestrates the full two-stage ingestion process:
    Stage 1 (D-22): Batch PDF extraction using OpenDataLoader (single JVM call)
    Stage 2 (D-22): Regex state-machine parsing for each PDF, in order (D-03):
        1. Companies Act 2013  -> Act object -> companies_act_2013_parsed.json
        2. Amendment Act 2026  -> AmendmentAct object -> amendment_act_2026_parsed.json
        3. Companies Rules 2014 -> RuleSet object -> companies_rules_2014_parsed.json

Each parsed JSON is accompanied by a summary report file (D-09):
    companies_act_2013_parsed_report.json
    amendment_act_2026_parsed_report.json
    companies_rules_2014_parsed_report.json

Security (T-01-08): PARSED_DIR and RAW_DIR are hardcoded relative to __file__
location — not user-configurable. mkdir with exist_ok prevents race conditions.

Usage:
    python -m ingestion.runner
    python -m ingestion          (via __main__.py)
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from ingestion.act_parser import parse_amendment_act, parse_companies_act
from ingestion.amendment_llm_parser import parse_amendment_with_llm
from ingestion.extractor import extract_pdfs, flatten_elements, load_raw_json
from ingestion.models import Act, AmendmentAct, ParseReport, RuleSet
from ingestion.preprocessor import preprocess_act, preprocess_ruleset
from ingestion.rules_parser import parse_companies_rules

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path constants — all derived from __file__, never user-configurable (T-01-08)
# ---------------------------------------------------------------------------

# Project root: backend/ingestion/runner.py -> resolve 3 levels up
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# backend/data/raw/ — OpenDataLoader JSON output (D-23)
RAW_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "raw"

# backend/data/parsed/ — Final parsed JSON files (D-07)
PARSED_DIR: Path = Path(__file__).resolve().parent.parent / "data" / "parsed"

# PDFs in parsing order per D-03: Act -> Amendment -> Rules
PDF_FILES: list[Path] = [
    PROJECT_ROOT / "Companies Act, 2013.pdf",
    PROJECT_ROOT / "Corporate Laws (Amendment) Act, 2026.pdf",
    PROJECT_ROOT / "Companies Rules, 2014.pdf",
]


# ---------------------------------------------------------------------------
# Node counting functions — produce ParseReport objects (D-09)
# ---------------------------------------------------------------------------

def count_nodes(act: Act) -> ParseReport:
    """Count all node types in a parsed Act and return a ParseReport.

    Recursively traverses the Act hierarchy to count sections, subsections,
    clauses, provisos, explanations, and definitions.

    Args:
        act: Fully populated Act object from parse_companies_act().

    Returns:
        ParseReport with counts for all hierarchical node types.
    """
    total_sections = 0
    total_subsections = 0
    total_clauses = 0
    total_provisos = 0
    total_explanations = 0
    total_definitions = 0

    for chapter in act.chapters:
        for section in chapter.sections:
            total_sections += 1
            total_provisos += len(section.provisos)
            total_explanations += len(section.explanations)
            total_definitions += len(section.definitions)
            total_clauses += len(section.clauses)

            for clause in section.clauses:
                total_provisos += len(clause.provisos)
                total_explanations += len(clause.explanations)
                for subclause in clause.subclauses:
                    total_provisos += len(subclause.provisos)
                    total_explanations += len(subclause.explanations)

            for subsection in section.subsections:
                total_subsections += 1
                total_provisos += len(subsection.provisos)
                total_explanations += len(subsection.explanations)
                total_definitions += len(subsection.definitions)
                total_clauses += len(subsection.clauses)

                for clause in subsection.clauses:
                    total_provisos += len(clause.provisos)
                    total_explanations += len(clause.explanations)
                    for subclause in clause.subclauses:
                        total_provisos += len(subclause.provisos)
                        total_explanations += len(subclause.explanations)

    # Coverage: use sections as proxy for "structured elements found"
    # A rough heuristic: Companies Act 2013 has 470 sections;
    # coverage_percent = min(total_sections / 470 * 100, 100.0)
    expected_sections = 470
    coverage_pct = min(round(total_sections / expected_sections * 100, 1), 100.0) if expected_sections > 0 else 0.0

    return ParseReport(
        source_pdf=act.source_pdf,
        total_sections=total_sections,
        total_subsections=total_subsections,
        total_clauses=total_clauses,
        total_provisos=total_provisos,
        total_explanations=total_explanations,
        total_definitions=total_definitions,
        warnings=[],
        coverage_percent=coverage_pct,
    )


def count_amendment_nodes(amendment: AmendmentAct) -> ParseReport:
    """Count nodes in a parsed AmendmentAct and return a ParseReport.

    Amendment Acts use entry-based structure, not a full hierarchy.
    Each AmendmentEntry targets a section — mapped to total_sections.
    Hierarchical fields (subsections, clauses, etc.) are set to 0.

    Args:
        amendment: Fully populated AmendmentAct object.

    Returns:
        ParseReport with total_sections = len(entries), hierarchical fields = 0.
    """
    total_entries = len(amendment.entries)

    # Coverage: classified entries (those with a recognised amendment_type)
    _VALID_AMENDMENT_TYPES = {"SUBSTITUTES", "INSERTS", "OMITS", "DECRIMINALIZES"}
    classified = sum(
        1 for e in amendment.entries
        if e.amendment_type in _VALID_AMENDMENT_TYPES
    )
    coverage_pct = round(classified / total_entries * 100, 1) if total_entries > 0 else 0.0

    warnings: list[str] = []
    for entry in amendment.entries:
        if entry.amendment_type not in _VALID_AMENDMENT_TYPES:
            warnings.append(
                f"Unrecognised amendment_type {entry.amendment_type!r} "
                f"for target {entry.target_section!r}"
            )

    return ParseReport(
        source_pdf=amendment.source_pdf,
        total_sections=total_entries,
        total_subsections=0,
        total_clauses=0,
        total_provisos=0,
        total_explanations=0,
        total_definitions=0,
        warnings=warnings,
        coverage_percent=coverage_pct,
    )


def count_rules_nodes(ruleset: RuleSet) -> ParseReport:
    """Count nodes in a parsed RuleSet and return a ParseReport.

    Rules use a two-level hierarchy: RuleSet -> Rule -> Form.
    Mapping:
        total_sections  = number of Rules
        total_clauses   = total number of Forms (reusing clauses field)

    Args:
        ruleset: Fully populated RuleSet object.

    Returns:
        ParseReport adapted for rules structure.
    """
    total_rules = len(ruleset.rules)
    total_forms = sum(len(r.forms) for r in ruleset.rules)

    # Coverage: rules that have title extracted vs total rules
    rules_with_title = sum(1 for r in ruleset.rules if r.title)
    coverage_pct = round(rules_with_title / total_rules * 100, 1) if total_rules > 0 else 0.0

    return ParseReport(
        source_pdf=ruleset.source_pdf,
        total_sections=total_rules,
        total_subsections=0,
        total_clauses=total_forms,
        total_provisos=0,
        total_explanations=0,
        total_definitions=0,
        warnings=[],
        coverage_percent=coverage_pct,
    )


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_output(data: Act | AmendmentAct | RuleSet, parsed_dir: Path, stem: str) -> Path:
    """Serialise a parsed legal object to JSON in the parsed output directory (D-02, D-07).

    Args:
        data: Any Pydantic model with a model_dump_json method.
        parsed_dir: Directory to write output into (created if absent).
        stem: Filename stem (e.g., "companies_act_2013") — "_parsed.json" is appended.

    Returns:
        Path to the written JSON file.
    """
    parsed_dir.mkdir(parents=True, exist_ok=True)
    output_path = parsed_dir / f"{stem}_parsed.json"
    output_path.write_text(data.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Wrote parsed output: %s (%.1f KB)", output_path.name, output_path.stat().st_size / 1024)
    return output_path


def write_report(report: ParseReport, parsed_dir: Path, stem: str) -> Path:
    """Write a ParseReport to a JSON summary file alongside the parsed output (D-09).

    Logs a human-readable summary of the report to INFO.

    Args:
        report: ParseReport object with node counts and coverage.
        parsed_dir: Directory to write the report into (should already exist).
        stem: Filename stem — "_parsed_report.json" is appended.

    Returns:
        Path to the written report file.
    """
    parsed_dir.mkdir(parents=True, exist_ok=True)
    report_path = parsed_dir / f"{stem}_parsed_report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "Report [%s]: %d sections, %d subsections, %d clauses, "
        "%d provisos, %d explanations, %d definitions | "
        "%d warnings | coverage: %.1f%%",
        stem,
        report.total_sections,
        report.total_subsections,
        report.total_clauses,
        report.total_provisos,
        report.total_explanations,
        report.total_definitions,
        len(report.warnings),
        report.coverage_percent,
    )
    return report_path


# ---------------------------------------------------------------------------
# File discovery helper
# ---------------------------------------------------------------------------

def resolve_raw_json(raw_outputs: dict[str, Path], keywords: list[str]) -> Path:
    """Find the raw JSON path whose stem matches any of the provided keywords.

    Case-insensitive substring match on the stem (filename without extension).
    All keywords are tried against all stems before raising.

    Args:
        raw_outputs: Dict mapping JSON stem -> Path (from extract_pdfs() output).
        keywords: List of keyword substrings to search for in stems.

    Returns:
        The Path for the first matching stem.

    Raises:
        FileNotFoundError: If no stem contains any of the provided keywords.
            The message lists expected keywords and available stems.
    """
    normalised = {stem.lower(): path for stem, path in raw_outputs.items()}

    for stem_lower, path in normalised.items():
        for kw in keywords:
            if kw.lower() in stem_lower:
                logger.debug("Resolved %s -> %s (matched keyword %r)", keywords, path.name, kw)
                return path

    raise FileNotFoundError(
        f"No raw JSON matching {keywords} found. "
        f"Available stems: {list(raw_outputs.keys())}"
    )


# ---------------------------------------------------------------------------
# Post-parse validation
# ---------------------------------------------------------------------------

def _validate_ingestion(act: Act, amendment: AmendmentAct, ruleset: RuleSet) -> None:
    """Log validation summary after parsing all PDFs."""
    section_count = sum(len(ch.sections) for ch in act.chapters)
    definition_count = sum(
        len(s.definitions)
        for ch in act.chapters
        for s in ch.sections
    ) + sum(
        len(ss.definitions)
        for ch in act.chapters
        for s in ch.sections
        for ss in s.subsections
    )
    amendment_count = len(amendment.entries)
    rule_count = len(ruleset.rules)

    logger.info("=" * 60)
    logger.info("INGESTION VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info("Sections:     %d (target: 400+)", section_count)
    logger.info("Definitions:  %d (target: 50+)", definition_count)
    logger.info("Amendments:   %d (target: >0)", amendment_count)
    logger.info("Rules:        %d (target: 30+)", rule_count)

    if section_count < 400:
        logger.warning("LOW SECTION COUNT: %d < 400 -- check parser coverage", section_count)
    if definition_count < 50:
        logger.warning("LOW DEFINITION COUNT: %d < 50 -- check DEFINITION_TERM_RE", definition_count)
    if amendment_count == 0:
        logger.warning("ZERO AMENDMENTS -- check amendment parser")
    if rule_count < 30:
        logger.warning("LOW RULE COUNT: %d < 30 -- check rules parser", rule_count)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------

def run_ingestion() -> None:
    """Orchestrate the full ingestion pipeline across all 3 PDFs.

    Stage 1: Batch-extract all PDFs to raw JSON (single OpenDataLoader JVM call).
    Stage 2: Parse each PDF in order (Act -> Amendment -> Rules per D-03).
    After each parse: write JSON output (D-02) and ParseReport (D-09) to PARSED_DIR.

    PDF processing order follows D-03:
        1. Companies Act, 2013
        2. Corporate Laws (Amendment) Act, 2026
        3. Companies Rules, 2014

    Raises:
        AssertionError: If extraction produces fewer than 3 JSON files.
        FileNotFoundError: If a PDF file is missing or a raw JSON cannot be matched.
    """
    pipeline_start = time.monotonic()
    logger.info("Starting ingestion pipeline...")
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Output directories: raw=%s, parsed=%s", RAW_DIR, PARSED_DIR)

    # ------------------------------------------------------------------
    # Stage 1: Extract all PDFs in a single batched call (D-21, D-22)
    # ------------------------------------------------------------------
    logger.info("Stage 1: Extracting %d PDF(s)...", len(PDF_FILES))
    for pdf in PDF_FILES:
        if not pdf.exists():
            logger.warning("PDF not found (will cause error): %s", pdf)

    raw_outputs: dict[str, Path] = extract_pdfs(PDF_FILES, RAW_DIR)
    logger.info("Extraction complete: %d JSON file(s) in %s", len(raw_outputs), RAW_DIR)

    assert len(raw_outputs) >= 3, (
        f"Expected at least 3 JSON outputs, got {len(raw_outputs)}: "
        f"{list(raw_outputs.keys())}"
    )

    # ------------------------------------------------------------------
    # Stage 2: Parse each PDF in D-03 order (Act -> Amendment -> Rules)
    # ------------------------------------------------------------------
    logger.info("Stage 2: Parsing extracted JSON files...")

    # --- 2a: Companies Act 2013 ---
    act_json_path = resolve_raw_json(raw_outputs, ["companies act", "companies_act"])
    logger.info("Parsing Companies Act from: %s", act_json_path.name)
    act_data = load_raw_json(act_json_path)
    act = parse_companies_act(flatten_elements(act_data["kids"]), source_pdf="Companies Act, 2013.pdf")

    # Stage 2.5a: Preprocess — normalize text, extract definitions, score quality
    logger.info("Preprocessing Companies Act...")
    act, act_preprocess_report = preprocess_act(act)

    write_output(act, PARSED_DIR, "companies_act_2013")
    write_report(count_nodes(act), PARSED_DIR, "companies_act_2013")

    # --- 2b: Amendment Act 2026 ---
    amend_json_path = resolve_raw_json(raw_outputs, ["amendment", "corporate laws"])
    logger.info("Parsing Amendment Act from: %s", amend_json_path.name)
    amend_data = load_raw_json(amend_json_path)

    # Try LLM parser first, fall back to regex
    amendment = parse_amendment_with_llm(
        flatten_elements(amend_data["kids"]),
        source_pdf="Corporate Laws (Amendment) Act, 2026.pdf",
    )
    if amendment is None or not amendment.entries:
        logger.info("Falling back to regex amendment parser")
        amend_data = load_raw_json(amend_json_path)  # reload since iterator was consumed
        amendment = parse_amendment_act(
            flatten_elements(amend_data["kids"]),
            source_pdf="Corporate Laws (Amendment) Act, 2026.pdf",
        )

    write_output(amendment, PARSED_DIR, "amendment_act_2026")
    write_report(count_amendment_nodes(amendment), PARSED_DIR, "amendment_act_2026")

    # --- 2c: Companies Rules 2014 ---
    rules_json_path = resolve_raw_json(raw_outputs, ["rules", "companies rules"])
    logger.info("Parsing Companies Rules from: %s", rules_json_path.name)
    rules_data = load_raw_json(rules_json_path)
    ruleset = parse_companies_rules(
        flatten_elements(rules_data["kids"]),
        source_pdf="Companies Rules, 2014.pdf",
    )

    # Stage 2.5c: Preprocess — clean garbled text, filter unusable rules, score quality
    logger.info("Preprocessing Companies Rules...")
    ruleset, rules_preprocess_report = preprocess_ruleset(ruleset)

    write_output(ruleset, PARSED_DIR, "companies_rules_2014")
    write_report(count_rules_nodes(ruleset), PARSED_DIR, "companies_rules_2014")

    _validate_ingestion(act, amendment, ruleset)

    elapsed = time.monotonic() - pipeline_start
    logger.info("Ingestion pipeline complete in %.1fs", elapsed)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_ingestion()
