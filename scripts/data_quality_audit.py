"""
Data Quality Audit for ViddhiAI Parsed Legal Data.

Analyzes all parsed JSON files and produces a comprehensive report on:
  - Text quality (garbled text, encoding issues, empty nodes)
  - Coverage gaps (missing sections, definitions, subsections)
  - Structural integrity (rule numbering, hierarchy consistency)
  - LLM readiness (text length distribution, noise ratio, formatting consistency)

Usage:
    python -m scripts.data_quality_audit
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

PARSED_DIR = Path(__file__).resolve().parent.parent / "data" / "parsed"

GARBLE_PATTERNS = [
    re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F]{2,}"),  # control chars
    re.compile(r"(?:[□○●◆■]{2,})"),                                 # box/shape runs
    re.compile(r"(?:\.\s+){5,}"),                                    # dot leaders
    re.compile(r"(?:\u0000{2,})"),                                   # null bytes
    re.compile(r"[ो]{3,}|[ा]{3,}|[ै]{3,}|[ी]{3,}"),               # repeated Devanagari vowel signs
]

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LEGAL_TERM_RE = re.compile(
    r"\b(section|sub-section|clause|proviso|provided|shall|may|"
    r"company|director|board|tribunal|registrar|act|rule|regulation)\b",
    re.IGNORECASE,
)
NOISE_CHARS = set("○●□■◆★☆※†‡‖¶•·…─━│┃┌┐└┘├┤┬┴┼╋")
FORM_TEMPLATE_RE = re.compile(
    r"(\.\.\.\.\.\.|_{5,}|\.{5,}|(?:Pre-fill)|"
    r"(?:\(?\s*/\s*/\s*\)?)|(?:ई-\s*ई))",
    re.IGNORECASE,
)


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def score_text_quality(text: str) -> dict:
    """Score a text node on multiple quality dimensions (0-100 each)."""
    if not text or not text.strip():
        return {
            "overall": 0,
            "length": 0,
            "legal_density": 0,
            "garble_score": 100,
            "encoding_issues": True,
            "is_empty": True,
            "noise_ratio": 1.0,
            "devanagari_ratio": 0.0,
            "issues": ["EMPTY_TEXT"],
        }

    issues: list[str] = []
    length = len(text.strip())

    non_ws = [ch for ch in text if not ch.isspace()]
    total_non_ws = len(non_ws) if non_ws else 1

    devanagari_count = sum(1 for ch in non_ws if DEVANAGARI_RE.match(ch))
    devanagari_ratio = devanagari_count / total_non_ws

    noise_count = sum(1 for ch in non_ws if ch in NOISE_CHARS)
    noise_ratio = noise_count / total_non_ws

    garble_hits = sum(1 for p in GARBLE_PATTERNS if p.search(text))
    null_count = text.count("\u0000")
    form_template_matches = len(FORM_TEMPLATE_RE.findall(text))

    legal_words = LEGAL_TERM_RE.findall(text)
    words = text.split()
    word_count = len(words) if words else 1
    legal_density = len(legal_words) / word_count

    alpha_count = sum(1 for ch in text if ch.isalpha())
    alpha_ratio = alpha_count / len(text) if text else 0

    punc_special = sum(1 for ch in text if ch in "()[]{}*○□●/\\|─")
    punc_ratio = punc_special / total_non_ws

    if length == 0:
        issues.append("EMPTY_TEXT")
    elif length < 20:
        issues.append("VERY_SHORT_TEXT")
    elif length < 50:
        issues.append("SHORT_TEXT")

    if null_count > 0:
        issues.append(f"NULL_BYTES({null_count})")
    if garble_hits > 0:
        issues.append(f"GARBLED_PATTERNS({garble_hits})")
    if devanagari_ratio > 0.3:
        issues.append(f"DEVANAGARI_MAJORITY({devanagari_ratio:.0%})")
    if devanagari_ratio > 0.05 and devanagari_ratio <= 0.3:
        issues.append(f"MIXED_SCRIPT({devanagari_ratio:.0%})")
    if noise_ratio > 0.1:
        issues.append(f"HIGH_NOISE({noise_ratio:.0%})")
    if form_template_matches > 3:
        issues.append(f"FORM_TEMPLATE({form_template_matches})")
    if punc_ratio > 0.4:
        issues.append(f"PUNCTUATION_HEAVY({punc_ratio:.0%})")
    if alpha_ratio < 0.3 and length > 20:
        issues.append(f"LOW_ALPHA({alpha_ratio:.0%})")
    if legal_density == 0 and length > 100:
        issues.append("NO_LEGAL_TERMS")
    if length > 5000:
        issues.append(f"OVERSIZED({length})")

    garble_penalty = min(garble_hits * 20 + null_count * 5, 100)
    garble_score = 100 - garble_penalty

    length_score = min(length / 50 * 100, 100) if length < 50 else 100

    legal_score = min(legal_density * 500, 100)

    encoding_issues = null_count > 0 or garble_hits > 0
    content_quality = max(0, 100 - noise_ratio * 200 - punc_ratio * 100 - form_template_matches * 10)

    overall = (
        garble_score * 0.3
        + length_score * 0.15
        + legal_score * 0.25
        + content_quality * 0.3
    )
    if devanagari_ratio > 0.3:
        overall *= 0.2
    if encoding_issues:
        overall *= 0.5

    return {
        "overall": round(max(0, min(100, overall)), 1),
        "length": length,
        "legal_density": round(legal_density, 3),
        "garble_score": round(garble_score, 1),
        "encoding_issues": encoding_issues,
        "is_empty": length == 0,
        "noise_ratio": round(noise_ratio, 3),
        "devanagari_ratio": round(devanagari_ratio, 3),
        "punc_ratio": round(punc_ratio, 3),
        "form_template_hits": form_template_matches,
        "issues": issues,
    }


def walk_nodes(obj: dict, depth: int = 0) -> list[dict]:
    """Recursively walk a parsed JSON tree and collect all nodes with text."""
    results = []
    node_type = obj.get("node_type", "Unknown")
    uid = obj.get("uid", "")
    text = obj.get("text", "")
    title = obj.get("title", "")
    number = obj.get("number", obj.get("label", ""))

    score = score_text_quality(text)
    results.append({
        "uid": uid,
        "node_type": node_type,
        "number": str(number),
        "title": str(title)[:80],
        "depth": depth,
        "text_preview": text[:120].replace("\n", " ") if text else "",
        "quality": score,
    })

    for key in [
        "chapters", "sections", "subsections", "clauses", "subclauses",
        "provisos", "explanations", "definitions", "rules", "forms",
        "entries", "schedules",
    ]:
        children = obj.get(key, [])
        for child in children:
            results.extend(walk_nodes(child, depth + 1))

    return results


def analyze_act(data: dict) -> dict:
    """Deep analysis of the Companies Act parsed data."""
    nodes = walk_nodes(data)

    sections = [n for n in nodes if n["node_type"] == "Section"]
    subsections = [n for n in nodes if n["node_type"] == "SubSection"]
    clauses = [n for n in nodes if n["node_type"] == "Clause"]
    provisos = [n for n in nodes if n["node_type"] == "Proviso"]
    explanations = [n for n in nodes if n["node_type"] == "Explanation"]
    definitions = [n for n in nodes if n["node_type"] == "Definition"]
    chapters = [n for n in nodes if n["node_type"] == "Chapter"]

    sec_numbers = sorted([int(s["number"]) for s in sections if s["number"].isdigit()])
    expected_sections = list(range(1, 471))
    missing_sections = [s for s in expected_sections if s not in sec_numbers]

    sec_texts = [n["quality"]["length"] for n in sections]
    empty_sections = [n for n in sections if n["quality"]["is_empty"]]
    short_sections = [n for n in sections if 0 < n["quality"]["length"] < 50]
    oversized_sections = [n for n in sections if n["quality"]["length"] > 5000]

    quality_scores = [n["quality"]["overall"] for n in nodes if n["node_type"] != "Act"]
    poor_nodes = [n for n in nodes if n["quality"]["overall"] < 30]
    good_nodes = [n for n in nodes if n["quality"]["overall"] >= 70]

    all_issues: Counter = Counter()
    for n in nodes:
        for issue in n["quality"]["issues"]:
            issue_type = issue.split("(")[0]
            all_issues[issue_type] += 1

    return {
        "document": "Companies Act, 2013",
        "total_nodes": len(nodes),
        "chapters": len(chapters),
        "sections_found": len(sections),
        "sections_expected": 470,
        "sections_missing": len(missing_sections),
        "missing_section_numbers": missing_sections[:30],
        "subsections": len(subsections),
        "clauses": len(clauses),
        "provisos": len(provisos),
        "explanations": len(explanations),
        "definitions": len(definitions),
        "coverage_pct": round(len(sections) / 470 * 100, 1),
        "empty_sections": len(empty_sections),
        "short_sections": len(short_sections),
        "oversized_sections": [(n["uid"], n["quality"]["length"]) for n in oversized_sections],
        "avg_section_length": round(statistics.mean(sec_texts)) if sec_texts else 0,
        "median_section_length": round(statistics.median(sec_texts)) if sec_texts else 0,
        "avg_quality_score": round(statistics.mean(quality_scores), 1) if quality_scores else 0,
        "poor_quality_nodes": len(poor_nodes),
        "good_quality_nodes": len(good_nodes),
        "issue_counts": dict(all_issues.most_common(20)),
        "worst_nodes": [
            {"uid": n["uid"], "type": n["node_type"], "score": n["quality"]["overall"],
             "issues": n["quality"]["issues"], "preview": n["text_preview"][:80]}
            for n in sorted(nodes, key=lambda x: x["quality"]["overall"])[:10]
        ],
        "key_findings": [],
    }


def analyze_rules(data: dict) -> dict:
    """Deep analysis of the Companies Rules parsed data."""
    nodes = walk_nodes(data)

    rules = [n for n in nodes if n["node_type"] == "Rule"]

    quality_scores = [n["quality"]["overall"] for n in nodes if n["node_type"] != "RuleSet"]
    garbled_rules = [n for n in rules if n["quality"]["garble_score"] < 50]
    encoding_issues = [n for n in rules if n["quality"]["encoding_issues"]]
    empty_rules = [n for n in rules if n["quality"]["is_empty"]]
    form_template_rules = [n for n in rules if n["quality"].get("form_template_hits", 0) > 3]

    rule_numbers = [r["number"] for r in rules]
    abnormal_numbers = [r for r in rules if len(str(r["number"])) > 4]

    devanagari_mixed = [n for n in rules if n["quality"]["devanagari_ratio"] > 0.05]

    all_issues: Counter = Counter()
    for n in nodes:
        for issue in n["quality"]["issues"]:
            issue_type = issue.split("(")[0]
            all_issues[issue_type] += 1

    usable_rules = [n for n in rules if n["quality"]["overall"] >= 40 and not n["quality"]["encoding_issues"]]

    return {
        "document": "Companies Rules, 2014",
        "total_nodes": len(nodes),
        "rules_found": len(rules),
        "usable_rules": len(usable_rules),
        "garbled_rules": len(garbled_rules),
        "encoding_issue_rules": len(encoding_issues),
        "empty_rules": len(empty_rules),
        "form_template_rules": len(form_template_rules),
        "devanagari_mixed_rules": len(devanagari_mixed),
        "abnormal_rule_numbers": [(r["uid"], r["number"]) for r in abnormal_numbers],
        "avg_quality_score": round(statistics.mean(quality_scores), 1) if quality_scores else 0,
        "issue_counts": dict(all_issues.most_common(20)),
        "worst_nodes": [
            {"uid": n["uid"], "type": n["node_type"], "score": n["quality"]["overall"],
             "issues": n["quality"]["issues"], "preview": n["text_preview"][:80]}
            for n in sorted(nodes, key=lambda x: x["quality"]["overall"])[:10]
        ],
        "key_findings": [],
    }


def analyze_amendment(data: dict) -> dict:
    """Analysis of the Amendment Act parsed data."""
    entries = data.get("entries", [])
    return {
        "document": "Corporate Laws (Amendment) Act, 2026",
        "entries_found": len(entries),
        "status": "TOTAL_FAILURE" if len(entries) == 0 else "OK",
        "key_findings": [
            "Zero amendment entries parsed — both LLM and regex parsers produced nothing",
            "Regex parser requires 'In the ... Act, in section X' pattern — may not match actual PDF format",
            "LLM parser requires GEMINI_API_KEY — likely skipped during extraction",
        ] if len(entries) == 0 else [],
    }


def generate_preprocessing_gaps(act_report: dict, rules_report: dict, amend_report: dict) -> list[dict]:
    """Identify specific preprocessing gaps that need to be addressed."""
    gaps = []

    # GAP 1: No text normalization
    gaps.append({
        "id": "GAP-01",
        "severity": "CRITICAL",
        "category": "Text Normalization",
        "description": "No normalization layer between parsed JSON and Neo4j/LLM consumption",
        "evidence": [
            f"Rules: {rules_report['garbled_rules']} rules have garbled text with encoding issues",
            f"Rules: {rules_report['encoding_issue_rules']} rules have null bytes or control characters",
            f"Act: {act_report['oversized_sections']} sections have >5000 chars (accumulated footnotes/notifications)",
            "Section 1 text is ~8000+ chars of commencement notifications, not the actual section text",
        ],
        "recommendation": "Add a TextNormalizer class that cleans text between parsing and graph loading",
    })

    # GAP 2: No quality gate
    gaps.append({
        "id": "GAP-02",
        "severity": "HIGH",
        "category": "Quality Gate",
        "description": "No quality scoring/filtering before graph ingestion — garbage goes directly into Neo4j",
        "evidence": [
            f"Rules: avg quality score = {rules_report['avg_quality_score']}/100",
            f"Act: {act_report['poor_quality_nodes']} nodes score below 30/100",
            "Garbled text like 'o o ( / ) , o o ( ) * ( )' is stored as rule text in Neo4j",
        ],
        "recommendation": "Add quality_score field to graph nodes; filter or flag low-quality nodes",
    })

    # GAP 3: Font/encoding recovery for Rules
    gaps.append({
        "id": "GAP-03",
        "severity": "CRITICAL",
        "category": "PDF Extraction",
        "description": "Companies Rules 2014 PDF uses non-standard fonts (Devnagari-ChanakyaBoldA, TT1CBt00) that OpenDataLoader cannot map to Unicode",
        "evidence": [
            "Raw JSON shows font 'TT1CBt00' and 'Devnagari-ChanakyaBoldA'",
            f"{rules_report['garbled_rules']} rules have garbled/unmappable text",
            "Rule numbers parsed as '110010', '111111', '112212' — clearly wrong",
            "Hindi portions are partially extracted but with missing characters",
        ],
        "recommendation": "Find an English-only source PDF for Companies Rules, OR use OCR fallback for pages with non-standard fonts",
    })

    # GAP 4: Missing definitions extraction
    gaps.append({
        "id": "GAP-04",
        "severity": "HIGH",
        "category": "Parser Coverage",
        "description": "Zero definitions extracted despite Section 2 containing 95 defined terms",
        "evidence": [
            f"Definitions found: {act_report['definitions']}",
            "DEFINITION_TERM_RE requires opening quote character — PDF may use different quote styles",
            "Section 2 subsections contain definitions like '\"company\" means...' but aren't tagged",
        ],
        "recommendation": "Fix DEFINITION_TERM_RE to handle PDF quote variants; post-process Section 2 subsections as definitions",
    })

    # GAP 5: Section coverage gap
    gaps.append({
        "id": "GAP-05",
        "severity": "HIGH",
        "category": "Parser Coverage",
        "description": f"Only {act_report['sections_found']}/470 sections extracted ({act_report['coverage_pct']}%)",
        "evidence": [
            f"Missing sections include: {act_report['missing_section_numbers'][:15]}...",
            "SECTION_RE may fail on section headings that span multiple elements",
            "Footnotes starting with numbers (e.g., '1. Ins. by Act 31 of 2016') may be wrongly deduplicated",
        ],
        "recommendation": "Audit SECTION_RE against raw elements; relax footnote detection; add section-number gap detection",
    })

    # GAP 6: No LLM-ready text formatting
    gaps.append({
        "id": "GAP-06",
        "severity": "MEDIUM",
        "category": "LLM Preprocessing",
        "description": "Node text is raw concatenated paragraphs — no structural markers for LLM consumption",
        "evidence": [
            "Section text merges subsection references without delimiters",
            "Provisos and explanations are separate nodes but parent text doesn't indicate their existence",
            "No consistent formatting: some nodes use '--' for em-dash, others have unicode",
        ],
        "recommendation": "Add a format_for_llm() function that adds structural context (section number, hierarchy path) to each node's text",
    })

    # GAP 7: Amendment Act total failure
    gaps.append({
        "id": "GAP-07",
        "severity": "CRITICAL",
        "category": "Document Parsing",
        "description": "Amendment Act produced zero entries — entire document is unparsed",
        "evidence": amend_report["key_findings"],
        "recommendation": "Debug amendment PDF raw JSON; adjust regex patterns; ensure GEMINI_API_KEY is set for LLM fallback",
    })

    # GAP 8: No text deduplication
    gaps.append({
        "id": "GAP-08",
        "severity": "MEDIUM",
        "category": "Text Normalization",
        "description": "Footnote and notification text is accumulated into section body text",
        "evidence": [
            "Section 1 contains pages of commencement notification dates — not the actual section text",
            "Definition subsections accumulate cross-reference footnotes",
            f"{len([s for s in act_report.get('oversized_sections', []) if s[1] > 5000])} sections are oversized",
        ],
        "recommendation": "Separate footnote/notification text from substantive legal text during parsing",
    })

    return gaps


def print_report(act_report: dict, rules_report: dict, amend_report: dict, gaps: list[dict]) -> None:
    """Print the full audit report."""
    print("=" * 80)
    print("  ViddhiAI DATA QUALITY AUDIT REPORT")
    print("=" * 80)

    # Executive Summary
    print("\n## EXECUTIVE SUMMARY\n")
    total_issues = sum(1 for g in gaps if g["severity"] == "CRITICAL")
    print(f"  CRITICAL issues: {total_issues}")
    print(f"  HIGH issues:     {sum(1 for g in gaps if g['severity'] == 'HIGH')}")
    print(f"  MEDIUM issues:   {sum(1 for g in gaps if g['severity'] == 'MEDIUM')}")
    print()

    overall_health = "POOR"
    if total_issues == 0:
        overall_health = "GOOD" if not any(g["severity"] == "HIGH" for g in gaps) else "FAIR"
    print(f"  Overall Data Health: {overall_health}")
    print(f"  LLM Readiness:      {'NOT READY' if total_issues > 0 else 'READY'}")

    # Companies Act
    print("\n" + "-" * 80)
    print("  COMPANIES ACT, 2013")
    print("-" * 80)
    r = act_report
    print(f"  Sections:      {r['sections_found']}/{r['sections_expected']} ({r['coverage_pct']}%)")
    print(f"  Subsections:   {r['subsections']}")
    print(f"  Clauses:       {r['clauses']}")
    print(f"  Provisos:      {r['provisos']}")
    print(f"  Explanations:  {r['explanations']}")
    print(f"  Definitions:   {r['definitions']}  *** EXPECTED ~95 ***")
    print(f"  Avg Quality:   {r['avg_quality_score']}/100")
    print(f"  Poor Nodes:    {r['poor_quality_nodes']} (score < 30)")
    print(f"  Good Nodes:    {r['good_quality_nodes']} (score >= 70)")
    print(f"  Empty:         {r['empty_sections']} sections")
    print(f"  Oversized:     {len(r['oversized_sections'])} sections")

    print(f"\n  Missing sections (first 20): {r['missing_section_numbers'][:20]}")
    print(f"\n  Top Issues:")
    for issue, count in sorted(r["issue_counts"].items(), key=lambda x: -x[1])[:10]:
        print(f"    {issue}: {count}")

    if r["worst_nodes"]:
        print(f"\n  Worst Quality Nodes:")
        for wn in r["worst_nodes"][:5]:
            print(f"    {wn['uid']} ({wn['type']}) score={wn['score']}")
            print(f"      Issues: {', '.join(wn['issues'][:3])}")
            print(f"      Preview: {wn['preview'][:60]}...")

    # Companies Rules
    print("\n" + "-" * 80)
    print("  COMPANIES RULES, 2014")
    print("-" * 80)
    r = rules_report
    print(f"  Rules Found:      {r['rules_found']}")
    print(f"  Usable Rules:     {r['usable_rules']}  *** Only rules with quality >= 40 and no encoding issues ***")
    print(f"  Garbled Rules:    {r['garbled_rules']}")
    print(f"  Encoding Issues:  {r['encoding_issue_rules']}")
    print(f"  Empty Rules:      {r['empty_rules']}")
    print(f"  Form Templates:   {r['form_template_rules']}")
    print(f"  Hindi Mixed:      {r['devanagari_mixed_rules']}")
    print(f"  Avg Quality:      {r['avg_quality_score']}/100")

    if r["abnormal_rule_numbers"]:
        print(f"\n  Abnormal Rule Numbers (>4 digits):")
        for uid, num in r["abnormal_rule_numbers"]:
            print(f"    {uid}: number='{num}'")

    print(f"\n  Top Issues:")
    for issue, count in sorted(r["issue_counts"].items(), key=lambda x: -x[1])[:10]:
        print(f"    {issue}: {count}")

    # Amendment Act
    print("\n" + "-" * 80)
    print("  CORPORATE LAWS (AMENDMENT) ACT, 2026")
    print("-" * 80)
    r = amend_report
    print(f"  Status:  {r['status']}")
    print(f"  Entries: {r['entries_found']}")
    for f in r["key_findings"]:
        print(f"  - {f}")

    # Preprocessing Gaps
    print("\n" + "=" * 80)
    print("  PREPROCESSING GAPS (Missing Pipeline Stages)")
    print("=" * 80)
    for gap in sorted(gaps, key=lambda g: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}.get(g["severity"], 3)):
        print(f"\n  [{gap['severity']}] {gap['id']}: {gap['category']}")
        print(f"  {gap['description']}")
        print(f"  Evidence:")
        for e in gap["evidence"][:3]:
            print(f"    - {e}")
        print(f"  Fix: {gap['recommendation']}")

    # Recommended Preprocessing Pipeline
    print("\n" + "=" * 80)
    print("  RECOMMENDED PREPROCESSING PIPELINE")
    print("=" * 80)
    print("""
  Current Flow:
    PDF -> OpenDataLoader -> flatten -> regex parse -> JSON -> Neo4j -> LLM query

  Recommended Flow (add stages marked with *):
    PDF -> OpenDataLoader -> flatten -> regex parse -> JSON
      -> *TextNormalizer (clean encoding, strip noise, normalize formatting)
      -> *QualityScorer (score each node 0-100, flag issues)
      -> *StructureValidator (verify numbering, hierarchy, cross-refs)
      -> *LLMFormatter (add context headers, hierarchy path, consistent format)
      -> Neo4j (with quality_score property on each node)
      -> LLM query (with pre-formatted context)

  New Module: backend/ingestion/preprocessor.py
    class TextNormalizer:
      - strip_null_bytes(text) -> str
      - strip_form_templates(text) -> str
      - normalize_quotes_dashes(text) -> str
      - separate_footnotes(text) -> (main_text, footnotes)
      - collapse_whitespace(text) -> str

    class QualityScorer:
      - score_node(node) -> float  (0-100)
      - is_usable(node) -> bool    (threshold-based)

    class LLMFormatter:
      - format_section(section) -> str
        Output: "Section 135 (Corporate Social Responsibility)\\n\\n(1) Every company..."
      - format_with_context(node, ancestors) -> str
        Output: "Companies Act 2013 > Chapter IX > Section 135 > Sub-section (1)\\n\\nText..."
      - format_definition(term, text) -> str
        Output: "Definition: 'company' means [from Section 2(20)]\\n\\nText..."
""")


def main():
    act_path = PARSED_DIR / "companies_act_2013_parsed.json"
    rules_path = PARSED_DIR / "companies_rules_2014_parsed.json"
    amend_path = PARSED_DIR / "amendment_act_2026_parsed.json"

    missing = [p for p in [act_path, rules_path, amend_path] if not p.exists()]
    if missing:
        print(f"ERROR: Missing parsed files: {[str(p) for p in missing]}", file=sys.stderr)
        sys.exit(1)

    print("Loading parsed data...")
    act_data = load_json(act_path)
    rules_data = load_json(rules_path)
    amend_data = load_json(amend_path)

    print("Analyzing Companies Act 2013...")
    act_report = analyze_act(act_data)

    print("Analyzing Companies Rules 2014...")
    rules_report = analyze_rules(rules_data)

    print("Analyzing Amendment Act 2026...")
    amend_report = analyze_amendment(amend_data)

    print("Identifying preprocessing gaps...\n")
    gaps = generate_preprocessing_gaps(act_report, rules_report, amend_report)

    print_report(act_report, rules_report, amend_report, gaps)

    report_path = PARSED_DIR / "DATA_QUALITY_AUDIT.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "act": act_report,
            "rules": rules_report,
            "amendment": amend_report,
            "gaps": gaps,
        }, f, indent=2, default=str)
    print(f"\n  Full JSON report saved to: {report_path}")


if __name__ == "__main__":
    main()
