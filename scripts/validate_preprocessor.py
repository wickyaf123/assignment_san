"""
Validate the preprocessor on existing parsed data.

Loads parsed JSON, applies preprocessing, and reports improvements
without needing PDFs or Java (uses already-parsed data).

Usage:
    python -m scripts.validate_preprocessor
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.models import Act, RuleSet
from ingestion.preprocessor import (
    LLMFormatter,
    QualityScorer,
    TextNormalizer,
    extract_definitions_from_section2,
    preprocess_act,
    preprocess_ruleset,
)

PARSED_DIR = Path(__file__).resolve().parent.parent / "data" / "parsed"


def main():
    print("=" * 70)
    print("  PREPROCESSOR VALIDATION")
    print("=" * 70)

    # --- Load existing parsed data ---
    act_path = PARSED_DIR / "companies_act_2013_parsed.json"
    rules_path = PARSED_DIR / "companies_rules_2014_parsed.json"

    if not act_path.exists() or not rules_path.exists():
        print("ERROR: Parsed files not found. Run ingestion first.", file=sys.stderr)
        sys.exit(1)

    print("\nLoading parsed data...")
    act = Act.model_validate_json(act_path.read_text(encoding="utf-8"))
    ruleset = RuleSet.model_validate_json(rules_path.read_text(encoding="utf-8"))

    # --- Before stats ---
    before_sections = sum(len(ch.sections) for ch in act.chapters)
    before_definitions = sum(
        len(d) for ch in act.chapters
        for s in ch.sections
        for d in [s.definitions] + [ss.definitions for ss in s.subsections]
    )
    before_rules = len(ruleset.rules)

    scorer = QualityScorer()
    before_act_scores = []
    for ch in act.chapters:
        for sec in ch.sections:
            q = scorer.score(sec.text, "Section")
            before_act_scores.append(q.score)

    before_rules_scores = []
    before_unusable_rules = 0
    for rule in ruleset.rules:
        q = scorer.score(rule.text, "Rule")
        before_rules_scores.append(q.score)
        if not q.is_usable:
            before_unusable_rules += 1

    print(f"\n--- BEFORE PREPROCESSING ---")
    print(f"  Act sections:     {before_sections}")
    print(f"  Act definitions:  {before_definitions}")
    print(f"  Act avg quality:  {sum(before_act_scores)/len(before_act_scores):.1f}/100" if before_act_scores else "  N/A")
    print(f"  Rules total:      {before_rules}")
    print(f"  Rules unusable:   {before_unusable_rules}")
    print(f"  Rules avg quality:{sum(before_rules_scores)/len(before_rules_scores):.1f}/100" if before_rules_scores else "  N/A")

    # --- Run preprocessor ---
    print("\nRunning preprocessor on Act...")
    act, act_report = preprocess_act(act)

    print("Running preprocessor on Rules...")
    ruleset, rules_report = preprocess_ruleset(ruleset)

    # --- After stats ---
    after_sections = sum(len(ch.sections) for ch in act.chapters)
    after_definitions = sum(
        len(d) for ch in act.chapters
        for s in ch.sections
        for d in [s.definitions] + [ss.definitions for ss in s.subsections]
    )
    after_rules = len(ruleset.rules)

    after_act_scores = []
    for ch in act.chapters:
        for sec in ch.sections:
            q = scorer.score(sec.text, "Section")
            after_act_scores.append(q.score)

    after_rules_scores = []
    after_unusable_rules = 0
    for rule in ruleset.rules:
        q = scorer.score(rule.text, "Rule")
        after_rules_scores.append(q.score)
        if not q.is_usable:
            after_unusable_rules += 1

    print(f"\n--- AFTER PREPROCESSING ---")
    print(f"  Act sections:     {after_sections}")
    print(f"  Act definitions:  {after_definitions} (extracted {act_report.definitions_extracted} new)")
    print(f"  Act avg quality:  {sum(after_act_scores)/len(after_act_scores):.1f}/100" if after_act_scores else "  N/A")
    print(f"  Act footnotes separated: {act_report.footnotes_separated}")
    print(f"  Act nodes flagged unusable: {act_report.nodes_flagged_unusable}")
    print(f"  Rules kept:       {after_rules} (from {before_rules})")
    print(f"  Rules unusable:   {after_unusable_rules}")
    print(f"  Rules avg quality:{sum(after_rules_scores)/len(after_rules_scores):.1f}/100" if after_rules_scores else "  N/A")
    print(f"  Rules garbled cleaned: {rules_report.garbled_nodes_cleaned}")
    print(f"  Rules flagged unusable: {rules_report.nodes_flagged_unusable}")

    # --- Improvement summary ---
    print(f"\n--- IMPROVEMENTS ---")
    def_delta = after_definitions - before_definitions
    print(f"  Definitions: {before_definitions} -> {after_definitions} (+{def_delta})")

    rules_removed = before_rules - after_rules
    print(f"  Rules: {before_rules} -> {after_rules} ({rules_removed} garbage rules removed)")

    if before_act_scores and after_act_scores:
        avg_before = sum(before_act_scores) / len(before_act_scores)
        avg_after = sum(after_act_scores) / len(after_act_scores)
        print(f"  Act quality: {avg_before:.1f} -> {avg_after:.1f}")

    if before_rules_scores and after_rules_scores:
        avg_before = sum(before_rules_scores) / len(before_rules_scores)
        avg_after = sum(after_rules_scores) / len(after_rules_scores)
        print(f"  Rules quality: {avg_before:.1f} -> {avg_after:.1f}")

    # --- Sample LLM-formatted output ---
    formatter = LLMFormatter()
    print(f"\n--- SAMPLE LLM-FORMATTED OUTPUT ---")
    for ch in act.chapters:
        for sec in ch.sections:
            if sec.number == 135:
                formatted = formatter.format_section(sec, ch)
                print(f"\n  Section 135 (CSR):")
                print(f"  {formatted[:300]}...")
                break

    for ch in act.chapters:
        for sec in ch.sections:
            if sec.number == 2:
                for ss in sec.subsections:
                    for defn in ss.definitions:
                        sample = formatter.format_definition(defn.term, defn.text[:150])
                        print(f"\n  Definition sample:")
                        print(f"  {sample[:200]}...")
                        break
                    if ss.definitions:
                        break
                break

    # --- Write preprocessed output ---
    preprocessed_dir = PARSED_DIR / "preprocessed"
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    act_out = preprocessed_dir / "companies_act_2013_preprocessed.json"
    act_out.write_text(act.model_dump_json(indent=2), encoding="utf-8")
    print(f"\n  Preprocessed Act saved: {act_out}")

    rules_out = preprocessed_dir / "companies_rules_2014_preprocessed.json"
    rules_out.write_text(ruleset.model_dump_json(indent=2), encoding="utf-8")
    print(f"  Preprocessed Rules saved: {rules_out}")

    print("\n" + "=" * 70)
    print("  VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
