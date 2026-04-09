# ViddhiAI — Data Quality Audit Report

**Date:** April 9, 2026
**Scope:** Full ingestion pipeline audit — PDF extraction through Neo4j graph loading
**Source documents:** Companies Act 2013, Companies Rules 2014, Corporate Laws (Amendment) Act 2026

---

## 1. Executive Summary

The ingestion pipeline had fundamental data quality gaps: the extractor was silently dropping content, the parser was misclassifying sections, and all Hindi (Devanagari) text was being discarded. Across two phases of work — parsing fixes and Devanagari support — the pipeline went from partial, English-only extraction to bilingual, near-complete coverage of the source legislation.

| Metric | Before | After | Change |
|---|---|---|---|
| Act sections extracted | 269 / 470 | **470 / 470** | +75% (100% coverage) |
| Definitions extracted | 0 | **70** | from zero |
| SubSections | 291 | **1,157** | +297% |
| Clauses | — | **1,955** | new granularity |
| Provisos | 352 | **242** | normalized (less noise) |
| Rules (English) | 22 usable | **56 usable** | +155% |
| Rules (Hindi) | 0 (discarded) | **38** | from zero |
| Total rules | 22 | **94** | +327% |
| Amendment entries | 0 | **102** | from zero |
| Estimated graph nodes | ~1,500 | **~4,173** | +178% |

---

## 2. Source Documents

| Document | PDF Pages | Raw Elements | Status |
|---|---|---|---|
| Companies Act, 2013 | 350+ | 149,247 JSON lines | Fully parsed |
| Companies Rules, 2014 | ~40 | 1,708 kids / 3,560 flat elements | Bilingual (EN+HI) |
| Corporate Laws (Amendment) Act, 2026 | — | — | 102 entries parsed |

---

## 3. Pipeline Architecture

```
PDF files
  │
  ▼
OpenDataLoader (Java) ──► Raw JSON (elements with type, font, content, page)
  │
  ▼
extractor.py ─────────► Flat element list (flatten_elements)
  │                      • Fixed: list items with empty kids now yielded
  │                      • Fixed: elements with content but empty kids now yielded
  ▼
act_parser.py ────────► Act model (Pydantic)
rules_parser.py ──────► RuleSet model       ← Hindi content now flows through
amendment_llm_parser.py► AmendmentAct model
  │
  ▼
preprocessor.py ──────► Cleaned + Scored models
  │                      • TextNormalizer (encoding, Devanagari cleaning)
  │                      • QualityScorer (bilingual scoring, Hindi legal terms)
  │                      • LLMFormatter (structural context headers)
  │                      • Language detection (en / hi / mixed)
  ▼
graph/loader.py ──────► Neo4j nodes with: text, quality_score, llm_text, language
  │
  ▼
query/preprocessor.py ► Bilingual query expansion (EN + HI abbreviations)
```

---

## 4. Companies Act, 2013 — Detailed Breakdown

**Coverage:** 29 chapters, 470 sections (100%)

| Node Type | Count | Notes |
|---|---|---|
| Chapters | 29 | All extracted |
| Sections | 470 | 100% coverage (was 269) |
| SubSections | 1,157 | Major increase from list-item fix |
| Clauses + SubClauses | 1,955 | Enabled by extractor fix |
| Provisos | 242 | Cleaned (was inflated by footnotes) |
| Explanations | 46 | — |
| Definitions | 70 | Extracted from Section 2 (was 0) |
| Schedules | 0 | Not yet parsed |
| Cross-references tagged | 4 | Cross-ref detector active |

### Quality Distribution (470 sections)

| Score Range | Count | % |
|---|---|---|
| 0–20 | 0 | 0% |
| 21–40 | 31 | 6.6% |
| 41–60 | 84 | 17.9% |
| 61–80 | 355 | 75.5% |
| 81–100 | 0 | 0% |

- **Average quality score:** 66.2 / 100
- **Empty sections:** 0
- **Short sections (<20 chars):** 25 (typically section stubs referencing rules)
- **Oversized sections (>5K chars):** 0 (footnote separation working)

### Sample Definitions Extracted (70 total)

`abridged prospectus` · `articles` · `banking company` · `books of account` · `charge` · `chartered accountant` · `chief executive officer` · `chief financial officer` · `company limited by shares` · `company liquidator` · `company secretary in practice` · `contributory` · `control` · `cost accountant` · `court` · `debenture` · `deposit` · `director` · `expert` · `financial year` · `holding company` · `key managerial personnel` · `managing director` · `member` · `net worth` · `officer` · `one person company` · `paid-up share capital` · `prescribed` · `promoter` · `public company` · `related party` · `share` · `subsidiary company` · `turnover` · `voting right` · and 34 more

---

## 5. Companies Rules, 2014 — Bilingual Breakdown

**Total rules kept: 94** (from 249 parsed, 155 filtered)

| Category | Count | Avg Quality | Notes |
|---|---|---|---|
| English rules | 56 | 68.1 | Same count as pre-Hindi work |
| Hindi/mixed rules | 38 | 47.8 | Previously all discarded |
| Garbled (dropped) | 40 | — | Font encoding corruption |
| Empty (dropped) | 85 | — | No text content |
| Low quality (dropped) | 30 | — | Fragments, short stubs |
| Forms | 5 | — | Linked to rules |

**Why Hindi quality is lower (47.8 vs 68.1):** The bilingual PDF uses non-standard Chanakya fonts for some Hindi content. OpenDataLoader extracts the Unicode Devanagari correctly for ~137 elements, but some have layout artifacts (doubled words, tripled parentheses from column-merge). The `clean_devanagari()` function mitigates this but cannot fully reconstruct missing characters.

---

## 6. Amendment Act, 2026

| Metric | Value |
|---|---|
| Entries parsed | 102 |
| Entries with text | 0 (structural entries only) |

The amendment entries are structural markers (section number + amendment type). Full amendment text reconstruction requires cross-referencing with the Act sections.

---

## 7. Root Causes Found and Fixed

| # | Issue | Root Cause | Fix | Impact |
|---|---|---|---|---|
| 1 | **Missing sections (269/470)** | `flatten_elements()` silently dropped `list item` elements with content but empty `kids` arrays | Yield element content as paragraph when `kids` is empty | +201 sections |
| 2 | **Zero definitions** | Definition terms live inside list items that were being dropped; also no post-parse extraction | Extractor fix + `extract_definitions_from_section2()` | +70 definitions |
| 3 | **Subsection count low** | TOC entries parsed as full sections, blocking real sections via `seen_section_numbers` | Content-length comparison: replace shorter TOC entries with substantive sections | +866 subsections |
| 4 | **All Hindi text discarded** | `rules_parser.py` had `continue` on `is_devanagari_majority()`; `QualityScorer` had 60-point penalty; `preprocess_ruleset` had explicit Devanagari skip | Removed all three barriers | +38 Hindi rules |
| 5 | **Hindi text flagged as garbled** | Python `isalpha()` returns False for Devanagari vowel signs (matras); low alpha ratio triggered garbled detection | `_is_content_char()` counts full Devanagari Unicode block | Correct classification |
| 6 | **Rule number filter bug** | `-dup-N` suffix in rule numbers like `1-dup-207` stripped to `1207`, exceeding 500 threshold | Extract base number before `-dup-` suffix | Rules no longer incorrectly dropped |

---

## 8. New Capabilities Added

### Devanagari Text Pipeline

- `TextNormalizer.clean_devanagari()` — fixes doubled words, repeated char sequences, tripled parentheses
- `TextNormalizer.detect_language()` — returns `"en"` / `"hi"` / `"mixed"`
- `TextNormalizer._is_content_char()` — Devanagari-aware content detection
- `BaseNode.language` field — persisted on all Neo4j nodes
- `_HINDI_LEGAL_TERMS_RE` — 18 Hindi legal terms for quality scoring (धारा, कंपनी, निदेशक, etc.)

### Bilingual Query Support

- Hindi abbreviation expansion: `सीएसआर` → `कॉर्पोरेट सामाजिक उत्तरदायित्व (धारा 135)`
- Hindi-to-English annotation: `धारा 135 क्या है?` → `धारा (section) 135 क्या है?`
- Automatic query language detection

---

## 9. Neo4j Graph Estimate

| Metric | Value | Limit (Aura Free) | Usage |
|---|---|---|---|
| Nodes | ~4,173 | 200,000 | 2.1% |
| Relationships | ~4,174 | 400,000 | 1.0% |

Well within Neo4j Aura Free tier limits.

---

## 10. Known Remaining Gaps

| ID | Severity | Issue | Notes |
|---|---|---|---|
| GAP-01 | Medium | ~25 definitions still missing (70/~95 expected) | Non-standard quote characters or formatting in source |
| GAP-02 | Low | Hindi rule text has residual artifacts (`M+`, `<+k`) | Chanakya font glyphs that didn't map to Unicode |
| GAP-03 | Low | Rules PDF font encoding is an upstream issue | OpenDataLoader can't fix non-Unicode fonts |
| GAP-04 | Low | Schedules not yet parsed from Act | Parser doesn't extract Schedule content |
| GAP-05 | Low | Amendment entries have no body text | Structural-only; need cross-ref to Act |
| GAP-06 | Low | Cross-reference tagging coverage is low (4 sections) | Detector exists but pattern matching needs expansion |
| GAP-07 | Info | Hindi text quality averages 47.8 vs 68.1 for English | Expected given PDF extraction quality difference |

---

## 11. Files Modified

| File | Lines | What Changed |
|---|---|---|
| `ingestion/preprocessor.py` | 670 | Devanagari cleaning, bilingual scoring, language detection, removed Hindi penalty |
| `ingestion/rules_parser.py` | 332 | Hindi content flows through instead of being skipped |
| `ingestion/models.py` | 320 | Added `language` field to `BaseNode` |
| `ingestion/extractor.py` | 250 | Fixed list-item content drop (root cause of missing data) |
| `ingestion/act_parser.py` | 613 | Relaxed section regex, duplicate handling, definition regex |
| `graph/loader.py` | 709 | Language property on Section, SubSection, Definition, Rule nodes |
| `query/preprocessor.py` | 92 | Hindi abbreviations, term annotation, language detection |

---

## 12. How to Re-Ingest

Since parser and preprocessor code was changed, you **must re-run the full ingestion pipeline** to update the parsed JSON files and reload the Neo4j graph with the new data (including Hindi rules, language tags, and fixed section coverage).

### Option A: API endpoint (if backend server is running)

```bash
curl -X POST http://localhost:8000/api/ingest
```

This triggers `run_ingestion()` (PDF extraction + parsing) followed by `run_graph_population()` (Neo4j loading). Takes ~5-15 minutes.

### Option B: CLI — Full pipeline (recommended)

```bash
cd backend

# Step 1: Re-extract and re-parse all PDFs → data/parsed/*.json
python -m ingestion.runner

# Step 2: Load parsed data into Neo4j graph
python -m graph
```

### Option C: CLI — Graph only (if parsed JSON is already updated)

If you already re-parsed (or the parsed JSON files are current from this session's work), you can skip Step 1 and just reload the graph:

```bash
cd backend
python -m graph
```

> **Note:** The Rules parsed JSON (`data/parsed/companies_rules_2014_parsed.json`) was already re-generated during this session with Hindi support. The Act parsed JSON was generated in the previous session. To get a fully consistent dataset, run the **full pipeline** (Option B).
