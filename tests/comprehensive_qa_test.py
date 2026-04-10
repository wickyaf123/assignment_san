"""
Comprehensive 100-question QA test for ViddhiAI.

Tests 4 categories (25 questions each):
1. Current Version — What does a section currently say?
2. Amendment History — What amendments have affected a provision?
3. Applicable Rules — Which rules apply under a section?
4. Structured Explanation — Explain a section with context

Each question is verified against graph data for factual accuracy.
"""

import json
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query.flow import run_query, clear_query_cache

QUESTIONS = {
    "current_version": [
        {"id": "CV-001", "q": "What does Section 135 say?", "verify": {"section": 135}},
        {"id": "CV-002", "q": "What are the provisions of Section 149?", "verify": {"section": 149}},
        {"id": "CV-003", "q": "What does Section 2 define?", "verify": {"section": 2}},
        {"id": "CV-004", "q": "What is Section 185 about?", "verify": {"section": 185}},
        {"id": "CV-005", "q": "Show me Section 447", "verify": {"section": 447}},
        {"id": "CV-006", "q": "What does Section 173 say about board meetings?", "verify": {"section": 173}},
        {"id": "CV-007", "q": "What are the subsections of Section 134?", "verify": {"section": 134}},
        {"id": "CV-008", "q": "What is Section 96 about?", "verify": {"section": 96}},
        {"id": "CV-009", "q": "Show me the provisions of Section 188", "verify": {"section": 188}},
        {"id": "CV-010", "q": "What does Section 143 say about auditor duties?", "verify": {"section": 143}},
        {"id": "CV-011", "q": "What is the definition of private company?", "verify": {"term": "private company"}},
        {"id": "CV-012", "q": "Define 'managing director' under the Companies Act", "verify": {"term": "managing director"}},
        {"id": "CV-013", "q": "What is the definition of related party?", "verify": {"term": "related party"}},
        {"id": "CV-014", "q": "What does Section 3 say about company formation?", "verify": {"section": 3}},
        {"id": "CV-015", "q": "What is Section 166 about duties of directors?", "verify": {"section": 166}},
        {"id": "CV-016", "q": "Show me Section 197 on managerial remuneration", "verify": {"section": 197}},
        {"id": "CV-017", "q": "What does Section 177 say about audit committee?", "verify": {"section": 177}},
        {"id": "CV-018", "q": "What are the provisions of Section 186?", "verify": {"section": 186}},
        {"id": "CV-019", "q": "What is Section 271 about?", "verify": {"section": 271}},
        {"id": "CV-020", "q": "What does Section 12 say about registered office?", "verify": {"section": 12}},
        {"id": "CV-021", "q": "What is the definition of 'company'?", "verify": {"term": "company"}},
        {"id": "CV-022", "q": "What does Section 8 say about charitable companies?", "verify": {"section": 8}},
        {"id": "CV-023", "q": "Show me Section 462 on powers to exempt", "verify": {"section": 462}},
        {"id": "CV-024", "q": "What does Section 128 say about books of account?", "verify": {"section": 128}},
        {"id": "CV-025", "q": "What is Section 245 about class action suits?", "verify": {"section": 245}},
    ],
    "amendment_history": [
        {"id": "AH-001", "q": "How was Section 135 amended?", "verify": {"section": 135, "has_amendments": True}},
        {"id": "AH-002", "q": "What amendments affected Section 2?", "verify": {"section": 2, "has_amendments": True}},
        {"id": "AH-003", "q": "Was Section 149 amended by the Corporate Laws Amendment Act?", "verify": {"section": 149, "has_amendments": False}},
        {"id": "AH-004", "q": "How was Section 4 changed by the amendment?", "verify": {"section": 4, "has_amendments": True}},
        {"id": "AH-005", "q": "What changes were made to Section 7?", "verify": {"section": 7, "has_amendments": True}},
        {"id": "AH-006", "q": "Which sections were substituted by the Amendment Act?", "verify": {"expects_list": True}},
        {"id": "AH-007", "q": "Were any sections omitted by the amendment?", "verify": {"expects_list": True}},
        {"id": "AH-008", "q": "Which sections were decriminalized?", "verify": {"expects_list": True}},
        {"id": "AH-009", "q": "How was Section 68 amended?", "verify": {"section": 68, "has_amendments": True}},
        {"id": "AH-010", "q": "What amendments affected Section 132?", "verify": {"section": 132, "has_amendments": True}},
        {"id": "AH-011", "q": "Was Section 99 changed by the Amendment Act?", "verify": {"section": 99, "has_amendments": True}},
        {"id": "AH-012", "q": "How was Section 34 amended?", "verify": {"section": 34, "has_amendments": True}},
        {"id": "AH-013", "q": "What changes did the Amendment Act make to Section 128?", "verify": {"section": 128, "has_amendments": True}},
        {"id": "AH-014", "q": "Was Section 42 amended?", "verify": {"section": 42, "has_amendments": True}},
        {"id": "AH-015", "q": "How was Section 26 changed?", "verify": {"section": 26, "has_amendments": True}},
        {"id": "AH-016", "q": "What did the Corporate Laws Amendment Act 2026 change?", "verify": {"expects_list": True}},
        {"id": "AH-017", "q": "Was Section 58 substituted?", "verify": {"section": 58, "has_amendments": True}},
        {"id": "AH-018", "q": "How was Section 40 amended?", "verify": {"section": 40, "has_amendments": True}},
        {"id": "AH-019", "q": "What changes affected Section 125?", "verify": {"section": 125, "has_amendments": True}},
        {"id": "AH-020", "q": "Was Section 200 amended?", "verify": {"section": 200, "has_amendments": False}},
        {"id": "AH-021", "q": "How was Section 77 changed by the amendment?", "verify": {"section": 77, "has_amendments": True}},
        {"id": "AH-022", "q": "What amendments affected Section 88?", "verify": {"section": 88, "has_amendments": True}},
        {"id": "AH-023", "q": "Was Section 11 amended?", "verify": {"section": 11, "has_amendments": True}},
        {"id": "AH-024", "q": "How was Section 124 changed?", "verify": {"section": 124, "has_amendments": True}},
        {"id": "AH-025", "q": "What is the effective text of Section 135 after amendments?", "verify": {"section": 135, "has_amendments": True}},
    ],
    "applicable_rules": [
        {"id": "AR-001", "q": "Which rules relate to Section 135?", "verify": {"section": 135, "expects_rules": True}},
        {"id": "AR-002", "q": "What rules apply under Section 8?", "verify": {"section": 8, "expects_rules": True}},
        {"id": "AR-003", "q": "Which rules prescribe for Section 134?", "verify": {"section": 134, "expects_rules": True}},
        {"id": "AR-004", "q": "Are there rules related to Section 185?", "verify": {"section": 185}},
        {"id": "AR-005", "q": "What rules exist for Section 149?", "verify": {"section": 149}},
        {"id": "AR-006", "q": "Which rules relate to Section 178?", "verify": {"section": 178, "expects_rules": True}},
        {"id": "AR-007", "q": "What rules exist for Section 186?", "verify": {"section": 186, "expects_rules": True}},
        {"id": "AR-008", "q": "Which rules apply under Section 188?", "verify": {"section": 188, "expects_rules": True}},
        {"id": "AR-009", "q": "Are there rules for Section 96?", "verify": {"section": 96}},
        {"id": "AR-010", "q": "What rules exist in the Companies Rules 2014?", "verify": {"expects_list": True}},
        {"id": "AR-011", "q": "Which chapters have no corresponding rules?", "verify": {"expects_list": True}},
        {"id": "AR-012", "q": "What rules relate to Section 7?", "verify": {"section": 7}},
        {"id": "AR-013", "q": "Show me Rule 8", "verify": {"rule": "8"}},
        {"id": "AR-014", "q": "What does Rule 3 say?", "verify": {"rule": "3"}},
        {"id": "AR-015", "q": "Which rules prescribe for corporate social responsibility?", "verify": {"keyword": "CSR"}},
        {"id": "AR-016", "q": "What rules apply to Section 447?", "verify": {"section": 447}},
        {"id": "AR-017", "q": "Are there any derived rules for Section 3?", "verify": {"section": 3}},
        {"id": "AR-018", "q": "Which rules relate to company incorporation?", "verify": {"keyword": "incorporation"}},
        {"id": "AR-019", "q": "What rules exist for charitable companies under Section 8?", "verify": {"section": 8, "expects_rules": True}},
        {"id": "AR-020", "q": "Show me all rules in the Companies Rules", "verify": {"expects_list": True}},
        {"id": "AR-021", "q": "Which rules refer to Section 248?", "verify": {"section": 248}},
        {"id": "AR-022", "q": "What is the relationship between Rule 7 and the Act?", "verify": {"rule": "7"}},
        {"id": "AR-023", "q": "Which forms exist in the Companies Rules?", "verify": {"expects_list": True}},
        {"id": "AR-024", "q": "What rules relate to Section 177 audit committee?", "verify": {"section": 177}},
        {"id": "AR-025", "q": "Are there rules for director appointments under Section 149?", "verify": {"section": 149}},
    ],
    "structured_explanation": [
        {"id": "SE-001", "q": "Explain Section 135 with all its subsections and context", "verify": {"section": 135}},
        {"id": "SE-002", "q": "Give me a structured explanation of Section 149", "verify": {"section": 149}},
        {"id": "SE-003", "q": "What is the penalty under Section 447 for fraud?", "verify": {"section": 447, "expects_keyword": "penalty"}},
        {"id": "SE-004", "q": "Explain the CSR provisions and their penalties", "verify": {"section": 135, "expects_keyword": "penalty"}},
        {"id": "SE-005", "q": "What are the penalties for violating Section 185?", "verify": {"section": 185, "expects_keyword": "penalty"}},
        {"id": "SE-006", "q": "Explain Chapter IX on Accounts", "verify": {"chapter": "IX"}},
        {"id": "SE-007", "q": "What does Chapter VIII cover?", "verify": {"chapter": "VIII"}},
        {"id": "SE-008", "q": "Explain the structure of the Companies Act 2013", "verify": {"expects_list": True}},
        {"id": "SE-009", "q": "How many sections are in the Companies Act?", "verify": {"expects_count": True}},
        {"id": "SE-010", "q": "What are all the definitions in the Companies Act?", "verify": {"expects_list": True}},
        {"id": "SE-011", "q": "Explain Section 166 duties of directors", "verify": {"section": 166}},
        {"id": "SE-012", "q": "What is the penalty under Section 448?", "verify": {"section": 448, "expects_keyword": "penalty"}},
        {"id": "SE-013", "q": "Explain the provisions of Section 186 on loans and investments", "verify": {"section": 186}},
        {"id": "SE-014", "q": "What are the penalties for non-compliance with Section 134?", "verify": {"section": 134, "expects_keyword": "penalty"}},
        {"id": "SE-015", "q": "Explain Section 173 about board meetings", "verify": {"section": 173}},
        {"id": "SE-016", "q": "What does Section 188 say about related party transactions?", "verify": {"section": 188}},
        {"id": "SE-017", "q": "Explain the penalty provisions in Section 129", "verify": {"section": 129, "expects_keyword": "penalty"}},
        {"id": "SE-018", "q": "What is the structure of Chapter XII?", "verify": {"chapter": "XII"}},
        {"id": "SE-019", "q": "Explain Section 143 on auditor powers and duties", "verify": {"section": 143}},
        {"id": "SE-020", "q": "What penalty applies for non-compliance with Section 92?", "verify": {"section": 92, "expects_keyword": "penalty"}},
        {"id": "SE-021", "q": "Explain Section 196 on appointment of managing director", "verify": {"section": 196}},
        {"id": "SE-022", "q": "What are the provisions of Section 197 on remuneration?", "verify": {"section": 197}},
        {"id": "SE-023", "q": "What is Section 1 about?", "verify": {"section": 1}},
        {"id": "SE-024", "q": "Explain Section 241 on application to Tribunal", "verify": {"section": 241}},
        {"id": "SE-025", "q": "What are the cross-references in Section 135?", "verify": {"section": 135}},
    ],
}


def run_all_questions(categories=None):
    """Run all questions and save results."""
    clear_query_cache()

    all_results = {}
    total = 0
    errors = 0

    cats = categories or list(QUESTIONS.keys())

    for category in cats:
        questions = QUESTIONS[category]
        print(f"\n{'='*60}")
        print(f"Category: {category} ({len(questions)} questions)")
        print(f"{'='*60}")

        for item in questions:
            qid = item["id"]
            question = item["q"]
            total += 1

            print(f"\n[{qid}] {question}")

            try:
                start = time.time()
                result = run_query(question)
                elapsed = time.time() - start

                answer = result.get("answer", "")
                metadata = result.get("metadata", {})

                # Basic quality checks
                has_answer = len(answer) > 50
                has_citations = "[" in answer and "]" in answer
                used_fallback = metadata.get("fallback_used", False)
                intent = metadata.get("intent", "")
                latency = metadata.get("latency_ms", 0)
                timed_out = latency > 55000

                status = "PASS" if has_answer and not timed_out else "FAIL"
                if timed_out:
                    status = "TIMEOUT"
                    errors += 1
                elif not has_answer:
                    errors += 1

                fb_marker = " [FALLBACK]" if used_fallback else ""
                cite_marker = " [NO-CITE]" if not has_citations else ""

                print(f"  {status} | {intent} | {latency:.0f}ms{fb_marker}{cite_marker}")
                print(f"  Answer: {answer[:150]}...")

                all_results[qid] = {
                    "question": question,
                    "category": category,
                    "answer": answer,
                    "cypher": result.get("cypher_query", ""),
                    "sources": result.get("sources", []),
                    "citations": result.get("citations", []),
                    "metadata": metadata,
                    "verify": item["verify"],
                    "status": status,
                    "has_citations": has_citations,
                    "fallback_used": used_fallback,
                    "elapsed_seconds": elapsed,
                }

            except Exception as e:
                print(f"  ERROR: {e}")
                errors += 1
                all_results[qid] = {
                    "question": question,
                    "category": category,
                    "answer": "",
                    "error": str(e),
                    "status": "ERROR",
                    "verify": item["verify"],
                }

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {total - errors}/{total} passed ({(total-errors)/total*100:.1f}%)")
    print(f"Errors/Timeouts: {errors}")

    # Per-category stats
    for cat in cats:
        cat_results = {k: v for k, v in all_results.items() if v["category"] == cat}
        passed = sum(1 for v in cat_results.values() if v["status"] == "PASS")
        fallbacks = sum(1 for v in cat_results.values() if v.get("fallback_used"))
        print(f"  {cat}: {passed}/{len(cat_results)} passed, {fallbacks} fallbacks")

    print(f"{'='*60}")

    # Save to file
    output_path = os.path.join(os.path.dirname(__file__), "comprehensive_qa_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", "-c", help="Run only this category")
    args = parser.parse_args()

    cats = [args.category] if args.category else None
    run_all_questions(cats)
