#!/bin/bash
# Complex QA Test Suite for ViddhiAI
# Tests 4 categories: Current Version, Amendment History, Applicable Rules, Structured Explanation

BASE_URL="${API_URL:-http://localhost:8000}"
ENDPOINT="$BASE_URL/api/query"
PASS=0
FAIL=0
TIMEOUT=0
TOTAL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

query() {
  local id="$1"
  local category="$2"
  local question="$3"
  local check_field="$4"   # field to verify exists in response
  local check_contains="$5" # substring expected in answer

  TOTAL=$((TOTAL + 1))
  printf "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
  printf "${BOLD}[%s] %s${NC}\n" "$id" "$category"
  printf "${YELLOW}Q: %s${NC}\n" "$question"

  RESPONSE=$(curl -sS --max-time 90 -X POST "$ENDPOINT" \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"$question\"}" 2>&1)

  EXIT_CODE=$?
  if [ $EXIT_CODE -ne 0 ]; then
    printf "${RED}  ✗ TIMEOUT/ERROR (curl exit code: %d)${NC}\n" "$EXIT_CODE"
    TIMEOUT=$((TIMEOUT + 1))
    return
  fi

  # Check for error response
  ERROR=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('detail',{}).get('error',''))" 2>/dev/null)
  if [ -n "$ERROR" ] && [ "$ERROR" != "" ]; then
    printf "${RED}  ✗ API ERROR: %s${NC}\n" "$ERROR"
    FAIL=$((FAIL + 1))
    return
  fi

  # Extract fields
  ANSWER=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('answer','')[:300])" 2>/dev/null)
  CYPHER=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('cypher_query','')[:200])" 2>/dev/null)
  SOURCES=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('sources',[])))" 2>/dev/null)
  GRAPH_PATH=$(echo "$RESPONSE" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('graph_path',[])))" 2>/dev/null)

  printf "${GREEN}  Answer (first 300 chars): %s${NC}\n" "$ANSWER"
  printf "  Cypher: %s\n" "$CYPHER"
  printf "  Sources: %s | Graph Path nodes: %s\n" "$SOURCES" "$GRAPH_PATH"

  # Validate
  if [ -n "$check_contains" ]; then
    if echo "$ANSWER" | grep -qi "$check_contains"; then
      printf "${GREEN}  ✓ PASS — contains '%s'${NC}\n" "$check_contains"
      PASS=$((PASS + 1))
    else
      printf "${RED}  ✗ FAIL — expected '%s' in answer${NC}\n" "$check_contains"
      FAIL=$((FAIL + 1))
    fi
  else
    if [ -n "$ANSWER" ] && [ "$ANSWER" != "None" ]; then
      printf "${GREEN}  ✓ PASS — got a non-empty answer${NC}\n"
      PASS=$((PASS + 1))
    else
      printf "${RED}  ✗ FAIL — empty answer${NC}\n"
      FAIL=$((FAIL + 1))
    fi
  fi
}

echo ""
printf "${BOLD}╔══════════════════════════════════════════════════════════════════╗${NC}\n"
printf "${BOLD}║     ViddhiAI Complex QA Test Suite — 4 Categories, 20 Qs       ║${NC}\n"
printf "${BOLD}╚══════════════════════════════════════════════════════════════════╝${NC}\n"
printf "Target: %s\n" "$ENDPOINT"

# ═══════════════════════════════════════════════════════════════════
# CATEGORY 1: CURRENT VERSION OF A SECTION
# ═══════════════════════════════════════════════════════════════════
printf "\n${BOLD}▶ CATEGORY 1: Current Version of a Section${NC}\n"

query "CQ-001" "CURRENT_VERSION" \
  "What is the current version of Section 135 after all amendments, including the substituted and inserted subsections? List each subsection with its current text." \
  "answer" "135"

query "CQ-002" "CURRENT_VERSION" \
  "Show me the current consolidated text of Section 149 on appointment of directors, identifying which subsections were substituted or inserted by the Amendment Act versus the original 2013 text." \
  "answer" "director"

query "CQ-003" "CURRENT_VERSION" \
  "What is the current operative version of Section 185 on loans to directors? Distinguish between the original provision, the substituted version, and any provisos that were added or modified." \
  "answer" "185"

query "CQ-004" "CURRENT_VERSION" \
  "Give me the current text of Section 2(68) defining private company. Has the threshold for number of members or paid-up share capital been changed by any amendment?" \
  "answer" "private company"

query "CQ-005" "CURRENT_VERSION" \
  "What is the current version of Section 12 on registered office after incorporating all amendments? Include any provisos or explanations that were inserted or substituted." \
  "answer" "registered office"

# ═══════════════════════════════════════════════════════════════════
# CATEGORY 2: AMENDMENTS AFFECTING A PROVISION
# ═══════════════════════════════════════════════════════════════════
printf "\n${BOLD}▶ CATEGORY 2: Amendments Affecting a Provision${NC}\n"

query "CQ-006" "AMENDMENT_HISTORY" \
  "Trace the complete amendment history of Section 135 on CSR. For each amendment entry, specify whether it was a substitution, insertion, or omission, which subsection was affected, and what the amendment changed." \
  "answer" "amend"

query "CQ-007" "AMENDMENT_HISTORY" \
  "Which specific subsections of Section 149 were substituted or inserted by the Companies (Amendment) Act? For each change, show the original text versus the amended text." \
  "answer" "149"

query "CQ-008" "AMENDMENT_HISTORY" \
  "List all provisions across the Companies Act 2013 that were omitted by the Amendment Act. For each omitted provision, state which section and subsection it belonged to." \
  "answer" "omit"

query "CQ-009" "AMENDMENT_HISTORY" \
  "How many amendment entries affect Chapter VIII (Declaration and Payment of Dividend)? Break down by type: substitutions, insertions, and omissions." \
  "answer" ""

query "CQ-010" "AMENDMENT_HISTORY" \
  "Compare the amendment impact on Section 185 (Loans to Directors) versus Section 188 (Related Party Transactions). Which section had more amendments, and what was the nature of each change?" \
  "answer" ""

# ═══════════════════════════════════════════════════════════════════
# CATEGORY 3: APPLICABLE RULES UNDER A SECTION
# ═══════════════════════════════════════════════════════════════════
printf "\n${BOLD}▶ CATEGORY 3: Applicable Rules Under a Section${NC}\n"

query "CQ-011" "APPLICABLE_RULES" \
  "Which rules from the Companies (Corporate Social Responsibility Policy) Rules 2014 are applicable under Section 135? List each rule number and its subject matter." \
  "answer" "rule"

query "CQ-012" "APPLICABLE_RULES" \
  "For Section 149 on appointment of directors, which specific rules prescribe the qualifications, disqualifications, and procedures? Map each rule to the subsection it implements." \
  "answer" "rule"

query "CQ-013" "APPLICABLE_RULES" \
  "What rules govern the compliance requirements under Section 12 (Registered Office)? Include any forms that are prescribed by these rules." \
  "answer" ""

query "CQ-014" "APPLICABLE_RULES" \
  "List all rules that are derived from or reference Section 185 and Section 186 together. Do any rules create additional compliance obligations beyond what the sections themselves require?" \
  "answer" ""

query "CQ-015" "APPLICABLE_RULES" \
  "Which rules under the Companies (Meetings of Board and its Powers) Rules apply to Section 173 on board meetings and Section 174 on quorum? Show the rule-to-section mapping." \
  "answer" ""

# ═══════════════════════════════════════════════════════════════════
# CATEGORY 4: STRUCTURED EXPLANATION WITH CONTEXT
# ═══════════════════════════════════════════════════════════════════
printf "\n${BOLD}▶ CATEGORY 4: Structured Explanation with Context${NC}\n"

query "CQ-016" "STRUCTURED_EXPLANATION" \
  "Provide a structured explanation of Section 135 (CSR) including: (a) which Chapter it belongs to, (b) all subsections and provisos, (c) applicable rules from the CSR Rules, (d) any amendments that modified it, and (e) cross-references to other sections." \
  "answer" "135"

query "CQ-017" "STRUCTURED_EXPLANATION" \
  "Explain Section 149 in full context: the Chapter hierarchy, each subsection on independent directors, the rules prescribing qualifications, any amendments, and sections that reference Section 149 via REFERS_TO or SUBJECT_TO relationships." \
  "answer" "149"

query "CQ-018" "STRUCTURED_EXPLANATION" \
  "Give a complete legal map of related party transactions: start from the definition of related party in Section 2(76), trace to Section 188 on related party transactions, include all applicable rules, amendments, penalties for non-compliance, and any cross-references." \
  "answer" ""

query "CQ-019" "STRUCTURED_EXPLANATION" \
  "Explain the full lifecycle of a directors appointment under the Companies Act: starting from the definition in Section 2(34), through eligibility under Section 149, appointment process in Section 152, disqualifications in Section 164, and removal in Section 169. Include applicable rules and any amendments for each section." \
  "answer" "director"

query "CQ-020" "STRUCTURED_EXPLANATION" \
  "Map the complete penalty framework for Chapter IX (Accounts of Companies): for each section in this chapter, identify the corresponding penalty provision, the amount of fine, whether imprisonment applies, and if any of these penalties were amended." \
  "answer" ""

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
printf "\n${BOLD}╔══════════════════════════════════════════════════════════════════╗${NC}\n"
printf "${BOLD}║                        TEST SUMMARY                             ║${NC}\n"
printf "${BOLD}╚══════════════════════════════════════════════════════════════════╝${NC}\n"
printf "  Total:    %d\n" "$TOTAL"
printf "  ${GREEN}Passed:   %d${NC}\n" "$PASS"
printf "  ${RED}Failed:   %d${NC}\n" "$FAIL"
printf "  ${YELLOW}Timeout:  %d${NC}\n" "$TIMEOUT"
printf "\n"
