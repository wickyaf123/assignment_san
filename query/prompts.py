"""
Prompt templates and few-shot examples for the ViddhiAI query engine.

All prompt constants are module-level strings so they can be imported
directly into LangGraph nodes without any runtime assembly cost.

Design choices (per 03-CONTEXT.md):
- D-01: Per-intent few-shot pools — examples are selected by intent after classification
- D-02: Schema injection at generation time — SCHEMA_TEMPLATE rendered with live node counts
- D-03: Explicit type annotations in schema (Section.number is STRING, not INT)
- D-04: Synthesis prompt forbids fact generation — LLM only formats graph data
- D-07: Fulltext index covers Section.text, Section.title, Definition.term, Rule.text
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Base system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = (
    "You are ViddhiAI, a legal intelligence system. "
    "You translate natural language questions about Indian corporate law into Cypher queries "
    "against a Neo4j knowledge graph. "
    "You NEVER generate facts — you only format graph data into readable prose. "
    "Every answer must be traceable to graph nodes. "
    "IMPORTANT: If the user query contains obvious spelling mistakes (e.g. 'direcor' → 'director', "
    "'penalti' → 'penalty', 'amendmant' → 'amendment'), silently correct them before generating "
    "Cypher. Always use correctly spelled legal terms in property value matches like WHERE, "
    "CONTAINS, and equality checks."
)

# ---------------------------------------------------------------------------
# Intent classification prompt
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT: str = """\
Classify the following legal query into exactly one of these 5 intent categories:

- structured_lookup: Questions about specific section content, definitions, or rules \
(e.g., 'What does Section 135 say?')
- amendment_query: Questions about how a section was changed, amended, or its history \
(e.g., 'How was Section 135 amended?')
- cross_reference: Questions about relationships between sections, references, or dependencies \
(e.g., 'Which sections refer to Section 135?')
- penalty_query: Questions about penalties, fines, or punishment under specific sections \
(e.g., 'What is the penalty under Section 447?')
- general_info: Broad questions about the Act, chapters, or general legal concepts \
(e.g., 'What does Chapter IX cover?')

Respond with ONLY the category name, nothing else.

Query: {query}
"""

# ---------------------------------------------------------------------------
# Schema template — injected into Cypher generation prompt at startup.
# {node_counts} is replaced with live counts from Neo4j after graph population.
# ---------------------------------------------------------------------------

SCHEMA_TEMPLATE: str = """\
Node types and properties:
- Act {{ uid: STRING, title: STRING, year: INT }}
- Chapter {{ uid: STRING, number: STRING, title: STRING }}
- Section {{ uid: STRING, number: INT, title: STRING, text: STRING }}
- SubSection {{ uid: STRING, number: INT, text: STRING }}
- Clause {{ uid: STRING, label: STRING, text: STRING }}
- SubClause {{ uid: STRING, label: STRING, text: STRING }}
- Proviso {{ uid: STRING, text: STRING }}
- Explanation {{ uid: STRING, text: STRING }}
- Definition {{ uid: STRING, term: STRING, text: STRING }}
- Rule {{ uid: STRING, number: STRING, text: STRING }}
- RuleSet {{ uid: STRING, title: STRING }}
- Schedule {{ uid: STRING, number: STRING, text: STRING }}
- Form {{ uid: STRING, form_number: STRING, text: STRING }}
- AmendmentAct {{ uid: STRING, title: STRING, year: INT }}

Relationship types:
- Structural: HAS_CHAPTER, HAS_SECTION, HAS_SUBSECTION, HAS_CLAUSE, HAS_SUBCLAUSE, \
HAS_PROVISO, HAS_EXPLANATION, HAS_DEFINITION, HAS_SCHEDULE, HAS_RULE, HAS_FORM
- Amendment: SUBSTITUTES, INSERTS, OMITS, DECRIMINALIZES \
(from AmendmentAct to Section, with properties: effective_date DATE, new_text STRING, old_text STRING) \
DECRIMINALIZES is used when a criminal penalty is replaced with a civil penalty or compoundable offence
- Cross-reference: REFERS_TO, SUBJECT_TO, NOTWITHSTANDING, PRESCRIBES_FOR, DERIVED_RULE \
(these edges exist from Section, SubSection, Clause, Proviso, Explanation, Definition, Rule, and Form nodes to Section nodes)
- Derived legislation: (Rule)-[:DERIVED_RULE]->(Section) links a rule to the section it is framed under
- Subordinate legislation: (RuleSet)-[:PRESCRIBES_FOR]->(Act) links the ruleset to the parent act

Known node identifiers:
- The Act node has uid: 'act-companies-2013', title: 'The Companies Act, 2013', year: 2013
- The RuleSet node has uid: 'ruleset-companies-rules', title: 'The Companies Rules, 2014'
- The AmendmentAct node has uid: 'amend-corporate-laws-2026', title: 'The Corporate Laws (Amendment) Act, 2026', year: 2026
- To match the Act: MATCH (a:Act {{uid: 'act-companies-2013'}}) — use uid for reliability
- To match the RuleSet: MATCH (rs:RuleSet {{uid: 'ruleset-companies-rules'}}) — use uid for reliability

CRITICAL — Definition.term values:
- Definition.term is stored in LOWERCASE multi-word form (e.g. 'private company', 'managing director', 'net worth')
- There is NO standalone 'company' definition — only specific compound terms like 'private company', 'public company', 'foreign company', 'small company', 'government company', 'listed company', 'one person company'
- For broad definition searches, use: MATCH (d:Definition) WHERE d.term CONTAINS 'company' RETURN d.uid, d.term, d.text
- For exact lookups, use the full lowercase term: {{term: 'private company'}}
- NOT ALL definitions have a Definition node — some are only in Section 2 subsection text. \
If a Definition search returns empty, fall back to: \
MATCH (s:Section {{number: 2}})-[:HAS_SUBSECTION]->(ss) WHERE toLower(ss.text) CONTAINS 'term_here' RETURN ss.uid, ss.text

HINDI QUERY HANDLING:
- Hindi queries arrive with an English translation appended: 'Hindi text [English: translated text]'
- ALWAYS use the English translation (inside the brackets) for generating Cypher
- The graph stores ALL content in English — never match Hindi text against graph properties
- Example: 'धारा 135 क्या कहती है? [English: section 135 what does it say]' → use section 135

CRITICAL — Chapter title casing:
- Chapter.title is stored in ALL CAPS (e.g. 'DECLARATION AND PAYMENT OF DIVIDEND', not 'Declaration and Payment of Dividend')
- ALWAYS use toUpper() when matching chapters by title: WHERE toUpper(c.title) = toUpper('Declaration and Payment of Dividend')
- Or match by Chapter.number (Roman numeral STRING): WHERE c.number = 'VIII'
- Chapter UIDs follow the pattern 'ch-VIII', 'ch-IX', etc.

IMPORTANT QUERY PATTERNS:
1. Section.number and SubSection.number are INT: WHERE s.number = 135 (not '135').
2. Rule.number is STRING: WHERE r.number = '8' (not 8).
3. Chapter.number is a Roman numeral STRING: WHERE c.number = 'VIII' (not 8).
4. To find Rules related to a Section, use: (r:Rule)-[:REFERS_TO|PRESCRIBES_FOR|DERIVED_RULE]->(s:Section)
5. To find penalty text, check subsections: (s:Section)-[:HAS_SUBSECTION]->(ss) WHERE ss.text CONTAINS 'penalty'
6. For text search when exact property matches fail, use: WHERE s.text CONTAINS 'keyword'
7. For negation queries (nodes WITHOUT a relationship), use: WHERE NOT EXISTS {{ MATCH (n)-[:REL]->(m) }}
8. To traverse from Act to chapters to sections: (a:Act)-[:HAS_CHAPTER]->(c:Chapter)-[:HAS_SECTION]->(s:Section)
9. All nodes have a `uid` property — always include it in RETURN for traceability.
10. For case-insensitive title matching, use toUpper(): WHERE toUpper(c.title) = toUpper('some title')
11. SUBJECT_TO edges come from BOTH Section AND SubSection nodes. To find what Section X is subject to, also check its subsections: \
MATCH (s:Section {{number: X}}) OPTIONAL MATCH (s)-[r1:SUBJECT_TO]->(t1) OPTIONAL MATCH (s)-[:HAS_SUBSECTION]->(ss)-[r2:SUBJECT_TO]->(t2) \
RETURN s.uid, t1.uid, t2.uid, ss.uid
12. ALWAYS use LIMIT in queries that may return large result sets. Use LIMIT 25 for listing queries and LIMIT 50 for structural queries.
13. When listing many nodes, use substring(n.text, 0, 150) AS text_preview instead of full n.text to avoid overwhelming response sizes.
{node_counts}
"""

# ---------------------------------------------------------------------------
# Cypher generation prompt
# ---------------------------------------------------------------------------

CYPHER_GENERATION_PROMPT: str = """\
{schema}

Examples of valid Cypher queries for this graph:
{examples}

IMPORTANT: Always include the `uid` property of matched nodes in your RETURN clause (e.g., s.uid, r.uid, d.uid) so results can be traced to source graph nodes.

Generate a single read-only Cypher query to answer the question below.
Use MATCH/WHERE/RETURN only. Do NOT use CREATE, SET, DELETE, MERGE, DROP.
Return the Cypher query only, no explanation.

Question: {query}
"""

# ---------------------------------------------------------------------------
# Response synthesis prompt
# ---------------------------------------------------------------------------

SYNTHESIS_PROMPT: str = """\
You are a legal document formatter. Format the graph query results below into readable prose.

Original question: {query}

Cypher query used:
{cypher}

Graph results:
{results}

Available source UIDs:
{sources}

Instructions:
- Use inline citations (e.g., 'Section 135(1) states...', 'Rule 8 provides...')
- IMPORTANT: After each factual claim, append a citation marker in the format [uid] where uid matches one of the Available source UIDs listed above. Example: 'Section 135 mandates CSR spending [sec-135]'
- You may cite multiple sources for one claim: 'penalties apply [sec-135] [sec-135-ss-7]'
- NEVER generate facts — only format the data provided above
- If results are empty, say you could not find the information in the graph
- Keep the response concise and legally precise
"""

# ---------------------------------------------------------------------------
# Retry prompt suffixes — appended to CYPHER_GENERATION_PROMPT on retry
# ---------------------------------------------------------------------------

RETRY_PROMPT_SUFFIX: str = (
    "The previous Cypher query failed with error: {error_message}. "
    "Please fix the query and try again."
)

BROADEN_PROMPT_SUFFIX: str = (
    "The previous query returned no results. Try broadening:\n"
    "- Chapter titles are ALL CAPS — use toUpper(c.title) CONTAINS toUpper('keyword') instead of exact match\n"
    "- Prefer matching Chapter by number (e.g. c.number = 'VIII') over title\n"
    "- Use WHERE s.text CONTAINS 'keyword' instead of exact property match\n"
    "- Use WHERE toUpper(s.title) CONTAINS toUpper('keyword') for title search\n"
    "- Try OPTIONAL MATCH for relationships that may not exist\n"
    "- Remove filters to check if base nodes exist\n"
    "- For definitions not found as Definition nodes, search Section 2 subsections: "
    "MATCH (s:Section {number: 2})-[:HAS_SUBSECTION]->(ss) WHERE toLower(ss.text) CONTAINS 'term'\n"
    "Example: MATCH (c:Chapter {number: 'IX'})-[:HAS_SECTION]->(s) "
    "RETURN c.uid, s.uid, s.number, s.title LIMIT 10"
)

# ---------------------------------------------------------------------------
# Per-intent few-shot examples (D-01)
#
# Each intent maps to a list of {question, cypher} dicts.
# At generation time, the classifier selects the matching pool and injects
# 2-3 examples into CYPHER_GENERATION_PROMPT via the {examples} placeholder.
# ---------------------------------------------------------------------------

INTENT_EXAMPLES: dict[str, list[dict[str, str]]] = {
    "structured_lookup": [
        {
            "question": "What does Section 135 say?",
            "cypher": "MATCH (s:Section {number: 135}) RETURN s.uid, s.title, s.text",
        },
        {
            "question": "What is the definition of 'company'?",
            "cypher": "MATCH (d:Definition) WHERE d.term CONTAINS 'company' RETURN d.uid, d.term, d.text",
        },
        {
            "question": "Show me Rule 8",
            "cypher": "MATCH (r:Rule {number: '8'}) RETURN r.uid, r.number, r.text",
        },
        {
            "question": "What are the subsections of Section 135?",
            "cypher": (
                "MATCH (s:Section {number: 135})-[:HAS_SUBSECTION]->(ss) "
                "RETURN s.uid, ss.uid, ss.number, ss.text ORDER BY ss.number"
            ),
        },
        {
            "question": "What does the term 'related party' mean?",
            "cypher": (
                "MATCH (d:Definition) WHERE toLower(d.term) CONTAINS 'related party' "
                "RETURN d.uid, d.term, d.text "
                "UNION "
                "MATCH (s:Section {number: 2})-[:HAS_SUBSECTION]->(ss) "
                "WHERE toLower(ss.text) CONTAINS 'related party' "
                "RETURN ss.uid AS `d.uid`, 'related party' AS `d.term`, ss.text AS `d.text`"
            ),
        },
        {
            "question": "Show me the provisions of Section 149 with all subsections",
            "cypher": (
                "MATCH (s:Section {number: 149}) "
                "OPTIONAL MATCH (s)-[:HAS_SUBSECTION]->(ss) "
                "RETURN s.uid, s.title, s.text, collect(ss.text) AS subsections"
            ),
        },
        {
            "question": "What are all definitions in the Companies Act?",
            "cypher": (
                "MATCH (d:Definition) RETURN d.uid, d.term, d.text ORDER BY d.term"
            ),
        },
        {
            "question": "Show me Schedule I",
            "cypher": "MATCH (sc:Schedule {number: 'I'}) RETURN sc.uid, sc.number, sc.text",
        },
    ],
    "amendment_query": [
        {
            "question": "How was Section 135 amended?",
            "cypher": (
                "MATCH (a:AmendmentAct)-[r]->(s:Section {number: 135}) "
                "WHERE type(r) IN ['SUBSTITUTES','INSERTS','OMITS','DECRIMINALIZES'] "
                "RETURN s.uid, a.uid, a.title, type(r) AS amendment_type, r.new_text, r.old_text, r.effective_date "
                "ORDER BY r.effective_date"
            ),
        },
        {
            "question": "What changes were made to Section 149?",
            "cypher": (
                "MATCH (a:AmendmentAct)-[r]->(s:Section {number: 149}) "
                "WHERE type(r) IN ['SUBSTITUTES','INSERTS','OMITS','DECRIMINALIZES'] "
                "RETURN s.uid, a.uid, a.title, type(r) AS amendment_type, r.new_text, r.effective_date"
            ),
        },
        {
            "question": "Which sections were substituted by the Amendment Act?",
            "cypher": (
                "MATCH (a:AmendmentAct)-[r:SUBSTITUTES]->(s:Section) "
                "RETURN s.uid, a.uid, s.number, s.title, r.new_text, r.effective_date "
                "ORDER BY s.number"
            ),
        },
        {
            "question": "Were any sections omitted by the amendment?",
            "cypher": (
                "MATCH (a:AmendmentAct)-[r:OMITS]->(s:Section) "
                "RETURN s.uid, a.uid, s.number, s.title, r.effective_date ORDER BY s.number"
            ),
        },
        {
            "question": "What sections were omitted by the Corporate Laws Amendment Act 2026?",
            "cypher": (
                "MATCH (a:AmendmentAct {uid: 'amend-corporate-laws-2026'})-[r:OMITS]->(s:Section) "
                "RETURN s.uid, a.uid, a.title, s.number, s.title, r.effective_date ORDER BY s.number"
            ),
        },
        {
            "question": "What is the effective text of Section 135 after amendments?",
            "cypher": (
                "MATCH (s:Section {number: 135}) "
                "OPTIONAL MATCH (a:AmendmentAct)-[r]->(s) "
                "WHERE type(r) IN ['SUBSTITUTES','INSERTS','OMITS','DECRIMINALIZES'] "
                "RETURN s.uid, s.title, s.text AS original_text, "
                "collect({type: type(r), new_text: r.new_text, old_text: r.old_text}) AS amendments"
            ),
        },
        {
            "question": "Compare the original Section 149 with changes made by the Amendment Act",
            "cypher": (
                "MATCH (s:Section {number: 149}) "
                "OPTIONAL MATCH (s)-[:HAS_SUBSECTION]->(ss) "
                "OPTIONAL MATCH (a:AmendmentAct)-[r]->(s) "
                "WHERE type(r) IN ['SUBSTITUTES','INSERTS','OMITS','DECRIMINALIZES'] "
                "RETURN s.uid, s.title, s.text AS original_text, "
                "collect(DISTINCT ss.text) AS subsections, "
                "collect(DISTINCT {amendment: a.title, type: type(r), new_text: r.new_text, old_text: r.old_text, date: r.effective_date}) AS amendments"
            ),
        },
        {
            "question": "Which sections have been amended and what are the amendment details?",
            "cypher": (
                "MATCH (a:AmendmentAct)-[r]->(s:Section) "
                "WHERE type(r) IN ['SUBSTITUTES','INSERTS','OMITS','DECRIMINALIZES'] "
                "RETURN s.uid, a.uid, s.number, s.title, a.title AS amendment_act, type(r) AS amendment_type, "
                "r.new_text, r.old_text, r.effective_date "
                "ORDER BY s.number LIMIT 25"
            ),
        },
        {
            "question": "How many amendment entries affect Chapter VIII (Declaration and Payment of Dividend)?",
            "cypher": (
                "MATCH (a:Act {uid: 'act-companies-2013'})-[:HAS_CHAPTER]->(c:Chapter {number: 'VIII'})-[:HAS_SECTION]->(s:Section) "
                "MATCH (am:AmendmentAct)-[r]->(s) "
                "WHERE type(r) IN ['SUBSTITUTES','INSERTS','OMITS','DECRIMINALIZES'] "
                "RETURN s.uid, am.uid, s.number, s.title, type(r) AS amendment_type "
                "ORDER BY s.number"
            ),
        },
        {
            "question": "What did the Corporate Laws Amendment Act 2026 change?",
            "cypher": (
                "MATCH (a:AmendmentAct {uid: 'amend-corporate-laws-2026'})-[r]->(s:Section) "
                "WHERE type(r) IN ['SUBSTITUTES','INSERTS','OMITS','DECRIMINALIZES'] "
                "RETURN s.uid, a.uid, a.title, s.number, s.title AS section_title, "
                "type(r) AS amendment_type, r.new_text, r.old_text, r.effective_date "
                "ORDER BY s.number LIMIT 25"
            ),
        },
        {
            "question": "Which sections were decriminalized by the Amendment Act?",
            "cypher": (
                "MATCH (a:AmendmentAct)-[r:DECRIMINALIZES]->(s:Section) "
                "RETURN s.uid, a.uid, s.number, s.title, r.new_text, r.old_text, r.effective_date "
                "ORDER BY s.number"
            ),
        },
    ],
    "cross_reference": [
        {
            "question": "Which sections refer to Section 135?",
            "cypher": (
                "MATCH (source)-[r:REFERS_TO]->(t:Section {number: 135}) "
                "RETURN source.uid, labels(source)[0] AS source_type, source.number, type(r)"
            ),
        },
        {
            "question": "What nodes reference Section 149?",
            "cypher": (
                "MATCH (source)-[r]->(t:Section {number: 149}) "
                "WHERE type(r) IN ['REFERS_TO', 'SUBJECT_TO', 'NOTWITHSTANDING', 'PRESCRIBES_FOR'] "
                "RETURN source.uid, labels(source)[0] AS source_type, source.number, type(r) AS rel_type"
            ),
        },
        {
            "question": "What is Section 149 subject to?",
            "cypher": (
                "MATCH (s:Section {number: 149}) "
                "OPTIONAL MATCH (s)-[r1:SUBJECT_TO]->(t1) "
                "OPTIONAL MATCH (s)-[:HAS_SUBSECTION]->(ss)-[r2:SUBJECT_TO]->(t2) "
                "RETURN s.uid, t1.uid AS direct_target, ss.uid AS subsection_uid, t2.uid AS sub_target, t2.number, t2.title"
            ),
        },
        {
            "question": "Which sections have notwithstanding clauses?",
            "cypher": (
                "MATCH (s)-[r:NOTWITHSTANDING]->(t) "
                "RETURN s.uid, t.uid, labels(s)[0] AS source_type, s.number, labels(t)[0] AS target_type, t.number"
            ),
        },
        {
            "question": "Which rules prescribe for Section 135?",
            "cypher": (
                "MATCH (r:Rule)-[:REFERS_TO|PRESCRIBES_FOR]->(s:Section {number: 135}) "
                "RETURN r.uid, s.uid, r.number, r.text"
            ),
        },
        {
            "question": "Which rules refer to Section 185?",
            "cypher": (
                "MATCH (r:Rule)-[rel:REFERS_TO|PRESCRIBES_FOR|SUBJECT_TO]->(s:Section {number: 185}) "
                "RETURN r.uid, s.uid, r.number, r.text, type(rel) AS relationship"
            ),
        },
        {
            "question": "Which chapters have no corresponding rules?",
            "cypher": (
                "MATCH (c:Chapter)-[:HAS_SECTION]->(s:Section) "
                "WHERE NOT EXISTS { MATCH (r:Rule)-[:REFERS_TO|PRESCRIBES_FOR]->(s) } "
                "WITH c, collect(s.number) AS sections "
                "RETURN c.uid, c.number, c.title, size(sections) AS section_count "
                "ORDER BY c.number"
            ),
        },
    ],
    "penalty_query": [
        {
            "question": "What is the penalty under Section 447?",
            "cypher": (
                "MATCH (s:Section {number: 447}) "
                "OPTIONAL MATCH (s)-[:HAS_SUBSECTION]->(ss) "
                "RETURN s.uid, s.title, s.text, collect(ss.text) AS subsections"
            ),
        },
        {
            "question": "What punishment does Section 448 prescribe?",
            "cypher": (
                "MATCH (s:Section {number: 448}) "
                "OPTIONAL MATCH (s)-[:HAS_SUBSECTION]->(ss) "
                "RETURN s.uid, s.title, s.text, collect(ss.text) AS subsections"
            ),
        },
        {
            "question": "What is the penalty for non-compliance with Section 135?",
            "cypher": (
                "MATCH (s:Section {number: 135}) "
                "OPTIONAL MATCH (s)-[:HAS_SUBSECTION]->(ss) "
                "WHERE ss.text CONTAINS 'penalty' OR ss.text CONTAINS 'liable' OR ss.text CONTAINS 'fine' "
                "RETURN s.uid, s.title, s.text, "
                "collect(DISTINCT ss.text) AS penalty_subsections"
            ),
        },
        {
            "question": "What are the penalties for violating Section 185?",
            "cypher": (
                "MATCH (s:Section {number: 185}) "
                "OPTIONAL MATCH (s)-[:HAS_SUBSECTION]->(ss) "
                "OPTIONAL MATCH (r:Rule)-[:REFERS_TO|PRESCRIBES_FOR]->(s) "
                "RETURN s.uid, s.title, s.text, "
                "collect(DISTINCT ss.text) AS subsections, "
                "collect(DISTINCT {rule_number: r.number, rule_text: r.text}) AS related_rules"
            ),
        },
        {
            "question": "Which sections were decriminalized by the Amendment Act?",
            "cypher": (
                "MATCH (a:AmendmentAct)-[r:DECRIMINALIZES]->(s:Section) "
                "RETURN s.uid, a.uid, s.number, s.title, r.new_text, r.old_text, r.effective_date "
                "ORDER BY s.number"
            ),
        },
        {
            "question": "List all sections that impose personal liability on directors",
            "cypher": (
                "MATCH (s:Section) "
                "WHERE s.text CONTAINS 'shall be liable' OR s.text CONTAINS 'personally liable' "
                "OPTIONAL MATCH (s)-[:HAS_SUBSECTION]->(ss) "
                "WHERE ss.text CONTAINS 'liable' OR ss.text CONTAINS 'penalty' "
                "RETURN s.uid, s.number, s.title, s.text, collect(ss.text) AS penalty_subsections "
                "ORDER BY s.number"
            ),
        },
    ],
    "general_info": [
        {
            "question": "What does Chapter IX cover?",
            "cypher": (
                "MATCH (c:Chapter {number: 'IX'})-[:HAS_SECTION]->(s) "
                "RETURN c.uid, s.uid, c.title, s.number, s.title ORDER BY s.number"
            ),
        },
        {
            "question": "What sections are in the chapter on Dividends?",
            "cypher": (
                "MATCH (c:Chapter)-[:HAS_SECTION]->(s:Section) "
                "WHERE toUpper(c.title) CONTAINS 'DIVIDEND' "
                "RETURN c.uid, s.uid, c.title, s.number, s.title ORDER BY s.number"
            ),
        },
        {
            "question": "How many sections are in the Companies Act?",
            "cypher": (
                "MATCH (a:Act)-[:HAS_CHAPTER]->()-[:HAS_SECTION]->(s) "
                "RETURN count(s) AS total_sections"
            ),
        },
        {
            "question": "How many chapters are in the Companies Act?",
            "cypher": (
                "MATCH (a:Act)-[:HAS_CHAPTER]->(c) "
                "RETURN count(c) AS total_chapters"
            ),
        },
        {
            "question": "What is the structure of the Companies Act?",
            "cypher": (
                "MATCH (a:Act {uid: 'act-companies-2013'})-[:HAS_CHAPTER]->(c) "
                "OPTIONAL MATCH (c)-[:HAS_SECTION]->(s) "
                "RETURN c.uid, c.number, c.title, count(s) AS section_count ORDER BY c.number LIMIT 50"
            ),
        },
        {
            "question": "What are the provisions of the Companies Act?",
            "cypher": (
                "MATCH (a:Act)-[:HAS_CHAPTER]->(c) "
                "RETURN c.uid, c.number, c.title ORDER BY c.number"
            ),
        },
        {
            "question": "Find sections about CSR",
            "cypher": (
                "MATCH (s:Section) WHERE s.text CONTAINS 'Corporate Social Responsibility' "
                "OR s.title CONTAINS 'Corporate Social Responsibility' "
                "RETURN s.uid, s.number, s.title, s.text LIMIT 10"
            ),
        },
        {
            "question": "धारा 135 क्या कहती है? [English: section 135 what does it say]",
            "cypher": "MATCH (s:Section {number: 135}) RETURN s.uid, s.title, s.text",
        },
        {
            "question": "कंपनी की परिभाषा क्या है? [English: definition of company what is]",
            "cypher": (
                "MATCH (d:Definition) WHERE d.term CONTAINS 'company' RETURN d.uid, d.term, d.text"
            ),
        },
        {
            "question": "What rules exist in the Companies Rules?",
            "cypher": (
                "MATCH (rs:RuleSet {uid: 'ruleset-companies-rules'})-[:HAS_RULE]->(r:Rule) "
                "RETURN r.uid, r.number, substring(r.text, 0, 150) AS text_preview ORDER BY r.number LIMIT 25"
            ),
        },
    ],
}

# ---------------------------------------------------------------------------
# Fallback synthesis prompt — used when BM25 fulltext results are returned
# instead of structured Cypher results
# ---------------------------------------------------------------------------

FALLBACK_SYNTHESIS_PROMPT: str = """\
You are a legal document formatter. The structured Cypher query did not return results, \
so a fulltext search was used instead. The results below are ranked by relevance score.

Original question: {query}

Fulltext search results:
{results}

Available source UIDs:
{sources}

Instructions:
- Identify the most relevant result(s) that answer the question
- Use inline citations (e.g., 'Section 135(1) states...', 'Rule 8 provides...')
- IMPORTANT: After each factual claim, append a citation marker in the format [uid] where uid matches one of the Available source UIDs listed above. Example: 'Section 135 mandates CSR spending [sec-135]'
- You may cite multiple sources for one claim: 'penalties apply [sec-135] [sec-135-ss-7]'
- NEVER generate facts — only format the data provided above
- If none of the results are relevant, say you could not find the information
- Note that these results came from a text search, not a structured graph query
- Keep the response concise and legally precise
"""
