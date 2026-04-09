"""
Cypher generator for ViddhiAI query engine.

Translates natural language queries into read-only Cypher using Gemini,
with per-intent few-shot examples and schema injection.

Design choices:
- D-01: Per-intent few-shot pools — INTENT_EXAMPLES selected by intent key
- D-02: Schema injected at generation time via get_schema_text()
- D-03: Schema explicitly annotates Section.number as STRING (Pitfall 5)
- D-06: Module-level LLM singleton (instantiated once at import time)
- T-03-05: Generator NEVER executes Cypher — returns string for validator.py
- T-03-08: User query placed in designated {query} slot; few-shots are pre-defined
- T-03-09: GEMINI_API_KEY read from env var, never logged
"""

from __future__ import annotations

import logging
import os
import re

from langchain_google_genai import ChatGoogleGenerativeAI

from graph.connection import DATABASE
from query.prompts import (
    BROADEN_PROMPT_SUFFIX,
    CYPHER_GENERATION_PROMPT,
    INTENT_EXAMPLES,
    RETRY_PROMPT_SUFFIX,
    SCHEMA_TEMPLATE,
    SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level LLM singleton (D-06, T-03-09)
# ---------------------------------------------------------------------------

_llm = ChatGoogleGenerativeAI(
    model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
    temperature=0.0,
    api_key=os.environ.get("GEMINI_API_KEY"),
)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def get_schema_text(driver=None) -> str:
    """Build the schema text string to inject into the Cypher generation prompt.

    Uses static schema without live node counts to avoid an expensive full-graph
    scan on startup. Node counts add minimal value for Cypher generation — the
    LLM needs structure, not volume.

    Args:
        driver: Neo4j Driver instance (unused — kept for API compatibility).

    Returns:
        Schema string formatted from SCHEMA_TEMPLATE.
    """
    return SCHEMA_TEMPLATE.format(node_counts="")


# ---------------------------------------------------------------------------
# Cypher cleaning helpers
# ---------------------------------------------------------------------------


def _clean_cypher(raw: str) -> str:
    """Strip markdown fences and extraneous prefixes from LLM-generated Cypher.

    Handles:
    - Leading/trailing whitespace
    - ```cypher ... ``` fences
    - ``` ... ``` fences (no language tag)
    - Leading 'CYPHER:' or 'cypher:' prefix

    Args:
        raw: Raw Cypher string as returned by the LLM.

    Returns:
        Cleaned Cypher string ready for validation and execution.
    """
    cleaned = raw.strip()

    # Remove ```cypher\n...\n``` or ```\n...\n``` fences
    # Handles both with and without newlines between fence and content
    cleaned = re.sub(r"^```(?:cypher)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned, flags=re.IGNORECASE)

    # Remove leading CYPHER: or cypher: prefix
    cleaned = re.sub(r"^(?:CYPHER|cypher):\s*", "", cleaned)

    return cleaned.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_cypher(
    query: str,
    intent: str,
    schema_text: str,
    error_context: str | None = None,
    empty_retry_count: int = 0,
) -> str:
    """Generate a read-only Cypher query from a natural language question.

    Selects few-shot examples for the given intent from INTENT_EXAMPLES (D-01),
    injects schema (D-02), and optionally appends retry or broadening suffixes.

    The generated Cypher is returned as a plain string — this function NEVER
    executes it. Execution is handled by executor.py after validator.py checks
    it (T-03-05).

    Args:
        query: Natural language question from the user.
        intent: One of the 5 intent categories from classify_intent().
        schema_text: Schema string from get_schema_text() — injected into prompt.
        error_context: Neo4j error message from the previous attempt (triggers retry suffix).
        empty_retry_count: Count of empty-result retries (triggers broadening suffix).

    Returns:
        Cleaned Cypher query string.
    """
    # Select per-intent few-shot examples (D-01)
    examples_list = INTENT_EXAMPLES.get(intent, [])
    if examples_list:
        examples_parts = []
        for i, ex in enumerate(examples_list, 1):
            examples_parts.append(
                f"Example {i}:\nQ: {ex['question']}\nCypher: {ex['cypher']}"
            )
        examples_text = "\n\n".join(examples_parts)
    else:
        examples_text = "(no examples available for this intent)"

    # Build base prompt with schema and examples (D-02, D-03)
    prompt = CYPHER_GENERATION_PROMPT.format(
        schema=schema_text,
        examples=examples_text,
        query=query,
    )

    # Append retry suffix if previous attempt errored
    if error_context is not None:
        prompt += "\n\n" + RETRY_PROMPT_SUFFIX.format(error_message=error_context)

    # Append broadening suffix if previous attempt returned empty results
    if empty_retry_count > 0:
        prompt += "\n\n" + BROADEN_PROMPT_SUFFIX

    # Call LLM with system prompt for role context
    from langchain_core.messages import HumanMessage, SystemMessage

    response = _llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
    raw_cypher = response.content

    cleaned = _clean_cypher(raw_cypher)
    logger.info(
        "Generated Cypher for intent '%s': %s",
        intent,
        cleaned[:120],
    )
    return cleaned
