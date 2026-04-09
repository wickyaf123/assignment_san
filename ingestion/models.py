"""
Pydantic v2 data models for the ViddhiAI legal document ingestion pipeline.

Models represent the full legislative hierarchy:
    Act -> Chapter -> Section -> SubSection -> Clause -> SubClause
                                                      -> Proviso
                                                      -> Explanation
                                                      -> Definition
    AmendmentAct -> AmendmentEntry
    RuleSet -> Rule -> Form
    Schedule

All models support JSON serialization via model.model_dump() and model.model_dump_json().
UIDs are deterministic — generated via ingestion.uid_generator functions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CrossReference(BaseModel):
    """A cross-reference from one legal provision to another.

    Attributes:
        target_pattern: The text pattern identifying the target (e.g., "section 149").
        ref_type: Relationship type — one of REFERS_TO, SUBJECT_TO, NOTWITHSTANDING.
        raw_text: The raw text excerpt from which this reference was extracted.
    """

    target_pattern: str
    ref_type: str  # REFERS_TO | SUBJECT_TO | NOTWITHSTANDING
    raw_text: str


class BaseNode(BaseModel):
    """Base class for all legal document node types.

    Attributes:
        uid: Deterministic unique identifier (e.g., "sec-135", "def-company").
        node_type: String discriminator identifying the node type.
        page_number: PDF page number where this node starts.
        text: Full extracted text content of this node.
        cross_references: Tagged cross-references found in this node's text.
        source_pdf: Filename of the source PDF (e.g., "Companies Act, 2013.pdf").
    """

    uid: str
    node_type: str
    page_number: int
    text: str
    cross_references: list[CrossReference] = Field(default_factory=list)
    source_pdf: str
    language: str = "en"


class Definition(BaseNode):
    """A defined term from Section 2 or an inline Explanation clause.

    Attributes:
        term: The defined term (e.g., "associate company"), enables exact-match lookups.
    """

    node_type: Literal["Definition"] = "Definition"
    term: str


class Proviso(BaseNode):
    """A proviso — a conditional or restrictive addition to a provision.

    Attributes:
        proviso_type: Distinguishes between proviso variants:
            "Provided that" | "Provided also that" | "Provided further that"
    """

    node_type: Literal["Proviso"] = "Proviso"
    proviso_type: str  # "Provided that" | "Provided also that" | "Provided further that"


class Explanation(BaseNode):
    """An explanatory clause attached to a provision.

    Attributes:
        label: Optional label for numbered explanations (e.g., "Explanation I", "Explanation II").
                Empty string for unlabelled explanations.
    """

    node_type: Literal["Explanation"] = "Explanation"
    label: str = ""


class SubClause(BaseNode):
    """A sub-clause within a Clause, identified by a Roman numeral label (e.g., "(i)").

    Attributes:
        label: The sub-clause label (e.g., "i", "ii", "iii").
        provisos: Provisos attached directly to this sub-clause.
        explanations: Explanations attached directly to this sub-clause.
    """

    node_type: Literal["SubClause"] = "SubClause"
    label: str
    provisos: list[Proviso] = Field(default_factory=list)
    explanations: list[Explanation] = Field(default_factory=list)


class Clause(BaseNode):
    """A clause within a SubSection or Section, identified by a letter label (e.g., "(a)").

    Attributes:
        label: The clause label (e.g., "a", "b", "c").
        subclauses: Sub-clauses nested within this clause.
        provisos: Provisos attached directly to this clause.
        explanations: Explanations attached directly to this clause.
    """

    node_type: Literal["Clause"] = "Clause"
    label: str
    subclauses: list[SubClause] = Field(default_factory=list)
    provisos: list[Proviso] = Field(default_factory=list)
    explanations: list[Explanation] = Field(default_factory=list)


class SubSection(BaseNode):
    """A sub-section within a Section, identified by a parenthetical numeral (e.g., "(1)").

    Attributes:
        number: The sub-section number (e.g., 1, 2, 3).
        clauses: Clauses nested within this sub-section.
        provisos: Provisos attached directly to this sub-section.
        explanations: Explanations attached directly to this sub-section.
        definitions: Inline definitions found within this sub-section.
    """

    node_type: Literal["SubSection"] = "SubSection"
    number: int
    clauses: list[Clause] = Field(default_factory=list)
    provisos: list[Proviso] = Field(default_factory=list)
    explanations: list[Explanation] = Field(default_factory=list)
    definitions: list[Definition] = Field(default_factory=list)


class Section(BaseNode):
    """A section of the Act.

    Attributes:
        number: The section number (e.g., 135).
        title: The marginal note / section title (e.g., "Corporate Social Responsibility").
        subsections: Sub-sections nested within this section.
        clauses: Clauses at section level (some sections have direct clauses, no sub-sections).
        provisos: Provisos attached directly to this section.
        explanations: Explanations attached directly to this section.
        definitions: Definitions found within this section (e.g., Section 2 definitions).
    """

    node_type: Literal["Section"] = "Section"
    number: int
    title: str
    subsections: list[SubSection] = Field(default_factory=list)
    clauses: list[Clause] = Field(default_factory=list)
    provisos: list[Proviso] = Field(default_factory=list)
    explanations: list[Explanation] = Field(default_factory=list)
    definitions: list[Definition] = Field(default_factory=list)


class Chapter(BaseNode):
    """A chapter of the Act.

    Attributes:
        number: The chapter number as a Roman numeral string (e.g., "I", "II", "XXIX").
        title: The chapter title (e.g., "PRELIMINARY").
        sections: All sections within this chapter.
    """

    node_type: Literal["Chapter"] = "Chapter"
    number: str  # Roman numeral, e.g., "I", "II", "XXIX"
    title: str
    sections: list[Section] = Field(default_factory=list)


class Act(BaseNode):
    """The top-level Act node.

    Attributes:
        title: Full title of the Act (e.g., "The Companies Act, 2013").
        year: Year of enactment (e.g., 2013).
        chapters: All chapters of the Act in order.
    """

    node_type: Literal["Act"] = "Act"
    title: str
    year: int
    chapters: list[Chapter] = Field(default_factory=list)


class AmendmentEntry(BaseModel):
    """A single amendment instruction from an Amendment Act.

    Attributes:
        target_section: Identifier of the section being amended (e.g., "section 135").
        amendment_type: Classification of the amendment operation:
            SUBSTITUTES | INSERTS | OMITS | DECRIMINALIZES
        new_text: The replacement or inserted text (empty for OMITS).
        removed_text: The text being omitted or replaced (empty for INSERTS).
        raw_text: The full raw text of this amendment entry.
    """

    target_section: str
    amendment_type: str  # SUBSTITUTES | INSERTS | OMITS | DECRIMINALIZES
    new_text: str = ""
    removed_text: str = ""
    raw_text: str


class AmendmentAct(BaseNode):
    """An Amendment Act node containing amendment instructions.

    Attributes:
        title: Full title of the Amendment Act.
        year: Year of the Amendment Act.
        entries: All amendment entries in this Act.
    """

    node_type: Literal["AmendmentAct"] = "AmendmentAct"
    title: str
    year: int
    entries: list[AmendmentEntry] = Field(default_factory=list)


class Form(BaseNode):
    """A statutory form associated with a Rule.

    Attributes:
        form_number: The form identifier (e.g., "INC-1", "INC-32A").
        title: The form title/description.
        associated_rule: Rule number this form is associated with.
    """

    node_type: Literal["Form"] = "Form"
    form_number: str
    title: str = ""
    associated_rule: str = ""


class Rule(BaseNode):
    """A rule within a RuleSet (subordinate legislation).

    Attributes:
        number: The rule number as a string (e.g., "3", "3A", "4").
        title: The rule title.
        forms: Forms prescribed by or associated with this rule.
    """

    node_type: Literal["Rule"] = "Rule"
    number: str
    title: str = ""
    forms: list[Form] = Field(default_factory=list)


class Schedule(BaseNode):
    """A schedule attached to an Act or Rules.

    Attributes:
        number: The schedule number/identifier (e.g., "I", "II", "VII").
        title: The schedule title.
    """

    node_type: Literal["Schedule"] = "Schedule"
    number: str
    title: str = ""


class RuleSet(BaseNode):
    """A set of rules (subordinate legislation) e.g., Companies (Incorporation) Rules, 2014.

    Attributes:
        title: Full title of the rule set.
        year: Year of notification/publication.
        rules: All rules within this rule set.
    """

    node_type: Literal["RuleSet"] = "RuleSet"
    title: str
    year: int = 0
    rules: list[Rule] = Field(default_factory=list)


class ParseReport(BaseModel):
    """Quality report produced after each ingestion run.

    Attributes:
        source_pdf: Filename of the PDF that was parsed.
        total_sections: Count of Section nodes extracted.
        total_subsections: Count of SubSection nodes extracted.
        total_clauses: Count of Clause nodes extracted.
        total_provisos: Count of Proviso nodes extracted.
        total_explanations: Count of Explanation nodes extracted.
        total_definitions: Count of Definition nodes extracted.
        warnings: Log of non-fatal issues encountered during parsing.
        coverage_percent: Percentage of pages that yielded structured content (0.0–100.0).
    """

    source_pdf: str
    total_sections: int
    total_subsections: int
    total_clauses: int
    total_provisos: int
    total_explanations: int
    total_definitions: int
    warnings: list[str]
    coverage_percent: float


# Resolve forward references for models that contain nested models defined later.
# Required by Pydantic v2 when using `from __future__ import annotations`.
Act.model_rebuild()
AmendmentAct.model_rebuild()
RuleSet.model_rebuild()
