"""common/schemas.py - Pydantic models for structured LLM outputs.

WHY THIS FILE EXISTS
--------------------
A raw LLM reply is free text. When the application needs *fields* it can
rely on - a category, an answer, key points - we define a schema and ask the
model to fill it:

    result = llm.with_structured_output(FinanceAnswer).invoke(messages)

LangChain converts the schema to the provider's native mechanism
(function/tool calling on Gemini, JSON schema on Ollama) and validates the
reply into a Pydantic object. If the model returns something invalid,
Pydantic raises - which the application catches and falls back to plain text
(see D1P1.py and D1P3.py).

API USED
--------
pydantic.BaseModel   - declares the fields of the expected JSON object.
pydantic.Field(...)  - adds a description the model uses to fill the field.
typing.Literal[...]  - restricts a field to a fixed set of allowed values.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# The six query categories understood by the FinDesk router (Session 3).
# Keeping them as a Literal means BOTH the LLM output and our Python code
# are checked against the same list.
# --------------------------------------------------------------------------
QueryCategory = Literal[
    "expense_policy",   # "what is the food allowance policy?"
    "invoice",          # "review invoice INV-2041"
    "reimbursement",    # "how do I claim travel reimbursement?"
    "budget",           # "budget utilisation of Department A"
    "general",          # "difference between invoice and purchase order"
    "unclear",          # too vague to classify
]


class FinanceAnswer(BaseModel):
    """Structured answer produced in Session 1 (`/structured` command).

    Attributes
    ----------
    category   : str            - topic detected in the question.
    answer     : str            - the actual plain-language answer.
    key_points : list[str]      - up to three bullet points (optional).
    """

    category: str = Field(
        description="Short topic label, e.g. expense_policy, invoice, budget"
    )
    answer: str = Field(
        description="Clear, complete answer for the employee, 2-5 sentences"
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="Up to three short bullet points summarising the answer",
    )


class QueryClassification(BaseModel):
    """Structured classification produced by the Session 3 router.

    The classifier LLM is asked to return exactly this object, so the graph
    receives a validated `category` string it can use in a conditional edge.
    """

    category: QueryCategory = Field(
        description="Best matching category for the employee's question"
    )
    reasoning: str = Field(
        description="One short sentence explaining why this category was chosen"
    )
