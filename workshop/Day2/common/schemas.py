"""common/schemas.py - Pydantic models for structured LLM outputs.

WHY THIS FILE EXISTS
--------------------
A raw LLM reply is free text. When the application needs *fields* it can
rely on - a category, a task plan, an answer - we define a schema and ask
the model to fill it:

    result = llm.with_structured_output(QueryClassification).invoke(...)

LangChain converts the schema to the provider's native mechanism
(function/tool calling on Gemini, JSON schema mode on Ollama) and validates
the reply into a Pydantic object. If the model returns something invalid,
Pydantic raises - which the calling code catches and falls back to a
deterministic default (see D2P1.py and D2P3.py).

Day 2 adds:
    FinanceAnswer / QueryClassification - carried over from Day 1
    TaskPlan / PlannedTask              - Session 6 task planner output
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# The six query categories understood by the FinDesk router (Day 1 Session 3).
# Kept so Day 2 code can classify questions with the same vocabulary.
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
    """Structured answer produced in Session 1 (`/structured` command)."""

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
    """Structured classification produced by the Day 1 Session 3 router."""

    category: QueryCategory = Field(
        description="Best matching category for the employee's question"
    )
    reasoning: str = Field(
        description="One short sentence explaining why this category was chosen"
    )


# ==========================================================================
# SESSION 6 - task planner output
# ==========================================================================
# The planner LLM is asked to return exactly this object so the graph
# receives a validated list of task instructions it can dispatch to the
# parallel worker nodes.
TaskType = Literal["budget", "invoice", "reimbursement", "compliance"]


class PlannedTask(BaseModel):
    """One independent task produced by the Session 6 planner."""

    task_type: TaskType = Field(
        description="Which worker handles this task: budget, invoice, "
                    "reimbursement or compliance"
    )
    instruction: str = Field(
        description="One short sentence describing what to analyse"
    )


class TaskPlan(BaseModel):
    """Full plan returned by the planner node.

    D2P3 validates this object; if the LLM call fails the graph falls back
    to a fixed four-task plan so the workflow always has work to dispatch.
    """

    department: str = Field(
        description="Department the request is about, e.g. 'Department A'; "
                    "use 'Department A' if not mentioned"
    )
    tasks: list[PlannedTask] = Field(
        description="3 to 5 independent tasks covering the request"
    )
