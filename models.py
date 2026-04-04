"""
Pydantic models for the Invoice Processing Pipeline environment.

Action:  Agent submits extracted/cleaned/reconciled invoice data as JSON.
Observation: Agent receives raw invoice text, feedback, and task context.
State:   Tracks episode progress, attempts, and scores.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class InvoiceAction(BaseModel):
    """Action the agent submits each step."""

    extracted_data: Dict[str, Any] = Field(
        ...,
        description=(
            "JSON object with extracted/cleaned invoice fields. "
            "Structure depends on the task. "
            "Easy: {vendor, date, currency, total, line_items: [{description, qty, unit_price, amount}]}. "
            "Medium: {invoices: [{vendor, date, currency, total, line_items}]} (batch of cleaned invoices). "
            "Hard: {invoices: [...], discrepancies: [{invoice_idx, type, detail, expected, actual}]}."
        ),
    )
    explanation: str = Field(
        default="",
        description="Optional reasoning about extraction or cleaning decisions.",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class InvoiceObservation(BaseModel):
    """What the agent sees each turn."""

    raw_text: str = Field(..., description="Raw invoice text (OCR-style or CSV-style)")
    task_id: str = Field(..., description="easy | medium | hard")
    difficulty: str = Field(..., description="Same as task_id")
    task_description: str = Field(..., description="What the agent should do")
    attempt_number: int = Field(default=0, description="Current attempt (0 = just reset)")
    max_attempts: int = Field(default=5, description="Max allowed attempts")
    feedback: str = Field(default="", description="Detailed grader feedback from last attempt")
    hint: str = Field(default="", description="Hint shown after 2+ failed attempts")
    reference_data: str = Field(
        default="",
        description="For hard task: purchase order data to reconcile against",
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class InvoiceState(BaseModel):
    """Internal episode state."""

    episode_id: str = Field(default="")
    task_id: str = Field(default="easy")
    step_count: int = Field(default=0)
    done: bool = Field(default=False)
    last_reward: float = Field(default=0.0)
    best_reward: float = Field(default=0.0)
    rewards: List[float] = Field(default_factory=list)