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
            "Hard: {invoices: [...], discrepancies: [{invoice_idx, type, detail, expected, actual}]}. "
            "Adversarial: same schema as easy — {vendor, date, currency, total, line_items}. "
            "Negotiate: either {'question': str} to ask a clarification, or the full extraction "
            "(same schema as easy). "
            "Supply_chain: {'anomalies': [{'delivery_id', 'anomaly_type', 'detail'}]}."
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
    task_id: str = Field(..., description="easy | medium | hard | expert | adversarial | negotiate | supply_chain")
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
    reward_breakdown: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Per-field score breakdown for easy, adversarial, and negotiate tasks. "
            "Example: {'vendor': {'score': 0.15, 'max': 0.15, 'status': 'correct'}, "
            "'date': {'score': 0.0, 'max': 0.10, 'status': 'wrong'}, ...}"
        ),
    )
    conversation_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="For negotiate task: list of {'role': 'agent'|'env', 'content': str} turns.",
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
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    clarification_count: int = Field(default=0)


# ---------------------------------------------------------------------------
# Supply Chain (documentation model)
# ---------------------------------------------------------------------------

class SupplyChainAnomalyItem(BaseModel):
    delivery_id: str
    anomaly_type: str  # quantity_shortfall | price_spike | unauthorized_substitution | phantom_delivery
    detail: str


class SupplyChainAction(BaseModel):
    anomalies: List[SupplyChainAnomalyItem]
