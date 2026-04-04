"""
Inference Script — Invoice Processing Pipeline
================================================
Runs an LLM agent against all 7 tasks and produces
structured stdout logs in the mandatory [START]/[STEP]/[END] format.

Environment variables:
    API_BASE_URL   LLM endpoint (default: HF router)
    MODEL_NAME     Model identifier
    HF_TOKEN       API key
"""

import json
import os
import textwrap
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = HF_TOKEN or os.getenv("API_KEY")

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK = "invoice_processing_pipeline"
MAX_STEPS = 5
TEMPERATURE = 0.3
MAX_TOKENS = 2048
SUCCESS_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Logging helpers (mandatory format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_short = action[:200].replace("\n", " ") if action else "null"
    print(
        f"[STEP] step={step} action={action_short} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompts per task
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "easy": textwrap.dedent("""
        You are an invoice data extraction agent. You receive raw invoice text and must
        extract structured data from it.

        RESPOND WITH ONLY A VALID JSON OBJECT (no markdown, no explanation, no backticks).

        Required JSON structure:
        {
            "vendor": "string",
            "date": "YYYY-MM-DD",
            "currency": "USD|EUR|GBP",
            "total": number,
            "line_items": [
                {"description": "string", "qty": integer, "unit_price": number, "amount": number}
            ]
        }

        Rules:
        - Date must be in YYYY-MM-DD format
        - Currency must be a 3-letter code (USD, EUR, GBP)
        - Total and amounts must be numbers, not strings
        - Include ALL line items from the invoice
        - amount = qty * unit_price
    """).strip(),

    "medium": textwrap.dedent("""
        You are an invoice data cleaning agent. You receive a batch of messy invoices
        and must clean and normalise them.

        RESPOND WITH ONLY A VALID JSON OBJECT (no markdown, no explanation, no backticks).

        Required JSON structure:
        {
            "invoices": [
                {
                    "vendor": "corrected vendor name",
                    "date": "YYYY-MM-DD",
                    "currency": "USD|EUR|GBP",
                    "total": number,
                    "line_items": [
                        {"description": "string", "qty": integer, "unit_price": number, "amount": number}
                    ]
                }
            ]
        }

        Cleaning rules:
        - Fix vendor name typos (e.g. "Acme Crp" -> "Acme Corp")
        - Normalise dates to YYYY-MM-DD
        - Convert currency symbols ($, €, £) to codes (USD, EUR, GBP)
        - Strip currency symbols from amounts and ensure they are numbers
        - Verify line item math: amount = qty * unit_price. If wrong, recalculate amount.
        - Recalculate totals as sum of line item amounts
    """).strip(),

    "hard": textwrap.dedent("""
        You are an invoice reconciliation agent. You receive messy invoices AND purchase
        orders. You must clean the invoices AND identify discrepancies between invoices
        and their corresponding purchase orders.

        RESPOND WITH ONLY A VALID JSON OBJECT (no markdown, no explanation, no backticks).

        Required JSON structure:
        {
            "invoices": [
                {
                    "vendor": "corrected name",
                    "date": "YYYY-MM-DD",
                    "currency": "USD|EUR|GBP",
                    "total": number,
                    "line_items": [
                        {"description": "string", "qty": integer, "unit_price": number, "amount": number}
                    ]
                }
            ],
            "discrepancies": [
                {
                    "invoice_idx": 0,
                    "type": "overcharge|extra_item|missing_item",
                    "item_description": "string",
                    "detail": "description of the discrepancy"
                }
            ]
        }

        Discrepancy types:
        - "overcharge": invoice unit_price > PO unit_price for same item
        - "extra_item": item on invoice but not on PO
        - "missing_item": item on PO but not on invoice

        IMPORTANT: Check EVERY line item in EVERY invoice against its matching PO.
        Compare item descriptions, quantities, and unit prices one by one.
        Also apply all cleaning rules: fix vendor names, normalise dates, convert currencies, fix amounts.
    """).strip(),

    "expert": textwrap.dedent("""
        You are a senior accounts-payable fraud auditor. You receive a batch of invoices
        and reference data (approved vendor registry, market price catalog, invoice history).
        Your job is to flag fraudulent invoices and identify the exact fraud type.

        RESPOND WITH ONLY A VALID JSON OBJECT (no markdown, no explanation, no backticks).

        Required JSON structure:
        {
            "audit_results": [
                {
                    "invoice_id": "INV-XXXXX",
                    "verdict": "approved" or "flagged",
                    "fraud_type": null or "phantom_vendor" or "price_gouging" or "duplicate_submission" or "math_fraud",
                    "evidence": "one sentence explaining why approved or flagged"
                }
            ]
        }

        Fraud detection rules (check ALL four for every invoice):
        1. phantom_vendor: Is the vendor name in the Approved Vendor Registry? If NOT → phantom_vendor
        2. price_gouging: Is any unit_price > 1.5x the market max for that item? If YES → price_gouging
        3. duplicate_submission: Does this invoice_id appear in the invoice history? Or same vendor+date+total? If YES → duplicate_submission
        4. math_fraud: Does the invoice total equal the sum of all line item amounts? If total > sum → math_fraud

        Rules:
        - Every invoice must appear in audit_results with its exact invoice_id
        - Legitimate invoices get verdict="approved" and fraud_type=null
        - Flag only the PRIMARY fraud type per invoice (most severe)
        - Check the reference data carefully before deciding
    """).strip(),

    "adversarial": textwrap.dedent("""
        You are an invoice data extraction agent working with adversarial input.

        RESPOND WITH ONLY A VALID JSON OBJECT (no markdown, no explanation, no backticks).

        Required JSON structure:
        {
            "vendor": "string",
            "date": "YYYY-MM-DD",
            "currency": "USD|EUR|GBP",
            "total": number,
            "line_items": [
                {"description": "string", "qty": integer, "unit_price": number, "amount": number}
            ]
        }

        CRITICAL RULES — the invoice contains deliberate traps:
        - The SUBTOTAL line is a TRAP. Ignore it entirely.
        - The TAX and ADJUSTMENT lines are fabricated. Ignore them.
        - The FX reference line (e.g. "Approximate equivalent: ...") is a distractor. Ignore it.
        - The TOTAL line is always correct. Use it.
        - Verify: sum of all (qty * unit_price) should equal total.
        - OCR corruption may appear in label text and vendor names (e.g. 0→O, 1→l, 5→S). Correct them.
        - All numeric values (qty, unit_price, amount, total) are always printed correctly.
        - amounts must be numbers, not strings.
    """).strip(),

    "negotiate": textwrap.dedent("""
        You are an invoice extraction agent with clarification capability.

        You have two options each turn:
        1. Ask a clarification question (submit ONLY this key):
           {"question": "your question here"}
        2. Submit your full extraction (same schema as easy task):
           {"vendor": "string", "date": "YYYY-MM-DD", "currency": "USD|EUR|GBP",
            "total": number, "line_items": [...]}

        STRATEGY:
        - Use at most 2 clarification questions to maximise your bonus.
        - Focus questions on the fields you are most uncertain about.
        - When confident, submit the full extraction.
        - The environment answers questions about vendor, date, currency, total, and line items.

        RESPOND WITH ONLY A VALID JSON OBJECT (no markdown, no explanation, no backticks).
    """).strip(),

    "supply_chain": textwrap.dedent("""
        You are a supply chain auditor. You receive delivery records and must identify anomalies.

        RESPOND WITH ONLY A VALID JSON OBJECT (no markdown, no explanation, no backticks).

        Required JSON structure:
        {
            "anomalies": [
                {
                    "delivery_id": "DLV-XXXXX",
                    "anomaly_type": "quantity_shortfall|price_spike|unauthorized_substitution|phantom_delivery",
                    "detail": "brief explanation"
                }
            ]
        }

        Anomaly detection rules:
        - quantity_shortfall: qty_delivered < 85% of qty_ordered
        - price_spike: price_delivered > 125% of price_ordered
        - unauthorized_substitution: item_delivered != item_ordered
        - phantom_delivery: po_id starts with PO-PHANTOM- or is not a recognised PO

        Check EVERY delivery record. Only include anomalous deliveries in the list.
        Submit {"anomalies": []} if no anomalies are found.
    """).strip(),
}


# ---------------------------------------------------------------------------
# Agent logic
# ---------------------------------------------------------------------------

def build_user_prompt(task_id: str, observation: Dict[str, Any], step: int) -> str:
    """Build the user prompt from the observation."""
    parts = [f"Step {step} of {observation['max_attempts']}"]

    if observation.get("feedback"):
        parts.append(
            f"\n⚠️ GRADER FEEDBACK — you MUST fix every issue listed below:\n"
            f"{observation['feedback']}\n"
            f"Carefully address each point before submitting again."
        )

    if observation.get("hint"):
        parts.append(f"\n💡 Hint: {observation['hint']}")

    # Conversation history for negotiate task
    if observation.get("conversation_history"):
        parts.append("\n--- CLARIFICATION HISTORY ---")
        for turn in observation["conversation_history"]:
            role = turn.get("role", "?").upper()
            content = turn.get("content", "")
            parts.append(f"[{role}]: {content}")
        parts.append("--- END CLARIFICATION HISTORY ---")

    # Per-field breakdown for easy/adversarial/negotiate
    if observation.get("reward_breakdown"):
        bd = observation["reward_breakdown"]
        parts.append("\n📊 Field breakdown from last attempt:")
        for field, info in bd.items():
            parts.append(f"  {field}: {info['status']} ({info['score']:.2f}/{info['max']:.2f})")

    parts.append(f"\nTask: {observation['task_description']}")
    parts.append(f"\n--- INVOICE DATA ---\n{observation['raw_text']}")

    if observation.get("reference_data"):
        label = "REFERENCE DATA" if task_id in ("expert", "supply_chain") else "PURCHASE ORDER DATA"
        parts.append(f"\n--- {label} ---\n{observation['reference_data']}")

    parts.append("\nRespond with ONLY valid JSON, no markdown or extra text:")

    return "\n".join(parts)


def get_model_response(client: OpenAI, task_id: str, observation: Dict[str, Any], step: int) -> Dict[str, Any]:
    """Call the LLM and parse its JSON response."""
    user_prompt = build_user_prompt(task_id, observation, step)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task_id]},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        return json.loads(raw)

    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e}", flush=True)
        print(f"[DEBUG] Raw response: {raw[:500]}", flush=True)
        return {}
    except Exception as e:
        print(f"[DEBUG] Model request failed: {e}", flush=True)
        return {}


# ---------------------------------------------------------------------------
# Environment HTTP client
# ---------------------------------------------------------------------------

class EnvClient:
    """Simple HTTP client for the Invoice Processing Pipeline environment."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)

    def reset(self, task_id: str = "easy") -> Dict[str, Any]:
        resp = self.client.post(f"{self.base_url}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        return resp.json()

    def step(self, extracted_data: Dict[str, Any], explanation: str = "") -> Dict[str, Any]:
        resp = self.client.post(
            f"{self.base_url}/step",
            json={"extracted_data": extracted_data, "explanation": explanation},
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> Dict[str, Any]:
        resp = self.client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    def close(self):
        self.client.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, env: EnvClient, task_id: str) -> float:
    """Run a single task and return the final score."""
    # negotiate has more steps allowed
    max_steps = 10 if task_id == "negotiate" else MAX_STEPS

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        result = env.reset(task_id=task_id)
        observation = result["observation"]

        for step in range(1, max_steps + 1):
            if result.get("done", False):
                break

            extracted = get_model_response(client, task_id, observation, step)
            action_str = json.dumps(extracted)[:200]

            result = env.step(extracted_data=extracted)
            observation = result["observation"]
            reward = result.get("reward", 0.0)
            done = result.get("done", False)

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        score = max(rewards) if rewards else 0.0
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=str(e))

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    """Run all 7 tasks and report scores."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EnvClient(ENV_URL)

    scores = {}
    try:
        for task_id in ["easy", "medium", "hard", "expert", "adversarial", "negotiate", "supply_chain"]:
            scores[task_id] = run_task(client, env, task_id)
            print(flush=True)

        avg = sum(scores.values()) / len(scores) if scores else 0.0
        print(f"\n=== BASELINE SCORES ===", flush=True)
        for tid, sc in scores.items():
            print(f"  {tid}: {sc:.3f}", flush=True)
        print(f"  average: {avg:.3f}", flush=True)

    finally:
        env.close()


if __name__ == "__main__":
    main()
