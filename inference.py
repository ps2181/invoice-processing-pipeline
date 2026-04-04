"""
Inference Script — Invoice Processing Pipeline
================================================
Runs an LLM agent against all 3 tasks (easy, medium, hard) and produces
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

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

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
    # Truncate action for readability
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

        Also apply all cleaning rules: fix vendor names, normalise dates, convert currencies, fix amounts.
    """).strip(),
}


# ---------------------------------------------------------------------------
# Agent logic
# ---------------------------------------------------------------------------

def build_user_prompt(task_id: str, observation: Dict[str, Any], step: int) -> str:
    """Build the user prompt from the observation."""
    parts = [f"Step {step} of {observation['max_attempts']}"]

    if observation.get("feedback"):
        parts.append(f"\nFeedback from previous attempt:\n{observation['feedback']}")

    if observation.get("hint"):
        parts.append(f"\nHint: {observation['hint']}")

    parts.append(f"\nTask: {observation['task_description']}")
    parts.append(f"\n--- RAW INVOICE DATA ---\n{observation['raw_text']}")

    if observation.get("reference_data"):
        parts.append(f"\n--- PURCHASE ORDER DATA ---\n{observation['reference_data']}")

    parts.append("\nExtract/clean the data and respond with ONLY valid JSON:")

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
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        result = env.reset(task_id=task_id)
        observation = result["observation"]

        for step in range(1, MAX_STEPS + 1):
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
    """Run all 3 tasks and report scores."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = EnvClient(ENV_URL)

    scores = {}
    try:
        for task_id in ["easy", "medium", "hard"]:
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