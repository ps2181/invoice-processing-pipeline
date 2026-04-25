"""
Gradio Web UI for Invoice Processing Pipeline
=============================================
Interactive tester — pick a task, see the invoice, run the LLM agent
or paste your own JSON, then inspect the grader feedback & score.

Mounted at /web on the main FastAPI app.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Tuple

import gradio as gr
import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Helpers — thin HTTP client talking to the same server
# ---------------------------------------------------------------------------

_SERVER_URL = "http://localhost:7860"


def _post(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    try:
        r = httpx.post(f"{_SERVER_URL}{path}", json=body, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _get(path: str) -> Dict[str, Any]:
    try:
        r = httpx.get(f"{_SERVER_URL}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# LLM agent helper
# ---------------------------------------------------------------------------

def _call_llm(task_id: str, obs: Dict[str, Any], step: int) -> Tuple[str, str]:
    """Call the configured LLM and return (json_str, status_msg)."""
    try:
        from openai import OpenAI
        from inference import SYSTEM_PROMPTS, build_user_prompt, MODEL_NAME, API_BASE_URL, API_KEY

        if not API_KEY:
            return "{}", "⚠️  No API key found — set HF_TOKEN or API_KEY env var."

        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        user_prompt = build_user_prompt(task_id, obs, step)

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS[task_id]},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        raw = (completion.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

        parsed = json.loads(raw)
        return json.dumps(parsed, indent=2), f"✅ LLM ({MODEL_NAME}) responded. Review then Submit."
    except json.JSONDecodeError as e:
        return "{}", f"❌ LLM returned invalid JSON: {e}"
    except Exception as e:
        return "{}", f"❌ LLM error: {e}"


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------

TASK_DESCRIPTIONS = {
    "easy":         "Extract structured fields from a single clean invoice.",
    "medium":       "Clean & normalise a batch of messy invoices (typos, date formats, currencies).",
    "hard":         "Clean invoices AND reconcile against purchase orders. Flag discrepancies.",
    "expert":       "Audit invoices for fraud: phantom vendors, price gouging, duplicates, math errors.",
    "adversarial":  "Extract from an invoice with OCR corruption, fake SUBTOTAL, and FX noise lines.",
    "negotiate":    "Ask clarification questions, then submit full extraction. Bonus for ≤2 questions.",
    "supply_chain": "Detect anomalies in delivery records: shortfalls, price spikes, substitutions, phantoms.",
}

PLACEHOLDER_JSON = "// Reset an episode first, then paste or generate JSON here."


def _get_regulator_report() -> str:
    data = _get("/regulator/report")
    forecast = _get("/regulator/forecast")
    calibration = _get("/regulator/calibration")
    if "error" in data:
        return f"Error: {data['error']}"

    lines = [
        f"Total audits recorded: {data.get('total_audits_recorded', 0)}  (window={data.get('window', 30)})",
        "",
        "DETECTION RATES  (trend: ↑ improving  ↓ declining)",
        "─" * 50,
    ]
    for ft, status in data.get("detection_rates", {}).items():
        lines.append(f"  {ft:<30} {status}")

    lines += [
        "",
        f"False Positive Rate: {data.get('false_positive_rate', 'no data')}",
        "",
        f"CRITICAL BLIND SPOTS:  {data.get('blind_spots', [])}",
        f"EMERGING BLIND SPOTS:  {data.get('emerging_blind_spots', [])}",
        "",
        "PREDICTIVE FORECAST",
        "─" * 50,
    ]
    if "error" not in forecast:
        for ft, trend in forecast.get("trends", {}).items():
            lines.append(f"  {ft:<30} {trend}")
        lines.append(f"\n  {forecast.get('recommendation', '')}")

    lines += [
        "",
        "CONFIDENCE CALIBRATION  (overconfident misses = dangerous)",
        "─" * 50,
    ]
    if "error" not in calibration:
        for ft, cal in calibration.items():
            if isinstance(cal, dict) and cal.get("status") != "no data":
                lines.append(f"  {ft:<30} {cal.get('status','')}")
                if cal.get("mean_confidence_when_wrong") is not None:
                    lines.append(f"    conf on wrong={cal['mean_confidence_when_wrong']:.2f}  conf on correct={cal.get('mean_confidence_when_correct', 'N/A')}")

    lines += [
        "",
        "GENERATOR WEIGHTS (next episode bias)",
        "─" * 50,
    ]
    for ft, w in data.get("generator_weights", {}).items():
        bar = "█" * int(w * 30)
        lines.append(f"  {ft:<30} {w:.3f}  {bar}")

    lines += ["", f"VERDICT: {data.get('verdict', '')}"]
    return "\n".join(lines)


def _seed_demo_data() -> str:
    data = _post("/regulator/demo_seed", {})
    if "error" in data:
        return f"Error: {data['error']}"
    return "✅ Demo data seeded — phantom_vendor at ~32% (blind spot), duplicate_submission declining (emerging)\n\n" + _get_regulator_report()


def build_ui() -> gr.Blocks:

    # ---- State per Gradio session ----------------------------------------
    # Stores: episode_id (str), last observation dict, step count
    init_state = {"episode_id": None, "obs": None, "step": 0, "history": []}

    # ---- Callbacks -------------------------------------------------------

    def do_reset(task_id: str, state: dict):
        data = _post("/reset", {"task_id": task_id})
        if "error" in data:
            return (
                state,
                gr.update(value=f"❌ Error: {data['error']}"),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(value=PLACEHOLDER_JSON),
                gr.update(value=""),
                gr.update(value=""),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )

        obs = data["observation"]
        ep = data["info"]["episode_id"]
        new_state = {"episode_id": ep, "obs": obs, "step": 0, "history": []}

        ref = obs.get("reference_data") or ""
        status = (
            f"✅ Episode started | task={task_id} | id={ep[:12]}…\n"
            f"Max attempts: {obs['max_attempts']}"
        )

        return (
            new_state,
            gr.update(value=status),
            gr.update(value=obs["task_description"]),
            gr.update(value=obs["raw_text"]),
            gr.update(value=ref),
            gr.update(value=PLACEHOLDER_JSON),
            gr.update(value=""),   # feedback
            gr.update(value=""),   # history
            gr.update(interactive=True),   # llm btn
            gr.update(interactive=True),   # submit btn
        )

    def do_llm(task_id: str, state: dict):
        if not state.get("obs"):
            return PLACEHOLDER_JSON, "⚠️  Reset an episode first."
        step = state["step"] + 1
        json_str, status = _call_llm(task_id, state["obs"], step)
        return json_str, status

    def do_submit(json_str: str, state: dict):
        if not state.get("episode_id"):
            return state, "⚠️  Reset an episode first.", "", "", ""

        try:
            extracted = json.loads(json_str)
        except json.JSONDecodeError as e:
            return state, f"❌ Invalid JSON: {e}", "", "", ""

        data = _post("/step", {
            "extracted_data": extracted,
            "episode_id": state["episode_id"],
        })

        if "error" in data:
            return state, f"❌ Error: {data['error']}", "", "", ""

        obs = data["observation"]
        reward = data.get("reward", 0.0)
        done = data.get("done", False)
        state["obs"] = obs
        state["step"] += 1

        # history
        entry = f"Step {state['step']}: reward={reward:.3f}" + (" ✓ done" if done else "")
        state["history"].append(entry)
        history_str = "\n".join(state["history"])

        feedback = obs.get("feedback") or "No feedback yet."

        bd = obs.get("reward_breakdown")
        breakdown_str = json.dumps(bd, indent=2) if bd else ""

        status = (
            f"Step {state['step']} / {obs['max_attempts']} | "
            f"Reward: {reward:.3f} | "
            f"{'🏁 Done' if done else 'In progress…'}"
        )

        return state, status, feedback, history_str, breakdown_str

    # ---- Layout ----------------------------------------------------------

    with gr.Blocks(
        title="Invoice Processing Pipeline",
        theme=gr.themes.Soft(),
        css=".gr-prose { font-family: monospace; }",
    ) as demo:

        gr.Markdown(
            "# 🧾 Invoice Processing Pipeline\n"
            "Interactive agent tester — select a task, reset to load an invoice, "
            "then use the LLM agent or paste your own JSON and submit."
        )

        session_state = gr.State(init_state)

        with gr.Tabs():

            # ================================================================
            # Tab 1 — Agent Tester
            # ================================================================
            with gr.Tab("Agent Tester"):

                # --- Controls row -----------------------------------------
                with gr.Row():
                    task_dd = gr.Dropdown(
                        choices=list(TASK_DESCRIPTIONS.keys()),
                        value="easy",
                        label="Task",
                        scale=1,
                    )
                    reset_btn = gr.Button("🔄 Reset Episode", variant="primary", scale=1)
                    status_box = gr.Textbox(
                        label="Status",
                        interactive=False,
                        scale=3,
                        lines=2,
                    )

                task_info = gr.Textbox(label="Task Description", interactive=False, lines=1)

                # --- Main two-column layout --------------------------------
                with gr.Row():
                    with gr.Column(scale=5):
                        invoice_box = gr.Textbox(
                            label="Invoice Data (raw text)",
                            interactive=False,
                            lines=16,
                            max_lines=30,
                        )
                        ref_box = gr.Textbox(
                            label="Reference Data (PO / vendor registry / catalog)",
                            interactive=False,
                            lines=8,
                            max_lines=16,
                        )

                    with gr.Column(scale=5):
                        json_box = gr.Code(
                            label="Extracted JSON",
                            language="json",
                            lines=16,
                            value=PLACEHOLDER_JSON,
                        )
                        with gr.Row():
                            llm_btn = gr.Button(
                                "🤖 Run LLM Agent",
                                variant="secondary",
                                interactive=False,
                            )
                            submit_btn = gr.Button(
                                "✅ Submit",
                                variant="primary",
                                interactive=False,
                            )
                        llm_status = gr.Textbox(
                            label="LLM status",
                            interactive=False,
                            lines=1,
                        )

                # --- Results row ------------------------------------------
                with gr.Row():
                    feedback_box = gr.Textbox(
                        label="Grader Feedback",
                        interactive=False,
                        lines=5,
                        scale=3,
                    )
                    breakdown_box = gr.Code(
                        label="Reward Breakdown",
                        language="json",
                        lines=5,
                        interactive=False,
                        scale=2,
                    )

                history_box = gr.Textbox(
                    label="Step History",
                    interactive=False,
                    lines=3,
                )

                # --- Wiring -----------------------------------------------
                task_dd.change(
                    fn=lambda t: TASK_DESCRIPTIONS.get(t, ""),
                    inputs=[task_dd],
                    outputs=[task_info],
                )

                reset_btn.click(
                    fn=do_reset,
                    inputs=[task_dd, session_state],
                    outputs=[
                        session_state, status_box, task_info,
                        invoice_box, ref_box, json_box,
                        feedback_box, history_box,
                        llm_btn, submit_btn,
                    ],
                )

                llm_btn.click(
                    fn=do_llm,
                    inputs=[task_dd, session_state],
                    outputs=[json_box, llm_status],
                )

                submit_btn.click(
                    fn=do_submit,
                    inputs=[json_box, session_state],
                    outputs=[session_state, status_box, feedback_box, history_box, breakdown_box],
                )

            # ================================================================
            # Tab 2 — Regulator Dashboard
            # ================================================================
            with gr.Tab("Regulator Dashboard"):

                gr.Markdown(
                    "## Regulator — Cross-Episode Auditor Oversight\n"
                    "Monitors Auditor detection rates over 30 episodes. "
                    "Detects blind spots and biases the Generator toward under-detected fraud types."
                )

                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh Report", variant="primary")
                    seed_btn = gr.Button("🌱 Seed Demo Data", variant="secondary")

                report_box = gr.Textbox(
                    label="Regulator Report",
                    interactive=False,
                    lines=22,
                    value="Click 'Refresh Report' or 'Seed Demo Data' to load.",
                )

                refresh_btn.click(fn=_get_regulator_report, inputs=[], outputs=[report_box])
                seed_btn.click(fn=_seed_demo_data, inputs=[], outputs=[report_box])

    return demo
