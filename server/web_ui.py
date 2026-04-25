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


def _run_pipeline_episode() -> str:
    """
    Run one complete multi-agent episode and return a formatted step-by-step trace.
    Generator → Extractor (rule-based demo) → Auditor (rule-based demo) → Approver → Regulator
    """
    lines = ["=" * 56, "  MULTI-AGENT PIPELINE — LIVE EPISODE", "=" * 56, ""]

    # ── Step 0: Regulator sets Generator weights ──────────────────
    report = _get("/regulator/report")
    blind_spots = report.get("blind_spots", [])
    weights = report.get("generator_weights", {})
    lines += [
        "STEP 0 — REGULATOR sets Generator weights",
        "─" * 40,
        f"  Blind spots detected: {blind_spots if blind_spots else 'none'}",
    ]
    for ft, w in weights.items():
        bar = "█" * max(1, int(w * 20))
        lines.append(f"  {ft:<28} {w:.2f}  {bar}")
    lines.append("")

    # ── Step 1: Generator creates biased episode ──────────────────
    ep = _post("/multi/reset", {})
    if "error" in ep:
        return f"Error starting episode: {ep['error']}"
    episode_id = ep["episode_id"]
    n_inv = ep["n_invoices"]
    fw = ep.get("fraud_weights_used", {})
    raw_text = ep.get("raw_text", "")

    # Figure out dominant fraud type from weights
    dominant = max(fw, key=lambda k: fw.get(k, 0)) if fw else "unknown"

    lines += [
        "STEP 1 — GENERATOR creates invoice batch",
        "─" * 40,
        f"  Episode ID:    {episode_id[:16]}…",
        f"  Invoices:      {n_inv}",
        f"  Dominant type: {dominant} ({fw.get(dominant, 0):.0%} weight — Regulator-biased)",
        f"  Invoice preview:",
    ]
    for line in raw_text.split("\n")[:12]:
        lines.append(f"    {line}")
    lines.append("    …")
    lines.append("")

    # ── Step 2: Extractor reads the invoice ───────────────────────
    # Use rule-based extraction for demo (no LLM needed)
    import re as _re
    vendor_match = _re.search(r"Vendor:\s*(.+)", raw_text)
    date_match   = _re.search(r"Date:\s*(\d{4}-\d{2}-\d{2})", raw_text)
    total_match  = _re.search(r"TOTAL\s+[\$£€]?([\d,.]+)", raw_text)
    vendor = vendor_match.group(1).strip() if vendor_match else "Unknown Vendor"
    date   = date_match.group(1).strip() if date_match else "2024-01-01"
    total  = float(total_match.group(1).replace(",", "")) if total_match else 0.0

    extracted = {
        "vendor": vendor, "date": date, "currency": "USD",
        "total": total,
        "line_items": [{"description": "Office Supplies", "qty": 1, "unit_price": total, "amount": total}]
    }
    ext_result = _post("/multi/extract", {"episode_id": episode_id, "extracted_data": extracted})
    ext_reward = ext_result.get("reward", 0)
    ext_bd = ext_result.get("breakdown", {})

    lines += [
        "STEP 2 — EXTRACTOR reads invoice → structured JSON",
        "─" * 40,
        f"  Vendor extracted:  {vendor}",
        f"  Date extracted:    {date}",
        f"  Total extracted:   {total}",
        f"  Extractor reward:  {ext_reward:.3f}",
        f"  Breakdown: format={ext_bd.get('format',0):.2f}  field={ext_bd.get('field_accuracy',0):.2f}  "
        f"math={ext_bd.get('math_consistency',0):.2f}  completeness={ext_bd.get('completeness',0):.2f}",
        "",
    ]

    # ── Step 3: Auditor reviews for fraud ─────────────────────────
    # Build audit results for all invoices in episode
    inv_ids = _re.findall(r"ID:\s*(INV-\d+)", raw_text)
    if not inv_ids:
        inv_ids = [f"INV-{i:05d}" for i in range(n_inv)]

    # Rule-based audit: flag phantom vendors (not in known list)
    known = ["acme corp","globaltech solutions","prime office supplies","datastream inc",
             "cloudnine services","metro logistics","pinnacle electronics","summit consulting",
             "vertex manufacturing","horizon digital","nexgen software","bluepeak analytics"]
    audit_results = []
    for inv_id in inv_ids[:n_inv]:
        is_phantom = vendor.lower() not in known
        audit_results.append({
            "invoice_id": inv_id,
            "verdict": "flagged" if is_phantom else "approved",
            "fraud_type": "phantom_vendor" if is_phantom else None,
            "confidence": 0.78 if is_phantom else 0.85,
        })

    aud_result = _post("/multi/audit", {"episode_id": episode_id, "audit_results": audit_results})
    aud_reward = aud_result.get("mean_reward", 0)
    aud_feedback = aud_result.get("feedback", "")
    new_report = aud_result.get("tracker_report", {})

    lines += [
        "STEP 3 — AUDITOR reviews for fraud",
        "─" * 40,
    ]
    for r in audit_results:
        verdict_str = f"FLAGGED ({r['fraud_type']})" if r["verdict"] == "flagged" else "APPROVED"
        lines.append(f"  {r['invoice_id']}  →  {verdict_str}  conf={r['confidence']:.2f}")
    lines += [
        f"  Mean Auditor reward: {aud_reward:.3f}",
        f"  Feedback: {aud_feedback[:120]}",
        "",
    ]

    # ── Step 4: Approver final decision ───────────────────────────
    approver_decisions = []
    for r in audit_results:
        if r["verdict"] == "flagged" and r["confidence"] >= 0.80:
            decision = "REJECT"
        elif r["verdict"] == "flagged":
            decision = "ESCALATE"
        else:
            decision = "APPROVE"
        approver_decisions.append((r["invoice_id"], decision))

    lines += [
        "STEP 4 — APPROVER final decision",
        "─" * 40,
    ]
    for inv_id, decision in approver_decisions:
        icon = "❌" if decision == "REJECT" else "⚠️" if decision == "ESCALATE" else "✅"
        lines.append(f"  {inv_id}  →  {icon} {decision}")
    lines.append("")

    # Generator adversarial reward
    n_evaded = sum(1 for r in audit_results if r["verdict"] == "approved")
    gen_reward = 0.85 if n_evaded == n_inv else (0.60 if n_evaded > 0 else 0.10)
    lines += [
        f"  Generator adversarial reward: {gen_reward:.2f}",
        f"  ({n_evaded}/{n_inv} invoices evaded Auditor)",
        "",
    ]

    # ── Step 5: Regulator updates ─────────────────────────────────
    new_blind_spots = new_report.get("blind_spots", [])
    new_emerging = new_report.get("emerging_blind_spots", [])
    new_weights = new_report.get("generator_weights", {})

    lines += [
        "STEP 5 — REGULATOR updates cross-episode tracker",
        "─" * 40,
        f"  Total audits recorded: {new_report.get('total_audits_recorded', '?')}",
        f"  Critical blind spots:  {new_blind_spots if new_blind_spots else 'none'}",
        f"  Emerging blind spots:  {new_emerging if new_emerging else 'none'}",
        "",
        "  Updated Generator weights for next episode:",
    ]
    for ft, w in new_weights.items():
        changed = "  ← BOOSTED" if w > 0.3 else ""
        lines.append(f"    {ft:<28} {w:.3f}{changed}")

    lines += [
        "",
        "=" * 56,
        "  LOOP COMPLETE — next episode uses updated weights",
        "=" * 56,
    ]
    return "\n".join(lines)


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
            # Tab 2 — Multi-Agent Pipeline Demo
            # ================================================================
            with gr.Tab("Multi-Agent Pipeline"):

                gr.Markdown(
                    "## Live 5-Agent Pipeline\n"
                    "Runs one complete episode through all agents in sequence:\n\n"
                    "**Regulator** sets weights → **Generator** creates biased invoice → "
                    "**Extractor** reads it → **Auditor** flags fraud → "
                    "**Approver** decides → **Regulator** updates tracker\n\n"
                    "Each run uses real live data from the deployed environment."
                )

                run_btn = gr.Button("▶ Run Full Pipeline Episode", variant="primary", size="lg")
                pipeline_output = gr.Textbox(
                    label="Pipeline Trace",
                    interactive=False,
                    lines=40,
                    value="Click 'Run Full Pipeline Episode' to start.",
                    elem_id="pipeline_trace",
                )

                run_btn.click(fn=_run_pipeline_episode, inputs=[], outputs=[pipeline_output])

            # ================================================================
            # Tab 3 — Regulator Dashboard
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
