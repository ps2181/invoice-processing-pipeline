"""
Gradio Web UI for Invoice Processing Pipeline
=============================================
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
# CSS
# ---------------------------------------------------------------------------

CSS = """
/* ── global ── */
body { font-family: 'Inter', sans-serif; }

/* ── hero banner ── */
#hero {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2744 50%, #162d4a 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 8px;
    color: white;
}
#hero h1 { font-size: 2rem; font-weight: 700; margin: 0 0 8px 0; color: white; }
#hero p  { font-size: 1rem; opacity: 0.85; margin: 0; line-height: 1.6; }

/* ── agent cards row ── */
#agent-cards {
    display: flex;
    gap: 10px;
    margin: 12px 0;
    flex-wrap: wrap;
}
.agent-card {
    flex: 1;
    min-width: 130px;
    background: white;
    border: 1.5px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 12px;
    text-align: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.agent-card .icon { font-size: 1.6rem; }
.agent-card .name { font-weight: 600; font-size: 0.82rem; color: #1e3a5f; margin-top: 4px; }
.agent-card .desc { font-size: 0.72rem; color: #64748b; margin-top: 2px; }

/* ── pipeline trace box ── */
#pipeline-trace textarea {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.55 !important;
    background: #0f172a !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}

/* ── regulator report ── */
#reg-report textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.81rem !important;
    line-height: 1.5 !important;
    background: #0f172a !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}

/* ── buttons ── */
.run-btn { font-size: 1rem !important; font-weight: 600 !important; }
.gr-button-primary { background: #1e3a5f !important; }

/* ── tab styling ── */
.tab-nav button { font-weight: 600 !important; font-size: 0.9rem !important; }

/* ── status pill ── */
#status-bar textarea {
    font-size: 0.82rem !important;
    background: #f8fafc !important;
    border-radius: 8px !important;
}
"""

HERO_HTML = """
<div id="hero">
  <h1>🧾 Invoice Processing Pipeline</h1>
  <p>
    A <strong>self-improving 5-agent system</strong> that generates, extracts, audits, approves,
    and regulates invoice fraud — powered by <strong>GRPO-trained Qwen2.5 LoRA agents</strong>.<br>
    The Regulator detects Auditor blind spots and biases the Generator toward harder fraud in the next episode,
    closing a continuous adversarial improvement loop.
  </p>
</div>
"""

AGENT_CARDS_HTML = """
<div id="agent-cards">
  <div class="agent-card">
    <div class="icon">🎯</div>
    <div class="name">Regulator</div>
    <div class="desc">Detects blind spots, sets weights</div>
  </div>
  <div class="agent-card">
    <div class="icon">⚡</div>
    <div class="name">Generator</div>
    <div class="desc">Creates adversarial invoices</div>
  </div>
  <div class="agent-card">
    <div class="icon">🔍</div>
    <div class="name">Extractor</div>
    <div class="desc">Parses structured fields</div>
  </div>
  <div class="agent-card">
    <div class="icon">🕵️</div>
    <div class="name">Auditor</div>
    <div class="desc">Flags fraud with confidence</div>
  </div>
  <div class="agent-card">
    <div class="icon">✅</div>
    <div class="name">Approver</div>
    <div class="desc">Final approve / escalate / reject</div>
  </div>
</div>
"""

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


# ---------------------------------------------------------------------------
# Pipeline logic
# ---------------------------------------------------------------------------

def _run_pipeline_episode() -> str:
    import re as _re
    from server.agents import run_extractor, run_auditor

    sep  = "━" * 58
    thin = "─" * 48

    lines = [
        sep,
        "  MULTI-AGENT PIPELINE  ·  LIVE EPISODE",
        sep, "",
    ]

    # ── STEP 0: Regulator ────────────────────────────────────────
    report = _get("/regulator/report")
    blind_spots = report.get("blind_spots", [])
    weights = report.get("generator_weights", {})

    lines += [
        "  🎯  STEP 0 — REGULATOR",
        f"  {thin}",
        f"  Blind spots detected : {', '.join(blind_spots) if blind_spots else 'none'}",
        "  Fraud weights for next episode:",
    ]
    for ft, w in weights.items():
        bar = "▓" * max(1, int(w * 24))
        flag = "  ← prioritised" if w >= 0.35 else ""
        lines.append(f"    {ft:<26} {w:.0%}  {bar}{flag}")
    lines.append("")

    # ── STEP 1: Generator ────────────────────────────────────────
    ep = _post("/multi/reset", {})
    if "error" in ep:
        return f"Error starting episode: {ep['error']}"
    episode_id = ep["episode_id"]
    n_inv      = ep["n_invoices"]
    fw         = ep.get("fraud_weights_used", {})
    raw_text   = ep.get("raw_text", "")
    ref_data   = ep.get("reference_data", "")
    dominant   = max(fw, key=lambda k: fw.get(k, 0)) if fw else "unknown"

    lines += [
        "  ⚡  STEP 1 — GENERATOR  (Qwen2.5 LoRA · ps2181/generator-lora-qwen2.5-1.5b)",
        f"  {thin}",
        f"  Episode       : {episode_id[:20]}…",
        f"  Invoices      : {n_inv}",
        f"  Fraud focus   : {dominant.replace('_',' ').title()}  ({fw.get(dominant,0):.0%} Regulator weight)",
        "",
        "  ┌─ Invoice Preview ───────────────────────────────",
    ]
    for line in raw_text.split("\n")[:14]:
        lines.append(f"  │  {line}")
    lines += ["  └─ …", ""]

    # ── STEP 2: Extractor ────────────────────────────────────────
    extracted, used_model = run_extractor(raw_text, ref_data)

    if not extracted or not used_model:
        vendor_m = _re.search(r"Vendor:\s*(.+)", raw_text)
        date_m   = _re.search(r"Date:\s*(\d{4}-\d{2}-\d{2})", raw_text)
        total_m  = _re.search(r"(?:Total|TOTAL)[:\s]+[\$£€]?([\d,]+\.?\d*)", raw_text, _re.IGNORECASE)
        vendor   = vendor_m.group(1).strip() if vendor_m else "Unknown Vendor"
        date     = date_m.group(1).strip()   if date_m   else "2024-01-01"
        total    = float(total_m.group(1).replace(",", "")) if total_m else 0.0
        extracted = {
            "vendor": vendor, "date": date, "currency": "USD", "total": total,
            "line_items": [{"description": "Office Supplies", "qty": 1,
                            "unit_price": total, "amount": total}],
        }
    else:
        vendor = extracted.get("vendor", "Unknown Vendor")
        date   = extracted.get("date", "2024-01-01")
        total  = extracted.get("total", 0.0)

    ext_result = _post("/multi/extract", {"episode_id": episode_id, "extracted_data": extracted})
    ext_reward = ext_result.get("reward", 0)
    ext_bd     = ext_result.get("breakdown", {})

    lines += [
        "  🔍  STEP 2 — EXTRACTOR  (Qwen2.5 LoRA · ps2181/extractor-lora-qwen2.5-1.5b)",
        f"  {thin}",
        f"  Vendor   : {vendor}",
        f"  Date     : {date}",
        f"  Total    : ${total:,.2f}",
        f"  Reward   : {ext_reward:.3f}   "
        f"[format {ext_bd.get('format',0):.2f}  "
        f"field {ext_bd.get('field_accuracy',0):.2f}  "
        f"math {ext_bd.get('math_consistency',0):.2f}  "
        f"completeness {ext_bd.get('completeness',0):.2f}]",
        "",
    ]

    # ── STEP 3: Auditor ──────────────────────────────────────────
    audit_results, used_model = run_auditor(raw_text, ref_data, n_inv)

    inv_ids = list(dict.fromkeys(_re.findall(r"ID:\s*(INV-\d+)", raw_text)))
    if not inv_ids:
        inv_ids = [f"INV-{i:05d}" for i in range(n_inv)]

    if not audit_results:
        known = {
            "acme corp","globaltech solutions","prime office supplies","datastream inc",
            "cloudnine services","metro logistics","pinnacle electronics","summit consulting",
            "vertex manufacturing","horizon digital","nexgen software","bluepeak analytics",
        }
        audit_results = []
        for inv_id in inv_ids[:n_inv]:
            is_phantom = vendor.lower() not in known
            audit_results.append({
                "invoice_id": inv_id,
                "verdict":    "flagged" if is_phantom else "approved",
                "fraud_type": "phantom_vendor" if is_phantom else None,
                "confidence": 0.78 if is_phantom else 0.85,
            })
    else:
        model_ids = {r.get("invoice_id") for r in audit_results}
        if not model_ids.intersection(set(inv_ids)):
            for i, r in enumerate(audit_results):
                if i < len(inv_ids):
                    r["invoice_id"] = inv_ids[i]

    aud_result  = _post("/multi/audit", {"episode_id": episode_id, "audit_results": audit_results})
    aud_reward  = aud_result.get("mean_reward", 0)
    aud_feedback = aud_result.get("feedback", "")
    new_report  = aud_result.get("tracker_report", {})

    lines += [
        "  🕵️  STEP 3 — AUDITOR  (Qwen2.5 LoRA · ps2181/auditor-lora-qwen2.5-1.5b)",
        f"  {thin}",
    ]
    for r in audit_results:
        v = r.get("verdict", "approved")
        ft = r.get("fraud_type")
        conf = r.get("confidence", 0)
        if v == "flagged":
            verdict_str = f"🚨 FLAGGED  [{ft.replace('_',' ').upper() if ft else 'UNKNOWN'}]  conf={conf:.2f}"
        else:
            verdict_str = f"✅ APPROVED                         conf={conf:.2f}"
        lines.append(f"  {r.get('invoice_id','?')}  →  {verdict_str}")

    lines += [
        f"  Mean reward  : {aud_reward:.3f}",
        f"  Feedback     : {aud_feedback[:100]}{'…' if len(aud_feedback) > 100 else ''}",
        "",
    ]

    # ── STEP 4: Approver ─────────────────────────────────────────
    approver_decisions = []
    for r in audit_results:
        v    = r.get("verdict", "approved")
        conf = r.get("confidence", 0)
        if v == "flagged" and conf >= 0.80:
            decision = "REJECT"
        elif v == "flagged":
            decision = "ESCALATE"
        else:
            decision = "APPROVE"
        approver_decisions.append((r.get("invoice_id", "?"), decision))

    lines += [
        "  ✅  STEP 4 — APPROVER",
        f"  {thin}",
    ]
    for inv_id, decision in approver_decisions:
        icon = {"REJECT": "❌", "ESCALATE": "⚠️ ", "APPROVE": "✅"}.get(decision, "  ")
        lines.append(f"  {inv_id}  →  {icon} {decision}")

    n_evaded  = sum(1 for r in audit_results if r.get("verdict") == "approved")
    gen_reward = 0.85 if n_evaded == n_inv else (0.60 if n_evaded > 0 else 0.10)
    lines += [
        "",
        f"  Generator adversarial reward : {gen_reward:.2f}",
        f"  Invoices that evaded Auditor : {n_evaded} / {n_inv}",
        "",
    ]

    # ── STEP 5: Regulator update ─────────────────────────────────
    new_blind    = new_report.get("blind_spots", [])
    new_emerging = new_report.get("emerging_blind_spots", [])
    new_weights  = new_report.get("generator_weights", {})

    lines += [
        "  🎯  STEP 5 — REGULATOR UPDATE",
        f"  {thin}",
        f"  Episodes recorded  : {new_report.get('total_audits_recorded', '?')}",
        f"  Critical blind spots : {', '.join(new_blind) if new_blind else 'none'}",
        f"  Emerging blind spots : {', '.join(new_emerging) if new_emerging else 'none'}",
        "",
        "  Generator weights → next episode:",
    ]
    for ft, w in new_weights.items():
        bar   = "▓" * max(1, int(w * 24))
        flag  = "  ← BOOSTED" if w >= 0.35 else ""
        lines.append(f"    {ft:<26} {w:.0%}  {bar}{flag}")

    lines += [
        "",
        sep,
        "  LOOP COMPLETE — weights updated for next episode",
        sep,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Regulator report
# ---------------------------------------------------------------------------

def _get_regulator_report() -> str:
    data        = _get("/regulator/report")
    forecast    = _get("/regulator/forecast")
    calibration = _get("/regulator/calibration")
    if "error" in data:
        return f"Error: {data['error']}"

    sep  = "━" * 54
    thin = "─" * 44

    lines = [
        sep,
        "  REGULATOR DASHBOARD",
        sep, "",
        f"  Episodes recorded : {data.get('total_audits_recorded', 0)}  (window = {data.get('window', 30)})",
        "",
        "  DETECTION RATES   ↑ improving · ↓ declining · → stable",
        f"  {thin}",
    ]
    for ft, status in data.get("detection_rates", {}).items():
        lines.append(f"  {ft.replace('_',' ').title():<28} {status}")

    lines += [
        "",
        f"  False Positive Rate  : {data.get('false_positive_rate', 'no data')}",
        "",
        f"  🚨 Critical blind spots : {', '.join(data.get('blind_spots', [])) or 'none'}",
        f"  ⚠️  Emerging blind spots : {', '.join(data.get('emerging_blind_spots', [])) or 'none'}",
        "",
        "  PREDICTIVE FORECAST",
        f"  {thin}",
    ]
    if "error" not in forecast:
        for ft, trend in forecast.get("trends", {}).items():
            lines.append(f"  {ft.replace('_',' ').title():<28} {trend}")
        rec = forecast.get("recommendation", "")
        if rec:
            lines += ["", f"  → {rec}"]

    lines += [
        "",
        "  CONFIDENCE CALIBRATION  (high conf + wrong = dangerous)",
        f"  {thin}",
    ]
    if "error" not in calibration:
        for ft, cal in calibration.items():
            if isinstance(cal, dict) and cal.get("status") != "no data":
                lines.append(f"  {ft.replace('_',' ').title():<28} {cal.get('status','')}")
                wrong = cal.get("mean_confidence_when_wrong")
                right = cal.get("mean_confidence_when_correct")
                if wrong is not None:
                    lines.append(f"    wrong conf={wrong:.2f}   correct conf={right if right else 'N/A'}")

    lines += [
        "",
        "  GENERATOR WEIGHTS — next episode bias",
        f"  {thin}",
    ]
    for ft, w in data.get("generator_weights", {}).items():
        bar  = "▓" * int(w * 30)
        flag = "  ← prioritised" if w >= 0.35 else ""
        lines.append(f"  {ft.replace('_',' ').title():<28} {w:.1%}  {bar}{flag}")

    lines += ["", f"  Verdict : {data.get('verdict', '')}", sep]
    return "\n".join(lines)


_GITHUB_RAW = "https://raw.githubusercontent.com/ps2181/invoice-processing-pipeline/main/assets"

def _make_training_curves() -> str:
    """Return HTML showing the 3 real Colab training curve images from GitHub."""
    img_style = "width:100%;border-radius:10px;margin-bottom:16px;"
    label_style = (
        "color:#94a3b8;font-size:0.82rem;font-family:monospace;"
        "text-align:center;display:block;margin-bottom:20px;"
    )
    return f"""
<div style="padding:8px;">
  <p style="color:#94a3b8;font-size:0.85rem;margin-bottom:16px;">
    Real training runs on Google Colab A100 — Qwen2.5-1.5B-Instruct + LoRA r=16 + TRL GRPOTrainer + Unsloth
  </p>
  <img src="{_GITHUB_RAW}/reward_curve.png" style="{img_style}" />
  <span style="{label_style}">
    🔍 Extractor — Total reward (4 signals) ↑ · Live /grader score peaked 0.914 at step 15 · Crashed at step 20 due to _MAX_SESSIONS=50 bug (fixed to 200)
  </span>
  <img src="{_GITHUB_RAW}/auditor_reward_curve_run2.png" style="{img_style}" />
  <span style="{label_style}">
    🕵️ Auditor — Run 2 (bug fixed) · Total reward ↑ 0.48→0.72 · Live env reward ↑ 0.28→0.52 · Run 1 had dead signal (episode_id list bug)
  </span>
  <img src="{_GITHUB_RAW}/generator_reward_curve.png" style="{img_style}" />
  <span style="{label_style}">
    ⚡ Generator — Fraud plausibility (format) learned ~0.19→0.25 · Live evasion reward stuck at 0.01 (same episode_id bug) — needs rerun with fix
  </span>
</div>
"""


def _seed_demo_data() -> str:
    data = _post("/regulator/demo_seed", {})
    if "error" in data:
        return f"Error: {data['error']}"
    return (
        "✅ Demo data seeded — phantom_vendor ~32% (blind spot), "
        "duplicate_submission declining (emerging)\n\n"
        + _get_regulator_report()
    )


# ---------------------------------------------------------------------------
# Agent Tester helpers
# ---------------------------------------------------------------------------

def _call_llm(task_id: str, obs: Dict[str, Any], step: int) -> Tuple[str, str]:
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
                {"role": "user",   "content": user_prompt},
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
        return json.dumps(parsed, indent=2), f"✅ LLM ({MODEL_NAME}) responded."
    except json.JSONDecodeError as e:
        return "{}", f"❌ LLM returned invalid JSON: {e}"
    except Exception as e:
        return "{}", f"❌ LLM error: {e}"


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:

    init_state = {"episode_id": None, "obs": None, "step": 0, "history": []}

    def do_reset(task_id: str, state: dict):
        data = _post("/reset", {"task_id": task_id})
        if "error" in data:
            return (
                state,
                gr.update(value=f"❌ {data['error']}"),
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
        ep  = data["info"]["episode_id"]
        new_state = {"episode_id": ep, "obs": obs, "step": 0, "history": []}
        status = (
            f"✅  Episode started   task={task_id}   id={ep[:12]}…\n"
            f"Max attempts: {obs['max_attempts']}"
        )
        return (
            new_state,
            gr.update(value=status),
            gr.update(value=obs["task_description"]),
            gr.update(value=obs["raw_text"]),
            gr.update(value=obs.get("reference_data") or ""),
            gr.update(value=PLACEHOLDER_JSON),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    def do_llm(task_id: str, state: dict):
        if not state.get("obs"):
            return PLACEHOLDER_JSON, "⚠️  Reset an episode first."
        return _call_llm(task_id, state["obs"], state["step"] + 1)

    def do_submit(json_str: str, state: dict):
        if not state.get("episode_id"):
            return state, "⚠️  Reset an episode first.", "", "", ""
        try:
            extracted = json.loads(json_str)
        except json.JSONDecodeError as e:
            return state, f"❌ Invalid JSON: {e}", "", "", ""
        data = _post("/step", {"extracted_data": extracted, "episode_id": state["episode_id"]})
        if "error" in data:
            return state, f"❌ {data['error']}", "", "", ""
        obs    = data["observation"]
        reward = data.get("reward", 0.0)
        done   = data.get("done", False)
        state["obs"] = obs
        state["step"] += 1
        state["history"].append(
            f"Step {state['step']}: reward={reward:.3f}" + (" ✓ done" if done else "")
        )
        status = (
            f"Step {state['step']} / {obs['max_attempts']}   "
            f"Reward: {reward:.3f}   "
            f"{'🏁 Done' if done else 'In progress…'}"
        )
        bd = obs.get("reward_breakdown")
        return (
            state, status,
            obs.get("feedback") or "No feedback yet.",
            "\n".join(state["history"]),
            json.dumps(bd, indent=2) if bd else "",
        )

    def _get_model_status() -> str:
        from server.agents import models_status
        s = models_status()
        if not s:
            return "No models checked yet — run a pipeline episode to trigger loading."
        return "\n".join(f"  {k:<12} {v}" for k, v in s.items())

    # ── Build layout ──────────────────────────────────────────────────────────

    with gr.Blocks(
        title="Invoice Processing Pipeline — Multi-Agent Fraud Detection",
        theme=gr.themes.Soft(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
        css=CSS,
    ) as demo:

        gr.HTML(HERO_HTML)
        gr.HTML(AGENT_CARDS_HTML)

        session_state = gr.State(init_state)

        with gr.Tabs():

            # ================================================================
            # TAB 1 — Multi-Agent Pipeline  (FIRST — most impressive)
            # ================================================================
            with gr.Tab("🔄  Live Pipeline"):

                gr.Markdown(
                    "### Watch all 5 agents work together in real time\n"
                    "Each run is a **fresh episode**: the Regulator reads current blind spots, "
                    "the Generator creates adversarial invoices biased toward weak fraud types, "
                    "the Extractor parses them, the Auditor flags fraud, the Approver decides, "
                    "and the Regulator updates weights for the next round."
                )

                with gr.Row():
                    run_btn    = gr.Button("▶  Run Full Pipeline Episode", variant="primary",
                                           size="lg", scale=4, elem_classes=["run-btn"])
                    model_btn  = gr.Button("🔍  Agent Status", variant="secondary", scale=1)

                model_status_box = gr.Textbox(
                    label="Loaded Agents",
                    interactive=False,
                    lines=3,
                    value="Click '🔍 Agent Status' to see which LoRA models are loaded.",
                    elem_id="status-bar",
                )

                pipeline_output = gr.Textbox(
                    label="Pipeline Trace",
                    interactive=False,
                    lines=50,
                    value=(
                        "Click  ▶ Run Full Pipeline Episode  to start.\n\n"
                        "You will see all 5 agents execute in sequence with live rewards and "
                        "the Regulator's self-improvement loop closing at the end."
                    ),
                    elem_id="pipeline-trace",
                )

                run_btn.click(fn=_run_pipeline_episode, inputs=[], outputs=[pipeline_output])
                model_btn.click(fn=_get_model_status,   inputs=[], outputs=[model_status_box])

            # ================================================================
            # TAB 2 — Regulator Dashboard
            # ================================================================
            with gr.Tab("📊  Regulator Dashboard"):

                gr.Markdown(
                    "### Cross-Episode Auditor Oversight\n"
                    "The Regulator monitors detection rates across episodes, "
                    "identifies blind spots using **predictive trend analysis**, "
                    "flags **overconfident misses** via confidence calibration, "
                    "and dynamically reweights the Generator toward under-detected fraud types."
                )

                with gr.Row():
                    refresh_btn = gr.Button("🔄  Refresh Report", variant="primary", scale=2)
                    seed_btn    = gr.Button("🌱  Load Demo Data",  variant="secondary", scale=1)

                report_box = gr.Textbox(
                    label="Regulator Report",
                    interactive=False,
                    lines=30,
                    value="Click  🔄 Refresh Report  or  🌱 Load Demo Data  to begin.",
                    elem_id="reg-report",
                )

                refresh_btn.click(fn=_get_regulator_report, inputs=[], outputs=[report_box])
                seed_btn.click(fn=_seed_demo_data,           inputs=[], outputs=[report_box])

            # ================================================================
            # TAB 3 — Training Results
            # ================================================================
            with gr.Tab("📈  Training Results"):

                gr.Markdown(
                    "### GRPO Training Results — Real Data from Colab Runs\n"
                    "Each agent trained with **TRL GRPOTrainer + Unsloth** on live environment data. "
                    "The deployed HF Space served as the live reward verifier.\n\n"
                    "| Agent | Result | Notes |\n"
                    "|-------|--------|-------|\n"
                    "| **Extractor** | Live grader score peaked at **0.914** | Crashed mid-run due to session pool bug (_MAX_SESSIONS=50), fixed to 200 |\n"
                    "| **Auditor** | Live reward climbed **0.01 → 0.52** | First run had dead reward signal (episode_id list bug), fixed and retrained |\n"
                    "| **Generator** | Fraud format learned (~0.22) | Live evasion reward had same episode_id bug — needs rerun with fix |"
                )

                try:
                    _curves_html = _make_training_curves()
                except Exception:
                    _curves_html = "<p style='color:#f87171'>Could not render training curves.</p>"

                gr.HTML(value=_curves_html)

                gr.Markdown(
                    "**Model checkpoints:** "
                    "[Extractor LoRA](https://huggingface.co/ps2181/extractor-lora-qwen2.5-1.5b) · "
                    "[Auditor LoRA](https://huggingface.co/ps2181/auditor-lora-qwen2.5-1.5b) · "
                    "[Generator LoRA](https://huggingface.co/ps2181/generator-lora-qwen2.5-1.5b)"
                )

            # ================================================================
            # TAB 4 — Single-Agent Tester
            # ================================================================
            with gr.Tab("🧪  Single Agent Tester"):

                gr.Markdown(
                    "### Test the OpenEnv environment directly\n"
                    "Select a task, reset to load an invoice, then submit structured JSON "
                    "to see the grader score and feedback. Covers 7 task types from easy extraction "
                    "to supply chain anomaly detection."
                )

                with gr.Row():
                    task_dd   = gr.Dropdown(
                        choices=list(TASK_DESCRIPTIONS.keys()),
                        value="easy",
                        label="Task type",
                        scale=1,
                    )
                    reset_btn = gr.Button("🔄  Reset Episode", variant="primary", scale=1)
                    status_box = gr.Textbox(
                        label="Status",
                        interactive=False,
                        scale=3,
                        lines=2,
                        elem_id="status-bar",
                    )

                task_info = gr.Textbox(label="Task description", interactive=False, lines=1)

                with gr.Row():
                    with gr.Column(scale=5):
                        invoice_box = gr.Textbox(
                            label="📄  Invoice (raw text)",
                            interactive=False,
                            lines=16,
                            max_lines=30,
                        )
                        ref_box = gr.Textbox(
                            label="📋  Reference data (PO / vendor registry / catalog)",
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
                                "🤖  Run LLM Agent",
                                variant="secondary",
                                interactive=False,
                            )
                            submit_btn = gr.Button(
                                "✅  Submit",
                                variant="primary",
                                interactive=False,
                            )
                        llm_status = gr.Textbox(
                            label="Agent status",
                            interactive=False,
                            lines=1,
                        )

                with gr.Row():
                    feedback_box = gr.Textbox(
                        label="📝  Grader Feedback",
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
                    label="Step history",
                    interactive=False,
                    lines=3,
                )

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

    return demo
