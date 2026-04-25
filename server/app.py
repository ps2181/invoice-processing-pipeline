"""
FastAPI server for Invoice Processing Pipeline environment.
Exposes /reset, /step, /state, /health, /tasks, /grader endpoints.

Session-based: each /reset creates an isolated InvoiceEnvironment instance
keyed by episode_id, supporting concurrent agents without state conflicts.
"""

from __future__ import annotations

import random
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import InvoiceAction, InvoiceObservation, InvoiceState
from server.environment import InvoiceEnvironment

app = FastAPI(
    title="Invoice Processing Pipeline",
    description="OpenEnv environment for invoice data extraction, cleaning, and reconciliation.",
    version="1.0.0",
)

# Auto-seed Regulator tracker on startup so pipeline demo has meaningful data immediately
from server.multi_agent_environment import tracker as _startup_tracker
_startup_tracker.reset_for_demo()

# Mount Gradio web UI at /web
try:
    import gradio as gr
    from server.web_ui import build_ui
    _gradio_app = build_ui()
    app = gr.mount_gradio_app(app, _gradio_app, path="/web")
    print("[startup] Gradio UI mounted at /web")
except Exception as _e:
    import traceback, warnings
    warnings.warn(f"Gradio UI not loaded: {_e}")
    traceback.print_exc()
    print(f"[startup] /web FAILED: {_e}")

# ---------------------------------------------------------------------------
# Session registry — one InvoiceEnvironment per episode_id
# Thread-safe, capped at MAX_SESSIONS to bound memory on vcpu=2 / 8gb
# ---------------------------------------------------------------------------

_MAX_SESSIONS = 200
_sessions: OrderedDict[str, InvoiceEnvironment] = OrderedDict()
_lock = threading.Lock()


def _new_session(task_id: str) -> tuple[InvoiceEnvironment, Any, float, bool, dict]:
    """Create a new env, run reset, register it, evict oldest if over cap."""
    env = InvoiceEnvironment()
    obs, reward, done, info = env.reset(task_id=task_id)
    episode_id = info["episode_id"]
    with _lock:
        _sessions[episode_id] = env
        while len(_sessions) > _MAX_SESSIONS:
            _sessions.popitem(last=False)
    return env, obs, reward, done, info


def _get_session(episode_id: Optional[str]) -> InvoiceEnvironment:
    """Return env for episode_id, or the most recent session if None."""
    with _lock:
        if episode_id and episode_id in _sessions:
            return _sessions[episode_id]
        if _sessions:
            return next(reversed(_sessions.values()))
    raise HTTPException(status_code=404, detail="No active session. Call /reset first.")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy"

class StepRequest(BaseModel):
    extracted_data: Dict[str, Any]
    explanation: str = ""
    episode_id: Optional[str] = None  # optional: route to specific session

class StateRequest(BaseModel):
    episode_id: Optional[str] = None

class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]

class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]

class StateResponse(BaseModel):
    episode_id: str
    task_id: str
    step_count: int
    done: bool
    last_reward: float
    best_reward: float
    rewards: list


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    with _lock:
        active = len(_sessions)
    return {"status": "ok", "environment": "invoice_processing_pipeline", "active_sessions": active}


@app.get("/tasks")
def list_tasks():
    """List available tasks with descriptions."""
    tasks = []
    for tid, info in InvoiceEnvironment.TASKS.items():
        tasks.append({
            "task_id": tid,
            "description": info["description"],
            "max_attempts": info["max_attempts"],
        })
    return {
        "tasks": tasks,
        "action_schema": InvoiceAction.model_json_schema(),
        "observation_schema": InvoiceObservation.model_json_schema(),
    }


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    _env, obs, reward, done, info = _new_session(task_id=req.task_id)
    return ResetResponse(
        observation=obs.model_dump(),
        reward=reward,
        done=done,
        info=info,
    )


@app.post("/step")
def step(req: StepRequest):
    env = _get_session(req.episode_id)
    if env.state.done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset first.")

    action = InvoiceAction(
        extracted_data=req.extracted_data,
        explanation=req.explanation,
    )
    obs, reward, done, info = env.step(action)
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state")
def get_state(episode_id: Optional[str] = None):
    env = _get_session(episode_id)
    s = env.state
    return StateResponse(
        episode_id=s.episode_id,
        task_id=s.task_id,
        step_count=s.step_count,
        done=s.done,
        last_reward=s.last_reward,
        best_reward=s.best_reward,
        rewards=s.rewards,
    )


@app.post("/grader")
def grader(req: StepRequest):
    """Score a submission without modifying episode state (for testing)."""
    env = _get_session(req.episode_id)
    action = InvoiceAction(extracted_data=req.extracted_data, explanation=req.explanation)

    task_id = env.state.task_id
    if task_id == "easy":
        from server.environment import _grade_easy
        score, feedback = _grade_easy(action.extracted_data, env._ground_truth)
    elif task_id == "medium":
        from server.environment import _grade_medium
        score, feedback = _grade_medium(action.extracted_data, env._ground_truth)
    elif task_id == "hard":
        from server.environment import _grade_hard
        score, feedback = _grade_hard(
            action.extracted_data, env._ground_truth, env._expected_discrepancies
        )
    elif task_id == "adversarial":
        from server.environment import _grade_adversarial
        score, feedback, _bd = _grade_adversarial(action.extracted_data, env._ground_truth)
    elif task_id == "negotiate":
        from server.environment import _grade_negotiate
        score, feedback, _bd = _grade_negotiate(
            action.extracted_data, env._ground_truth, env._state.clarification_count
        )
    elif task_id == "supply_chain":
        from server.environment import _grade_supply_chain
        score, feedback = _grade_supply_chain(
            action.extracted_data, env._expected_sc_anomalies
        )
    elif task_id == "long_horizon":
        from server.environment import _grade_long_horizon
        score, feedback = _grade_long_horizon(
            action.extracted_data, env._state, env._lh_gt,
            env._expected_discrepancies, env._lh_expert_gt, env._lh_po_texts,
        )
    elif task_id == "personalized":
        from server.environment import _grade_personalized
        score, feedback, _ = _grade_personalized(action.extracted_data, env._personalized_gt)
    elif task_id == "curriculum":
        from server.environment import _curriculum_grade
        score, feedback = _curriculum_grade(
            env._curriculum_stage, action.extracted_data,
            env._curriculum_gt, env._curriculum_extra,
        )
    else:  # expert
        from server.environment import _grade_expert
        score, feedback = _grade_expert(action.extracted_data, env._expert_ground_truth)

    return {"score": score, "feedback": feedback}


def _clamp(v: float) -> float:
    return max(0.01, min(0.99, float(v)))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint — required by openenv-core GenericEnvClient."""
    await websocket.accept()
    env = InvoiceEnvironment()

    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type")
            data = msg.get("data", {})

            if msg_type == "reset":
                task_id = data.get("task_id", "easy")
                try:
                    obs, reward, done, info = env.reset(task_id=task_id)
                except Exception as e:
                    await websocket.send_json({"type": "error", "data": {"message": str(e)}})
                    continue
                await websocket.send_json({
                    "type": "observation",
                    "data": {
                        "observation": obs.model_dump(),
                        "reward": _clamp(reward),
                        "done": done,
                        "info": info,
                    },
                })

            elif msg_type == "step":
                extracted = data.get("extracted_data", {})
                explanation = data.get("explanation", "")
                action = InvoiceAction(extracted_data=extracted, explanation=explanation)
                obs, reward, done, info = env.step(action)
                await websocket.send_json({
                    "type": "observation",
                    "data": {
                        "observation": obs.model_dump(),
                        "reward": _clamp(reward),
                        "done": done,
                        "info": info,
                    },
                })

            elif msg_type == "state":
                await websocket.send_json({
                    "type": "state",
                    "data": env.state.model_dump(),
                })

            elif msg_type == "close":
                break

            else:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": f"Unknown message type: {msg_type}"},
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "data": {"message": str(e)}})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Multi-agent endpoints
# ---------------------------------------------------------------------------

from server.multi_agent_environment import (
    create_episode,
    get_episode,
    handle_extract,
    handle_audit,
    handle_approve,
    tracker as _regulator_tracker,
    compute_regulator_reward,
)


class MultiResetResponse(BaseModel):
    episode_id: str
    raw_text: str
    reference_data: str
    fraud_weights_used: Dict[str, Any]
    n_invoices: int


class MultiExtractRequest(BaseModel):
    episode_id: str
    extracted_data: Dict[str, Any]


class MultiAuditRequest(BaseModel):
    episode_id: str
    audit_results: list


class RegulatorPredictRequest(BaseModel):
    predicted_blind_spots: list
    predicted_emerging: Optional[list] = None


@app.post("/multi/reset")
def multi_reset():
    """Start a new multi-agent episode. Generator is biased by Regulator blind spots."""
    ep = create_episode()
    return MultiResetResponse(
        episode_id=ep.episode_id,
        raw_text=ep.raw_text,
        reference_data=ep.reference_data,
        fraud_weights_used=ep.fraud_weights_used,
        n_invoices=len(ep.invoices),
    )


@app.post("/multi/extract")
def multi_extract(req: MultiExtractRequest):
    """Score Extractor output with 4 independent reward signals."""
    result = handle_extract(req.episode_id, req.extracted_data)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/multi/audit")
def multi_audit(req: MultiAuditRequest):
    """Score Auditor output. Records to AuditorPerformanceTracker."""
    result = handle_audit(req.episode_id, req.audit_results)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


class MultiApproveRequest(BaseModel):
    episode_id: str


@app.post("/multi/approve")
def multi_approve(req: MultiApproveRequest):
    """Run rule-based Approver. Computes Generator adversarial reward."""
    result = handle_approve(req.episode_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/multi/state/{episode_id}")
def multi_state(episode_id: str):
    """Get current state of a multi-agent episode."""
    ep = get_episode(episode_id)
    if ep is None:
        raise HTTPException(status_code=404, detail="Episode not found")
    return {
        "episode_id": ep.episode_id,
        "n_invoices": len(ep.invoices),
        "fraud_weights_used": ep.fraud_weights_used,
        "extractor_reward": ep.extractor_reward,
        "extractor_breakdown": ep.extractor_breakdown,
        "mean_auditor_reward": ep.mean_auditor_reward,
        "mean_generator_reward": ep.mean_generator_reward,
        "done": ep.done,
    }


@app.get("/regulator/report")
def regulator_report():
    """Get the Regulator's current cross-episode Auditor performance report."""
    return _regulator_tracker.report()


@app.post("/regulator/predict")
def regulator_predict(req: RegulatorPredictRequest):
    """Score a Regulator agent's blind spot predictions against actual tracker state.
    Optional: predicted_emerging for Option A early-warning bonus."""
    actual = _regulator_tracker.blind_spots()
    reward, feedback = compute_regulator_reward(
        req.predicted_blind_spots, actual, req.predicted_emerging
    )
    return {
        "reward": reward,
        "feedback": feedback,
        "actual_blind_spots": actual,
        "actual_emerging": [e["fraud_type"] for e in _regulator_tracker.emerging_blind_spots()],
        "predicted_blind_spots": req.predicted_blind_spots,
        "predicted_emerging": req.predicted_emerging,
    }


@app.get("/regulator/forecast")
def regulator_forecast():
    """Option A: Predictive Regulator — trend analysis + emerging blind spot warnings."""
    return _regulator_tracker.forecast()


@app.get("/regulator/calibration")
def regulator_calibration():
    """Option C: Auditor confidence calibration report — detects overconfident misses."""
    return _regulator_tracker.calibration_report()


# ---------------------------------------------------------------------------
# Generator training endpoint
# ---------------------------------------------------------------------------

class GeneratorScoreRequest(BaseModel):
    invoice_json: Dict[str, Any]          # generated invoice to evaluate
    fraud_type: str                        # claimed fraud type
    context: Optional[str] = ""           # optional: generation reasoning

class GeneratorScoreResponse(BaseModel):
    reward: float
    auditor_detected: bool
    approver_decision: str
    feedback: str
    regulator_weights: Dict[str, Any]     # current blind spot weights for next generation


@app.post("/generator/score")
def generator_score(req: GeneratorScoreRequest):
    """
    Score a Generator-produced invoice through the full Auditor+Approver pipeline.
    Used as the live reward signal during Generator GRPO training.

    The Generator's job: produce a fraudulent invoice that evades the Auditor.
    Reward: 0.85 (evades both), 0.60 (evades Auditor, Approver catches), 0.10 (Auditor catches)
    """
    from server.multi_agent_environment import (
        compute_generator_reward, approver_decision as _approver_decision,
        PHANTOM_VENDORS, FRAUD_TYPES,
    )
    from server.environment import VENDORS, MARKET_PRICE_MAX

    inv = req.invoice_json
    fraud_type = req.fraud_type.lower().strip()

    # --- Rule-based Auditor simulation (fast, no LLM needed for scoring) ---
    # This mirrors what a trained Auditor would do, seeded from tracker blind spots
    vendor = inv.get("vendor", "")
    line_items = inv.get("line_items", [])
    total = float(inv.get("total", 0))

    auditor_verdict = "approved"
    auditor_confidence = 0.5
    auditor_fraud_type = None

    # Check phantom vendor
    known_vendors_lower = [v.lower() for v in VENDORS]
    if vendor.lower() not in known_vendors_lower:
        # Auditor detection rate for phantom_vendor from tracker
        detection_rate = _regulator_tracker.detection_rates().get("phantom_vendor") or 0.5
        if random.random() < detection_rate:
            auditor_verdict = "flagged"
            auditor_fraud_type = "phantom_vendor"
            auditor_confidence = round(0.5 + detection_rate * 0.4, 2)

    # Check price gouging
    if auditor_verdict == "approved":
        for item in line_items:
            desc = item.get("description", "")
            up = float(item.get("unit_price", 0))
            market_max = MARKET_PRICE_MAX.get(desc, up)
            if up > market_max * 1.5:
                detection_rate = _regulator_tracker.detection_rates().get("price_gouging") or 0.5
                if random.random() < detection_rate:
                    auditor_verdict = "flagged"
                    auditor_fraud_type = "price_gouging"
                    auditor_confidence = round(0.5 + detection_rate * 0.4, 2)
                break

    # Check math fraud
    if auditor_verdict == "approved" and line_items:
        computed = round(sum(float(it.get("amount", 0)) for it in line_items), 2)
        if abs(computed - total) > 0.05 * total:
            detection_rate = _regulator_tracker.detection_rates().get("math_fraud") or 0.5
            if random.random() < detection_rate:
                auditor_verdict = "flagged"
                auditor_fraud_type = "math_fraud"
                auditor_confidence = round(0.5 + detection_rate * 0.4, 2)

    auditor_detected = auditor_verdict == "flagged"

    # --- Approver decision ---
    approver = _approver_decision(auditor_verdict, auditor_confidence, auditor_fraud_type)
    approver_approved = approver["decision"] == "approve"

    # --- Generator reward ---
    reward = compute_generator_reward(auditor_detected, approver_approved)

    feedback_parts = [
        f"Auditor: {'DETECTED' if auditor_detected else 'MISSED'} (conf={auditor_confidence:.2f})",
        f"Approver: {approver['decision']} — {approver['reason']}",
        f"Generator reward: {reward:.2f}",
    ]

    return GeneratorScoreResponse(
        reward=reward,
        auditor_detected=auditor_detected,
        approver_decision=approver["decision"],
        feedback=" | ".join(feedback_parts),
        regulator_weights=_regulator_tracker.generator_weights(),
    )


@app.get("/metrics")
def metrics():
    """Environment-wide aggregate metrics: episode counts, per-task averages, all-time bests."""
    from server.environment import _PERF_HISTORY, _PERF_LOCK
    with _PERF_LOCK:
        per_task = {}
        total_episodes = 0
        for task_id, history in _PERF_HISTORY.items():
            h = list(history)
            total_episodes += len(h)
            if h:
                per_task[task_id] = {
                    "episodes": len(h),
                    "avg_score": round(sum(h) / len(h), 4),
                    "best_score": round(max(h), 4),
                    "latest_score": round(h[-1], 4),
                }
            else:
                per_task[task_id] = {"episodes": 0, "avg_score": None, "best_score": None, "latest_score": None}

    with _lock:
        active_sessions = len(_sessions)

    return {
        "total_episodes": total_episodes,
        "active_sessions": active_sessions,
        "per_task": per_task,
        "regulator": _regulator_tracker.report(),
    }


@app.post("/regulator/demo_seed")
def regulator_demo_seed():
    """Seed the tracker with realistic demo data (phantom_vendor weak at 31%)."""
    _regulator_tracker.reset_for_demo()
    return {"status": "seeded", "report": _regulator_tracker.report()}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
