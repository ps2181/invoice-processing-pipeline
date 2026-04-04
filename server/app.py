"""
FastAPI server for Invoice Processing Pipeline environment.
Exposes /reset, /step, /state, /health, /tasks, /grader endpoints.

Session-based: each /reset creates an isolated InvoiceEnvironment instance
keyed by episode_id, supporting concurrent agents without state conflicts.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
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

# ---------------------------------------------------------------------------
# Session registry — one InvoiceEnvironment per episode_id
# Thread-safe, capped at MAX_SESSIONS to bound memory on vcpu=2 / 8gb
# ---------------------------------------------------------------------------

_MAX_SESSIONS = 50
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
    else:
        from server.environment import _grade_expert
        score, feedback = _grade_expert(action.extracted_data, env._expert_ground_truth)

    return {"score": score, "feedback": feedback}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
