"""
FastAPI server for Invoice Processing Pipeline environment.
Exposes /reset, /step, /state, /health, /tasks, /grader endpoints.
"""

from __future__ import annotations

import json
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

# Single environment instance (one episode at a time for the HF Space)
env = InvoiceEnvironment()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy"

class StepRequest(BaseModel):
    extracted_data: Dict[str, Any]
    explanation: str = ""

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
    return {"status": "ok", "environment": "invoice_processing_pipeline"}


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
    obs, reward, done, info = env.reset(task_id=req.task_id)
    return ResetResponse(
        observation=obs.model_dump(),
        reward=reward,
        done=done,
        info=info,
    )


@app.post("/step")
def step(req: StepRequest):
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
def get_state():
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
    import copy
    saved_state = copy.deepcopy(env._state)
    action = InvoiceAction(extracted_data=req.extracted_data, explanation=req.explanation)

    task_id = env.state.task_id
    if task_id == "easy":
        from server.environment import _grade_easy
        score, feedback = _grade_easy(action.extracted_data, env._ground_truth)
    elif task_id == "medium":
        from server.environment import _grade_medium
        score, feedback = _grade_medium(action.extracted_data, env._ground_truth)
    else:
        from server.environment import _grade_hard
        score, feedback = _grade_hard(
            action.extracted_data, env._ground_truth, env._expected_discrepancies
        )

    return {"score": score, "feedback": feedback}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)