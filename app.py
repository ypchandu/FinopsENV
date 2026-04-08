"""FastAPI server – OpenEnv-compliant endpoints for the Autonomous FinOps Agent.

Endpoints
---------
POST /reset   – Reset env to a task's starting state.  Returns Observation.
POST /step    – Execute one action (ActionEnvelope).    Returns StepResult.
GET  /state   – Current observation (read-only).        Returns Observation.
GET  /grade   – Grade the current trajectory.           Returns GraderResult.
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from schemas import ActionEnvelope, GraderResult, Observation, StepResult
from environment import FinOpsEnv
from graders import grade_easy, grade_medium, grade_hard

# ── App + CORS ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Autonomous FinOps Agent Environment",
    version="1.0.0",
    description="RL environment for SaaS seat pruning and LLM API tier routing.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Single global environment instance ───────────────────────────────────────

env = FinOpsEnv()

# ── Request schemas ──────────────────────────────────────────────────────────


class ResetRequest(BaseModel):
    task: Literal["easy", "medium", "hard"]


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.post("/reset", response_model=Observation)
def reset(payload: ResetRequest) -> Observation:
    """Reset the environment to the starting state for a given task."""
    try:
        obs: Observation = env.reset(payload.task)
        return obs
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step", response_model=StepResult)
def step(action: ActionEnvelope) -> StepResult:
    """Execute one action and advance the simulation by one week."""
    if env.episode_done:
        raise HTTPException(
            status_code=400,
            detail="Episode is already done. POST /reset to start a new episode.",
        )
    try:
        result: StepResult = env.step(action)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state", response_model=Observation)
def state() -> Observation:
    """Return the current observation without advancing the simulation."""
    if env.budget is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialised. POST /reset first.",
        )
    return env.state()


@app.get("/grade", response_model=GraderResult)
def grade(
    task: Literal["easy", "medium", "hard"] = Query(
        ..., description="Task ID to grade the current trajectory for."
    ),
) -> GraderResult:
    """Grade the current trajectory for the specified task."""
    if not env.trajectory:
        raise HTTPException(
            status_code=400,
            detail="No trajectory to grade. POST /reset first.",
        )
    grader_map = {
        "easy": grade_easy,
        "medium": grade_medium,
        "hard": grade_hard,
    }
    return grader_map[task](env.trajectory)


@app.get("/health")
def health() -> dict:
    """Return environment health status."""
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict:
    """Return environment metadata."""
    return {
        "name": "Autonomous FinOps Agent Environment",
        "description": "RL environment for SaaS seat pruning and LLM API tier routing."
    }


@app.get("/schema")
def schema() -> dict:
    """Return JSON schemas for action, observation, and state."""
    return {
        "action": ActionEnvelope.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": Observation.model_json_schema(),
    }


@app.post("/mcp")
def mcp(payload: dict) -> dict:
    """Model Context Protocol endpoint."""
    return {
        "jsonrpc": "2.0",
        "id": payload.get("id"),
        "result": {}
    }
