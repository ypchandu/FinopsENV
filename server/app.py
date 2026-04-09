"""FastAPI server – OpenEnv-compliant endpoints for the Autonomous FinOps Agent.

Endpoints
---------
POST /reset   – Reset env to a task's starting state.  Returns Observation.
POST /step    – Execute one action (ActionEnvelope).    Returns StepResult.
GET  /state   – Current observation (read-only).        Returns Observation.
GET  /grade   – Grade the current trajectory.           Returns GraderResult.
"""

from __future__ import annotations
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import RedirectResponse

from schemas import ActionEnvelope, GraderResult, Observation, StepResult
from environment import FinOpsEnv
from graders import grade_easy, grade_medium, grade_hard

# ── App + CORS ───────────────────────────────────────────────────────────────

# CUSTOM INSTRUCTION BOX FOR JUDGES
instructions = """
## 🛠️ How to Use the FinOps Environment
This dashboard allows you to manually interact with the Autonomous FinOps Agent simulation.

| Step | Action | Endpoint | Description |
| :--- | :--- | :--- | :--- |
| **1** | **Initialize** | `POST /reset` | Choose a task (**easy**, **medium**, **hard**) to start the 52-week simulation. |
| **2** | **Execute** | `POST /step` | Submit an action (Prune Seats or Route Traffic). Each step advances time by 1 week. |
| **3** | **Monitor** | `GET /state` | View current budget, SLA status, and SaaS telemetry without advancing time. |
| **4** | **Evaluate** | `GET /grade` | Once the episode is done (Week 52 or Bankruptcy), get your final performance score. |

---
"""

app = FastAPI(
    title="Autonomous FinOps Agent Environment",
    version="1.0.0",
    description=instructions + "RL environment for SaaS seat pruning and LLM API tier routing.",
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
    task: Literal["easy", "medium", "hard"] = "easy"

class GradeRequest(BaseModel):
    task: Literal["easy", "medium", "hard"]
    trajectory: list[dict]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    """Redirects visitors to the interactive dashboard."""
    return RedirectResponse(url="/docs")

@app.post("/reset", response_model=Observation)
def reset(payload: Optional[ResetRequest] = None) -> Observation:
    """Reset the environment to the starting state for a given task."""
    try:
        task_name = payload.task if payload else "easy"
        obs: Observation = env.reset(task_name)
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
    
    # Use the trajectory from the environment state
    trajectory = env.trajectory
    
    if task == "easy":
        return grade_easy(trajectory)
    elif task == "medium":
        return grade_medium(trajectory)
    elif task == "hard":
        return grade_hard(trajectory)
    else:
        raise HTTPException(status_code=404, detail="Task not found")

@app.post("/grade", response_model=GraderResult)
def grade_post(payload: GradeRequest) -> GraderResult:
    """Automated Grader hook for Phase 2 Task Validation tests."""
    raw_traj = payload.trajectory
    parsed_traj = []

    for entry in raw_traj:
        # Clone the entry to avoid side effects
        parsed_entry = entry.copy()
        
        # Convert dictionary observation back into a Pydantic object
        if "observation" in entry and isinstance(entry["observation"], dict):
            parsed_entry["observation"] = Observation.model_validate(entry["observation"])
            
        # Convert dictionary action envelope back into a Pydantic object
        if "action" in entry and isinstance(entry["action"], dict):
            parsed_entry["action"] = ActionEnvelope.model_validate(entry["action"])
            
        parsed_traj.append(parsed_entry)

    if payload.task == "easy":
        return grade_easy(parsed_traj)
    elif payload.task == "medium":
        return grade_medium(parsed_traj)
    elif payload.task == "hard":
        return grade_hard(parsed_traj)
    else:
        raise HTTPException(status_code=404, detail=f"Task '{payload.task}' not found")


@app.get("/health")
def health() -> dict:
    """Return environment health status."""
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict:
    """Return environment metadata."""
    return {
        "tasks": [
            {"id": "easy", "description": "Easy Task"},
            {"id": "medium", "description": "Medium Task"},
            {"id": "hard", "description": "Hard Task"}
        ]
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

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()