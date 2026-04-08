from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request

from app import environment
from app.models import Action, Observation, Reward

app = FastAPI(title="SRE Incident Response Environment", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    return {
        "name": "sre-incident-response",
        "description": (
            "An SRE on-call simulation where an agent triages production alerts, "
            "investigates logs and metrics, and applies remediation actions."
        ),
        "version": "1.0.0",
        "tags": ["sre", "incident-response", "observability", "devops", "openenv"],
    }


@app.get("/schema")
def schema():
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": {
            "type": "object",
            "description": "Raw episode state including internal fields (debug only)",
            "properties": {
                "task_id": {"type": "string"},
                "step": {"type": "integer"},
                "done": {"type": "boolean"},
                "services": {"type": "array"},
                "alerts": {"type": "array"},
                "logs": {"type": "object"},
                "action_history": {"type": "array"},
            },
        },
        "reward": Reward.model_json_schema(),
    }


@app.post("/mcp")
async def mcp(request: Request):
    """Minimal MCP/JSON-RPC 2.0 endpoint for OpenEnv compliance."""
    try:
        body = await request.json()
    except Exception:
        body = {}
    return {
        "jsonrpc": "2.0",
        "id": body.get("id", 1),
        "result": {
            "name": "sre-incident-response",
            "description": "SRE incident triage environment",
        },
    }


@app.get("/tasks")
def tasks():
    spec = yaml.safe_load(Path("openenv.yaml").read_text())
    return spec.get("tasks", [])


@app.post("/reset")
def reset(body: dict):
    task_id = body.get("task_id")
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")
    try:
        obs = environment.reset(task_id)
        return obs
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step")
def step(action: Action):
    try:
        result = environment.step(action.model_dump())
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state")
def state():
    return environment.get_state()
