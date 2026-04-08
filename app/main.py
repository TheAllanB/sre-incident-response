from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from app import environment, graders
from app.models import Action, Observation, Reward

app = FastAPI(title="SRE Incident Response Environment", version="1.0.0")

REQUIRED_ENDPOINTS = ["/health", "/tasks", "/reset", "/step", "/state", "/schema", "/grader"]


@app.get("/", response_class=HTMLResponse)
def root():
    html_file = Path("static/index.html")
    if html_file.exists():
        return HTMLResponse(content=html_file.read_text())
    # Fallback JSON if static file missing
    return HTMLResponse(content=str({
        "name": "sre-incident-response",
        "status": "running",
        "docs": "/docs",
        "required_endpoints": REQUIRED_ENDPOINTS,
    }))


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


@app.post("/grader")
def grader(body: dict):
    """
    Score a completed action history without running a live episode.
    Body: { "task_id": "single_service_crash", "action_history": [...], "ground_truth": {...} }
    Returns: { "score": 0.0-1.0 }
    """
    task_id = body.get("task_id")
    action_history = body.get("action_history", [])
    ground_truth = body.get("ground_truth")

    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")

    # Allow caller to omit ground_truth — load it from the scenario file
    if ground_truth is None:
        try:
            from app.scenarios import load_scenario
            _, ground_truth = load_scenario(task_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    state = {"action_history": action_history, "task_id": task_id, "ground_truth": ground_truth}
    score = max(0.001, min(0.999, graders.grade(task_id, state, ground_truth)))
    return {"task_id": task_id, "score": round(score, 3), "success": score >= 0.8}


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
async def reset(request: Request):
    task_id = None

    # Try JSON body — check multiple possible key names
    try:
        body = await request.json()
        if isinstance(body, dict):
            task_id = body.get("task_id") or body.get("task") or body.get("id")
    except Exception:
        pass

    # Try query params as fallback
    if not task_id:
        task_id = request.query_params.get("task_id") or request.query_params.get("task")

    # Default to first task if checker sends empty body (connectivity probe)
    if not task_id:
        task_id = "single_service_crash"

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
