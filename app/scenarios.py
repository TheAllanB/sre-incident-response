import json
from pathlib import Path

SCENARIO_MAP = {
    "single_service_crash":        "easy.json",
    "cascading_failure":           "medium.json",
    "silent_data_corruption":      "hard.json",
    "db_connection_pool_exhaustion": "medium_hard.json",
    "tls_certificate_expiry":      "hard2.json",
}

SCENARIOS_DIR = Path(__file__).parent.parent / "data" / "scenarios"


def load_scenario(task_id: str) -> tuple[dict, dict]:
    """Returns (initial_state, ground_truth). Ground truth is embedded in
    initial_state under 'ground_truth' key (for environment use) but also
    returned separately so callers can pass it directly to graders."""
    filename = SCENARIO_MAP.get(task_id)
    if filename is None:
        raise ValueError(f"Unknown task_id: {task_id!r}. Valid: {list(SCENARIO_MAP)}")

    data = json.loads((SCENARIOS_DIR / filename).read_text())
    ground_truth = data["ground_truth"]

    initial_state = {
        "task_id": data["task_id"],
        "max_steps": data["max_steps"],
        "step": 0,
        "done": False,
        "services": data["services"],
        "alerts": data["alerts"],
        "logs": data["logs"],
        "extended_metrics": data.get("extended_metrics", {}),
        "slo_status": data.get("slo_status", []),
        "queried_logs": [],
        "last_action_result": "",
        "action_history": [],
        "ground_truth": ground_truth,
    }
    return initial_state, ground_truth


def list_tasks() -> list[dict]:
    return [
        {
            "id": "single_service_crash",
            "difficulty": "easy",
            "description": "A single service is down with clear error logs. Identify and restart it.",
        },
        {
            "id": "cascading_failure",
            "difficulty": "medium",
            "description": "Two services are failing due to a dependency issue. Find root cause and fix in order.",
        },
        {
            "id": "silent_data_corruption",
            "difficulty": "hard",
            "description": "A service appears healthy but produces corrupted output. Misleading metrics included.",
        },
        {
            "id": "db_connection_pool_exhaustion",
            "difficulty": "medium-hard",
            "description": "Database connection pool exhausted by a leaking worker service. The culprit appears healthy by standard metrics.",
        },
        {
            "id": "tls_certificate_expiry",
            "difficulty": "hard",
            "description": "TLS certificate expired causing 91% user-facing failure. The gateway reports healthy internally. A recent deploy is a red herring.",
        },
    ]
