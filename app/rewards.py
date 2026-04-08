from app.graders import grade

_REMEDIATION_TYPES = {"restart_service", "rollback_deploy", "apply_fix"}
_INVESTIGATIVE_TYPES = {"query_logs", "query_metrics", "inspect_dependencies"}


def compute_reward(state: dict, action_record: dict) -> float:
    """Per-step reward signal. Clamped to [-1.0, 1.0]."""
    history = state.get("action_history", [])
    action_type = action_record["action_type"]
    target = action_record["target_service"]
    ground_truth = state.get("ground_truth", {})
    task_id = state.get("task_id", "")

    # Repeated identical action (same type + same target)
    if len(history) >= 2:
        prev = history[-2]
        if prev["action_type"] == action_type and prev["target_service"] == target:
            return -0.1

    # inspect_dependencies always returns the same graph — penalise any call after the first
    if action_type == "inspect_dependencies":
        prior_dep_calls = [a for a in history[:-1] if a["action_type"] == "inspect_dependencies"]
        if prior_dep_calls:
            return -0.1

    # Escalate without any prior fix attempt
    if action_type == "escalate":
        prior_fixes = [a for a in history[:-1] if a["action_type"] in _REMEDIATION_TYPES]
        if not prior_fixes:
            return -0.3

    # Remediation actions
    if action_type in _REMEDIATION_TYPES:
        root = ground_truth.get("root_cause_service")
        # Wrong service targeted (with cascading-failure exception for order-service)
        if root and target != root:
            if not (task_id == "cascading_failure" and target == "order-service"):
                return -0.2

        # Check if episode is solved (score threshold reached)
        score = grade(task_id, state, ground_truth)
        if score >= 0.8:
            return 1.0

    # Investigative action before any remediation
    if action_type in _INVESTIGATIVE_TYPES:
        prior_remediations = [a for a in history[:-1] if a["action_type"] in _REMEDIATION_TYPES]
        if not prior_remediations:
            return 0.1

    return 0.0
