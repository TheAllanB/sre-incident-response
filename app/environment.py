from app import db, scenarios, graders, rewards

# Single active episode per process (sufficient for single-agent hackathon use)
_current_episode_id: str | None = None


def reset(task_id: str) -> dict:
    global _current_episode_id
    initial_state, _ = scenarios.load_scenario(task_id)
    episode_id = db.create_episode(initial_state)
    _current_episode_id = episode_id
    state = db.read_episode(episode_id)
    return _to_observation(state)


def step(action_data: dict) -> dict:
    global _current_episode_id
    if _current_episode_id is None:
        raise ValueError("No active episode. Call /reset first.")

    state = db.read_episode(_current_episode_id)

    # Validate target service exists (escalate is exempt — it ends the episode regardless)
    service_names = [s["name"] for s in state["services"]]
    if action_data["action_type"] != "escalate" and action_data["target_service"] not in service_names:
        return {
            "observation": _to_observation(state),
            "reward": {"value": -0.1},
            "done": state.get("done", False),
            "info": {"error": f"Unknown service: {action_data['target_service']!r}"},
        }

    # Record action in history
    action_record = {
        "action_type": action_data["action_type"],
        "target_service": action_data["target_service"],
        "parameters": action_data.get("parameters", {}),
        "step": state["step"] + 1,
    }
    state["step"] += 1
    state["action_history"].append(action_record)

    # Apply effect and get result message
    result_msg, state = _apply_action(action_data, state)
    state["last_action_result"] = result_msg

    # Compute per-step reward
    reward_value = rewards.compute_reward(state, action_record)
    reward_value = max(-1.0, min(1.0, reward_value))

    # Check done conditions
    ground_truth = state["ground_truth"]
    task_id = state["task_id"]
    score = max(0.001, min(0.999, graders.grade(task_id, state, ground_truth)))

    done = False
    if score >= 0.8:
        done = True
    if state["step"] >= state.get("max_steps", 10):
        done = True
    if action_data["action_type"] == "escalate":
        done = True

    state["done"] = done
    db.write_episode(_current_episode_id, state)

    return {
        "observation": _to_observation(state),
        "reward": {"value": reward_value},
        "done": done,
        "info": {"score": score, "step": state["step"]},
    }


def get_state() -> dict:
    if _current_episode_id is None:
        return {}
    return db.read_episode(_current_episode_id)


# ─── Action Effects ────────────────────────────────────────────────────────────

def _apply_action(action: dict, state: dict) -> tuple[str, dict]:
    action_type = action["action_type"]
    target = action["target_service"]
    params = action.get("parameters", {})

    if action_type == "query_logs":
        logs = state["logs"].get(target, [])
        state["queried_logs"] = logs
        if not logs:
            return f"No logs found for {target}.", state
        return f"Retrieved {len(logs)} log entries for {target}.", state

    if action_type == "query_metrics":
        service = next((s for s in state["services"] if s["name"] == target), None)
        state["queried_logs"] = []
        if not service:
            return f"Service {target!r} not found.", state
        window = params.get("window_minutes", 60)
        msg = (
            f"Metrics for {target} (last {window}m): "
            f"health={service['health']}, cpu={service['cpu_pct']}%, "
            f"memory={service['memory_pct']}%, error_rate={service['error_rate'] * 100:.1f}%"
        )
        if window >= 120:
            extended = state.get("extended_metrics", {}).get(target)
            if extended:
                msg += f"\n\n{extended}"
        return msg, state

    if action_type == "inspect_dependencies":
        state["queried_logs"] = []
        deps = _dependency_graph(state["task_id"])
        return f"Dependency graph:\n{deps}", state

    if action_type == "restart_service":
        return _handle_restart(target, state)

    if action_type == "rollback_deploy":
        return _handle_rollback(target, state)

    if action_type == "apply_fix":
        return _handle_apply_fix(target, params, state)

    if action_type == "silence_alert":
        state["queried_logs"] = []
        for alert in state["alerts"]:
            if alert["service"] == target:
                alert["firing"] = False
        return f"Silenced all alerts for {target}.", state

    if action_type == "escalate":
        return "Incident escalated to senior on-call engineer.", state

    return f"Action {action_type!r} completed on {target}.", state


def _handle_restart(target: str, state: dict) -> tuple[str, dict]:
    gt = state["ground_truth"]
    root = gt.get("root_cause_service")
    correct_action = gt.get("correct_action")
    task_id = state["task_id"]
    state["queried_logs"] = []

    if task_id == "cascading_failure" and target == "order-service":
        rollback_done = any(
            a["action_type"] == "rollback_deploy" and a["target_service"] == "inventory-api"
            for a in state["action_history"][:-1]
        )
        if rollback_done:
            _set_healthy(target, state)
            return f"Restarted {target} — now healthy. inventory-api dependency resolved.", state
        return f"Restarted {target} — crashed again immediately. inventory-api is still down.", state

    if target == root and correct_action == "restart_service":
        _set_healthy(target, state)
        return f"Restarted {target} successfully. Service is now healthy.", state

    return f"Restarted {target}. Issues persist — root cause may be elsewhere.", state


def _handle_rollback(target: str, state: dict) -> tuple[str, dict]:
    gt = state["ground_truth"]
    root = gt.get("root_cause_service")
    state["queried_logs"] = []
    if target == root:
        _set_healthy(target, state)
        return f"Rolled back deploy on {target}. Service recovering.", state
    return f"Rolled back deploy on {target}. No improvement observed.", state


def _handle_apply_fix(target: str, params: dict, state: dict) -> tuple[str, dict]:
    gt = state["ground_truth"]
    state["queried_logs"] = []
    if target == gt.get("root_cause_service") and params.get("fix_id") == gt.get("correct_fix_id"):
        _set_healthy(target, state)
        # Also restore any services that recover once the root cause is fixed
        for svc_name in gt.get("post_fix_services", []):
            _set_healthy(svc_name, state)
        for alert in state["alerts"]:
            alert["firing"] = False
        return f"Applied fix '{params.get('fix_id')}' to {target}. Issue resolved.", state
    fix_id = params.get("fix_id")
    if not fix_id:
        return (
            f"apply_fix failed: fix_id is required. "
            f"Query logs and metrics first to identify the correct fix_id."
        ), state
    return (
        f"Applied fix '{fix_id}' to {target}. No improvement observed — "
        f"fix_id may be incorrect or wrong service targeted. "
        f"Check logs and metrics for the specific config flag or setting that needs reverting."
    ), state


def _set_healthy(service_name: str, state: dict) -> None:
    for s in state["services"]:
        if s["name"] == service_name:
            s["health"] = "healthy"
            s["error_rate"] = 0.0
            s["memory_pct"] = min(s["memory_pct"], 50.0)
    for alert in state["alerts"]:
        if alert["service"] == service_name:
            alert["firing"] = False


def _dependency_graph(task_id: str) -> str:
    graphs = {
        "single_service_crash": (
            "payment-api → cache-service, database\n"
            "auth-service → database"
        ),
        "cascading_failure": (
            "order-service → inventory-api, cache-service\n"
            "inventory-api → database\n"
            "notification-service → order-service"
        ),
        "silent_data_corruption": (
            "analytics-service → reporting-service\n"
            "reporting-service → database\n"
            "api-gateway → analytics-service, reporting-service"
        ),
        "db_connection_pool_exhaustion": (
            "api-server → postgres-db, cache-service\n"
            "worker-service → postgres-db\n"
            "notification-svc → api-server"
        ),
        "tls_certificate_expiry": (
            "frontend → api-gateway (HTTPS :443)\n"
            "api-gateway → backend-api, auth-service\n"
            "cdn-service → api-gateway (internal HTTP, separate cert)"
        ),
    }
    return graphs.get(task_id, "No dependency information available.")


def _to_observation(state: dict) -> dict:
    return {
        "step": state["step"],
        "done": state.get("done", False),
        "alerts": [a for a in state["alerts"] if a.get("firing")],
        "services": state["services"],
        "recent_logs": state.get("queried_logs", []),
        "last_action_result": state.get("last_action_result", ""),
        "episode_id": state.get("episode_id", ""),
        "slo_status": state.get("slo_status", []),
    }
