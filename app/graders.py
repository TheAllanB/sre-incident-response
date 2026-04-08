def grade(task_id: str, state: dict, ground_truth: dict) -> float:
    """Dispatch to the correct grader and return a score in [0.0, 1.0]."""
    graders = {
        "single_service_crash":          _grade_single_service_crash,
        "cascading_failure":             _grade_cascading_failure,
        "silent_data_corruption":        _grade_silent_data_corruption,
        "db_connection_pool_exhaustion": _grade_db_connection_pool_exhaustion,
        "tls_certificate_expiry":        _grade_tls_certificate_expiry,
    }
    fn = graders.get(task_id)
    if fn is None:
        return 0.001
    return fn(state, ground_truth)


def _grade_single_service_crash(state: dict, ground_truth: dict) -> float:
    history = state.get("action_history", [])
    root = ground_truth["root_cause_service"]
    correct_action = ground_truth["correct_action"]
    score = 0.0

    restarts = [a for a in history if a["action_type"] == correct_action and a["target_service"] == root]
    if restarts:
        score += 1.0
        first_restart_step = restarts[0]["step"]
        logs_before = [
            a for a in history
            if a["action_type"] == "query_logs" and a["step"] < first_restart_step
        ]
        if logs_before:
            score += 0.5

    wrong_restarts = [
        a for a in history
        if a["action_type"] == correct_action and a["target_service"] != root
    ]
    if wrong_restarts:
        first_wrong = wrong_restarts[0]["step"]
        first_correct = restarts[0]["step"] if restarts else float("inf")
        if first_wrong < first_correct:
            score -= 0.3

    escalations = [a for a in history if a["action_type"] == "escalate"]
    fix_attempts = [a for a in history if a["action_type"] in ("restart_service", "rollback_deploy", "apply_fix")]
    if escalations and not fix_attempts:
        score -= 0.5

    return max(0.001, min(0.999, score))


def _grade_cascading_failure(state: dict, ground_truth: dict) -> float:
    history = state.get("action_history", [])
    score = 0.0

    rollbacks = [a for a in history if a["action_type"] == "rollback_deploy" and a["target_service"] == "inventory-api"]
    order_restarts = [a for a in history if a["action_type"] == "restart_service" and a["target_service"] == "order-service"]

    if rollbacks and order_restarts:
        first_rollback = rollbacks[0]["step"]
        first_restart = order_restarts[0]["step"]
        if first_rollback < first_restart:
            score += 0.4  # correct order
            score += 0.3  # order-service restarted after rollback
        else:
            score -= 0.4  # wrong order
    elif rollbacks:
        score += 0.4  # rollback done, order-service not yet restarted
    elif order_restarts:
        score -= 0.4  # restarted order-service before fixing root cause

    if any(a["action_type"] == "inspect_dependencies" for a in history):
        score += 0.2

    remediation_types = {"restart_service", "rollback_deploy", "apply_fix"}
    remediations = [a for a in history if a["action_type"] in remediation_types]
    if remediations:
        first_step = remediations[0]["step"]
        if any(a["action_type"] == "query_logs" and a["step"] < first_step for a in history):
            score += 0.1

    bad_actions = [a for a in history if a["action_type"] in ("escalate", "silence_alert")]
    score -= min(len(bad_actions) * 0.2, 0.4)

    return max(0.001, min(0.999, score))


def _grade_silent_data_corruption(state: dict, ground_truth: dict) -> float:
    history = state.get("action_history", [])
    score = 0.0

    correct_fixes = [
        a for a in history
        if a["action_type"] == "apply_fix"
        and a["target_service"] == "reporting-service"
        and a.get("parameters", {}).get("fix_id") == "revert_config_flag"
    ]
    if correct_fixes:
        score += 0.5

    reporting_actions = [a for a in history if a["target_service"] == "reporting-service"]
    analytics_logs = [
        a for a in history
        if a["action_type"] == "query_logs" and a["target_service"] == "analytics-service"
    ]
    if reporting_actions and analytics_logs:
        first_reporting_step = reporting_actions[0]["step"]
        if any(a["step"] < first_reporting_step for a in analytics_logs):
            score += 0.2

    wide_metric_queries = [
        a for a in history
        if a["action_type"] == "query_metrics"
        and a["target_service"] == "reporting-service"
        and a.get("parameters", {}).get("window_minutes", 0) >= 120
    ]
    if wide_metric_queries:
        score += 0.2

    cache_actions = [a for a in history if a["target_service"] == "cache-service"]
    if not cache_actions:
        score += 0.1
    else:
        score -= 0.1

    wrong_fixes = [
        a for a in history
        if a["action_type"] == "apply_fix" and a["target_service"] != "reporting-service"
    ]
    if wrong_fixes:
        score -= 0.3

    reporting_restarts = [
        a for a in history
        if a["action_type"] == "restart_service" and a["target_service"] == "reporting-service"
    ]
    if reporting_restarts:
        score -= 0.2

    return max(0.001, min(0.999, score))


def _grade_db_connection_pool_exhaustion(state: dict, ground_truth: dict) -> float:
    """
    Correct path:
      +0.50  apply_fix(worker-service, drain_connection_pool) — the actual fix
      +0.25  queried worker-service logs — found the connection leak evidence
      +0.25  queried postgres-db logs OR metrics on postgres-db with window>=120

    Penalties:
      -0.30  restart_service on postgres-db — bandaid; pool re-fills in minutes
      -0.10  any action targeting cache-service (red herring)
    """
    history = state.get("action_history", [])
    score = 0.0

    # Core fix
    correct_fixes = [
        a for a in history
        if a["action_type"] == "apply_fix"
        and a["target_service"] == "worker-service"
        and a.get("parameters", {}).get("fix_id") == "drain_connection_pool"
    ]
    if correct_fixes:
        score += 0.55

    # Investigated the actual leak source
    worker_logs = [
        a for a in history
        if a["action_type"] == "query_logs" and a["target_service"] == "worker-service"
    ]
    if worker_logs:
        score += 0.25

    # Confirmed pool exhaustion at postgres-db level
    postgres_investigated = any(
        (a["action_type"] == "query_logs" and a["target_service"] == "postgres-db")
        or (
            a["action_type"] == "query_metrics"
            and a["target_service"] == "postgres-db"
            and a.get("parameters", {}).get("window_minutes", 0) >= 120
        )
        for a in history
    )
    if postgres_investigated:
        score += 0.25

    # Penalty: restarting postgres-db doesn't fix a connection leak
    postgres_restarts = [
        a for a in history
        if a["action_type"] == "restart_service" and a["target_service"] == "postgres-db"
    ]
    if postgres_restarts:
        score -= 0.30

    # Penalty: chasing the cache-service red herring
    cache_actions = [a for a in history if a["target_service"] == "cache-service"]
    if cache_actions:
        score -= 0.10

    return max(0.001, min(0.999, score))


def _grade_tls_certificate_expiry(state: dict, ground_truth: dict) -> float:
    """
    Correct path:
      +0.50  apply_fix(api-gateway, rotate_certificate)
      +0.25  query_metrics(api-gateway, window>=120) — reveals expiry timestamp and fix_id
      +0.15  query_logs(api-gateway) — TLS error evidence
      +0.10  never touched backend-api with rollback_deploy (red herring avoidance)

    Penalties:
      -0.40  rollback_deploy on backend-api (destructive red herring action)
      -0.20  restart_service on api-gateway (cert still expired after restart)
    """
    history = state.get("action_history", [])
    score = 0.0

    # Core fix
    correct_fixes = [
        a for a in history
        if a["action_type"] == "apply_fix"
        and a["target_service"] == "api-gateway"
        and a.get("parameters", {}).get("fix_id") == "rotate_certificate"
    ]
    if correct_fixes:
        score += 0.50

    # Queried metrics with wide window — this is how the agent discovers the cert expiry timeline
    wide_gateway_metrics = [
        a for a in history
        if a["action_type"] == "query_metrics"
        and a["target_service"] == "api-gateway"
        and a.get("parameters", {}).get("window_minutes", 0) >= 120
    ]
    if wide_gateway_metrics:
        score += 0.25

    # Queried api-gateway logs directly
    gateway_logs = [
        a for a in history
        if a["action_type"] == "query_logs" and a["target_service"] == "api-gateway"
    ]
    if gateway_logs:
        score += 0.15

    # Bonus for not chasing the backend-api red herring — only awarded when the agent
    # actually solved the incident (prevents an empty history from earning free points)
    backend_rollbacks = [
        a for a in history
        if a["action_type"] == "rollback_deploy" and a["target_service"] == "backend-api"
    ]
    if backend_rollbacks:
        # Heavy penalty — rollback is a destructive action taken on wrong evidence
        score -= 0.40
    elif correct_fixes:
        # Reward avoiding the red herring only when the correct fix was also applied
        score += 0.10

    # Penalty: restarting api-gateway won't rotate its TLS certificate
    gateway_restarts = [
        a for a in history
        if a["action_type"] == "restart_service" and a["target_service"] == "api-gateway"
    ]
    if gateway_restarts:
        score -= 0.20

    return max(0.001, min(0.999, score))
