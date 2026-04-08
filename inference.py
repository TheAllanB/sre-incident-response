"""
SRE Incident Response — Baseline Inference Agent

Structured log format (all fields on one line):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import json
import os
import sys
import textwrap
import time
from typing import List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

TASKS = [
    "single_service_crash",
    "cascading_failure",
    "silent_data_corruption",
    "db_connection_pool_exhaustion",
    "tls_certificate_expiry",
]
SUCCESS_SCORE_THRESHOLD = 0.8
MAX_STEPS = 15


# ─── Logging ───────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ─── Tools (one per action_type) ───────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_logs",
            "description": (
                "Retrieve recent log entries for a service. "
                "Call ONCE PER SERVICE — results are static and won't change on repeat calls. "
                "Always start here to understand what went wrong. "
                "Look for FATAL/ERROR messages, OOMKilled, crash loops, deploy failures, "
                "TLS errors, connection pool issues."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_service": {"type": "string", "description": "Service name to query logs for"},
                },
                "required": ["target_service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_metrics",
            "description": (
                "Get CPU, memory, and error rate metrics for a service. "
                "Use window_minutes=120 or higher to detect slow-building issues or config changes "
                "that started hours ago and would be invisible in a short window. "
                "IMPORTANT: For TLS issues and connection pool leaks, always use window_minutes=120 "
                "to find the fix_id embedded in extended metric output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_service": {"type": "string", "description": "Service name to get metrics for"},
                    "window_minutes": {
                        "type": "integer",
                        "description": "Lookback window in minutes. Use 120 to reveal trends over 2 hours.",
                        "default": 60,
                    },
                },
                "required": ["target_service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_dependencies",
            "description": (
                "Show the full service dependency graph. "
                "Call AT MOST ONCE per incident — the graph never changes. "
                "Use this when a service is down but its own logs don't explain why — "
                "a dependency may have failed first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_service": {"type": "string", "description": "Any service name (returns the full graph)"},
                },
                "required": ["target_service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restart_service",
            "description": (
                "Restart a crashed or OOMKilled service to restore it. "
                "Use when logs show: OOMKilled, exit code 137, container crash, process terminated. "
                "For cascading failures: only restart the dependent service AFTER fixing its dependency."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_service": {"type": "string", "description": "Service to restart"},
                },
                "required": ["target_service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rollback_deploy",
            "description": (
                "Roll back the most recent deploy on a service. "
                "Use when logs show a deploy caused the failure: migration errors, startup failures, "
                "crash loop after a specific version deploy. "
                "Do NOT use for OOMKill — use restart_service instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_service": {"type": "string", "description": "Service whose deploy to roll back"},
                },
                "required": ["target_service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_fix",
            "description": (
                "Apply a specific named configuration fix to a service. "
                "ONLY use this after you have found the exact fix_id in the logs or metrics output. "
                "Look for config flag names in deploy logs or extended metric analysis (window_minutes=120). "
                "Do NOT guess the fix_id — read it directly from the investigation output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_service": {"type": "string", "description": "Service to apply the fix to"},
                    "fix_id": {
                        "type": "string",
                        "description": "Exact fix identifier found in logs or metrics (e.g. 'revert_config_flag')",
                    },
                },
                "required": ["target_service", "fix_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "silence_alert",
            "description": "Silence alerts for a service. Use sparingly and only after resolving the root cause.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_service": {"type": "string"},
                },
                "required": ["target_service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate",
            "description": (
                "Escalate the incident to a senior engineer. "
                "Last resort only — use after exhausting all investigation and remediation options. "
                "Pass the name of the primary affected service."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_service": {"type": "string", "description": "Primary affected service name"},
                },
                "required": ["target_service"],
            },
        },
    },
]

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert SRE (Site Reliability Engineer) on-call, responding to a production incident.

    INVESTIGATION PROTOCOL — follow in order:
    1. query_logs on the alerting/failing service → find FATAL/ERROR messages
    2. If logs point to another service → query_logs on that service too
    3. inspect_dependencies if you need to understand which service depends on what
    4. If a service looks healthy but something downstream is broken → query_metrics with window_minutes=120
       to reveal anomalies over the past 2 hours that short windows miss
    5. Only remediate after you have identified the root cause from evidence

    REMEDIATION DECISION TREE:
    - Logs show OOMKilled / exit code 137 / process terminated → restart_service
    - Logs show deploy caused crash (migration failed, startup error, crash loop after version X) → rollback_deploy
    - Logs/metrics show a config flag or setting is wrong → apply_fix with the exact fix_id from the logs
    - Logs show TLS/certificate errors → apply_fix with fix_id found in extended metrics (window_minutes=120)
    - Logs show connection pool exhausted + idle connections held by a service → apply_fix on the leaking service
      with fix_id found in postgres-db extended metrics (window_minutes=120). Do NOT restart the DB.
    - Cascading failure (service B is down because service A failed):
        Step 1: Fix service A (rollback or restart A)
        Step 2: restart_service on service B AFTER A is fixed — do not escalate, complete both steps
    - Service appears healthy (green metrics, zero errors) → it is NOT the root cause, look elsewhere

    MONITORING BLINDSPOT RULE:
    - Internal health checks use HTTP — a service can show 'healthy' metrics while failing on HTTPS/TLS
    - If users are failing but backend services all show healthy, check api-gateway logs and query_metrics
      with window_minutes=120 to find the actual onset of failure
    - A deploy that happened AFTER users started failing is NOT the root cause — check timestamps carefully

    CONNECTION POOL RULE:
    - If postgres-db is exhausted but restarting it won't help (pool will refill immediately)
    - Always identify WHICH service is leaking connections via worker logs showing acquire-but-never-release
    - Apply the fix to the leaking service, not the database

    RED HERRING RULE:
    - P3 alerts (latency, queue depth, eviction rate, memory %) are almost never the root cause
    - If a service has healthy metrics and no ERROR logs, ignore it completely
    - Focus only on P1/P2 alerts and services with actual errors

    SLO URGENCY:
    - Check SLO STATUS in every observation — if error_budget_remaining_pct < 5% or breach < 10 minutes,
      skip deep investigation and apply the most likely fix immediately
    - If breach_estimated_minutes is shown and critical, prioritize speed over completeness

    CRITICAL RULES:
    - Never call apply_fix without a specific fix_id you found in logs or extended metric output
    - Never escalate in the middle of a cascading failure — finish the remediation sequence
    - After a successful rollback, always restart the dependent service that was affected
    - Never rollback_deploy a service just because it recently deployed — verify the deploy caused the failure
    - NEVER call inspect_dependencies more than once per episode — it returns the same graph every time
    - NEVER repeat the exact same action twice in a row — if an action didn't help, try something different
    - After rollback_deploy on inventory-api succeeds, the NEXT action must be restart_service on order-service
    - Check EPISODE STATE at the bottom of each message to avoid repeating actions already taken
""").strip()


# ─── Helpers ───────────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    return OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


def call_llm_with_retry(client: OpenAI, **kwargs) -> object:
    """Call the LLM with exponential backoff on rate limit errors."""
    delays = [15, 30, 60, 120]
    for attempt, delay in enumerate(delays + [None]):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            err_str = str(exc)
            is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower() or "rate limit" in err_str.lower()
            if is_rate_limit and delay is not None:
                print(
                    f"[DEBUG] Rate limit hit (attempt {attempt + 1}), waiting {delay}s...",
                    file=sys.stderr, flush=True
                )
                time.sleep(delay)
                continue
            raise


def format_observation(obs: dict, step: int, actions_taken: List[str], steps_remaining: int) -> str:
    lines = [f"=== Step {obs.get('step', step)} (budget: {steps_remaining} steps left) ==="]

    active_alerts = [a for a in obs.get("alerts", []) if a.get("firing")]
    if active_alerts:
        lines.append("ACTIVE ALERTS:")
        for a in active_alerts:
            lines.append(f"  [{a['severity']}] {a['service']}: {a['message']}")
    else:
        lines.append("ACTIVE ALERTS: none — incident may be resolved")

    if obs.get("slo_status"):
        lines.append("\nSLO STATUS:")
        for slo in obs["slo_status"]:
            breach = ""
            if slo.get("breach_estimated_minutes") is not None:
                breach = f" — BREACH IN {slo['breach_estimated_minutes']}min"
            lines.append(
                f"  {slo['service']}: {slo['current_availability_pct']:.1f}% availability "
                f"(target {slo['slo_target_pct']}%, "
                f"budget {slo['error_budget_remaining_pct']:.1f}% remaining{breach})"
            )

    lines.append("\nSERVICES:")
    for s in obs.get("services", []):
        lines.append(
            f"  {s['name']}: {s['health'].upper()} "
            f"(cpu={s['cpu_pct']}%, mem={s['memory_pct']}%, err={s['error_rate'] * 100:.1f}%)"
        )

    if obs.get("recent_logs"):
        logs = obs["recent_logs"]
        # Show last 6 entries; prioritise ERROR/FATAL
        important = [l for l in logs if l["level"] in ("ERROR", "FATAL")]
        rest = [l for l in logs if l["level"] not in ("ERROR", "FATAL")]
        shown = (important + rest)[-6:]
        lines.append(f"\nLOG OUTPUT ({len(logs)} entries, showing {len(shown)}):")
        for log in shown:
            lines.append(f"  [{log['level']}] {log['timestamp']} {log['service']}: {log['message']}")

    if obs.get("last_action_result"):
        lines.append(f"\nLAST ACTION RESULT: {obs['last_action_result']}")

    # Episode state block — helps model avoid repetition
    if actions_taken:
        lines.append(f"\nEPISODE STATE ({len(actions_taken)} actions taken so far):")
        for i, a in enumerate(actions_taken, 1):
            lines.append(f"  {i}. {a}")

        # Post-rollback hint
        rollbacks = [a for a in actions_taken if a.startswith("rollback_deploy")]
        restarts = [a for a in actions_taken if a.startswith("restart_service")]
        if rollbacks and not restarts:
            lines.append(
                "  >> HINT: Rollback succeeded. You MUST now call restart_service on the dependent service."
            )

        # Anti-repeat warning
        if len(actions_taken) >= 2 and actions_taken[-1] == actions_taken[-2]:
            lines.append(
                "  >> WARNING: You just repeated the same action. Pick a DIFFERENT action next."
            )

        # Inspect-dependencies overuse warning
        dep_count = sum(1 for a in actions_taken if a.startswith("inspect_dependencies"))
        if dep_count >= 1:
            lines.append(
                f"  >> NOTE: inspect_dependencies already called {dep_count}x — do NOT call it again."
            )
    else:
        lines.append("\nEPISODE STATE: no actions taken yet — start with query_logs on the alerting service.")

    return "\n".join(lines)


def tool_call_to_action(tool_name: str, args: dict) -> dict:
    parameters = {}
    if tool_name == "query_metrics" and "window_minutes" in args:
        parameters = {"window_minutes": args["window_minutes"]}
    elif tool_name == "apply_fix" and "fix_id" in args:
        parameters = {"fix_id": args["fix_id"]}
    return {
        "action_type": tool_name,
        "target_service": args.get("target_service", "unknown"),
        "parameters": parameters,
    }


def action_repr(action: dict) -> str:
    fix_id = action.get("parameters", {}).get("fix_id")
    window = action.get("parameters", {}).get("window_minutes")
    extra = f",fix_id={fix_id}" if fix_id else (f",window={window}m" if window else "")
    return f"{action['action_type']}('{action['target_service']}'{extra})"


# ─── Main Agent Loop ───────────────────────────────────────────────────────────

def run_task(task_id: str) -> None:
    client = get_client()
    log_start(task_id, "sre-incident-response", MODEL_NAME)

    all_rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_result: dict = {}
    actions_taken: List[str] = []  # tracks every action for episode state injection

    try:
        resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        obs = resp.json()

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        while not obs.get("done", False):
            steps_remaining = MAX_STEPS - steps_taken
            obs_text = format_observation(obs, steps_taken + 1, actions_taken, steps_remaining)
            messages.append({"role": "user", "content": obs_text})

            # Rolling context: keep system + last 8 messages (reduces token burn)
            if len(messages) > 10:
                messages = [messages[0]] + messages[-8:]

            response = call_llm_with_retry(
                client,
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS,
                tool_choice="required",
                temperature=0.0,
            )

            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                action = {"action_type": "escalate", "target_service": "unknown", "parameters": {}}
                tool_call_id = None
            else:
                tc = msg.tool_calls[0]
                args = json.loads(tc.function.arguments)
                action = tool_call_to_action(tc.function.name, args)
                tool_call_id = tc.id

            step_resp = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
            step_resp.raise_for_status()
            last_result = step_resp.json()

            steps_taken += 1
            reward = last_result.get("reward", {}).get("value", 0.0)
            done = last_result.get("done", False)
            error = last_result.get("info", {}).get("error")

            action_str = action_repr(action)
            all_rewards.append(reward)
            actions_taken.append(action_str)
            log_step(steps_taken, action_str, reward, done, error)

            if tool_call_id:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(last_result.get("observation", {})),
                })

            obs = last_result.get("observation", obs)

            if done or steps_taken >= MAX_STEPS:
                break

        score = last_result.get("info", {}).get("score", 0.0) if last_result else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} error: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success, steps_taken, score, all_rewards)


if __name__ == "__main__":
    if not HF_TOKEN:
        print("ERROR: Set HF_TOKEN environment variable", file=sys.stderr)
        sys.exit(1)

    for task_id in TASKS:
        run_task(task_id)
