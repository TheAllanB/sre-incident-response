---
title: SRE Incident Response
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
tags:
  - openenv
  - sre
  - incident-response
  - reinforcement-learning
---

# SRE Incident Response Environment

An OpenEnv-compliant environment simulating real-world SRE on-call incident triage. An AI agent
receives production alerts, service metrics, and logs from a broken system and must diagnose
root causes and apply correct remediations.

## Environment Description

The environment presents the agent with a production system in a degraded state. At each step
the agent receives an observation (active alerts, service health, recent logs, SLO status) and must
choose an action to investigate or remediate the incident. The episode ends when the agent
resolves the incident, exhausts its step budget, or escalates.

Each scenario includes realistic noise — red herring alerts, healthy-looking culprits, and
misleading metrics — requiring the agent to reason carefully rather than pattern-match on surface signals.

## Tasks

| ID | Difficulty | Max Steps | Description |
|----|-----------|-----------|-------------|
| `single_service_crash` | Easy | 10 | `payment-api` is OOMKilled. Logs clearly show the cause. Query logs → restart service. |
| `cascading_failure` | Medium | 15 | `order-service` is down because its dependency `inventory-api` failed after a bad deploy. Must fix root cause first, then restart the dependent service. |
| `silent_data_corruption` | Hard | 20 | `reporting-service` appears healthy but writes corrupted output due to a bad config flag. Downstream `analytics-service` logs reveal the issue. Red herring: `cache-service` latency spike. |
| `db_connection_pool_exhaustion` | Medium-Hard | 15 | `postgres-db` connection pool saturated (500/500). Root cause: `worker-service` v2.1.1 introduced a connection leak. The worker appears healthy by CPU/memory metrics. Red herring: `cache-service` memory alert. |
| `tls_certificate_expiry` | Hard | 20 | 91% of user requests failing but `api-gateway` reports healthy (internal HTTP checks pass, external HTTPS fails). TLS certificate expired. Red herring: a recent backend deploy. |

## Baseline Agent Results

Tested with `llama-3.1-8b-instant` via Groq (OpenAI-compatible API):

| Task | Score | Steps | Result |
|------|-------|-------|--------|
| `single_service_crash` | 1.000 | 2 | ✓ |
| `cascading_failure` | 1.000 | 6 | ✓ |
| `silent_data_corruption` | 0.000 | 15 | ✗ |
| `db_connection_pool_exhaustion` | 0.750 | 7 | ✗ |
| `tls_certificate_expiry` | 0.850 | 5 | ✓ |
| **Average** | **0.720** | | **3/5 tasks** |

The two harder tasks require multi-hop reasoning (identifying an upstream config change as the
root cause, and correlating a connection leak across service boundaries) — capabilities that
improve significantly with larger models.

## Observation Space

```json
{
  "step": 3,
  "done": false,
  "episode_id": "uuid",
  "alerts": [
    {"id": "alert-001", "service": "payment-api", "severity": "P1", "message": "...", "firing": true}
  ],
  "services": [
    {"name": "payment-api", "health": "down", "cpu_pct": 0.0, "memory_pct": 100.0, "error_rate": 1.0}
  ],
  "recent_logs": [
    {"timestamp": "2026-04-06T03:11:33Z", "service": "payment-api", "level": "FATAL", "message": "OOMKilled"}
  ],
  "last_action_result": "Retrieved 5 log entries for payment-api.",
  "slo_status": [
    {
      "service": "payment-api",
      "slo_target_pct": 99.95,
      "current_availability_pct": 0.0,
      "error_budget_remaining_pct": 31.2,
      "breach_estimated_minutes": 94
    }
  ]
}
```

`recent_logs` is populated after `query_logs`. `last_action_result` contains the text response
for all other action types. `slo_status` conveys real SRE urgency — `breach_estimated_minutes`
signals how long before the quarterly error budget is consumed.

## Action Space

```json
{
  "action_type": "query_logs | query_metrics | inspect_dependencies | restart_service | rollback_deploy | apply_fix | silence_alert | escalate",
  "target_service": "service-name",
  "parameters": {}
}
```

| Action | Parameters | Effect |
|--------|-----------|--------|
| `query_logs` | — | Populates `recent_logs` with log entries for the service |
| `query_metrics` | `window_minutes: int` (default 60) | Returns metrics; use ≥120 to reveal slow-building issues |
| `inspect_dependencies` | — | Returns the full service dependency graph |
| `restart_service` | — | Restarts the service (use for OOMKill / process crash) |
| `rollback_deploy` | — | Reverts the most recent deploy (use when deploy caused the failure) |
| `apply_fix` | `fix_id: str` | Applies a named configuration fix found in logs or extended metrics |
| `silence_alert` | — | Silences alerts for a service |
| `escalate` | — | Ends the episode immediately |

## Reward Function

| Condition | Reward |
|-----------|--------|
| Correct terminal action resolves root cause | +1.0 |
| Investigative action before first remediation | +0.1 |
| Wrong service targeted for remediation | −0.2 |
| Repeated identical action | −0.1 |
| Repeated `inspect_dependencies` call | −0.1 |
| Escalate without any fix attempt | −0.3 |
| Invalid service name | −0.1 |
| All other valid steps | 0.0 |

All rewards are clamped to `[-1.0, 1.0]`. Final episode score is clamped to `[0.0, 1.0]`.
Success threshold: score ≥ 0.8.

## Design Highlights

- **Red herrings in every scenario** — P3 alerts, healthy-looking culprits, and post-incident
  deploys force the agent to reason from evidence rather than heuristics.
- **Extended metrics** — `query_metrics` with `window_minutes=120` reveals trends invisible in
  short windows (TLS cert expiry timestamps, connection leak growth curves). The `fix_id` for
  config-based incidents is only discoverable through extended metric analysis.
- **SLO urgency** — every observation includes error budget remaining and estimated breach time,
  rewarding agents that prioritise speed when the budget is critical.
- **Partial credit graders** — scores reflect investigation quality, not just the final action.
  An agent that finds the right service but applies the wrong fix scores higher than one that
  guesses randomly.
- **Cascading failure ordering** — the grader enforces correct remediation order: fixing the
  dependency before restarting the affected service.

## API Reference

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness check — returns `{"status": "ok"}` |
| GET | `/tasks` | List all available tasks |
| POST | `/reset` | Start episode: `{"task_id": "single_service_crash"}` |
| POST | `/step` | Take action, returns `{observation, reward, done, info}` |
| POST | `/grader` | Score an action history directly without running an episode |
| GET | `/state` | Raw episode state including ground truth (debug) |
| GET | `/schema` | JSON schemas for Action, Observation, and Reward |
| GET | `/docs` | Interactive Swagger UI |

### `/grader` usage

```bash
curl -X POST /grader \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "single_service_crash",
    "action_history": [
      {"action_type": "query_logs", "target_service": "payment-api", "parameters": {}, "step": 1},
      {"action_type": "restart_service", "target_service": "payment-api", "parameters": {}, "step": 2}
    ]
  }'
# → {"task_id": "single_service_crash", "score": 1.0, "success": true}
```

## Setup

### Local

```bash
pip install -r requirements.txt
cp .env.example .env   # add your LLM API key
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

### Docker

```bash
docker build -t sre-env .
docker run -p 7860:7860 --env-file .env sre-env
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | LLM API endpoint (e.g. `https://api.groq.com/openai/v1`) |
| `MODEL_NAME` | Yes | Model identifier (e.g. `llama-3.3-70b-versatile`) |
| `HF_TOKEN` | Yes | API key for the LLM provider |
| `OPENAI_API_KEY` | Optional | OpenAI key (preferred over HF_TOKEN if set) |
| `DB_PATH` | Optional | SQLite file path (default: `./sre_env.db`) |
| `ENV_BASE_URL` | Optional | Environment server URL for inference.py (default: `http://localhost:7860`) |

## Running the Baseline Agent

With the server running on port 7860:

```bash
python inference.py
```

The agent runs all 5 tasks and prints structured logs:

```
[START] task=single_service_crash env=sre-incident-response model=llama-3.1-8b-instant
[STEP] step=1 action=query_logs('payment-api') reward=0.10 done=false error=null
[STEP] step=2 action=restart_service('payment-api') reward=1.00 done=true error=null
[END] success=true steps=2 score=1.000 rewards=0.10,1.00

[START] task=cascading_failure env=sre-incident-response model=llama-3.1-8b-instant
...
[END] success=true steps=6 score=1.000 rewards=0.10,0.10,0.10,0.10,0.00,1.00

[START] task=tls_certificate_expiry env=sre-incident-response model=llama-3.1-8b-instant
...
[END] success=true steps=5 score=0.850 rewards=0.10,0.10,0.10,0.10,1.00
```

The baseline uses tool-calling with the OpenAI client (compatible with any OpenAI-spec provider).
