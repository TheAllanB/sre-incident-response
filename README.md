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
the agent receives an observation (active alerts, service metrics, recent log results) and must
choose an action to investigate or remediate the incident. The episode ends when the agent
resolves the incident, exhausts its step budget, or escalates.

## Tasks

| ID | Difficulty | Max Steps | Description |
|----|-----------|-----------|-------------|
| `single_service_crash` | Easy | 10 | `payment-api` is OOMKilled. Logs clearly show the cause. Query logs → restart service. |
| `cascading_failure` | Medium | 15 | `order-service` is down because its dependency `inventory-api` failed after a bad deploy. Must fix root cause first, then the dependent service. |
| `silent_data_corruption` | Hard | 20 | `reporting-service` appears healthy but writes corrupted output due to a bad config flag. Downstream `analytics-service` logs reveal the issue. Red herring: `cache-service` latency spike (unrelated). |
| `db_connection_pool_exhaustion` | Medium-Hard | 15 | `postgres-db` connection pool is saturated (500/500). Root cause: `worker-service` v2.1.1 introduced a connection leak — acquires DB connections but never releases them. The worker itself appears healthy by CPU/memory metrics. Red herring: `cache-service` memory alert. Fix: `apply_fix(worker-service, drain_connection_pool)`. |
| `tls_certificate_expiry` | Hard | 20 | 91% of user requests failing but `api-gateway` reports healthy (internal HTTP checks pass while external HTTPS fails). TLS certificate expired at 01:00 UTC. Red herring: `backend-api` v2.3.1 was deployed 2.5h after the incident began. Fix: `apply_fix(api-gateway, rotate_certificate)`. |

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

`recent_logs` is only populated after a `query_logs` action. `last_action_result` contains the
text output of the last action for all other action types.

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
| `query_metrics` | `window_minutes: int` (default 60) | Returns metrics; use ≥120 for trend analysis |
| `inspect_dependencies` | — | Returns service dependency graph |
| `restart_service` | — | Restarts the service |
| `rollback_deploy` | — | Reverts the most recent deploy |
| `apply_fix` | `fix_id: str` | Applies a named configuration fix |
| `silence_alert` | — | Silences alerts for the service |
| `escalate` | — | Ends episode immediately |

## Reward Function

| Condition | Reward |
|-----------|--------|
| Correct terminal action resolves root cause | +1.0 |
| Investigative action before first remediation | +0.1 |
| Wrong service targeted for remediation | −0.2 |
| Repeated identical action | −0.1 |
| Escalate without any fix attempt | −0.3 |
| All other valid steps | 0.0 |
| Invalid service name | −0.1 |

All rewards are clamped to `[-1.0, 1.0]`. Final episode score is clamped to `[0.0, 1.0]`.

`slo_status` is present in every observation and conveys real SRE urgency — `breach_estimated_minutes` signals how long before the quarterly error budget is consumed.

## Setup

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)

### Local

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
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
| `API_BASE_URL` | Yes | LLM API endpoint (e.g. `https://api.openai.com/v1`) |
| `MODEL_NAME` | Yes | Model identifier (e.g. `gpt-4o`) |
| `HF_TOKEN` | Yes | Hugging Face / API key (used as fallback if OPENAI_API_KEY not set) |
| `OPENAI_API_KEY` | Optional | OpenAI API key (preferred over HF_TOKEN if set) |
| `DB_PATH` | Optional | SQLite file path (default: `./sre_env.db`) |
| `ENV_BASE_URL` | Optional | Environment server URL for inference.py (default: `http://localhost:7860`) |

## Running the Baseline Agent

With the server running on port 7860:

```bash
python inference.py
```

The agent runs all 3 tasks and prints structured logs:

```
[START] task=single_service_crash env=sre-incident-response model=gpt-4o
[STEP] step=1 action=query_logs('payment-api') reward=0.10 done=false error=null
[STEP] step=2 action=restart_service('payment-api') reward=1.00 done=true error=null
[END] success=true steps=2 score=1.000 rewards=0.10,1.00

[START] task=cascading_failure env=sre-incident-response model=gpt-4o
...
[END] success=true steps=4 score=1.000 rewards=0.10,0.10,1.00,0.00

[START] task=silent_data_corruption env=sre-incident-response model=gpt-4o
...
[END] success=true steps=3 score=1.000 rewards=0.10,0.10,1.00
```

## API Reference

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness check — returns `{"status": "ok"}` |
| POST | `/reset` | Start episode: `{"task_id": "single_service_crash"}` |
| POST | `/step` | Take action, returns `{observation, reward, done, info}` |
| GET | `/state` | Raw episode state including ground truth (debug only) |
| GET | `/tasks` | List all available tasks |

## Tests

```bash
pytest tests/ -v
```

All 49 tests should pass.

## Hugging Face Space Deployment

1. Create a new Space (Docker SDK type) at huggingface.co/new-space
2. Tag the Space with `openenv`
3. Push this repository to the Space
4. Add all required env vars via Settings → Variables and secrets:
   - `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`, `OPENAI_API_KEY`, `DB_PATH`
5. The Space will build automatically and serve on port 7860
