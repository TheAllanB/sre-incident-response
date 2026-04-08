from typing import Literal, Optional
from pydantic import BaseModel


class Alert(BaseModel):
    id: str
    service: str
    severity: Literal["P1", "P2", "P3"]
    message: str
    firing: bool


class ServiceStatus(BaseModel):
    name: str
    health: Literal["healthy", "degraded", "down"]
    cpu_pct: float
    memory_pct: float
    error_rate: float


class LogEntry(BaseModel):
    timestamp: str
    service: str
    level: Literal["INFO", "WARN", "ERROR", "FATAL"]
    message: str


class SLOStatus(BaseModel):
    service: str
    slo_target_pct: float
    current_availability_pct: float
    error_budget_remaining_pct: float
    breach_estimated_minutes: Optional[int] = None


class Observation(BaseModel):
    step: int
    done: bool
    alerts: list[Alert]
    services: list[ServiceStatus]
    recent_logs: list[LogEntry] = []
    last_action_result: str = ""
    episode_id: str
    slo_status: list[SLOStatus] = []


class Action(BaseModel):
    action_type: Literal[
        "query_logs",
        "query_metrics",
        "inspect_dependencies",
        "restart_service",
        "rollback_deploy",
        "apply_fix",
        "silence_alert",
        "escalate",
    ]
    target_service: str
    parameters: dict = {}


class Reward(BaseModel):
    value: float
