"""Pydantic request/response schemas for the API."""
from __future__ import annotations

from pydantic import BaseModel

from app.domain.enums import ControllerVerdict, DetectionMethod, LivenessVerdict, TurnstileState


class LivenessResponse(BaseModel):
    verdict: LivenessVerdict
    confidence: float
    method: DetectionMethod
    details: dict = {}


class SessionResponse(BaseModel):
    session_id: str


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    models_loaded: bool
    asset_error: str | None = None


class DecisionSnapshot(BaseModel):
    session_id: str
    state: TurnstileState
    verdict: ControllerVerdict | None = None
    confidence: float
    reason: str
    reason_codes: list[str]
    latency_ms: float
    camera_id: str
    device_id: str
    timestamp_utc: str
    live_score: float
    heuristic_score: float
    deep_learning_score: float | None = None
    details: dict = {}


class LatestDecisionResponse(BaseModel):
    status: str
    latest_decision: DecisionSnapshot | None = None
