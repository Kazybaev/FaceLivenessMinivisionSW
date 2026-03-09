"""Pydantic request/response schemas for the API."""
from __future__ import annotations

from pydantic import BaseModel

from app.domain.enums import LivenessVerdict, DetectionMethod


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
