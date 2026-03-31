from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.core.models import AntiSpoofLabel, DecisionVerdict, EventSeverity, SessionState


class AntiSpoofResultSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    label: AntiSpoofLabel
    confidence: float
    model_name: str
    details: dict = Field(default_factory=dict)


class DecisionRecordSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    session_id: str
    verdict: DecisionVerdict
    state: SessionState
    allow_face_recognition: bool
    confidence: float
    reason: str
    timestamp: str
    cooldown_until: float | None = None
    details: dict = Field(default_factory=dict)


class SecurityEventSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    event_id: str
    timestamp: str
    event_type: str
    severity: EventSeverity
    message: str
    session_id: str | None = None
    payload: dict = Field(default_factory=dict)


class SessionSnapshotSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    session_id: str
    state: SessionState
    track_id: int | None = None
    blocked_by_suspicious_object: bool
    suspicious_object_seen: bool
    suspicious_object_types: list[str]
    suspicious_labels: list[str]
    buffered_frames: int
    cooldown_remaining_seconds: float
    blocked_reason: str | None = None
    last_anti_spoof_result: AntiSpoofResultSchema | None = None
    last_decision: DecisionRecordSchema | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    status: str
    issues: list[str] = Field(default_factory=list)


class PerformanceSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    processed_fps: float
    last_process_ms: float
    avg_process_ms: float
    frames_processed_total: int
    anti_spoof_cache_hits: int
    anti_spoof_inferences: int


class StatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    runtime_running: bool
    camera_running: bool
    camera_error: str | None = None
    camera_source: str | int | None = None
    yolo_backend: str
    yolo_model: str
    anti_spoof_backend: str
    active_liveness_backend: str
    frame_counter: int
    ready: bool
    readiness_issues: list[str]
    performance: PerformanceSchema
    session: SessionSnapshotSchema


class SessionResetResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    status: str
    session_id: str


class EventsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    events: list[SecurityEventSchema]


class CurrentDecisionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    decision: DecisionRecordSchema | None = None
