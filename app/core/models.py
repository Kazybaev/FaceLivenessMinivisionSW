from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class SessionState(str, Enum):
    IDLE = "idle"
    OBSERVING = "observing"
    SUSPICIOUS_OBJECT_DETECTED = "suspicious_object_detected"
    SPOOF_DETECTED = "spoof_detected"
    REAL_DETECTED = "real_detected"
    BLOCKED = "blocked"
    COOLDOWN = "cooldown"
    ALLOWED = "allowed"


class AntiSpoofLabel(str, Enum):
    REAL = "real"
    SPOOF = "spoof"
    UNCERTAIN = "uncertain"


class DecisionVerdict(str, Enum):
    PENDING = "pending"
    ALLOW = "allow"
    DENY = "deny"


class EventSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(slots=True)
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


@dataclass(slots=True)
class FramePacket:
    frame_id: int
    timestamp: float
    frame: np.ndarray


@dataclass(slots=True)
class ObjectDetection:
    label: str
    confidence: float
    bbox: BoundingBox


@dataclass(slots=True)
class FaceDetection:
    bbox: BoundingBox
    confidence: float


@dataclass(slots=True)
class TrackedFace:
    track_id: int
    bbox: BoundingBox
    confidence: float
    last_seen_at: float


@dataclass(slots=True)
class AntiSpoofResult:
    label: AntiSpoofLabel
    confidence: float
    model_name: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DecisionRecord:
    session_id: str
    verdict: DecisionVerdict
    state: SessionState
    allow_face_recognition: bool
    confidence: float
    reason: str
    timestamp: str
    cooldown_until: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SecurityEvent:
    event_id: str
    timestamp: str
    event_type: str
    severity: EventSeverity
    message: str
    session_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessSession:
    session_id: str
    state: SessionState = SessionState.IDLE
    track_id: int | None = None
    started_at: float = 0.0
    last_seen_at: float = 0.0
    cooldown_until: float | None = None
    state_deadline: float | None = None
    last_suspicious_at: float | None = None
    blocked_by_suspicious_object: bool = False
    suspicious_object_seen: bool = False
    suspicious_object_types: list[str] = field(default_factory=list)
    suspicious_labels: set[str] = field(default_factory=set)
    frame_buffer: deque[FramePacket] = field(default_factory=deque)
    last_anti_spoof_result: AntiSpoofResult | None = None
    last_decision: DecisionRecord | None = None
    blocked_reason: str | None = None
