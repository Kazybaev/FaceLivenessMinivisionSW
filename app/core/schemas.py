"""Dataclasses shared between runtime layers."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.core.enums import AntiSpoofLabel


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
        return self.width * self.height

    def clamp(self, frame_width: int, frame_height: int) -> BoundingBox:
        x1 = max(0, self.x)
        y1 = max(0, self.y)
        x2 = min(frame_width, self.x2)
        y2 = min(frame_height, self.y2)
        return BoundingBox(
            x=x1,
            y=y1,
            width=max(0, x2 - x1),
            height=max(0, y2 - y1),
        )


@dataclass(slots=True)
class FaceDetection:
    bbox: BoundingBox
    confidence: float = 1.0


@dataclass(slots=True)
class ModelScore:
    model_name: str
    real_score: float
    spoof_score: float
    confidence: float
    weight: float


@dataclass(slots=True)
class TemporalEvidence:
    live_score: float
    spoof_score: float
    motion_score: float
    diversity_score: float
    rigidity_score: float
    frame_count: int
    detail: str


@dataclass(slots=True)
class PresentationAttackEvidence:
    attack_score: float
    bezel_score: float
    rectangle_score: float
    glare_score: float
    detail: str


@dataclass(slots=True)
class LandmarkObservation:
    points: np.ndarray
    depth_values: np.ndarray


@dataclass(slots=True)
class LandmarkEvidence:
    live_score: float
    spoof_score: float
    motion_score: float
    depth_score: float
    rigidity_score: float
    frame_count: int
    point_count: int
    detail: str


@dataclass(slots=True)
class AntiSpoofInference:
    real_score: float
    spoof_score: float
    inference_time_ms: float
    raw_scores: np.ndarray | None
    window_size: int = 1
    model_scores: list[ModelScore] = field(default_factory=list)
    temporal: TemporalEvidence | None = None
    presentation: PresentationAttackEvidence | None = None
    landmarks: LandmarkEvidence | None = None
    landmark_points: np.ndarray | None = None


@dataclass(slots=True)
class Decision:
    label: AntiSpoofLabel
    confidence: float
    real_score: float
    spoof_score: float
    reason: str


@dataclass(slots=True)
class FrameAnalysis:
    decision: Decision
    face_detection: FaceDetection | None
    inference_time_ms: float
    fps: float
    inference: AntiSpoofInference | None = None
