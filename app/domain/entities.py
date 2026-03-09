from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque

import numpy as np

from app.domain.enums import LivenessVerdict, DetectionMethod


@dataclass
class BBox:
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

    def clamp(self, img_width: int, img_height: int) -> BBox:
        x = max(0, self.x)
        y = max(0, self.y)
        x2 = min(img_width, self.x2)
        y2 = min(img_height, self.y2)
        return BBox(x=x, y=y, width=x2 - x, height=y2 - y)


@dataclass
class FaceRegion:
    bbox: BBox
    image: np.ndarray
    landmarks: np.ndarray | None = None


@dataclass
class LivenessResult:
    verdict: LivenessVerdict
    confidence: float
    method: DetectionMethod
    details: dict = field(default_factory=dict)


@dataclass
class CheckResult:
    name: str
    passed: bool
    value: float
    threshold: float


@dataclass
class HeuristicState:
    blink_count: int = 0
    blink_frames: int = 0
    move_count: int = 0
    ratio_ok_count: int = 0
    ear_buf: deque = field(default_factory=lambda: deque(maxlen=10))
    nose_buf: deque = field(default_factory=lambda: deque(maxlen=6))
    leye_buf: deque = field(default_factory=lambda: deque(maxlen=6))
    reye_buf: deque = field(default_factory=lambda: deque(maxlen=6))
    texture_buf: deque = field(default_factory=lambda: deque(maxlen=20))

    def reset(self) -> None:
        self.blink_count = 0
        self.blink_frames = 0
        self.move_count = 0
        self.ratio_ok_count = 0
        self.ear_buf.clear()
        self.nose_buf.clear()
        self.leye_buf.clear()
        self.reye_buf.clear()
        self.texture_buf.clear()


@dataclass
class ModelInfo:
    name: str
    path: str
    h_input: int
    w_input: int
    model_type: str
    scale: float | None
