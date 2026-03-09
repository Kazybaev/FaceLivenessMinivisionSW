from __future__ import annotations

from typing import Protocol

import numpy as np

from app.domain.entities import FaceRegion, LivenessResult


class LivenessAnalyzerPort(Protocol):
    def analyze(self, image: np.ndarray, face: FaceRegion) -> LivenessResult: ...


class StatefulLivenessAnalyzerPort(Protocol):
    def process_frame(self, image: np.ndarray, face: FaceRegion) -> LivenessResult: ...
    def reset(self) -> None: ...
