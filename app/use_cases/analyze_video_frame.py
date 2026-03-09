"""Use case: analyze video frames with stateful tracking."""
from __future__ import annotations

import numpy as np

from app.domain.entities import LivenessResult
from app.domain.enums import LivenessVerdict, DetectionMethod
from app.ports.face_detector import FaceDetectorPort
from app.ports.liveness_analyzer import StatefulLivenessAnalyzerPort


class AnalyzeVideoFrameUseCase:
    def __init__(
        self,
        detector: FaceDetectorPort,
        analyzer: StatefulLivenessAnalyzerPort,
    ):
        self._detector = detector
        self._analyzer = analyzer

    def execute(self, frame: np.ndarray) -> LivenessResult:
        face = self._detector.detect(frame)
        if face is None:
            return LivenessResult(
                verdict=LivenessVerdict.NO_FACE,
                confidence=0.0,
                method=DetectionMethod.HEURISTIC,
            )

        return self._analyzer.process_frame(frame, face)

    def reset(self) -> None:
        self._analyzer.reset()
