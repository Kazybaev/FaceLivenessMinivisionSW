"""Combined analyzer: weighted combination of DL + heuristic results."""
from __future__ import annotations

import numpy as np
import structlog

from app.domain.entities import FaceRegion, LivenessResult
from app.domain.enums import LivenessVerdict, DetectionMethod
from app.infrastructure.config import CombinedConfig
from app.adapters.analyzers.deep_learning_analyzer import DeepLearningAnalyzer
from app.adapters.analyzers.heuristic_analyzer import HeuristicAnalyzer

logger = structlog.get_logger(__name__)


class CombinedAnalyzer:
    def __init__(
        self,
        config: CombinedConfig,
        dl_analyzer: DeepLearningAnalyzer,
        heuristic_analyzer: HeuristicAnalyzer,
    ):
        self._config = config
        self._dl = dl_analyzer
        self._heuristic = heuristic_analyzer

    def analyze(self, image: np.ndarray, face: FaceRegion) -> LivenessResult:
        dl_result = self._dl.analyze(image, face)
        heuristic_result = self._heuristic.analyze(image, face)
        return self._combine(dl_result, heuristic_result)

    def process_frame(self, image: np.ndarray, face: FaceRegion) -> LivenessResult:
        dl_result = self._dl.process_frame(image, face)
        heuristic_result = self._heuristic.process_frame(image, face)
        return self._combine(dl_result, heuristic_result)

    def reset(self) -> None:
        self._dl.reset()
        self._heuristic.reset()

    def _combine(self, dl_result: LivenessResult, heuristic_result: LivenessResult) -> LivenessResult:
        combined_confidence = (
            self._config.dl_weight * dl_result.confidence
            + self._config.heuristic_weight * heuristic_result.confidence
        )

        if combined_confidence >= 0.75:
            verdict = LivenessVerdict.REAL
        elif combined_confidence >= 0.4:
            verdict = LivenessVerdict.UNCERTAIN
        else:
            verdict = LivenessVerdict.FAKE

        return LivenessResult(
            verdict=verdict,
            confidence=float(combined_confidence),
            method=DetectionMethod.COMBINED,
            details={
                "dl": {
                    "verdict": dl_result.verdict.value,
                    "confidence": dl_result.confidence,
                    **dl_result.details,
                },
                "heuristic": {
                    "verdict": heuristic_result.verdict.value,
                    "confidence": heuristic_result.confidence,
                    **heuristic_result.details,
                },
            },
        )
