"""Use case: analyze a single image for liveness."""
from __future__ import annotations

import numpy as np
import structlog

from app.domain.entities import LivenessResult
from app.domain.enums import LivenessVerdict, DetectionMethod
from app.domain.exceptions import FaceNotFoundError, InvalidImageError
from app.ports.face_detector import FaceDetectorPort
from app.ports.liveness_analyzer import LivenessAnalyzerPort

logger = structlog.get_logger(__name__)


class AnalyzeSingleImageUseCase:
    def __init__(
        self,
        detector: FaceDetectorPort,
        analyzer: LivenessAnalyzerPort,
    ):
        self._detector = detector
        self._analyzer = analyzer

    def execute(self, image: np.ndarray) -> LivenessResult:
        if image is None or image.size == 0:
            raise InvalidImageError("Image is empty or None")

        face = self._detector.detect(image)
        if face is None:
            logger.info("no_face_detected")
            return LivenessResult(
                verdict=LivenessVerdict.NO_FACE,
                confidence=0.0,
                method=DetectionMethod.DEEP_LEARNING,
            )

        if face.image.size == 0 or face.bbox.width < 30 or face.bbox.height < 30:
            logger.info("face_too_small", width=face.bbox.width, height=face.bbox.height)
            return LivenessResult(
                verdict=LivenessVerdict.NO_FACE,
                confidence=0.0,
                method=DetectionMethod.DEEP_LEARNING,
                details={"reason": "face_too_small"},
            )

        result = self._analyzer.analyze(image, face)
        logger.info("analysis_complete", verdict=result.verdict.value,
                     confidence=result.confidence)
        return result
