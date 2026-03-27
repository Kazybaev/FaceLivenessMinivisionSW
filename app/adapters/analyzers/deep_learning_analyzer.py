"""Deep learning liveness analyzer using MiniFASNet ensemble.

Original: src/anti_spoof_predict.py:62-111 LivenessModel
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import structlog

from app.domain.entities import FaceRegion, LivenessResult, ModelInfo
from app.domain.enums import LivenessVerdict, DetectionMethod
from app.infrastructure.config import DeepLearningConfig
from app.ports.model_repository import ModelRepositoryPort
from app.ml.data.transforms import Compose, ToTensor

logger = structlog.get_logger(__name__)


class DeepLearningAnalyzer:
    def __init__(
        self,
        config: DeepLearningConfig,
        model_repo: ModelRepositoryPort,
    ):
        self._config = config
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._transform = Compose([ToTensor()])  # Created once, not per predict()
        self._models: dict[str, tuple[torch.nn.Module, int, int]] = {}
        self._score_history: deque[float] = deque(maxlen=config.smoothing_frames)
        self._load_models(model_repo)

    @property
    def model_names(self) -> list[str]:
        return sorted(self._models.keys())

    def _load_models(self, model_repo: ModelRepositoryPort) -> None:
        model_infos = model_repo.list_models()
        for info in model_infos:
            model = model_repo.load_model(info)
            model.to(self._device)
            model.eval()
            self._models[info.name] = (model, info.h_input, info.w_input)
        logger.info("models_loaded", count=len(self._models))

    def analyze(self, image: np.ndarray, face: FaceRegion) -> LivenessResult:
        prediction = self.predict_face(face.image, smooth=False)
        real_prob = prediction["real_prob"]
        fake_prob = prediction["fake_prob"]
        if real_prob is None:
            return LivenessResult(
                verdict=LivenessVerdict.UNCERTAIN,
                confidence=0.0,
                method=DetectionMethod.DEEP_LEARNING,
            )

        is_real = real_prob >= 0.5
        confidence = real_prob if is_real else (1 - real_prob)

        if confidence < self._config.confidence_threshold:
            verdict = LivenessVerdict.UNCERTAIN
        elif is_real:
            verdict = LivenessVerdict.REAL
        else:
            verdict = LivenessVerdict.FAKE

        return LivenessResult(
            verdict=verdict,
            confidence=float(confidence),
            method=DetectionMethod.DEEP_LEARNING,
            details={
                "real_prob": float(real_prob),
                "fake_prob": float(fake_prob),
            },
        )

    def process_frame(self, image: np.ndarray, face: FaceRegion) -> LivenessResult:
        """Stateful version with score smoothing."""
        prediction = self.predict_face(face.image, smooth=True)
        real_prob = prediction["real_prob"]
        fake_prob = prediction["fake_prob"]
        smoothed = prediction["smoothed"]
        if real_prob is None:
            return LivenessResult(
                verdict=LivenessVerdict.UNCERTAIN,
                confidence=0.0,
                method=DetectionMethod.DEEP_LEARNING,
            )

        is_real = smoothed >= 0.5
        confidence = smoothed if is_real else (1 - smoothed)

        if confidence < self._config.confidence_threshold:
            verdict = LivenessVerdict.UNCERTAIN
        elif is_real:
            verdict = LivenessVerdict.REAL
        else:
            verdict = LivenessVerdict.FAKE

        return LivenessResult(
            verdict=verdict,
            confidence=float(confidence),
            method=DetectionMethod.DEEP_LEARNING,
            details={
                "real_prob": float(real_prob),
                "fake_prob": float(fake_prob),
                "smoothed": float(smoothed),
            },
        )

    def reset(self) -> None:
        self._score_history.clear()

    def predict_face(self, face: np.ndarray, smooth: bool = False) -> dict[str, float | None]:
        real_prob, fake_prob = self._predict(face)
        if real_prob is None:
            return {"real_prob": None, "fake_prob": None, "smoothed": None}

        if smooth:
            self._score_history.append(real_prob)
            smoothed = float(np.mean(self._score_history))
        else:
            smoothed = float(real_prob)

        return {
            "real_prob": float(real_prob),
            "fake_prob": float(fake_prob),
            "smoothed": float(smoothed),
        }

    def _predict(self, face: np.ndarray) -> tuple[float | None, float | None]:
        real_scores: list[float] = []
        fake_scores: list[float] = []

        for model_name, (model, h_input, w_input) in self._models.items():
            face_resized = cv2.resize(face, (w_input, h_input))
            img = self._transform(face_resized).unsqueeze(0).to(self._device)
            with torch.inference_mode():
                result = model(img)
                result = F.softmax(result, dim=1).cpu().numpy()
            fake_scores.append(float(result[0][0]))
            real_scores.append(float(result[0][1]))

        if not real_scores:
            return None, None

        avg_real = float(np.mean(real_scores))
        avg_fake = float(np.mean(fake_scores))
        total = avg_real + avg_fake
        if total == 0:
            return None, None
        return avg_real / total, avg_fake / total
