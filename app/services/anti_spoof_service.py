"""MiniFAS plus DeepPixBiS ensemble service."""
from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from app.config import Settings
from app.core.schemas import AntiSpoofInference, BoundingBox, ModelScore
from app.models.deeppixbis import DeepPixBiSPredictor
from app.models.mini_fas import MiniFASPredictor
from app.utils.device import resolve_execution_providers, resolve_torch_device


class AntiSpoofService:
    def __init__(self, settings: Settings) -> None:
        self._logger = logging.getLogger(__name__)
        torch_device = resolve_torch_device(settings.device)
        providers = resolve_execution_providers(settings.device)
        self._enable_tta = settings.enable_tta
        self._models = [
            MiniFASPredictor(settings.mini_fas_v2_weights, torch_device, settings.mini_fas_v2_weight),
            MiniFASPredictor(settings.mini_fas_v1se_weights, torch_device, settings.mini_fas_v1se_weight),
        ]
        self._deeppix_enabled = False
        deeppix_weights = settings.deeppixbis_weights
        if deeppix_weights.exists():
            self._models.extend(
                [
                    DeepPixBiSPredictor(
                        deeppix_weights,
                        torch_device,
                        providers,
                        settings.deeppixbis_weight,
                        settings.deeppixbis_crop_scale,
                        model_alias="DeepPixBiS",
                    ),
                    DeepPixBiSPredictor(
                        deeppix_weights,
                        torch_device,
                        providers,
                        settings.deeppixbis_context_weight,
                        settings.deeppixbis_context_crop_scale,
                        model_alias="DeepPixBiSContext",
                    ),
                ]
            )
            self._deeppix_enabled = True
            self._logger.info("DeepPixBiS loaded from %s", deeppix_weights)
        elif not settings.allow_minifas_only_fallback:
            raise FileNotFoundError(
                "DeepPixBiS weights were not found. "
                f"Expected file: {deeppix_weights}"
            )
        else:
            self._logger.warning(
                "DeepPixBiS weights are missing. Running in stricter MiniFAS-only fallback mode."
            )

    def infer(self, frame_bgr: np.ndarray, face_bbox: BoundingBox) -> AntiSpoofInference:
        started = time.perf_counter()
        flipped_frame: np.ndarray | None = None
        flipped_bbox: BoundingBox | None = None
        if self._enable_tta:
            flipped_frame = cv2.flip(frame_bgr, 1)
            flipped_bbox = BoundingBox(
                x=frame_bgr.shape[1] - face_bbox.x2,
                y=face_bbox.y,
                width=face_bbox.width,
                height=face_bbox.height,
            )

        model_scores: list[ModelScore] = []
        for model in self._models:
            base_score = model.predict(frame_bgr, face_bbox)
            if flipped_frame is None or flipped_bbox is None:
                model_scores.append(base_score)
                continue
            flip_score = model.predict(flipped_frame, flipped_bbox)
            model_scores.append(
                ModelScore(
                    model_name=base_score.model_name,
                    real_score=float((base_score.real_score + flip_score.real_score) / 2.0),
                    spoof_score=float((base_score.spoof_score + flip_score.spoof_score) / 2.0),
                    confidence=float(
                        max(
                            (base_score.real_score + flip_score.real_score) / 2.0,
                            (base_score.spoof_score + flip_score.spoof_score) / 2.0,
                        )
                    ),
                    weight=base_score.weight,
                )
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        total_weight = sum(score.weight for score in model_scores)
        real_score = sum(score.real_score * score.weight for score in model_scores) / total_weight
        spoof_score = sum(score.spoof_score * score.weight for score in model_scores) / total_weight
        return AntiSpoofInference(
            real_score=float(real_score),
            spoof_score=float(spoof_score),
            inference_time_ms=elapsed_ms,
            raw_scores=np.asarray([real_score, spoof_score], dtype=np.float32),
            model_scores=model_scores,
        )
