"""Simple JSONL frame logger for calibration."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.core.schemas import AntiSpoofInference, BoundingBox, Decision


class EventLogger:
    def __init__(self, log_path: Path, logger_name: str = "hard.antispoof") -> None:
        self._log_path = log_path
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(logger_name)

    def log_frame(
        self,
        *,
        decision: Decision,
        inference: AntiSpoofInference | None,
        inference_time_ms: float,
        face_detected: bool,
        bbox: BoundingBox | None,
        error: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "label": decision.label.value,
            "confidence": round(decision.confidence, 4),
            "real_score": round(decision.real_score, 4),
            "spoof_score": round(decision.spoof_score, 4),
            "reason": decision.reason,
            "inference_time_ms": round(inference_time_ms, 3),
            "face_detected": face_detected,
            "bbox": None if bbox is None else {
                "x": bbox.x,
                "y": bbox.y,
                "width": bbox.width,
                "height": bbox.height,
            },
            "model_scores": [] if inference is None else [
                {
                    "model_name": score.model_name,
                    "real_score": round(score.real_score, 4),
                    "spoof_score": round(score.spoof_score, 4),
                    "confidence": round(score.confidence, 4),
                    "weight": round(score.weight, 4),
                }
                for score in inference.model_scores
            ],
            "temporal": None if inference is None or inference.temporal is None else {
                "live_score": round(inference.temporal.live_score, 4),
                "spoof_score": round(inference.temporal.spoof_score, 4),
                "motion_score": round(inference.temporal.motion_score, 4),
                "diversity_score": round(inference.temporal.diversity_score, 4),
                "rigidity_score": round(inference.temporal.rigidity_score, 4),
                "frame_count": inference.temporal.frame_count,
                "detail": inference.temporal.detail,
            },
            "presentation": None if inference is None or inference.presentation is None else {
                "attack_score": round(inference.presentation.attack_score, 4),
                "bezel_score": round(inference.presentation.bezel_score, 4),
                "rectangle_score": round(inference.presentation.rectangle_score, 4),
                "glare_score": round(inference.presentation.glare_score, 4),
                "detail": inference.presentation.detail,
            },
            "landmarks": None if inference is None or inference.landmarks is None else {
                "live_score": round(inference.landmarks.live_score, 4),
                "spoof_score": round(inference.landmarks.spoof_score, 4),
                "motion_score": round(inference.landmarks.motion_score, 4),
                "depth_score": round(inference.landmarks.depth_score, 4),
                "rigidity_score": round(inference.landmarks.rigidity_score, 4),
                "frame_count": inference.landmarks.frame_count,
                "point_count": inference.landmarks.point_count,
                "detail": inference.landmarks.detail,
            },
            "error": error,
        }
        self._logger.info("%s | conf=%.3f", decision.label.value, decision.confidence)
        with self._log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")
