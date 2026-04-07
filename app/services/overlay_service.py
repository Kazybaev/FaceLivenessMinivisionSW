"""Overlay rendering for the webcam window."""
from __future__ import annotations

import cv2
import numpy as np

from app.config import Settings
from app.core.enums import AntiSpoofLabel
from app.core.schemas import AntiSpoofInference, Decision, FaceDetection


class OverlayService:
    _COLORS = {
        AntiSpoofLabel.REAL: (40, 210, 40),
        AntiSpoofLabel.FAKE: (40, 40, 230),
        AntiSpoofLabel.NO_FACE: (190, 190, 190),
    }

    def __init__(self, settings: Settings) -> None:
        self._show_fps = settings.show_fps
        self._screen_attack_real_max = settings.screen_attack_real_max

    def draw(
        self,
        frame: np.ndarray,
        *,
        face_detection: FaceDetection | None,
        decision: Decision,
        inference: AntiSpoofInference | None,
        inference_time_ms: float,
        fps: float,
    ) -> np.ndarray:
        canvas = frame.copy()
        color = self._COLORS[decision.label]

        if face_detection is not None:
            bbox = face_detection.bbox
            cv2.rectangle(canvas, (bbox.x, bbox.y), (bbox.x2, bbox.y2), color, 2)
        if inference is not None and inference.landmark_points is not None:
            for point in inference.landmark_points:
                px = int(round(float(point[0])))
                py = int(round(float(point[1])))
                cv2.circle(canvas, (px, py), 3, (70, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 5, (12, 32, 40), 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 212), (18, 22, 28), -1)
        cv2.putText(canvas, decision.label.value.upper(), (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"conf={decision.confidence:.2f} real={decision.real_score:.2f} spoof={decision.spoof_score:.2f}",
            (18, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "STRICT MINIFAS + DEEPPIXBIS",
            (18, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (180, 220, 255),
            1,
            cv2.LINE_AA,
        )

        if inference is not None and inference.model_scores:
            mini_scores = [score for score in inference.model_scores if "MiniFAS" in score.model_name]
            deep_scores = [score for score in inference.model_scores if "DeepPixBiS" in score.model_name]
            if mini_scores:
                mini_real = sum(score.real_score for score in mini_scores) / len(mini_scores)
                mini_spoof = sum(score.spoof_score for score in mini_scores) / len(mini_scores)
                cv2.putText(
                    canvas,
                    f"MiniFAS family real={mini_real:.2f} spoof={mini_spoof:.2f}",
                    (18, 126),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (120, 220, 120) if mini_real >= mini_spoof else (100, 120, 255),
                    1,
                    cv2.LINE_AA,
                )
            if deep_scores:
                deep_real = sum(score.real_score for score in deep_scores) / len(deep_scores)
                deep_spoof = sum(score.spoof_score for score in deep_scores) / len(deep_scores)
                cv2.putText(
                    canvas,
                    f"DeepPixBiS family real={deep_real:.2f} spoof={deep_spoof:.2f}",
                    (18, 144),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (120, 220, 120) if deep_real >= deep_spoof else (100, 120, 255),
                    1,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    canvas,
                    "DeepPixBiS not loaded: strict MiniFAS-only fallback",
                    (18, 144),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (120, 180, 255),
                    1,
                    cv2.LINE_AA,
                )
            if inference.temporal is not None:
                cv2.putText(
                    canvas,
                    f"Temporal {inference.temporal.detail}",
                    (18, 162),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (180, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
            if inference.presentation is not None:
                cv2.putText(
                    canvas,
                    f"Screen {inference.presentation.detail}",
                    (18, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (255, 210, 150) if inference.presentation.attack_score < self._screen_attack_real_max else (100, 120, 255),
                    1,
                    cv2.LINE_AA,
                )
            if inference.landmarks is not None:
                cv2.putText(
                    canvas,
                    f"Points {inference.landmarks.detail}",
                    (18, 198),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.44,
                    (150, 230, 255) if inference.landmarks.live_score >= inference.landmarks.spoof_score else (100, 120, 255),
                    1,
                    cv2.LINE_AA,
                )

        info_line = f"infer={inference_time_ms:.1f} ms"
        if self._show_fps:
            info_line += f" fps={fps:.1f}"
        cv2.putText(
            canvas,
            info_line,
            (canvas.shape[1] - 280, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (210, 210, 210),
            2,
            cv2.LINE_AA,
        )
        reason = decision.reason if len(decision.reason) <= 140 else decision.reason[:137] + "..."
        cv2.putText(
            canvas,
            reason,
            (18, canvas.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
            cv2.LINE_AA,
        )
        return canvas
