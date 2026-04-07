"""Face quality checks and normalization."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import Settings


@dataclass(slots=True)
class QualityAssessment:
    ok: bool
    reasons: list[str]
    brightness: float
    contrast: float
    face_size: int
    backlight_delta: float

    @property
    def message(self) -> str:
        if self.ok:
            return "Face quality looks good."
        return "; ".join(self.reasons)


class FaceQualityService:
    def __init__(self, settings: Settings) -> None:
        self._min_face_brightness = settings.min_face_brightness
        self._min_face_contrast = settings.min_face_contrast
        self._min_face_size = settings.min_face_size
        self._max_backlight_delta = settings.max_backlight_delta

    def assess(self, frame: np.ndarray, face_crop: np.ndarray) -> QualityAssessment:
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness = float(face_gray.mean())
        contrast = float(face_gray.std())
        face_size = int(min(face_crop.shape[:2]))
        backlight_delta = float(frame_gray.mean() - brightness)

        reasons: list[str] = []
        if brightness < self._min_face_brightness:
            reasons.append("Face is too dark.")
        if contrast < self._min_face_contrast:
            reasons.append("Face contrast is too low.")
        if face_size < self._min_face_size:
            reasons.append("Move closer to the camera.")
        if backlight_delta > self._max_backlight_delta:
            reasons.append("Strong backlight detected.")

        return QualityAssessment(
            ok=not reasons,
            reasons=reasons,
            brightness=brightness,
            contrast=contrast,
            face_size=face_size,
            backlight_delta=backlight_delta,
        )

    def prepare_for_inference(self, face_crop: np.ndarray, assessment: QualityAssessment) -> np.ndarray:
        prepared = face_crop.copy()
        if assessment.brightness >= self._min_face_brightness and assessment.backlight_delta <= self._max_backlight_delta:
            return prepared

        lab = cv2.cvtColor(prepared, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        normalized_lab = cv2.merge((l_channel, a_channel, b_channel))
        prepared = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)

        current_brightness = float(cv2.cvtColor(prepared, cv2.COLOR_BGR2GRAY).mean())
        if current_brightness < 95.0:
            gain = min(1.8, 110.0 / max(current_brightness, 1.0))
            prepared = np.clip(prepared.astype(np.float32) * gain, 0.0, 255.0).astype(np.uint8)
        return prepared
