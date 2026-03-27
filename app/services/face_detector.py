from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from app.config import FaceSettings
from app.core.models import BoundingBox, FaceDetection


class BaseFaceDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        ...


class OpenCvHaarFaceDetector:
    """Simple face detector that keeps the MVP runnable without extra models."""

    def __init__(self, settings: FaceSettings):
        self._settings = settings
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = max(0.1, min(self._settings.detection_scale, 1.0))
        if scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        detections = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(
                max(16, int(self._settings.min_face_size * scale)),
                max(16, int(self._settings.min_face_size * scale)),
            ),
        )
        faces = [
            FaceDetection(
                bbox=BoundingBox(
                    x=int(x / scale),
                    y=int(y / scale),
                    width=int(w / scale),
                    height=int(h / scale),
                ),
                confidence=1.0,
            )
            for x, y, w, h in detections
        ]
        faces.sort(key=lambda item: item.bbox.area, reverse=True)
        return faces
