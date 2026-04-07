"""Lightweight face detector based on OpenCV Haar cascade."""
from __future__ import annotations

import cv2
import numpy as np

from app.config import Settings
from app.core.schemas import BoundingBox, FaceDetection
from app.utils.image_utils import select_largest_box


class FaceDetector:
    def __init__(self, settings: Settings) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(cascade_path)
        if self._detector.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}.")
        self._scale_factor = settings.detection_scale_factor
        self._min_neighbors = settings.detection_min_neighbors
        self._min_size = (settings.detection_min_size, settings.detection_min_size)

    def detect(self, frame: np.ndarray) -> FaceDetection | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._detector.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
        )
        boxes = [BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h)) for x, y, w, h in faces]
        largest = select_largest_box(boxes)
        if largest is None:
            return None
        return FaceDetection(bbox=largest, confidence=1.0)
