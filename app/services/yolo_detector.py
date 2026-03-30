from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import structlog

from app.config import YoloSettings
from app.core.detection_policy import filter_suspicious_detections, normalize_detection_label
from app.core.models import BoundingBox, ObjectDetection

logger = structlog.get_logger(__name__)


class BaseDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[ObjectDetection]:
        ...


class MockYoloDetector:
    """Returns no suspicious objects but keeps the pipeline contract intact."""

    backend_name = "mock"

    def detect(self, frame: np.ndarray) -> list[ObjectDetection]:
        return []


class YoloDetector:
    """Optional YOLO detector with graceful fallback to a mock implementation."""

    def __init__(self, settings: YoloSettings):
        self._settings = settings
        self._detector: BaseDetector
        self.backend_name = settings.backend
        self.model_name = "mock"
        self._call_counter = 0
        self._last_detections: list[ObjectDetection] = []

        if settings.backend != "ultralytics" or not settings.model_path:
            self._detector = MockYoloDetector()
            self.backend_name = self._detector.backend_name
            self.model_name = self._detector.backend_name
            return

        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(settings.model_path)
            self._names = self._model.model.names
            self._detector = self
            self.backend_name = "ultralytics"
            self.model_name = Path(settings.model_path).name
        except Exception as exc:  # pragma: no cover - depends on optional runtime.
            logger.warning("yolo_fallback_to_mock", error=str(exc))
            self._detector = MockYoloDetector()
            self.backend_name = self._detector.backend_name
            self.model_name = self._detector.backend_name

    def detect(self, frame: np.ndarray) -> list[ObjectDetection]:
        if self._detector is not self:
            return self._detector.detect(frame)

        self._call_counter += 1
        if self._settings.run_every_n_frames > 1 and self._call_counter % self._settings.run_every_n_frames != 0:
            return list(self._last_detections)

        results = self._model.predict(
            frame,
            verbose=False,
            conf=self._settings.confidence_threshold,
            imgsz=self._settings.inference_size,
        )
        if not results:
            self._last_detections = []
            return []

        raw_detections: list[ObjectDetection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for xyxy, cls_idx, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                label = normalize_detection_label(str(self._names[int(cls_idx)]))
                x1, y1, x2, y2 = [int(v) for v in xyxy.tolist()]
                raw_detections.append(
                    ObjectDetection(
                        label=label,
                        confidence=float(conf),
                        bbox=BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
                    )
                )
        self._last_detections = filter_suspicious_detections(raw_detections, self._settings.suspicious_labels)
        return list(self._last_detections)
