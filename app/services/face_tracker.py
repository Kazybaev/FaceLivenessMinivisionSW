from __future__ import annotations

import itertools

from app.config import FaceSettings
from app.core.models import FaceDetection, TrackedFace
from app.utils.geometry import bbox_iou


class FaceTracker:
    """Tracks one face across frames using a lightweight IoU matcher."""

    def __init__(self, settings: FaceSettings):
        self._settings = settings
        self._id_seq = itertools.count(1)
        self._tracked: TrackedFace | None = None

    def update(self, detections: list[FaceDetection], now: float) -> TrackedFace | None:
        if not detections:
            if self._tracked is not None and now - self._tracked.last_seen_at > self._settings.tracker_max_missing_seconds:
                self._tracked = None
            return self._tracked

        best = detections[0]
        if self._tracked is None:
            self._tracked = TrackedFace(
                track_id=next(self._id_seq),
                bbox=best.bbox,
                confidence=best.confidence,
                last_seen_at=now,
            )
            return self._tracked

        iou = bbox_iou(best.bbox, self._tracked.bbox)
        if iou < self._settings.tracker_iou_threshold:
            self._tracked = TrackedFace(
                track_id=next(self._id_seq),
                bbox=best.bbox,
                confidence=best.confidence,
                last_seen_at=now,
            )
            return self._tracked

        self._tracked.bbox = best.bbox
        self._tracked.confidence = best.confidence
        self._tracked.last_seen_at = now
        return self._tracked
