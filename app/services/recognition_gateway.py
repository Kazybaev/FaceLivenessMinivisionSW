from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.core.models import TrackedFace


@dataclass
class RecognitionSubmission:
    track_id: int
    timestamp: float
    bbox: dict[str, int]


class RecognitionGateway:
    """Stub for future face-recognition integration."""

    def __init__(self):
        self._last_submission: RecognitionSubmission | None = None

    def submit(self, tracked_face: TrackedFace, frame: np.ndarray, timestamp: float) -> None:
        self._last_submission = RecognitionSubmission(
            track_id=tracked_face.track_id,
            timestamp=timestamp,
            bbox={
                "x": tracked_face.bbox.x,
                "y": tracked_face.bbox.y,
                "width": tracked_face.bbox.width,
                "height": tracked_face.bbox.height,
            },
        )

    def last_submission(self) -> RecognitionSubmission | None:
        return self._last_submission
