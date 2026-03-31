from __future__ import annotations

import time

import numpy as np

from app.config import get_settings
from app.core.models import (
    AntiSpoofLabel,
    AntiSpoofResult,
    BoundingBox,
    FaceDetection,
    FramePacket,
    TrackedFace,
)
from app.core.pipeline import AccessControlPipeline


class _FakeYoloDetector:
    backend_name = "ultralytics"
    model_name = "fake_yolo.pt"

    def detect(self, frame):
        return []


class _FakeFaceDetector:
    def detect(self, frame):
        return [FaceDetection(bbox=BoundingBox(x=10, y=10, width=40, height=40), confidence=0.98)]


class _FakeFaceTracker:
    def update(self, detections, now: float):
        if not detections:
            return None
        return TrackedFace(
            track_id=1,
            bbox=detections[0].bbox,
            confidence=detections[0].confidence,
            last_seen_at=now,
        )


class _FakeAntiSpoofModel:
    backend_name = "fake_temporal"

    def __init__(self):
        self.calls = 0

    def predict(self, frames, tracked_face):
        self.calls += 1
        return AntiSpoofResult(
            label=AntiSpoofLabel.REAL,
            confidence=0.91,
            model_name="fake_temporal_v1",
        )


def _frame(frame_id: int, timestamp: float) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        timestamp=timestamp,
        frame=np.zeros((64, 64, 3), dtype=np.uint8),
    )


class TestAccessMvpPipeline:
    def test_reuses_recent_anti_spoof_result_between_close_frames(self):
        settings = get_settings().model_copy(deep=True)
        settings.runtime.require_real_backends = False
        settings.anti_spoof.min_frames_for_inference = 1
        settings.anti_spoof.inference_interval_frames = 3
        settings.anti_spoof.max_cached_result_age_seconds = 10.0
        settings.anti_spoof.rerun_iou_threshold = 0.8
        pipeline = AccessControlPipeline(settings)
        pipeline._yolo_detector = _FakeYoloDetector()
        pipeline._face_detector = _FakeFaceDetector()
        pipeline._face_tracker = _FakeFaceTracker()
        pipeline._anti_spoof_model = _FakeAntiSpoofModel()

        now = time.monotonic()
        pipeline._process_packet(_frame(1, now))
        pipeline._process_packet(_frame(2, now + 0.02))

        status = pipeline.get_status()

        assert pipeline._anti_spoof_model.calls == 1
        assert status["performance"]["anti_spoof_cache_hits"] == 1
        assert status["performance"]["anti_spoof_inferences"] == 1
