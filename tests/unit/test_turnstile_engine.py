from __future__ import annotations

import json
import time

import numpy as np

from app.domain.entities import AssetManifest, BBox, FaceRegion
from app.domain.enums import ControllerVerdict, TurnstileState
from app.infrastructure.config import CameraConfig, TurnstileConfig, TurnstileThresholdsConfig
from app.turnstile.engine import TurnstileDecisionEngine


def _build_face_region(frame: np.ndarray) -> FaceRegion:
    bbox = BBox(x=140, y=90, width=180, height=210)
    face = frame[bbox.y:bbox.y2, bbox.x:bbox.x2].copy()
    landmarks = np.zeros((478, 2), dtype=np.float32)
    center_x = bbox.x + bbox.width / 2
    center_y = bbox.y + bbox.height / 2
    landmarks[:] = [center_x, center_y]
    landmarks[1] = [center_x, center_y]
    landmarks[33] = [bbox.x + 55, bbox.y + 70]
    landmarks[133] = [bbox.x + 75, bbox.y + 72]
    landmarks[263] = [bbox.x + 125, bbox.y + 70]
    landmarks[362] = [bbox.x + 105, bbox.y + 72]
    return FaceRegion(bbox=bbox, image=face, landmarks=landmarks)


class FakeDetector:
    def __init__(self, face: FaceRegion | None):
        self._face = face

    def detect_video_frame(self, image: np.ndarray, timestamp_ms: int) -> FaceRegion | None:
        return self._face


class FakeDLAnalyzer:
    def __init__(self, real_prob: float = 0.9, fake_prob: float = 0.1, smoothed: float | None = None):
        self._real_prob = real_prob
        self._fake_prob = fake_prob
        self._smoothed = real_prob if smoothed is None else smoothed
        self.reset_calls = 0

    def predict_face(self, face: np.ndarray, smooth: bool = False) -> dict[str, float]:
        return {
            "real_prob": self._real_prob,
            "fake_prob": self._fake_prob,
            "smoothed": self._smoothed,
        }

    def reset(self) -> None:
        self.reset_calls += 1


def _build_engine(
    tmp_path,
    detector,
    dl_analyzer,
    webhook_url: str | None = None,
) -> TurnstileDecisionEngine:
    thresholds = TurnstileThresholdsConfig(dl_interval_frames=1)
    cfg = TurnstileConfig(
        decision_window_ms=10,
        cooldown_ms=100,
        min_good_frames=1,
        webhook_url=webhook_url,
        audit_log_dir=str(tmp_path),
        thresholds=thresholds,
    )
    assets = AssetManifest(
        mediapipe_model_path="resources/face_landmarker.task",
        retinaface_prototxt="resources/detection_model/deploy.prototxt",
        retinaface_caffemodel="resources/detection_model/Widerface-RetinaFace.caffemodel",
        anti_spoof_model_dir="resources/anti_spoof_models",
        anti_spoof_model_names=["model_a.pth"],
    )
    return TurnstileDecisionEngine(
        camera_config=CameraConfig(id="camera-test", index=0, width=640, height=480),
        turnstile_config=cfg,
        detector=detector,
        dl_analyzer=dl_analyzer,
        assets=assets,
    )


class TestTurnstileDecisionEngine:
    def test_no_face_returns_no_face(self, sample_image, tmp_path):
        engine = _build_engine(tmp_path, FakeDetector(None), FakeDLAnalyzer())
        decision = engine.process_frame(sample_image)

        assert decision.state == TurnstileState.NO_FACE
        assert decision.controller_verdict is None
        assert decision.reason_codes == ["no_face"]

    def test_access_granted_when_scores_are_high(self, sample_image, tmp_path):
        face = _build_face_region(sample_image)
        engine = _build_engine(tmp_path, FakeDetector(face), FakeDLAnalyzer(real_prob=0.92, fake_prob=0.08))
        engine._state.started_at = time.time() - 1.0
        engine._state.texture_hist.extend([0.60] * 8)
        engine._state.edge_hist.extend([120.0] * 8)
        engine._state.motion_hist.extend([3.0] * 8)
        engine._state.flow_hist.extend([0.35] * 8)
        engine._state.parallax_hist.extend([0.45, 0.47, 0.43, 0.48, 0.44, 0.46])

        decision = engine.process_frame(sample_image)

        assert decision.state == TurnstileState.ACCESS_GRANTED
        assert decision.controller_verdict == ControllerVerdict.ACCESS_GRANTED
        assert decision.is_final is True
        assert engine.get_latest_decision() is not None

    def test_access_denied_on_high_fake_override(self, sample_image, tmp_path):
        face = _build_face_region(sample_image)
        engine = _build_engine(tmp_path, FakeDetector(face), FakeDLAnalyzer(real_prob=0.04, fake_prob=0.96, smoothed=0.04))
        engine._state.started_at = time.time() - 1.0
        engine._state.texture_hist.extend([0.50] * 6)
        engine._state.edge_hist.extend([140.0] * 6)
        engine._state.motion_hist.extend([2.5] * 6)
        engine._state.flow_hist.extend([0.40] * 6)
        engine._state.parallax_hist.extend([0.40, 0.41, 0.39, 0.42])

        decision = engine.process_frame(sample_image)

        assert decision.state == TurnstileState.ACCESS_DENIED
        assert decision.controller_verdict == ControllerVerdict.ACCESS_DENIED
        assert decision.is_final is True

    def test_final_decision_writes_audit_log(self, sample_image, tmp_path):
        face = _build_face_region(sample_image)
        engine = _build_engine(tmp_path, FakeDetector(face), FakeDLAnalyzer(real_prob=0.92, fake_prob=0.08))
        engine._state.started_at = time.time() - 1.0
        engine._state.texture_hist.extend([0.60] * 8)
        engine._state.edge_hist.extend([120.0] * 8)
        engine._state.motion_hist.extend([3.0] * 8)
        engine._state.flow_hist.extend([0.35] * 8)
        engine._state.parallax_hist.extend([0.45, 0.47, 0.43, 0.48, 0.44, 0.46])

        decision = engine.process_frame(sample_image)
        audit_files = list(tmp_path.glob("*.jsonl"))

        assert decision.is_final is True
        assert len(audit_files) == 1
        payload = json.loads(audit_files[0].read_text(encoding="utf-8").strip())
        assert payload["session_id"] == decision.session_id
        assert payload["verdict"] == "ACCESS_GRANTED"

    def test_webhook_payload_contains_expected_fields(self, sample_image, tmp_path, monkeypatch):
        import app.turnstile.engine as engine_module

        sent: dict[str, object] = {}

        class DummyResponse:
            status = 202

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class ImmediateThread:
            def __init__(self, target=None, daemon=None):
                self._target = target

            def start(self):
                if self._target is not None:
                    self._target()

        def fake_urlopen(request, timeout=0):
            sent["url"] = request.full_url
            sent["payload"] = json.loads(request.data.decode("utf-8"))
            return DummyResponse()

        monkeypatch.setattr(engine_module.threading, "Thread", ImmediateThread)
        monkeypatch.setattr(engine_module.urllib.request, "urlopen", fake_urlopen)

        face = _build_face_region(sample_image)
        engine = _build_engine(
            tmp_path,
            FakeDetector(face),
            FakeDLAnalyzer(real_prob=0.92, fake_prob=0.08),
            webhook_url="http://127.0.0.1/webhook",
        )
        engine._state.started_at = time.time() - 1.0
        engine._state.texture_hist.extend([0.60] * 8)
        engine._state.edge_hist.extend([120.0] * 8)
        engine._state.motion_hist.extend([3.0] * 8)
        engine._state.flow_hist.extend([0.35] * 8)
        engine._state.parallax_hist.extend([0.45, 0.47, 0.43, 0.48, 0.44, 0.46])

        engine.process_frame(sample_image)

        assert sent["url"] == "http://127.0.0.1/webhook"
        assert sent["payload"]["verdict"] == "ACCESS_GRANTED"
        assert sent["payload"]["camera_id"] == "camera-test"
        assert "model_versions" in sent["payload"]
