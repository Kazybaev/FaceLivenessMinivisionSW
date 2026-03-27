from __future__ import annotations

from fastapi.testclient import TestClient

from app.domain.entities import TurnstileDecision
from app.domain.enums import ControllerVerdict, TurnstileState
from app.infrastructure.api.dependencies import set_container
from app.main import create_app


class _FakeTurnstileEngine:
    def __init__(self, decision: TurnstileDecision):
        self._decision = decision
        self.reset_called = False

    def get_latest_decision(self):
        return self._decision

    def reset(self):
        self.reset_called = True


class _FakeContainer:
    def __init__(self, decision: TurnstileDecision):
        self.turnstile_engine = _FakeTurnstileEngine(decision)
        self.readiness = {"status": "ready", "models_loaded": True, "asset_error": None}


def _build_decision() -> TurnstileDecision:
    return TurnstileDecision(
        session_id="session-1",
        state=TurnstileState.ACCESS_GRANTED,
        reason="Live person confirmed",
        confidence=0.93,
        reason_codes=["live_confirmed"],
        controller_verdict=ControllerVerdict.ACCESS_GRANTED,
        timestamp_utc="2026-03-27T00:00:00+00:00",
        latency_ms=420.0,
        camera_id="camera-test",
        device_id="device-test",
        live_score=0.91,
        heuristic_score=0.87,
        deep_learning_score=0.96,
        is_final=True,
        details={"model_versions": {"mediapipe": "face_landmarker.task", "anti_spoof_models": ["model_a.pth"]}},
    )


class TestTurnstileControlPlane:
    def setup_method(self):
        self._container = _FakeContainer(_build_decision())
        set_container(self._container)
        self._client = TestClient(create_app())

    def teardown_method(self):
        set_container(None)

    def test_ready_alias(self):
        response = self._client.get("/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"

    def test_latest_decision(self):
        response = self._client.get("/decision/latest")
        assert response.status_code == 200
        payload = response.json()["latest_decision"]
        assert payload["verdict"] == "ACCESS_GRANTED"
        assert payload["state"] == "ACCESS_GRANTED"
        assert payload["camera_id"] == "camera-test"

    def test_turnstile_reset(self):
        response = self._client.post("/api/v1/turnstile/reset")
        assert response.status_code == 200
        assert self._container.turnstile_engine.reset_called is True
