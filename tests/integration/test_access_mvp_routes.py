from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.models import DecisionRecord, DecisionVerdict, SessionState
from app.main import create_app


class _FakePipeline:
    def __init__(self):
        self._decision = DecisionRecord(
            session_id="session-42",
            verdict=DecisionVerdict.DENY,
            state=SessionState.SPOOF_DETECTED,
            allow_face_recognition=False,
            confidence=0.88,
            reason="Anti-spoofing returned spoof",
            timestamp="2026-03-27T00:00:00+00:00",
        )

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def get_status(self):
        return {
            "runtime_running": True,
            "camera_running": True,
            "camera_error": None,
            "yolo_backend": "mock",
            "anti_spoof_backend": "mock",
            "frame_counter": 11,
            "session": {
                "session_id": "session-42",
                "state": SessionState.OBSERVING,
                "track_id": 1,
                "blocked_by_suspicious_object": False,
                "suspicious_object_seen": False,
                "suspicious_object_types": [],
                "suspicious_labels": [],
                "buffered_frames": 5,
                "cooldown_remaining_seconds": 0.0,
                "blocked_reason": None,
                "last_anti_spoof_result": None,
                "last_decision": self._decision,
            },
        }

    def get_events(self, limit: int = 100):
        return []

    def get_current_decision(self):
        return self._decision

    def reset_session(self) -> str:
        return "session-reset"


class TestAccessMvpRoutes:
    def test_status_events_and_decision_routes(self):
        app = create_app()
        app.state.access_pipeline = _FakePipeline()
        client = TestClient(app)

        status_response = client.get("/status")
        decision_response = client.get("/current-decision")
        events_response = client.get("/events")
        reset_response = client.post("/session/reset")

        assert status_response.status_code == 200
        assert status_response.json()["runtime_running"] is True
        assert decision_response.status_code == 200
        assert decision_response.json()["decision"]["state"] == "spoof_detected"
        assert events_response.status_code == 200
        assert events_response.json()["events"] == []
        assert reset_response.status_code == 200
        assert reset_response.json()["session_id"] == "session-reset"
