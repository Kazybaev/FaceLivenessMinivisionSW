"""Integration tests for API routes using TestClient."""
from __future__ import annotations

import io
import numpy as np
import cv2
import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from app.config import get_settings as get_access_settings
from app.main import create_app
from app.domain.entities import LivenessResult
from app.domain.enums import LivenessVerdict, DetectionMethod


@pytest.fixture
def client():
    settings = get_access_settings().model_copy(deep=True)
    settings.runtime.autostart = False
    settings.runtime.auto_enable_local_backends = False
    settings.runtime.require_real_backends = False
    settings.active_liveness.enabled = False
    app = create_app(settings)
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


class TestHealthEndpoints:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_readiness(self, client):
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert "models_loaded" in data


class TestSessionEndpoints:
    def test_create_and_delete_session(self, client):
        # Create
        response = client.post("/api/v1/liveness/session")
        assert response.status_code == 200
        session_id = response.json()["session_id"]
        assert session_id

        # Delete
        response = client.delete(f"/api/v1/liveness/session/{session_id}")
        assert response.status_code == 200

    def test_delete_nonexistent_session(self, client):
        response = client.delete("/api/v1/liveness/session/nonexistent")
        assert response.status_code == 404

    def test_frame_nonexistent_session(self, client, sample_image_bytes):
        response = client.post(
            "/api/v1/liveness/session/nonexistent/frame",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")},
        )
        assert response.status_code == 404
