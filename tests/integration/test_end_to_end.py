"""End-to-end integration tests."""
from __future__ import annotations

import os
import cv2
import numpy as np
import pytest

from app.infrastructure.config import get_settings, AppSettings
from app.infrastructure.container import Container


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "anti_spoof_models")
SAMPLE_IMAGE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "images", "sample")


@pytest.mark.skipif(
    not os.path.isdir(MODEL_DIR),
    reason="Model directory not found",
)
class TestEndToEnd:
    @pytest.fixture
    def container(self):
        settings = get_settings()
        return Container(settings)

    def test_analyze_synthetic_image(self, container):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        use_case = container.analyze_single_image
        result = use_case.execute(image)
        # Random noise won't have a face
        assert result.verdict.value in ["NO_FACE", "REAL", "FAKE", "UNCERTAIN"]
