from __future__ import annotations

import pytest

from app.domain.exceptions import AssetValidationError
from app.infrastructure.assets import validate_assets
from app.infrastructure.config import get_settings


class TestAssetValidation:
    def test_validate_assets_success(self):
        settings = get_settings()
        manifest = validate_assets(settings)
        assert manifest.mediapipe_model_path.endswith("face_landmarker.task")
        assert len(manifest.anti_spoof_model_names) >= 1

    def test_validate_assets_missing_mediapipe(self):
        settings = get_settings().model_copy(deep=True)
        settings.detector.mediapipe.model_path = "resources/missing_face_landmarker.task"

        with pytest.raises(AssetValidationError):
            validate_assets(settings)
