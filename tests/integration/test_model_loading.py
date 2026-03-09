"""Integration tests for model loading from filesystem."""
from __future__ import annotations

import os
import pytest

from app.adapters.repositories.filesystem_model_repo import FilesystemModelRepo


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "anti_spoof_models")


@pytest.mark.skipif(
    not os.path.isdir(MODEL_DIR),
    reason="Model directory not found — skipping model loading tests",
)
class TestModelLoading:
    def test_list_models(self):
        repo = FilesystemModelRepo(MODEL_DIR)
        models = repo.list_models()
        assert len(models) >= 1
        for m in models:
            assert m.name.endswith(".pth")
            assert m.h_input > 0
            assert m.w_input > 0

    def test_load_model(self):
        repo = FilesystemModelRepo(MODEL_DIR)
        models = repo.list_models()
        assert len(models) > 0
        model = repo.load_model(models[0])
        assert model is not None

    def test_nonexistent_dir(self):
        repo = FilesystemModelRepo("/nonexistent/path")
        models = repo.list_models()
        assert len(models) == 0
