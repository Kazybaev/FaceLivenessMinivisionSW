"""Tests for deep learning analyzer with mock models."""
import numpy as np
import pytest
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from app.adapters.analyzers.deep_learning_analyzer import DeepLearningAnalyzer
from app.infrastructure.config import DeepLearningConfig
from app.domain.entities import BBox, FaceRegion, ModelInfo
from app.domain.enums import LivenessVerdict


class FakeModel(nn.Module):
    def __init__(self, output: list[float]):
        super().__init__()
        self._output = torch.tensor([output])

    def forward(self, x):
        return self._output


class FakeRepo:
    def __init__(self, models: dict[str, FakeModel]):
        self._models = models

    def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(name=name, path=f"/fake/{name}", h_input=80, w_input=80,
                      model_type="MiniFASNetV2", scale=2.7)
            for name in self._models
        ]

    def load_model(self, info: ModelInfo) -> nn.Module:
        return self._models[info.name]


class TestDeepLearningAnalyzer:
    def test_predict_real(self):
        # Model outputs high real prob: softmax([0, 5, 0]) ≈ [0.007, 0.986, 0.007]
        repo = FakeRepo({"model1.pth": FakeModel([0.0, 5.0, 0.0])})
        config = DeepLearningConfig(model_dir="/fake", confidence_threshold=0.6)
        analyzer = DeepLearningAnalyzer(config, repo)

        face = FaceRegion(
            bbox=BBox(x=0, y=0, width=80, height=80),
            image=np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8),
        )
        result = analyzer.analyze(np.zeros((480, 640, 3), dtype=np.uint8), face)
        assert result.verdict == LivenessVerdict.REAL
        assert result.confidence > 0.6

    def test_predict_fake(self):
        # Model outputs high fake prob: softmax([5, 0, 0]) ≈ [0.986, 0.007, 0.007]
        repo = FakeRepo({"model1.pth": FakeModel([5.0, 0.0, 0.0])})
        config = DeepLearningConfig(model_dir="/fake", confidence_threshold=0.6)
        analyzer = DeepLearningAnalyzer(config, repo)

        face = FaceRegion(
            bbox=BBox(x=0, y=0, width=80, height=80),
            image=np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8),
        )
        result = analyzer.analyze(np.zeros((480, 640, 3), dtype=np.uint8), face)
        assert result.verdict == LivenessVerdict.FAKE

    def test_ensemble_averaging(self):
        # Two models: one says real, one says fake — should be uncertain
        repo = FakeRepo({
            "model1.pth": FakeModel([5.0, 0.0, 0.0]),
            "model2.pth": FakeModel([0.0, 5.0, 0.0]),
        })
        config = DeepLearningConfig(model_dir="/fake", confidence_threshold=0.6)
        analyzer = DeepLearningAnalyzer(config, repo)

        face = FaceRegion(
            bbox=BBox(x=0, y=0, width=80, height=80),
            image=np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8),
        )
        result = analyzer.analyze(np.zeros((480, 640, 3), dtype=np.uint8), face)
        assert result.confidence < 0.7

    def test_score_smoothing(self):
        repo = FakeRepo({"model1.pth": FakeModel([0.0, 5.0, 0.0])})
        config = DeepLearningConfig(model_dir="/fake", smoothing_frames=3)
        analyzer = DeepLearningAnalyzer(config, repo)

        face = FaceRegion(
            bbox=BBox(x=0, y=0, width=80, height=80),
            image=np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8),
        )
        img = np.zeros((480, 640, 3), dtype=np.uint8)

        # Process multiple frames
        for _ in range(3):
            result = analyzer.process_frame(img, face)
        assert len(analyzer._score_history) == 3

    def test_reset(self):
        repo = FakeRepo({"model1.pth": FakeModel([0.0, 5.0, 0.0])})
        config = DeepLearningConfig(model_dir="/fake")
        analyzer = DeepLearningAnalyzer(config, repo)

        face = FaceRegion(
            bbox=BBox(x=0, y=0, width=80, height=80),
            image=np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8),
        )
        analyzer.process_frame(np.zeros((480, 640, 3), dtype=np.uint8), face)
        analyzer.reset()
        assert len(analyzer._score_history) == 0
