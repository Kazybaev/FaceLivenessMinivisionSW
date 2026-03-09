"""Tests for heuristic analyzer: calc_ear, blink detection, movement, ratio, texture."""
import json

import numpy as np
import pytest

from app.adapters.analyzers.heuristic_analyzer import calc_ear, HeuristicAnalyzer
from app.infrastructure.config import HeuristicConfig
from app.domain.entities import BBox, FaceRegion
from app.domain.enums import LivenessVerdict


class TestCalcEar:
    def test_open_eye(self):
        pts = np.array([
            [0.0, 0.5],   # p0
            [0.25, 1.0],  # p1
            [0.75, 1.0],  # p2
            [1.0, 0.5],   # p3
            [0.75, 0.0],  # p4
            [0.25, 0.0],  # p5
        ])
        ear = calc_ear(pts)
        assert ear > 0.3

    def test_closed_eye(self):
        pts = np.array([
            [0.0, 0.5],   # p0
            [0.25, 0.55], # p1 — very close to p5
            [0.75, 0.55], # p2 — very close to p4
            [1.0, 0.5],   # p3
            [0.75, 0.45], # p4
            [0.25, 0.45], # p5
        ])
        ear = calc_ear(pts)
        assert ear < 0.22

    def test_no_division_by_zero(self):
        pts = np.array([[0, 0], [0, 1], [0, 1], [0, 0], [0, -1], [0, -1]])
        ear = calc_ear(pts)
        assert np.isfinite(ear)


class TestHeuristicAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return HeuristicAnalyzer(HeuristicConfig())

    def test_reset(self, analyzer):
        analyzer._state.blink_count = 5
        analyzer.reset()
        assert analyzer._state.blink_count == 0

    def test_no_landmarks_returns_uncertain(self, analyzer, sample_image):
        face = FaceRegion(
            bbox=BBox(x=0, y=0, width=100, height=100),
            image=sample_image[:100, :100],
            landmarks=None,
        )
        result = analyzer.process_frame(sample_image, face)
        assert result.verdict == LivenessVerdict.UNCERTAIN

    def test_process_frame_with_landmarks(self, analyzer, sample_image):
        landmarks = np.random.rand(478, 2).astype(np.float32)
        landmarks[:, 0] *= sample_image.shape[1]
        landmarks[:, 1] *= sample_image.shape[0]
        face = FaceRegion(
            bbox=BBox(x=0, y=0, width=100, height=100),
            image=sample_image[:100, :100],
            landmarks=landmarks,
        )
        result = analyzer.process_frame(sample_image, face)
        assert result.verdict in [LivenessVerdict.REAL, LivenessVerdict.FAKE]
        assert 0.0 <= result.confidence <= 1.0
        assert "checks" in result.details
        json.dumps(result.details)
