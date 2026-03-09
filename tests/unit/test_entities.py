"""Tests for domain entities."""
import numpy as np
import pytest

from app.domain.entities import BBox, FaceRegion, LivenessResult, CheckResult, HeuristicState, ModelInfo
from app.domain.enums import LivenessVerdict, DetectionMethod


class TestBBox:
    def test_x2_y2(self):
        bbox = BBox(x=10, y=20, width=100, height=50)
        assert bbox.x2 == 110
        assert bbox.y2 == 70

    def test_clamp(self):
        bbox = BBox(x=-5, y=-10, width=100, height=50)
        clamped = bbox.clamp(640, 480)
        assert clamped.x == 0
        assert clamped.y == 0
        assert clamped.width == 95
        assert clamped.height == 40

    def test_clamp_overflow(self):
        bbox = BBox(x=600, y=450, width=100, height=100)
        clamped = bbox.clamp(640, 480)
        assert clamped.x == 600
        assert clamped.y == 450
        assert clamped.x2 == 640
        assert clamped.y2 == 480


class TestHeuristicState:
    def test_reset(self):
        state = HeuristicState()
        state.blink_count = 5
        state.move_count = 10
        state.ratio_ok_count = 3
        state.ear_buf.append(0.3)
        state.texture_buf.append(100.0)
        state.reset()
        assert state.blink_count == 0
        assert state.move_count == 0
        assert state.ratio_ok_count == 0
        assert len(state.ear_buf) == 0
        assert len(state.texture_buf) == 0


class TestModelInfo:
    def test_creation(self):
        info = ModelInfo(
            name="2.7_80x80_MiniFASNetV2.pth",
            path="/models/2.7_80x80_MiniFASNetV2.pth",
            h_input=80, w_input=80,
            model_type="MiniFASNetV2",
            scale=2.7,
        )
        assert info.h_input == 80
        assert info.model_type == "MiniFASNetV2"
        assert info.scale == 2.7


class TestLivenessResult:
    def test_creation(self):
        result = LivenessResult(
            verdict=LivenessVerdict.REAL,
            confidence=0.95,
            method=DetectionMethod.DEEP_LEARNING,
        )
        assert result.verdict == LivenessVerdict.REAL
        assert result.confidence == 0.95
        assert result.details == {}
