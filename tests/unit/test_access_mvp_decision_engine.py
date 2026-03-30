from __future__ import annotations

import time
import uuid

from app.config import AntiSpoofSettings, SessionSettings
from app.core.detection_policy import should_block_session
from app.core.decision_engine import DecisionEngine
from app.core.models import (
    AccessSession,
    AntiSpoofLabel,
    AntiSpoofResult,
    BoundingBox,
    DecisionVerdict,
    ObjectDetection,
    SessionState,
)


def _session() -> AccessSession:
    return AccessSession(session_id=str(uuid.uuid4()), state=SessionState.OBSERVING, started_at=time.monotonic())


class TestAccessMvpDecisionEngine:
    def test_suspicious_object_blocks_session(self):
        engine = DecisionEngine(SessionSettings(), AntiSpoofSettings())
        session = _session()
        detections = [
            ObjectDetection(
                label="phone",
                confidence=0.91,
                bbox=BoundingBox(x=10, y=10, width=100, height=200),
            )
        ]

        decision = engine.evaluate(session, detections, None, time.monotonic())

        assert decision is not None
        assert decision.verdict == DecisionVerdict.DENY
        assert decision.state == SessionState.SUSPICIOUS_OBJECT_DETECTED

    def test_real_anti_spoof_allows(self):
        engine = DecisionEngine(SessionSettings(), AntiSpoofSettings(real_confidence_threshold=0.75))
        session = _session()
        anti_spoof = AntiSpoofResult(
            label=AntiSpoofLabel.REAL,
            confidence=0.82,
            model_name="mock_temporal_v1",
        )

        decision = engine.evaluate(session, [], anti_spoof, time.monotonic())

        assert decision is not None
        assert decision.verdict == DecisionVerdict.ALLOW
        assert decision.state == SessionState.REAL_DETECTED
        assert decision.allow_face_recognition is True

    def test_uncertain_anti_spoof_denies(self):
        engine = DecisionEngine(SessionSettings(), AntiSpoofSettings())
        session = _session()
        anti_spoof = AntiSpoofResult(
            label=AntiSpoofLabel.UNCERTAIN,
            confidence=0.55,
            model_name="mock_temporal_v1",
        )

        decision = engine.evaluate(session, [], anti_spoof, time.monotonic())

        assert decision is not None
        assert decision.verdict == DecisionVerdict.DENY
        assert decision.state == SessionState.SPOOF_DETECTED

    def test_suspicious_object_hard_block_beats_real_anti_spoof(self):
        engine = DecisionEngine(SessionSettings(), AntiSpoofSettings(real_confidence_threshold=0.75))
        session = _session()
        session.blocked_by_suspicious_object = True
        session.suspicious_object_seen = True
        session.suspicious_object_types = ["phone"]
        anti_spoof = AntiSpoofResult(
            label=AntiSpoofLabel.REAL,
            confidence=0.99,
            model_name="mock_temporal_v1",
        )

        decision = engine.evaluate(session, [], anti_spoof, time.monotonic())

        assert decision is not None
        assert decision.verdict == DecisionVerdict.DENY
        assert decision.reason == "Suspicious object detected: phone. Remove it and try again"

    def test_real_is_allowed_again_after_suspicious_flag_is_cleared(self):
        engine = DecisionEngine(SessionSettings(), AntiSpoofSettings(real_confidence_threshold=0.75))
        session = _session()
        session.blocked_by_suspicious_object = False
        session.suspicious_object_seen = False
        session.suspicious_object_types = []
        anti_spoof = AntiSpoofResult(
            label=AntiSpoofLabel.REAL,
            confidence=0.91,
            model_name="mock_temporal_v1",
        )

        decision = engine.evaluate(session, [], anti_spoof, time.monotonic())

        assert decision is not None
        assert decision.verdict == DecisionVerdict.ALLOW
        assert decision.reason == "Real person confirmed"

    def test_should_block_session_normalizes_detector_labels(self):
        detections = [
            ObjectDetection(
                label="cell phone",
                confidence=0.84,
                bbox=BoundingBox(x=0, y=0, width=50, height=90),
            ),
            ObjectDetection(
                label="display",
                confidence=0.76,
                bbox=BoundingBox(x=120, y=10, width=100, height=80),
            ),
        ]

        should_block, types = should_block_session(
            detections,
            ["phone", "screen", "tablet", "printed photo"],
        )

        assert should_block is True
        assert types == ["phone", "screen"]
