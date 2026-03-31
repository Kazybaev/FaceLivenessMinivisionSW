from __future__ import annotations

import time

import numpy as np

from app.config import AntiSpoofSettings, SessionSettings
from app.core.models import (
    BoundingBox,
    DecisionRecord,
    DecisionVerdict,
    FramePacket,
    ObjectDetection,
    SessionState,
    TrackedFace,
)
from app.core.session_manager import SessionManager
from app.services.event_logger import EventLogger


def _frame(frame_id: int, timestamp: float) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        timestamp=timestamp,
        frame=np.zeros((32, 32, 3), dtype=np.uint8),
    )


def _face(timestamp: float, track_id: int = 1) -> TrackedFace:
    return TrackedFace(
        track_id=track_id,
        bbox=BoundingBox(x=5, y=5, width=20, height=20),
        confidence=0.95,
        last_seen_at=timestamp,
    )


class TestAccessMvpSessionManager:
    def test_phone_denies_while_present_and_clears_after_clean_frames(self):
        now = time.monotonic()
        manager = SessionManager(
            SessionSettings(state_display_seconds=0.0, suspicious_hold_seconds=0.05, sticky_suspicious_block=False),
            AntiSpoofSettings(),
            EventLogger(),
        )

        manager.on_frame(_frame(1, now), _face(now))
        detections = [
            ObjectDetection(
                label="phone",
                confidence=0.93,
                bbox=BoundingBox(x=0, y=0, width=12, height=12),
            )
        ]
        manager.update_suspicious_state(detections, now)
        manager.apply_decision(
            DecisionRecord(
                session_id=manager.current_session.session_id,
                verdict=DecisionVerdict.DENY,
                state=SessionState.SUSPICIOUS_OBJECT_DETECTED,
                allow_face_recognition=False,
                confidence=0.99,
                reason="Suspicious object detected: phone",
                timestamp="2026-03-27T00:00:00+00:00",
                cooldown_until=now,
            ),
            now,
        )

        blocked_session = manager.current_session
        assert blocked_session.state == SessionState.SUSPICIOUS_OBJECT_DETECTED
        assert blocked_session.blocked_by_suspicious_object is True
        assert blocked_session.suspicious_object_types == ["phone"]

        manager.on_frame(_frame(2, now + 0.1), _face(now + 0.1))
        manager.update_suspicious_state([], now + 0.1)
        session = manager.current_session

        assert session.state == SessionState.OBSERVING
        assert session.blocked_by_suspicious_object is False
        assert session.suspicious_object_types == []
        assert session.blocked_reason is None
        assert len(session.frame_buffer) == 0

    def test_track_change_resets_to_a_new_session(self):
        now = time.monotonic()
        manager = SessionManager(
            SessionSettings(),
            AntiSpoofSettings(),
            EventLogger(),
        )

        first = manager.on_frame(_frame(1, now), _face(now, track_id=1))
        first_session_id = first.session_id

        second = manager.on_frame(_frame(2, now + 0.1), _face(now + 0.1, track_id=2))

        assert second.session_id != first_session_id
        assert second.state == SessionState.OBSERVING
        assert second.track_id == 2
        assert len(second.frame_buffer) == 1

    def test_allowed_session_resets_after_face_disappears(self):
        now = time.monotonic()
        manager = SessionManager(
            SessionSettings(no_face_reset_seconds=0.05),
            AntiSpoofSettings(),
            EventLogger(),
        )

        session = manager.on_frame(_frame(1, now), _face(now))
        manager.apply_decision(
            DecisionRecord(
                session_id=session.session_id,
                verdict=DecisionVerdict.ALLOW,
                state=SessionState.REAL_DETECTED,
                allow_face_recognition=True,
                confidence=0.95,
                reason="Real person confirmed",
                timestamp="2026-03-31T00:00:00+00:00",
            ),
            now,
        )

        reset_session = manager.on_frame(_frame(2, now + 0.1), None)

        assert reset_session.state == SessionState.IDLE
        assert reset_session.track_id is None
        assert reset_session.session_id != session.session_id
