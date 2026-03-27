from __future__ import annotations

from collections import deque
import uuid

from app.config import AntiSpoofSettings, SessionSettings
from app.core.models import (
    AccessSession,
    AntiSpoofResult,
    DecisionRecord,
    FramePacket,
    ObjectDetection,
    SessionState,
    TrackedFace,
)
from app.services.event_logger import EventLogger
from app.utils.timers import remaining_seconds


class SessionManager:
    """Owns the active access-check session and its frame buffer."""

    def __init__(
        self,
        session_settings: SessionSettings,
        anti_spoof_settings: AntiSpoofSettings,
        event_logger: EventLogger,
    ):
        self._session_settings = session_settings
        self._anti_spoof_settings = anti_spoof_settings
        self._event_logger = event_logger
        self._session = self._new_session()

    @property
    def current_session(self) -> AccessSession:
        return self._session

    def reset(self, now: float, reason: str = "manual_reset") -> AccessSession:
        self._session = self._new_session()
        self._event_logger.log(
            "session_reset",
            f"Session reset: {reason}",
            session_id=self._session.session_id,
            payload={"reason": reason},
        )
        self._session.started_at = now
        return self._session

    def on_frame(self, frame_packet: FramePacket, tracked_face: TrackedFace | None) -> AccessSession:
        now = frame_packet.timestamp
        session = self._session

        if session.state in {SessionState.COOLDOWN, SessionState.BLOCKED} and session.cooldown_until is not None:
            if now >= session.cooldown_until:
                session = self.reset(now, reason="cooldown_expired")
            else:
                return session

        if session.state == SessionState.SPOOF_DETECTED:
            if session.state_deadline is not None and now >= session.state_deadline:
                session.state = SessionState.COOLDOWN
                return session

        if tracked_face is None:
            if session.state == SessionState.ALLOWED:
                return session
            if session.last_seen_at and now - session.last_seen_at > self._session_settings.no_face_reset_seconds:
                session = self.reset(now, reason="no_face_timeout")
            return session

        if session.state == SessionState.IDLE:
            session.session_id = str(uuid.uuid4())
            session.started_at = now
            session.state = SessionState.OBSERVING

        session.track_id = tracked_face.track_id
        session.last_seen_at = now
        session.frame_buffer.append(frame_packet)
        while len(session.frame_buffer) > self._anti_spoof_settings.frame_buffer_size:
            session.frame_buffer.popleft()
        return session

    def mark_suspicious(self, objects: list[ObjectDetection]) -> None:
        session = self._session
        session.blocked_by_suspicious_object = True
        session.suspicious_object_seen = True
        for obj in objects:
            if obj.label not in session.suspicious_object_types:
                session.suspicious_object_types.append(obj.label)
            session.suspicious_labels.add(obj.label)
        labels = ", ".join(session.suspicious_object_types)
        session.blocked_reason = f"Suspicious object detected: {labels}"
        session.frame_buffer.clear()
        session.last_anti_spoof_result = None

    def update_suspicious_state(self, objects: list[ObjectDetection], now: float) -> None:
        session = self._session
        if objects:
            session.last_suspicious_at = now
            self.mark_suspicious(objects)
            return

        if not session.blocked_by_suspicious_object:
            return
        if self._session_settings.sticky_suspicious_block:
            return
        if session.last_suspicious_at is None:
            self._clear_suspicious_state()
            return
        if now - session.last_suspicious_at < self._session_settings.suspicious_hold_seconds:
            return
        self._clear_suspicious_state()

    def record_anti_spoof_result(self, result: AntiSpoofResult | None) -> None:
        if result is None:
            return
        self._session.last_anti_spoof_result = result

    def apply_decision(self, decision: DecisionRecord, now: float) -> None:
        session = self._session
        session.last_decision = decision

        if decision.state == SessionState.SUSPICIOUS_OBJECT_DETECTED:
            session.state = SessionState.SUSPICIOUS_OBJECT_DETECTED
            session.blocked_reason = decision.reason
            session.cooldown_until = decision.cooldown_until
            session.state_deadline = None
            return

        if decision.state == SessionState.BLOCKED:
            session.state = SessionState.BLOCKED
            session.blocked_reason = decision.reason
            session.cooldown_until = decision.cooldown_until
            return

        if decision.state == SessionState.SPOOF_DETECTED:
            session.state = SessionState.SPOOF_DETECTED
            session.blocked_reason = decision.reason
            session.cooldown_until = decision.cooldown_until
            session.state_deadline = now + self._session_settings.state_display_seconds
            return

        if decision.state == SessionState.REAL_DETECTED:
            session.state = SessionState.ALLOWED
            session.blocked_reason = None
            return

        session.state = decision.state

    def snapshot(self, now: float) -> dict[str, object]:
        session = self._session
        return {
            "session_id": session.session_id,
            "state": session.state,
            "track_id": session.track_id,
            "blocked_by_suspicious_object": session.blocked_by_suspicious_object,
            "suspicious_object_seen": session.suspicious_object_seen,
            "suspicious_object_types": list(session.suspicious_object_types),
            "suspicious_labels": sorted(session.suspicious_labels),
            "buffered_frames": len(session.frame_buffer),
            "cooldown_remaining_seconds": remaining_seconds(session.cooldown_until, now),
            "blocked_reason": session.blocked_reason,
            "last_anti_spoof_result": session.last_anti_spoof_result,
            "last_decision": session.last_decision,
        }

    def _new_session(self) -> AccessSession:
        return AccessSession(
            session_id=str(uuid.uuid4()),
            frame_buffer=deque(maxlen=self._anti_spoof_settings.frame_buffer_size),
        )

    def _clear_suspicious_state(self) -> None:
        session = self._session
        session.blocked_by_suspicious_object = False
        session.suspicious_object_seen = False
        session.suspicious_object_types.clear()
        session.suspicious_labels.clear()
        session.blocked_reason = None
        session.cooldown_until = None
        session.last_suspicious_at = None
        session.frame_buffer.clear()
        session.last_anti_spoof_result = None
        if session.state in {SessionState.SUSPICIOUS_OBJECT_DETECTED, SessionState.BLOCKED}:
            session.state = SessionState.OBSERVING
