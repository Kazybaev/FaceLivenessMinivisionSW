from __future__ import annotations

from app.config import AntiSpoofSettings, SessionSettings
from app.core.models import (
    AccessSession,
    AntiSpoofLabel,
    AntiSpoofResult,
    DecisionRecord,
    DecisionVerdict,
    ObjectDetection,
    SessionState,
)
from app.utils.timers import utc_iso


def make_access_decision(
    session: AccessSession,
    anti_spoof_result: AntiSpoofResult | None,
    *,
    now: float,
    suspicious_object_types: list[str],
    session_settings: SessionSettings,
    anti_spoof_settings: AntiSpoofSettings,
) -> DecisionRecord:
    # Suspicious objects are a hard rule and can never be overridden by anti-spoof scores.
    if session.blocked_by_suspicious_object or suspicious_object_types:
        labels = session.suspicious_object_types or suspicious_object_types
        decision_state = SessionState.BLOCKED if session.state == SessionState.BLOCKED else SessionState.SUSPICIOUS_OBJECT_DETECTED
        return DecisionRecord(
            session_id=session.session_id,
            verdict=DecisionVerdict.DENY,
            state=decision_state,
            allow_face_recognition=False,
            confidence=0.99,
            reason=f"Suspicious object detected: {', '.join(labels)}. Remove it and try again",
            timestamp=utc_iso(),
            cooldown_until=now + session_settings.suspicious_cooldown_seconds,
            details={"suspicious_object_types": labels},
        )

    if session.state == SessionState.COOLDOWN and session.cooldown_until is not None and now < session.cooldown_until:
        return DecisionRecord(
            session_id=session.session_id,
            verdict=DecisionVerdict.DENY,
            state=SessionState.COOLDOWN,
            allow_face_recognition=False,
            confidence=1.0,
            reason="Session is in cooldown",
            timestamp=utc_iso(),
            cooldown_until=session.cooldown_until,
        )

    if anti_spoof_result is None:
        return DecisionRecord(
            session_id=session.session_id,
            verdict=DecisionVerdict.PENDING,
            state=SessionState.OBSERVING,
            allow_face_recognition=False,
            confidence=0.0,
            reason="Collecting frame buffer for anti-spoofing",
            timestamp=utc_iso(),
        )

    if anti_spoof_result.label == AntiSpoofLabel.SPOOF:
        return DecisionRecord(
            session_id=session.session_id,
            verdict=DecisionVerdict.DENY,
            state=SessionState.SPOOF_DETECTED,
            allow_face_recognition=False,
            confidence=anti_spoof_result.confidence,
            reason="Spoof detected. Try again",
            timestamp=utc_iso(),
            cooldown_until=now + session_settings.spoof_cooldown_seconds,
            details=anti_spoof_result.details,
        )

    if anti_spoof_result.label == AntiSpoofLabel.UNCERTAIN:
        return DecisionRecord(
            session_id=session.session_id,
            verdict=DecisionVerdict.DENY,
            state=SessionState.SPOOF_DETECTED,
            allow_face_recognition=False,
            confidence=anti_spoof_result.confidence,
            reason="Face not confirmed as real. Try again",
            timestamp=utc_iso(),
            cooldown_until=now + session_settings.spoof_cooldown_seconds,
            details=anti_spoof_result.details,
        )

    if anti_spoof_result.confidence < anti_spoof_settings.real_confidence_threshold:
        return DecisionRecord(
            session_id=session.session_id,
            verdict=DecisionVerdict.DENY,
            state=SessionState.SPOOF_DETECTED,
            allow_face_recognition=False,
            confidence=anti_spoof_result.confidence,
            reason="Real face confidence is too low. Try again",
            timestamp=utc_iso(),
            cooldown_until=now + session_settings.spoof_cooldown_seconds,
            details=anti_spoof_result.details,
        )

    return DecisionRecord(
        session_id=session.session_id,
        verdict=DecisionVerdict.ALLOW,
        state=SessionState.REAL_DETECTED,
        allow_face_recognition=True,
        confidence=anti_spoof_result.confidence,
        reason="Real person confirmed",
        timestamp=utc_iso(),
        details=anti_spoof_result.details,
    )


class DecisionEngine:
    """Final allow/deny policy for the current access session."""

    def __init__(self, settings: SessionSettings, anti_spoof_settings: AntiSpoofSettings | None = None):
        self._settings = settings
        self._anti_spoof_settings = anti_spoof_settings or AntiSpoofSettings()

    def evaluate(
        self,
        session: AccessSession,
        suspicious_objects: list[ObjectDetection],
        anti_spoof_result: AntiSpoofResult | None,
        now: float,
    ) -> DecisionRecord | None:
        suspicious_object_types = sorted(
            {obj.label for obj in suspicious_objects} | set(session.suspicious_object_types)
        )
        return make_access_decision(
            session,
            anti_spoof_result,
            now=now,
            suspicious_object_types=suspicious_object_types,
            session_settings=self._settings,
            anti_spoof_settings=self._anti_spoof_settings,
        )
