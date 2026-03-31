from __future__ import annotations

from app.config import AntiSpoofSettings, SessionSettings
from app.core.models import (
    AccessSession,
    ActiveLivenessResult,
    ActiveLivenessVerdict,
    AntiSpoofLabel,
    AntiSpoofResult,
    DecisionRecord,
    DecisionVerdict,
    ObjectDetection,
    SessionState,
)
from app.utils.timers import utc_iso


def _anti_spoof_metrics(result: AntiSpoofResult) -> tuple[float, float, float]:
    details = result.details or {}
    combined_real = float(details.get("combined_real", result.confidence))
    dl_real = float(details.get("dl_real", combined_real))
    dl_fake = float(details.get("dl_fake", 0.0))
    return combined_real, dl_real, dl_fake


def _active_assisted_real_allowed(
    anti_spoof_result: AntiSpoofResult,
    active_liveness_result: ActiveLivenessResult | None,
    anti_spoof_settings: AntiSpoofSettings,
) -> bool:
    if active_liveness_result is None or active_liveness_result.verdict != ActiveLivenessVerdict.REAL:
        return False
    combined_real, dl_real, dl_fake = _anti_spoof_metrics(anti_spoof_result)
    evidence = max(anti_spoof_result.confidence, combined_real, dl_real)
    return (
        evidence >= anti_spoof_settings.active_assisted_real_threshold
        and dl_fake <= anti_spoof_settings.active_assisted_max_fake_probability
    )


def _strong_fake_signal(
    anti_spoof_result: AntiSpoofResult,
    anti_spoof_settings: AntiSpoofSettings,
) -> bool:
    combined_real, _, dl_fake = _anti_spoof_metrics(anti_spoof_result)
    return (
        anti_spoof_result.label == AntiSpoofLabel.SPOOF
        or dl_fake >= anti_spoof_settings.fake_threshold
        or combined_real <= 0.45
    )


def make_access_decision(
    session: AccessSession,
    anti_spoof_result: AntiSpoofResult | None,
    active_liveness_result: ActiveLivenessResult | None,
    *,
    now: float,
    suspicious_object_types: list[str],
    session_settings: SessionSettings,
    anti_spoof_settings: AntiSpoofSettings,
    active_liveness_required: bool,
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

    if active_liveness_required:
        if active_liveness_result is None:
            active_liveness_result = ActiveLivenessResult(
                verdict=ActiveLivenessVerdict.PENDING,
                confidence=0.0,
                reason="Blink or turn your head slightly",
            )

        if active_liveness_result.verdict == ActiveLivenessVerdict.UNAVAILABLE:
            return DecisionRecord(
                session_id=session.session_id,
                verdict=DecisionVerdict.DENY,
                state=SessionState.BLOCKED,
                allow_face_recognition=False,
                confidence=1.0,
                reason=active_liveness_result.reason,
                timestamp=utc_iso(),
                details=active_liveness_result.details,
            )

        if active_liveness_result.verdict in {ActiveLivenessVerdict.SPOOF, ActiveLivenessVerdict.FAILED}:
            return DecisionRecord(
                session_id=session.session_id,
                verdict=DecisionVerdict.DENY,
                state=SessionState.SPOOF_DETECTED,
                allow_face_recognition=False,
                confidence=active_liveness_result.confidence,
                reason=active_liveness_result.reason,
                timestamp=utc_iso(),
                cooldown_until=now + session_settings.spoof_cooldown_seconds,
                details=active_liveness_result.details,
            )

    if anti_spoof_result is None:
        return DecisionRecord(
            session_id=session.session_id,
            verdict=DecisionVerdict.PENDING,
            state=SessionState.OBSERVING,
            allow_face_recognition=False,
            confidence=active_liveness_result.confidence if active_liveness_result is not None else 0.0,
            reason="Collecting frame buffer for anti-spoofing",
            timestamp=utc_iso(),
            details={} if active_liveness_result is None else active_liveness_result.details,
        )

    if anti_spoof_result.label == AntiSpoofLabel.SPOOF:
        if _active_assisted_real_allowed(anti_spoof_result, active_liveness_result, anti_spoof_settings):
            return DecisionRecord(
                session_id=session.session_id,
                verdict=DecisionVerdict.PENDING,
                state=SessionState.OBSERVING,
                allow_face_recognition=False,
                confidence=max(
                    anti_spoof_result.confidence,
                    active_liveness_result.confidence if active_liveness_result is not None else 0.0,
                ),
                reason="Hold still, checking for spoof signals again",
                timestamp=utc_iso(),
                details=anti_spoof_result.details,
            )
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
        if _active_assisted_real_allowed(anti_spoof_result, active_liveness_result, anti_spoof_settings):
            return DecisionRecord(
                session_id=session.session_id,
                verdict=DecisionVerdict.ALLOW,
                state=SessionState.REAL_DETECTED,
                allow_face_recognition=True,
                confidence=max(
                    anti_spoof_result.confidence,
                    active_liveness_result.confidence if active_liveness_result is not None else 0.0,
                ),
                reason="Real person confirmed with active live response",
                timestamp=utc_iso(),
                details={
                    **anti_spoof_result.details,
                    "active_liveness": None if active_liveness_result is None else active_liveness_result.details,
                },
            )
        if _strong_fake_signal(anti_spoof_result, anti_spoof_settings):
            return DecisionRecord(
                session_id=session.session_id,
                verdict=DecisionVerdict.DENY,
                state=SessionState.SPOOF_DETECTED,
                allow_face_recognition=False,
                confidence=anti_spoof_result.confidence,
                reason="Spoof signals are too strong. Try again",
                timestamp=utc_iso(),
                cooldown_until=now + session_settings.spoof_cooldown_seconds,
                details=anti_spoof_result.details,
            )
        return DecisionRecord(
            session_id=session.session_id,
            verdict=DecisionVerdict.PENDING,
            state=SessionState.OBSERVING,
            allow_face_recognition=False,
            confidence=anti_spoof_result.confidence,
            reason="Analyzing live face. Hold still or blink",
            timestamp=utc_iso(),
            details=anti_spoof_result.details,
        )

    if anti_spoof_result.confidence < anti_spoof_settings.real_confidence_threshold:
        if _active_assisted_real_allowed(anti_spoof_result, active_liveness_result, anti_spoof_settings):
            return DecisionRecord(
                session_id=session.session_id,
                verdict=DecisionVerdict.ALLOW,
                state=SessionState.REAL_DETECTED,
                allow_face_recognition=True,
                confidence=max(
                    anti_spoof_result.confidence,
                    active_liveness_result.confidence if active_liveness_result is not None else 0.0,
                ),
                reason="Real person confirmed with active live response",
                timestamp=utc_iso(),
                details={
                    **anti_spoof_result.details,
                    "active_liveness": None if active_liveness_result is None else active_liveness_result.details,
                },
            )
        return DecisionRecord(
            session_id=session.session_id,
            verdict=DecisionVerdict.PENDING,
            state=SessionState.OBSERVING,
            allow_face_recognition=False,
            confidence=anti_spoof_result.confidence,
            reason="Need a little more live evidence",
            timestamp=utc_iso(),
            details=anti_spoof_result.details,
        )

    if active_liveness_required and active_liveness_result.verdict != ActiveLivenessVerdict.REAL:
        return DecisionRecord(
            session_id=session.session_id,
            verdict=DecisionVerdict.PENDING,
            state=SessionState.OBSERVING,
            allow_face_recognition=False,
            confidence=active_liveness_result.confidence,
            reason=active_liveness_result.reason,
            timestamp=utc_iso(),
            details=active_liveness_result.details,
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

    def __init__(
        self,
        settings: SessionSettings,
        anti_spoof_settings: AntiSpoofSettings | None = None,
        *,
        active_liveness_required: bool = True,
    ):
        self._settings = settings
        self._anti_spoof_settings = anti_spoof_settings or AntiSpoofSettings()
        self._active_liveness_required = active_liveness_required

    def evaluate(
        self,
        session: AccessSession,
        suspicious_objects: list[ObjectDetection],
        anti_spoof_result: AntiSpoofResult | None,
        active_liveness_result: ActiveLivenessResult | None,
        now: float,
    ) -> DecisionRecord | None:
        suspicious_object_types = sorted(
            {obj.label for obj in suspicious_objects} | set(session.suspicious_object_types)
        )
        return make_access_decision(
            session,
            anti_spoof_result,
            active_liveness_result,
            now=now,
            suspicious_object_types=suspicious_object_types,
            session_settings=self._settings,
            anti_spoof_settings=self._anti_spoof_settings,
            active_liveness_required=self._active_liveness_required,
        )
