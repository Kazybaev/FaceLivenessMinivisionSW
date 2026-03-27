from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
import uuid

import cv2
import numpy as np
import structlog

from app.adapters.analyzers.deep_learning_analyzer import DeepLearningAnalyzer
from app.adapters.detectors.mediapipe_detector import MediaPipeDetector
from app.domain.entities import AssetManifest, TurnstileDecision
from app.domain.enums import ControllerVerdict, TurnstileState
from app.infrastructure.assets import resolve_path
from app.infrastructure.config import CameraConfig, TurnstileConfig, TurnstileThresholdsConfig

logger = structlog.get_logger(__name__)

LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
NOSE_TIP = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _linear_band_score(x: float, low: float, high: float) -> float:
    if x <= low:
        return 0.0
    if x >= high:
        return 1.0
    return float((x - low) / max(high - low, 1e-6))


def _dominant_frequency(signal: np.ndarray, fps: float, fmin: float, fmax: float) -> tuple[float, float]:
    if len(signal) < 12 or fps <= 1.0:
        return 0.0, 0.0
    centered = signal.astype(np.float32) - np.mean(signal)
    if np.std(centered) < 1e-6:
        return 0.0, 0.0
    window = np.hanning(len(centered))
    xf = np.fft.rfft(centered * window)
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / fps)
    amps = np.abs(xf)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0, 0.0
    masked_freqs = freqs[mask]
    masked_amps = amps[mask]
    idx = int(np.argmax(masked_amps))
    return float(masked_freqs[idx]), float(masked_amps[idx] / len(centered))


def _lbp_mean(gray_roi: np.ndarray) -> float:
    if gray_roi.shape[0] < 8 or gray_roi.shape[1] < 8:
        return 0.0
    center = gray_roi[1:-1, 1:-1]
    code = np.zeros_like(center, dtype=np.uint8)
    neighbors = [
        gray_roi[:-2, :-2],
        gray_roi[:-2, 1:-1],
        gray_roi[:-2, 2:],
        gray_roi[1:-1, 2:],
        gray_roi[2:, 2:],
        gray_roi[2:, 1:-1],
        gray_roi[2:, :-2],
        gray_roi[1:-1, :-2],
    ]
    for idx, neighbor in enumerate(neighbors):
        code |= ((neighbor >= center).astype(np.uint8) << idx)
    hist = cv2.calcHist([code], [0], None, [256], [0, 256]).flatten()
    hist /= max(hist.sum(), 1.0)
    uniformity = float(np.max(hist))
    entropy = float(-np.sum(hist * np.log2(hist + 1e-8))) / 8.0
    return float(np.clip(0.55 * entropy + 0.45 * (1.0 - uniformity), 0.0, 1.0))


def _compute_flow_rigidity(prev_gray_face: np.ndarray | None, gray_face: np.ndarray) -> float | None:
    if prev_gray_face is None:
        return None
    if prev_gray_face.shape != gray_face.shape:
        gray_face = cv2.resize(gray_face, (prev_gray_face.shape[1], prev_gray_face.shape[0]))
    if prev_gray_face.shape[0] < 20 or prev_gray_face.shape[1] < 20:
        return None
    pts = cv2.goodFeaturesToTrack(prev_gray_face, maxCorners=30, qualityLevel=0.01, minDistance=5)
    if pts is None or len(pts) < 6:
        return None
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray_face, gray_face, pts, None)
    if next_pts is None or status is None:
        return None
    good = status.reshape(-1).astype(bool)
    if good.sum() < 6:
        return None
    p0 = pts[good].reshape(-1, 2)
    p1 = next_pts[good].reshape(-1, 2)
    flow = p1 - p0
    mags = np.linalg.norm(flow, axis=1)
    mean_mag = float(np.mean(mags))
    if mean_mag < 0.2:
        return 1.0
    directions = flow / (mags[:, None] + 1e-6)
    return float(np.clip(np.linalg.norm(np.mean(directions, axis=0)), 0.0, 1.0))


@dataclass
class _SessionState:
    session_id: str
    started_at: float
    good_frames: int = 0
    last_face_center: np.ndarray | None = None
    prev_gray_face: np.ndarray | None = None
    texture_hist: deque = field(default_factory=lambda: deque(maxlen=40))
    edge_hist: deque = field(default_factory=lambda: deque(maxlen=40))
    motion_hist: deque = field(default_factory=lambda: deque(maxlen=25))
    parallax_hist: deque = field(default_factory=lambda: deque(maxlen=25))
    flow_hist: deque = field(default_factory=lambda: deque(maxlen=25))
    flicker_hist: deque = field(default_factory=lambda: deque(maxlen=60))
    flags: set[str] = field(default_factory=set)
    last_dl_real: float | None = None
    last_dl_fake: float | None = None
    last_dl_smoothed: float | None = None
    final_decision: TurnstileDecision | None = None
    cooldown_until: float = 0.0


class TurnstileDecisionEngine:
    def __init__(
        self,
        camera_config: CameraConfig,
        turnstile_config: TurnstileConfig,
        detector: MediaPipeDetector,
        dl_analyzer: DeepLearningAnalyzer,
        assets: AssetManifest,
    ):
        self._camera_cfg = camera_config
        self._cfg = turnstile_config
        self._thr = turnstile_config.thresholds
        self._detector = detector
        self._dl = dl_analyzer
        self._assets = assets
        self._lock = threading.Lock()
        self._latest_final_decision: TurnstileDecision | None = None
        self._fps_hist: deque[float] = deque(maxlen=30)
        self._prev_frame_ts: float | None = None
        self._state = self._new_session()

    @property
    def model_versions(self) -> dict[str, object]:
        return {
            "mediapipe": Path(self._assets.mediapipe_model_path).name,
            "anti_spoof_models": list(self._assets.anti_spoof_model_names),
        }

    def reset(self) -> None:
        self._dl.reset()
        self._state = self._new_session()
        logger.info("turnstile_session_reset", session_id=self._state.session_id)

    def get_latest_decision(self) -> TurnstileDecision | None:
        with self._lock:
            return self._latest_final_decision

    def process_frame(self, frame: np.ndarray, timestamp_ms: int | None = None) -> TurnstileDecision:
        now = time.time()
        if timestamp_ms is None:
            timestamp_ms = int(now * 1000)
        self._update_fps(now)

        if self._state.final_decision is not None and now < self._state.cooldown_until:
            return self._cooldown_decision()
        if self._state.final_decision is not None and now >= self._state.cooldown_until:
            self.reset()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._state.flicker_hist.append(float(gray.mean()) / 255.0)

        face = self._detector.detect_video_frame(frame, timestamp_ms)
        if face is None or face.landmarks is None:
            self.reset()
            return self._build_decision(
                state=TurnstileState.NO_FACE,
                reason="Show your face to camera",
                reason_codes=["no_face"],
                confidence=0.0,
                is_final=False,
                details={},
            )

        x1 = face.bbox.x
        y1 = face.bbox.y
        x2 = face.bbox.x2
        y2 = face.bbox.y2
        face_gray = gray[y1:y2, x1:x2]
        quality_ok, quality_reason, quality_codes = self._quality_check(gray, face_gray, face.bbox)

        landmarks = face.landmarks
        w = frame.shape[1]
        h = frame.shape[0]
        nose = np.array([landmarks[NOSE_TIP][0], landmarks[NOSE_TIP][1]], dtype=np.float32)
        left_eye = 0.5 * (
            np.array(landmarks[LEFT_EYE_OUTER], dtype=np.float32)
            + np.array(landmarks[LEFT_EYE_INNER], dtype=np.float32)
        )
        right_eye = 0.5 * (
            np.array(landmarks[RIGHT_EYE_OUTER], dtype=np.float32)
            + np.array(landmarks[RIGHT_EYE_INNER], dtype=np.float32)
        )
        eyes_center = 0.5 * (left_eye + right_eye)

        if self._state.last_face_center is not None:
            motion = float(np.linalg.norm(nose - self._state.last_face_center))
            self._state.motion_hist.append(motion)
        self._state.last_face_center = nose.copy()

        if quality_ok:
            self._state.good_frames += 1
            self._state.texture_hist.append(_lbp_mean(face_gray))
            edge_val = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
            self._state.edge_hist.append(edge_val)

            flow_rigid = _compute_flow_rigidity(self._state.prev_gray_face, face_gray)
            if flow_rigid is not None:
                self._state.flow_hist.append(flow_rigid)
            self._state.prev_gray_face = face_gray.copy()

            inter_eye = _dist(left_eye, right_eye) + 1e-6
            self._state.parallax_hist.append(_dist(nose, eyes_center) / inter_eye)

            if float(np.mean(face_gray > 245)) > self._thr.glare_ratio:
                self._state.flags.add("glare")

            if self._state.good_frames % max(self._thr.dl_interval_frames, 1) == 0:
                dl_result = self._dl.predict_face(face.image, smooth=True)
                self._state.last_dl_real = dl_result["real_prob"]
                self._state.last_dl_fake = dl_result["fake_prob"]
                self._state.last_dl_smoothed = dl_result["smoothed"]
        else:
            self._state.prev_gray_face = None

        heuristic_score, metrics = self._compute_heuristic_score()
        dl_score = self._state.last_dl_smoothed
        analysis_elapsed_ms = (now - self._state.started_at) * 1000.0

        common_details = {
            "face_bbox": {"x": x1, "y": y1, "x2": x2, "y2": y2},
            "quality_ok": quality_ok,
            "quality_reason": quality_reason,
            "good_frames": self._state.good_frames,
            "heuristic_score": heuristic_score,
            "deep_learning_score": dl_score,
            "metrics": metrics,
        }

        if not quality_ok:
            return self._build_decision(
                state=TurnstileState.POSITIONING,
                reason=quality_reason,
                reason_codes=quality_codes,
                confidence=0.0,
                is_final=False,
                details=common_details,
            )

        if (
            analysis_elapsed_ms < self._cfg.decision_window_ms
            or self._state.good_frames < self._cfg.min_good_frames
        ):
            return self._build_decision(
                state=TurnstileState.ANALYZING,
                reason="Analyzing face",
                reason_codes=["analyzing"],
                confidence=max(heuristic_score, dl_score or 0.0),
                is_final=False,
                details=common_details,
            )

        fusion_score = self._fuse_scores(heuristic_score, dl_score)
        strong_spoof, spoof_codes = self._strong_spoof(metrics)
        grant_threshold_met = fusion_score >= self._thr.live_score_threshold and (
            dl_score is None or dl_score >= self._thr.dl_real_min_threshold
        )

        if strong_spoof or (self._state.last_dl_fake or 0.0) >= self._thr.dl_fake_override_threshold:
            final_state = TurnstileState.ACCESS_DENIED
            reason = "Photo / screen / replay suspected"
            reason_codes = spoof_codes or ["deep_learning_fake_high"]
            confidence = max(
                1.0 - fusion_score,
                self._state.last_dl_fake or 0.0,
                1.0 - heuristic_score,
            )
        elif grant_threshold_met:
            final_state = TurnstileState.ACCESS_GRANTED
            reason = "Live person confirmed"
            reason_codes = ["live_confirmed"]
            confidence = max(fusion_score, heuristic_score, dl_score or 0.0)
        elif fusion_score <= self._thr.spoof_score_threshold:
            final_state = TurnstileState.ACCESS_DENIED
            reason = "Not enough live evidence"
            reason_codes = ["fusion_score_low"]
            confidence = max(1.0 - fusion_score, self._state.last_dl_fake or 0.0)
        else:
            final_state = TurnstileState.ACCESS_DENIED
            reason = "Security-first deny"
            reason_codes = ["security_first_deny"]
            confidence = max(1.0 - fusion_score, 0.5)

        decision = self._build_decision(
            state=final_state,
            reason=reason,
            reason_codes=reason_codes,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            is_final=True,
            live_score=float(fusion_score),
            heuristic_score=heuristic_score,
            deep_learning_score=dl_score,
            details=common_details,
        )
        self._finalize(decision)
        return decision

    def _new_session(self) -> _SessionState:
        return _SessionState(session_id=str(uuid.uuid4()), started_at=time.time())

    def _update_fps(self, now: float) -> None:
        if self._prev_frame_ts is not None:
            dt = max(now - self._prev_frame_ts, 1e-6)
            self._fps_hist.append(1.0 / dt)
        self._prev_frame_ts = now

    def _fps(self) -> float:
        return float(np.mean(self._fps_hist)) if self._fps_hist else 30.0

    def _quality_check(self, gray: np.ndarray, face_gray: np.ndarray, bbox) -> tuple[bool, str, list[str]]:
        frame_h, frame_w = gray.shape[:2]
        face_ratio = (bbox.width * bbox.height) / max(frame_w * frame_h, 1)
        brightness = float(face_gray.mean()) / 255.0
        blur_var = float(cv2.Laplacian(face_gray, cv2.CV_64F).var()) if face_gray.size else 0.0

        if face_ratio < self._thr.min_face_ratio:
            return False, "Move closer", ["face_too_small"]
        if face_ratio > self._thr.max_face_ratio:
            return False, "Move back a little", ["face_too_large"]
        if brightness < self._thr.min_brightness:
            return False, "More light on face", ["low_light"]
        if brightness > self._thr.max_brightness:
            return False, "Reduce strong light", ["overexposed"]
        if blur_var < self._thr.min_blur_var:
            return False, "Hold still", ["blurred_face"]
        return True, "Good", ["quality_ok"]

    def _compute_heuristic_score(self) -> tuple[float, dict[str, object]]:
        flags = set(self._state.flags)
        texture = float(np.mean(self._state.texture_hist)) if self._state.texture_hist else 0.0
        edge = float(np.mean(self._state.edge_hist)) if self._state.edge_hist else 0.0
        motion = float(np.mean(self._state.motion_hist)) if self._state.motion_hist else 0.0
        flow = float(np.mean(self._state.flow_hist)) if self._state.flow_hist else 1.0

        if len(self._state.parallax_hist) >= 2:
            vals = np.array(self._state.parallax_hist, dtype=np.float32)
            parallax = float(np.mean(np.abs(vals - np.mean(vals))))
        else:
            parallax = 0.0

        flicker_hz, flicker_amp = _dominant_frequency(
            np.array(self._state.flicker_hist, dtype=np.float32),
            self._fps(),
            self._thr.flicker_min_hz,
            self._thr.flicker_max_hz,
        )

        texture_score = _linear_band_score(
            texture,
            self._thr.texture_real_min,
            self._thr.texture_real_min + 0.22,
        )

        motion_score = 0.0
        if self._thr.motion_min <= motion <= self._thr.motion_max:
            center = (self._thr.motion_min + self._thr.motion_max) / 2.0
            motion_score = max(0.0, 1.0 - abs(motion - center) / center)

        parallax_score = _linear_band_score(
            parallax,
            self._thr.parallax_live_min,
            self._thr.parallax_live_min + 0.02,
        )
        flow_score = 1.0 - _linear_band_score(flow, self._thr.flow_rigidity_spoof, 0.995)

        edge_score = 1.0
        if edge > self._thr.edge_screen_high:
            edge_score = max(0.0, 1.0 - (edge - self._thr.edge_screen_high) / 220.0)

        flicker_score = 1.0
        if (
            flicker_amp > self._thr.flicker_amp_spoof
            and self._thr.flicker_min_hz <= flicker_hz <= self._thr.flicker_max_hz
        ):
            flicker_score = 0.0
            flags.add("screen_flicker")

        if flow > 0.97 and len(self._state.flow_hist) >= 8:
            flags.add("flat_motion")
        if edge > self._thr.edge_screen_high + 90:
            flags.add("sharp_edges")

        score = (
            0.26 * texture_score
            + 0.18 * motion_score
            + 0.22 * parallax_score
            + 0.22 * flow_score
            + 0.06 * edge_score
            + 0.06 * flicker_score
        )
        score = float(np.clip(score, 0.0, 1.0))
        self._state.flags = flags

        return score, {
            "texture": texture,
            "motion": motion,
            "parallax": parallax,
            "flow": flow,
            "edge": edge,
            "flicker_hz": flicker_hz,
            "flicker_amp": flicker_amp,
            "flags": sorted(flags),
        }

    def _strong_spoof(self, metrics: dict[str, object]) -> tuple[bool, list[str]]:
        flags = set(metrics.get("flags", []))
        reason_codes: list[str] = []
        if "screen_flicker" in flags:
            reason_codes.append("screen_flicker")
        if "flat_motion" in flags and float(metrics.get("parallax", 0.0)) < self._thr.parallax_live_min * 0.8:
            reason_codes.append("flat_motion_low_parallax")
        if (
            float(metrics.get("flow", 0.0)) > 0.985
            and float(metrics.get("texture", 0.0)) < self._thr.texture_real_min
        ):
            reason_codes.append("rigid_low_texture")
        return bool(reason_codes), reason_codes

    def _fuse_scores(self, heuristic_score: float, dl_score: float | None) -> float:
        if dl_score is None:
            return heuristic_score
        weighted = (
            self._thr.heuristic_weight * heuristic_score
            + self._thr.deep_learning_weight * dl_score
        )
        total_weight = self._thr.heuristic_weight + self._thr.deep_learning_weight
        return float(weighted / max(total_weight, 1e-6))

    def _controller_verdict(self, state: TurnstileState, is_final: bool) -> ControllerVerdict | None:
        if not is_final:
            return None
        if state == TurnstileState.ACCESS_GRANTED:
            return ControllerVerdict.ACCESS_GRANTED
        return ControllerVerdict.ACCESS_DENIED

    def _build_decision(
        self,
        *,
        state: TurnstileState,
        reason: str,
        reason_codes: list[str],
        confidence: float,
        is_final: bool,
        details: dict[str, object],
        live_score: float | None = None,
        heuristic_score: float | None = None,
        deep_learning_score: float | None = None,
    ) -> TurnstileDecision:
        return TurnstileDecision(
            session_id=self._state.session_id,
            state=state,
            reason=reason,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            reason_codes=reason_codes,
            controller_verdict=self._controller_verdict(state, is_final),
            timestamp_utc=_utc_now(),
            latency_ms=(time.time() - self._state.started_at) * 1000.0,
            camera_id=self._camera_cfg.id,
            device_id=self._cfg.device_id,
            live_score=float(live_score if live_score is not None else confidence),
            heuristic_score=float(heuristic_score if heuristic_score is not None else confidence),
            deep_learning_score=deep_learning_score,
            is_final=is_final,
            details={
                **details,
                "model_versions": self.model_versions,
            },
        )

    def _finalize(self, decision: TurnstileDecision) -> None:
        self._state.final_decision = decision
        self._state.cooldown_until = time.time() + (self._cfg.cooldown_ms / 1000.0)
        with self._lock:
            self._latest_final_decision = decision
        self._write_audit(decision)
        self._publish_webhook(decision)
        logger.info(
            "turnstile_decision_finalized",
            session_id=decision.session_id,
            verdict=decision.controller_verdict.value if decision.controller_verdict else None,
            confidence=round(decision.confidence, 4),
            reason_codes=decision.reason_codes,
            latency_ms=round(decision.latency_ms, 1),
        )

    def _cooldown_decision(self) -> TurnstileDecision:
        final_decision = self._state.final_decision
        assert final_decision is not None
        return TurnstileDecision(
            session_id=final_decision.session_id,
            state=TurnstileState.COOLDOWN,
            reason=f"Cooldown after {final_decision.state.value}",
            confidence=final_decision.confidence,
            reason_codes=list(final_decision.reason_codes),
            controller_verdict=final_decision.controller_verdict,
            timestamp_utc=_utc_now(),
            latency_ms=final_decision.latency_ms,
            camera_id=final_decision.camera_id,
            device_id=final_decision.device_id,
            live_score=final_decision.live_score,
            heuristic_score=final_decision.heuristic_score,
            deep_learning_score=final_decision.deep_learning_score,
            is_final=False,
            details=final_decision.details,
        )

    def _decision_payload(self, decision: TurnstileDecision) -> dict[str, object]:
        return {
            "session_id": decision.session_id,
            "device_id": decision.device_id,
            "camera_id": decision.camera_id,
            "timestamp_utc": decision.timestamp_utc,
            "state": decision.state.value,
            "verdict": decision.controller_verdict.value if decision.controller_verdict else None,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "reason_codes": list(decision.reason_codes),
            "latency_ms": decision.latency_ms,
            "live_score": decision.live_score,
            "heuristic_score": decision.heuristic_score,
            "deep_learning_score": decision.deep_learning_score,
            "model_versions": self.model_versions,
            "details": decision.details,
        }

    def _write_audit(self, decision: TurnstileDecision) -> None:
        audit_dir = resolve_path(self._cfg.audit_log_dir)
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_file = audit_dir / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
        with audit_file.open("a", encoding="utf-8") as file_obj:
            json.dump(self._decision_payload(decision), file_obj, ensure_ascii=True)
            file_obj.write("\n")

    def _publish_webhook(self, decision: TurnstileDecision) -> None:
        if not self._cfg.webhook_url:
            return

        payload = json.dumps(self._decision_payload(decision)).encode("utf-8")

        def _send() -> None:
            request = urllib.request.Request(
                self._cfg.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=max(self._cfg.webhook_timeout_ms / 1000.0, 0.1),
                ) as response:
                    logger.info(
                        "turnstile_webhook_sent",
                        session_id=decision.session_id,
                        status=getattr(response, "status", 200),
                    )
            except (urllib.error.URLError, TimeoutError) as exc:
                logger.warning(
                    "turnstile_webhook_failed",
                    session_id=decision.session_id,
                    error=str(exc),
                )

        threading.Thread(target=_send, daemon=True).start()
