from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import cv2
import numpy as np

from app.adapters.analyzers.heuristic_analyzer import HeuristicAnalyzer
from app.config import ActiveLivenessSettings
from app.core.models import (
    ActiveLivenessResult,
    ActiveLivenessVerdict,
    BoundingBox,
    FramePacket,
    TrackedFace,
)
from app.domain.entities import BBox, FaceRegion
from app.utils.geometry import bbox_iou

try:
    from app.adapters.detectors.mediapipe_detector import MediaPipeDetector
    from app.domain.exceptions import AssetValidationError
    from app.infrastructure.config import HeuristicConfig, MediaPipeConfig
    _ACTIVE_LIVENESS_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - depends on host runtime packages
    MediaPipeDetector = None  # type: ignore[assignment]
    AssetValidationError = Exception  # type: ignore[assignment]
    HeuristicConfig = None  # type: ignore[assignment]
    MediaPipeConfig = None  # type: ignore[assignment]
    _ACTIVE_LIVENESS_IMPORT_ERROR = exc

LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
NOSE_TIP = 1


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


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


def _tracked_bbox(box: BBox) -> BoundingBox:
    return BoundingBox(x=box.x, y=box.y, width=box.width, height=box.height)


@dataclass
class _ActiveLivenessState:
    session_id: str
    started_at: float
    analyzed_frames: int = 0
    last_result: ActiveLivenessResult | None = None
    prev_gray_face: np.ndarray | None = None
    texture_hist: deque[float] = field(default_factory=lambda: deque(maxlen=24))
    flow_hist: deque[float] = field(default_factory=lambda: deque(maxlen=24))
    parallax_hist: deque[float] = field(default_factory=lambda: deque(maxlen=24))
    flicker_hist: deque[float] = field(default_factory=lambda: deque(maxlen=48))
    timestamp_hist: deque[float] = field(default_factory=lambda: deque(maxlen=48))


class BaseActiveLivenessService(Protocol):
    backend_name: str

    def evaluate(
        self,
        session_id: str,
        frame_packet: FramePacket,
        tracked_face: TrackedFace,
    ) -> ActiveLivenessResult | None:
        ...


class DisabledActiveLivenessService:
    backend_name = "disabled"

    def evaluate(
        self,
        session_id: str,
        frame_packet: FramePacket,
        tracked_face: TrackedFace,
    ) -> ActiveLivenessResult | None:
        return None


class UnavailableActiveLivenessService:
    backend_name = "mock_unavailable"

    def __init__(self, reason: str):
        self._reason = reason

    def evaluate(
        self,
        session_id: str,
        frame_packet: FramePacket,
        tracked_face: TrackedFace,
    ) -> ActiveLivenessResult | None:
        return ActiveLivenessResult(
            verdict=ActiveLivenessVerdict.UNAVAILABLE,
            confidence=0.0,
            reason=self._reason,
            details={"reason": self._reason},
        )


class MediaPipeActiveLivenessService:
    backend_name = "mediapipe_active_liveness"

    def __init__(self, settings: ActiveLivenessSettings):
        self._settings = settings
        if not settings.model_path:
            raise ValueError("Active liveness model_path is required for MediaPipe backend")
        if MediaPipeDetector is None or MediaPipeConfig is None or HeuristicConfig is None:
            raise ImportError(str(_ACTIVE_LIVENESS_IMPORT_ERROR or "MediaPipe runtime is unavailable"))

        self._detector = MediaPipeDetector(
            MediaPipeConfig(model_path=settings.model_path),
            running_mode="VIDEO",
        )
        self._heuristic = HeuristicAnalyzer(
            HeuristicConfig(
                blinks_needed=settings.blinks_needed,
                ear_threshold=settings.ear_threshold,
                min_blink_frames=settings.min_blink_frames,
                moves_needed=settings.moves_needed,
                move_pixels=settings.move_pixels,
                move_max_jump=settings.move_max_jump,
                smooth_window=settings.smooth_window,
                ratio_diff_min=settings.ratio_diff_min,
                ratio_checks_min=settings.ratio_checks_min,
                texture_min=settings.texture_min,
                texture_frames=settings.texture_frames,
            )
        )
        self._state = _ActiveLivenessState(session_id="", started_at=0.0)

    def evaluate(
        self,
        session_id: str,
        frame_packet: FramePacket,
        tracked_face: TrackedFace,
    ) -> ActiveLivenessResult | None:
        if self._state.session_id != session_id:
            self._reset(session_id, frame_packet.timestamp)

        if self._settings.run_every_n_frames > 1 and self._state.analyzed_frames > 0:
            if self._state.analyzed_frames % self._settings.run_every_n_frames != 0:
                self._state.analyzed_frames += 1
                return self._state.last_result or self._pending_result(frame_packet.timestamp)

        self._state.analyzed_frames += 1
        timestamp_ms = int(frame_packet.timestamp * 1000)
        face_region = self._detector.detect_video_frame(frame_packet.frame, timestamp_ms)
        if face_region is None or face_region.landmarks is None:
            result = self._timeout_or_pending(
                frame_packet.timestamp,
                reason="Look straight at the camera",
            )
            self._state.last_result = result
            return result

        if bbox_iou(_tracked_bbox(face_region.bbox), tracked_face.bbox) < self._settings.min_face_iou:
            result = self._timeout_or_pending(
                frame_packet.timestamp,
                reason="Keep only one face in view",
            )
            self._state.last_result = result
            return result

        heuristic = self._heuristic.process_frame(frame_packet.frame, face_region)
        checks = {
            item["name"]: item
            for item in heuristic.details.get("checks", [])
            if isinstance(item, dict) and "name" in item
        }
        blink_ok = bool(checks.get("blink", {}).get("passed", False))
        movement_ok = bool(checks.get("movement", {}).get("passed", False))
        ratio_ok = bool(checks.get("ratio", {}).get("passed", False))

        metrics = self._screen_metrics(face_region, frame_packet.timestamp)
        spoof_codes = self._strong_spoof(metrics, frame_packet.timestamp)
        details = {
            "heuristic": heuristic.details,
            "screen_metrics": metrics,
            "elapsed_seconds": frame_packet.timestamp - self._state.started_at,
            "backend": self.backend_name,
        }

        if spoof_codes:
            result = ActiveLivenessResult(
                verdict=ActiveLivenessVerdict.SPOOF,
                confidence=0.94,
                reason="Screen / photo attack suspected",
                details={**details, "reason_codes": spoof_codes},
            )
        elif blink_ok or (movement_ok and ratio_ok):
            confidence = max(float(heuristic.confidence), 0.78 if blink_ok else 0.72)
            result = ActiveLivenessResult(
                verdict=ActiveLivenessVerdict.REAL,
                confidence=confidence,
                reason="Live facial response confirmed",
                details=details,
            )
        elif frame_packet.timestamp - self._state.started_at >= self._settings.challenge_timeout_seconds:
            if blink_ok or movement_ok or ratio_ok:
                result = self._pending_result(
                    frame_packet.timestamp,
                    reason="Continue: blink or turn your head a bit more",
                    details=details,
                )
            else:
                result = ActiveLivenessResult(
                    verdict=ActiveLivenessVerdict.FAILED,
                    confidence=max(float(heuristic.confidence), 0.65),
                    reason="Blink or turn your head slightly and try again",
                    details=details,
                )
        else:
            result = self._pending_result(
                frame_packet.timestamp,
                reason="Blink or turn your head slightly",
                details=details,
            )

        self._state.last_result = result
        return result

    def _screen_metrics(self, face_region: FaceRegion, timestamp: float) -> dict[str, float | list[str]]:
        gray = cv2.cvtColor(face_region.image, cv2.COLOR_BGR2GRAY)
        self._state.flicker_hist.append(float(gray.mean()) / 255.0)
        self._state.timestamp_hist.append(timestamp)
        self._state.texture_hist.append(_lbp_mean(gray))

        flow = _compute_flow_rigidity(self._state.prev_gray_face, gray)
        if flow is not None:
            self._state.flow_hist.append(flow)
        self._state.prev_gray_face = gray.copy()

        lm = face_region.landmarks
        nose = np.array([lm[NOSE_TIP][0], lm[NOSE_TIP][1]], dtype=np.float32)
        left_eye = 0.5 * (
            np.array(lm[LEFT_EYE_OUTER], dtype=np.float32)
            + np.array(lm[LEFT_EYE_INNER], dtype=np.float32)
        )
        right_eye = 0.5 * (
            np.array(lm[RIGHT_EYE_OUTER], dtype=np.float32)
            + np.array(lm[RIGHT_EYE_INNER], dtype=np.float32)
        )
        inter_eye = _dist(left_eye, right_eye) + 1e-6
        eyes_center = 0.5 * (left_eye + right_eye)
        self._state.parallax_hist.append(_dist(nose, eyes_center) / inter_eye)

        if len(self._state.parallax_hist) >= 2:
            vals = np.array(self._state.parallax_hist, dtype=np.float32)
            parallax = float(np.mean(np.abs(vals - np.mean(vals))))
        else:
            parallax = 0.0

        if len(self._state.timestamp_hist) >= 2:
            elapsed = self._state.timestamp_hist[-1] - self._state.timestamp_hist[0]
            fps = (len(self._state.timestamp_hist) - 1) / elapsed if elapsed > 0 else 30.0
        else:
            fps = 30.0
        flicker_hz, flicker_amp = _dominant_frequency(
            np.array(self._state.flicker_hist, dtype=np.float32),
            fps,
            self._settings.flicker_min_hz,
            self._settings.flicker_max_hz,
        )
        return {
            "texture": float(np.mean(self._state.texture_hist)) if self._state.texture_hist else 0.0,
            "flow": float(np.mean(self._state.flow_hist)) if self._state.flow_hist else 0.0,
            "parallax": parallax,
            "flicker_hz": flicker_hz,
            "flicker_amp": flicker_amp,
        }

    def _strong_spoof(self, metrics: dict[str, float | list[str]], now: float) -> list[str]:
        elapsed = now - self._state.started_at
        if elapsed < self._settings.min_spoof_signal_seconds:
            return []

        reason_codes: list[str] = []
        flow = float(metrics.get("flow", 0.0))
        parallax = float(metrics.get("parallax", 0.0))
        texture = float(metrics.get("texture", 0.0))
        flicker_hz = float(metrics.get("flicker_hz", 0.0))
        flicker_amp = float(metrics.get("flicker_amp", 0.0))

        if (
            flicker_amp > self._settings.flicker_amp_spoof
            and self._settings.flicker_min_hz <= flicker_hz <= self._settings.flicker_max_hz
        ):
            reason_codes.append("screen_flicker")
        if flow > self._settings.flow_rigidity_spoof and parallax < self._settings.parallax_live_min:
            reason_codes.append("flat_motion_low_parallax")
        if flow > self._settings.flow_rigidity_spoof and texture < self._settings.texture_spoof_max:
            reason_codes.append("rigid_low_texture")
        return reason_codes

    def _pending_result(
        self,
        now: float,
        *,
        reason: str = "Blink or turn your head slightly",
        details: dict[str, object] | None = None,
    ) -> ActiveLivenessResult:
        elapsed = max(now - self._state.started_at, 0.0)
        progress = min(elapsed / max(self._settings.challenge_timeout_seconds, 1e-6), 0.95)
        return ActiveLivenessResult(
            verdict=ActiveLivenessVerdict.PENDING,
            confidence=progress,
            reason=reason,
            details=details or {"elapsed_seconds": elapsed, "backend": self.backend_name},
        )

    def _timeout_or_pending(self, now: float, *, reason: str) -> ActiveLivenessResult:
        elapsed = max(now - self._state.started_at, 0.0)
        if elapsed >= self._settings.challenge_timeout_seconds:
            return ActiveLivenessResult(
                verdict=ActiveLivenessVerdict.FAILED,
                confidence=0.7,
                reason="Live facial response was not confirmed",
                details={"elapsed_seconds": elapsed, "backend": self.backend_name},
            )
        return self._pending_result(now, reason=reason)

    def _reset(self, session_id: str, started_at: float) -> None:
        self._heuristic.reset()
        self._state = _ActiveLivenessState(session_id=session_id, started_at=started_at)


def create_active_liveness_service(
    settings: ActiveLivenessSettings,
) -> BaseActiveLivenessService:
    if not settings.enabled:
        return DisabledActiveLivenessService()
    if settings.backend != "mediapipe" or not settings.model_path:
        return UnavailableActiveLivenessService("Active liveness backend is unavailable")
    try:
        return MediaPipeActiveLivenessService(settings)
    except (ImportError, AssetValidationError, ValueError) as exc:
        return UnavailableActiveLivenessService(str(exc))
