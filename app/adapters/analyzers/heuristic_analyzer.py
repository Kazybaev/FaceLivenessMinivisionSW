"""Heuristic liveness analyzer — 4 checks: EAR/blink, movement, ratio, texture.

Original: liveness_detection.py:71-211
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np
import structlog

from app.domain.entities import (
    FaceRegion, LivenessResult, CheckResult, HeuristicState,
)
from app.domain.enums import LivenessVerdict, DetectionMethod
from app.infrastructure.config import HeuristicConfig

logger = structlog.get_logger(__name__)


def calc_ear(pts: np.ndarray) -> float:
    """Eye Aspect Ratio: (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)"""
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def _to_python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


class HeuristicAnalyzer:
    """Stateful liveness analyzer using 4 heuristic checks."""

    def __init__(self, config: HeuristicConfig):
        self._cfg = config
        self._state = HeuristicState(
            nose_buf=deque(maxlen=config.smooth_window),
            leye_buf=deque(maxlen=config.smooth_window),
            reye_buf=deque(maxlen=config.smooth_window),
            texture_buf=deque(maxlen=config.texture_frames),
        )

    def process_frame(self, image: np.ndarray, face: FaceRegion) -> LivenessResult:
        if face.landmarks is None:
            return LivenessResult(
                verdict=LivenessVerdict.UNCERTAIN,
                confidence=0.0,
                method=DetectionMethod.HEURISTIC,
                details={"error": "no landmarks"},
            )

        h, w = image.shape[:2]
        lm = face.landmarks
        cfg = self._cfg
        state = self._state

        # 1. EAR / Blink
        left_pts = np.array([lm[i] for i in cfg.left_eye])
        right_pts = np.array([lm[i] for i in cfg.right_eye])
        current_ear = (calc_ear(left_pts) + calc_ear(right_pts)) / 2.0
        state.ear_buf.append(current_ear)

        if current_ear < cfg.ear_threshold:
            state.blink_frames += 1
        else:
            if state.blink_frames >= cfg.min_blink_frames:
                state.blink_count += 1
            state.blink_frames = 0

        # 2. Head movement (smoothed)
        nose_raw = lm[cfg.nose_tip]
        lcor_raw = lm[cfg.left_corner]
        rcor_raw = lm[cfg.right_corner]

        state.nose_buf.append(nose_raw)
        state.leye_buf.append(lcor_raw)
        state.reye_buf.append(rcor_raw)

        ratio_checked = False
        if len(state.nose_buf) == cfg.smooth_window:
            s_nose = np.mean(state.nose_buf, axis=0)
            s_leye = np.mean(state.leye_buf, axis=0)
            s_reye = np.mean(state.reye_buf, axis=0)

            nose_dist = float(np.linalg.norm(s_nose - state.nose_buf[0]))
            leye_dist = float(np.linalg.norm(s_leye - state.leye_buf[0]))
            reye_dist = float(np.linalg.norm(s_reye - state.reye_buf[0]))
            eye_dist = (leye_dist + reye_dist) / 2.0

            if cfg.move_pixels < nose_dist < cfg.move_max_jump:
                state.move_count += 1

            # 3. Nose/Eye ratio
            if eye_dist > 2.0:
                ratio = nose_dist / (eye_dist + 1e-6)
                if abs(ratio - 1.0) > cfg.ratio_diff_min:
                    state.ratio_ok_count += 1
                ratio_checked = True

        # 4. Skin texture
        xs = lm[:, 0].astype(int)
        ys = lm[:, 1].astype(int)
        fx1 = max(0, int(xs.min()) - 10)
        fx2 = min(w, int(xs.max()) + 10)
        fy1 = max(0, int(ys.min()) - 10)
        fy2 = min(h, int(ys.max()) + 10)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_roi = gray[fy1:fy2, fx1:fx2]
        if face_roi.size > 100:
            lap = float(cv2.Laplacian(face_roi, cv2.CV_64F).var())
            state.texture_buf.append(lap)

        avg_texture = float(np.mean(state.texture_buf)) if state.texture_buf else 0.0
        blink_ok = state.blink_count >= cfg.blinks_needed
        move_ok = state.move_count >= cfg.moves_needed
        ratio_ok = state.ratio_ok_count >= cfg.ratio_checks_min
        texture_ok = avg_texture >= cfg.texture_min

        checks = [
            CheckResult("blink", blink_ok, state.blink_count, cfg.blinks_needed),
            CheckResult("movement", move_ok, state.move_count, cfg.moves_needed),
            CheckResult("ratio", ratio_ok, state.ratio_ok_count, cfg.ratio_checks_min),
            CheckResult("texture", texture_ok, avg_texture, cfg.texture_min),
        ]
        checks_passed = sum(c.passed for c in checks)
        is_real = checks_passed == 4

        if is_real:
            verdict = LivenessVerdict.REAL
        else:
            verdict = LivenessVerdict.FAKE

        return LivenessResult(
            verdict=verdict,
            confidence=checks_passed / 4.0,
            method=DetectionMethod.HEURISTIC,
            details={
                "checks": [
                    {
                        "name": c.name,
                        "passed": bool(c.passed),
                        "value": _to_python_scalar(c.value),
                        "threshold": _to_python_scalar(c.threshold),
                    }
                    for c in checks
                ],
                "checks_passed": int(checks_passed),
                "ear": _to_python_scalar(current_ear),
                "avg_texture": _to_python_scalar(avg_texture),
            },
        )

    def reset(self) -> None:
        self._state.reset()

    def analyze(self, image: np.ndarray, face: FaceRegion) -> LivenessResult:
        return self.process_frame(image, face)
