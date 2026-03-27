# -*- coding: utf-8 -*-
"""
Liveness Detection v7.0 - Clear UI / Single file / No file logs

What changed vs previous version:
- no Russian text in OpenCV overlay (prevents ???? on many systems)
- clearer final statuses: NO FACE / CALIBRATING / CHECKING / LIVE / SPOOF SUSPECT / RETRY
- bigger user hints on screen
- softer thresholds for low-quality webcams
- face guide box in the center
- no .log files are saved; logs go to console only

Install:
    pip install opencv-python mediapipe numpy

Run:
    python liveness_v7_clear_ui.py
    python liveness_v7_clear_ui.py --camera 0
    python liveness_v7_clear_ui.py --session-sec 14 --debug

Controls:
    Q - quit
    R - reset session
"""

import argparse
import collections
from dataclasses import dataclass, field
import logging
import math
import os
import random
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

# -----------------------------------------------------------------------------
# LOGGING - console only
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("liveness_v7")

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
VERSION = "7.0"
BUILD_DATE = "2026-03-26"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = "face_landmarker.task"

DEFAULT_SESSION_SEC = 12.0
MIN_GOOD_FRAMES = 28
ALLOW_SCORE = 0.66
RETRY_SCORE = 0.44
HARD_BLOCK_SCORE = 0.30

# softer quality gate for weak webcams
MIN_FACE_RATIO = 0.055
MAX_FACE_RATIO = 0.60
MIN_BRIGHTNESS = 0.10
MAX_BRIGHTNESS = 0.96
MIN_BLUR_VAR = 20.0
MAX_ABS_YAW = 35.0
MAX_ABS_PITCH = 28.0
MAX_ABS_ROLL = 30.0
MIN_STABLE_FACE_FRAMES = 6

EAR_BASELINE_FRAMES = 24
EAR_CLOSED_RATIO = 0.79
BLINK_MIN_FRAMES = 1
BLINK_MAX_FRAMES = 10
BLINKS_REQUIRED = 1

TEXTURE_WINDOW = 20
EDGE_WINDOW = 20
FLICKER_WINDOW = 96
FLOW_WINDOW = 12
POSE_WINDOW = 18
MOVE_WINDOW = 10
EXPR_WINDOW = 12

TEXTURE_REAL_MIN = 0.34
EDGE_SCREEN_MAX = 220.0
FLICKER_AMP_THRESH = 0.014
FLICKER_MIN_HZ = 8.0
FLICKER_MAX_HZ = 40.0
FLOW_RIGIDITY_MAX = 0.93
POSE_PARALLAX_MIN = 0.006
MOTION_MIN_PX = 0.6
MOTION_MAX_PX = 22.0
EXPR_MIN_STD = 0.45
RPPG_MIN_PEAK = 0.0022
RPPG_MIN_HZ = 0.75
RPPG_MAX_HZ = 2.4

CHALLENGE_TIMEOUT = 5.0
CHALLENGE_MIN_HOLD = 0.35
NOSE_TURN_LEFT = -9.0
NOSE_TURN_RIGHT = 9.0
MOUTH_OPEN_RATIO = 1.05
BROW_RAISE_DELTA = -2.0

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
NOSE_TIP = 1
CHIN = 152
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
LEFT_BROW = 70
RIGHT_BROW = 300

FACE_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1),
], dtype=np.float64)

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def ensure_model(path: str = MODEL_PATH) -> str:
    if os.path.exists(path):
        return path
    log.info("Downloading MediaPipe model...")
    urllib.request.urlretrieve(MODEL_URL, path)
    log.info("Model saved to %s", path)
    return path


def build_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model()),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.45,
        min_face_presence_confidence=0.45,
        min_tracking_confidence=0.45,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


def p2(lm, idx, w, h):
    return np.array([lm[idx].x * w, lm[idx].y * h], dtype=np.float32)


def dist(a, b):
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(points):
    p1, p2_, p3, p4, p5, p6 = points
    vertical1 = np.linalg.norm(p2_ - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4) + 1e-6
    return float((vertical1 + vertical2) / (2.0 * horizontal))


def clip_roi(x1, y1, x2, y2, w, h):
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(1, min(w, int(x2)))
    y2 = max(1, min(h, int(y2)))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def face_bbox_from_landmarks(lm, w, h, pad=0.20):
    xs = [pt.x * w for pt in lm]
    ys = [pt.y * h for pt in lm]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * pad
    x2 += bw * pad
    y1 -= bh * pad
    y2 += bh * pad
    return clip_roi(x1, y1, x2, y2, w, h)


def lbp_mean(gray_roi: np.ndarray) -> float:
    if gray_roi.shape[0] < 8 or gray_roi.shape[1] < 8:
        return 0.0
    c = gray_roi[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.uint8)
    neigh = [
        gray_roi[:-2, :-2], gray_roi[:-2, 1:-1], gray_roi[:-2, 2:],
        gray_roi[1:-1, 2:], gray_roi[2:, 2:], gray_roi[2:, 1:-1],
        gray_roi[2:, :-2], gray_roi[1:-1, :-2],
    ]
    for i, n in enumerate(neigh):
        code |= ((n >= c).astype(np.uint8) << i)
    hist = cv2.calcHist([code], [0], None, [256], [0, 256]).flatten()
    hist /= max(hist.sum(), 1.0)
    uniformity = float(np.max(hist))
    entropy = float(-np.sum(hist * np.log2(hist + 1e-8))) / 8.0
    value = 0.55 * entropy + 0.45 * (1.0 - uniformity)
    return float(np.clip(value, 0.0, 1.0))


def dominant_frequency(signal: np.ndarray, fps: float, fmin: float, fmax: float):
    if len(signal) < 12 or fps <= 1.0:
        return 0.0, 0.0
    x = signal.astype(np.float32)
    x = x - np.mean(x)
    if np.std(x) < 1e-6:
        return 0.0, 0.0
    window = np.hanning(len(x))
    xf = np.fft.rfft(x * window)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fps)
    amps = np.abs(xf)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0, 0.0
    idx = np.argmax(amps[mask])
    freqs_m = freqs[mask]
    amps_m = amps[mask]
    return float(freqs_m[idx]), float(amps_m[idx] / len(x))


def estimate_head_pose(image_size, image_points):
    h, w = image_size
    focal = float(w)
    center = (w / 2.0, h / 2.0)
    cam_matrix = np.array([
        [focal, 0, center[0]],
        [0, focal, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(
        FACE_3D_POINTS,
        image_points.astype(np.float64),
        cam_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rot_mat, _ = cv2.Rodrigues(rvec)
    proj_mat = np.hstack((rot_mat, tvec))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)
    euler = np.asarray(euler, dtype=np.float64).reshape(-1)
    if euler.size < 3:
        return None
    pitch = float(euler[0])
    yaw = float(euler[1])
    roll = float(euler[2])
    return yaw, pitch, roll


def safe_mean(values, default=0.0):
    return float(np.mean(values)) if len(values) else float(default)


def safe_std(values, default=0.0):
    return float(np.std(values)) if len(values) else float(default)


def linear_band_score(x, low, high):
    if x <= low:
        return 0.0
    if x >= high:
        return 1.0
    return float((x - low) / max(high - low, 1e-6))


def draw_bar(img, x, y, value, color, width=220, height=10, bg=(40, 40, 45)):
    value = float(np.clip(value, 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + width, y + height), bg, -1)
    cv2.rectangle(img, (x, y), (x + int(width * value), y + height), color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (70, 70, 80), 1)


def overlay_text(img, text, x, y, color=(220, 220, 220), scale=0.55, thickness=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_center_guide(img):
    h, w = img.shape[:2]
    gw = int(w * 0.33)
    gh = int(h * 0.56)
    x1 = (w - gw) // 2
    y1 = (h - gh) // 2
    x2 = x1 + gw
    y2 = y1 + gh
    cv2.rectangle(img, (x1, y1), (x2, y2), (70, 90, 100), 1)
    return (x1, y1, x2, y2)


def bbox_overlap_ratio(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter / area_a

# -----------------------------------------------------------------------------
# STATE
# -----------------------------------------------------------------------------
@dataclass
class QualityInfo:
    ok: bool = False
    brightness: float = 0.0
    blur_var: float = 0.0
    face_ratio: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    reason: str = ""


@dataclass
class FeatureState:
    ear_baseline_buf: collections.deque = field(default_factory=lambda: collections.deque(maxlen=EAR_BASELINE_FRAMES))
    nose_hist: collections.deque = field(default_factory=lambda: collections.deque(maxlen=MOVE_WINDOW))
    mouth_hist: collections.deque = field(default_factory=lambda: collections.deque(maxlen=EXPR_WINDOW))
    brow_hist: collections.deque = field(default_factory=lambda: collections.deque(maxlen=EXPR_WINDOW))
    texture_hist: collections.deque = field(default_factory=lambda: collections.deque(maxlen=TEXTURE_WINDOW))
    edge_hist: collections.deque = field(default_factory=lambda: collections.deque(maxlen=EDGE_WINDOW))
    flicker_hist: collections.deque = field(default_factory=lambda: collections.deque(maxlen=FLICKER_WINDOW))
    rppg_hist: collections.deque = field(default_factory=lambda: collections.deque(maxlen=FLICKER_WINDOW))
    flow_rigidity_hist: collections.deque = field(default_factory=lambda: collections.deque(maxlen=FLOW_WINDOW))
    pose_parallax_hist: collections.deque = field(default_factory=lambda: collections.deque(maxlen=POSE_WINDOW))
    motion_hist: collections.deque = field(default_factory=lambda: collections.deque(maxlen=MOVE_WINDOW))

    good_quality_frames: int = 0
    stable_face_frames: int = 0
    blink_count: int = 0
    blink_run: int = 0
    baseline_done: bool = False
    ear_baseline: float = 0.0
    last_face_center: np.ndarray = None
    prev_gray_face: np.ndarray = None
    hard_flags: set = field(default_factory=set)
    quality_reason: str = ""
    session_start: float = field(default_factory=time.time)


class ChallengeManager:
    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.total_needed = 2
        self.reset()

    def reset(self):
        pool = ["blink", "turn_left", "turn_right", "open_mouth"]
        self.queue = self.rng.sample(pool, k=self.total_needed)
        self.current = None
        self.current_started = None
        self.current_hold_started = None
        self.completed = 0
        self.failed = 0

    def tick(self, now: float, baseline_ready: bool):
        if not baseline_ready:
            return
        if self.current is None and self.queue:
            self.current = self.queue.pop(0)
            self.current_started = now
            self.current_hold_started = None
        elif self.current is not None and now - self.current_started > CHALLENGE_TIMEOUT:
            self.failed += 1
            self.current = None
            self.current_started = None
            self.current_hold_started = None

    def _hold_check(self, condition: bool, now: float) -> bool:
        if condition:
            if self.current_hold_started is None:
                self.current_hold_started = now
            elif now - self.current_hold_started >= CHALLENGE_MIN_HOLD:
                return True
        else:
            self.current_hold_started = None
        return False

    def check(self, now: float, blinked: bool, nose_offset: float, mouth_ratio: float):
        if self.current is None:
            return
        done = False
        if self.current == "blink":
            done = blinked
        elif self.current == "turn_left":
            done = self._hold_check(nose_offset <= NOSE_TURN_LEFT, now)
        elif self.current == "turn_right":
            done = self._hold_check(nose_offset >= NOSE_TURN_RIGHT, now)
        elif self.current == "open_mouth":
            done = self._hold_check(mouth_ratio >= MOUTH_OPEN_RATIO, now)
        if done:
            self.completed += 1
            self.current = None
            self.current_started = None
            self.current_hold_started = None

    @property
    def done(self):
        return self.completed >= self.total_needed

    @property
    def progress(self):
        return self.completed / max(1, self.total_needed)

    def label(self):
        labels = {
            "blink": "Blink once",
            "turn_left": "Turn head left",
            "turn_right": "Turn head right",
            "open_mouth": "Open mouth",
        }
        if self.current is None:
            return "Done" if self.done else "Wait..."
        return labels.get(self.current, self.current)

# -----------------------------------------------------------------------------
# QUALITY / FEATURES / SCORING
# -----------------------------------------------------------------------------
def evaluate_quality(frame_gray, face_gray, face_bbox, pose_tuple, stable_face_frames, guide_bbox):
    h, w = frame_gray.shape[:2]
    x1, y1, x2, y2 = face_bbox
    face_ratio = ((x2 - x1) * (y2 - y1)) / max(w * h, 1)
    brightness = float(face_gray.mean()) / 255.0
    blur_var = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
    yaw, pitch, roll = pose_tuple
    overlap = bbox_overlap_ratio(face_bbox, guide_bbox)

    if face_ratio < MIN_FACE_RATIO:
        return QualityInfo(False, brightness, blur_var, face_ratio, yaw, pitch, roll, "Move closer")
    if face_ratio > MAX_FACE_RATIO:
        return QualityInfo(False, brightness, blur_var, face_ratio, yaw, pitch, roll, "Move slightly back")
    if brightness < MIN_BRIGHTNESS:
        return QualityInfo(False, brightness, blur_var, face_ratio, yaw, pitch, roll, "More light on face")
    if brightness > MAX_BRIGHTNESS:
        return QualityInfo(False, brightness, blur_var, face_ratio, yaw, pitch, roll, "Reduce strong light")
    if blur_var < MIN_BLUR_VAR:
        return QualityInfo(False, brightness, blur_var, face_ratio, yaw, pitch, roll, "Hold still")
    if abs(yaw) > MAX_ABS_YAW or abs(pitch) > MAX_ABS_PITCH or abs(roll) > MAX_ABS_ROLL:
        return QualityInfo(False, brightness, blur_var, face_ratio, yaw, pitch, roll, "Look at camera")
    if stable_face_frames < MIN_STABLE_FACE_FRAMES:
        return QualityInfo(False, brightness, blur_var, face_ratio, yaw, pitch, roll, "Hold face steady")
    if overlap < 0.40:
        return QualityInfo(False, brightness, blur_var, face_ratio, yaw, pitch, roll, "Center your face")
    return QualityInfo(True, brightness, blur_var, face_ratio, yaw, pitch, roll, "Good")


def compute_optical_flow_rigidity(prev_gray_face, gray_face):
    if prev_gray_face is None:
        return None
    prev = prev_gray_face
    curr = gray_face
    h, w = prev.shape[:2]
    if h < 20 or w < 20:
        return None
    points = cv2.goodFeaturesToTrack(prev, maxCorners=25, qualityLevel=0.01, minDistance=5)
    if points is None or len(points) < 5:
        return None
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, points, None)
    if next_pts is None or status is None:
        return None
    status = status.reshape(-1).astype(bool)
    if status.sum() < 5:
        return None
    p0 = points[status].reshape(-1, 2)
    p1 = next_pts[status].reshape(-1, 2)
    flow = p1 - p0
    mags = np.linalg.norm(flow, axis=1)
    mean_mag = float(np.mean(mags))
    if mean_mag < 0.18:
        return 1.0
    dirs = flow / (mags[:, None] + 1e-6)
    mean_dir = np.linalg.norm(np.mean(dirs, axis=0))
    return float(np.clip(mean_dir, 0.0, 1.0))


def compute_passive_score(state: FeatureState, fps: float):
    hard_flags = set(state.hard_flags)

    blink_score = linear_band_score(state.blink_count, BLINKS_REQUIRED, BLINKS_REQUIRED + 1)

    motion_mean = safe_mean(state.motion_hist)
    motion_score = 0.0
    if MOTION_MIN_PX <= motion_mean <= MOTION_MAX_PX:
        center = (MOTION_MIN_PX + MOTION_MAX_PX) / 2.0
        motion_score = max(0.0, 1.0 - abs(motion_mean - center) / center)

    texture_mean = safe_mean(state.texture_hist)
    texture_score = linear_band_score(texture_mean, TEXTURE_REAL_MIN, TEXTURE_REAL_MIN + 0.20)

    edge_mean = safe_mean(state.edge_hist, default=0.0)
    edge_score = 1.0 - linear_band_score(edge_mean, EDGE_SCREEN_MAX, EDGE_SCREEN_MAX + 160.0)
    if edge_mean > EDGE_SCREEN_MAX + 130:
        hard_flags.add("sharp_edges")

    flow_rigid = safe_mean(state.flow_rigidity_hist, default=1.0)
    flow_score = 1.0 - linear_band_score(flow_rigid, FLOW_RIGIDITY_MAX, 0.99)
    if flow_rigid > 0.975 and len(state.flow_rigidity_hist) >= 8:
        hard_flags.add("flat_motion")

    pose_parallax = safe_mean(state.pose_parallax_hist, default=0.0)
    pose_score = linear_band_score(pose_parallax, POSE_PARALLAX_MIN, POSE_PARALLAX_MIN + 0.030)

    expr_std = max(safe_std(state.mouth_hist), safe_std(state.brow_hist))
    expr_score = linear_band_score(expr_std, EXPR_MIN_STD, EXPR_MIN_STD + 2.0)

    flicker_hz, flicker_amp = dominant_frequency(np.array(state.flicker_hist), fps, FLICKER_MIN_HZ, FLICKER_MAX_HZ)
    if flicker_amp > FLICKER_AMP_THRESH and FLICKER_MIN_HZ <= flicker_hz <= FLICKER_MAX_HZ:
        flicker_score = 0.0
        hard_flags.add("screen_flicker")
    else:
        flicker_score = 1.0 if len(state.flicker_hist) >= 40 else 0.45

    _, rppg_amp = dominant_frequency(np.array(state.rppg_hist), fps, RPPG_MIN_HZ, RPPG_MAX_HZ)
    rppg_score = linear_band_score(rppg_amp, RPPG_MIN_PEAK, RPPG_MIN_PEAK * 2.5)

    components = {
        "blink": blink_score,
        "motion": motion_score,
        "texture": texture_score,
        "edge": edge_score,
        "flow": flow_score,
        "pose": pose_score,
        "expr": expr_score,
        "flicker": flicker_score,
        "rppg": rppg_score,
    }
    weights = {
        "blink": 0.12,
        "motion": 0.10,
        "texture": 0.14,
        "edge": 0.09,
        "flow": 0.15,
        "pose": 0.13,
        "expr": 0.10,
        "flicker": 0.09,
        "rppg": 0.08,
    }
    total = sum(components[k] * weights[k] for k in components)

    return float(np.clip(total, 0.0, 1.0)), components, {
        "motion_mean": motion_mean,
        "texture_mean": texture_mean,
        "edge_mean": edge_mean,
        "flow_rigid": flow_rigid,
        "pose_parallax": pose_parallax,
        "expr_std": expr_std,
        "flicker_hz": flicker_hz,
        "flicker_amp": flicker_amp,
        "rppg_amp": rppg_amp,
        "hard_flags": hard_flags,
    }


def final_decision(face_found, quality_ok, state, challenge: ChallengeManager, passive_score, details):
    hard_flags = details["hard_flags"]
    if not face_found:
        return "NO FACE", (0, 220, 220), "Show your face to camera"
    if not state.baseline_done:
        return "CALIBRATING", (0, 180, 220), "Keep eyes open for 1-2 sec"
    if not quality_ok:
        return "RETRY", (0, 210, 230), state.quality_reason or "Improve camera view"
    if state.good_quality_frames < MIN_GOOD_FRAMES:
        return "CHECKING", (0, 210, 230), "Stay still, collecting signals"

    if "screen_flicker" in hard_flags and passive_score < 0.62:
        return "SPOOF SUSPECT", (20, 40, 240), "Screen / replay pattern detected"
    if len(hard_flags.intersection({"sharp_edges", "flat_motion"})) == 2 and passive_score < 0.58:
        return "SPOOF SUSPECT", (20, 40, 240), "Flat movement detected"

    if challenge.done and passive_score >= ALLOW_SCORE:
        return "LIVE", (0, 220, 0), "Live person confirmed"
    if passive_score <= HARD_BLOCK_SCORE and challenge.failed >= 1:
        return "SPOOF SUSPECT", (20, 40, 240), "Too few live signals"
    if passive_score >= RETRY_SCORE:
        return "RETRY", (0, 210, 230), "Almost there - do the challenge"
    return "SPOOF SUSPECT", (20, 40, 240), "Live confidence too low"


def explain_for_user(verdict, quality_reason, challenge_label, challenge_done, passive_score):
    if verdict == "NO FACE":
        return "Put your face inside the frame"
    if verdict == "CALIBRATING":
        return "Look at camera and keep eyes open"
    if verdict == "CHECKING":
        return challenge_label if not challenge_done else "Hold still for a moment"
    if verdict == "RETRY":
        if challenge_label not in {"Wait...", "Done"}:
            return challenge_label
        return quality_reason or "Try again with better light"
    if verdict == "LIVE":
        return "Access can be allowed"
    if passive_score < 0.35:
        return "Photo / screen / replay may be detected"
    return "Try again from a different angle"

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Liveness detection clear UI")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--session-sec", type=float, default=DEFAULT_SESSION_SEC)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"""
============================================================
LIVENESS DETECTION v{VERSION}  Build {BUILD_DATE}
Clear UI / no file logs / better for weak webcams
Controls: Q quit | R reset
============================================================
""")

    landmarker = build_landmarker()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        log.error("Cannot open camera %s", args.camera)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    state = FeatureState()
    challenge = ChallengeManager()
    fps_buf = collections.deque(maxlen=30)
    prev_time = time.time()

    log.info("Camera opened. Press Q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            log.warning("Frame read failed")
            time.sleep(0.03)
            continue

        now = time.time()
        dt = max(now - prev_time, 1e-6)
        prev_time = now
        fps_buf.append(1.0 / dt)
        fps = float(np.mean(fps_buf)) if fps_buf else 25.0

        h, w = frame.shape[:2]
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        guide_bbox = draw_center_guide(display)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(now * 1000))
        face_found = bool(result.face_landmarks)

        passive_score = 0.0
        components = {}
        details = {"hard_flags": set()}
        verdict = "NO FACE"
        verdict_reason = "Show your face to camera"
        verdict_color = (0, 220, 220)
        blinked_this_frame = False
        mouth_ratio = 1.0
        nose_offset = 0.0
        quality = QualityInfo(False, reason="Show your face to camera")

        state.flicker_hist.append(float(gray.mean()) / 255.0)

        if face_found:
            lm = result.face_landmarks[0]
            face_bbox = face_bbox_from_landmarks(lm, w, h, pad=0.20)
            x1, y1, x2, y2 = face_bbox
            face_gray = gray[y1:y2, x1:x2]
            face_bgr = frame[y1:y2, x1:x2]
            cv2.rectangle(display, (x1, y1), (x2, y2), (50, 220, 210), 2)

            image_points = np.array([
                p2(lm, NOSE_TIP, w, h),
                p2(lm, CHIN, w, h),
                p2(lm, LEFT_EYE_OUTER, w, h),
                p2(lm, RIGHT_EYE_OUTER, w, h),
                p2(lm, LEFT_MOUTH, w, h),
                p2(lm, RIGHT_MOUTH, w, h),
            ], dtype=np.float64)
            pose = estimate_head_pose((h, w), image_points)
            yaw, pitch, roll = pose if pose is not None else (0.0, 0.0, 0.0)

            face_center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
            if state.last_face_center is not None:
                center_shift = np.linalg.norm(face_center - state.last_face_center)
                state.stable_face_frames = state.stable_face_frames + 1 if center_shift < 28.0 else 0
            else:
                state.stable_face_frames = 1
            state.last_face_center = face_center

            quality = evaluate_quality(gray, face_gray, face_bbox, (yaw, pitch, roll), state.stable_face_frames, guide_bbox)
            state.quality_reason = quality.reason

            le = [p2(lm, idx, w, h) for idx in LEFT_EYE]
            re = [p2(lm, idx, w, h) for idx in RIGHT_EYE]
            ear = 0.5 * (eye_aspect_ratio(le) + eye_aspect_ratio(re))

            if not state.baseline_done:
                state.ear_baseline_buf.append(ear)
                if len(state.ear_baseline_buf) >= EAR_BASELINE_FRAMES:
                    state.ear_baseline = float(np.median(state.ear_baseline_buf))
                    state.baseline_done = True
                    log.info("EAR baseline ready: %.4f", state.ear_baseline)

            if state.baseline_done:
                closed_thr = state.ear_baseline * EAR_CLOSED_RATIO
                if ear < closed_thr:
                    state.blink_run += 1
                else:
                    if BLINK_MIN_FRAMES <= state.blink_run <= BLINK_MAX_FRAMES:
                        state.blink_count += 1
                        blinked_this_frame = True
                    state.blink_run = 0

            nose = p2(lm, NOSE_TIP, w, h)
            le_center = 0.5 * (p2(lm, LEFT_EYE_OUTER, w, h) + p2(lm, LEFT_EYE_INNER, w, h))
            re_center = 0.5 * (p2(lm, RIGHT_EYE_OUTER, w, h) + p2(lm, RIGHT_EYE_INNER, w, h))
            eyes_center = 0.5 * (le_center + re_center)
            nose_offset = float(nose[0] - eyes_center[0])
            state.nose_hist.append(nose)

            if len(state.nose_hist) >= 2:
                motion = float(np.linalg.norm(state.nose_hist[-1] - state.nose_hist[-2]))
                state.motion_hist.append(motion)

            mouth_h = dist(p2(lm, UPPER_LIP, w, h), p2(lm, LOWER_LIP, w, h))
            mouth_w = dist(p2(lm, LEFT_MOUTH, w, h), p2(lm, RIGHT_MOUTH, w, h)) + 1e-6
            mouth_ratio = mouth_h / mouth_w
            state.mouth_hist.append(mouth_h)

            brow_left = p2(lm, LEFT_BROW, w, h)
            brow_right = p2(lm, RIGHT_BROW, w, h)
            brow_mid = 0.5 * (brow_left + brow_right)
            brow_delta = float(brow_mid[1] - eyes_center[1])
            state.brow_hist.append(brow_delta)

            if quality.ok:
                state.good_quality_frames += 1

                texture_val = lbp_mean(face_gray)
                state.texture_hist.append(texture_val)
                edge_val = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
                state.edge_hist.append(edge_val)

                flow_rigid = compute_optical_flow_rigidity(state.prev_gray_face, face_gray)
                if flow_rigid is not None:
                    state.flow_rigidity_hist.append(flow_rigid)
                state.prev_gray_face = face_gray.copy()

                inter_eye = dist(le_center, re_center) + 1e-6
                nose_to_eye = dist(nose, eyes_center) / inter_eye
                state.pose_parallax_hist.append(nose_to_eye)

                g = face_bgr[:, :, 1].astype(np.float32) / 255.0
                state.rppg_hist.append(float(np.mean(g)))
            else:
                state.prev_gray_face = None

            sat_ratio = float(np.mean(face_gray > 245))
            if sat_ratio > 0.16:
                state.hard_flags.add("glare")

            challenge.tick(now, state.baseline_done)
            challenge.check(now, blinked_this_frame, nose_offset, mouth_ratio)

            if args.debug:
                for idx in [NOSE_TIP, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, LEFT_MOUTH, RIGHT_MOUTH, CHIN]:
                    pt = p2(lm, idx, w, h).astype(int)
                    cv2.circle(display, tuple(pt), 2, (0, 220, 180), -1)

        if len(state.pose_parallax_hist) >= 2:
            vals = np.array(state.pose_parallax_hist, dtype=np.float32)
            centered = np.abs(vals - np.mean(vals))
            state.pose_parallax_hist = collections.deque(list(centered), maxlen=POSE_WINDOW)

        passive_score, components, details = compute_passive_score(state, fps)
        verdict, verdict_color, verdict_reason = final_decision(
            face_found, quality.ok, state, challenge, passive_score, details
        )

        elapsed = now - state.session_start
        remaining = max(0.0, args.session_sec - elapsed)
        helper_text = explain_for_user(verdict, quality.reason, challenge.label(), challenge.done, passive_score)

        if elapsed >= args.session_sec:
            log.info(
                "Session result: verdict=%s score=%.3f challenge=%s/%s flags=%s",
                verdict, passive_score, challenge.completed, challenge.total_needed,
                sorted(list(details.get("hard_flags", set())))
            )
            state = FeatureState()
            challenge.reset()

        # UI
        panel_w = 380
        cv2.rectangle(display, (w - panel_w, 0), (w, h), (12, 12, 16), -1)
        cv2.rectangle(display, (0, 0), (w, 88), (10, 10, 14), -1)

        overlay_text(display, f"LIVENESS v{VERSION} | {verdict}", 16, 34, verdict_color, 0.95, 2)
        overlay_text(display, helper_text, 16, 66, (235, 235, 235), 0.72, 2)

        px = w - panel_w + 16
        y = 30
        overlay_text(display, f"FPS: {fps:4.1f}", px, y, (180, 180, 195), 0.52, 1)
        overlay_text(display, f"Time left: {remaining:4.1f}s", px + 150, y, (180, 180, 195), 0.52, 1)

        y += 34
        overlay_text(display, f"Live score: {passive_score * 100:5.1f}%", px, y, (220, 220, 220), 0.60, 1)
        draw_bar(display, px, y + 8, passive_score, verdict_color, width=panel_w - 34, height=12)

        y += 40
        q_progress = min(1.0, state.good_quality_frames / max(MIN_GOOD_FRAMES, 1))
        qcol = (0, 200, 0) if quality.ok else (0, 200, 220)
        overlay_text(display, f"Camera view: {quality.reason}", px, y, qcol, 0.55, 1)
        draw_bar(display, px, y + 8, q_progress, qcol, width=panel_w - 34, height=10)

        y += 38
        calib = min(1.0, len(state.ear_baseline_buf) / max(EAR_BASELINE_FRAMES, 1))
        ccol = (0, 200, 0) if state.baseline_done else (0, 180, 220)
        overlay_text(display, f"Eye baseline: {'ready' if state.baseline_done else f'{calib * 100:.0f}%'}", px, y, ccol, 0.55, 1)
        draw_bar(display, px, y + 8, 1.0 if state.baseline_done else calib, ccol, width=panel_w - 34, height=10)

        y += 38
        overlay_text(display, f"Challenge: {challenge.label()}", px, y, (220, 220, 220), 0.55, 1)
        draw_bar(display, px, y + 8, challenge.progress, (30, 180, 250), width=panel_w - 34, height=10)
        overlay_text(display, f"Done {challenge.completed}/{challenge.total_needed} | Fail {challenge.failed}", px, y + 28, (170, 170, 180), 0.45, 1)

        y += 58
        metric_lines = [
            ("Blinks", f"{state.blink_count}"),
            ("Motion", f"{details.get('motion_mean', 0.0):.2f}"),
            ("Texture", f"{details.get('texture_mean', 0.0):.3f}"),
            ("Edge", f"{details.get('edge_mean', 0.0):.1f}"),
            ("Flow rigid", f"{details.get('flow_rigid', 0.0):.3f}"),
            ("Parallax", f"{details.get('pose_parallax', 0.0):.3f}"),
            ("Expr std", f"{details.get('expr_std', 0.0):.3f}"),
            ("rPPG amp", f"{details.get('rppg_amp', 0.0):.4f}"),
            ("Flicker", f"{details.get('flicker_hz', 0.0):.1f} / {details.get('flicker_amp', 0.0):.4f}"),
        ]
        for name, val in metric_lines:
            overlay_text(display, f"{name:10s}: {val}", px, y, (195, 195, 205), 0.47, 1)
            y += 21

        y += 6
        flags_text = ", ".join(sorted(list(details.get("hard_flags", set())))) or "none"
        overlay_text(display, f"Flags: {flags_text}", px, min(y, h - 16), (220, 180, 80), 0.45, 1)

        overlay_text(display, "Q quit | R reset", 14, h - 14, (150, 150, 165), 0.45, 1)
        cv2.imshow(f"Liveness Detection v{VERSION}", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            log.info("Manual reset")
            state = FeatureState()
            challenge.reset()

    cap.release()
    cv2.destroyAllWindows()
    log.info("Done")


if __name__ == "__main__":
    main()
