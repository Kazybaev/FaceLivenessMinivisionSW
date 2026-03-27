
# -*- coding: utf-8 -*-
"""
Turnstile Liveness Fast - green/red access light

Goal:
- Person looks at camera for a short moment
- If likely real live face -> GREEN / ACCESS GRANTED
- If likely phone/photo/video/spoof -> RED / ACCESS DENIED
- No blink challenge, no long interaction

Install:
    pip install opencv-python mediapipe numpy

Run:
    python liveness_turnstile_green_red.py
    python liveness_turnstile_green_red.py --camera 0
    python liveness_turnstile_green_red.py --debug
"""

import argparse
import collections
import logging
import math
import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

# -----------------------------------------------------------------------------
# Logging: console only, no file logs
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("turnstile_liveness")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
VERSION = "1.0"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_PATH = "face_landmarker.task"

WINDOW_TITLE = "Turnstile Liveness"

# Fast decision
DECISION_WINDOW = 1.2          # seconds
COOLDOWN_SEC = 1.2             # keep green/red light visible
MIN_GOOD_FRAMES = 8

# Quality gate (soft, for weak cameras)
MIN_FACE_RATIO = 0.08
MAX_FACE_RATIO = 0.58
MIN_BRIGHTNESS = 0.14
MAX_BRIGHTNESS = 0.95
MIN_BLUR_VAR = 25.0

# Anti-spoof heuristics
TEXTURE_REAL_MIN = 0.30
FLOW_RIGIDITY_SPOOF = 0.94
PARALLAX_LIVE_MIN = 0.0035
MOTION_MIN = 0.45
MOTION_MAX = 14.0
EDGE_SCREEN_HIGH = 240.0
FLICKER_MIN_HZ = 7.0
FLICKER_MAX_HZ = 35.0
FLICKER_AMP_SPOOF = 0.010

# Final thresholds
LIVE_SCORE_THR = 0.62
SPOOF_SCORE_THR = 0.42

# MediaPipe landmarks
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
NOSE_TIP = 1
CHIN = 152
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

FACE_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -63.6, -12.5),    # Chin
    (-43.3, 32.7, -26.0),   # Left eye outer
    (43.3, 32.7, -26.0),    # Right eye outer
    (-28.9, -28.9, -24.1),  # Left mouth
    (28.9, -28.9, -24.1),   # Right mouth
], dtype=np.float64)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def ensure_model(path: str = MODEL_PATH) -> str:
    if os.path.exists(path):
        return path
    log.info("Downloading MediaPipe model...")
    urllib.request.urlretrieve(MODEL_URL, path)
    log.info("Model saved: %s", path)
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


def face_bbox_from_landmarks(lm, w, h, pad=0.18):
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


def linear_band_score(x, low, high):
    if x <= low:
        return 0.0
    if x >= high:
        return 1.0
    return float((x - low) / max(high - low, 1e-6))


def draw_text(img, text, x, y, color=(230, 230, 230), scale=0.7, thickness=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_progress(img, x, y, value, width, height, color):
    value = float(np.clip(value, 0.0, 1.0))
    cv2.rectangle(img, (x, y), (x + width, y + height), (45, 45, 50), -1)
    cv2.rectangle(img, (x, y), (x + int(width * value), y + height), color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (80, 80, 90), 1)


def compute_flow_rigidity(prev_gray_face, gray_face):
    if prev_gray_face is None:
        return None
    prev = prev_gray_face
    curr = gray_face
    if prev.shape[0] < 20 or prev.shape[1] < 20:
        return None
    pts = cv2.goodFeaturesToTrack(prev, maxCorners=30, qualityLevel=0.01, minDistance=5)
    if pts is None or len(pts) < 6:
        return None
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, pts, None)
    if next_pts is None:
        return None
    status = status.reshape(-1).astype(bool)
    if status.sum() < 6:
        return None
    p0 = pts[status].reshape(-1, 2)
    p1 = next_pts[status].reshape(-1, 2)
    flow = p1 - p0
    mags = np.linalg.norm(flow, axis=1)
    mean_mag = float(np.mean(mags))
    if mean_mag < 0.2:
        return 1.0
    dirs = flow / (mags[:, None] + 1e-6)
    mean_dir = np.linalg.norm(np.mean(dirs, axis=0))
    return float(np.clip(mean_dir, 0.0, 1.0))


class SessionState:
    def __init__(self):
        self.start_ts = time.time()
        self.last_face_center = None
        self.prev_gray_face = None
        self.good_frames = 0
        self.texture_hist = collections.deque(maxlen=40)
        self.edge_hist = collections.deque(maxlen=40)
        self.motion_hist = collections.deque(maxlen=25)
        self.parallax_hist = collections.deque(maxlen=25)
        self.flow_hist = collections.deque(maxlen=25)
        self.flicker_hist = collections.deque(maxlen=60)
        self.flags = set()

    def reset(self):
        self.__init__()


def compute_scores(state: SessionState, fps: float):
    texture = float(np.mean(state.texture_hist)) if state.texture_hist else 0.0
    edge = float(np.mean(state.edge_hist)) if state.edge_hist else 0.0
    motion = float(np.mean(state.motion_hist)) if state.motion_hist else 0.0
    parallax = float(np.mean(state.parallax_hist)) if state.parallax_hist else 0.0
    flow = float(np.mean(state.flow_hist)) if state.flow_hist else 1.0
    flicker_hz, flicker_amp = dominant_frequency(
        np.array(state.flicker_hist, dtype=np.float32), fps, FLICKER_MIN_HZ, FLICKER_MAX_HZ
    )

    texture_score = linear_band_score(texture, TEXTURE_REAL_MIN, TEXTURE_REAL_MIN + 0.22)

    motion_score = 0.0
    if MOTION_MIN <= motion <= MOTION_MAX:
        center = (MOTION_MIN + MOTION_MAX) / 2.0
        motion_score = max(0.0, 1.0 - abs(motion - center) / center)

    parallax_score = linear_band_score(parallax, PARALLAX_LIVE_MIN, PARALLAX_LIVE_MIN + 0.02)
    flow_score = 1.0 - linear_band_score(flow, FLOW_RIGIDITY_SPOOF, 0.995)

    edge_score = 1.0
    if edge > EDGE_SCREEN_HIGH:
        edge_score = max(0.0, 1.0 - (edge - EDGE_SCREEN_HIGH) / 220.0)

    flicker_score = 1.0
    if flicker_amp > FLICKER_AMP_SPOOF and FLICKER_MIN_HZ <= flicker_hz <= FLICKER_MAX_HZ:
        flicker_score = 0.0
        state.flags.add("screen_flicker")

    if flow > 0.97 and len(state.flow_hist) >= 8:
        state.flags.add("flat_motion")
    if edge > EDGE_SCREEN_HIGH + 90:
        state.flags.add("sharp_edges")

    live_score = (
        0.26 * texture_score +
        0.18 * motion_score +
        0.22 * parallax_score +
        0.22 * flow_score +
        0.06 * edge_score +
        0.06 * flicker_score
    )
    live_score = float(np.clip(live_score, 0.0, 1.0))

    return live_score, {
        "texture": texture,
        "motion": motion,
        "parallax": parallax,
        "flow": flow,
        "edge": edge,
        "flicker_hz": flicker_hz,
        "flicker_amp": flicker_amp,
        "flags": set(state.flags),
    }


def quality_check(gray, face_gray, face_bbox):
    h, w = gray.shape[:2]
    x1, y1, x2, y2 = face_bbox
    face_ratio = ((x2 - x1) * (y2 - y1)) / max(w * h, 1)
    brightness = float(face_gray.mean()) / 255.0
    blur_var = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())

    if face_ratio < MIN_FACE_RATIO:
        return False, "Move closer"
    if face_ratio > MAX_FACE_RATIO:
        return False, "Move a little back"
    if brightness < MIN_BRIGHTNESS:
        return False, "More light on face"
    if brightness > MAX_BRIGHTNESS:
        return False, "Too much light"
    if blur_var < MIN_BLUR_VAR:
        return False, "Hold still"
    return True, "Good"


def main():
    parser = argparse.ArgumentParser(description="Fast turnstile liveness detector")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(f"\nTURNSTILE LIVENESS v{VERSION}\nGreen = real person | Red = spoof\nQ quit | R reset\n")

    landmarker = build_landmarker()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        log.error("Cannot open camera %s", args.camera)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    state = SessionState()
    fps_buf = collections.deque(maxlen=30)
    prev_time = time.time()

    locked_result = None
    locked_color = (0, 200, 220)
    locked_reason = "Look at camera"
    lock_until = 0.0

    log.info("Camera opened. Q quit, R reset.")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue

        now = time.time()
        dt = max(now - prev_time, 1e-6)
        prev_time = now
        fps_buf.append(1.0 / dt)
        fps = float(np.mean(fps_buf)) if fps_buf else 30.0

        h, w = frame.shape[:2]
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ROI guide
        gx1, gy1 = int(w * 0.20), int(h * 0.18)
        gx2, gy2 = int(w * 0.55), int(h * 0.82)
        cv2.rectangle(display, (gx1, gy1), (gx2, gy2), (90, 90, 100), 2)

        state.flicker_hist.append(float(gray.mean()) / 255.0)

        face_found = False
        quality_ok = False
        quality_reason = "Look at camera"
        metrics = {
            "texture": 0.0, "motion": 0.0, "parallax": 0.0, "flow": 1.0,
            "edge": 0.0, "flicker_hz": 0.0, "flicker_amp": 0.0, "flags": set()
        }
        live_score = 0.0

        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(now * 1000))
        if result.face_landmarks:
            face_found = True
            lm = result.face_landmarks[0]
            bbox = face_bbox_from_landmarks(lm, w, h, pad=0.20)
            x1, y1, x2, y2 = bbox
            face_gray = gray[y1:y2, x1:x2]
            face_bgr = frame[y1:y2, x1:x2]

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 220), 2)

            quality_ok, quality_reason = quality_check(gray, face_gray, bbox)

            nose = p2(lm, NOSE_TIP, w, h)
            le = 0.5 * (p2(lm, LEFT_EYE_OUTER, w, h) + p2(lm, LEFT_EYE_INNER, w, h))
            re = 0.5 * (p2(lm, RIGHT_EYE_OUTER, w, h) + p2(lm, RIGHT_EYE_INNER, w, h))
            eyes_center = 0.5 * (le + re)

            if state.last_face_center is not None:
                motion = float(np.linalg.norm(nose - state.last_face_center))
                state.motion_hist.append(motion)
            state.last_face_center = nose.copy()

            if quality_ok:
                state.good_frames += 1

                texture_val = lbp_mean(face_gray)
                state.texture_hist.append(texture_val)

                edge_val = float(cv2.Laplacian(face_gray, cv2.CV_64F).var())
                state.edge_hist.append(edge_val)

                flow_rigid = compute_flow_rigidity(state.prev_gray_face, face_gray)
                if flow_rigid is not None:
                    state.flow_hist.append(flow_rigid)
                state.prev_gray_face = face_gray.copy()

                inter_eye = dist(le, re) + 1e-6
                nose_to_eye = dist(nose, eyes_center) / inter_eye
                state.parallax_hist.append(nose_to_eye)

                # simple glare heuristic
                if float(np.mean(face_gray > 245)) > 0.13:
                    state.flags.add("glare")

                if args.debug:
                    for idx in [NOSE_TIP, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, LEFT_MOUTH, RIGHT_MOUTH, CHIN]:
                        pt = p2(lm, idx, w, h).astype(int)
                        cv2.circle(display, tuple(pt), 2, (0, 255, 0), -1)
            else:
                state.prev_gray_face = None

        # Normalize parallax dynamics
        if len(state.parallax_hist) >= 2:
            vals = np.array(state.parallax_hist, dtype=np.float32)
            dyn = np.abs(vals - np.mean(vals))
            state.parallax_hist = collections.deque(list(dyn)[-25:], maxlen=25)

        live_score, metrics = compute_scores(state, fps)

        decision = "CHECKING"
        decision_color = (0, 200, 220)
        reason = quality_reason if face_found else "No face"

        # Hold decision light for a short time
        if now < lock_until and locked_result is not None:
            decision = locked_result
            decision_color = locked_color
            reason = locked_reason
        else:
            elapsed = now - state.start_ts

            if not face_found:
                decision = "NO FACE"
                decision_color = (0, 200, 220)
                reason = "Look at camera"
            elif not quality_ok:
                decision = "CHECKING"
                decision_color = (0, 200, 220)
                reason = quality_reason
            elif elapsed < DECISION_WINDOW or state.good_frames < MIN_GOOD_FRAMES:
                decision = "CHECKING"
                decision_color = (0, 200, 220)
                reason = "Analyzing face"
            else:
                flags = metrics.get("flags", set())
                strong_spoof = (
                    ("screen_flicker" in flags) or
                    ("flat_motion" in flags and metrics["parallax"] < PARALLAX_LIVE_MIN * 0.8) or
                    (metrics["flow"] > 0.985 and metrics["texture"] < TEXTURE_REAL_MIN)
                )
                if strong_spoof or live_score <= SPOOF_SCORE_THR:
                    decision = "ACCESS DENIED"
                    decision_color = (0, 0, 255)
                    reason = "Spoof suspected"
                elif live_score >= LIVE_SCORE_THR:
                    decision = "ACCESS GRANTED"
                    decision_color = (0, 255, 0)
                    reason = "Real person"
                else:
                    decision = "ACCESS DENIED"
                    decision_color = (0, 0, 255)
                    reason = "Not enough live evidence"

                locked_result = decision
                locked_color = decision_color
                locked_reason = reason
                lock_until = now + COOLDOWN_SEC
                state.reset()

        # Top bar
        cv2.rectangle(display, (0, 0), (w, 100), (10, 10, 14), -1)
        draw_text(display, f"TURNSTILE LIVENESS v{VERSION}", 18, 45, (0, 220, 255), 1.1, 3)
        draw_text(display, decision, 18, 85, decision_color, 1.0, 3)
        draw_text(display, reason, 420, 52, (230, 230, 235), 0.8, 2)

        # Traffic light on right
        panel_x1 = w - 320
        panel_x2 = w - 20
        cv2.rectangle(display, (panel_x1, 120), (panel_x2, h - 20), (12, 12, 16), -1)

        cx = panel_x1 + 150
        r = 42
        red_col = (0, 0, 90)
        green_col = (0, 90, 0)

        if decision == "ACCESS GRANTED":
            green_col = (0, 255, 0)
            red_col = (0, 0, 60)
        elif decision == "ACCESS DENIED":
            red_col = (0, 0, 255)
            green_col = (0, 60, 0)

        cv2.circle(display, (cx, 220), r, red_col, -1)
        cv2.circle(display, (cx, 340), r, green_col, -1)
        cv2.circle(display, (cx, 220), r, (80, 80, 80), 2)
        cv2.circle(display, (cx, 340), r, (80, 80, 80), 2)
        draw_text(display, "RED", cx - 30, 285, (210, 210, 215), 0.7, 2)
        draw_text(display, "GREEN", cx - 48, 405, (210, 210, 215), 0.7, 2)

        # Info
        draw_text(display, f"Live score: {live_score * 100:5.1f}%", panel_x1 + 20, 470, (230, 230, 235), 0.7, 2)
        draw_progress(display, panel_x1 + 20, 490, live_score, 240, 14, decision_color)
        draw_text(display, f"Face view: {quality_reason}", panel_x1 + 20, 535, (220, 220, 100), 0.65, 2)
        draw_text(display, f"FPS: {fps:4.1f}", panel_x1 + 20, 575, (180, 180, 190), 0.6, 1)
        draw_text(display, f"Good frames: {state.good_frames}", panel_x1 + 20, 605, (180, 180, 190), 0.6, 1)

        flags_text = ", ".join(sorted(metrics.get("flags", set()))) or "none"
        draw_text(display, f"Flags: {flags_text}", panel_x1 + 20, 635, (120, 200, 255), 0.55, 1)

        if args.debug:
            debug_lines = [
                f"texture: {metrics['texture']:.3f}",
                f"motion: {metrics['motion']:.2f}",
                f"parallax: {metrics['parallax']:.4f}",
                f"flow: {metrics['flow']:.3f}",
                f"edge: {metrics['edge']:.1f}",
                f"flicker: {metrics['flicker_hz']:.1f} / {metrics['flicker_amp']:.4f}",
            ]
            yy = 140
            for line in debug_lines:
                draw_text(display, line, 18, yy, (220, 220, 220), 0.55, 1)
                yy += 26

        draw_text(display, "Q quit | R reset", 18, h - 18, (170, 170, 180), 0.55, 1)

        cv2.imshow(WINDOW_TITLE, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            state.reset()
            locked_result = None
            lock_until = 0.0
            log.info("Manual reset")

    cap.release()
    cv2.destroyAllWindows()
    log.info("Done")


if __name__ == "__main__":
    main()
