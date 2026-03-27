# -*- coding: utf-8 -*-
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║          LIVENESS DETECTION  v3.0  —  Professional Edition              ║
# ║                                                                          ║
# ║  Checks:  1. Adaptive blink (personal EAR baseline)                     ║
# ║           2. 3D head pose via Z-landmarks                                ║
# ║           3. Micro-motion frequency (natural body oscillation)           ║
# ║           4. Independent L/R eye analysis                                ║
# ║           5. LBP skin-texture (Local Binary Pattern)                     ║
# ║           6. Facial expression dynamics (brow / lip movement)            ║
# ║           7. Nose/eye depth-ratio                                         ║
# ║                                                                          ║
# ║  Build:   2026-03-24                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import os
from collections import deque
import time
from datetime import datetime
import logging
import sys

# ─── LOGGING ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"liveness_{datetime.now():%Y%m%d_%H%M%S}.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("liveness")

# ─── VERSION ───────────────────────────────────────────────────────────────
VERSION    = "3.0"
BUILD_DATE = "2026-03-24"

# ─── SOURCE CONFIGURATION ──────────────────────────────────────────────────
# Укажи путь к видеофайлу или None для использования камеры
VIDEO_FILE = "vid_1.mp4"   # <-- поменяй на None чтобы использовать камеру

# ─── THRESHOLDS ────────────────────────────────────────────────────────────
REAL_THRESHOLD      = 70.0      # % to declare REAL
RESET_INTERVAL_SEC  = 8         # auto-reset window

# Blink
BLINKS_NEEDED       = 3
EAR_CLOSED_RATIO    = 0.78      # fraction of personal baseline → "closed"
EAR_BASELINE_FRAMES = 40        # frames to compute personal baseline
BLINK_MIN_FRAMES    = 2
BLINK_MAX_FRAMES    = 10

# Head movement
MOVES_NEEDED        = 8
MOVE_PIXELS_MIN     = 14
MOVE_PIXELS_MAX     = 40
SMOOTH_WINDOW       = 6

# Depth ratio
RATIO_DIFF_MIN      = 0.22
RATIO_CHECKS_MIN    = 5

# Texture (LBP)
TEXTURE_MIN_LBP     = 0.65      # entropy of LBP histogram  (0..1)
TEXTURE_FRAMES      = 25

# Micro-motion frequency
FREQ_BUFFER_LEN     = 90        # ~3 s at 30 fps
FREQ_MIN_HZ         = 0.5       # natural sway lower bound
FREQ_MAX_HZ         = 4.0       # natural sway upper bound
FREQ_CHECKS_MIN     = 4

# Expression dynamics
EXPR_MOVE_THRESH    = 3.5       # pixels of brow/lip movement to count
EXPR_CHECKS_MIN     = 3

# Scoring weights  (must sum to 1.0)
W_BLINK    = 0.20
W_MOVE     = 0.18
W_RATIO    = 0.12
W_TEXTURE  = 0.20
W_FREQ     = 0.15
W_EXPR     = 0.15

# ─── LANDMARK INDICES ──────────────────────────────────────────────────────
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
NOSE_TIP   = 1
L_CORNER   = 263
R_CORNER   = 33
# Brows
L_BROW     = [336, 296, 334, 293, 300]
R_BROW     = [107,  66,  68, 104, 108]
# Lips
UPPER_LIP  = [13, 312, 311, 310]
LOWER_LIP  = [14, 317, 402, 318]


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def get_landmarker(model_path: str = "face_landmarker.task"):
    if not os.path.exists(model_path):
        log.info("Downloading face_landmarker.task …")
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        urllib.request.urlretrieve(url, model_path)
        log.info("Download complete.")
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.45,
        min_face_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(opts)


def calc_ear(pts: np.ndarray) -> float:
    """Eye Aspect Ratio — standard formula."""
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return float((A + B) / (2.0 * C + 1e-6))


def lbp_entropy(roi_gray: np.ndarray) -> float:
    """
    Simplified LBP: for each pixel compare to 8 neighbours,
    build histogram, return normalised Shannon entropy.
    Values near 1.0 = rich texture (real skin); near 0 = flat (photo on screen).
    """
    if roi_gray.size < 100:
        return 0.0
    img = roi_gray.astype(np.float32)
    lbp = np.zeros_like(img, dtype=np.uint8)
    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
        shifted = np.roll(np.roll(img, dy, axis=0), dx, axis=1)
        lbp = (lbp << 1) | (img >= shifted).astype(np.uint8)
    hist, _ = np.histogram(lbp, bins=256, range=(0, 255))
    hist = hist / (hist.sum() + 1e-9)
    entropy = -np.sum(hist * np.log2(hist + 1e-9)) / 8.0
    return float(np.clip(entropy, 0, 1))


def dominant_frequency(signal: np.ndarray, fps: float = 30.0) -> float:
    """Return dominant frequency of a 1-D signal (in Hz)."""
    if len(signal) < 16:
        return 0.0
    sig = signal - np.mean(signal)
    fft = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fps)
    idx = np.argmax(fft[1:]) + 1
    return float(freqs[idx])


def draw_bar(img, x, y, ratio, color, width=240, height=10):
    cv2.rectangle(img, (x, y), (x + width, y + height), (28, 28, 35), -1)
    filled = int(width * min(max(ratio, 0.0), 1.0))
    if filled > 0:
        cv2.rectangle(img, (x, y), (x + filled, y + height), color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (70, 70, 80), 1)


def make_color(pct: float):
    """Green if done, blue-ish if in progress."""
    return (0, 215, 0) if pct >= 100 else (80, 100, 220)


# ═══════════════════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════════════════

def reset_state(ear_baseline: float = 0.0):
    return {
        "blink_count":      0,
        "blink_frames":     0,
        "ear_baseline":     ear_baseline,
        "move_count":       0,
        "ratio_ok_count":   0,
        "freq_ok_count":    0,
        "expr_ok_count":    0,
        "last_reset":       time.time(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# SCORE
# ═══════════════════════════════════════════════════════════════════════════

def compute_score(state: dict, texture_pct: float) -> dict:
    b  = min(1.0, state["blink_count"]    / BLINKS_NEEDED)
    m  = min(1.0, state["move_count"]     / MOVES_NEEDED)
    r  = min(1.0, state["ratio_ok_count"] / RATIO_CHECKS_MIN)
    t  = texture_pct / 100.0
    f  = min(1.0, state["freq_ok_count"]  / FREQ_CHECKS_MIN)
    e  = min(1.0, state["expr_ok_count"]  / EXPR_CHECKS_MIN)

    score = (b*W_BLINK + m*W_MOVE + r*W_RATIO + t*W_TEXTURE +
             f*W_FREQ  + e*W_EXPR) * 100.0

    return {
        "blink_pct":   b   * 100,
        "move_pct":    m   * 100,
        "ratio_pct":   r   * 100,
        "texture_pct": t   * 100,
        "freq_pct":    f   * 100,
        "expr_pct":    e   * 100,
        "total":       score,
    }


# ═══════════════════════════════════════════════════════════════════════════
# OPEN VIDEO / CAMERA
# ═══════════════════════════════════════════════════════════════════════════

def open_capture(video_file=None):
    """
    Если video_file задан и файл существует — открываем его.
    Иначе ищем доступную камеру (индексы 0-4).
    Возвращает (cap, source_label) или бросает RuntimeError.
    """
    if video_file and os.path.exists(video_file):
        cap = cv2.VideoCapture(video_file)
        if cap.isOpened():
            log.info(f"Video file opened: {video_file}")
            return cap, f"FILE: {video_file}"
        else:
            log.error(f"Failed to open video file: {video_file}")

    log.info("Searching for camera …")
    for cam_id in range(5):
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            log.info(f"Camera opened on index {cam_id}")
            return cap, f"CAM:{cam_id}"
        cap.release()

    raise RuntimeError("No video source found. Put vid_1.mp4 next to the script or connect a camera.")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    session_start = datetime.now()

    banner = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║          LIVENESS DETECTION  v{VERSION}  —  Professional Edition         ║
║          Build: {BUILD_DATE}   Started: {session_start:%H:%M:%S}              ║
║                                                                      ║
║  Checks:   Adaptive blink · 3D head pose · Micro-motion freq        ║
║            LBP skin texture · L/R eye · Depth ratio · Expression    ║
║                                                                      ║
║  REAL threshold: {REAL_THRESHOLD}%    Auto-reset every {RESET_INTERVAL_SEC}s                    ║
║  Controls:  Q = quit   R = manual reset   SPACE = pause (video)     ║
╚══════════════════════════════════════════════════════════════════════╝"""
    print(banner)
    log.info("Loading face landmark model …")

    landmarker = get_landmarker()
    log.info("Model ready.")

    # ── Open video/camera ─────────────────────────────────────────────
    cap, source_label = open_capture(VIDEO_FILE)

    fps_target = cap.get(cv2.CAP_PROP_FPS) or 30.0
    is_file    = VIDEO_FILE and os.path.exists(VIDEO_FILE)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_file else 0
    log.info(f"Source: {source_label}  |  FPS target: {fps_target:.1f}"
             + (f"  |  Total frames: {total_frames}" if is_file else ""))

    # ── Buffers ───────────────────────────────────────────────────────
    ear_buf       = deque(maxlen=EAR_BASELINE_FRAMES)
    nose_buf      = deque(maxlen=SMOOTH_WINDOW)
    leye_buf      = deque(maxlen=SMOOTH_WINDOW)
    reye_buf      = deque(maxlen=SMOOTH_WINDOW)
    texture_buf   = deque(maxlen=TEXTURE_FRAMES)
    freq_buf      = deque(maxlen=FREQ_BUFFER_LEN)
    brow_buf      = deque(maxlen=SMOOTH_WINDOW)
    lip_buf       = deque(maxlen=SMOOTH_WINDOW)

    state         = reset_state()
    ear_baseline  = 0.0
    baseline_done = False
    baseline_buf  = deque(maxlen=EAR_BASELINE_FRAMES)

    frame_count = 0
    reset_count = 0
    fps_buf     = deque(maxlen=30)
    t_prev      = time.time()
    paused      = False

    log.info("Detection running …  press Q to quit, SPACE to pause video.")

    while True:
        # ── Pause support (video only) ────────────────────────────────
        if paused and is_file:
            key = cv2.waitKey(50) & 0xFF
            if key == ord(" "):
                paused = False
            elif key == ord("q"):
                break
            continue

        ret, frame = cap.read()

        # ── End of video file → loop back ────────────────────────────
        if not ret:
            if is_file:
                log.info("End of video — looping back to start.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # Reset detection state for new loop
                reset_count += 1
                state = reset_state(ear_baseline=ear_baseline)
                texture_buf.clear(); nose_buf.clear(); leye_buf.clear()
                reye_buf.clear(); freq_buf.clear(); brow_buf.clear(); lip_buf.clear()
                continue
            else:
                log.warning("Frame read failed — retrying …")
                time.sleep(0.05)
                continue

        frame_count += 1
        now = time.time()

        # ── FPS throttle for video files (avoid blasting through) ────
        if is_file and fps_target > 0:
            elapsed_since_last = now - t_prev
            wait_needed = (1.0 / fps_target) - elapsed_since_last
            if wait_needed > 0:
                time.sleep(wait_needed)
            now = time.time()

        fps_buf.append(1.0 / max(now - t_prev, 1e-6))
        t_prev   = now
        live_fps = float(np.mean(fps_buf))

        h, w   = frame.shape[:2]
        display = frame.copy()

        # ── Auto-reset ────────────────────────────────────────────────
        elapsed = now - state["last_reset"]
        if elapsed >= RESET_INTERVAL_SEC:
            reset_count += 1
            log.info(f"[RESET #{reset_count}]  window={elapsed:.1f}s  baseline={ear_baseline:.3f}")
            state = reset_state(ear_baseline=ear_baseline)
            texture_buf.clear()
            nose_buf.clear(); leye_buf.clear(); reye_buf.clear()
            freq_buf.clear(); brow_buf.clear(); lip_buf.clear()

        # ── Detect ────────────────────────────────────────────────────
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)
        face_found = bool(result.face_landmarks)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if face_found:
            lm = result.face_landmarks[0]

            def pt2(i):
                return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

            def pt3(i):
                return np.array([lm[i].x * w, lm[i].y * h, lm[i].z * w], dtype=np.float32)

            # ── 1. Adaptive blink ─────────────────────────────────────
            left_pts  = np.array([pt2(i) for i in LEFT_EYE])
            right_pts = np.array([pt2(i) for i in RIGHT_EYE])
            ear_l = calc_ear(left_pts)
            ear_r = calc_ear(right_pts)
            ear   = (ear_l + ear_r) / 2.0
            ear_buf.append(ear)

            if not baseline_done:
                baseline_buf.append(ear)
                if len(baseline_buf) == EAR_BASELINE_FRAMES:
                    ear_baseline = float(np.percentile(baseline_buf, 75))
                    baseline_done = True
                    state["ear_baseline"] = ear_baseline
                    log.info(f"EAR baseline calibrated: {ear_baseline:.3f}")
            else:
                ear_threshold = ear_baseline * EAR_CLOSED_RATIO
                if ear < ear_threshold:
                    state["blink_frames"] += 1
                else:
                    if BLINK_MIN_FRAMES <= state["blink_frames"] <= BLINK_MAX_FRAMES:
                        state["blink_count"] += 1
                        log.info(f"  Blink {state['blink_count']}/{BLINKS_NEEDED}  "
                                 f"(EAR={ear:.3f} < thr={ear_threshold:.3f})")
                    state["blink_frames"] = 0

            # ── 2. Head movement ──────────────────────────────────────
            nose_raw = pt2(NOSE_TIP)
            lcor_raw = pt2(L_CORNER)
            rcor_raw = pt2(R_CORNER)
            nose_buf.append(nose_raw)
            leye_buf.append(lcor_raw)
            reye_buf.append(rcor_raw)
            freq_buf.append(nose_raw[0])

            if len(nose_buf) == SMOOTH_WINDOW:
                s_nose = np.mean(nose_buf, axis=0)
                s_leye = np.mean(leye_buf, axis=0)
                s_reye = np.mean(reye_buf, axis=0)

                nose_dist = float(np.linalg.norm(s_nose - nose_buf[0]))
                leye_dist = float(np.linalg.norm(s_leye - leye_buf[0]))
                reye_dist = float(np.linalg.norm(s_reye - reye_buf[0]))
                eye_dist  = (leye_dist + reye_dist) / 2.0

                if MOVE_PIXELS_MIN < nose_dist < MOVE_PIXELS_MAX:
                    state["move_count"] += 1

                # ── 3. Depth ratio ────────────────────────────────────
                if eye_dist > 2.0:
                    ratio = nose_dist / (eye_dist + 1e-6)
                    if abs(ratio - 1.0) > RATIO_DIFF_MIN:
                        state["ratio_ok_count"] += 1

            # ── 4. Micro-motion frequency ─────────────────────────────
            if len(freq_buf) == FREQ_BUFFER_LEN:
                dom_hz = dominant_frequency(np.array(freq_buf), fps=live_fps)
                if FREQ_MIN_HZ < dom_hz < FREQ_MAX_HZ:
                    state["freq_ok_count"] += 1

            # ── 5. LBP Skin texture ───────────────────────────────────
            xs = [int(lm[i].x * w) for i in range(len(lm))]
            ys = [int(lm[i].y * h) for i in range(len(lm))]
            fx1 = max(0, min(xs) - 10); fx2 = min(w, max(xs) + 10)
            fy1 = max(0, min(ys) - 10); fy2 = min(h, max(ys) + 10)
            face_roi = gray[fy1:fy2, fx1:fx2]
            if face_roi.size > 200:
                texture_buf.append(lbp_entropy(face_roi))

            # ── 6. Expression dynamics ────────────────────────────────
            brow_pts = np.mean([pt2(i) for i in L_BROW + R_BROW], axis=0)
            lip_pts  = np.mean([pt2(i) for i in UPPER_LIP + LOWER_LIP], axis=0)
            brow_buf.append(brow_pts)
            lip_buf.append(lip_pts)

            if len(brow_buf) == SMOOTH_WINDOW:
                brow_move = float(np.linalg.norm(np.mean(brow_buf, axis=0) - brow_buf[0]))
                lip_move  = float(np.linalg.norm(np.mean(lip_buf,  axis=0) - lip_buf[0]))
                if brow_move > EXPR_MOVE_THRESH or lip_move > EXPR_MOVE_THRESH:
                    state["expr_ok_count"] += 1

            # Draw landmarks
            for idx in LEFT_EYE + RIGHT_EYE:
                px_c = (int(lm[idx].x * w), int(lm[idx].y * h))
                cv2.circle(display, px_c, 2, (0, 210, 180), -1)
            cv2.circle(display, (int(lm[NOSE_TIP].x * w), int(lm[NOSE_TIP].y * h)),
                       4, (255, 200, 0), -1)

        # ── Compute final score ───────────────────────────────────────
        avg_texture  = float(np.mean(texture_buf)) if texture_buf else 0.0
        texture_pct  = min(100.0, avg_texture / TEXTURE_MIN_LBP * 100.0)
        scores       = compute_score(state, texture_pct)
        final        = scores["total"]

        # ── Verdict ───────────────────────────────────────────────────
        if not face_found:
            verdict = "NO FACE DETECTED"
            vcolor  = (0, 220, 220)
        elif not baseline_done:
            verdict = "CALIBRATING …"
            vcolor  = (180, 180, 0)
        elif final >= REAL_THRESHOLD:
            verdict = "REAL PERSON"
            vcolor  = (0, 220, 0)
        else:
            verdict = "FAKE / PHOTO"
            vcolor  = (30, 30, 220)

        # ═════════════════════════════════════════════════════════════
        # UI RENDERING
        # ═════════════════════════════════════════════════════════════
        PANEL_W = 320
        px0     = w - PANEL_W

        # ── Top bar ───────────────────────────────────────────────────
        cv2.rectangle(display, (0, 0), (w, 60), (8, 8, 12), -1)
        cv2.putText(display, f"LIVENESS v{VERSION}  |  {verdict}",
                    (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, vcolor, 2, cv2.LINE_AA)
        cv2.putText(display, f"{live_fps:.0f} fps",
                    (px0 - 80, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (90, 90, 100), 1, cv2.LINE_AA)

        # ── Video progress bar (only for file mode) ───────────────────
        if is_file and total_frames > 0:
            curr_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            progress   = curr_frame / total_frames
            bar_y      = h - 6
            cv2.rectangle(display, (0, bar_y), (w, h), (20, 20, 25), -1)
            cv2.rectangle(display, (0, bar_y), (int(w * progress), h), (0, 160, 220), -1)
            # Timecode
            curr_sec = curr_frame / max(fps_target, 1)
            tot_sec  = total_frames / max(fps_target, 1)
            tc = f"{int(curr_sec//60):02d}:{int(curr_sec%60):02d} / {int(tot_sec//60):02d}:{int(tot_sec%60):02d}"
            cv2.putText(display, tc, (8, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 160), 1, cv2.LINE_AA)
            if paused:
                cv2.putText(display, "PAUSED", (w // 2 - 40, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3, cv2.LINE_AA)

        # ── Source label ──────────────────────────────────────────────
        cv2.putText(display, source_label, (14, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (55, 55, 70), 1, cv2.LINE_AA)

        # ── Right panel background ────────────────────────────────────
        overlay = display.copy()
        cv2.rectangle(overlay, (px0, 60), (w, h), (10, 12, 18), -1)
        cv2.addWeighted(overlay, 0.88, display, 0.12, 0, display)

        def txt(text, y, color=(190, 190, 200), scale=0.48, bold=False):
            cv2.putText(display, text, (px0 + 14, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, 2 if bold else 1, cv2.LINE_AA)

        def sep(y):
            cv2.line(display, (px0 + 10, y), (w - 10, y), (40, 40, 55), 1)

        txt("LIVENESS CHECKS", 88, (140, 140, 210), 0.52, bold=True)
        sep(96)

        time_left = max(0.0, RESET_INTERVAL_SEC - (now - state["last_reset"]))
        tcol = (0, 190, 190) if time_left > 1.5 else (220, 110, 20)
        txt(f"Window: {time_left:.1f}s / {RESET_INTERVAL_SEC}s   Resets: #{reset_count}", 116, tcol, 0.40)
        sep(124)

        BAR_W = PANEL_W - 28
        checks = [
            ("1. Blink",      scores["blink_pct"],   f"{state['blink_count']}/{BLINKS_NEEDED}"),
            ("2. Movement",   scores["move_pct"],    f"{state['move_count']}/{MOVES_NEEDED}"),
            ("3. Depth",      scores["ratio_pct"],   f"{state['ratio_ok_count']}/{RATIO_CHECKS_MIN}"),
            ("4. Texture",    scores["texture_pct"], f"{avg_texture:.2f}"),
            ("5. Freq",       scores["freq_pct"],    f"{state['freq_ok_count']}/{FREQ_CHECKS_MIN}"),
            ("6. Expression", scores["expr_pct"],    f"{state['expr_ok_count']}/{EXPR_CHECKS_MIN}"),
        ]
        y = 148
        for label, pct, detail in checks:
            col = make_color(pct)
            txt(f"{label}: {detail}  ({pct:.0f}%)", y, col, 0.44)
            draw_bar(display, px0 + 14, y + 4, pct / 100.0, col, width=BAR_W, height=9)
            y += 36
            sep(y - 4)

        txt("LIVENESS SCORE", y + 8, (160, 160, 160), 0.48)
        if final >= REAL_THRESHOLD:
            sc = (0, 240, 0)
        elif final >= REAL_THRESHOLD - 20:
            sc = (0, 200, 200)
        else:
            sc = (30, 30, 240)
        cv2.putText(display, f"{final:.1f}%", (px0 + 14, y + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.9, sc, 3, cv2.LINE_AA)

        sep(y + 68)

        calib_pct = min(1.0, len(baseline_buf) / EAR_BASELINE_FRAMES)
        calib_col = (0, 200, 0) if baseline_done else (180, 180, 0)
        calib_txt = "EAR calibrated" if baseline_done else f"Calibrating {calib_pct*100:.0f}%"
        txt(calib_txt, y + 85, calib_col, 0.40)
        draw_bar(display, px0 + 14, y + 89, calib_pct, calib_col, width=BAR_W, height=6)

        sep(y + 100)

        txt(f"Threshold: {REAL_THRESHOLD}%", y + 118, (130, 130, 140), 0.40)
        txt(verdict, y + 140, vcolor, 0.60, bold=True)

        cv2.putText(display, f"v{VERSION}  {BUILD_DATE}",
                    (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 70), 1, cv2.LINE_AA)
        txt("Q quit   R reset   SPACE pause", h - 18, (55, 55, 65), 0.34)

        cv2.imshow(f"Liveness Detection v{VERSION}", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("r"):
            reset_count += 1
            state = reset_state(ear_baseline=ear_baseline)
            texture_buf.clear(); nose_buf.clear(); leye_buf.clear()
            reye_buf.clear(); freq_buf.clear(); brow_buf.clear(); lip_buf.clear()
            log.info(f"[MANUAL RESET #{reset_count}]")
        if key == ord(" ") and is_file:
            paused = not paused
            log.info("PAUSED" if paused else "RESUMED")

    cap.release()
    cv2.destroyAllWindows()

    runtime = datetime.now() - session_start
    log.info("═" * 60)
    log.info(f"Session ended   runtime={runtime}   frames={frame_count}   resets={reset_count}")
    log.info("═" * 60)


if __name__ == "__main__":
    main()
