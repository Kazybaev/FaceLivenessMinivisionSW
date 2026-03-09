# -*- coding: utf-8 -*-
# Liveness Detection — Real person vs Photo/Screen
#
# Checks:
#   1. Blinks         — photo never blinks
#   2. Head movement  — INDEPENDENT from eye movement (photo moves together)
#   3. Nose/Eye ratio — on photo nose and eyes move in sync, on real person they differ
#   4. Skin texture   — photo/screen is too smooth (low Laplacian variance)
#
# Install: pip install opencv-python mediapipe numpy

import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import os
from collections import deque

# ─── SETTINGS (strict) ─────────────────────────────────────────────────────
BLINKS_NEEDED     = 3      # blinks required to confirm

# Head movement — strict
MOVES_NEEDED      = 10     # independent head movements required
MOVE_PIXELS       = 16     # min pixels to count as movement
MOVE_MAX_JUMP     = 35     # max jump (above = camera shake, ignored)
SMOOTH_WINDOW     = 6      # nose smoothing frames

# Nose vs Eye ratio check
# On a PHOTO: nose and eyes move together (ratio ~1.0)
# On a REAL person: head turns, nose moves MORE than eye corners
RATIO_DIFF_MIN    = 0.25   # nose must move at least 25% MORE than eyes
RATIO_CHECKS_MIN  = 6      # how many frames must pass this check

# Skin texture
TEXTURE_MIN       = 90.0   # Laplacian variance — below = too smooth = photo
TEXTURE_FRAMES    = 20     # average over N frames

# ALL 4 checks must pass for REAL
# (blink + movement + ratio + texture)

# Landmark indices
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
NOSE_TIP   = 1
# Eye corners for ratio check
L_CORNER   = 263   # left eye outer corner
R_CORNER   = 33    # right eye outer corner


# ─── Download model ────────────────────────────────────────────────────────
def get_landmarker():
    model_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        print("Downloading face_landmarker.task (~2 MB)...")
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        urllib.request.urlretrieve(url, model_path)
        print("Done.")
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(opts)


# ─── Eye Aspect Ratio ──────────────────────────────────────────────────────
def calc_ear(pts):
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)


# ─── Draw progress bar ─────────────────────────────────────────────────────
def draw_bar(img, x, y, ratio, color, width=200, height=10):
    cv2.rectangle(img, (x, y), (x+width, y+height), (45,45,45), -1)
    filled = int(width * min(max(ratio, 0), 1.0))
    if filled > 0:
        cv2.rectangle(img, (x, y), (x+filled, y+height), color, -1)
    cv2.rectangle(img, (x, y), (x+width, y+height), (80,80,80), 1)


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    print("Starting...")
    landmarker = get_landmarker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found!")
        return

    # State
    blink_count    = 0
    blink_frames   = 0
    move_count     = 0
    ratio_ok_count = 0
    ear_buf        = deque(maxlen=10)
    nose_buf       = deque(maxlen=SMOOTH_WINDOW)
    leye_buf       = deque(maxlen=SMOOTH_WINDOW)
    reye_buf       = deque(maxlen=SMOOTH_WINDOW)
    texture_buf    = deque(maxlen=TEXTURE_FRAMES)

    print("Ready!  Q = quit,  R = reset")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w    = frame.shape[:2]
        display = frame.copy()

        # Detect
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)

        face_found  = bool(result.face_landmarks)
        current_ear = 0.25

        # ── Skin texture (whole frame gray) ────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if face_found:
            lm = result.face_landmarks[0]

            def pt(i):
                return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)

            # ── 1. EAR / Blink ─────────────────────────────────────────────
            left_pts  = np.array([pt(i) for i in LEFT_EYE])
            right_pts = np.array([pt(i) for i in RIGHT_EYE])
            current_ear = (calc_ear(left_pts) + calc_ear(right_pts)) / 2.0
            ear_buf.append(current_ear)

            if current_ear < 0.22:
                blink_frames += 1
            else:
                if blink_frames >= 2:
                    blink_count += 1
                blink_frames = 0

            # ── 2. Head movement (smoothed, strict) ────────────────────────
            nose_raw = pt(NOSE_TIP)
            lcor_raw = pt(L_CORNER)
            rcor_raw = pt(R_CORNER)

            nose_buf.append(nose_raw)
            leye_buf.append(lcor_raw)
            reye_buf.append(rcor_raw)

            if len(nose_buf) == SMOOTH_WINDOW:
                s_nose = np.mean(nose_buf, axis=0)
                s_leye = np.mean(leye_buf, axis=0)
                s_reye = np.mean(reye_buf, axis=0)

                nose_dist = float(np.linalg.norm(s_nose - nose_buf[0]))
                leye_dist = float(np.linalg.norm(s_leye - leye_buf[0]))
                reye_dist = float(np.linalg.norm(s_reye - reye_buf[0]))
                eye_dist  = (leye_dist + reye_dist) / 2.0

                # Valid movement: not camera shake
                if MOVE_PIXELS < nose_dist < MOVE_MAX_JUMP:
                    move_count += 1

                # ── 3. Nose/Eye ratio check ─────────────────────────────────
                # On photo: camera shakes → nose and eyes move the same amount
                # On real person: head turns → nose moves differently than eye corners
                if eye_dist > 2.0:  # avoid division by near-zero
                    ratio = nose_dist / (eye_dist + 1e-6)
                    # Real person: nose moves relatively more (head rotation)
                    # Photo: ratio is ~1.0 (everything moves together)
                    if abs(ratio - 1.0) > RATIO_DIFF_MIN:
                        ratio_ok_count += 1

            # ── 4. Skin texture ────────────────────────────────────────────
            # Crop face region
            xs = [int(lm[i].x * w) for i in range(len(lm))]
            ys = [int(lm[i].y * h) for i in range(len(lm))]
            fx1 = max(0, min(xs)-10); fx2 = min(w, max(xs)+10)
            fy1 = max(0, min(ys)-10); fy2 = min(h, max(ys)+10)
            face_roi = gray[fy1:fy2, fx1:fx2]
            if face_roi.size > 100:
                lap = float(cv2.Laplacian(face_roi, cv2.CV_64F).var())
                texture_buf.append(lap)

            # Draw eye dots
            for i in LEFT_EYE + RIGHT_EYE:
                cv2.circle(display,
                           (int(lm[i].x * w), int(lm[i].y * h)),
                           2, (0, 255, 180), -1)
            cv2.circle(display,
                       (int(lm[NOSE_TIP].x * w), int(lm[NOSE_TIP].y * h)),
                       4, (255, 200, 0), -1)

        # ── Evaluate all checks ────────────────────────────────────────────
        avg_texture  = float(np.mean(texture_buf)) if texture_buf else 0.0

        blink_ok   = blink_count    >= BLINKS_NEEDED
        move_ok    = move_count     >= MOVES_NEEDED
        ratio_ok   = ratio_ok_count >= RATIO_CHECKS_MIN
        texture_ok = avg_texture    >= TEXTURE_MIN

        # ALL must pass
        checks_passed = sum([blink_ok, move_ok, ratio_ok, texture_ok])
        is_real = checks_passed == 4

        # ── Verdict ────────────────────────────────────────────────────────
        if not face_found:
            verdict      = "NO FACE"
            verdict_col  = (0, 220, 220)
        elif is_real:
            verdict      = "REAL PERSON"
            verdict_col  = (0, 220, 0)
        else:
            verdict      = "PHOTO / SCREEN"
            verdict_col  = (30, 30, 220)

        # ── Draw UI ────────────────────────────────────────────────────────
        # Top bar
        cv2.rectangle(display, (0, 0), (w, 54), (10,10,10), -1)
        cv2.putText(display, verdict, (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, verdict_col, 2, cv2.LINE_AA)

        # Right panel
        px = w - 250
        cv2.rectangle(display, (px, 54), (w, h), (10,10,20), -1)

        def txt(text, y, color=(200,200,200), scale=0.48, bold=False):
            cv2.putText(display, text, (px+12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                        2 if bold else 1, cv2.LINE_AA)

        txt("LIVENESS CHECK", 82, (140,140,220), 0.54, bold=True)
        cv2.line(display, (px+8, 90), (w-8, 90), (55,55,80), 1)

        # Check 1: Blinks
        c1 = (0,210,0) if blink_ok else (80,80,200)
        txt(f"1. Blinks: {blink_count}/{BLINKS_NEEDED}", 114, c1, 0.50)
        draw_bar(display, px+12, 120, blink_count/BLINKS_NEEDED, c1, width=220)

        cv2.line(display, (px+8, 138), (w-8, 138), (40,40,60), 1)

        # Check 2: Movement
        c2 = (0,210,0) if move_ok else (80,80,200)
        txt(f"2. Head move: {move_count}/{MOVES_NEEDED}", 158, c2, 0.50)
        draw_bar(display, px+12, 164, move_count/MOVES_NEEDED, c2, width=220)

        cv2.line(display, (px+8, 182), (w-8, 182), (40,40,60), 1)

        # Check 3: Nose/Eye ratio
        c3 = (0,210,0) if ratio_ok else (80,80,200)
        txt(f"3. Face depth: {ratio_ok_count}/{RATIO_CHECKS_MIN}", 202, c3, 0.50)
        draw_bar(display, px+12, 208, ratio_ok_count/RATIO_CHECKS_MIN, c3, width=220)
        txt("(nose vs eyes move diff)", 224, (100,100,120), 0.36)

        cv2.line(display, (px+8, 232), (w-8, 232), (40,40,60), 1)

        # Check 4: Texture
        c4 = (0,210,0) if texture_ok else (80,80,200)
        txt(f"4. Skin texture: {avg_texture:.0f}", 252, c4, 0.50)
        draw_bar(display, px+12, 258, avg_texture/200.0, c4, width=220)
        txt(f"(min={TEXTURE_MIN:.0f}, photo=low)", 274, (100,100,120), 0.36)

        cv2.line(display, (px+8, 282), (w-8, 282), (40,40,60), 1)

        # Score
        score_col = (0,200,0) if checks_passed >= 3 else (200,140,0) if checks_passed >= 2 else (200,60,60)
        txt(f"Score: {checks_passed}/4", 304, score_col, 0.54, bold=True)

        cv2.line(display, (px+8, 316), (w-8, 316), (40,40,60), 1)

        # Final
        txt("RESULT:", 338, (180,180,180), 0.46)
        txt(verdict, 362, verdict_col, 0.58, bold=True)

        cv2.line(display, (px+8, 374), (w-8, 374), (40,40,60), 1)

        # Hint — what's missing
        if not face_found:
            txt("Look at camera", h-40, (120,120,150), 0.42)
        elif is_real:
            txt("Confirmed!", h-40, (0,200,0), 0.50, bold=True)
        else:
            missing = []
            if not blink_ok:   missing.append(f"Blink {BLINKS_NEEDED}x")
            if not move_ok:    missing.append("Move head")
            if not ratio_ok:   missing.append("Turn head slowly")
            if not texture_ok: missing.append("Too smooth=photo")
            if missing:
                txt(missing[0], h-40, (140,140,180), 0.42)

        txt("Q-quit   R-reset", h-14, (65,65,85), 0.37)

        cv2.imshow("Liveness Detection", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key == ord('r'):
            blink_count    = 0
            blink_frames   = 0
            move_count     = 0
            ratio_ok_count = 0
            ear_buf.clear()
            nose_buf.clear()
            leye_buf.clear()
            reye_buf.clear()
            texture_buf.clear()
            print("Reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
