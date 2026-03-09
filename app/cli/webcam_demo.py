"""CLI: real-time webcam liveness detection with heuristic checks.

Replaces: liveness_detection.py
Usage: python -m app.cli.webcam_demo
"""
from __future__ import annotations

import cv2
import numpy as np
from collections import deque

from app.infrastructure.config import get_settings
from app.infrastructure.logging_setup import setup_logging
from app.adapters.detectors.mediapipe_detector import MediaPipeDetector
from app.adapters.analyzers.heuristic_analyzer import HeuristicAnalyzer, calc_ear


def draw_bar(img, x, y, ratio, color, width=200, height=10):
    cv2.rectangle(img, (x, y), (x + width, y + height), (45, 45, 45), -1)
    filled = int(width * min(max(ratio, 0), 1.0))
    if filled > 0:
        cv2.rectangle(img, (x, y), (x + filled, y + height), color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (80, 80, 80), 1)


def main():
    setup_logging()
    settings = get_settings()
    cfg = settings.analyzer.heuristic

    print("Starting...")
    detector = MediaPipeDetector(settings.detector.mediapipe)
    analyzer = HeuristicAnalyzer(cfg)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not found!")
        return

    print("Ready!  Q = quit,  R = reset")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        display = frame.copy()

        face = detector.detect(frame)
        face_found = face is not None

        if face_found:
            result = analyzer.process_frame(frame, face)
            state = analyzer._state
            checks = result.details.get("checks", [])

            # Draw eye dots
            if face.landmarks is not None:
                for i in cfg.left_eye + cfg.right_eye:
                    pt = face.landmarks[i].astype(int)
                    cv2.circle(display, tuple(pt), 2, (0, 255, 180), -1)
                nose_pt = face.landmarks[cfg.nose_tip].astype(int)
                cv2.circle(display, tuple(nose_pt), 4, (255, 200, 0), -1)
        else:
            result = None
            state = analyzer._state

        # Evaluate
        avg_texture = result.details.get("avg_texture", 0.0) if result else 0.0
        blink_ok = state.blink_count >= cfg.blinks_needed
        move_ok = state.move_count >= cfg.moves_needed
        ratio_ok = state.ratio_ok_count >= cfg.ratio_checks_min
        texture_ok = avg_texture >= cfg.texture_min
        checks_passed = sum([blink_ok, move_ok, ratio_ok, texture_ok])
        is_real = checks_passed == 4

        if not face_found:
            verdict = "NO FACE"
            verdict_col = (0, 220, 220)
        elif is_real:
            verdict = "REAL PERSON"
            verdict_col = (0, 220, 0)
        else:
            verdict = "PHOTO / SCREEN"
            verdict_col = (30, 30, 220)

        # Draw UI
        cv2.rectangle(display, (0, 0), (w, 54), (10, 10, 10), -1)
        cv2.putText(display, verdict, (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, verdict_col, 2, cv2.LINE_AA)

        px = w - 250
        cv2.rectangle(display, (px, 54), (w, h), (10, 10, 20), -1)

        def txt(text, y, color=(200, 200, 200), scale=0.48, bold=False):
            cv2.putText(display, text, (px + 12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                        2 if bold else 1, cv2.LINE_AA)

        txt("LIVENESS CHECK", 82, (140, 140, 220), 0.54, bold=True)
        cv2.line(display, (px + 8, 90), (w - 8, 90), (55, 55, 80), 1)

        c1 = (0, 210, 0) if blink_ok else (80, 80, 200)
        txt(f"1. Blinks: {state.blink_count}/{cfg.blinks_needed}", 114, c1, 0.50)
        draw_bar(display, px + 12, 120, state.blink_count / cfg.blinks_needed, c1, width=220)
        cv2.line(display, (px + 8, 138), (w - 8, 138), (40, 40, 60), 1)

        c2 = (0, 210, 0) if move_ok else (80, 80, 200)
        txt(f"2. Head move: {state.move_count}/{cfg.moves_needed}", 158, c2, 0.50)
        draw_bar(display, px + 12, 164, state.move_count / cfg.moves_needed, c2, width=220)
        cv2.line(display, (px + 8, 182), (w - 8, 182), (40, 40, 60), 1)

        c3 = (0, 210, 0) if ratio_ok else (80, 80, 200)
        txt(f"3. Face depth: {state.ratio_ok_count}/{cfg.ratio_checks_min}", 202, c3, 0.50)
        draw_bar(display, px + 12, 208, state.ratio_ok_count / cfg.ratio_checks_min, c3, width=220)
        txt("(nose vs eyes move diff)", 224, (100, 100, 120), 0.36)
        cv2.line(display, (px + 8, 232), (w - 8, 232), (40, 40, 60), 1)

        c4 = (0, 210, 0) if texture_ok else (80, 80, 200)
        txt(f"4. Skin texture: {avg_texture:.0f}", 252, c4, 0.50)
        draw_bar(display, px + 12, 258, avg_texture / 200.0, c4, width=220)
        txt(f"(min={cfg.texture_min:.0f}, photo=low)", 274, (100, 100, 120), 0.36)
        cv2.line(display, (px + 8, 282), (w - 8, 282), (40, 40, 60), 1)

        score_col = ((0, 200, 0) if checks_passed >= 3
                     else (200, 140, 0) if checks_passed >= 2
                     else (200, 60, 60))
        txt(f"Score: {checks_passed}/4", 304, score_col, 0.54, bold=True)
        cv2.line(display, (px + 8, 316), (w - 8, 316), (40, 40, 60), 1)

        txt("RESULT:", 338, (180, 180, 180), 0.46)
        txt(verdict, 362, verdict_col, 0.58, bold=True)
        cv2.line(display, (px + 8, 374), (w - 8, 374), (40, 40, 60), 1)

        if not face_found:
            txt("Look at camera", h - 40, (120, 120, 150), 0.42)
        elif is_real:
            txt("Confirmed!", h - 40, (0, 200, 0), 0.50, bold=True)
        else:
            missing = []
            if not blink_ok:
                missing.append(f"Blink {cfg.blinks_needed}x")
            if not move_ok:
                missing.append("Move head")
            if not ratio_ok:
                missing.append("Turn head slowly")
            if not texture_ok:
                missing.append("Too smooth=photo")
            if missing:
                txt(missing[0], h - 40, (140, 140, 180), 0.42)

        txt("Q-quit   R-reset", h - 14, (65, 65, 85), 0.37)

        cv2.imshow("Liveness Detection", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        if key == ord('r'):
            analyzer.reset()
            print("Reset.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
