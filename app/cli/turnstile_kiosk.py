"""Windows edge turnstile kiosk runtime."""
from __future__ import annotations

import argparse
import threading
import time

import cv2

from app.domain.enums import ControllerVerdict, TurnstileState
from app.infrastructure.api.dependencies import set_container
from app.infrastructure.config import get_settings
from app.infrastructure.container import Container
from app.infrastructure.logging_setup import setup_logging


def _draw_text(img, text, x, y, color=(230, 230, 230), scale=0.7, thickness=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_progress(img, x, y, value, width, height, color):
    value = max(0.0, min(float(value), 1.0))
    cv2.rectangle(img, (x, y), (x + width, y + height), (45, 45, 50), -1)
    cv2.rectangle(img, (x, y), (x + int(width * value), y + height), color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (80, 80, 90), 1)


def _state_color(state, controller_verdict):
    if controller_verdict == ControllerVerdict.ACCESS_GRANTED:
        return (0, 255, 0)
    if controller_verdict == ControllerVerdict.ACCESS_DENIED:
        return (0, 0, 255)
    if state == TurnstileState.POSITIONING:
        return (0, 200, 220)
    if state == TurnstileState.ANALYZING:
        return (0, 220, 255)
    return (0, 200, 220)


def _start_control_plane(container: Container, host: str, port: int):
    try:
        import uvicorn
    except ImportError:
        return None, None, "uvicorn not installed"

    from app.main import create_app

    set_container(container)
    app = create_app()
    config = uvicorn.Config(app=app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(0.2)
    return server, thread, None


def main():
    parser = argparse.ArgumentParser(description="Turnstile kiosk runtime")
    parser.add_argument("--camera", type=int, default=None, help="Camera index override")
    parser.add_argument("--width", type=int, default=None, help="Camera width override")
    parser.add_argument("--height", type=int, default=None, help="Camera height override")
    parser.add_argument("--no-control-plane", action="store_true", help="Disable embedded local API")
    args = parser.parse_args()

    setup_logging()
    settings = get_settings().model_copy(deep=True)
    if args.camera is not None:
        settings.camera.index = args.camera
    if args.width is not None:
        settings.camera.width = args.width
    if args.height is not None:
        settings.camera.height = args.height

    container = Container(settings)
    if not container.is_ready:
        print(f"NOT READY: {container.asset_error}")
        return

    engine = container.turnstile_engine
    control_plane_enabled = settings.turnstile.control_plane_enabled and not args.no_control_plane
    server = None
    server_thread = None
    control_plane_error = None
    if control_plane_enabled:
        server, server_thread, control_plane_error = _start_control_plane(
            container,
            settings.turnstile.control_plane_host,
            settings.turnstile.control_plane_port,
        )

    cap = cv2.VideoCapture(settings.camera.index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera.height)
    if not cap.isOpened():
        print(f"Cannot open camera {settings.camera.index}")
        return

    print("TURNSTILE KIOSK")
    print("Q quit | R reset")
    if control_plane_enabled and control_plane_error is None:
        print(f"Control plane: http://{settings.turnstile.control_plane_host}:{settings.turnstile.control_plane_port}")
    elif control_plane_error is not None:
        print(f"Control plane disabled: {control_plane_error}")

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue

        decision = engine.process_frame(frame)
        display = frame.copy()
        h, w = display.shape[:2]
        color = _state_color(decision.state, decision.controller_verdict)

        guide_x1 = int(w * 0.20)
        guide_y1 = int(h * 0.16)
        guide_x2 = int(w * 0.56)
        guide_y2 = int(h * 0.84)
        cv2.rectangle(display, (guide_x1, guide_y1), (guide_x2, guide_y2), (90, 90, 100), 2)

        bbox = decision.details.get("face_bbox")
        if bbox:
            cv2.rectangle(
                display,
                (int(bbox["x"]), int(bbox["y"])),
                (int(bbox["x2"]), int(bbox["y2"])),
                (0, 220, 220),
                2,
            )

        cv2.rectangle(display, (0, 0), (w, 100), (10, 10, 14), -1)
        title = decision.state.value
        if decision.controller_verdict is not None and decision.state == TurnstileState.COOLDOWN:
            title = f"COOLDOWN | {decision.controller_verdict.value}"
        _draw_text(display, "TURNSTILE EDGE RUNTIME", 18, 40, (0, 220, 255), 0.95, 2)
        _draw_text(display, title, 18, 80, color, 1.0, 3)
        _draw_text(display, decision.reason, 420, 50, (230, 230, 235), 0.78, 2)

        panel_x1 = w - 340
        panel_x2 = w - 20
        cv2.rectangle(display, (panel_x1, 120), (panel_x2, h - 20), (12, 12, 16), -1)

        cx = panel_x1 + 160
        radius = 42
        red_color = (0, 0, 90)
        green_color = (0, 90, 0)
        if decision.controller_verdict == ControllerVerdict.ACCESS_GRANTED:
            green_color = (0, 255, 0)
            red_color = (0, 0, 60)
        elif decision.controller_verdict == ControllerVerdict.ACCESS_DENIED:
            red_color = (0, 0, 255)
            green_color = (0, 60, 0)

        cv2.circle(display, (cx, 220), radius, red_color, -1)
        cv2.circle(display, (cx, 340), radius, green_color, -1)
        cv2.circle(display, (cx, 220), radius, (80, 80, 80), 2)
        cv2.circle(display, (cx, 340), radius, (80, 80, 80), 2)
        _draw_text(display, "RED", cx - 30, 285, (210, 210, 215), 0.7, 2)
        _draw_text(display, "GREEN", cx - 48, 405, (210, 210, 215), 0.7, 2)

        _draw_text(display, f"Session: {decision.session_id[:8]}", panel_x1 + 20, 460, (230, 230, 235), 0.60, 1)
        _draw_text(display, f"Confidence: {decision.confidence * 100:5.1f}%", panel_x1 + 20, 490, (230, 230, 235), 0.65, 2)
        _draw_progress(display, panel_x1 + 20, 505, decision.confidence, 250, 12, color)
        _draw_text(display, f"Fusion score: {decision.live_score:0.3f}", panel_x1 + 20, 540, (230, 230, 235), 0.55, 1)
        _draw_text(display, f"Heuristic: {decision.heuristic_score:0.3f}", panel_x1 + 20, 565, (180, 180, 190), 0.55, 1)
        _draw_text(
            display,
            f"Deep learning: {decision.deep_learning_score if decision.deep_learning_score is not None else 'n/a'}",
            panel_x1 + 20,
            590,
            (180, 180, 190),
            0.55,
            1,
        )
        flags = ", ".join(decision.reason_codes) or "none"
        _draw_text(display, f"Reason codes: {flags}", panel_x1 + 20, 625, (120, 200, 255), 0.50, 1)
        metrics = decision.details.get("metrics", {})
        _draw_text(display, f"Quality: {decision.details.get('quality_reason', 'n/a')}", panel_x1 + 20, 650, (220, 220, 100), 0.50, 1)
        _draw_text(display, f"Flow: {metrics.get('flow', 0.0):0.3f}", panel_x1 + 20, 675, (170, 170, 180), 0.48, 1)
        _draw_text(display, f"Parallax: {metrics.get('parallax', 0.0):0.4f}", panel_x1 + 20, 695, (170, 170, 180), 0.48, 1)

        footer = "Q quit | R reset"
        if control_plane_enabled and control_plane_error is None:
            footer += f" | API http://{settings.turnstile.control_plane_host}:{settings.turnstile.control_plane_port}"
        _draw_text(display, footer, 18, h - 18, (170, 170, 180), 0.50, 1)

        cv2.imshow("Turnstile Edge Runtime", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            engine.reset()

    cap.release()
    cv2.destroyAllWindows()
    if server is not None:
        server.should_exit = True
    if server_thread is not None:
        server_thread.join(timeout=2.0)
    set_container(None)


if __name__ == "__main__":
    main()
