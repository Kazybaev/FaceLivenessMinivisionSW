"""Visible preview runtime for the continuous access-control MVP."""
from __future__ import annotations

import argparse
import threading
import time

import cv2
import numpy as np

from app.config import Settings, get_settings
from app.core.pipeline import AccessControlPipeline
from app.infrastructure.logging_setup import setup_logging
from app.main import create_app


def _draw_text(img, text, x, y, color=(230, 230, 230), scale=0.7, thickness=2):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _start_api_server(settings: Settings, pipeline: AccessControlPipeline):
    try:
        import uvicorn
    except ImportError:
        return None, None, "uvicorn not installed"

    app = create_app(settings)
    app.state.access_pipeline = pipeline
    config = uvicorn.Config(app=app, host=settings.api.host, port=settings.api.port, log_level="info")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(0.4)
    return server, thread, None


def _placeholder_frame(settings: Settings, message: str) -> np.ndarray:
    frame = np.zeros((settings.runtime.preview_height, settings.runtime.preview_width, 3), dtype=np.uint8)
    _draw_text(frame, "ACCESS CONTROL PREVIEW", 28, 54, (0, 220, 255), 1.0, 2)
    _draw_text(frame, message, 28, 108, (220, 220, 220), 0.8, 2)
    return frame


def _draw_bbox(img, bbox: dict[str, int], color, label: str | None = None) -> None:
    cv2.rectangle(img, (bbox["x"], bbox["y"]), (bbox["x2"], bbox["y2"]), color, 2)
    if label:
        _draw_text(img, label, bbox["x"], max(22, bbox["y"] - 8), color, 0.55, 2)


def _render_preview(frame: np.ndarray, snapshot: dict[str, object], status: dict[str, object]) -> np.ndarray:
    display = frame.copy()
    height, width = display.shape[:2]
    cv2.rectangle(display, (0, 0), (width, 150), (8, 8, 12), -1)

    face = snapshot.get("tracked_face")
    if isinstance(face, dict):
        _draw_bbox(display, face, (0, 220, 220), "face")

    suspicious_boxes = {
        (
            item["bbox"]["x"],
            item["bbox"]["y"],
            item["bbox"]["x2"],
            item["bbox"]["y2"],
        )
        for item in snapshot.get("suspicious_objects", [])
        if isinstance(item, dict)
    }
    for item in snapshot.get("objects", []):
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox")
        if not isinstance(bbox, dict):
            continue
        key = (bbox["x"], bbox["y"], bbox["x2"], bbox["y2"])
        color = (0, 0, 255) if key in suspicious_boxes else (70, 170, 255)
        label = f'{item["label"]} {float(item["confidence"]) * 100:0.0f}%'
        _draw_bbox(display, bbox, color, label)

    decision = snapshot.get("decision") if isinstance(snapshot.get("decision"), dict) else None
    anti_spoof = snapshot.get("anti_spoof_result") if isinstance(snapshot.get("anti_spoof_result"), dict) else None
    session_state = str(snapshot.get("session_state", "idle")).upper()
    if decision and decision.get("verdict") == "deny":
        session_state = "FAKE"
    elif decision and decision.get("verdict") == "allow":
        session_state = "REAL"
    elif bool(snapshot.get("blocked_by_suspicious_object")):
        session_state = "FAKE"
    blocked_reason = str(snapshot.get("blocked_reason") or "")
    suspicious_types = snapshot.get("suspicious_object_types", [])
    if isinstance(suspicious_types, list):
        suspicious_line = ", ".join(str(item) for item in suspicious_types) or "none"
    else:
        suspicious_line = "none"

    title_color = (0, 220, 255)
    if decision and decision.get("verdict") == "allow":
        title_color = (0, 255, 0)
    elif decision and decision.get("verdict") == "deny":
        title_color = (0, 0, 255)

    _draw_text(display, "ACCESS CONTROL PREVIEW", 20, 40, (0, 220, 255), 0.95, 2)
    _draw_text(display, f"STATE: {session_state}", 20, 82, title_color, 0.88, 2)
    _draw_text(display, f'FRAME: {snapshot.get("frame_id", 0)}', 20, 112, (185, 185, 190), 0.50, 1)
    _draw_text(display, f'BUFFER: {snapshot.get("buffered_frames", 0)}', 180, 112, (185, 185, 190), 0.50, 1)
    _draw_text(
        display,
        f'CAMERA: {"ON" if status["camera_running"] else "OFF"}',
        320,
        112,
        (185, 185, 190),
        0.50,
        1,
    )
    _draw_text(
        display,
        f'YOLO: {str(status.get("yolo_backend", "unknown")).upper()}',
        470,
        112,
        (185, 185, 190),
        0.50,
        1,
    )
    if status.get("yolo_backend") == "mock":
        _draw_text(display, "WARNING: YOLO backend is mock, object blocking is inactive", 20, 124, (0, 165, 255), 0.52, 1)

    info_y = height - 96
    cv2.rectangle(display, (0, info_y - 34), (width, height), (8, 8, 12), -1)
    if decision:
        verdict = str(decision.get("verdict", "pending")).upper()
        confidence = float(decision.get("confidence", 0.0)) * 100.0
        _draw_text(display, f"DECISION: {verdict} {confidence:0.1f}%", 20, info_y, title_color, 0.72, 2)
        _draw_text(display, f'REASON: {str(decision.get("reason", ""))}', 20, info_y + 32, (220, 220, 225), 0.58, 1)
    elif anti_spoof:
        confidence = float(anti_spoof.get("confidence", 0.0)) * 100.0
        _draw_text(
            display,
            f'ANTI-SPOOF: {str(anti_spoof.get("label", "pending")).upper()} {confidence:0.1f}%',
            20,
            info_y,
            (220, 220, 100),
            0.72,
            2,
        )
    else:
        _draw_text(display, "ANTI-SPOOF: collecting sequence", 20, info_y, (220, 220, 100), 0.72, 2)

    if blocked_reason:
        _draw_text(display, blocked_reason, 20, 146, (0, 0, 255), 0.56, 1)
    _draw_text(display, f"SUSPICIOUS: {suspicious_line}", 20, height - 24, (120, 200, 255), 0.54, 1)
    _draw_text(display, "Q quit | R reset session", width - 300, height - 24, (170, 170, 180), 0.54, 1)
    return display


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Visible preview for the access-control MVP")
    parser.add_argument("--camera", type=int, default=None, help="Camera index override")
    parser.add_argument("--video", type=str, default=None, help="Video file path instead of a live camera")
    parser.add_argument("--width", type=int, default=None, help="Camera width override")
    parser.add_argument("--height", type=int, default=None, help="Camera height override")
    parser.add_argument("--no-api", action="store_true", help="Do not start the FastAPI server")
    args = parser.parse_args(argv)

    setup_logging()
    settings = get_settings().model_copy(deep=True)
    if args.camera is not None:
        settings.camera.index = args.camera
    if args.video is not None:
        settings.camera.source = args.video
    if args.width is not None:
        settings.camera.width = args.width
    if args.height is not None:
        settings.camera.height = args.height

    pipeline = AccessControlPipeline(settings)
    server = None
    server_thread = None
    api_error = None

    if args.no_api:
        pipeline.start()
    else:
        server, server_thread, api_error = _start_api_server(settings, pipeline)
        if api_error is not None:
            pipeline.start()

    window_name = settings.runtime.preview_window_name
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, settings.runtime.preview_width, settings.runtime.preview_height)

    print("ACCESS CONTROL PREVIEW")
    print("Q quit | R reset session")
    if settings.camera.source:
        print(f"Video source: {settings.camera.source}")
    else:
        print(f"Camera index: {settings.camera.index}")
    if not args.no_api and api_error is None:
        print(f"API: http://{settings.api.host}:{settings.api.port}")
    elif api_error is not None:
        print(f"API disabled: {api_error}")

    try:
        while True:
            snapshot = pipeline.get_preview_snapshot()
            status = pipeline.get_status()
            frame = snapshot.get("frame")
            if isinstance(frame, np.ndarray):
                display = _render_preview(frame, snapshot, status)
            else:
                message = status["camera_error"] or "Waiting for the first camera frame"
                display = _placeholder_frame(settings, str(message))

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                pipeline.reset_session()

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            time.sleep(settings.runtime.loop_sleep_seconds)
    finally:
        cv2.destroyAllWindows()
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=2.0)
        if args.no_api or api_error is not None:
            pipeline.stop()


if __name__ == "__main__":
    main()
