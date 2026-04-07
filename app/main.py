"""OpenCV entrypoint for the strict MiniFAS plus DeepPixBiS runtime."""
from __future__ import annotations

import logging
import sys

import cv2

from app.config import Settings
from app.core.pipeline import AntiSpoofPipeline
from app.services.camera_service import CameraService


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> int:
    settings = Settings.from_env()
    configure_logging(settings.log_level)

    try:
        pipeline = AntiSpoofPipeline(settings)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Failed to initialize anti-spoof pipeline: {exc}", file=sys.stderr)
        return 1

    camera = CameraService(settings.camera_index)
    try:
        camera.open()
    except Exception as exc:
        print(f"Failed to open camera {settings.camera_index}: {exc}", file=sys.stderr)
        return 1

    cv2.namedWindow(settings.window_name, cv2.WINDOW_NORMAL)
    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
            overlay, _ = pipeline.process_frame(frame)
            cv2.imshow(settings.window_name, overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
