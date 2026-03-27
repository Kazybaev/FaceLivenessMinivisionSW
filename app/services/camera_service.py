from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from app.config import CameraSettings
from app.core.models import EventSeverity, FramePacket
from app.services.event_logger import EventLogger


class CameraService:
    """Continuously reads frames from a camera in a dedicated thread."""

    def __init__(self, settings: CameraSettings, event_logger: EventLogger):
        self._settings = settings
        self._event_logger = event_logger
        self._capture: cv2.VideoCapture | None = None
        self._latest_frame: FramePacket | None = None
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._frame_counter = 0
        self._camera_error: str | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def get_latest_frame(self) -> FramePacket | None:
        with self._lock:
            return self._latest_frame

    def status(self) -> dict[str, object]:
        return {
            "running": self._running and self._capture is not None and self._capture.isOpened(),
            "frame_counter": self._frame_counter,
            "error": self._camera_error,
            "source": self._source_description(),
        }

    def _loop(self) -> None:
        while self._running:
            if self._capture is None or not self._capture.isOpened():
                self._open_camera()
                if self._capture is None or not self._capture.isOpened():
                    time.sleep(self._settings.reconnect_interval_seconds)
                    continue

            ok, frame = self._capture.read()
            if not ok or frame is None:
                if self._settings.source and self._settings.loop_video and self._restart_video():
                    time.sleep(self._settings.read_sleep_seconds)
                    continue

                self._camera_error = f"Failed to read frame from {self._source_description()}"
                self._event_logger.log(
                    "camera_read_failed",
                    self._camera_error,
                    severity=EventSeverity.WARNING,
                )
                time.sleep(self._settings.read_sleep_seconds)
                continue

            self._frame_counter += 1
            packet = FramePacket(
                frame_id=self._frame_counter,
                timestamp=time.monotonic(),
                frame=frame,
            )
            with self._lock:
                self._latest_frame = packet
            time.sleep(self._settings.read_sleep_seconds)

    def _open_camera(self) -> None:
        if self._capture is not None:
            self._capture.release()
        source: int | str = self._settings.source if self._settings.source else self._settings.index
        self._capture = cv2.VideoCapture(source)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self._settings.source:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.height)
        if not self._capture.isOpened():
            self._camera_error = f"Cannot open {self._source_description()}"
            self._event_logger.log(
                "camera_open_failed",
                self._camera_error,
                severity=EventSeverity.ERROR,
            )
            return

        self._camera_error = None
        self._event_logger.log(
            "camera_started",
            f"Video source started: {self._source_description()}",
            payload={
                "camera_index": self._settings.index,
                "source": self._settings.source,
                "loop_video": self._settings.loop_video,
            },
        )

    def _restart_video(self) -> bool:
        if self._capture is None or not self._settings.source:
            return False
        restarted = self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if restarted:
            self._camera_error = None
            self._event_logger.log(
                "video_restarted",
                "Video file reached the end and restarted from frame 0",
                payload={"source": self._settings.source},
            )
        return restarted

    def _source_description(self) -> str:
        if self._settings.source:
            return f"video file {self._settings.source}"
        return f"camera index {self._settings.index}"
