"""Camera wrapper with simple Windows-friendly backend selection."""
from __future__ import annotations

import platform

import cv2
import numpy as np


class CameraService:
    def __init__(self, camera_index: int) -> None:
        self._camera_index = camera_index
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        if self._capture is not None:
            return
        for backend in self._backend_candidates():
            capture = cv2.VideoCapture(self._camera_index) if backend is None else cv2.VideoCapture(self._camera_index, backend)
            if not capture.isOpened():
                capture.release()
                continue
            ok, frame = capture.read()
            if ok and frame is not None:
                self._capture = capture
                return
            capture.release()
        raise RuntimeError(f"Unable to open camera index {self._camera_index}.")

    def read(self) -> np.ndarray | None:
        if self._capture is None:
            raise RuntimeError("Camera is not open.")
        ok, frame = self._capture.read()
        if not ok:
            return None
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @staticmethod
    def _backend_candidates() -> list[int | None]:
        if platform.system().lower().startswith("win"):
            return [cv2.CAP_DSHOW, None, cv2.CAP_MSMF]
        return [None]
