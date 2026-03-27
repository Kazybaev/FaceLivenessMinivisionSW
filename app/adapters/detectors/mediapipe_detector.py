"""MediaPipe Face Landmarker detector."""
from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import structlog

from app.domain.entities import BBox, FaceRegion
from app.domain.exceptions import AssetValidationError
from app.infrastructure.assets import resolve_path
from app.infrastructure.config import MediaPipeConfig

logger = structlog.get_logger(__name__)


class MediaPipeDetector:
    def __init__(self, config: MediaPipeConfig, running_mode: str = "IMAGE"):
        self._config = config
        self._running_mode = running_mode.upper()
        self._landmarker = self._init_landmarker()
        logger.info("mediapipe_detector_initialized", running_mode=self._running_mode)

    @property
    def model_path(self) -> str:
        return str(resolve_path(self._config.model_path))

    def _init_landmarker(self) -> mp.tasks.vision.FaceLandmarker:
        model_path = resolve_path(self._config.model_path)
        if not model_path.is_file():
            raise AssetValidationError(f"MediaPipe model not found: {model_path}")

        if self._running_mode == "VIDEO":
            running_mode = mp.tasks.vision.RunningMode.VIDEO
        else:
            running_mode = mp.tasks.vision.RunningMode.IMAGE

        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=running_mode,
            num_faces=self._config.num_faces,
            min_face_detection_confidence=self._config.min_face_detection_confidence,
            min_face_presence_confidence=self._config.min_face_presence_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(opts)

    def detect(self, image: np.ndarray) -> FaceRegion | None:
        result = self._detect_raw(image)
        return self._face_region_from_result(image, result)

    def detect_video_frame(self, image: np.ndarray, timestamp_ms: int) -> FaceRegion | None:
        result = self._detect_raw(image, timestamp_ms=timestamp_ms)
        return self._face_region_from_result(image, result)

    def detect_with_landmarks(self, image: np.ndarray) -> FaceRegion | None:
        return self.detect(image)

    def _detect_raw(
        self,
        image: np.ndarray,
        timestamp_ms: int | None = None,
    ) -> mp.tasks.vision.FaceLandmarkerResult:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._running_mode == "VIDEO":
            if timestamp_ms is None:
                raise ValueError("timestamp_ms is required in VIDEO mode")
            return self._landmarker.detect_for_video(mp_img, timestamp_ms)
        return self._landmarker.detect(mp_img)

    def _face_region_from_result(
        self,
        image: np.ndarray,
        result: mp.tasks.vision.FaceLandmarkerResult,
    ) -> FaceRegion | None:
        if not result.face_landmarks:
            return None
        h, w = image.shape[:2]
        lm = result.face_landmarks[0]
        landmarks = np.array([[l.x * w, l.y * h] for l in lm], dtype=np.float32)
        xs = landmarks[:, 0].astype(int)
        ys = landmarks[:, 1].astype(int)
        x1 = max(0, int(xs.min()) - 10)
        y1 = max(0, int(ys.min()) - 10)
        x2 = min(w, int(xs.max()) + 10)
        y2 = min(h, int(ys.max()) + 10)
        bbox = BBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None
        return FaceRegion(bbox=bbox, image=face_crop, landmarks=landmarks)
