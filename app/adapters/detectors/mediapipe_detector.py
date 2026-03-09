"""MediaPipe Face Landmarker detector.

Original: liveness_detection.py:51-67 get_landmarker()
"""
from __future__ import annotations

import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
import structlog

from app.domain.entities import BBox, FaceRegion
from app.infrastructure.config import MediaPipeConfig

logger = structlog.get_logger(__name__)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_PATH = "face_landmarker.task"


class MediaPipeDetector:
    def __init__(self, config: MediaPipeConfig):
        self._config = config
        self._landmarker = self._init_landmarker()
        logger.info("mediapipe_detector_initialized")

    def _init_landmarker(self) -> mp.tasks.vision.FaceLandmarker:
        if not os.path.exists(_MODEL_PATH):
            logger.info("downloading_mediapipe_model")
            urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=_MODEL_PATH,
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=self._config.num_faces,
            min_face_detection_confidence=self._config.min_face_detection_confidence,
            min_face_presence_confidence=self._config.min_face_presence_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(opts)

    def detect(self, image: np.ndarray) -> FaceRegion | None:
        result = self._detect_raw(image)
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
        return FaceRegion(bbox=bbox, image=face_crop, landmarks=landmarks)

    def detect_with_landmarks(self, image: np.ndarray) -> FaceRegion | None:
        return self.detect(image)

    def _detect_raw(self, image: np.ndarray) -> mp.tasks.vision.FaceLandmarkerResult:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._landmarker.detect(mp_img)
