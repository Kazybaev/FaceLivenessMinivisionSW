"""Face landmark extraction for passive point-based liveness."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions

from app.core.schemas import LandmarkObservation


_KEYPOINT_IDS = (
    10,   # forehead
    1,    # nose tip
    152,  # chin
    33,   # left eye outer
    133,  # left eye inner
    263,  # right eye outer
    362,  # right eye inner
    61,   # mouth left
    291,  # mouth right
    13,   # upper lip
    14,   # lower lip
    234,  # left cheek
    454,  # right cheek
)


class LandmarkLivenessService:
    def __init__(self, model_path: Path, logger_name: str = __name__) -> None:
        self._logger = logging.getLogger(logger_name)
        self._landmarker: FaceLandmarker | None = None
        self._enabled = False
        if not model_path.exists():
            self._logger.warning("Face landmarker model is missing: %s. Point-based liveness disabled.", model_path)
            return

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path.as_posix()),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._enabled = True

    def detect(self, face_crop_bgr: np.ndarray) -> LandmarkObservation | None:
        if not self._enabled or self._landmarker is None or face_crop_bgr.size == 0:
            return None
        rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        points = np.asarray(
            [[float(landmarks[idx].x), float(landmarks[idx].y)] for idx in _KEYPOINT_IDS],
            dtype=np.float32,
        )
        depth_values = np.asarray([float(landmarks[idx].z) for idx in _KEYPOINT_IDS], dtype=np.float32)
        return LandmarkObservation(points=points, depth_values=depth_values)
