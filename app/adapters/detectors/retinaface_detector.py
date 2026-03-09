"""RetinaFace-based face detector using Caffe DNN.

Original: src/anti_spoof_predict.py:34-59 FaceDetector class
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import structlog

from app.domain.entities import BBox, FaceRegion
from app.infrastructure.config import RetinaFaceConfig

logger = structlog.get_logger(__name__)


class RetinaFaceDetector:
    def __init__(self, config: RetinaFaceConfig):
        self._config = config
        self._detector = cv2.dnn.readNetFromCaffe(config.prototxt, config.caffemodel)
        self._confidence_thresh = config.confidence_threshold
        logger.info("retinaface_detector_initialized")

    def detect(self, image: np.ndarray) -> FaceRegion | None:
        bbox = self._get_bbox(image)
        if bbox is None:
            return None
        clamped = bbox.clamp(image.shape[1], image.shape[0])
        face_crop = image[clamped.y:clamped.y2, clamped.x:clamped.x2]
        if face_crop.size == 0:
            return None
        return FaceRegion(bbox=clamped, image=face_crop)

    def detect_with_landmarks(self, image: np.ndarray) -> FaceRegion | None:
        return self.detect(image)

    def _get_bbox(self, img: np.ndarray) -> BBox | None:
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= self._config.input_size * self._config.input_size:
            img = cv2.resize(
                img,
                (int(self._config.input_size * math.sqrt(aspect_ratio)),
                 int(self._config.input_size / math.sqrt(aspect_ratio))),
                interpolation=cv2.INTER_LINEAR,
            )
        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self._detector.setInput(blob, 'data')
        out = self._detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        if out[max_conf_index, 2] < self._confidence_thresh:
            return None
        left = out[max_conf_index, 3] * width
        top = out[max_conf_index, 4] * height
        right = out[max_conf_index, 5] * width
        bottom = out[max_conf_index, 6] * height
        return BBox(
            x=int(left),
            y=int(top),
            width=int(right - left + 1),
            height=int(bottom - top + 1),
        )
