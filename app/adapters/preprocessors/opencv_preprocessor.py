"""OpenCV-based image preprocessor for face crops.

Original: src/generate_patches.py CropImage class
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from app.domain.entities import BBox
from app.ml.data.transforms import to_tensor


class OpenCVPreprocessor:
    @staticmethod
    def crop_face(
        image: np.ndarray,
        bbox: BBox,
        scale: float,
        out_w: int,
        out_h: int,
    ) -> np.ndarray:
        if scale is None:
            return cv2.resize(image, (out_w, out_h))

        src_h, src_w = image.shape[:2]
        x = bbox.x
        y = bbox.y
        box_w = bbox.width
        box_h = bbox.height

        scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))

        new_width = box_w * scale
        new_height = box_h * scale
        center_x = box_w / 2 + x
        center_y = box_h / 2 + y

        left_top_x = center_x - new_width / 2
        left_top_y = center_y - new_height / 2
        right_bottom_x = center_x + new_width / 2
        right_bottom_y = center_y + new_height / 2

        if left_top_x < 0:
            right_bottom_x -= left_top_x
            left_top_x = 0
        if left_top_y < 0:
            right_bottom_y -= left_top_y
            left_top_y = 0
        if right_bottom_x > src_w - 1:
            left_top_x -= right_bottom_x - src_w + 1
            right_bottom_x = src_w - 1
        if right_bottom_y > src_h - 1:
            left_top_y -= right_bottom_y - src_h + 1
            right_bottom_y = src_h - 1

        img = image[
            int(left_top_y):int(right_bottom_y) + 1,
            int(left_top_x):int(right_bottom_x) + 1,
        ]
        return cv2.resize(img, (out_w, out_h))

    @staticmethod
    def to_tensor(image: np.ndarray) -> torch.Tensor:
        return to_tensor(image)
