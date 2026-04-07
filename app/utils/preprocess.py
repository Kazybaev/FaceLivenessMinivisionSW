"""Generic preprocessing helper."""
from __future__ import annotations

import cv2
import numpy as np


def preprocess_face_crop(
    face_bgr: np.ndarray,
    input_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(face_rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    image = resized.astype(np.float32) / 255.0
    chw = image.transpose(2, 0, 1)
    mean_array = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std_array = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    normalized = (chw - mean_array) / std_array
    return np.expand_dims(normalized.astype(np.float32), axis=0)
