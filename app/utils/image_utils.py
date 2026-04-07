"""Image helper functions."""
from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from app.core.schemas import BoundingBox


def select_largest_box(boxes: Iterable[BoundingBox]) -> BoundingBox | None:
    items = list(boxes)
    if not items:
        return None
    return max(items, key=lambda item: item.area)


def resize_frame_if_needed(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
