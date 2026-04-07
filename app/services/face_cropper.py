"""Face crop helper."""
from __future__ import annotations

import numpy as np

from app.core.schemas import BoundingBox


class FaceCropper:
    def __init__(self, margin_ratio: float) -> None:
        self._margin_ratio = margin_ratio

    def crop(self, frame: np.ndarray, bbox: BoundingBox) -> tuple[np.ndarray, BoundingBox]:
        margin_x = int(bbox.width * self._margin_ratio)
        margin_y = int(bbox.height * self._margin_ratio)
        crop_box = BoundingBox(
            x=bbox.x - margin_x,
            y=bbox.y - margin_y,
            width=bbox.width + 2 * margin_x,
            height=bbox.height + 2 * margin_y,
        ).clamp(frame.shape[1], frame.shape[0])
        crop = frame[crop_box.y:crop_box.y2, crop_box.x:crop_box.x2]
        if crop.size == 0:
            raise ValueError("Computed face crop is empty.")
        return crop, crop_box
