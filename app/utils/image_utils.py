from __future__ import annotations

import cv2
import numpy as np

from app.core.models import BoundingBox


def clip_box(box: BoundingBox, width: int, height: int) -> BoundingBox:
    x1 = max(0, min(width - 1, box.x))
    y1 = max(0, min(height - 1, box.y))
    x2 = max(x1 + 1, min(width, box.x2))
    y2 = max(y1 + 1, min(height, box.y2))
    return BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)


def crop_bbox(frame: np.ndarray, box: BoundingBox) -> np.ndarray:
    clipped = clip_box(box, frame.shape[1], frame.shape[0])
    return frame[clipped.y:clipped.y2, clipped.x:clipped.x2].copy()


def laplacian_variance(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
