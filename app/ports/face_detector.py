from __future__ import annotations

from typing import Protocol

import numpy as np

from app.domain.entities import FaceRegion


class FaceDetectorPort(Protocol):
    def detect(self, image: np.ndarray) -> FaceRegion | None: ...
    def detect_with_landmarks(self, image: np.ndarray) -> FaceRegion | None: ...
