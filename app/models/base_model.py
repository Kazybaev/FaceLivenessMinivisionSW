"""Base model abstractions."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.core.schemas import BoundingBox, ModelScore


class BaseAntiSpoofModel(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def predict(self, frame_bgr: np.ndarray, face_bbox: BoundingBox) -> ModelScore:
        raise NotImplementedError
