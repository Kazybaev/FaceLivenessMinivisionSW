from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from app.domain.entities import BBox


class ImagePreprocessorPort(Protocol):
    def crop_face(
        self,
        image: np.ndarray,
        bbox: BBox,
        scale: float,
        out_w: int,
        out_h: int,
    ) -> np.ndarray: ...

    def to_tensor(self, image: np.ndarray) -> torch.Tensor: ...
