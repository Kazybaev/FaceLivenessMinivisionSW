"""MiniFAS model wrappers."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from app.core.schemas import BoundingBox, ModelScore
from app.models.base_model import BaseAntiSpoofModel
from src.generate_patches import CropImage
from src.model_lib.MiniFASNet import MiniFASNetV1SE, MiniFASNetV2
from src.utility import get_kernel, parse_model_name


_MODEL_FACTORY = {
    "MiniFASNetV2": MiniFASNetV2,
    "MiniFASNetV1SE": MiniFASNetV1SE,
}


class MiniFASPredictor(BaseAntiSpoofModel):
    def __init__(self, weights_path: Path, device: torch.device, ensemble_weight: float) -> None:
        if not weights_path.exists():
            raise FileNotFoundError(f"MiniFAS weights were not found: {weights_path}")

        self._weights_path = weights_path
        self._device = device
        self._ensemble_weight = ensemble_weight
        self._cropper = CropImage()

        height, width, model_type, scale = parse_model_name(weights_path.name)
        self._model_name = model_type
        self._input_height = int(height)
        self._input_width = int(width)
        self._scale = float(scale or 1.0)
        kernel_size = get_kernel(height, width)
        self._model = _MODEL_FACTORY[model_type](conv6_kernel=kernel_size).to(device)

        state_dict = torch.load(weights_path, map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = OrderedDict((key[7:], value) for key, value in state_dict.items())
        self._model.load_state_dict(state_dict, strict=True)
        self._model.eval()

    @property
    def model_name(self) -> str:
        return self._model_name

    def predict(self, frame_bgr: np.ndarray, face_bbox: BoundingBox) -> ModelScore:
        patch = self._cropper.crop(
            frame_bgr,
            [face_bbox.x, face_bbox.y, face_bbox.width, face_bbox.height],
            self._scale,
            self._input_width,
            self._input_height,
            crop=True,
        )
        tensor = self._to_tensor(patch).to(self._device)
        with torch.inference_mode():
            logits = self._model(tensor)
            probabilities = F.softmax(logits, dim=1).detach().cpu().numpy().reshape(-1)

        spoof_score = float(probabilities[0])
        real_score = float(probabilities[1])
        total = max(real_score + spoof_score, 1e-6)
        real_score /= total
        spoof_score /= total
        return ModelScore(
            model_name=self.model_name,
            real_score=real_score,
            spoof_score=spoof_score,
            confidence=max(real_score, spoof_score),
            weight=self._ensemble_weight,
        )

    @staticmethod
    def _to_tensor(patch_bgr: np.ndarray) -> torch.Tensor:
        image = patch_bgr.astype(np.float32) / 255.0
        chw = image.transpose(2, 0, 1)
        return torch.from_numpy(chw).unsqueeze(0)
