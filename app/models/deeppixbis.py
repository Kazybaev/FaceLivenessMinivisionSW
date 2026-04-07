"""DeepPixBiS wrappers for ONNX or PyTorch checkpoints."""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from app.core.schemas import BoundingBox, ModelScore
from app.models.base_model import BaseAntiSpoofModel
from src.generate_patches import CropImage


def _build_densenet161() -> nn.Module:
    try:
        return models.densenet161(weights=None)
    except TypeError:
        return models.densenet161(pretrained=False)


class _DeepPixBiSNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dense = _build_densenet161()
        features = list(dense.features.children())
        self.enc = nn.Sequential(*features[:8])
        self.dec = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(14 * 14, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.enc(x)
        mask = torch.sigmoid(self.dec(encoded))
        binary = torch.sigmoid(self.linear(mask.view(-1, 14 * 14)))
        return mask, torch.flatten(binary)


class DeepPixBiSPredictor(BaseAntiSpoofModel):
    def __init__(
        self,
        weights_path: Path,
        device: torch.device,
        providers: list[str],
        ensemble_weight: float,
        crop_scale: float,
        model_alias: str,
    ) -> None:
        if not weights_path.exists():
            raise FileNotFoundError(
                "DeepPixBiS weights were not found. "
                f"Expected file: {weights_path}"
            )
        self._weights_path = weights_path
        self._device = device
        self._ensemble_weight = ensemble_weight
        self._crop_scale = crop_scale
        self._model_alias = model_alias
        self._cropper = CropImage()
        self._providers = providers
        self._backend = weights_path.suffix.lower()
        self._input_height = 224
        self._input_width = 224

        if self._backend == ".onnx":
            import onnxruntime as ort

            self._session = ort.InferenceSession(weights_path.as_posix(), providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            _, _, self._input_height, self._input_width = self._session.get_inputs()[0].shape
            self._model = None
            return

        if self._backend not in {".pth", ".pt"}:
            raise ValueError(
                "DeepPixBiS weights must be .pth/.pt or .onnx. "
                f"Received: {weights_path}"
            )

        self._model = _DeepPixBiSNetwork().to(device)
        self._session = None
        self._input_name = None
        state_dict = torch.load(weights_path, map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if any(key.startswith("module.") for key in state_dict.keys()):
            state_dict = OrderedDict((key[7:], value) for key, value in state_dict.items())
        self._model.load_state_dict(state_dict, strict=True)
        self._model.eval()

    @property
    def model_name(self) -> str:
        return self._model_alias

    def predict(self, frame_bgr: np.ndarray, face_bbox: BoundingBox) -> ModelScore:
        patch = self._cropper.crop(
            frame_bgr,
            [face_bbox.x, face_bbox.y, face_bbox.width, face_bbox.height],
            self._crop_scale,
            int(self._input_width),
            int(self._input_height),
            crop=True,
        )
        face_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(face_rgb, (int(self._input_width), int(self._input_height)))
        if self._backend == ".onnx":
            tensor = np.expand_dims(resized.transpose(2, 0, 1).astype(np.float32) / 255.0, axis=0)
            output = self._session.run(None, {self._input_name: tensor})[0]
            output = np.asarray(output, dtype=np.float32).reshape(-1)

            if output.size == 1:
                real_score = float(1.0 / (1.0 + np.exp(-output[0])))
                spoof_score = float(1.0 - real_score)
            elif output.size >= 2:
                shifted = output[:2] - np.max(output[:2])
                probs = np.exp(shifted) / np.exp(shifted).sum()
                real_score = float(probs[1])
                spoof_score = float(probs[0])
            else:
                raise ValueError("DeepPixBiS returned unsupported output.")
        else:
            tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().div(255.0)
            tensor = tensor.sub(0.5).div(0.5).unsqueeze(0).to(self._device)
            with torch.inference_mode():
                mask, binary = self._model(tensor)
            pixel_score = float(mask.mean().item())
            binary_score = float(binary.mean().item())
            real_score = float(np.clip((pixel_score * 0.75) + (binary_score * 0.25), 0.0, 1.0))
            spoof_score = float(1.0 - real_score)

        return ModelScore(
            model_name=self.model_name,
            real_score=real_score,
            spoof_score=spoof_score,
            confidence=max(real_score, spoof_score),
            weight=self._ensemble_weight,
        )
