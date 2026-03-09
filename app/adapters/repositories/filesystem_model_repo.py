"""Filesystem-based model repository.

Original: src/anti_spoof_predict.py:68-88 _load_all_models()
Preserves `module.` prefix stripping for DataParallel state dicts.
"""
from __future__ import annotations

import os
from collections import OrderedDict

import torch
import torch.nn as nn
import structlog

from app.domain.entities import ModelInfo
from app.domain.exceptions import ModelLoadError
from app.ml.utils import parse_model_name, get_kernel
from app.ml.models.minifasnet import MODEL_MAPPING

logger = structlog.get_logger(__name__)


class FilesystemModelRepo:
    def __init__(self, model_dir: str):
        self._model_dir = model_dir

    def list_models(self) -> list[ModelInfo]:
        models = []
        if not os.path.isdir(self._model_dir):
            logger.warning("model_dir_not_found", path=self._model_dir)
            return models
        for name in sorted(os.listdir(self._model_dir)):
            if not name.endswith(".pth"):
                continue
            try:
                h_input, w_input, model_type, scale = parse_model_name(name)
                models.append(ModelInfo(
                    name=name,
                    path=os.path.join(self._model_dir, name),
                    h_input=h_input,
                    w_input=w_input,
                    model_type=model_type,
                    scale=scale,
                ))
            except Exception as e:
                logger.warning("skip_model_file", name=name, error=str(e))
        return models

    def load_model(self, info: ModelInfo) -> nn.Module:
        if info.model_type not in MODEL_MAPPING:
            raise ModelLoadError(f"Unknown model type: {info.model_type}")

        kernel_size = get_kernel(info.h_input, info.w_input)
        model = MODEL_MAPPING[info.model_type](conv6_kernel=kernel_size)

        state_dict = torch.load(info.path, map_location="cpu", weights_only=True)

        # Strip `module.` prefix from DataParallel-saved state dicts
        first_key = next(iter(state_dict))
        if first_key.startswith('module.'):
            new_sd = OrderedDict()
            for k, v in state_dict.items():
                new_sd[k[7:]] = v
            state_dict = new_sd

        model.load_state_dict(state_dict)
        logger.info("model_loaded", name=info.name, type=info.model_type,
                     input_size=f"{info.h_input}x{info.w_input}")
        return model
