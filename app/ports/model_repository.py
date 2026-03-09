from __future__ import annotations

from typing import Protocol

import torch.nn as nn

from app.domain.entities import ModelInfo


class ModelRepositoryPort(Protocol):
    def list_models(self) -> list[ModelInfo]: ...
    def load_model(self, info: ModelInfo) -> nn.Module: ...
