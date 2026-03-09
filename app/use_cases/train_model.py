"""Use case: train anti-spoofing model."""
from __future__ import annotations

import structlog

from app.ml.training.trainer import Trainer
from app.ml.training.config import TrainConfig

logger = structlog.get_logger(__name__)


class TrainModelUseCase:
    def __init__(self, config: TrainConfig):
        self._config = config

    def execute(self, devices: list[int], patch_info: str) -> None:
        self._config.update_from_args(devices, patch_info)
        logger.info("training_started", patch_info=patch_info, devices=devices)
        trainer = Trainer(self._config)
        trainer.train()
        logger.info("training_completed")
