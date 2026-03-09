"""CLI: train anti-spoofing model.

Replaces: train.py
Usage: python -m app.cli.train --device_ids 0 --patch_info 1_80x80
"""
from __future__ import annotations

import argparse
import os

from app.infrastructure.logging_setup import setup_logging
from app.ml.training.config import TrainConfig
from app.use_cases.train_model import TrainModelUseCase


def parse_args():
    parser = argparse.ArgumentParser(description="Silence-FAS Training")
    parser.add_argument("--device_ids", type=str, default="1", help="GPU IDs, e.g. 0123")
    parser.add_argument("--patch_info", type=str, default="1_80x80",
                        help="[org_1_80x60 / 1_80x80 / 2.7_80x80 / 4_80x80]")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    cuda_devices = [int(elem) for elem in args.device_ids]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, cuda_devices))
    devices = list(range(len(cuda_devices)))

    config = TrainConfig()
    use_case = TrainModelUseCase(config)
    use_case.execute(devices, args.patch_info)
