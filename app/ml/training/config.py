"""Training configuration.

Original: src/default_config.py (Minivision, zhuying)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import torch

from app.ml.utils import make_if_not_exist, get_width_height, get_kernel


@dataclass
class TrainConfig:
    lr: float = 0.1
    milestones: list[int] = field(default_factory=lambda: [10, 15, 22])
    gamma: float = 0.1
    epochs: int = 25
    momentum: float = 0.9
    weight_decay: float = 5e-4
    batch_size: int = 1024
    num_classes: int = 3
    input_channel: int = 3
    embedding_size: int = 128
    train_root_path: str = './datasets/rgb_image'
    snapshot_dir_path: str = './saved_logs/snapshot'
    log_path: str = './saved_logs/jobs'
    board_loss_every: int = 10
    save_every: int = 30
    num_workers: int = 16
    cls_loss_weight: float = 0.5
    ft_loss_weight: float = 0.5

    # Set by update_from_args
    devices: list[int] = field(default_factory=lambda: [0])
    patch_info: str = '1_80x80'
    input_size: list[int] = field(default_factory=lambda: [80, 80])
    kernel_size: tuple[int, int] = (5, 5)
    device: str = 'cpu'
    ft_height: int = 10
    ft_width: int = 10
    model_path: str = ''
    job_name: str = ''

    def update_from_args(self, devices: list[int], patch_info: str) -> None:
        self.devices = devices
        self.patch_info = patch_info
        w_input, h_input = get_width_height(patch_info)
        self.input_size = [h_input, w_input]
        self.kernel_size = get_kernel(h_input, w_input)
        self.device = f"cuda:{devices[0]}" if torch.cuda.is_available() else "cpu"

        self.ft_height = 2 * self.kernel_size[0]
        self.ft_width = 2 * self.kernel_size[1]

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.job_name = f'Anti_Spoofing_{patch_info}'
        log_path = f'{self.log_path}/{self.job_name}/{current_time} '
        snapshot_dir = f'{self.snapshot_dir_path}/{self.job_name}'

        make_if_not_exist(snapshot_dir)
        make_if_not_exist(log_path)

        self.model_path = snapshot_dir
        self.log_path = log_path
