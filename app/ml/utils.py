"""Utility functions for ML models.

Original: src/utility.py (Minivision, zhuying)
"""
from __future__ import annotations

import os
from datetime import datetime


def get_time() -> str:
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def get_kernel(height: int, width: int) -> tuple[int, int]:
    return ((height + 15) // 16, (width + 15) // 16)


def get_width_height(patch_info: str) -> tuple[int, int]:
    w_input = int(patch_info.split('x')[-1])
    h_input = int(patch_info.split('x')[0].split('_')[-1])
    return w_input, h_input


def parse_model_name(model_name: str) -> tuple[int, int, str, float | None]:
    info = model_name.split('_')[0:-1]
    h_input, w_input = info[-1].split('x')
    model_type = model_name.split('.pth')[0].split('_')[-1]

    if info[0] == "org":
        scale = None
    else:
        scale = float(info[0])
    return int(h_input), int(w_input), model_type, scale


def make_if_not_exist(folder_path: str) -> None:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
