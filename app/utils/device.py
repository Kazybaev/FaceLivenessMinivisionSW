"""Execution provider helpers."""
from __future__ import annotations

import onnxruntime as ort
import torch


def resolve_torch_device(device_name: str) -> torch.device:
    normalized = device_name.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda requested, but CUDA is not available.")
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported DEVICE value: {device_name!r}")


def resolve_execution_providers(device_name: str) -> list[str]:
    normalized = device_name.strip().lower()
    available = ort.get_available_providers()
    if normalized == "auto":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    if normalized == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("DEVICE=cuda requested, but CUDAExecutionProvider is not available.")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if normalized == "cpu":
        return ["CPUExecutionProvider"]
    raise ValueError(f"Unsupported DEVICE value: {device_name!r}")
