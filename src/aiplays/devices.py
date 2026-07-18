from __future__ import annotations

import torch


def select_device(requested: str = "auto") -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA was requested but PyTorch cannot see a compatible NVIDIA GPU")
    return "cuda" if requested == "cuda" or (requested == "auto" and available) else "cpu"


def device_description(device: str) -> str:
    if device == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return "cpu"
