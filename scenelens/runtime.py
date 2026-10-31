"""Prefer CUDA for inference, falling back to CPU when CUDA is unavailable."""

import torch

from scenelens import config


def resolve_device(device=None):
    selected = torch.device(device if device is not None else config.DEFAULT_DEVICE)
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            return "cpu"
        index = selected.index if selected.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} does not exist.")
        return f"cuda:{index}"
    return str(selected)
