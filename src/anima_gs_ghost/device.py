"""Backend selection helpers for Mac-first development and CUDA training."""

from __future__ import annotations

import os
from functools import lru_cache


def _detect_backend() -> str:
    forced = os.environ.get("ANIMA_BACKEND")
    if forced:
        return forced

    try:
        import mlx.core as mx  # noqa: F401
    except Exception:
        pass
    else:
        return "mlx"

    try:
        import torch
    except Exception:
        return "cpu"

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_backend() -> str:
    return _detect_backend()


def get_device() -> str:
    backend = get_backend()
    if backend == "cuda":
        return os.environ.get("ANIMA_CUDA_DEVICE", "cuda:0")
    if backend == "mlx":
        return "mlx"
    return "cpu"

