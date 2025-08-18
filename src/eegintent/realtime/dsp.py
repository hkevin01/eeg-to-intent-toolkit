from __future__ import annotations

import numpy as np


def sliding_mean(x: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1:
        return x
    pad = k // 2
    xpad = np.pad(x, ((0, 0), (0, 0), (pad, pad)), mode="edge")
    out = np.empty_like(x)
    for i in range(x.shape[-1]):
        out[..., i] = xpad[..., i : i + k].mean(axis=-1)
    return out
