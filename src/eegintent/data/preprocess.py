"""Preprocessing utilities: filters, referencing, epoching, normalization."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def bandpass(x: np.ndarray, low_hz: float, high_hz: float, sfreq: float) -> np.ndarray:
    """No-op stub bandpass; replace with MNE filtering.

    Args:
        x: (n_samples, n_channels, n_times)
        low_hz: low cut
        high_hz: high cut
        sfreq: sampling frequency
    """
    _ = (low_hz, high_hz, sfreq)
    return x


def zscore(x: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    mu = x.mean(axis=axis, keepdims=True)
    sigma = x.std(axis=axis, keepdims=True)
    return (x - mu) / (sigma + eps)


def epoch_sliding(
    x: np.ndarray, y: np.ndarray, window: int, step: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Make sliding windows from continuous or trial data.

    If x is trialized already, we just return x; this is a stub.
    """
    _ = (window, step, y)
    return x, y
