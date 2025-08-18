"""Preprocessing utilities: filters, referencing, epoching, normalization."""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass(
    x: np.ndarray, low_hz: float, high_hz: float, sfreq: float, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth band-pass filter along time axis.

    Args:
        x: array of shape (n_samples, n_channels, n_times)
        low_hz: low cutoff in Hz (> 0)
        high_hz: high cutoff in Hz (< sfreq/2)
        sfreq: sampling frequency in Hz
        order: filter order
    """
    nyq = 0.5 * sfreq
    low = low_hz / nyq if low_hz is not None else 0.0
    high = high_hz / nyq if high_hz is not None else 1.0
    if not (0.0 <= low < high <= 1.0):
        raise ValueError("Invalid band edges for bandpass filter.")
    b, a = cast(
        tuple[np.ndarray, np.ndarray],
        butter(order, [low, high], btype="bandpass"),
    )
    # reshape to (n_samples*n_channels, n_times) for vectorized filtering
    ns, nc, nt = x.shape
    x2d = x.reshape(ns * nc, nt)
    y2d = filtfilt(b, a, x2d, axis=-1).astype(x.dtype, copy=False)
    return y2d.reshape(ns, nc, nt)


def notch(x: np.ndarray, freq: float, sfreq: float, quality: float = 30.0) -> np.ndarray:
    """IIR notch filter around powerline frequency.

    Args:
        x: (n_samples, n_channels, n_times)
        freq: notch center frequency (e.g., 50 or 60)
        sfreq: sampling frequency
        quality: quality factor (higher = narrower notch)
    """
    b, a = cast(
        tuple[np.ndarray, np.ndarray],
        iirnotch(w0=freq / (sfreq / 2.0), Q=quality),
    )
    ns, nc, nt = x.shape
    x2d = x.reshape(ns * nc, nt)
    y2d = filtfilt(b, a, x2d, axis=-1).astype(x.dtype, copy=False)
    return y2d.reshape(ns, nc, nt)


def car(x: np.ndarray) -> np.ndarray:
    """Common average reference across channels.

    Subtracts the per-sample mean across channels.
    """
    mean_ch = x.mean(axis=1, keepdims=True)
    return x - mean_ch


def preprocess_pipeline(
    x: np.ndarray,
    sfreq: float,
    do_notch: bool | float | None = None,
    band: tuple[float, float] | None = None,
    do_car: bool = True,
    do_zscore: bool = True,
) -> np.ndarray:
    """Configurable preprocessing steps applied in sequence.

    Args:
        x: (n_samples, n_channels, n_times)
        sfreq: sampling frequency
        do_notch: False/None to skip, or powerline frequency (50/60)
        band: (low, high) Hz for bandpass, or None to skip
        do_car: apply common average reference
        do_zscore: z-score normalize per-channel per-sample
    """
    out = x
    if do_notch:
        f0 = float(do_notch) if isinstance(do_notch, (int, float)) else 50.0
        out = notch(out, f0, sfreq)
    if band is not None:
        out = bandpass(out, band[0], band[1], sfreq)
    if do_car:
        out = car(out)
    if do_zscore:
        out = zscore(out)
    return out


def zscore(x: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    mu = x.mean(axis=axis, keepdims=True)
    sigma = x.std(axis=axis, keepdims=True)
    return (x - mu) / (sigma + eps)


def epoch_sliding(
    x: np.ndarray, y: np.ndarray, window: int, step: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows along time for each sample.

    Args:
        x: (n_samples, n_channels, n_times)
        y: (n_samples,) labels (propagated to all windows of a sample)
        window: window length in samples
        step: stride in samples

    Returns:
    xw: (n_windows_total, n_channels, window)
    yw: (n_windows_total,)
    """
    ns, nc, nt = x.shape
    if window <= 0 or step <= 0 or window > nt:
        raise ValueError("Invalid window/step for epoch_sliding.")
    windows = []
    labels = []
    for i in range(ns):
        start = 0
        while start + window <= nt:
            windows.append(x[i, :, start : start + window])
            labels.append(y[i])
            start += step
    xw = np.stack(windows, axis=0) if windows else np.empty((0, nc, window), dtype=x.dtype)
    yw = np.asarray(labels, dtype=y.dtype) if labels else np.empty((0,), dtype=y.dtype)
    return xw, yw
