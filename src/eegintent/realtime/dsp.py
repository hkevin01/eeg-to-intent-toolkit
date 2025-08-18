"""Streaming-friendly DSP filters for real-time EEG."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # optional SciPy
    from scipy.signal import butter, iirnotch, lfilter, lfilter_zi
except ImportError:  # pragma: no cover - scipy optional
    butter = None  # type: ignore
    iirnotch = None  # type: ignore
    lfilter = None  # type: ignore
    lfilter_zi = None  # type: ignore


@dataclass
class IIRBandpass:
    fs: float
    low: float
    high: float
    order: int = 4
    _b: np.ndarray | None = None
    _a: np.ndarray | None = None
    _zi: np.ndarray | None = None

    def __post_init__(self) -> None:
        if butter is None:
            raise RuntimeError("scipy is required for IIRBandpass")
        nyq = 0.5 * self.fs
        low = self.low / nyq
        high = self.high / nyq
        ba = butter(self.order, [low, high], btype="bandpass")
        self._b, self._a = ba  # type: ignore[misc]

    def reset(self, n_channels: int) -> None:
        if lfilter_zi is None or self._b is None or self._a is None:
            self._zi = None
            return
        zi = lfilter_zi(self._b, self._a)
        self._zi = np.tile(zi[:, None], (1, n_channels))

    def process(self, x: np.ndarray) -> np.ndarray:
        if lfilter is None or self._b is None or self._a is None:
            return x
        if self._zi is None:
            self.reset(x.shape[1])
        y, self._zi = lfilter(self._b, self._a, x, axis=0, zi=self._zi)
        return y


@dataclass
class NotchFilter:
    fs: float
    freq: float = 50.0
    q: float = 30.0
    _b: np.ndarray | None = None
    _a: np.ndarray | None = None
    _zi: np.ndarray | None = None

    def __post_init__(self) -> None:
        if iirnotch is None:
            raise RuntimeError("scipy is required for NotchFilter")
        w0 = self.freq / (self.fs / 2.0)
        ba = iirnotch(w0, self.q)
        self._b, self._a = ba  # type: ignore[misc]

    def reset(self, n_channels: int) -> None:
        if lfilter_zi is None or self._b is None or self._a is None:
            self._zi = None
            return
        zi = lfilter_zi(self._b, self._a)
        self._zi = np.tile(zi[:, None], (1, n_channels))

    def process(self, x: np.ndarray) -> np.ndarray:
        if lfilter is None or self._b is None or self._a is None:
            return x
        if self._zi is None:
            self.reset(x.shape[1])
        y, self._zi = lfilter(self._b, self._a, x, axis=0, zi=self._zi)
        return y


@dataclass
class AdaptiveNoiseGate:
    """Simple noise gate using RMS threshold in a trailing window."""

    window: int = 256
    ratio: float = 3.0
    min_scale: float = 0.3

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] < self.window:
            return x
        y = x.copy()
        w = self.window
        rms = np.sqrt(np.mean(y[-w:] ** 2, axis=0) + 1e-8)
        med = np.median(rms)
        if med <= 0:
            return y
        scale = np.clip(self.ratio * med / (rms + 1e-8), self.min_scale, 1.0)
        y[-w:] *= scale
        return y


class StreamingPipeline:
    """Compose multiple processors with .process(x) API."""

    def __init__(self, *ops) -> None:
        self.ops = list(ops)

    def process(self, x: np.ndarray) -> np.ndarray:
        y = x
        for op in self.ops:
            y = op.process(y)
        return y


def sliding_mean(x: np.ndarray, k: int = 3) -> np.ndarray:
    """Simple moving average along the last axis.

    Parameters
    - x: array of shape (..., T)
    - k: window size (odd preferred)
    """
    if k <= 1:
        return x
    kernel = np.ones((k,), dtype=x.dtype) / float(k)
    # Apply along the last axis via convolution with 'same' output size

    def _conv_last(v: np.ndarray) -> np.ndarray:
        return np.convolve(v, kernel, mode="same")

    return np.apply_along_axis(_conv_last, axis=-1, arr=x)
