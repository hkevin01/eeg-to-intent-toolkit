"""Simple numpy ring buffers for real-time EEG."""

from __future__ import annotations

import threading

import numpy as np


class RingBuffer:
    """Fixed-size ring buffer for (n_samples, n_channels) arrays."""

    def __init__(self, capacity: int, n_channels: int, dtype=np.float32) -> None:
        self.capacity = int(capacity)
        self.n_channels = int(n_channels)
        self._buf = np.zeros((self.capacity, self.n_channels), dtype=dtype)
        self._idx = 0
        self._full = False
        self._lock = threading.Lock()

    def append(self, x: np.ndarray) -> None:
        arr = np.asarray(x)
        if arr.ndim == 1:
            arr = arr[None, :]
        with self._lock:
            for row in arr:
                self._buf[self._idx] = row
                self._idx = (self._idx + 1) % self.capacity
                if self._idx == 0:
                    self._full = True

    def latest(self, n: int) -> np.ndarray:
        with self._lock:
            n = min(n, self.size)
            if n <= 0:
                return self._buf[:0]
            start = (self._idx - n) % self.capacity
            if start + n <= self.capacity:
                return self._buf[start : start + n]
            first = self.capacity - start
            return np.vstack((self._buf[start:], self._buf[: n - first]))

    @property
    def size(self) -> int:
        return self.capacity if self._full else self._idx


class TimestampedRingBuffer(RingBuffer):
    """Ring buffer keeping parallel timestamp array."""

    def __init__(self, capacity: int, n_channels: int, dtype=np.float32) -> None:
        super().__init__(capacity, n_channels, dtype=dtype)
        self._ts = np.zeros((self.capacity,), dtype=np.float64)

    def append_with_ts(self, x: np.ndarray, ts: np.ndarray) -> None:
        arr = np.asarray(x)
        t = np.asarray(ts)
        if arr.ndim == 1:
            arr = arr[None, :]
        assert arr.shape[0] == t.shape[0]
        with self._lock:
            for i in range(arr.shape[0]):
                self._buf[self._idx] = arr[i]
                self._ts[self._idx] = float(t[i])
                self._idx = (self._idx + 1) % self.capacity
                if self._idx == 0:
                    self._full = True

    def latest_with_ts(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            n = min(n, self.size)
            if n <= 0:
                return self._buf[:0], self._ts[:0]
            start = (self._idx - n) % self.capacity
            if start + n <= self.capacity:
                return (
                    self._buf[start : start + n],
                    self._ts[start : start + n],
                )
            first = self.capacity - start
            data = np.vstack((self._buf[start:], self._buf[: n - first]))
            ts = np.concatenate((self._ts[start:], self._ts[: n - first]))
            return data, ts
