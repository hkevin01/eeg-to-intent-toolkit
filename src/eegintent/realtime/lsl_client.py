"""Minimal LSL receiver wrapper with optional pylsl dependency.

Provides discovery, connect, read, and close APIs for EEG streams.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np


class LSLNotAvailableError(RuntimeError):
    """Raised when pylsl is missing but LSL functionality is requested."""


@dataclass
class LSLStreamInfo:
    name: str
    type: str
    channel_count: int
    nominal_srate: float
    source_id: str | None = None


class LSLReceiver:
    """Receiver for EEG data via Lab Streaming Layer (LSL)."""

    def __init__(self) -> None:
        self._pylsl: Any = None
        self._inlet: Any = None
        self._info: LSLStreamInfo | None = None

    def _ensure_pylsl(self) -> None:
        if self._pylsl is None:
            try:
                self._pylsl = importlib.import_module("pylsl")
            except Exception as exc:  # pragma: no cover
                raise LSLNotAvailableError(
                    "pylsl not installed. Install via `pip install pylsl`."
                ) from exc

    def discover(
        self, stream_type: str = "EEG", timeout: float = 2.0
    ) -> list[LSLStreamInfo]:
        """Discover available LSL streams of a given type."""
        self._ensure_pylsl()
        infos = self._pylsl.resolve_stream(
            "type", stream_type, timeout=timeout
        )
        out: list[LSLStreamInfo] = []
        for info in infos:
            out.append(
                LSLStreamInfo(
                    name=info.name(),
                    type=info.type(),
                    channel_count=info.channel_count(),
                    nominal_srate=info.nominal_srate(),
                    source_id=(
                        info.source_id()
                        if hasattr(info, "source_id")
                        else None
                    ),
                )
            )
        return out

    def connect(
        self,
        stream_name: str | None = None,
        stream_type: str = "EEG",
        timeout: float = 5.0,
    ) -> LSLStreamInfo:
        """Connect to an LSL stream by name or first available of type."""
        self._ensure_pylsl()
        infos = self._pylsl.resolve_stream(
            "type", stream_type, timeout=timeout
        )
        if not infos:
            msg = (
                f"No LSL streams found of type={stream_type} within {timeout}s"
            )
            raise RuntimeError(msg)
        target = None
        if stream_name is None:
            target = infos[0]
        else:
            for info in infos:
                if info.name() == stream_name:
                    target = info
                    break
        if target is None:
            msg = (
                f"LSL stream named '{stream_name}' not found "
                f"(type={stream_type})"
            )
            raise RuntimeError(msg)
        self._inlet = self._pylsl.StreamInlet(target)
        self._info = LSLStreamInfo(
            name=target.name(),
            type=target.type(),
            channel_count=target.channel_count(),
            nominal_srate=target.nominal_srate(),
            source_id=(
                target.source_id() if hasattr(target, "source_id") else None
            ),
        )
        return self._info

    @property
    def info(self) -> LSLStreamInfo | None:
        return self._info

    def read(
        self, n_samples: int, timeout: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read a batch of samples and timestamps.

        Returns a tuple (data, timestamps).
        """
        if self._inlet is None or self._info is None:
            raise RuntimeError("Not connected; call connect() first.")
        data = np.zeros(
            (n_samples, self._info.channel_count), dtype=np.float32
        )
        ts = np.zeros((n_samples,), dtype=np.float64)
        filled = 0
        while filled < n_samples:
            sample, t = self._inlet.pull_sample(timeout=timeout)
            if sample is None:
                break
            data[filled] = np.asarray(sample, dtype=np.float32)
            ts[filled] = float(t)
            filled += 1
        if filled < n_samples:
            data = data[:filled]
            ts = ts[:filled]
        return data, ts

    def close(self) -> None:
        self._inlet = None
        self._info = None
