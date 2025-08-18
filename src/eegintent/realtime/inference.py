"""Sliding-window inference and smoothing for real-time EEG."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from importlib import import_module

import numpy as np


class Backend(enum.Enum):
    TORCHSCRIPT = "torchscript"
    ONNX = "onnx"


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax for numpy arrays."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


@dataclass
class SmoothingConfig:
    avg_last: int = 3
    refractory_ms: int = 500


@dataclass
class PredictorConfig:
    """Configuration for SlidingWindowPredictor."""

    window_size: int
    step_size: int
    n_channels: int
    device: str | None = None
    smoothing: SmoothingConfig | None = None


class SlidingWindowPredictor:
    """Windowed inference with output smoothing and refractory logic.

    Contract
    - Inputs: data [N, C], timestamps [N]; requires N>=window_size
    - Outputs: (pred_class | None, avg_probs | None)
        - Emits at most once per refractory_ms and when step_size new samples
            arrive
    """

    def __init__(self, model_path: str, backend: Backend, cfg: PredictorConfig) -> None:
        self.model_path = model_path
        self.backend = backend
        self.window_size = int(cfg.window_size)
        self.step_size = int(cfg.step_size)
        self.n_channels = int(cfg.n_channels)
        self.device = cfg.device
        self.smoothing = cfg.smoothing or SmoothingConfig()
        self._probs_hist: list[np.ndarray] = []
        self._last_emit_ts: float = 0.0
        self._last_infer_end: int = 0

        if self.backend == Backend.TORCHSCRIPT:
            # Lazy import via import_module to avoid hard dependency if unused
            torch = import_module("torch")
            self._torch = torch  # stash for later use
            map_loc = cfg.device or "cpu"
            self._model = torch.jit.load(model_path, map_location=map_loc)
            self._model.eval()
        elif self.backend == Backend.ONNX:
            # Lazy import via import_module to avoid hard dependency if unused
            ort = import_module("onnxruntime")
            providers = ["CPUExecutionProvider"]
            self._ort_sess = ort.InferenceSession(model_path, providers=providers)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported backend: {backend}")

    def _infer_torch(self, x: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            xt = torch.from_numpy(x[None]).float()
            if self.device:
                xt = xt.to(self.device)
                self._model.to(self.device)
            logits = self._model(xt)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return probs

    def _infer_onnx(self, x: np.ndarray) -> np.ndarray:
        # Names for first input/output; typical single-input classification
        inp = self._ort_sess.get_inputs()[0]  # type: ignore[attr-defined]
        out = self._ort_sess.get_outputs()[0]  # type: ignore[attr-defined]
        inp_name = inp.name
        out_name = out.name
        x_ = x.astype(np.float32)[None]
        out_arr = self._ort_sess.run([out_name], {inp_name: x_})[0]
        batched_ndims = 2
        if out_arr.ndim == batched_ndims:
            out_arr = out_arr[0]
        return _softmax(out_arr, axis=-1)

    def _infer(self, x: np.ndarray) -> np.ndarray:
        if self.backend == Backend.TORCHSCRIPT:
            return self._infer_torch(x)
        return self._infer_onnx(x)

    def process_stream(
        self, data: np.ndarray, timestamps: np.ndarray
    ) -> tuple[int | None, np.ndarray | None]:
        """Process incoming samples and return a prediction when ready.

        Uses the last full window [N-window_size:N] when at least step_size
        new samples have arrived since the last inference. Applies
        averaging over the last K probability vectors and respects a
        refractory window for emitting discrete predictions.
        """
        n = int(data.shape[0])
        if n < self.window_size:
            return None, None

        # Enforce step size relative to last inference end
        if n - self._last_infer_end < self.step_size:
            return None, None

        # Use last full window
        x = data[-self.window_size :]
        probs = self._infer(x)
        self._probs_hist.append(probs)
        if len(self._probs_hist) > self.smoothing.avg_last:
            self._probs_hist.pop(0)
        avg = np.mean(np.stack(self._probs_hist, axis=0), axis=0)
        pred = int(np.argmax(avg))

        now = float(timestamps[-1]) if timestamps.size else time.time()
        self._last_infer_end = n

        # Refractory gate for emitting discrete predictions
        if (now - self._last_emit_ts) * 1000.0 < self.smoothing.refractory_ms:
            return None, avg

        self._last_emit_ts = now
        return pred, avg
