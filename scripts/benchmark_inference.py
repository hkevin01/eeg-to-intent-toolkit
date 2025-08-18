#!/usr/bin/env python3
"""Benchmark inference latency for SlidingWindowPredictor.

Generates synthetic data windows and measures per-call latency for
TorchScript or ONNX backends. Prints mean and percentile stats.

Usage (optional):
    python scripts/benchmark_inference.py \
        --model /path/to/model.onnx --backend onnx \
        --window 256 --step 32 --channels 8 --runs 200

If no model is provided, the script will create a dummy ONNX-like path
and skip actual backend load to sanity-check the timing harness.
"""
from __future__ import annotations

import argparse
import statistics as stats
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import List

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        type=str,
        default="",
        help="Path to ONNX or TorchScript model",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="onnx",
        choices=["onnx", "torchscript"],
        help="Inference backend",
    )
    p.add_argument(
        "--window",
        type=int,
        default=256,
        help="Window size (samples)",
    )
    p.add_argument("--step", type=int, default=32, help="Step size (samples)")
    p.add_argument(
        "--channels",
        type=int,
        default=8,
        help="Number of channels",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=200,
        help="Number of timing runs",
    )
    p.add_argument(
        "--classes",
        type=int,
        default=4,
        help="Assumed number of classes (for softmax shape)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure repo 'src' is on sys.path when running from checkout
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

    # Import inference module dynamically
    try:
        inf = import_module("eegintent.realtime.inference")
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Failed to import realtime inference modules: "
            f"{e}. Ensure PYTHONPATH includes ./src or install the package"
        ) from e

    Backend = inf.Backend
    PredictorConfig = inf.PredictorConfig
    SlidingWindowPredictor = inf.SlidingWindowPredictor
    SmoothingConfig = inf.SmoothingConfig

    backend = Backend.TORCHSCRIPT if args.backend == "torchscript" else Backend.ONNX

    # Prepare predictor
    if args.model:
        cfg = PredictorConfig(
            window_size=args.window,
            step_size=args.step,
            n_channels=args.channels,
            device=None,
            smoothing=SmoothingConfig(avg_last=1, refractory_ms=0),
        )
        predictor = SlidingWindowPredictor(args.model, backend, cfg)
    else:
        # When no model is provided, mock predictor wrapping process_stream
        class _MockPredictor:
            def __init__(self, window: int, classes: int) -> None:
                self.window = window
                self.classes = classes

            def process_stream(self, data: np.ndarray, _ts: np.ndarray):
                # Simulate compute with a tiny matmul and softmax
                x = data[-self.window :]
                w = np.random.randn(self.window, self.classes).astype(np.float32)
                logits = x.reshape(-1)[: self.window].astype(np.float32) @ w
                e = np.exp(logits - logits.max())
                probs = e / e.sum()
                return int(np.argmax(probs)), probs

        predictor = _MockPredictor(args.window, args.classes)

    # Synthetic data (random noise)
    N = args.window * 2 + args.step * 4
    data = (np.random.randn(N, args.channels)).astype(np.float32)
    # Assume ~250 Hz
    timestamps = np.linspace(0, N / 250.0, N).astype(np.float64)

    # Warm-up
    for _ in range(5):
        predictor.process_stream(data, timestamps)

    # Timed runs
    times_ms: List[float] = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        predictor.process_stream(data, timestamps)
        dt = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt)

    # Stats
    mean_ms = stats.mean(times_ms)
    p50 = float(np.percentile(times_ms, 50))
    p90 = float(np.percentile(times_ms, 90))
    p99 = float(np.percentile(times_ms, 99))

    print("Inference latency (ms):")
    print(f"  mean: {mean_ms:.3f}")
    print(f"  p50 : {p50:.3f}")
    print(f"  p90 : {p90:.3f}")
    print(f"  p99 : {p99:.3f}")


if __name__ == "__main__":
    main()
