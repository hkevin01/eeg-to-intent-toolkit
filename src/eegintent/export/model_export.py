"""Model export utilities for ONNX and TorchScript deployment."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch
    from torch import nn
else:
    # Handle optional dependencies
    try:
        import torch
        from torch import nn
    except ImportError:
        torch = None
        nn = None

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    onnx = None
    ort = None


def export_to_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    export_path: str | Path,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict | None = None,
    opset_version: int = 11,
) -> bool:
    """Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        dummy_input: Example input tensor for tracing
        export_path: Path to save ONNX model
        input_names: Names for input tensors
        output_names: Names for output tensors
        dynamic_axes: Dynamic axes specification
        opset_version: ONNX opset version

    Returns:
        True if export successful
    """
    if torch is None:
        raise ImportError("torch required for ONNX export")

    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    # Set default names
    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    try:
        model.eval()

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                str(export_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )

        print(f"Successfully exported ONNX model to {export_path}")
        return True

    except Exception as e:
        print(f"ONNX export failed: {e}")
        return False


def export_to_torchscript(
    model: nn.Module,
    dummy_input: torch.Tensor,
    export_path: str | Path,
    method: str = "trace",
) -> bool:
    """Export PyTorch model to TorchScript format.

    Args:
        model: PyTorch model to export
        dummy_input: Example input tensor (for tracing)
        export_path: Path to save TorchScript model
        method: Export method ("trace" or "script")

    Returns:
        True if export successful
    """
    if torch is None:
        raise ImportError("torch required for TorchScript export")

    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        model.eval()

        if method == "trace":
            with torch.no_grad():
                traced_model = torch.jit.trace(model, dummy_input)
        elif method == "script":
            traced_model = torch.jit.script(model)
        else:
            raise ValueError(f"Unknown method: {method}")

        traced_model.save(str(export_path))

        print(f"Successfully exported TorchScript model to {export_path}")
        return True

    except Exception as e:
        print(f"TorchScript export failed: {e}")
        return False


def validate_onnx_model(
    onnx_path: str | Path,
    pytorch_model: nn.Module,
    dummy_input: torch.Tensor,
    tolerance: float = 1e-5,
) -> bool:
    """Validate ONNX model against original PyTorch model.

    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model
        dummy_input: Test input tensor
        tolerance: Numerical tolerance for comparison

    Returns:
        True if outputs match within tolerance
    """
    if torch is None or ort is None:
        print("torch and onnxruntime required for validation")
        return False

    try:
        # PyTorch prediction
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(dummy_input).numpy()

        # ONNX prediction
        session = ort.InferenceSession(str(onnx_path))
        input_name = session.get_inputs()[0].name
        onnx_output = session.run(None, {input_name: dummy_input.numpy()})[0]

        # Compare outputs
        diff = np.abs(pytorch_output - onnx_output)
        max_diff = np.max(diff)

        if max_diff < tolerance:
            print(f"✓ ONNX validation passed (max diff: {max_diff:.2e})")
            return True
        else:
            print(f"✗ ONNX validation failed (max diff: {max_diff:.2e})")
            return False

    except Exception as e:
        print(f"ONNX validation error: {e}")
        return False


def benchmark_model_latency(
    model_path: str | Path,
    input_shape: tuple[int, ...],
    model_type: str = "onnx",
    n_runs: int = 100,
    warmup_runs: int = 10,
    device: str = "cpu",
) -> dict[str, float]:
    """Benchmark model inference latency.

    Args:
        model_path: Path to model file
        input_shape: Shape of input tensor (batch_size, ...)
        model_type: Model type ("onnx" or "torchscript")
        n_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        device: Device for inference

    Returns:
        Dictionary with timing statistics
    """
    if model_type == "onnx":
        return _benchmark_onnx_latency(model_path, input_shape, n_runs, warmup_runs)
    elif model_type == "torchscript":
        return _benchmark_torchscript_latency(model_path, input_shape, n_runs, warmup_runs, device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _benchmark_onnx_latency(
    model_path: str | Path,
    input_shape: tuple[int, ...],
    n_runs: int,
    warmup_runs: int,
) -> dict[str, float]:
    """Benchmark ONNX model latency."""
    if ort is None:
        raise ImportError("onnxruntime required for ONNX benchmarking")

    # Load model
    session = ort.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name

    # Create dummy input
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup_runs):
        session.run(None, {input_name: dummy_input})

    # Benchmark
    latencies = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)  # Convert to ms

    return _compute_latency_stats(latencies)


def _benchmark_torchscript_latency(
    model_path: str | Path,
    input_shape: tuple[int, ...],
    n_runs: int,
    warmup_runs: int,
    device: str,
) -> dict[str, float]:
    """Benchmark TorchScript model latency."""
    if torch is None:
        raise ImportError("torch required for TorchScript benchmarking")

    # Load model
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            model(dummy_input)

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            start_time = time.perf_counter()
            model(dummy_input)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to ms

    return _compute_latency_stats(latencies)


def _compute_latency_stats(latencies: list[float]) -> dict[str, float]:
    """Compute latency statistics."""
    latencies = np.array(latencies)

    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
    }


def create_deployment_config(
    model_path: str | Path,
    model_type: str,
    input_shape: tuple[int, ...],
    output_classes: list[str],
    sampling_rate: float,
    window_size_ms: int,
    preprocessing_config: dict | None = None,
) -> dict:
    """Create deployment configuration file.

    Args:
        model_path: Path to exported model
        model_type: Type of model ("onnx" or "torchscript")
        input_shape: Expected input shape
        output_classes: List of class names
        sampling_rate: EEG sampling rate
        window_size_ms: Inference window size in milliseconds
        preprocessing_config: Preprocessing configuration

    Returns:
        Deployment configuration dictionary
    """
    config = {
        "model": {
            "path": str(model_path),
            "type": model_type,
            "input_shape": list(input_shape),
            "output_classes": output_classes,
        },
        "signal": {
            "sampling_rate": sampling_rate,
            "window_size_ms": window_size_ms,
            "n_channels": input_shape[1] if len(input_shape) > 1 else 1,
        },
        "inference": {
            "confidence_threshold": 0.6,
            "smoothing_window": 5,
            "refractory_period_ms": 500,
        },
        "preprocessing": preprocessing_config
        or {
            "bandpass_low": 1.0,
            "bandpass_high": 50.0,
            "notch_freq": 50.0,
            "use_car": True,
            "use_adaptive_noise": True,
        },
    }

    return config


def export_eeg_model_complete(
    model: nn.Module,
    save_dir: str | Path,
    model_name: str,
    n_channels: int = 8,
    n_classes: int = 2,
    sampling_rate: float = 250.0,
    window_size_ms: int = 1000,
    class_names: list[str] | None = None,
) -> dict[str, str]:
    """Complete export pipeline for EEG classification model.

    Args:
        model: PyTorch model to export
        save_dir: Directory to save exported files
        model_name: Base name for exported files
        n_channels: Number of EEG channels
        n_classes: Number of output classes
        sampling_rate: EEG sampling rate
        window_size_ms: Inference window size
        class_names: Names of output classes

    Returns:
        Dictionary with paths to exported files
    """
    if torch is None:
        raise ImportError("torch required for model export")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Calculate input shape
    window_samples = int(window_size_ms * sampling_rate / 1000)
    input_shape = (1, n_channels, window_samples)
    dummy_input = torch.randn(*input_shape)

    # Default class names
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(n_classes)]

    exported_files = {}

    # Export to ONNX
    onnx_path = save_dir / f"{model_name}.onnx"
    if export_to_onnx(model, dummy_input, onnx_path):
        exported_files["onnx"] = str(onnx_path)

        # Validate ONNX
        if validate_onnx_model(onnx_path, model, dummy_input):
            # Benchmark ONNX
            stats = benchmark_model_latency(onnx_path, input_shape, "onnx")
            print(f"ONNX benchmark: {stats['mean_ms']:.2f}ms average")

    # Export to TorchScript
    torchscript_path = save_dir / f"{model_name}.pt"
    if export_to_torchscript(model, dummy_input, torchscript_path):
        exported_files["torchscript"] = str(torchscript_path)

        # Benchmark TorchScript
        stats = benchmark_model_latency(torchscript_path, input_shape, "torchscript")
        print(f"TorchScript benchmark: {stats['mean_ms']:.2f}ms average")

    # Create deployment config
    config = create_deployment_config(
        onnx_path if "onnx" in exported_files else torchscript_path,
        "onnx" if "onnx" in exported_files else "torchscript",
        input_shape,
        class_names,
        sampling_rate,
        window_size_ms,
    )

    # Save config
    config_path = save_dir / f"{model_name}_config.json"
    import json

    with config_path.open("w") as f:
        json.dump(config, f, indent=2)

    exported_files["config"] = str(config_path)

    print(f"Export complete. Files saved to {save_dir}")
    return exported_files


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Export EEG model")
    parser.add_argument("--model", type=str, required=True, help="PyTorch model path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--name", type=str, default="eeg_model", help="Model name")

    args = parser.parse_args()

    if torch is not None:
        # Load model (example)
        model = torch.load(args.model, map_location="cpu")

        # Export
        exported = export_eeg_model_complete(
            model=model,
            save_dir=args.output,
            model_name=args.name,
        )

        print(f"Exported files: {exported}")
    else:
        print("PyTorch not available for export")
