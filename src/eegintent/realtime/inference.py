"""Sliding-window inference and smoothing for real-time EEG."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


class Backend(enum.Enum):
    TORCHSCRIPT = "torchscript"
    ONNX = "onnx"


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


@dataclass
class SmoothingConfig:
    avg_last: int = 3
    refractory_ms: int = 500


class SlidingWindowPredictor:
    """Windowed inference with output smoothing and refractory logic."""

    def __init__(
        self,
        model_path: str,
        backend: Backend,
        window_size: int,
        step_size: int,
        n_channels: int,
        device: Optional[str] = None,
        smoothing: Optional[SmoothingConfig] = None,
    ) -> None:
        self.model_path = model_path
        self.backend = backend
        self.window_size = int(window_size)
        self.step_size = int(step_size)
        self.n_channels = int(n_channels)
        self.device = device
        self.smoothing = smoothing or SmoothingConfig()
        self._probs_hist: list[np.ndarray] = []
        self._last_emit_ts: float = 0.0

        if self.backend == Backend.TORCHSCRIPT:
            import torch

            self._model = torch.jit.load(model_path, map_location=device or "cpu")
            self._model.eval()
        elif self.backend == Backend.ONNX:
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            self._ort = ort.InferenceSession(model_path, providers=providers)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported backend: {backend}")

    def _infer_torch(self, x: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            xt = torch.from_numpy(x[None]).float()
            if self.device:
                xt = xt.to(self.device)
                self._model.to(self.device)
            logits = self._model(xt)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return probs

    def _infer_onnx(self, x: np.ndarray) -> np.ndarray:
        inp_name = self._ort.get_inputs()[0].name  # type: ignore[attr-defined]
        out_name = self._ort.get_outputs()[0].name  # type: ignore[attr-defined]
        x_ = x.astype(np.float32)[None]
        out = self._ort.run([out_name], {inp_name: x_})[0]
        if out.ndim == 2:
            out = out[0]
        return _softmax(out, axis=-1)

    def _infer(self, x: np.ndarray) -> np.ndarray:
        if self.backend == Backend.TORCHSCRIPT:
            return self._infer_torch(x)
        return self._infer_onnx(x)

    def process_stream(
        self, data: np.ndarray, timestamps: np.ndarray
    ) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """Process incoming samples and return (predicted_class, probs) when ready.

        Accumulates windows based on step_size; returns None when insufficient data.
        """
        n = data.shape[0]
        if n < self.window_size:
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
        if (now - self._last_emit_ts) * 1000.0 < self.smoothing.refractory_ms:
            return None, avg
        self._last_emit_ts = now
        return pred, avg


"""Real-time inference engine with sliding windows and temporal smoothing."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
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
    import onnxruntime as ort
except ImportError:
    ort = None


@dataclass
class Prediction:
    """Single prediction result."""

    probabilities: np.ndarray  # Class probabilities
    predicted_class: int  # Predicted class index
    confidence: float  # Max probability
    timestamp: float  # Prediction timestamp
    latency_ms: float  # Processing latency in milliseconds


@dataclass
class InferenceConfig:
    """Configuration for real-time inference."""

    window_size_ms: int = 1000  # Window size in milliseconds
    step_size_ms: int = 100  # Step size in milliseconds
    n_classes: int = 2  # Number of output classes
    smoothing_window: int = 5  # Temporal smoothing window
    confidence_threshold: float = 0.6  # Minimum confidence for prediction
    refractory_period_ms: int = 500  # Minimum time between predictions


class SlidingWindowPredictor:
    """Sliding window inference for real-time EEG classification."""

    def __init__(
        self,
        model: nn.Module | str,  # PyTorch model or ONNX path
        config: InferenceConfig,
        sampling_rate: float,
        n_channels: int,
        device: str = "cpu",
    ):
        """Initialize sliding window predictor.

        Args:
            model: PyTorch model or path to ONNX model
            config: Inference configuration
            sampling_rate: EEG sampling rate in Hz
            n_channels: Number of EEG channels
            device: Device for PyTorch inference
        """
        self.config = config
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.device = device

        # Calculate window sizes in samples
        self.window_samples = int(config.window_size_ms * sampling_rate / 1000)
        self.step_samples = int(config.step_size_ms * sampling_rate / 1000)

        # Load model
        self.model_type = "pytorch"
        self.pytorch_model = None
        self.onnx_session = None

        if isinstance(model, str):
            # ONNX model path
            if ort is None:
                raise ImportError("onnxruntime required for ONNX inference")
            self.onnx_session = ort.InferenceSession(model)
            self.model_type = "onnx"
        else:
            # PyTorch model
            if torch is None:
                raise ImportError("torch required for PyTorch inference")
            self.pytorch_model = model
            self.pytorch_model.eval()
            if hasattr(self.pytorch_model, "to"):
                self.pytorch_model.to(device)

        # Prediction smoothing
        self.prediction_history: deque[Prediction] = deque(maxlen=config.smoothing_window)

        # Timing
        self.last_prediction_time = 0.0

        # Threading
        self.lock = threading.Lock()

        print(f"Initialized {self.model_type} predictor:")
        print(f"  Window: {config.window_size_ms}ms " f"({self.window_samples} samples)")
        print(f"  Step: {config.step_size_ms}ms ({self.step_samples} samples)")

    def _predict_pytorch(self, data: np.ndarray) -> np.ndarray:
        """Run PyTorch model inference."""
        if self.pytorch_model is None or torch is None:
            raise RuntimeError("PyTorch model not available")

        # Convert to tensor
        x = torch.from_numpy(data).float()
        if self.device != "cpu":
            x = x.to(self.device)

        # Add batch dimension if needed
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (1, n_channels, n_samples)

        with torch.no_grad():
            outputs = self.pytorch_model(x)

            # Apply softmax if not already applied
            if hasattr(torch.nn.functional, "softmax"):
                probs = torch.nn.functional.softmax(outputs, dim=-1)
            else:
                probs = outputs

            return probs.cpu().numpy().squeeze()

    def _predict_onnx(self, data: np.ndarray) -> np.ndarray:
        """Run ONNX model inference."""
        if self.onnx_session is None:
            raise RuntimeError("ONNX session not available")

        # Prepare input
        if data.ndim == 2:
            data = data[np.newaxis, ...]  # Add batch dimension

        input_name = self.onnx_session.get_inputs()[0].name
        outputs = self.onnx_session.run(None, {input_name: data.astype(np.float32)})

        return outputs[0].squeeze()

    def predict_window(self, data: np.ndarray) -> Prediction | None:
        """Make prediction on a data window.

        Args:
            data: EEG data window (n_channels, window_samples)

        Returns:
            Prediction result or None if conditions not met
        """
        start_time = time.time()

        # Check refractory period
        current_time = time.time()
        if current_time - self.last_prediction_time < self.config.refractory_period_ms / 1000:
            return None

        # Validate input shape
        if data.shape != (self.n_channels, self.window_samples):
            return None

        try:
            # Run inference
            if self.model_type == "pytorch":
                probabilities = self._predict_pytorch(data)
            else:
                probabilities = self._predict_onnx(data)

            # Extract prediction info
            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])

            # Check confidence threshold
            if confidence < self.config.confidence_threshold:
                return None

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Create prediction
            prediction = Prediction(
                probabilities=probabilities,
                predicted_class=predicted_class,
                confidence=confidence,
                timestamp=current_time,
                latency_ms=latency_ms,
            )

            # Update prediction history
            with self.lock:
                self.prediction_history.append(prediction)
                self.last_prediction_time = current_time

            return prediction

        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def get_smoothed_prediction(self) -> Prediction | None:
        """Get temporally smoothed prediction.

        Returns:
            Smoothed prediction or None if not enough history
        """
        with self.lock:
            if len(self.prediction_history) < 2:
                return None

            # Get recent predictions
            recent_predictions = list(self.prediction_history)

            # Average probabilities
            prob_matrix = np.array([p.probabilities for p in recent_predictions])
            avg_probabilities = np.mean(prob_matrix, axis=0)

            # Get smoothed prediction
            predicted_class = int(np.argmax(avg_probabilities))
            confidence = float(avg_probabilities[predicted_class])

            # Use most recent timestamp and latency
            latest = recent_predictions[-1]

            return Prediction(
                probabilities=avg_probabilities,
                predicted_class=predicted_class,
                confidence=confidence,
                timestamp=latest.timestamp,
                latency_ms=latest.latency_ms,
            )

    def reset(self) -> None:
        """Reset prediction history."""
        with self.lock:
            self.prediction_history.clear()
            self.last_prediction_time = 0.0


class RealtimeInferenceEngine:
    """Complete real-time inference engine."""

    def __init__(
        self,
        predictor: SlidingWindowPredictor,
        min_data_duration: float = 2.0,  # Minimum data duration before starting
    ):
        """Initialize inference engine.

        Args:
            predictor: Sliding window predictor
            min_data_duration: Minimum data duration in seconds
        """
        self.predictor = predictor
        self.min_data_duration = min_data_duration
        self.min_samples = int(min_data_duration * predictor.sampling_rate)

        # State
        self.is_running = False
        self.data_buffer: deque[tuple[np.ndarray, float]] = deque()

        # Threading
        self._inference_thread = None
        self._stop_event = threading.Event()
        self.lock = threading.Lock()

        # Callbacks
        self.prediction_callback = None

        # Stats
        self.total_predictions = 0
        self.total_latency = 0.0

    def set_prediction_callback(self, callback) -> None:
        """Set callback for predictions.

        Args:
            callback: Function(prediction) called for each prediction
        """
        self.prediction_callback = callback

    def add_data(self, data: np.ndarray, timestamp: float) -> None:
        """Add new EEG data for inference.

        Args:
            data: EEG sample (n_channels,)
            timestamp: Sample timestamp
        """
        with self.lock:
            self.data_buffer.append((data, timestamp))

            # Keep only recent data
            max_samples = int(
                self.predictor.sampling_rate * (self.predictor.config.window_size_ms / 1000 + 5.0)
            )

            while len(self.data_buffer) > max_samples:
                self.data_buffer.popleft()

    def start_inference(self) -> bool:
        """Start real-time inference in background thread.

        Returns:
            True if started successfully
        """
        if self.is_running:
            return True

        self._stop_event.clear()
        self._inference_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True,
        )
        self._inference_thread.start()
        self.is_running = True

        print("Started real-time inference")
        return True

    def stop_inference(self) -> None:
        """Stop real-time inference."""
        if not self.is_running:
            return

        self._stop_event.set()
        if self._inference_thread and self._inference_thread.is_alive():
            self._inference_thread.join(timeout=1.0)

        self.is_running = False
        print("Stopped real-time inference")

    def _inference_loop(self) -> None:
        """Main inference loop (runs in background thread)."""
        step_interval = self.predictor.config.step_size_ms / 1000

        while not self._stop_event.is_set():
            try:
                # Check if we have enough data
                with self.lock:
                    if len(self.data_buffer) < self.min_samples:
                        time.sleep(step_interval)
                        continue

                    # Extract window
                    buffer_data = list(self.data_buffer)

                # Get latest window
                window_data = np.array([sample[0] for sample in buffer_data])
                window_data = window_data.T  # (n_channels, n_samples)

                # Extract window of appropriate size
                if window_data.shape[1] >= self.predictor.window_samples:
                    start_idx = window_data.shape[1] - self.predictor.window_samples
                    window = window_data[:, start_idx:]

                    # Make prediction
                    prediction = self.predictor.predict_window(window)

                    if prediction:
                        self.total_predictions += 1
                        self.total_latency += prediction.latency_ms

                        # Call callback
                        if self.prediction_callback:
                            self.prediction_callback(prediction)

                # Sleep until next step
                time.sleep(step_interval)

            except Exception as e:
                print(f"Inference loop error: {e}")
                time.sleep(step_interval)

    def get_stats(self) -> dict[str, float]:
        """Get inference statistics.

        Returns:
            Dictionary with performance stats
        """
        if self.total_predictions == 0:
            return {
                "total_predictions": 0,
                "avg_latency_ms": 0.0,
                "predictions_per_second": 0.0,
            }

        avg_latency = self.total_latency / self.total_predictions

        return {
            "total_predictions": self.total_predictions,
            "avg_latency_ms": avg_latency,
            "predictions_per_second": self.total_predictions / 60.0,  # Rough estimate
        }


def create_torchscript_model(
    model: nn.Module, example_input: torch.Tensor
) -> torch.jit.ScriptModule:
    """Convert PyTorch model to TorchScript for faster inference.

    Args:
        model: PyTorch model
        example_input: Example input tensor for tracing

    Returns:
        TorchScript model
    """
    if torch is None:
        raise ImportError("torch required for TorchScript")

    model.eval()
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)

    return traced_model


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Real-time inference example")
    parser.add_argument("--model", type=str, help="Path to ONNX model")
    parser.add_argument("--duration", type=float, default=10.0, help="Test duration")

    args = parser.parse_args()

    if args.model:
        # Test with ONNX model
        config = InferenceConfig()
        predictor = SlidingWindowPredictor(
            model=args.model,
            config=config,
            sampling_rate=250.0,
            n_channels=8,
        )

        engine = RealtimeInferenceEngine(predictor)

        def print_prediction(pred: Prediction) -> None:
            print(
                f"Prediction: Class {pred.predicted_class} "
                f"({pred.confidence:.3f} confidence, "
                f"{pred.latency_ms:.1f}ms latency)"
            )

        engine.set_prediction_callback(print_prediction)
        engine.start_inference()

        # Simulate data
        print("Simulating EEG data...")
        for i in range(int(args.duration * 250)):  # 250 Hz
            fake_data = np.random.randn(8) * 0.1
            timestamp = time.time()
            engine.add_data(fake_data, timestamp)
            time.sleep(1 / 250)  # 250 Hz

        engine.stop_inference()
        stats = engine.get_stats()
        print(f"Stats: {stats}")

    else:
        print("Please provide --model path to test inference")
