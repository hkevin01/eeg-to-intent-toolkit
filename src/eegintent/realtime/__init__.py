"""Real-time EEG streaming and inference utilities.

Modules
-------
- lsl_client: LSL-based stream discovery and receiving.
- buffers: Lock-free ring buffers with timestamp alignment.
- dsp: Streaming-friendly DSP filters (IIR/FIR, notch) and artifact gating.
- inference: Sliding-window inference with smoothing and refractory logic.
"""

from .buffers import RingBuffer, TimestampedRingBuffer
from .dsp import AdaptiveNoiseGate, IIRBandpass, NotchFilter, StreamingPipeline
from .inference import Backend, PredictorConfig, SlidingWindowPredictor, SmoothingConfig
from .lsl_client import LSLNotAvailableError, LSLReceiver, LSLStreamInfo

__all__ = [
    "AdaptiveNoiseGate",
    "Backend",
    "IIRBandpass",
    "LSLNotAvailableError",
    "LSLReceiver",
    "LSLStreamInfo",
    "NotchFilter",
    "PredictorConfig",
    "RingBuffer",
    "SlidingWindowPredictor",
    "SmoothingConfig",
    "StreamingPipeline",
    "TimestampedRingBuffer",
]
"""Realtime modules: LSL client, DSP, inference, app."""
