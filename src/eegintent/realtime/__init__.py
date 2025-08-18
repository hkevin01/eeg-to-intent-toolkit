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
from .inference import Backend, SlidingWindowPredictor
from .lsl_client import LSLNotAvailableError, LSLReceiver, LSLStreamInfo

__all__ = [
    "LSLReceiver",
    "LSLStreamInfo",
    "LSLNotAvailableError",
    "RingBuffer",
    "TimestampedRingBuffer",
    "IIRBandpass",
    "NotchFilter",
    "AdaptiveNoiseGate",
    "StreamingPipeline",
    "SlidingWindowPredictor",
    "Backend",
]
"""Realtime modules: LSL client, DSP, inference, app."""
