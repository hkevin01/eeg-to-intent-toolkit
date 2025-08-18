"""Real-time DSP filters and processing for EEG signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from scipy import signal
else:
    # Handle optional scipy dependency
    try:
        from scipy import signal
    except ImportError:
        signal = None


@dataclass
class FilterState:
    """State information for real-time IIR filters."""

    zi: np.ndarray  # Initial conditions for filter
    b: np.ndarray  # Numerator coefficients
    a: np.ndarray  # Denominator coefficients


class RealtimeFilter:
    """Base class for real-time filters."""

    def __init__(self, n_channels: int):
        """Initialize filter.

        Args:
            n_channels: Number of EEG channels
        """
        self.n_channels = n_channels
        self.is_initialized = False

    def reset(self) -> None:
        """Reset filter state."""
        self.is_initialized = False

    def process_sample(self, sample: np.ndarray) -> np.ndarray:
        """Process a single sample.

        Args:
            sample: Input sample (n_channels,)

        Returns:
            Filtered sample (n_channels,)
        """
        raise NotImplementedError

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Process a chunk of samples.

        Args:
            chunk: Input chunk (n_channels, n_samples)

        Returns:
            Filtered chunk (n_channels, n_samples)
        """
        n_channels, n_samples = chunk.shape
        output = np.zeros_like(chunk)

        for i in range(n_samples):
            output[:, i] = self.process_sample(chunk[:, i])

        return output


class BandpassFilter(RealtimeFilter):
    """Real-time bandpass filter using IIR (Butterworth)."""

    def __init__(
        self,
        n_channels: int,
        low_freq: float,
        high_freq: float,
        sampling_rate: float,
        order: int = 4,
    ):
        """Initialize bandpass filter.

        Args:
            n_channels: Number of EEG channels
            low_freq: Low cutoff frequency in Hz
            high_freq: High cutoff frequency in Hz
            sampling_rate: Sampling rate in Hz
            order: Filter order
        """
        super().__init__(n_channels)

        if signal is None:
            raise ImportError("scipy.signal required for filtering")

        self.low_freq = low_freq
        self.high_freq = high_freq
        self.sampling_rate = sampling_rate
        self.order = order

        # Design filter
        nyquist = sampling_rate / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist

        self.b, self.a = signal.butter(order, [low_norm, high_norm], btype="band")

        # Initialize filter states for each channel
        self.filter_states: list[np.ndarray] = []
        for _ in range(n_channels):
            zi = signal.lfilter_zi(self.b, self.a)
            self.filter_states.append(zi)

    def reset(self) -> None:
        """Reset filter state."""
        super().reset()
        if signal is not None:
            for i in range(self.n_channels):
                self.filter_states[i] = signal.lfilter_zi(self.b, self.a)

    def process_sample(self, sample: np.ndarray) -> np.ndarray:
        """Process a single sample through bandpass filter."""
        if signal is None:
            return sample

        output = np.zeros_like(sample)

        for ch in range(self.n_channels):
            # Apply IIR filter to single sample
            y, self.filter_states[ch] = signal.lfilter(
                self.b, self.a, [sample[ch]], zi=self.filter_states[ch]
            )
            output[ch] = y[0]

        return output


class NotchFilter(RealtimeFilter):
    """Real-time notch filter for powerline interference."""

    def __init__(
        self,
        n_channels: int,
        notch_freq: float,
        sampling_rate: float,
        quality_factor: float = 30.0,
    ):
        """Initialize notch filter.

        Args:
            n_channels: Number of EEG channels
            notch_freq: Notch frequency in Hz (e.g., 50 or 60)
            sampling_rate: Sampling rate in Hz
            quality_factor: Quality factor (higher = narrower notch)
        """
        super().__init__(n_channels)

        if signal is None:
            raise ImportError("scipy.signal required for filtering")

        self.notch_freq = notch_freq
        self.sampling_rate = sampling_rate
        self.quality_factor = quality_factor

        # Design notch filter
        nyquist = sampling_rate / 2
        freq_norm = notch_freq / nyquist

        self.b, self.a = signal.iirnotch(freq_norm, quality_factor)

        # Initialize filter states
        self.filter_states: list[np.ndarray] = []
        for _ in range(n_channels):
            zi = signal.lfilter_zi(self.b, self.a)
            self.filter_states.append(zi)

    def reset(self) -> None:
        """Reset filter state."""
        super().reset()
        if signal is not None:
            for i in range(self.n_channels):
                self.filter_states[i] = signal.lfilter_zi(self.b, self.a)

    def process_sample(self, sample: np.ndarray) -> np.ndarray:
        """Process a single sample through notch filter."""
        if signal is None:
            return sample

        output = np.zeros_like(sample)

        for ch in range(self.n_channels):
            y, self.filter_states[ch] = signal.lfilter(
                self.b, self.a, [sample[ch]], zi=self.filter_states[ch]
            )
            output[ch] = y[0]

        return output


class CommonAverageReference(RealtimeFilter):
    """Real-time Common Average Reference (CAR) filter."""

    def __init__(self, n_channels: int):
        """Initialize CAR filter.

        Args:
            n_channels: Number of EEG channels
        """
        super().__init__(n_channels)

    def process_sample(self, sample: np.ndarray) -> np.ndarray:
        """Apply CAR to a single sample."""
        # Subtract average across all channels
        average = np.mean(sample)
        return sample - average


class AdaptiveNoiseFilter(RealtimeFilter):
    """Adaptive noise reduction filter using moving statistics."""

    def __init__(
        self,
        n_channels: int,
        window_size: int = 50,
        threshold_std: float = 3.0,
    ):
        """Initialize adaptive noise filter.

        Args:
            n_channels: Number of EEG channels
            window_size: Size of moving window for statistics
            threshold_std: Threshold in standard deviations for outlier detection
        """
        super().__init__(n_channels)
        self.window_size = window_size
        self.threshold_std = threshold_std

        # Circular buffers for each channel
        self.buffers = [np.zeros(window_size) for _ in range(n_channels)]
        self.buffer_indices = [0] * n_channels
        self.buffer_filled = [False] * n_channels

    def reset(self) -> None:
        """Reset filter state."""
        super().reset()
        for ch in range(self.n_channels):
            self.buffers[ch].fill(0)
            self.buffer_indices[ch] = 0
            self.buffer_filled[ch] = False

    def process_sample(self, sample: np.ndarray) -> np.ndarray:
        """Process sample with adaptive noise reduction."""
        output = sample.copy()

        for ch in range(self.n_channels):
            # Update circular buffer
            idx = self.buffer_indices[ch]
            self.buffers[ch][idx] = sample[ch]
            self.buffer_indices[ch] = (idx + 1) % self.window_size

            if idx == self.window_size - 1:
                self.buffer_filled[ch] = True

            # Apply noise reduction if enough samples
            if self.buffer_filled[ch]:
                buffer_data = self.buffers[ch]
                mean_val = np.mean(buffer_data)
                std_val = np.std(buffer_data)

                # Check if current sample is an outlier
                if abs(sample[ch] - mean_val) > self.threshold_std * std_val:
                    # Replace with median of recent samples
                    output[ch] = np.median(buffer_data[-10:])

        return output


class FilterPipeline:
    """Pipeline of multiple real-time filters."""

    def __init__(self, n_channels: int):
        """Initialize filter pipeline.

        Args:
            n_channels: Number of EEG channels
        """
        self.n_channels = n_channels
        self.filters: list[RealtimeFilter] = []

    def add_filter(self, filter_obj: RealtimeFilter) -> None:
        """Add a filter to the pipeline.

        Args:
            filter_obj: Filter to add
        """
        if filter_obj.n_channels != self.n_channels:
            raise ValueError(
                f"Filter has {filter_obj.n_channels} channels, "
                f"pipeline expects {self.n_channels}"
            )
        self.filters.append(filter_obj)

    def process_sample(self, sample: np.ndarray) -> np.ndarray:
        """Process sample through all filters in sequence."""
        output = sample.copy()
        for filt in self.filters:
            output = filt.process_sample(output)
        return output

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Process chunk through all filters in sequence."""
        output = chunk.copy()
        for filt in self.filters:
            output = filt.process_chunk(output)
        return output

    def reset(self) -> None:
        """Reset all filters in the pipeline."""
        for filt in self.filters:
            filt.reset()


def create_standard_eeg_pipeline(
    n_channels: int,
    sampling_rate: float,
    low_freq: float = 1.0,
    high_freq: float = 50.0,
    notch_freq: float = 50.0,
    use_car: bool = True,
    use_adaptive: bool = True,
) -> FilterPipeline:
    """Create a standard EEG preprocessing pipeline.

    Args:
        n_channels: Number of EEG channels
        sampling_rate: Sampling rate in Hz
        low_freq: Low-pass cutoff frequency
        high_freq: High-pass cutoff frequency
        notch_freq: Notch filter frequency (50/60 Hz)
        use_car: Whether to apply Common Average Reference
        use_adaptive: Whether to apply adaptive noise reduction

    Returns:
        Configured filter pipeline
    """
    pipeline = FilterPipeline(n_channels)

    # Bandpass filter (1-50 Hz typical for EEG)
    if signal is not None:
        bandpass = BandpassFilter(n_channels, low_freq, high_freq, sampling_rate)
        pipeline.add_filter(bandpass)

    # Notch filter for powerline interference
    if signal is not None:
        notch = NotchFilter(n_channels, notch_freq, sampling_rate)
        pipeline.add_filter(notch)

    # Common Average Reference
    if use_car:
        car = CommonAverageReference(n_channels)
        pipeline.add_filter(car)

    # Adaptive noise reduction
    if use_adaptive:
        adaptive = AdaptiveNoiseFilter(n_channels)
        pipeline.add_filter(adaptive)

    return pipeline


def sliding_mean(x: np.ndarray, k: int = 3) -> np.ndarray:
    """Apply sliding mean filter (legacy function)."""
    if k <= 1:
        return x
    pad = k // 2
    xpad = np.pad(x, ((0, 0), (0, 0), (pad, pad)), mode="edge")
    out = np.empty_like(x)
    for i in range(x.shape[-1]):
        out[..., i] = xpad[..., i : i + k].mean(axis=-1)
    return out
