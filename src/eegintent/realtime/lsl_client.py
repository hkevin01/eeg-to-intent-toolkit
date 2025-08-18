"""Lab Streaming Layer (LSL) client for real-time EEG acquisition."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

try:
    import pylsl as lsl
except ImportError:
    lsl = None


@dataclass
class StreamInfo:
    """Information about an LSL stream."""

    name: str
    type: str
    channel_count: int
    sampling_rate: float
    channel_format: str
    source_id: str
    hostname: str
    uid: str


@dataclass
class EEGSample:
    """Single EEG sample with metadata."""

    data: np.ndarray  # Shape: (n_channels,)
    timestamp: float  # LSL timestamp
    local_clock: float  # Local system time
    sample_id: int  # Sequential sample ID


class RingBuffer:
    """Thread-safe ring buffer for EEG samples."""

    def __init__(self, capacity: int):
        """Initialize ring buffer.

        Args:
            capacity: Maximum number of samples to store
        """
        self.capacity = capacity
        self.buffer: deque[EEGSample] = deque(maxlen=capacity)
        self.lock = threading.Lock()
        self._sample_counter = 0

    def put(self, data: np.ndarray, timestamp: float) -> None:
        """Add a sample to the buffer.

        Args:
            data: EEG data for this sample
            timestamp: LSL timestamp
        """
        sample = EEGSample(
            data=data.copy(),
            timestamp=timestamp,
            local_clock=time.time(),
            sample_id=self._sample_counter,
        )

        with self.lock:
            self.buffer.append(sample)
            self._sample_counter += 1

    def get_latest(self, n_samples: int = 1) -> list[EEGSample]:
        """Get the most recent samples.

        Args:
            n_samples: Number of samples to retrieve

        Returns:
            List of EEG samples (most recent first)
        """
        with self.lock:
            if len(self.buffer) == 0:
                return []

            # Get last n samples
            n_available = min(n_samples, len(self.buffer))
            return list(self.buffer)[-n_available:]

    def get_window(
        self,
        duration: float,
        sampling_rate: float,
        end_time: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get a time window of EEG data.

        Args:
            duration: Window duration in seconds
            sampling_rate: Expected sampling rate in Hz
            end_time: End timestamp (None for most recent)

        Returns:
            Tuple of (data_matrix, timestamps)
            data_matrix: Shape (n_channels, n_samples)
            timestamps: Shape (n_samples,)
        """
        with self.lock:
            if len(self.buffer) == 0:
                return np.array([]), np.array([])

            samples = list(self.buffer)

            # Determine end time
            if end_time is None:
                end_time = samples[-1].timestamp

            start_time = end_time - duration

            # Filter samples in time window
            window_samples = [s for s in samples if start_time <= s.timestamp <= end_time]

            if not window_samples:
                return np.array([]), np.array([])

            # Extract data and timestamps
            data_list = [s.data for s in window_samples]
            timestamps = np.array([s.timestamp for s in window_samples])

            # Stack data: (n_samples, n_channels) -> (n_channels, n_samples)
            data_matrix = np.stack(data_list, axis=0).T

            return data_matrix, timestamps

    def clear(self) -> None:
        """Clear all samples from the buffer."""
        with self.lock:
            self.buffer.clear()
            self._sample_counter = 0

    def __len__(self) -> int:
        """Get current number of samples in buffer."""
        with self.lock:
            return len(self.buffer)


class LSLReceiver:
    """LSL client for receiving EEG streams."""

    def __init__(
        self,
        stream_name: str | None = None,
        stream_type: str = "EEG",
        buffer_duration: float = 10.0,
        timeout: float = 5.0,
    ):
        """Initialize LSL receiver.

        Args:
            stream_name: Name of stream to connect to (None for first found)
            stream_type: Type of stream to look for
            buffer_duration: Buffer duration in seconds
            timeout: Timeout for stream discovery in seconds
        """
        if lsl is None:
            raise ImportError("pylsl not available. Install with: pip install pylsl")

        self.stream_name = stream_name
        self.stream_type = stream_type
        self.buffer_duration = buffer_duration
        self.timeout = timeout

        self.inlet: lsl.StreamInlet | None = None
        self.stream_info: StreamInfo | None = None
        self.buffer: RingBuffer | None = None
        self.is_connected = False
        self.is_receiving = False

        # Threading
        self._receive_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Callbacks
        self.data_callback: Callable[[np.ndarray, float], None] | None = None

    def discover_streams(self) -> list[StreamInfo]:
        """Discover available LSL streams.

        Returns:
            List of available stream information
        """
        if lsl is None:
            return []

        print(f"Looking for {self.stream_type} streams...")
        streams = lsl.resolve_stream("type", self.stream_type, timeout=self.timeout)

        stream_infos = []
        for stream in streams:
            info = StreamInfo(
                name=stream.name(),
                type=stream.type(),
                channel_count=stream.channel_count(),
                sampling_rate=stream.nominal_srate(),
                channel_format=stream.channel_format(),
                source_id=stream.source_id(),
                hostname=stream.hostname(),
                uid=stream.uid(),
            )
            stream_infos.append(info)

        return stream_infos

    def connect(self, stream_info: StreamInfo | None = None) -> bool:
        """Connect to an LSL stream.

        Args:
            stream_info: Specific stream to connect to (None for auto-discovery)

        Returns:
            True if connection successful
        """
        if self.is_connected:
            print("Already connected to a stream")
            return True

        try:
            # Discover streams if none specified
            if stream_info is None:
                available_streams = self.discover_streams()

                if not available_streams:
                    print(f"No {self.stream_type} streams found")
                    return False

                # Filter by name if specified
                if self.stream_name:
                    matching_streams = [s for s in available_streams if s.name == self.stream_name]
                    if matching_streams:
                        stream_info = matching_streams[0]
                    else:
                        print(f"Stream '{self.stream_name}' not found")
                        return False
                else:
                    # Use first available stream
                    stream_info = available_streams[0]

            # Create inlet
            print(f"Connecting to stream: {stream_info.name}")
            streams = lsl.resolve_stream("uid", stream_info.uid)

            if not streams:
                print(f"Could not resolve stream {stream_info.uid}")
                return False

            self.inlet = lsl.StreamInlet(streams[0])
            self.stream_info = stream_info

            # Initialize buffer
            estimated_samples = int(stream_info.sampling_rate * self.buffer_duration)
            self.buffer = RingBuffer(capacity=estimated_samples)

            self.is_connected = True
            print(
                f"Connected to {stream_info.name} "
                f"({stream_info.channel_count} channels, "
                f"{stream_info.sampling_rate} Hz)"
            )

            return True

        except Exception as e:
            print(f"Connection failed: {e}")
            return False

    def start_receiving(self) -> bool:
        """Start receiving data in background thread.

        Returns:
            True if receiving started successfully
        """
        if not self.is_connected or self.inlet is None:
            print("Not connected to any stream")
            return False

        if self.is_receiving:
            print("Already receiving data")
            return True

        self._stop_event.clear()
        self._receive_thread = threading.Thread(
            target=self._receive_loop,
            daemon=True,
        )
        self._receive_thread.start()
        self.is_receiving = True

        print("Started receiving data")
        return True

    def stop_receiving(self) -> None:
        """Stop receiving data."""
        if not self.is_receiving:
            return

        print("Stopping data reception...")
        self._stop_event.set()

        if self._receive_thread and self._receive_thread.is_alive():
            self._receive_thread.join(timeout=1.0)

        self.is_receiving = False
        print("Data reception stopped")

    def disconnect(self) -> None:
        """Disconnect from the stream."""
        self.stop_receiving()

        if self.inlet:
            self.inlet.close_stream()
            self.inlet = None

        self.is_connected = False
        self.stream_info = None

        if self.buffer:
            self.buffer.clear()

        print("Disconnected from stream")

    def _receive_loop(self) -> None:
        """Main data receiving loop (runs in background thread)."""
        if not self.inlet or not self.buffer:
            return

        print("Data receiving loop started")

        while not self._stop_event.is_set():
            try:
                # Pull sample with timeout
                sample, timestamp = self.inlet.pull_sample(timeout=0.1)

                if sample is not None:
                    # Convert to numpy array
                    data = np.array(sample, dtype=np.float32)

                    # Store in buffer
                    self.buffer.put(data, timestamp)

                    # Call callback if set
                    if self.data_callback:
                        self.data_callback(data, timestamp)

            except Exception as e:
                print(f"Error in receive loop: {e}")
                break

        print("Data receiving loop ended")

    def get_latest_window(
        self,
        duration: float,
        end_time: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get latest window of EEG data.

        Args:
            duration: Window duration in seconds
            end_time: End timestamp (None for most recent)

        Returns:
            Tuple of (data_matrix, timestamps)
        """
        if not self.buffer or not self.stream_info:
            return np.array([]), np.array([])

        return self.buffer.get_window(
            duration=duration,
            sampling_rate=self.stream_info.sampling_rate,
            end_time=end_time,
        )

    def get_stream_info(self) -> StreamInfo | None:
        """Get information about the connected stream."""
        return self.stream_info

    def set_data_callback(
        self,
        callback: Callable[[np.ndarray, float], None],
    ) -> None:
        """Set callback function for incoming data.

        Args:
            callback: Function called for each sample (data, timestamp)
        """
        self.data_callback = callback


def create_mock_stream(
    name: str = "MockEEG",
    n_channels: int = 8,
    sampling_rate: float = 250.0,
    duration: float = 60.0,
) -> None:
    """Create a mock EEG stream for testing.

    Args:
        name: Stream name
        n_channels: Number of EEG channels
        sampling_rate: Sampling rate in Hz
        duration: How long to stream in seconds
    """
    if lsl is None:
        raise ImportError("pylsl not available for mock stream")

    # Create stream info
    info = lsl.StreamInfo(
        name=name,
        type="EEG",
        channel_count=n_channels,
        nominal_srate=sampling_rate,
        channel_format=lsl.cf_float32,
        source_id="mock_eeg_001",
    )

    # Create outlet
    outlet = lsl.StreamOutlet(info)

    print(f"Starting mock EEG stream: {name}")
    print(f"Channels: {n_channels}, Rate: {sampling_rate} Hz")

    try:
        # Generate synthetic EEG data
        n_samples = int(duration * sampling_rate)
        sample_interval = 1.0 / sampling_rate

        for i in range(n_samples):
            # Generate synthetic EEG-like data
            t = i / sampling_rate

            # Mix of sine waves with noise (simulating EEG rhythms)
            alpha = 0.5 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
            beta = 0.3 * np.sin(2 * np.pi * 20 * t)  # 20 Hz beta
            noise = 0.2 * np.random.randn()

            # Create multichannel data with slight variations
            sample = []
            for ch in range(n_channels):
                # Add channel-specific phase and amplitude variations
                phase_shift = ch * 0.1
                amplitude = 1.0 + 0.2 * ch

                value = amplitude * (
                    alpha * np.cos(phase_shift) + beta * np.sin(phase_shift) + noise
                )
                sample.append(value)

            # Send sample
            outlet.push_sample(sample)

            # Sleep to maintain sampling rate
            time.sleep(sample_interval)

            if i % int(sampling_rate) == 0:
                print(f"Sent {i} samples ({i/sampling_rate:.1f}s)")

    except KeyboardInterrupt:
        print("Mock stream interrupted")

    finally:
        print("Mock stream ended")


if __name__ == "__main__":
    # Example usage / testing
    import argparse

    parser = argparse.ArgumentParser(description="LSL EEG Receiver")
    parser.add_argument("--mock", action="store_true", help="Create mock stream")
    parser.add_argument("--receive", action="store_true", help="Receive from stream")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration")

    args = parser.parse_args()

    if args.mock:
        create_mock_stream(duration=args.duration)
    elif args.receive:
        receiver = LSLReceiver()

        def print_data(data: np.ndarray, timestamp: float) -> None:
            print(f"Sample: {data[:3]:.3f}... at {timestamp:.3f}")

        receiver.set_data_callback(print_data)

        if receiver.connect():
            receiver.start_receiving()

            try:
                time.sleep(args.duration)
            except KeyboardInterrupt:
                pass

            receiver.disconnect()
    else:
        # Just discover streams
        receiver = LSLReceiver()
        streams = receiver.discover_streams()
        print(f"Found {len(streams)} streams:")
        for stream in streams:
            print(f"  {stream.name} ({stream.type}, {stream.channel_count} ch)")
