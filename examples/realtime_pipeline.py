#!/usr/bin/env python3
"""
Example: Complete real-time EEG processing pipeline.

This example demonstrates the full pipeline from LSL stream acquisition
through real-time filtering and model inference.
"""

import logging
import time
from pathlib import Path

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_lsl_connection():
    """Set up LSL connection."""
    from eegintent.realtime.lsl_client import LSLReceiver, create_mock_stream

    # Create mock EEG stream for demonstration
    logger.info("Creating mock EEG stream...")
    mock_stream = create_mock_stream(name="MockEEG", n_channels=8, sampling_rate=250.0)

    # Connect to LSL stream
    logger.info("Connecting to LSL stream...")
    receiver = LSLReceiver(stream_name="MockEEG")

    if not receiver.connect():
        logger.error("Failed to connect to LSL stream")
        return None, None

    logger.info(f"Connected to stream: {receiver.get_stream_info().name}")
    return receiver, mock_stream


def setup_dsp_pipeline():
    """Set up DSP processing pipeline."""
    from eegintent.realtime.dsp import create_standard_eeg_pipeline

    logger.info("Setting up DSP pipeline...")
    return create_standard_eeg_pipeline(
        sampling_rate=250.0, bandpass_range=(1.0, 50.0), notch_freq=50.0, use_car=True
    )


def collect_and_process_data(receiver, dsp_pipeline, duration=10):
    """Collect and process EEG data."""
    processed_data = []
    raw_data = []

    def on_raw_data(data, timestamp):
        """Handle raw EEG data."""
        raw_data.append((data, timestamp))

        # Apply DSP filtering
        if len(raw_data) > 10:  # Need some history for filtering
            recent_samples = np.array([d[0] for d in raw_data[-10:]])
            filtered = dsp_pipeline.process_samples(recent_samples.T)
            processed_data.append((filtered[:, -1], timestamp))

    receiver.set_data_callback(on_raw_data)
    receiver.start_receiving()

    # Collect data
    logger.info(f"Collecting data for {duration} seconds...")
    time.sleep(duration)

    return raw_data, processed_data


def demonstrate_inference(processed_data):
    """Demonstrate model inference setup."""
    from eegintent.realtime.inference import InferenceConfig, SlidingWindowPredictor

    logger.info("Simulating model inference...")

    config = InferenceConfig(window_size_ms=1000, step_size_ms=250, confidence_threshold=0.7)

    # Create a dummy model path for demonstration
    dummy_model_path = Path("dummy_model.onnx")

    try:
        # This would normally load a real model
        SlidingWindowPredictor(
            model=str(dummy_model_path), config=config, sampling_rate=250.0, n_channels=8
        )
        logger.info("Model predictor created (would load real model)")

        # Simulate making a prediction
        if processed_data:
            dummy_window = np.random.randn(8, 250)  # 1 second window
            logger.info(f"Would predict on window shape: {dummy_window.shape}")

    except Exception as e:
        logger.info(f"Skipping model inference (no model): {e}")


def run_realtime_example():
    """Run a complete real-time processing example."""
    try:
        logger.info("Starting real-time EEG processing example...")

        # Set up components
        receiver, mock_stream = setup_lsl_connection()
        if not receiver:
            return

        dsp_pipeline = setup_dsp_pipeline()

        # Collect and process data
        raw_data, processed_data = collect_and_process_data(receiver, dsp_pipeline)

        # Show results
        logger.info(f"Collected {len(raw_data)} raw samples")
        logger.info(f"Processed {len(processed_data)} filtered samples")

        if processed_data:
            # Calculate some basic statistics
            all_filtered = np.array([d[0] for d in processed_data])
            logger.info(f"Filtered data shape: {all_filtered.shape}")
            logger.info(f"Mean amplitude: {np.mean(all_filtered):.3f} μV")
            logger.info(f"Std amplitude: {np.std(all_filtered):.3f} μV")

        # Demonstrate inference
        demonstrate_inference(processed_data)

        # Clean up
        receiver.disconnect()
        if mock_stream and hasattr(mock_stream, "stop_streaming"):
            mock_stream.stop_streaming()

        logger.info("Real-time example completed successfully!")

    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.info("Try installing with: pip install pylsl")
    except Exception as e:
        logger.error(f"Error in real-time example: {e}")


def run_export_example():
    """Demonstrate model export capabilities."""
    try:
        # Just check if export module is available
        import eegintent.export.model_export  # noqa: F401

        logger.info("Model export example...")
        logger.info("Export example requires a trained PyTorch model")
        logger.info("See training examples for how to create and export models")

    except ImportError as e:
        logger.error(f"Missing dependencies for export: {e}")
        logger.info("Try installing with: pip install torch onnx")


def main():
    """Run all examples."""
    print("EEG-to-Intent Real-time Processing Examples")
    print("=" * 50)

    # Run real-time processing example
    run_realtime_example()

    print("\n" + "=" * 50)

    # Run export example
    run_export_example()

    print("\nFor interactive dashboard, run:")
    print("python -m eegintent.run_dashboard")


if __name__ == "__main__":
    main()
