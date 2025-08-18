"""Real-time EEG dashboard using Streamlit."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import streamlit as st

if TYPE_CHECKING:
    import plotly.express as px
    import plotly.graph_objects as go
else:
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        go = None
        px = None

# Try to import our real-time modules
try:
    from ..realtime.dsp import create_standard_eeg_pipeline
    from ..realtime.inference import (
        InferenceConfig,
        RealtimeInferenceEngine,
        SlidingWindowPredictor,
    )
    from ..realtime.lsl_client import LSLReceiver, create_mock_stream
except ImportError:
    LSLReceiver = None
    create_mock_stream = None
    create_standard_eeg_pipeline = None
    RealtimeInferenceEngine = None
    SlidingWindowPredictor = None
    InferenceConfig = None


def init_session_state():
    """Initialize Streamlit session state."""
    if "lsl_receiver" not in st.session_state:
        st.session_state.lsl_receiver = None
    if "inference_engine" not in st.session_state:
        st.session_state.inference_engine = None
    if "dsp_pipeline" not in st.session_state:
        st.session_state.dsp_pipeline = None
    if "is_recording" not in st.session_state:
        st.session_state.is_recording = False
    if "predictions" not in st.session_state:
        st.session_state.predictions = []
    if "eeg_data" not in st.session_state:
        st.session_state.eeg_data = []
    if "timestamps" not in st.session_state:
        st.session_state.timestamps = []


def create_eeg_plot(data: np.ndarray, timestamps: np.ndarray, channels: list[str]):
    """Create real-time EEG plot."""
    if go is None:
        st.error("Plotly not available for plotting")
        return None

    fig = go.Figure()

    # Plot each channel with offset
    for i, channel in enumerate(channels):
        if i < data.shape[0]:
            offset = i * 100  # Vertical offset between channels
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=data[i] + offset,
                    mode="lines",
                    name=channel,
                    line=dict(width=1),
                )
            )

    fig.update_layout(
        title="Real-time EEG Signals",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (μV)",
        height=400,
        showlegend=True,
    )

    return fig


def create_prediction_plot(predictions: list):
    """Create prediction visualization."""
    if not predictions or go is None:
        return None

    # Extract data
    times = [p["timestamp"] for p in predictions]
    classes = [p["predicted_class"] for p in predictions]
    confidences = [p["confidence"] for p in predictions]

    fig = go.Figure()

    # Prediction timeline
    fig.add_trace(
        go.Scatter(
            x=times,
            y=classes,
            mode="markers+lines",
            marker=dict(
                size=10,
                color=confidences,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Confidence"),
            ),
            name="Predictions",
        )
    )

    fig.update_layout(
        title="Prediction Timeline",
        xaxis_title="Time",
        yaxis_title="Predicted Class",
        height=300,
    )

    return fig


def create_latency_plot(predictions: list):
    """Create latency monitoring plot."""
    if not predictions or go is None:
        return None

    latencies = [p["latency_ms"] for p in predictions]
    times = [p["timestamp"] for p in predictions]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=times,
            y=latencies,
            mode="lines+markers",
            name="Latency",
            line=dict(color="red"),
        )
    )

    # Add mean line
    if latencies:
        mean_latency = np.mean(latencies)
        fig.add_hline(
            y=mean_latency,
            line_dash="dash",
            annotation_text=f"Mean: {mean_latency:.1f}ms",
        )

    fig.update_layout(
        title="Inference Latency",
        xaxis_title="Time",
        yaxis_title="Latency (ms)",
        height=250,
    )

    return fig


def sidebar_controls():
    """Create sidebar controls."""
    st.sidebar.header("⚙️ Control Panel")

    # LSL Connection
    st.sidebar.subheader("LSL Stream")

    if st.sidebar.button("Discover Streams"):
        if LSLReceiver is not None:
            receiver = LSLReceiver()
            streams = receiver.discover_streams()

            if streams:
                st.sidebar.success(f"Found {len(streams)} streams:")
                for stream in streams:
                    st.sidebar.text(f"• {stream.name} ({stream.type})")
            else:
                st.sidebar.warning("No streams found")
        else:
            st.sidebar.error("LSL not available")

    # Stream selection
    stream_name = st.sidebar.text_input("Stream Name (optional)")

    # Model configuration
    st.sidebar.subheader("Model Configuration")

    model_path = st.sidebar.text_input("Model Path (ONNX/TorchScript)")

    config_expander = st.sidebar.expander("Inference Config")
    with config_expander:
        window_size = st.number_input("Window Size (ms)", value=1000, min_value=100)
        step_size = st.number_input("Step Size (ms)", value=100, min_value=50)
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.6)

    # DSP Configuration
    dsp_expander = st.sidebar.expander("DSP Settings")
    with dsp_expander:
        use_bandpass = st.checkbox("Bandpass Filter", value=True)
        low_freq = st.number_input("Low Freq (Hz)", value=1.0, min_value=0.1)
        high_freq = st.number_input("High Freq (Hz)", value=50.0, min_value=1.0)
        use_notch = st.checkbox("Notch Filter", value=True)
        notch_freq = st.selectbox("Notch Freq", [50, 60], index=0)
        use_car = st.checkbox("Common Average Reference", value=True)

    return {
        "stream_name": stream_name,
        "model_path": model_path,
        "window_size": window_size,
        "step_size": step_size,
        "confidence_threshold": confidence_threshold,
        "dsp_config": {
            "use_bandpass": use_bandpass,
            "low_freq": low_freq,
            "high_freq": high_freq,
            "use_notch": use_notch,
            "notch_freq": notch_freq,
            "use_car": use_car,
        },
    }


def main_dashboard():
    """Main dashboard layout."""
    st.title("🧠 EEG-to-Intent Real-time Dashboard")

    # Initialize session state
    init_session_state()

    # Get controls
    config = sidebar_controls()

    # Main controls
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🟢 Start Recording", disabled=st.session_state.is_recording):
            start_recording(config)

    with col2:
        if st.button("🔴 Stop Recording", disabled=not st.session_state.is_recording):
            stop_recording()

    with col3:
        if st.button("🔄 Reset"):
            reset_session()

    # Status display
    if st.session_state.is_recording:
        st.success("🟢 Recording active")
    else:
        st.info("⚫ Not recording")

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Real-time", "🎯 Predictions", "⏱️ Latency", "📋 Logs"])

    with tab1:
        st.subheader("Live EEG Signals")
        eeg_placeholder = st.empty()

        # Show live EEG data
        if st.session_state.eeg_data and st.session_state.timestamps:
            # Last 500 samples
            data = np.array(st.session_state.eeg_data[-500:])
            timestamps = np.array(st.session_state.timestamps[-500:])

            if data.size > 0:
                channels = [f"Ch{i+1}" for i in range(data.shape[1])]
                fig = create_eeg_plot(data.T, timestamps, channels)
                if fig:
                    eeg_placeholder.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Classification Results")
        pred_placeholder = st.empty()

        if st.session_state.predictions:
            fig = create_prediction_plot(st.session_state.predictions)
            if fig:
                pred_placeholder.plotly_chart(fig, use_container_width=True)

            # Recent predictions table
            st.subheader("Recent Predictions")
            recent_preds = st.session_state.predictions[-10:]
            if recent_preds:
                df_data = []
                for p in recent_preds:
                    df_data.append(
                        {
                            "Time": time.strftime("%H:%M:%S", time.localtime(p["timestamp"])),
                            "Class": p["predicted_class"],
                            "Confidence": f"{p['confidence']:.3f}",
                            "Latency (ms)": f"{p['latency_ms']:.1f}",
                        }
                    )
                st.table(df_data)

    with tab3:
        st.subheader("Performance Monitoring")

        if st.session_state.predictions:
            fig = create_latency_plot(st.session_state.predictions)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Statistics
            latencies = [p["latency_ms"] for p in st.session_state.predictions]
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Mean Latency", f"{np.mean(latencies):.1f} ms")
            with col2:
                st.metric("Max Latency", f"{np.max(latencies):.1f} ms")
            with col3:
                st.metric("Total Predictions", len(st.session_state.predictions))
            with col4:
                if latencies:
                    start_time = st.session_state.predictions[0]["timestamp"]
                    elapsed = max(1, (time.time() - start_time))
                    fps = len(latencies) / elapsed
                    st.metric("Predictions/sec", f"{fps:.1f}")

    with tab4:
        st.subheader("System Logs")

        # Show connection status
        if st.session_state.lsl_receiver:
            stream_info = st.session_state.lsl_receiver.get_stream_info()
            if stream_info:
                st.json(
                    {
                        "Stream Name": stream_info.name,
                        "Channels": stream_info.channel_count,
                        "Sampling Rate": stream_info.sampling_rate,
                        "Source ID": stream_info.source_id,
                    }
                )

        # Configuration display
        st.subheader("Current Configuration")
        st.json(config)


def start_recording(config):
    """Start recording and inference."""
    try:
        # Initialize LSL receiver
        if LSLReceiver is not None:
            receiver = LSLReceiver(
                stream_name=(config["stream_name"] if config["stream_name"] else None)
            )

            if receiver.connect():
                st.session_state.lsl_receiver = receiver

                # Set up data callback
                def on_data(data, timestamp):
                    st.session_state.eeg_data.append(data)
                    st.session_state.timestamps.append(timestamp)

                    # Keep only recent data
                    max_samples = 5000
                    if len(st.session_state.eeg_data) > max_samples:
                        eeg_data = st.session_state.eeg_data
                        timestamps = st.session_state.timestamps
                        st.session_state.eeg_data = eeg_data[-max_samples:]
                        st.session_state.timestamps = timestamps[-max_samples:]

                receiver.set_data_callback(on_data)
                receiver.start_receiving()

                # Initialize inference if model provided
                model_path = config["model_path"]
                if model_path and Path(model_path).exists():
                    setup_inference(config)

                st.session_state.is_recording = True
                st.success("Started recording!")
            else:
                st.error("Failed to connect to LSL stream")
        else:
            st.error("LSL not available")

    except Exception as e:
        st.error(f"Error starting recording: {e}")


def setup_inference(config):
    """Set up inference engine."""
    try:
        if InferenceConfig is not None and SlidingWindowPredictor is not None:
            # Create inference config
            inf_config = InferenceConfig(
                window_size_ms=config["window_size"],
                step_size_ms=config["step_size"],
                confidence_threshold=config["confidence_threshold"],
            )

            # Create predictor
            predictor = SlidingWindowPredictor(
                model=config["model_path"],
                config=inf_config,
                sampling_rate=250.0,  # Assume 250 Hz
                n_channels=8,  # Assume 8 channels
            )

            # Create inference engine
            if RealtimeInferenceEngine is not None:
                engine = RealtimeInferenceEngine(predictor)

                # Set prediction callback
                def on_prediction(pred):
                    pred_dict = {
                        "timestamp": pred.timestamp,
                        "predicted_class": pred.predicted_class,
                        "confidence": pred.confidence,
                        "latency_ms": pred.latency_ms,
                        "probabilities": pred.probabilities.tolist(),
                    }
                    st.session_state.predictions.append(pred_dict)

                    # Keep only recent predictions
                    if len(st.session_state.predictions) > 1000:
                        predictions = st.session_state.predictions
                        st.session_state.predictions = predictions[-1000:]

                engine.set_prediction_callback(on_prediction)
                engine.start_inference()

                st.session_state.inference_engine = engine

    except Exception as e:
        st.error(f"Error setting up inference: {e}")


def stop_recording():
    """Stop recording and inference."""
    if st.session_state.lsl_receiver:
        st.session_state.lsl_receiver.disconnect()
        st.session_state.lsl_receiver = None

    if st.session_state.inference_engine:
        st.session_state.inference_engine.stop_inference()
        st.session_state.inference_engine = None

    st.session_state.is_recording = False
    st.success("Stopped recording!")


def reset_session():
    """Reset all session data."""
    stop_recording()
    st.session_state.eeg_data = []
    st.session_state.timestamps = []
    st.session_state.predictions = []
    st.success("Session reset!")


def calibration_wizard():
    """Calibration wizard page."""
    st.title("🎯 Calibration Wizard")

    st.markdown(
        """
    This wizard will guide you through calibrating the EEG classification
    system for optimal performance with your specific setup.
    """
    )

    # Step 1: Hardware setup
    with st.expander("Step 1: Hardware Setup", expanded=True):
        st.markdown(
            """
        1. Ensure your EEG headset is properly connected
        2. Check electrode impedances (should be < 10kΩ)
        3. Start your LSL stream from the EEG software
        """
        )

        if st.button("Test LSL Connection"):
            if LSLReceiver is not None:
                receiver = LSLReceiver()
                streams = receiver.discover_streams()
                if streams:
                    st.success(f"✓ Found {len(streams)} LSL streams")
                    for stream in streams:
                        st.info(f"Stream: {stream.name} ({stream.channel_count} channels)")
                else:
                    st.warning("No LSL streams found")

    # Step 2: Signal quality
    with st.expander("Step 2: Signal Quality Check"):
        st.markdown(
            """
        Record a baseline measurement to check signal quality:
        - Sit still and relax for 30 seconds
        - Minimize movement and muscle tension
        """
        )

        if st.button("Start Baseline Recording"):
            st.info("Baseline recording would start here...")

    # Step 3: Task calibration
    with st.expander("Step 3: Task Calibration"):
        st.markdown(
            """
        Perform the mental tasks to calibrate the classifier:
        1. Follow the on-screen instructions
        2. Perform each task for the specified duration
        3. Rest between tasks
        """
        )

        task_type = st.selectbox(
            "Calibration Task",
            [
                "Motor Imagery (Left vs Right Hand)",
                "Visual Attention (Left vs Right)",
                "Mental Math vs Rest",
            ],
        )

        n_trials = st.number_input("Number of trials per class", value=20, min_value=5)
        trial_duration = st.number_input("Trial duration (seconds)", value=4, min_value=2)

        if st.button("Start Calibration"):
            st.info(f"Calibration for {task_type} would start here...")

    # Step 4: Validation
    with st.expander("Step 4: Validation"):
        st.markdown(
            """
        Test the calibrated model:
        - Perform sample tasks
        - Check classification accuracy
        - Adjust parameters if needed
        """
        )

        if st.button("Validate Model"):
            st.info("Model validation would start here...")


# Main app
def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="EEG-to-Intent Dashboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["🏠 Dashboard", "🎯 Calibration", "📖 Help"],
    )

    if page == "🏠 Dashboard":
        main_dashboard()
    elif page == "🎯 Calibration":
        calibration_wizard()
    elif page == "📖 Help":
        st.title("📖 Help & Documentation")
        st.markdown(
            """
        ## Getting Started

        1. **Connect EEG Device**: Start your EEG software and enable LSL streaming
        2. **Discover Streams**: Use the sidebar to find available LSL streams
        3. **Load Model**: Provide path to trained ONNX or TorchScript model
        4. **Start Recording**: Begin real-time data acquisition and inference

        ## Features

        - **Real-time EEG visualization**
        - **Live classification predictions**
        - **Performance monitoring**
        - **Calibration wizard**

        ## Troubleshooting

        - **No streams found**: Check if EEG software is running and LSL is enabled
        - **Poor predictions**: Run calibration wizard or retrain model
        - **High latency**: Reduce window size or use CPU optimization
        """
        )


if __name__ == "__main__":
    main()
