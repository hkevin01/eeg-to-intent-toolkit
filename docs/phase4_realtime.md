# Phase 4: Real-time Inference System

This directory contains the complete real-time EEG processing and inference system for the EEG-to-Intent toolkit.

## Overview

Phase 4 implements a production-ready real-time inference system that:

- **Acquires EEG data** from Lab Streaming Layer (LSL) streams
- **Applies real-time filtering** using optimized DSP pipelines
- **Performs sliding-window inference** with temporal smoothing
- **Provides interactive dashboard** for monitoring and calibration
- **Exports models** for deployment optimization

## Quick Start

### 1. Install Dependencies

```bash
# Core dependencies
pip install pylsl scipy numpy

# For model inference
pip install torch onnxruntime

# For dashboard (optional)
pip install -r requirements-dashboard.txt
```

### 2. Run the Dashboard

```bash
python -m eegintent.run_dashboard
```

Open your browser to `http://localhost:8501` to access the interactive dashboard.

### 3. Run Example Pipeline

```bash
python examples/realtime_pipeline.py
```

## Architecture

### LSL Stream Management (`lsl_client.py`)

```python
from eegintent.realtime.lsl_client import LSLReceiver

# Connect to EEG stream
receiver = LSLReceiver(stream_name="YourEEGDevice")
receiver.connect()

# Set up data callback
def on_data(data, timestamp):
    print(f"Received {data.shape} at {timestamp}")

receiver.set_data_callback(on_data)
receiver.start_receiving()
```

**Features:**
- Automatic stream discovery
- Thread-safe ring buffer
- Timestamp synchronization
- Mock stream generation for testing

### Real-time DSP (`dsp.py`)

```python
from eegintent.realtime.dsp import create_standard_eeg_pipeline

# Create processing pipeline
pipeline = create_standard_eeg_pipeline(
    sampling_rate=250.0,
    bandpass_range=(1.0, 50.0),
    notch_freq=50.0,
    use_car=True
)

# Process incoming data
filtered_data = pipeline.process_samples(raw_eeg_data)
```

**Filters Available:**
- **Bandpass Filter**: IIR-based frequency filtering
- **Notch Filter**: Power line interference removal (50/60 Hz)
- **Common Average Reference**: Spatial filtering
- **Adaptive Noise Filter**: Dynamic artifact reduction

### Sliding Window Inference (`inference.py`)

```python
from eegintent.realtime.inference import (
    SlidingWindowPredictor,
    InferenceConfig
)

# Configure inference
config = InferenceConfig(
    window_size_ms=1000,
    step_size_ms=250,
    confidence_threshold=0.7
)

# Create predictor
predictor = SlidingWindowPredictor(
    model="path/to/model.onnx",
    config=config,
    sampling_rate=250.0,
    n_channels=8
)

# Make predictions
prediction = predictor.predict(eeg_window)
print(f"Class: {prediction.predicted_class}, Confidence: {prediction.confidence}")
```

**Features:**
- PyTorch and ONNX model support
- Temporal smoothing and voting
- Confidence thresholding
- Latency monitoring

### Model Export (`model_export.py`)

```python
from eegintent.export.model_export import export_eeg_model_complete

# Export trained model for deployment
export_eeg_model_complete(
    pytorch_model=model,
    dummy_input=torch.randn(1, 8, 250),
    output_dir="deployment/",
    model_name="eeg_classifier"
)
```

**Export Formats:**
- **ONNX**: Cross-platform optimization
- **TorchScript**: PyTorch-native deployment
- **Benchmarking**: Latency and throughput testing
- **Config Generation**: Deployment metadata

## Dashboard Features

The Streamlit-based dashboard provides:

### Real-time Monitoring
- Live EEG signal visualization
- Multi-channel time-series plots
- Signal quality indicators

### Prediction Visualization
- Classification results timeline
- Confidence score tracking
- Class probability distributions

### Performance Monitoring
- Inference latency tracking
- Throughput measurement
- System resource usage

### Calibration Wizard
- Guided setup process
- Signal quality assessment
- Model validation tools

## Configuration

### LSL Settings
```python
# Stream discovery
receiver = LSLReceiver()
streams = receiver.discover_streams()

# Manual connection
receiver = LSLReceiver(
    stream_name="BrainVision RDA",
    stream_type="EEG"
)
```

### DSP Parameters
```python
pipeline = create_standard_eeg_pipeline(
    sampling_rate=250.0,      # Hz
    bandpass_range=(0.5, 50), # Hz
    notch_freq=50,            # Hz (50 or 60)
    use_car=True,             # Common average reference
    buffer_size=1000          # Samples
)
```

### Inference Configuration
```python
config = InferenceConfig(
    window_size_ms=1000,      # Analysis window
    step_size_ms=250,         # Sliding step
    confidence_threshold=0.6, # Minimum confidence
    smoothing_window=5,       # Temporal smoothing
    voting_strategy="majority" # or "confidence_weighted"
)
```

## Performance Optimization

### Model Optimization
- Use ONNX for cross-platform deployment
- Apply model quantization for edge devices
- Optimize batch sizes for throughput

### Memory Management
- Configure ring buffer sizes appropriately
- Limit prediction history retention
- Use streaming processing where possible

### Latency Reduction
- Minimize window sizes when possible
- Use efficient model architectures
- Enable GPU acceleration if available

## Integration Examples

### With OpenBCI
```python
# OpenBCI LSL stream
receiver = LSLReceiver(stream_name="OpenBCIGUI")
```

### With BrainVision
```python
# BrainVision RDA stream
receiver = LSLReceiver(stream_name="BrainVision RDA")
```

### With g.tec
```python
# g.tec streaming
receiver = LSLReceiver(stream_name="g.USBamp")
```

## Troubleshooting

### Common Issues

**No LSL Streams Found**
- Ensure EEG software is running
- Check LSL streaming is enabled
- Verify network connectivity

**High Latency**
- Reduce window size
- Optimize model architecture
- Check system resources

**Poor Predictions**
- Run calibration wizard
- Check signal quality
- Retrain or fine-tune model

**Connection Drops**
- Check USB/network stability
- Increase buffer sizes
- Add connection retry logic

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Detailed LSL information
receiver = LSLReceiver(debug=True)
```

## Deployment

### Production Checklist
- [ ] Model exported and validated
- [ ] LSL streams tested
- [ ] Latency benchmarked
- [ ] Error handling verified
- [ ] Monitoring configured

### Docker Deployment
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ /app/src/
WORKDIR /app
CMD ["python", "-m", "eegintent.run_dashboard"]
```

## Next Steps

After completing Phase 4, consider:

1. **Performance Benchmarking** - Comprehensive latency and accuracy testing
2. **Model Zoo Development** - Pre-trained models for common tasks
3. **Advanced Visualizations** - 3D brain mapping and source localization
4. **Multi-modal Integration** - Combine with other biosignals
5. **Edge Deployment** - Optimize for mobile and embedded devices

## API Reference

See individual module documentation for detailed API reference:

- [`lsl_client.py`](src/eegintent/realtime/lsl_client.py) - LSL stream management
- [`dsp.py`](src/eegintent/realtime/dsp.py) - Real-time signal processing
- [`inference.py`](src/eegintent/realtime/inference.py) - Model inference engine
- [`dashboard.py`](src/eegintent/realtime/dashboard.py) - Interactive dashboard
- [`model_export.py`](src/eegintent/export/model_export.py) - Model deployment utilities
