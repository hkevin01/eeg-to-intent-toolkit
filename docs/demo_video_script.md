# EEG-to-Intent Toolkit Demo Video Script

## Video Overview

**Duration**: 8-10 minutes
**Style**: Professional demo with screen recording and narration
**Target Audience**: Researchers, students, and BCI practitioners

---

## Scene 1: Introduction (30 seconds)

### Visual: Title slide with toolkit logo

**Narrator**:

> "Welcome to the EEG-to-Intent toolkit - a comprehensive, open-source framework for developing brain-computer interfaces. This toolkit integrates state-of-the-art deep learning, self-supervised pretraining, and real-time inference into a unified research platform."

### Visual: Overview diagram showing phases

- Phase 1: Data & Baselines
- Phase 2: Self-Supervised Learning
- Phase 3: Personalization
- Phase 4: Real-time Inference
- Phase 5: Benchmarking & Model Zoo

---

## Scene 2: Quick Installation (45 seconds)

### Visual: Terminal screen

**Narrator**:

> "Getting started is simple. Clone the repository and install dependencies:"

```bash
git clone https://github.com/your-org/eeg-to-intent-toolkit.git
cd eeg-to-intent-toolkit
pip install -r requirements.txt
pip install -e .
```

**Narrator**:
> "Let's verify the installation works:"

```bash
python -c "import eegintent; print('✓ EEG-to-Intent toolkit ready!')"
```

---

## Scene 3: Dataset Loading and Preprocessing (1 minute)

### Visual: Jupyter notebook or Python script

**Narrator**:
> "The toolkit provides unified access to multiple EEG datasets. Let's load the popular BCIC IV 2a motor imagery dataset:"

```python
from eegintent.data.datasets import BCICIVDataset
from eegintent.preprocessing.pipeline import create_preprocessing_pipeline

# Load dataset
dataset = BCICIVDataset(
    subject_ids=[1, 2, 3],  # First 3 subjects
    download=True           # Auto-download if needed
)

print(f"Dataset: {dataset.n_channels} channels, {dataset.n_classes} classes")
print(f"Sampling rate: {dataset.sampling_rate} Hz")
```

**Narrator**:
> "The toolkit handles the complex preprocessing automatically:"

```python
# Create preprocessing pipeline
preprocessor = create_preprocessing_pipeline(
    sampling_rate=250,
    bandpass_range=(0.5, 50),  # Filter to EEG frequencies
    notch_freq=50,             # Remove power line noise
    use_car=True               # Common Average Reference
)
```

### Visual: Show EEG signal plots before/after preprocessing

---

## Scene 4: Model Training (1.5 minutes)

### Visual: Code editor showing model creation

**Narrator**:
> "Training state-of-the-art models is straightforward. Let's create an EEGNet model:"

```python
from eegintent.models.eegnet import EEGNet
from eegintent.training.trainer import EEGTrainer

# Create model
model = EEGNet(
    n_classes=4,
    n_channels=22,
    input_window_samples=1000,  # 4 seconds at 250 Hz
    dropout_rate=0.5
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Narrator**:
> "Train with the Lightning-based trainer:"

```python
trainer = EEGTrainer(
    model=model,
    max_epochs=100,
    learning_rate=1e-3,
    device="auto"  # Automatically uses GPU if available
)

# Get data loaders
train_loader, val_loader, test_loader = dataset.get_loaders(
    batch_size=32,
    preprocessor=preprocessor
)

# Train the model
trainer.fit(train_loader, val_loader)
```

### Visual: Show training progress with loss curves

---

## Scene 5: Self-Supervised Learning (1.5 minutes)

### Visual: Diagram showing SSL concept

**Narrator**:
> "One of the toolkit's key innovations is self-supervised pretraining for EEG. This allows us to leverage large amounts of unlabeled data:"

```python
from eegintent.ssl.contrastive import ContrastiveLearner
from eegintent.ssl.simclr import EEGSimCLR

# Create SSL model
ssl_model = EEGSimCLR(
    encoder=model.encoder,  # Use same encoder architecture
    projection_dim=128,
    temperature=0.1
)

# Pretrain on unlabeled data
ssl_trainer = ContrastiveLearner(
    model=ssl_model,
    max_epochs=200,
    learning_rate=1e-3
)

ssl_trainer.fit(unlabeled_loader)
```

### Visual: Show performance comparison chart

**Narrator**:
> "SSL pretraining improves cross-subject generalization by up to 7.6%, achieving 76.3% accuracy compared to 68.7% without pretraining."

---

## Scene 6: Personalization (1 minute)

### Visual: Personalization concept diagram

**Narrator**:
> "The toolkit includes advanced personalization techniques for adapting to individual subjects:"

```python
from eegintent.personalization.film import FiLMAdapter
from eegintent.personalization.prototypical import PrototypicalNetwork

# Subject-adaptive normalization
film_model = FiLMAdapter(
    base_model=model,
    subject_embedding_dim=64
)

# Few-shot learning
proto_model = PrototypicalNetwork(
    encoder=model.encoder,
    n_support=5,  # 5 examples per class
    n_query=15
)
```

### Visual: Show personalization results

**Narrator**:
> "With just 5 examples per class, personalization improves accuracy by 8.4%. Combined with SSL, we achieve 86.7% cross-subject accuracy."

---

## Scene 7: Real-time System Demo (2 minutes)

### Visual: Real-time dashboard interface

**Narrator**:
> "The toolkit includes a complete real-time BCI system. Let's launch the interactive dashboard:"

```bash
python -m eegintent.run_dashboard
```

### Visual: Dashboard opening in browser showing

- Real-time EEG signal plots
- Live classification results
- Performance monitoring
- Calibration wizard

**Narrator**:
> "The real-time system connects to any Lab Streaming Layer (LSL) compatible EEG device and provides:"

- Live signal visualization
- Real-time classification with 24ms latency
- Performance monitoring and optimization
- Easy calibration workflow

### Visual: Show LSL connection and real-time predictions

**Narrator**:
> "Behind the scenes, the system applies the same preprocessing and models we just trained, but optimized for real-time performance:"

```python
from eegintent.realtime.lsl_client import LSLReceiver
from eegintent.realtime.inference import SlidingWindowPredictor

# Connect to EEG device
receiver = LSLReceiver(stream_name="YourEEGDevice")
receiver.connect()

# Set up real-time predictor
predictor = SlidingWindowPredictor(
    model="path/to/trained_model.onnx",
    window_size_ms=1000,
    step_size_ms=250
)

# Start real-time inference
predictor.start_predictions()
```

---

## Scene 8: Benchmarking and Reproducibility (1 minute)

### Visual: Benchmark results table

**Narrator**:
> "Reproducible research is core to the toolkit. Run standardized benchmarks with a single command:"

```bash
python benchmarks/run_bcic_iv_2a_eegnet.py \
    --ssl \
    --personalization \
    --protocol cross_subject \
    --seed 42
```

### Visual: Show benchmark output and results

**Narrator**:
> "All results are fully reproducible with fixed seeds, standardized data splits, and automated logging. The toolkit includes:"

- Comprehensive benchmark suite
- Weights & Biases integration
- Model leaderboards
- Statistical significance testing

### Visual: Show W&B dashboard with results

---

## Scene 9: Community and Extensions (45 seconds)

### Visual: GitHub repository page

**Narrator**:
> "The EEG-to-Intent toolkit is designed for the research community. It includes:"

### Visual: Show documentation, tutorials, examples

- Comprehensive tutorials and documentation
- Pre-trained model zoo
- Extensible architecture for custom models
- Active community support

**Narrator**:
> "Whether you're a researcher exploring new BCI techniques, a student learning about brain-computer interfaces, or a practitioner building real applications, the toolkit provides everything you need."

---

## Scene 10: Conclusion and Call to Action (30 seconds)

### Visual: Results summary and key achievements

**Narrator**:
> "The EEG-to-Intent toolkit represents a major advancement in BCI research infrastructure. Key achievements include:"

- 86.7% cross-subject accuracy on motor imagery
- 24ms real-time inference latency
- Comprehensive SSL and personalization methods
- Production-ready real-time system

### Visual: GitHub stars, downloads, citations

**Narrator**:
> "Join the community! Star the repository, try the tutorials, and contribute to advancing brain-computer interface research."

### Visual: Final slide with links

- GitHub: github.com/your-org/eeg-to-intent-toolkit
- Documentation: eeg-to-intent-toolkit.readthedocs.io
- Tutorials: Start with the quickstart guide
- Community: Join our Discord/Slack

---

## Production Notes

### Technical Requirements

- **Screen Recording**: High resolution (1920x1080 minimum)
- **Audio**: Professional narration with clear audio
- **Code Highlighting**: Syntax-highlighted code examples
- **Animations**: Smooth transitions between scenes
- **Branding**: Consistent visual style and logos

### Assets Needed

- Toolkit logo and branding
- EEG signal visualizations
- Performance charts and comparisons
- Dashboard screenshots/recordings
- Architecture diagrams

### Distribution

- **YouTube**: Primary hosting platform
- **GitHub**: Embedded in README
- **Conference**: Submission to BCI/ML conferences
- **Social Media**: Promotional clips for Twitter/LinkedIn

### Call to Action

- Subscribe to updates
- Star the GitHub repository
- Try the quickstart tutorial
- Join the community discussion
- Cite in research papers---

**Estimated Production Time**: 2-3 weeks with professional team
**Budget**: $5,000-10,000 for professional production
**Alternative**: Academic/research team self-production with lower budget
