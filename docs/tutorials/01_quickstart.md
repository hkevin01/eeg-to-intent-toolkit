# Quickstart Guide

Welcome to the EEG-to-Intent toolkit! This quickstart guide will get you up and running in under 15 minutes.

## Learning Objectives

By the end of this tutorial, you'll be able to:

- Load and preprocess EEG datasets
- Train a baseline classification model
- Evaluate model performance
- Run reproducible benchmarks

## Prerequisites

- Python 3.9+ with PyTorch installed
- Basic understanding of EEG and machine learning
- Completed installation (see main [README](../../README.md))

## Step 1: Environment Setup

First, let's verify your installation and set up the environment:

```bash
# Activate your environment
conda activate eeg-intent  # or your preferred environment

# Verify installation
python -c "import eegintent; print('✓ EEG-to-Intent toolkit installed')"

# Check GPU availability (optional)
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
```

## Step 2: Quick Dataset Demo

Let's start with a simple dataset loading example:

```python
# examples/quickstart_demo.py
import numpy as np
from eegintent.data.datasets import BCICIVDataset
from eegintent.preprocessing.pipeline import create_preprocessing_pipeline
from eegintent.models.eegnet import EEGNet
from eegintent.training.trainer import EEGTrainer

def quickstart_demo():
    """Quickstart demonstration of the EEG-to-Intent toolkit."""

    print("🧠 EEG-to-Intent Quickstart Demo")
    print("="*40)

    # Step 1: Load dataset
    print("📊 Loading BCIC IV 2a dataset...")
    dataset = BCICIVDataset(
        subject_ids=[1, 2],  # Use first 2 subjects for demo
        download=True,       # Auto-download if needed
    )

    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Channels: {dataset.n_channels}")
    print(f"Classes: {dataset.n_classes}")
    print(f"Sampling rate: {dataset.sampling_rate} Hz")

    # Step 2: Create preprocessing pipeline
    print("\n🔧 Setting up preprocessing...")
    preprocessor = create_preprocessing_pipeline(
        sampling_rate=dataset.sampling_rate,
        bandpass_range=(0.5, 50),  # Standard EEG frequency range
        notch_freq=50,             # Remove power line noise
        use_car=True,              # Common Average Reference
    )

    # Step 3: Get data loaders
    print("\n📦 Creating data loaders...")
    train_loader, val_loader, test_loader = dataset.get_loaders(
        batch_size=32,
        num_workers=4,
        preprocessor=preprocessor,
    )

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Step 4: Create model
    print("\n🤖 Creating EEGNet model...")
    model = EEGNet(
        n_classes=dataset.n_classes,
        n_channels=dataset.n_channels,
        input_window_samples=1000,  # 4 seconds at 250 Hz
        dropout_rate=0.5,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Step 5: Train model
    print("\n🚀 Training model...")
    trainer = EEGTrainer(
        model=model,
        max_epochs=5,  # Short training for demo
        learning_rate=1e-3,
        device="auto",
    )

    # Train the model
    trainer.fit(train_loader, val_loader)

    # Step 6: Evaluate
    print("\n📊 Evaluating model...")
    test_results = trainer.test(test_loader)

    print("Test Results:")
    for metric, value in test_results.items():
        print(f"  {metric}: {value:.3f}")

    # Step 7: Make predictions
    print("\n🔮 Making predictions...")
    predictions, targets = trainer.predict(test_loader)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")
    print(f"Sample targets: {targets[:5]}")

    print("\n✅ Quickstart demo completed!")
    print("Next steps:")
    print("  1. Try different models (ShallowConvNet, DeepConvNet)")
    print("  2. Experiment with preprocessing options")
    print("  3. Explore self-supervised pretraining")
    print("  4. Try personalization techniques")

if __name__ == "__main__":
    quickstart_demo()
```

Run the demo:

```bash
python examples/quickstart_demo.py
```

## Step 3: Run a Benchmark

Let's run a standardized benchmark:

```bash
# Run a quick benchmark with the demo script
python benchmarks/run_demo_benchmark.py \
    --dataset bcic_iv_2a \
    --model eegnet \
    --protocol within_subject \
    --seed 42

# Check the results
ls outputs/bcic_iv_2a_eegnet_within_subject_seed42/
```

You should see:
```
results.json  # Summary metrics
results.npz   # Detailed predictions
logs/         # Training logs
```

## Step 4: Explore the Results

Let's examine the benchmark results:

```python
# examples/analyze_results.py
import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_benchmark_results(results_dir="outputs/bcic_iv_2a_eegnet_within_subject_seed42"):
    """Analyze benchmark results."""

    # Load summary results
    with open(f"{results_dir}/results.json", 'r') as f:
        summary = json.load(f)

    print("📊 Benchmark Results Analysis")
    print("="*40)
    print(f"Experiment: {summary['experiment_name']}")
    print(f"Dataset: {summary['dataset']}")
    print(f"Model: {summary['model']}")
    print(f"Protocol: {summary['protocol']}")
    print()

    # Show performance metrics
    print("Performance Metrics:")
    if 'accuracy_mean' in summary:
        print(f"  Accuracy: {summary['accuracy_mean']:.3f} ± {summary.get('accuracy_std', 0):.3f}")
    if 'kappa_mean' in summary:
        print(f"  Kappa: {summary['kappa_mean']:.3f} ± {summary.get('kappa_std', 0):.3f}")
    if 'f1_score_mean' in summary:
        print(f"  F1 Score: {summary['f1_score_mean']:.3f} ± {summary.get('f1_score_std', 0):.3f}")

    # Load detailed results
    detailed = np.load(f"{results_dir}/results.npz", allow_pickle=True)

    print(f"\nDetailed Results Available:")
    for key in detailed.files:
        print(f"  {key}: {detailed[key].shape if hasattr(detailed[key], 'shape') else type(detailed[key])}")

    # Subject-wise performance (if available)
    if 'accuracy_subjects' in summary:
        print(f"\nSubject-wise Accuracy:")
        for subject, acc in summary['accuracy_subjects'].items():
            print(f"  Subject {subject}: {acc:.3f}")

if __name__ == "__main__":
    analyze_benchmark_results()
```

## Step 5: Compare Different Models

Let's compare multiple models:

```bash
# Compare EEGNet vs ShallowConvNet
python benchmarks/run_demo_benchmark.py --model eegnet --seed 42
python benchmarks/run_demo_benchmark.py --model shallow_convnet --seed 42

# The results will be saved in separate directories:
# outputs/bcic_iv_2a_eegnet_within_subject_seed42/
# outputs/bcic_iv_2a_shallow_convnet_within_subject_seed42/
```

## What's Next?

Congratulations! You've completed the quickstart guide. Here are your next steps:

### 🧠 **Dive Deeper into Models**
- Try different architectures: `deep_convnet`, `eegconformer`
- Experiment with hyperparameters
- Compare within-subject vs cross-subject evaluation

### 📊 **Explore More Datasets**
- Load different datasets: `physionet_mi`, `cho2017`
- Work with your own EEG data
- Try different preprocessing pipelines

### 🚀 **Advanced Techniques**
- [Self-supervised pretraining](02_ssl_pretraining.md) for better representations
- [Personalization techniques](03_personalization.md) for subject adaptation
- [Real-time inference](04_realtime_demo.md) for live BCI applications

### 📈 **Research and Benchmarking**
- Run comprehensive [benchmarks](05_benchmarking.md)
- Contribute to the research leaderboard
- Develop custom models and methods

## Troubleshooting

### Common Issues

**ImportError: No module named 'eegintent'**
```bash
# Make sure you installed in development mode
pip install -e .
```

**CUDA out of memory**
```bash
# Reduce batch size or use CPU
python benchmarks/run_demo_benchmark.py --batch-size 16
```

**Dataset download fails**
```bash
# Check internet connection and try manual download
python -c "from eegintent.data.datasets import BCICIVDataset; BCICIVDataset(download=True)"
```

**Slow training**
```bash
# Check if GPU is being used
python -c "import torch; print(torch.cuda.is_available())"
```

### Getting Help

- Check the [FAQ](../faq.md)
- Browse [GitHub Issues](https://github.com/your-org/eeg-to-intent-toolkit/issues)
- Ask questions in [Discussions](https://github.com/your-org/eeg-to-intent-toolkit/discussions)

---

**Next Tutorial**: [Self-Supervised Learning](02_ssl_pretraining.md)
