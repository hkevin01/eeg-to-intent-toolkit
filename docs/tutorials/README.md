# EEG-to-Intent Toolkit Tutorials

Welcome to the comprehensive tutorial series for the EEG-to-Intent toolkit! This collection of tutorials will guide you through all aspects of the toolkit, from basic usage to advanced research applications.

## Tutorial Overview

### 🚀 [Quickstart Guide](01_quickstart.md)
Get up and running with the toolkit in minutes. Learn the basic workflow from data loading to model training and evaluation.

### 🧠 [Self-Supervised Learning](02_ssl_pretraining.md)
Master self-supervised pretraining techniques for EEG data. Learn how to leverage unlabeled data to improve your models.

### 👤 [Personalization Techniques](03_personalization.md)
Explore advanced personalization methods including few-shot learning, domain adaptation, and subject-specific fine-tuning.

### ⚡ [Real-time Inference](04_realtime_demo.md)
Build real-time BCI applications with LSL integration, live signal processing, and interactive dashboards.

### 📊 [Benchmarking and Evaluation](05_benchmarking.md)
Learn how to run reproducible benchmarks, compare models, and contribute to the research leaderboard.

### 🔧 [Advanced Customization](06_advanced_usage.md)
Deep dive into toolkit internals, custom model development, and research-oriented features.

## Prerequisites

Before starting the tutorials, make sure you have:

1. **Python 3.9+** installed
2. **Basic EEG knowledge** (what are EEG signals, common artifacts, basic preprocessing)
3. **PyTorch familiarity** (tensors, basic neural networks, training loops)
4. **Optional**: EEG hardware for real-time tutorials

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/eeg-to-intent-toolkit.git
cd eeg-to-intent-toolkit

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Optional: Install real-time dependencies
pip install -r requirements-realtime.txt

# Optional: Install dashboard dependencies
pip install -r requirements-dashboard.txt
```

## Quick Test

Verify your installation:

```bash
# Run a quick benchmark demo
python benchmarks/run_demo_benchmark.py --dataset bcic_iv_2a --model eegnet

# Test real-time components
python examples/realtime_pipeline.py

# Launch the dashboard (optional)
python -m eegintent.run_dashboard
```

## Tutorial Structure

Each tutorial follows a consistent structure:

- **Learning Objectives**: What you'll accomplish
- **Prerequisites**: Required knowledge and setup
- **Step-by-Step Guide**: Detailed instructions with code examples
- **Exercises**: Hands-on practice opportunities
- **Advanced Topics**: Extensions and research directions
- **Further Reading**: Additional resources

## Getting Help

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/your-org/eeg-to-intent-toolkit/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/your-org/eeg-to-intent-toolkit/discussions)
- **Documentation**: Browse the [API documentation](../api/)
- **Community**: Join our research community on Discord/Slack

## Contributing to Tutorials

We welcome contributions to improve these tutorials! Please see our [contribution guidelines](../contributing.md) for details on:

- Reporting tutorial issues
- Suggesting improvements
- Adding new tutorial content
- Translating tutorials

## Tutorial Data

Some tutorials use example datasets. These will be automatically downloaded when needed. For offline usage, you can pre-download:

```bash
# Download tutorial datasets
python scripts/download_tutorial_data.py
```

## Citing the Toolkit

If you use this toolkit in your research, please cite:

```bibtex
@software{eeg_to_intent_toolkit,
  title={EEG-to-Intent Toolkit: A Comprehensive Framework for Brain-Computer Interfaces},
  author={Your Name and Contributors},
  year={2025},
  url={https://github.com/your-org/eeg-to-intent-toolkit}
}
```

---

Ready to get started? Begin with the [Quickstart Guide](01_quickstart.md)!
