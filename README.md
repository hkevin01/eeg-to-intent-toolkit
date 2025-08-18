# EEG-to-Intent Toolkit

Robust, open-source pipeline to detect actionable mental states from EEG (consumer and research-grade). Train SOTA models (EEGNet, Shallow/Deep ConvNets, lightweight Transformers), leverage self-supervised pretraining, and run real-time inference for biofeedback or simple control—all locally on a single GPU.

## Why this matters
- Bridges noisy consumer EEG to usable control signals.
- Emphasizes self-supervised pretraining and personalization for practical use.
- Fully local, privacy-preserving, extensible to EMG/EOG.

## Features
- Data: Unified loaders for MOABB/MNE/BIDS; configs for public datasets (e.g., PhysioNet MI, BCI IV 2a/2b, OpenMIIR).
- Preprocessing: Band-pass, notch, re-reference (CAR), epoching, ICA/ASR.
- Models: EEGNet, Shallow/Deep ConvNets, spectrogram-based Transformer.
- SSL: SimCLR-style contrastive, masked spectrogram autoencoding, multi-view time/freq.
- Training: PyTorch + Lightning, mixed precision, W&B/MLflow optional.
- Real-time: LSL receiver, sliding-window inference, smoothing; Streamlit/Gradio UI.
- Export: TorchScript/ONNX for low-latency inference.

## Install
Choose one of the following. No local virtualenv is created by default; a virtualenv is provisioned inside Docker.

- Poetry (recommended):
  - Ensure Poetry is installed, then run `poetry install`.
- uv (alternative):
  - `uv pip install -e .`
- pip (basic):
  - `pip install -e .`

GPU: Install a CUDA-enabled PyTorch matching your GPU/driver per https://pytorch.org/get-started/locally/.

## Quick start
- Prepare data (example MOABB dataset config):
  - `eegintent prep --config configs/data/bci_iv_2a.yaml`
- Train a baseline:
  - `eegintent train --config configs/train/eegnet_bci2a.yaml`
- Evaluate:
  - `eegintent eval --checkpoint checkpoints/eegnet_bci2a.ckpt --protocol loso`
- Real-time demo:
  - `eegintent realtime --checkpoint exports/model.onnx --lsl-stream EEG`

The CLI is defined in `src/eegintent/cli.py` with stub commands ready to extend.

## Project structure
- `src/` — package code (models, data, training, realtime, utils)
- `tests/` — unit/integration tests
- `docs/` — documentation (see `docs/project_plan.md`)
- `configs/` — YAML configs for data/models/training
- `scripts/` — helper scripts (lint, test, docker)
- `data/`, `assets/` — local data caches and static assets (not versioned)
- `.github/` — CI workflows and templates
- `.vscode/` — workspace settings for consistent dev experience

## Contributing
- Read `.github/CONTRIBUTING.md`.
- Use pre-commit hooks: `pre-commit install`.
- Style: black + ruff; type hints encouraged, mypy checked in CI.

## License
MIT License (see `LICENSE`).

## Acknowledgments
- MNE, MOABB, Braindecode, PyTorch Lightning, and the open EEG community.

## Universal Docker Development

Use the Docker Compose stack for a consistent, production-like dev environment.

- Quickstart:
  - Copy `.env.example` to `.env`
  - make up
  - make exec
  - make test / make precommit

- More details and troubleshooting: see `docs/universal_docker.md`.
