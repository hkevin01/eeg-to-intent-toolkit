# Project Goals

## Purpose
Build a robust, local, GPU-accelerated EEG-to-Intent toolkit enabling practical mental state decoding and real-time interaction using open data and models.

## Short-term goals (0–2 months)
- Baseline data loaders, preprocessing, and models (EEGNet, ShallowConvNet).
- Lightning training loop with metrics and checkpoints.
- Self-supervised pretraining (SimCLR) prototype.
- Real-time LSL receiver and sliding-window inference stub.
- CI (lint, type, tests) and docs scaffold.

## Long-term goals (3–6+ months)
- Multi-view SSL (time/freq) and spectrogram MAE.
- Personalization (prototypical, adapters, conditional norms).
- Model zoo with reproducible benchmarks and public leaderboard.
- Streamlit/Gradio dashboard with calibration wizard.
- Export and deployment via ONNX/TorchScript microservice.

## Audience
- BCI practitioners, researchers, and makers.
- Students and hobbyists using consumer EEG.
- Privacy-conscious users preferring local inference.
