# Project Plan: EEG-to-Intent Toolkit

This project builds a robust, local, GPU-accelerated pipeline to decode actionable mental states from EEG using supervised and self-supervised learning, plus a real-time app for practical interaction.

Overview of current repo: initial scaffold with a modern src layout, CI, Docker, and a minimal CLI. Core functionality (data loaders, preprocessing, models, SSL, realtime) is stubbed and incrementally implemented.

## Phases and checklists

### Phase 1 — Data + Baselines
- [x] Implement unified dataset loaders (MOABB, BCI IV 2a/2b, PhysioNet MI)
  - Action items: MNE-BIDS support; caching; subject/session splits; channel mapping.
  - Options: Use MOABB pipelines vs custom MNE pipelines.
- [x] Preprocessing pipeline (filters, notch, CAR, ICA/ASR, epoching)
  - Action items: config-driven steps; quality metrics; artifact logs.
  - Options: `autoreject` vs manual thresholds; ASR via `mne-icalabel` alternative.
- [x] Baseline models (EEGNet, Shallow/Deep ConvNet)
  - Action items: implement modules; unit tests for shapes; LightningModule.
  - Options: import from Braindecode for baselines vs own re-implementation.
- [x] Training loop (Lightning) with mixed precision and checkpointing
  - Action items: metrics (balanced acc, kappa, F1); W&B/MLflow toggles.
  - Options: Fabric (custom loops) vs Lightning Trainer.
- [x] Evaluation protocols (within-subject, cross-subject, LOSO)
  - Action items: standardized splits; reporting; reproducibility seeds.

### Phase 2 — Self-Supervised Pretraining
- [x] Contrastive learning (SimCLR/InfoNCE) on time-series and spectrograms
  - Action items: augmentations; projection head; temperature schedule.
  - Options: NT-Xent vs Circle loss; multi-crop variants.
- [x] Multi-view learning (TS-TCC/CMC variants)
  - Action items: dual encoders (time/freq); cross-modal contrastive loss.
  - Options: shared vs separate backbones; momentum encoder.
- [x] Masked autoencoder for spectrograms (ViT-lite)
  - Action items: patchify; masking policy; lightweight decoder; recon loss.
  - Options: channel-wise vs time-freq masking; conv-ViT hybrids.
- [x] Pretraining data pool
  - Action items: MOABB aggregation; normalization alignment; license notes.
  - Options: subject-agnostic vs subject-aware objectives.
- [x] Fine-tuning recipes
  - Action items: layer-wise LR decay; freezing policies; adapters.

### Phase 3 — Personalization + Domain Adaptation
- [x] Subject-adaptive normalization (FiLM, conditional BN)
  - Action items: subject embeddings; per-subject gamma/beta. (Done)
  - Options: adapters vs LoRA-style low-rank updates.
- [x] Few-shot calibration (prototypical networks)
  - Action items: centroid computation; episodic training. (Done - EpisodeSampler, ProtoNetLightning, episodic training functions)
  - Options: ProtoMAML vs simple nearest-centroid.
- [x] Riemannian alignment baselines
  - Action items: covariance features; tangent space; CSP+LDA. (CSP+LDA via pyriemann)
  - Options: pyriemann pipelines vs custom.
- [x] Robustness & ablations
  - Action items: augmentation sweeps; SSL vs scratch; personalization off/on. (Done - SSL backbone loading, augmentation toggles, personalization switches)
  - Options: domain-adversarial training (DANN) add-on.
- [x] Reproducibility & artifacts
  - Action items: version pinning; seeds; artifact registry. (Done - comprehensive ArtifactRegistry system with experiment tracking)

### Phase 4 — Real-time Inference App
- [ ] LSL-based receiver and stream buffers
  - Action items: stream discovery; ring buffers; timestamp alignment.
  - Options: WebSockets bridge for browsers.
- [ ] Real-time DSP (filters, artifact handling)
  - Action items: IIR/FIR options; notch; adaptive noise.
  - Options: fastpath C++ filter via ctypes for low latency.
- [ ] Sliding-window inference + smoothing
  - Action items: softmax averaging; refractory period; HMM/CRF experiment.
  - Options: TorchScript vs ONNX runtime execution.
- [ ] Dashboard (Streamlit/Gradio)
  - Action items: calibration wizard; feedback widgets; latency monitor.
  - Options: FastAPI backend with WebSocket stream.
- [ ] Packaging & export
  - Action items: ONNX/TorchScript exporters; benchmark latency; configs.

### Phase 5 — Benchmarks, Docs, and Model Zoo
- [ ] Reproducible benchmark scripts
  - Action items: run_{dataset}_{model}.py; seed control; logs.
- [ ] Public W&B leaderboard and model registry
  - Action items: sweep configs; artifact naming.
- [ ] Tutorials and curriculum
  - Action items: quickstart; SSL; personalization; realtime demo.
- [ ] Pre-trained checkpoints hosting (no data redistribution)
  - Action items: license review; hosting strategy.
- [ ] Paper-style report and demo video
  - Action items: ablations; figures; narrative.

## Risks & mitigations
- Data heterogeneity → strict channel mapping, normalization alignment.
- Latency constraints → export + runtime optimization; batching and CUDA graphs.
- Personalization overfitting → few-shot regularization, subject splits.
