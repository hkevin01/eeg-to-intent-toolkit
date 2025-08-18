from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """Minimal EEGNet-like placeholder producing logits from (B, C, T)."""

    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 16, kernel_size=7, padding=3),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(16, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).squeeze(-1)
        return self.classifier(h)
