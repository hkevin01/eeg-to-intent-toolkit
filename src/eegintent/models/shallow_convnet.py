from __future__ import annotations

import torch
import torch.nn as nn


class ShallowConvNet(nn.Module):
    def __init__(self, n_channels: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, 8, kernel_size=11, padding=5),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(8, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(x).squeeze(-1))
