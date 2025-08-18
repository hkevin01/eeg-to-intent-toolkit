from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
