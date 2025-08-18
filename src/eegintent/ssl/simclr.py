from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    reps = torch.cat([z1, z2], dim=0)
    sim = reps @ reps.t()
    mask = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    sim = sim[~mask].view(sim.size(0), -1)
    labels = torch.arange(z1.size(0), device=sim.device)
    labels = torch.cat([labels + z1.size(0) - 1, labels], dim=0)  # shifted due to mask removal
    sim = sim / temperature
    loss = F.cross_entropy(sim, labels)
    return loss
