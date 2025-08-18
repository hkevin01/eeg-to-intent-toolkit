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
    """Normalized temperature-scaled cross entropy (SimCLR).

    Expects z1 and z2 as two augmented views, same batch size.
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    reps = torch.cat([z1, z2], dim=0)  # [2B, D]
    sim = reps @ reps.t()
    b = z1.size(0)
    # mask self-similarities
    mask = torch.eye(2 * b, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(mask, float("-inf"))
    sim = sim / temperature
    # positives are at offset b for each sample in both halves
    labels = torch.cat(
        [
            torch.arange(b, 2 * b, device=sim.device),
            torch.arange(0, b, device=sim.device),
        ]
    )
    loss = F.cross_entropy(sim, labels)
    return loss
