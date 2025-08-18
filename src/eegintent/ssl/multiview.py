from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _nt_xent_logits(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    reps = torch.cat([a, b], dim=0)
    sim = reps @ reps.t()
    n = a.size(0)
    mask = torch.eye(2 * n, dtype=torch.bool, device=sim.device)
    sim = sim.masked_fill(mask, float("-inf")) / temperature
    return sim


def cross_modal_contrastive(
    z_time: torch.Tensor, z_freq: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """CMC-style loss between time and frequency view projections.

    z_time and z_freq are projections of the same batch under two encoders.
    """
    logits = _nt_xent_logits(z_time, z_freq, temperature)
    b = z_time.size(0)
    labels = torch.cat(
        [
            torch.arange(b, 2 * b, device=logits.device),
            torch.arange(0, b, device=logits.device),
        ]
    )
    return F.cross_entropy(logits, labels)


class DualEncoder(nn.Module):
    def __init__(self, enc_time: nn.Module, enc_freq: nn.Module, proj_dim: int = 128):
        super().__init__()
        self.enc_time = enc_time
        self.enc_freq = enc_freq
        self.proj_time = nn.Linear(enc_time.out_dim, proj_dim)
        self.proj_freq = nn.Linear(enc_freq.out_dim, proj_dim)

    def forward(
        self, x_time: torch.Tensor, x_freq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zt = self.proj_time(self.enc_time(x_time))
        zf = self.proj_freq(self.enc_freq(x_freq))
        return zt, zf
