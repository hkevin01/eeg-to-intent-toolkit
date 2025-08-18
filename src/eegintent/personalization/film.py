from __future__ import annotations

import torch
from torch import nn

FEAT_2D = 2
FEAT_3D = 3


class FiLM(nn.Module):
    """Feature-wise Linear Modulation with learnable subject embeddings.

    Given subject_id, produce gamma/beta and modulate features.
    """

    def __init__(self, n_subjects: int, feat_dim: int, emb_dim: int = 32):
        super().__init__()
        self.emb = nn.Embedding(n_subjects, emb_dim)
        self.gamma = nn.Linear(emb_dim, feat_dim)
        self.beta = nn.Linear(emb_dim, feat_dim)

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        # x: (B, D) or (B, C, T) -> we broadcast on last dim
        e = self.emb(subject_ids)
        g = self.gamma(e)
        b = self.beta(e)
        if x.dim() == FEAT_2D:
            return x * (1 + g) + b
        elif x.dim() == FEAT_3D:
            return x * (1 + g.unsqueeze(-1)) + b.unsqueeze(-1)
        else:
            raise ValueError("Unsupported input shape for FiLM")


class ConditionalBatchNorm1d(nn.Module):
    """CBN with per-subject scale/shift via embeddings."""

    def __init__(self, num_features: int, n_subjects: int, emb_dim: int = 32):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.emb = nn.Embedding(n_subjects, emb_dim)
        self.to_gamma = nn.Linear(emb_dim, num_features)
        self.to_beta = nn.Linear(emb_dim, num_features)

    def forward(self, x: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> BatchNorm1d supports (N, C, L)
        y = self.bn(x)
        e = self.emb(subject_ids)
        g = 1 + self.to_gamma(e).unsqueeze(-1)
        beta = self.to_beta(e).unsqueeze(-1)
        return y * g + beta
