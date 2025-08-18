from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def patchify(x: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
    """Split spectrogram (B, C, H, W) into patches (B, N, P).

    P = C*ph*pw, N = (H/ph)*(W/pw)
    """
    b, c, h, w = x.shape
    assert h % ph == 0 and w % pw == 0
    x = x.view(b, c, h // ph, ph, w // pw, pw)
    # (B, H/ph, W/pw, C, ph, pw)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    n = (h // ph) * (w // pw)
    return x.view(b, n, c * ph * pw)


class SimpleMAE(nn.Module):
    def __init__(self, in_ch: int = 1, ph: int = 4, pw: int = 4, hidden: int = 256):
        super().__init__()
        self.ph, self.pw = ph, pw
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Linear(32 * ph * pw, hidden)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_ch * ph * pw),
        )

    def forward(
        self, x: torch.Tensor, mask_ratio: float = 0.5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W)
        feat = self.encoder(x)
        patches = patchify(feat, self.ph, self.pw)
        _, n, _ = patches.shape
        k = int(n * mask_ratio)
        idx = torch.randperm(n, device=x.device)
        keep = idx[k:]
        tokens = self.proj(patches[:, keep])  # (B, N_keep, H)
        recon = self.decoder(tokens)  # (B, N_keep, C*ph*pw)
        target = patches[:, keep]
        loss = F.mse_loss(recon, target)
        return recon, loss
