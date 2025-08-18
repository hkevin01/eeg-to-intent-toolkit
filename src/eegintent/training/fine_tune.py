from __future__ import annotations

from torch import nn


def freeze(module: nn.Module, except_heads: bool = True) -> None:
    for name, p in module.named_parameters():
        if except_heads and ("head" in name or "classifier" in name):
            p.requires_grad = True
        else:
            p.requires_grad = False


def layerwise_lr(model: nn.Module, base_lr: float, decay: float = 0.9) -> list[dict]:
    """Assign decreasing LR by depth (last layers get base_lr)."""
    params = list(model.named_parameters())
    groups: list[dict] = []
    n = len(params)
    for i, (_, p) in enumerate(params):
        lr = base_lr * (decay ** (n - i - 1))
        groups.append({"params": [p], "lr": lr})
    return groups
