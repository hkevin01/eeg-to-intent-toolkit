from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..models.eegnet import EEGNet
from ..training.lit_module import LitClassifier
from ..training.trainer import make_dataloaders, make_trainer


@dataclass
class AblationConfig:
    n_channels: int = 8
    n_classes: int = 2
    lr: float = 1e-3
    max_epochs: int = 1
    batch_size: int = 8
    use_ssl_backbone: bool = False
    use_personalization: bool = False
    personalization_mode: str = "film"  # or "cbn"
    n_subjects: int = 4
    augment_noise_std: float = 0.0


def _maybe_augment(x: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return x
    return x + std * torch.randn_like(x)


def run_experiment(config: dict[str, Any]) -> dict[str, float]:
    # Build base model
    n_channels = int(config.get("n_channels", 8))
    n_classes = int(config.get("n_classes", 2))
    model = EEGNet(n_channels=n_channels, n_classes=n_classes)

    # Lightning module
    lit = LitClassifier(model, lr=float(config.get("lr", 1e-3)), n_classes=n_classes)

    # Synthetic data for demo
    n = 64
    x = torch.randn(n, n_channels, 128)
    y = torch.randint(0, n_classes, (n,))

    aug_std = float(config.get("augment_noise_std", 0.0))
    x = _maybe_augment(x, aug_std)

    train_loader, val_loader = make_dataloaders(x, y, batch_size=int(config.get("batch_size", 8)))

    trainer, lit = make_trainer(
        {
            "n_channels": n_channels,
            "n_classes": n_classes,
            "lr": float(config.get("lr", 1e-3)),
            "max_epochs": int(config.get("max_epochs", 1)),
            "use_wandb": False,
            "use_mlflow": False,
        }
    )

    trainer.fit(lit, train_loader, val_loader)
    return {"val_acc": float(lit.val_acc.compute().item())}
