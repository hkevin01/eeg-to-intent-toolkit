from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..personalization.film import ConditionalBatchNorm1d, FiLM
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
    """Run a small ablation experiment on synthetic data.

    Config keys supported:
    - n_channels, n_classes, lr, max_epochs, batch_size
    - augment_noise_std
    - use_personalization (bool)
    - personalization_mode ("film"|"cbn")
    - n_subjects (int)
    - use_ssl_backbone (bool) — placeholder toggle for now
    """
    # Parse config
    n_channels = int(config.get("n_channels", 8))
    n_classes = int(config.get("n_classes", 2))
    batch_size = int(config.get("batch_size", 8))
    max_epochs = int(config.get("max_epochs", 1))
    lr = float(config.get("lr", 1e-3))
    use_personalization = bool(config.get("use_personalization", False))
    personalization_mode = str(config.get("personalization_mode", "film")).lower()
    n_subjects = int(config.get("n_subjects", 4))
    use_ssl_backbone = bool(config.get("use_ssl_backbone", False))

    # Build base model (SSL toggle is a stub here).
    # A future version could select a pre-trained encoder when
    # use_ssl_backbone is True.

    # Personalization layer (optional)
    personalization_layer = None
    if use_personalization:
        if personalization_mode == "film":
            personalization_layer = FiLM(n_subjects=n_subjects, feat_dim=n_channels)
        elif personalization_mode == "cbn":
            personalization_layer = ConditionalBatchNorm1d(
                num_features=n_channels, n_subjects=n_subjects
            )
        else:
            raise ValueError("personalization_mode must be 'film' or 'cbn'")

    # Synthetic data for demo
    n = 64
    x = torch.randn(n, n_channels, 128)
    y = torch.randint(0, n_classes, (n,))
    subject_ids = torch.randint(0, n_subjects, (n,)) if use_personalization else None

    # Optional augmentation
    aug_std = float(config.get("augment_noise_std", 0.0))
    x = _maybe_augment(x, aug_std)

    # Dataloaders (subject-aware when needed)
    train_loader, val_loader = make_dataloaders(
        x, y, batch_size=batch_size, subject_ids=subject_ids
    )

    # Trainer + Lightning module
    trainer, lit = make_trainer(
        {
            "n_channels": n_channels,
            "n_classes": n_classes,
            "lr": lr,
            "max_epochs": max_epochs,
            "use_wandb": False,
            "use_mlflow": False,
            "use_personalization": use_personalization,
            "personalization_layer": personalization_layer,
            "use_ssl_backbone": use_ssl_backbone,
        }
    )

    trainer.fit(lit, train_loader, val_loader)
    return {"val_acc": float(lit.val_acc.compute().item())}
