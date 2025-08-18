from __future__ import annotations

from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ..models.eegnet import EEGNet
from .lit_module import LitClassifier


def make_trainer(config: Dict) -> Tuple[pl.Trainer, LitClassifier]:
    n_channels = config.get("n_channels", 8)
    n_classes = config.get("n_classes", 2)
    lr = config.get("lr", 1e-3)
    use_wandb = bool(config.get("use_wandb", False))
    use_mlflow = bool(config.get("use_mlflow", False))

    model = EEGNet(n_channels=n_channels, n_classes=n_classes)
    lit = LitClassifier(
        model, lr=lr, n_classes=n_classes, use_wandb=use_wandb, use_mlflow=use_mlflow
    )

    loggers = []
    if use_wandb:
        try:
            from pytorch_lightning.loggers import WandbLogger  # type: ignore

            loggers.append(WandbLogger(project=config.get("wandb_project", "eegintent")))
        except Exception:
            pass
    if use_mlflow:
        try:
            from pytorch_lightning.loggers import MLFlowLogger  # type: ignore

            loggers.append(
                MLFlowLogger(experiment_name=config.get("mlflow_experiment", "eegintent"))
            )
        except Exception:
            pass

    trainer = pl.Trainer(
        max_epochs=config.get("max_epochs", 1),
        enable_checkpointing=False,
        logger=loggers if loggers else True,
    )
    return trainer, lit


class TensorDatasetWithSubject(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, subject_ids: torch.Tensor):
        assert len(x) == len(y) == len(subject_ids)
        self.x = x
        self.y = y
        self.sids = subject_ids

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.sids[idx]


def make_dataloaders(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 8,
    subject_ids: torch.Tensor | None = None,
):
    if subject_ids is not None:
        ds: Dataset = TensorDatasetWithSubject(x, y, subject_ids)
    else:
        ds = TensorDataset(x, y)
    return (
        DataLoader(ds, batch_size=batch_size),
        DataLoader(ds, batch_size=batch_size),
    )
