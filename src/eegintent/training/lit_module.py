from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import MulticlassAccuracy


@dataclass
class TrainConfig:
    lr: float = 1e-3
    n_classes: int = 2


class LitClassifier(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        lr: float = 1e-3,
        n_classes: int = 2,
        use_wandb: bool = False,
        use_mlflow: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["backbone"])  # avoid pickling the module twice
        self.backbone = backbone
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=n_classes)
        self.val_acc = MulticlassAccuracy(num_classes=n_classes)
        self.use_wandb = use_wandb
        self.use_mlflow = use_mlflow

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=-1)
        acc = self.train_acc(preds, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore[override]
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=-1)
        acc = self.val_acc(preds, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):  # type: ignore[override]
        return optim.AdamW(self.parameters(), lr=self.lr)
