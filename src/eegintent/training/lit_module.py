from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn, optim
from torchmetrics.classification import MulticlassAccuracy


@dataclass
class TrainConfig:
    lr: float = 1e-3
    n_classes: int = 2


class LitClassifier(pl.LightningModule):
    def __init__(
        self,
        backbone: nn.Module,
        n_classes: int = 2,
        lr: float = 1e-3,
        options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        # avoid pickling the module twice
        self.save_hyperparameters(ignore=["backbone"])
        self.backbone = backbone
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=n_classes)
        self.val_acc = MulticlassAccuracy(num_classes=n_classes)
        opts = options or {}
        self.use_wandb = bool(opts.get("use_wandb", False))
        self.use_mlflow = bool(opts.get("use_mlflow", False))
        self.use_personalization = bool(opts.get("use_personalization", False))
        self.personalization_layer = opts.get("personalization_layer")

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            self.use_personalization
            and self.personalization_layer is not None
            and subject_ids is not None
        ):
            x = self.personalization_layer(x, subject_ids)
        return self.backbone(x)

    def training_step(self, batch, _):  # type: ignore[override]
        has_sids = 3
        if len(batch) == has_sids:
            x, y, sids = batch
            logits = self(x, sids)
        else:
            x, y = batch
            logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=-1)
        acc = self.train_acc(preds, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):  # type: ignore[override]
        has_sids = 3
        if len(batch) == has_sids:
            x, y, sids = batch
            logits = self(x, sids)
        else:
            x, y = batch
            logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=-1)
        acc = self.val_acc(preds, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):  # type: ignore[override]
        return optim.AdamW(self.parameters(), lr=self.lr)
