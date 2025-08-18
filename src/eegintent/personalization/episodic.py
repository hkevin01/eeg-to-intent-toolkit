"""Lightning module for prototypical networks episodic training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
from torch import nn, optim
from torchmetrics import MeanMetric

if TYPE_CHECKING:
    from .proto import EpisodeSampler

try:
    from .proto import episodic_train_step, evaluate_few_shot
except ImportError:
    episodic_train_step = None
    evaluate_few_shot = None


class ProtoNetLightning(pl.LightningModule):
    """Lightning module for prototypical networks."""

    def __init__(
        self,
        encoder: nn.Module,
        lr: float = 1e-3,
        n_way: int = 2,
        n_support: int = 5,
        n_query: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder"])
        self.encoder = encoder
        self.lr = lr
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query

        # Metrics
        self.train_loss = MeanMetric()
        self.train_acc = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_acc = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder."""
        return self.encoder(x)

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        """Training step for episodic learning."""
        if episodic_train_step is None:
            raise ImportError("episodic_train_step not available")

        support_x, support_y, query_x, query_y = batch

        loss, metrics = episodic_train_step(self.encoder, support_x, support_y, query_x, query_y)

        # Update metrics
        self.train_loss(loss)
        self.train_acc(metrics["episode_acc"])

        # Log
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", metrics["episode_acc"], on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, _):  # type: ignore[override]
        """Validation step for episodic learning."""
        if evaluate_few_shot is None:
            raise ImportError("evaluate_few_shot not available")

        support_x, support_y, query_x, query_y = batch

        metrics = evaluate_few_shot(self.encoder, support_x, support_y, query_x, query_y)

        # Update metrics
        self.val_loss(metrics["loss"])
        self.val_acc(metrics["accuracy"])

        # Log
        self.log("val/loss", metrics["loss"], on_step=False, on_epoch=True)
        self.log("val/acc", metrics["accuracy"], on_step=False, on_epoch=True)

    def configure_optimizers(self):  # type: ignore[override]
        """Configure optimizer."""
        return optim.AdamW(self.encoder.parameters(), lr=self.lr)

    def on_train_epoch_end(self):  # type: ignore[override]
        """Log epoch metrics."""
        self.log("train/epoch_loss", self.train_loss.compute())
        self.log("train/epoch_acc", self.train_acc.compute())
        self.train_loss.reset()
        self.train_acc.reset()

    def on_validation_epoch_end(self):  # type: ignore[override]
        """Log validation epoch metrics."""
        self.log("val/epoch_loss", self.val_loss.compute())
        self.log("val/epoch_acc", self.val_acc.compute())
        self.val_loss.reset()
        self.val_acc.reset()


class EpisodeDataLoader:
    """DataLoader wrapper for episodic sampling."""

    def __init__(self, episode_sampler: EpisodeSampler):
        self.episode_sampler = episode_sampler

    def __iter__(self):
        """Iterate over episodes."""
        return iter(self.episode_sampler)

    def __len__(self) -> int:
        """Number of episodes."""
        return len(self.episode_sampler)


def create_prototypical_trainer(
    data: torch.Tensor,
    labels: torch.Tensor,
    encoder: nn.Module,
    n_way: int = 2,
    n_support: int = 5,
    n_query: int = 10,
    n_train_episodes: int = 100,
    n_val_episodes: int = 50,
    max_epochs: int = 10,
    lr: float = 1e-3,
) -> tuple[pl.Trainer, ProtoNetLightning, EpisodeDataLoader, EpisodeDataLoader]:
    """Create trainer and module for prototypical networks.

    Args:
        data: Training data tensor
        labels: Training labels tensor
        encoder: Encoder network
        n_way: Number of classes per episode
        n_support: Support samples per class
        n_query: Query samples per class
        n_train_episodes: Episodes per training epoch
        n_val_episodes: Episodes per validation epoch
        max_epochs: Maximum training epochs
        lr: Learning rate

    Returns:
        Trainer and Lightning module
    """
    try:
        from .proto import EpisodeSampler
    except ImportError as exc:
        raise ImportError("EpisodeSampler not available") from exc

    # Create episode samplers
    train_sampler = EpisodeSampler(data, labels, n_way, n_support, n_query, n_train_episodes)
    val_sampler = EpisodeSampler(data, labels, n_way, n_support, n_query, n_val_episodes)

    # Create data loaders
    train_loader = EpisodeDataLoader(train_sampler)
    val_loader = EpisodeDataLoader(val_sampler)

    # Create Lightning module
    model = ProtoNetLightning(
        encoder=encoder,
        lr=lr,
        n_way=n_way,
        n_support=n_support,
        n_query=n_query,
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        enable_checkpointing=False,
        logger=False,  # Disable logging for simplicity
    )

    return trainer, model, train_loader, val_loader
