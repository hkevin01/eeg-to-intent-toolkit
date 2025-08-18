from __future__ import annotations

import random
from typing import Iterator

import torch
import torch.nn.functional as f


def compute_prototypes(
    emb: torch.Tensor,
    labels: torch.Tensor,
) -> dict[int, torch.Tensor]:
    """Compute class prototypes as mean embeddings per label."""
    classes = labels.unique().tolist()
    protos = {}
    for c in classes:
        m = labels == c
        protos[int(c)] = emb[m].mean(dim=0)
    return protos


def prototypical_loss(
    emb: torch.Tensor,
    labels: torch.Tensor,
    prototypes: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Negative log-likelihood over distances to class prototypes."""
    # stack prototypes
    keys = sorted(prototypes.keys())
    proto_mat = torch.stack([prototypes[k] for k in keys], dim=0)  # (K, D)
    # distances: (B, K)
    dists = torch.cdist(emb.unsqueeze(0), proto_mat.unsqueeze(0)).squeeze(0)
    logits = -dists
    # map labels to prototype indices
    label_to_idx = {k: i for i, k in enumerate(keys)}
    idxs = [label_to_idx[int(lbl.item())] for lbl in labels]
    y = torch.tensor(idxs, device=emb.device)
    return f.cross_entropy(logits, y)


class EpisodeSampler:
    """Sample episodes for prototypical networks training."""

    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        n_way: int = 2,
        n_support: int = 5,
        n_query: int = 10,
        n_episodes: int = 100,
    ):
        """Initialize episode sampler.

        Args:
            data: Input data tensor (N, ...)
            labels: Labels tensor (N,)
            n_way: Number of classes per episode
            n_support: Number of support samples per class
            n_query: Number of query samples per class
            n_episodes: Number of episodes to generate
        """
        self.data = data
        self.labels = labels
        self.n_way = n_way
        self.n_support = n_support
        self.n_query = n_query
        self.n_episodes = n_episodes

        # Group indices by class
        self.class_indices = {}
        for i, label in enumerate(labels):
            class_id = int(label.item())
            if class_id not in self.class_indices:
                self.class_indices[class_id] = []
            self.class_indices[class_id].append(i)

        self.classes = list(self.class_indices.keys())

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Iterate over episodes.

        Yields:
            support_x: Support set data (n_way * n_support, ...)
            support_y: Support set labels (n_way * n_support,)
            query_x: Query set data (n_way * n_query, ...)
            query_y: Query set labels (n_way * n_query,)
        """
        for _ in range(self.n_episodes):
            # Sample classes for this episode
            episode_classes = random.sample(self.classes, self.n_way)

            support_indices = []
            query_indices = []
            support_labels = []
            query_labels = []

            for new_label, class_id in enumerate(episode_classes):
                # Sample indices for this class
                class_idx = self.class_indices[class_id]
                if len(class_idx) < self.n_support + self.n_query:
                    # Not enough samples, sample with replacement
                    total_needed = self.n_support + self.n_query
                    sampled = random.choices(class_idx, k=total_needed)
                else:
                    total_needed = self.n_support + self.n_query
                    sampled = random.sample(class_idx, total_needed)

                # Split into support and query
                support_idx = sampled[: self.n_support]
                query_end = self.n_support + self.n_query
                query_idx = sampled[self.n_support : query_end]

                support_indices.extend(support_idx)
                query_indices.extend(query_idx)

                # Use episode-specific labels (0, 1, ..., n_way-1)
                support_labels.extend([new_label] * self.n_support)
                query_labels.extend([new_label] * self.n_query)

            # Extract data
            support_x = self.data[support_indices]
            query_x = self.data[query_indices]
            support_y = torch.tensor(support_labels, device=self.data.device)
            query_y = torch.tensor(query_labels, device=self.data.device)

            yield support_x, support_y, query_x, query_y

    def __len__(self) -> int:
        return self.n_episodes


def episodic_train_step(
    model: torch.nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Single episodic training step for prototypical networks.

    Args:
        model: Encoder model that outputs embeddings
        support_x: Support set data
        support_y: Support set labels (episode-specific)
        query_x: Query set data
        query_y: Query set labels (episode-specific)

    Returns:
        loss: Prototypical loss
        metrics: Dictionary with accuracy and other metrics
    """
    # Get embeddings
    support_emb = model(support_x)
    query_emb = model(query_x)

    # Compute prototypes from support set
    prototypes = compute_prototypes(support_emb, support_y)

    # Compute loss on query set
    loss = prototypical_loss(query_emb, query_y, prototypes)

    # Compute accuracy
    keys = sorted(prototypes.keys())
    proto_mat = torch.stack([prototypes[k] for k in keys], dim=0)
    dists = torch.cdist(query_emb.unsqueeze(0), proto_mat.unsqueeze(0)).squeeze(0)
    pred_labels = dists.argmin(dim=1)
    acc = (pred_labels == query_y).float().mean()

    metrics = {
        "episode_acc": acc.item(),
        "episode_loss": loss.item(),
    }

    return loss, metrics


def evaluate_few_shot(
    model: torch.nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    query_x: torch.Tensor,
    query_y: torch.Tensor,
) -> dict[str, float]:
    """Evaluate model on a few-shot episode.

    Args:
        model: Trained encoder model
        support_x: Support set data
        support_y: Support set labels
        query_x: Query set data
        query_y: Query set labels

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    with torch.no_grad():
        # Get embeddings
        support_emb = model(support_x)
        query_emb = model(query_x)

        # Compute prototypes
        prototypes = compute_prototypes(support_emb, support_y)

        # Compute predictions
        keys = sorted(prototypes.keys())
        proto_mat = torch.stack([prototypes[k] for k in keys], dim=0)
        dists = torch.cdist(query_emb.unsqueeze(0), proto_mat.unsqueeze(0)).squeeze(0)
        pred_labels = dists.argmin(dim=1)

        # Compute metrics
        acc = (pred_labels == query_y).float().mean()
        loss = prototypical_loss(query_emb, query_y, prototypes)

        return {
            "accuracy": acc.item(),
            "loss": loss.item(),
            "n_support": len(support_x),
            "n_query": len(query_x),
        }
