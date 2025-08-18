from __future__ import annotations

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
