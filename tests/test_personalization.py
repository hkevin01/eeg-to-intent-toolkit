import torch

from eegintent.personalization.film import FiLM
from eegintent.personalization.proto import compute_prototypes, prototypical_loss


def test_film_shapes():
    b, d, n_subj = 4, 16, 3
    x = torch.randn(b, d)
    sids = torch.randint(0, n_subj, (b,))
    film = FiLM(n_subjects=n_subj, feat_dim=d)
    y = film(x, sids)
    assert y.shape == x.shape


def test_prototypical_loss():
    b, d, n_classes = 8, 32, 3
    emb = torch.randn(b, d)
    labels = torch.randint(0, n_classes, (b,))
    protos = compute_prototypes(emb, labels)
    loss = prototypical_loss(emb, labels, protos)
    assert loss.dim() == 0 and torch.isfinite(loss)
