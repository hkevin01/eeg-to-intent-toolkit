import torch

from eegintent.ssl.mae import SimpleMAE
from eegintent.ssl.multiview import cross_modal_contrastive


def test_cmc_loss_runs():
    b, d = 8, 64
    zt = torch.randn(b, d)
    zf = torch.randn(b, d)
    loss = cross_modal_contrastive(zt, zf)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_mae_forward():
    b, c, h, w = 2, 1, 32, 32
    x = torch.randn(b, c, h, w)
    mae = SimpleMAE(in_ch=c, ph=4, pw=4)
    recon, loss = mae(x, mask_ratio=0.5)
    assert recon.shape[0] == b and torch.isfinite(loss)
