import torch

from eegintent.models.eegnet import EEGNet
from eegintent.models.shallow_convnet import ShallowConvNet


def test_eegnet_forward():
    x = torch.randn(2, 8, 128)
    model = EEGNet(n_channels=8, n_classes=3)
    y = model(x)
    assert y.shape == (2, 3)


def test_shallow_convnet_forward():
    x = torch.randn(2, 8, 128)
    model = ShallowConvNet(n_channels=8, n_classes=2)
    y = model(x)
    assert y.shape == (2, 2)
