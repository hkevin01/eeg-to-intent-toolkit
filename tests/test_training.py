import torch

from eegintent.training.trainer import make_dataloaders, make_trainer


def test_trainer_smoke():
    config = {"n_channels": 4, "n_classes": 2, "max_epochs": 1}
    trainer, lit = make_trainer(config)
    X = torch.randn(16, 4, 64)
    y = torch.randint(0, 2, (16,))
    train_loader, val_loader = make_dataloaders(X, y, batch_size=8)
    trainer.fit(lit, train_loader, val_loader)
