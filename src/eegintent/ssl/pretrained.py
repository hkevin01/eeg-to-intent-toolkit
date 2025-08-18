"""SSL pretrained backbone loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from torch import nn

    from ..models.eegnet import EEGNet
else:
    # Handle optional dependencies
    try:
        import torch
        from torch import nn

        from ..models.eegnet import EEGNet
    except ImportError:
        torch = None
        nn = None
        EEGNet = None

# Import SSL modules conditionally
try:
    from .mae import SimpleMAE
except ImportError:
    SimpleMAE = None


class SSLEncoder(nn.Module):
    """Wrapper for SSL pre-trained encoders.

    This extracts the feature representation from SSL models,
    removing any task-specific heads or projection layers.
    """

    def __init__(self, backbone: nn.Module, feature_dim: int):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input."""
        # For most SSL models, we want the representation before
        # any projection heads or classifiers
        if hasattr(self.backbone, "features"):
            return self.backbone.features(x)
        elif hasattr(self.backbone, "encoder"):
            return self.backbone.encoder(x)
        else:
            # Fallback: use full backbone (might include classification head)
            return self.backbone(x)


def load_ssl_backbone(
    checkpoint_path: str | Path,
    backbone_type: str = "simclr",
    n_channels: int = 8,
    **kwargs: Any,
) -> nn.Module:
    """Load a pre-trained SSL backbone from checkpoint.

    Args:
        checkpoint_path: Path to the SSL model checkpoint
        backbone_type: Type of SSL model ("simclr", "mae", "multiview")
        n_channels: Number of input channels
        **kwargs: Additional arguments for backbone construction

    Returns:
        A feature encoder ready for fine-tuning
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SSL checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract state dict (handle Lightning checkpoints)
    state_dict = checkpoint.get("state_dict", checkpoint)

    # Create appropriate backbone based on type
    if backbone_type == "simclr":
        # For SimCLR, we want the encoder without projection head
        # Create a base encoder (could be EEGNet or other)
        # temp n_classes for feature extraction
        backbone = EEGNet(n_channels=n_channels, n_classes=128)
        # Remove classifier for feature extraction
        backbone.classifier = nn.Identity()
        feature_dim = 16  # EEGNet feature dim

    elif backbone_type == "mae":
        # For MAE, we want the encoder part
        if SimpleMAE is None:
            raise ImportError("SimpleMAE not available")
        mae_model = SimpleMAE(in_ch=1, **kwargs)
        # Extract encoder
        backbone = mae_model.encoder
        feature_dim = 32  # From MAE encoder output

    elif backbone_type == "multiview":
        # For multiview, we might want one of the encoders
        # This is more complex as it has dual encoders
        backbone = EEGNet(n_channels=n_channels, n_classes=128)
        backbone.classifier = nn.Identity()
        feature_dim = 16

    else:
        raise ValueError(f"Unknown backbone_type: {backbone_type}")

    # Load weights with prefix handling for Lightning modules
    try:
        # Try direct loading
        backbone.load_state_dict(state_dict, strict=False)
    except RuntimeError:
        # Handle prefixed keys (e.g., "backbone.", "encoder.", etc.)
        filtered_dict = {}
        for orig_key, value in state_dict.items():
            # Remove common prefixes
            new_key = orig_key
            for prefix in ["backbone.", "encoder.", "enc_time.", "model."]:
                if orig_key.startswith(prefix):
                    new_key = orig_key[len(prefix) :]
                    break
            filtered_dict[new_key] = value

        backbone.load_state_dict(filtered_dict, strict=False)

    return SSLEncoder(backbone, feature_dim)


def create_ssl_backbone_factory(
    ssl_checkpoint_dir: str | Path = "checkpoints/ssl",
    default_backbone_type: str = "simclr",
) -> dict[str, Any]:
    """Create a factory function for SSL backbones.

    Returns a config dict that can be used in make_trainer.
    """
    ssl_dir = Path(ssl_checkpoint_dir)

    # Look for available checkpoints
    available_checkpoints = {}
    if ssl_dir.exists():
        for ckpt_file in ssl_dir.glob("*.ckpt"):
            # Infer type from filename
            name = ckpt_file.stem.lower()
            if "simclr" in name:
                available_checkpoints["simclr"] = ckpt_file
            elif "mae" in name:
                available_checkpoints["mae"] = ckpt_file
            elif "multiview" in name or "cmc" in name:
                available_checkpoints["multiview"] = ckpt_file

    def ssl_backbone_factory(n_channels: int, n_classes: int, **kwargs) -> nn.Module:
        """Factory to create SSL backbone with downstream head."""
        backbone_type = kwargs.get("ssl_backbone_type", default_backbone_type)

        if backbone_type in available_checkpoints:
            # Load pre-trained SSL backbone
            ssl_encoder = load_ssl_backbone(
                available_checkpoints[backbone_type],
                backbone_type=backbone_type,
                n_channels=n_channels,
            )
            # Add classification head
            classifier = nn.Linear(ssl_encoder.feature_dim, n_classes)
            return nn.Sequential(ssl_encoder, classifier)
        else:
            # Fallback to random init if no checkpoint available
            msg = f"Warning: No SSL checkpoint found for {backbone_type}, " "using random init"
            print(msg)
            return EEGNet(n_channels=n_channels, n_classes=n_classes)

    return {
        "backbone_factory": ssl_backbone_factory,
        "available_checkpoints": list(available_checkpoints.keys()),
    }
