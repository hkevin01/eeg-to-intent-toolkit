"""
Model hosting and distribution strategy for EEG-to-Intent toolkit.

This module provides utilities for hosting, distributing, and managing
pre-trained model checkpoints while ensuring license compliance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class ModelInfo:
    """Information about a hosted model."""

    # Model identification
    name: str
    dataset: str
    architecture: str
    method: str  # supervised, ssl_pretrained, personalized, etc.

    # Version and metadata
    version: str
    created_date: str
    description: str

    # Performance metrics
    accuracy: float
    kappa: float
    f1_score: float
    cross_subject_accuracy: Optional[float] = None

    # Technical details
    model_size_mb: float
    input_shape: List[int]
    n_classes: int
    n_channels: int
    sampling_rate: float

    # Files and checksums
    checkpoint_url: str
    config_url: str
    checkpoint_sha256: str
    config_sha256: str

    # Legal and usage
    license: str
    citation: str
    training_data_sources: List[str]
    usage_restrictions: Optional[str] = None

    # Authors and contact
    authors: List[str]
    contact_email: str
    repository_url: str


class ModelRegistry:
    """Registry for managing and accessing hosted models."""

    def __init__(self, registry_url: str = None):
        """Initialize model registry."""
        self.registry_url = registry_url or self._get_default_registry_url()
        self.models: Dict[str, ModelInfo] = {}
        self._load_registry()

    def _get_default_registry_url(self) -> str:
        """Get default model registry URL."""
        # In practice, this would point to a hosted registry
        return (
            "https://github.com/your-org/eeg-intent-models/releases/latest/download/registry.json"
        )

    def _load_registry(self):
        """Load model registry from remote source."""
        if not REQUESTS_AVAILABLE:
            print("Warning: requests not available, using local registry")
            self._load_local_registry()
            return

        try:
            response = requests.get(self.registry_url, timeout=10)
            response.raise_for_status()
            registry_data = response.json()

            for model_data in registry_data.get("models", []):
                model_info = ModelInfo(**model_data)
                self.models[model_info.name] = model_info

        except Exception as e:
            print(f"Failed to load remote registry: {e}")
            print("Falling back to local registry")
            self._load_local_registry()

    def _load_local_registry(self):
        """Load local model registry as fallback."""
        # Define some example models for demonstration
        self.models = {
            "bcic_iv_2a_eegnet_supervised": ModelInfo(
                name="bcic_iv_2a_eegnet_supervised",
                dataset="bcic_iv_2a",
                architecture="eegnet",
                method="supervised",
                version="1.0.0",
                created_date="2025-01-15",
                description="EEGNet trained on BCIC IV 2a dataset with standard supervised learning",
                accuracy=0.823,
                kappa=0.764,
                f1_score=0.815,
                cross_subject_accuracy=0.687,
                model_size_mb=2.3,
                input_shape=[22, 1000],  # channels x samples
                n_classes=4,
                n_channels=22,
                sampling_rate=250.0,
                checkpoint_url="https://github.com/your-org/eeg-intent-models/releases/download/v1.0.0/bcic_iv_2a_eegnet_supervised.ckpt",
                config_url="https://github.com/your-org/eeg-intent-models/releases/download/v1.0.0/bcic_iv_2a_eegnet_supervised_config.yaml",
                checkpoint_sha256="a1b2c3d4e5f6...",
                config_sha256="f6e5d4c3b2a1...",
                license="MIT",
                citation="@article{eeg_intent_2025, title={EEG-to-Intent Toolkit}, author={Your Name}, year={2025}}",
                training_data_sources=["BCIC IV 2a (publicly available)"],
                usage_restrictions=None,
                authors=["Your Name", "Collaborator Name"],
                contact_email="your.email@institution.edu",
                repository_url="https://github.com/your-org/eeg-to-intent-toolkit",
            ),
            "bcic_iv_2a_eegnet_ssl": ModelInfo(
                name="bcic_iv_2a_eegnet_ssl",
                dataset="bcic_iv_2a",
                architecture="eegnet",
                method="ssl_pretrained",
                version="1.0.0",
                created_date="2025-01-15",
                description="EEGNet with SSL pretraining on large-scale EEG data, fine-tuned on BCIC IV 2a",
                accuracy=0.867,
                kappa=0.823,
                f1_score=0.858,
                cross_subject_accuracy=0.745,
                model_size_mb=2.3,
                input_shape=[22, 1000],
                n_classes=4,
                n_channels=22,
                sampling_rate=250.0,
                checkpoint_url="https://github.com/your-org/eeg-intent-models/releases/download/v1.0.0/bcic_iv_2a_eegnet_ssl.ckpt",
                config_url="https://github.com/your-org/eeg-intent-models/releases/download/v1.0.0/bcic_iv_2a_eegnet_ssl_config.yaml",
                checkpoint_sha256="b2c3d4e5f6a1...",
                config_sha256="e5f6a1b2c3d4...",
                license="MIT",
                citation="@article{eeg_intent_ssl_2025, title={Self-Supervised Learning for EEG}, author={Your Name}, year={2025}}",
                training_data_sources=["BCIC IV 2a", "PhysioNet MI", "Large-scale unlabeled EEG"],
                usage_restrictions="Academic use only for SSL pretraining data",
                authors=["Your Name", "SSL Expert"],
                contact_email="your.email@institution.edu",
                repository_url="https://github.com/your-org/eeg-to-intent-toolkit",
            ),
        }

    def list_models(
        self,
        dataset: Optional[str] = None,
        architecture: Optional[str] = None,
        method: Optional[str] = None,
    ) -> List[ModelInfo]:
        """List available models with optional filtering."""
        models = list(self.models.values())

        if dataset:
            models = [m for m in models if m.dataset == dataset]
        if architecture:
            models = [m for m in models if m.architecture == architecture]
        if method:
            models = [m for m in models if m.method == method]

        return sorted(models, key=lambda x: x.accuracy, reverse=True)

    def get_model(self, name: str) -> Optional[ModelInfo]:
        """Get model info by name."""
        return self.models.get(name)

    def download_model(
        self, name: str, cache_dir: str = "~/.cache/eeg_intent_models", verify_checksum: bool = True
    ) -> Dict[str, str]:
        """Download model files to local cache."""

        model_info = self.get_model(name)
        if not model_info:
            raise ValueError(f"Model '{name}' not found in registry")

        cache_path = Path(cache_dir).expanduser()
        cache_path.mkdir(parents=True, exist_ok=True)

        # Download checkpoint
        checkpoint_path = cache_path / f"{name}.ckpt"
        config_path = cache_path / f"{name}_config.yaml"

        if not checkpoint_path.exists():
            print(f"Downloading checkpoint for {name}...")
            self._download_file(model_info.checkpoint_url, checkpoint_path)

            if verify_checksum:
                self._verify_checksum(checkpoint_path, model_info.checkpoint_sha256)

        if not config_path.exists():
            print(f"Downloading config for {name}...")
            self._download_file(model_info.config_url, config_path)

            if verify_checksum:
                self._verify_checksum(config_path, model_info.config_sha256)

        return {
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "model_info": model_info,
        }

    def _download_file(self, url: str, output_path: Path):
        """Download a file from URL."""
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests required for downloading models")

        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    def _verify_checksum(self, file_path: Path, expected_sha256: str):
        """Verify file checksum."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)

        computed_hash = sha256_hash.hexdigest()
        if computed_hash != expected_sha256:
            raise ValueError(f"Checksum mismatch for {file_path}")


def load_pretrained_model(
    name: str, cache_dir: str = "~/.cache/eeg_intent_models", device: str = "auto"
) -> Dict[str, Any]:
    """Load a pretrained model from the registry."""

    registry = ModelRegistry()
    model_files = registry.download_model(name, cache_dir)

    # Load model (implementation would depend on your model loading system)
    print(f"Loading model from {model_files['checkpoint']}")

    # This is a placeholder - actual implementation would load the model
    return {
        "model": None,  # Loaded model object
        "config": model_files["config"],
        "model_info": model_files["model_info"],
    }


def create_model_card(model_info: ModelInfo, output_path: str):
    """Create a model card with detailed information."""

    card_content = f"""# {model_info.name}

## Model Description

**Architecture**: {model_info.architecture}
**Dataset**: {model_info.dataset}
**Method**: {model_info.method}
**Version**: {model_info.version}

{model_info.description}

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | {model_info.accuracy:.3f} |
| Kappa | {model_info.kappa:.3f} |
| F1 Score | {model_info.f1_score:.3f} |
| Cross-Subject Accuracy | {model_info.cross_subject_accuracy or 'N/A'} |

## Technical Details

- **Input Shape**: {model_info.input_shape}
- **Number of Classes**: {model_info.n_classes}
- **Number of Channels**: {model_info.n_channels}
- **Sampling Rate**: {model_info.sampling_rate} Hz
- **Model Size**: {model_info.model_size_mb} MB

## Usage

```python
from eegintent.hosting import load_pretrained_model

# Load the model
model_data = load_pretrained_model("{model_info.name}")
model = model_data["model"]
config = model_data["config"]

# Use for inference
predictions = model(eeg_data)
```

## Training Data

{chr(10).join(f"- {source}" for source in model_info.training_data_sources)}

## License and Usage

**License**: {model_info.license}

{model_info.usage_restrictions or "No specific usage restrictions."}

## Citation

```bibtex
{model_info.citation}
```

## Authors and Contact

**Authors**: {", ".join(model_info.authors)}
**Contact**: {model_info.contact_email}
**Repository**: {model_info.repository_url}

## Model Files

- **Checkpoint**: [Download]({model_info.checkpoint_url})
- **Config**: [Download]({model_info.config_url})

**Checksums**:
- Checkpoint SHA256: `{model_info.checkpoint_sha256}`
- Config SHA256: `{model_info.config_sha256}`

---

*Generated on {model_info.created_date}*
"""

    with open(output_path, "w") as f:
        f.write(card_content)

    print(f"Model card saved to {output_path}")


def hosting_strategy_summary():
    """Print hosting strategy summary."""

    print("🏠 EEG-to-Intent Model Hosting Strategy")
    print("=" * 50)

    print("\n📦 Hosting Platforms:")
    print("1. GitHub Releases - Free, public, version controlled")
    print("2. Hugging Face Hub - ML-focused, good discoverability")
    print("3. Zenodo - Academic, DOI assignment, long-term preservation")
    print("4. AWS S3/Google Cloud - Scalable, paid, custom domains")

    print("\n📋 File Structure:")
    print("models/")
    print("├── registry.json              # Model registry")
    print("├── {dataset}_{model}_{method}/")
    print("│   ├── model.ckpt            # Model checkpoint")
    print("│   ├── config.yaml           # Configuration")
    print("│   ├── model_card.md         # Documentation")
    print("│   └── metrics.json          # Performance metrics")

    print("\n⚖️ License Compliance:")
    print("- MIT License for code and supervised models")
    print("- Clear attribution for datasets used")
    print("- Academic-use restrictions for some SSL models")
    print("- No redistribution of original datasets")
    print("- Checksum verification for integrity")

    print("\n🔍 Model Registry:")
    print("- Centralized JSON registry with metadata")
    print("- Version tracking and change logs")
    print("- Performance benchmarks and comparisons")
    print("- Download statistics and usage tracking")

    print("\n💾 Storage Strategy:")
    print("- Compressed checkpoints (.ckpt files)")
    print("- Separate configuration files")
    print("- Model cards with full documentation")
    print("- Automated CI/CD for model updates")


if __name__ == "__main__":
    # Demonstrate the model registry
    registry = ModelRegistry()

    print("Available Models:")
    for model in registry.list_models():
        print(f"  {model.name} - {model.accuracy:.3f} accuracy")

    # Create example model card
    model_info = registry.get_model("bcic_iv_2a_eegnet_supervised")
    if model_info:
        create_model_card(model_info, "model_card_example.md")

    # Show hosting strategy
    hosting_strategy_summary()
