"""
Benchmarking configuration and utilities.

This module provides standardized configurations and utilities for
reproducible benchmarking across datasets and models.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""

    # Experiment identification
    experiment_name: str
    dataset_name: str
    model_name: str

    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = True

    # Data configuration
    subject_ids: Optional[List[int]] = None
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Model configuration
    model_config: Dict[str, Any] = field(default_factory=dict)

    # Training configuration
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # SSL configuration (if applicable)
    use_ssl_pretrained: bool = False
    ssl_checkpoint_path: Optional[str] = None
    freeze_encoder: bool = False

    # Personalization configuration
    use_personalization: bool = False
    personalization_method: str = "film"  # film, prototypical, riemannian
    few_shot_samples: int = 5

    # Evaluation protocol
    # Options: within_subject, cross_subject, loso
    evaluation_protocol: str = "within_subject"
    cross_validation_folds: int = 5

    # Logging and artifacts
    log_level: str = "INFO"
    save_predictions: bool = True
    save_model: bool = True
    wandb_project: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)

    # Compute configuration
    accelerator: str = "auto"
    devices: Union[int, str] = "auto"
    precision: str = "16-mixed"

    # Output paths
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set environment variables for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Enable deterministic operations
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def setup_logging(config: BenchmarkConfig) -> logging.Logger:
    """Set up logging configuration."""
    log_level = getattr(logging, config.log_level.upper())

    # Create log directory
    log_dir = Path(config.output_dir) / config.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_file = log_dir / f"{config.experiment_name}.log"

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured. Log file: {log_file}")

    return logger


def create_experiment_name(
    dataset: str,
    model: str,
    protocol: str,
    ssl: bool = False,
    personalization: bool = False,
    seed: int = 42,
) -> str:
    """Create standardized experiment name."""
    components = [dataset, model, protocol]

    if ssl:
        components.append("ssl")

    if personalization:
        components.append("personal")

    components.append(f"seed{seed}")

    return "_".join(components)


def load_config(config_path: Union[str, Path]) -> BenchmarkConfig:
    """Load configuration from YAML file."""
    config_dict = OmegaConf.load(config_path)

    # Convert to BenchmarkConfig
    return BenchmarkConfig(**config_dict)


def save_config(config: BenchmarkConfig, output_path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    config_dict = OmegaConf.structured(config)
    OmegaConf.save(config_dict, output_path)


def get_default_configs() -> Dict[str, BenchmarkConfig]:
    """Get default configurations for common benchmark scenarios."""
    configs = {}

    # Base configuration
    base_config = BenchmarkConfig(
        experiment_name="base",
        dataset_name="bcic_iv_2a",
        model_name="eegnet",
    )

    # Within-subject baseline
    configs["within_subject_baseline"] = BenchmarkConfig(
        **base_config.__dict__,
        experiment_name="within_subject_baseline",
        evaluation_protocol="within_subject",
    )

    # Cross-subject baseline
    configs["cross_subject_baseline"] = BenchmarkConfig(
        **base_config.__dict__,
        experiment_name="cross_subject_baseline",
        evaluation_protocol="cross_subject",
    )

    # SSL pretraining
    configs["ssl_pretrained"] = BenchmarkConfig(
        **base_config.__dict__,
        experiment_name="ssl_pretrained",
        use_ssl_pretrained=True,
        evaluation_protocol="cross_subject",
    )

    # Personalization
    configs["personalization"] = BenchmarkConfig(
        **base_config.__dict__,
        experiment_name="personalization",
        use_personalization=True,
        personalization_method="film",
        few_shot_samples=5,
        evaluation_protocol="cross_subject",
    )

    # SSL + Personalization
    configs["ssl_personalization"] = BenchmarkConfig(
        **base_config.__dict__,
        experiment_name="ssl_personalization",
        use_ssl_pretrained=True,
        use_personalization=True,
        personalization_method="prototypical",
        few_shot_samples=5,
        evaluation_protocol="cross_subject",
    )

    return configs


class BenchmarkResults:
    """Container for benchmark results."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.metrics = {}
        self.predictions = {}
        self.model_path = None
        self.artifacts = {}

    def add_metric(self, name: str, value: float, subject_id: Optional[int] = None):
        """Add a metric value."""
        if subject_id is not None:
            if name not in self.metrics:
                self.metrics[name] = {}
            self.metrics[name][subject_id] = value
        else:
            self.metrics[name] = value

    def add_predictions(
        self, predictions: np.ndarray, targets: np.ndarray, subject_id: Optional[int] = None
    ):
        """Add predictions and targets."""
        key = f"subject_{subject_id}" if subject_id else "global"
        self.predictions[key] = {
            "predictions": predictions,
            "targets": targets,
        }

    def add_artifact(self, name: str, path: Union[str, Path]):
        """Add artifact path."""
        self.artifacts[name] = str(path)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of results."""
        summary = {
            "experiment_name": self.config.experiment_name,
            "dataset": self.config.dataset_name,
            "model": self.config.model_name,
            "protocol": self.config.evaluation_protocol,
            "seed": self.config.seed,
        }

        # Add averaged metrics
        for metric_name, values in self.metrics.items():
            if isinstance(values, dict):
                # Subject-wise metrics
                summary[f"{metric_name}_mean"] = np.mean(list(values.values()))
                summary[f"{metric_name}_std"] = np.std(list(values.values()))
                summary[f"{metric_name}_subjects"] = values
            else:
                # Global metrics
                summary[metric_name] = values

        return summary

    def save(self, output_path: Union[str, Path]):
        """Save results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save summary as JSON
        summary = self.get_summary()
        with open(output_path.with_suffix(".json"), "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Save detailed predictions separately
        predictions_data = {}
        for key, pred_data in self.predictions.items():
            predictions_data[f"{key}_predictions"] = pred_data["predictions"]
            predictions_data[f"{key}_targets"] = pred_data["targets"]

        if predictions_data:
            np.savez_compressed(output_path.with_suffix(".npz"), **predictions_data)

        logger.info(f"Results saved to {output_path}")


def create_wandb_config(config: BenchmarkConfig) -> Dict[str, Any]:
    """Create W&B configuration from benchmark config."""
    wandb_config = {
        # Experiment metadata
        "experiment_name": config.experiment_name,
        "dataset": config.dataset_name,
        "model": config.model_name,
        "evaluation_protocol": config.evaluation_protocol,
        # Training parameters
        "max_epochs": config.max_epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        # Model configuration
        **config.model_config,
        # SSL and personalization flags
        "use_ssl_pretrained": config.use_ssl_pretrained,
        "use_personalization": config.use_personalization,
        "personalization_method": config.personalization_method,
        # Reproducibility
        "seed": config.seed,
        "deterministic": config.deterministic,
    }

    return wandb_config
