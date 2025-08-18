"""
Weights & Biases configuration and leaderboard setup.

This module provides utilities for setting up W&B experiments,
sweeps, and maintaining a public leaderboard for EEG-to-Intent benchmarks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


# Standard project configuration
WANDB_PROJECT = "eeg-to-intent-benchmarks"
WANDB_ENTITY = "eeg-research"  # Replace with your W&B entity

# Standard tags for organization
DATASET_TAGS = ["bcic_iv_2a", "bcic_iv_2b", "physionet_mi", "cho2017", "lee2019", "zhou2016"]

MODEL_TAGS = [
    "eegnet",
    "shallow_convnet",
    "deep_convnet",
    "eegconformer",
    "eegchannelnet",
    "fbcspnet",
]

PROTOCOL_TAGS = ["within_subject", "cross_subject", "loso", "few_shot", "zero_shot"]

METHOD_TAGS = [
    "supervised",
    "ssl_pretrained",
    "personalized",
    "domain_adapted",
    "transfer_learning",
]


def get_wandb_config() -> Dict[str, Any]:
    """Get standard W&B configuration."""
    return {
        "project": WANDB_PROJECT,
        "entity": WANDB_ENTITY,
        "save_code": True,
        "notes": "EEG-to-Intent Toolkit Benchmark",
        "job_type": "benchmark",
    }


def create_sweep_config(
    dataset: str = "bcic_iv_2a",
    model: str = "eegnet",
    method: str = "supervised",
) -> Dict[str, Any]:
    """Create W&B sweep configuration for hyperparameter optimization."""

    # Base configuration
    config = {
        "name": f"{dataset}_{model}_{method}_sweep",
        "method": "bayes",  # bayesian optimization
        "metric": {"name": "test_accuracy", "goal": "maximize"},
        "early_terminate": {"type": "hyperband", "min_iter": 10, "eta": 2},
        "parameters": {
            # Model hyperparameters
            "learning_rate": {"distribution": "log_uniform", "min": 1e-5, "max": 1e-2},
            "weight_decay": {"distribution": "log_uniform", "min": 1e-6, "max": 1e-2},
            "batch_size": {"values": [16, 32, 64, 128]},
            "dropout_rate": {"distribution": "uniform", "min": 0.1, "max": 0.8},
            # Training hyperparameters
            "max_epochs": {"value": 200},
            "patience": {"values": [10, 15, 20]},
            "lr_scheduler": {"values": ["cosine", "step", "exponential", "none"]},
            # Data augmentation
            "augmentation_strength": {"distribution": "uniform", "min": 0.0, "max": 0.5},
            "noise_std": {"distribution": "uniform", "min": 0.0, "max": 0.1},
        },
    }

    # Model-specific parameters
    if model == "eegnet":
        config["parameters"].update(
            {
                "F1": {"values": [4, 8, 16]},
                "D": {"values": [2, 4, 8]},
                "F2": {"values": [8, 16, 32]},
                "kernel_length": {"values": [32, 64, 128]},
            }
        )
    elif model in ["shallow_convnet", "deep_convnet"]:
        config["parameters"].update(
            {
                "n_filters_time": {"values": [25, 40, 60]},
                "filter_time_length": {"values": [10, 25, 40]},
                "n_filters_spat": {"values": [25, 40, 60]},
                "pool_time_length": {"values": [30, 50, 75]},
            }
        )

    # Method-specific parameters
    if method == "ssl_pretrained":
        config["parameters"].update(
            {
                "ssl_learning_rate": {"distribution": "log_uniform", "min": 1e-5, "max": 1e-2},
                "freeze_encoder": {"values": [True, False]},
                "fine_tune_epochs": {"values": [10, 25, 50]},
            }
        )
    elif method == "personalized":
        config["parameters"].update(
            {
                "personalization_method": {"values": ["film", "prototypical", "riemannian"]},
                "few_shot_samples": {"values": [5, 10, 20]},
                "adaptation_lr": {"distribution": "log_uniform", "min": 1e-4, "max": 1e-2},
            }
        )

    return config


def setup_wandb_run(
    config: Dict[str, Any],
    tags: Optional[List[str]] = None,
    group: Optional[str] = None,
    job_type: str = "benchmark",
) -> Any:
    """Set up a W&B run with standard configuration."""

    if not WANDB_AVAILABLE:
        raise RuntimeError("wandb not available. Install with: pip install wandb")

    # Merge with standard config
    wandb_config = get_wandb_config()
    wandb_config.update(
        {
            "tags": tags or [],
            "group": group,
            "job_type": job_type,
        }
    )

    # Initialize run
    run = wandb.init(
        **wandb_config,
        config=config,
    )

    return run


def log_benchmark_results(
    results: Dict[str, Any],
    model_path: Optional[str] = None,
    config_path: Optional[str] = None,
) -> None:
    """Log benchmark results to W&B."""

    if not WANDB_AVAILABLE or not wandb.run:
        return

    # Log metrics
    wandb.log(results)

    # Log artifacts
    if model_path and Path(model_path).exists():
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

    if config_path and Path(config_path).exists():
        artifact = wandb.Artifact("config", type="config")
        artifact.add_file(config_path)
        wandb.log_artifact(artifact)


def create_leaderboard_table(
    results_list: List[Dict[str, Any]],
    metrics: List[str] = ["accuracy", "kappa", "f1_score"],
) -> Any:
    """Create a W&B table for leaderboard visualization."""

    if not WANDB_AVAILABLE:
        return None

    # Define columns
    columns = (
        ["experiment_name", "dataset", "model", "protocol", "method", "seed"]
        + metrics
        + ["timestamp", "run_id"]
    )

    # Create table data
    data = []
    for result in results_list:
        row = []
        for col in columns:
            if col in result:
                row.append(result[col])
            else:
                row.append(None)
        data.append(row)

    # Create W&B table
    table = wandb.Table(columns=columns, data=data)

    return table


def get_sweep_configs() -> Dict[str, Dict[str, Any]]:
    """Get predefined sweep configurations for different scenarios."""

    sweeps = {}

    # Basic supervised learning sweeps
    for dataset in ["bcic_iv_2a", "physionet_mi"]:
        for model in ["eegnet", "shallow_convnet"]:
            sweep_name = f"{dataset}_{model}_supervised"
            sweeps[sweep_name] = create_sweep_config(
                dataset=dataset, model=model, method="supervised"
            )

    # SSL pretraining sweeps
    for dataset in ["bcic_iv_2a"]:
        for model in ["eegnet"]:
            sweep_name = f"{dataset}_{model}_ssl"
            sweeps[sweep_name] = create_sweep_config(
                dataset=dataset, model=model, method="ssl_pretrained"
            )

    # Personalization sweeps
    for dataset in ["bcic_iv_2a"]:
        for model in ["eegnet"]:
            sweep_name = f"{dataset}_{model}_personalized"
            sweeps[sweep_name] = create_sweep_config(
                dataset=dataset, model=model, method="personalized"
            )

    return sweeps


def save_sweep_configs(output_dir: str = "configs/sweeps") -> None:
    """Save all sweep configurations to YAML files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sweeps = get_sweep_configs()

    for name, config in sweeps.items():
        config_file = output_path / f"{name}.yaml"

        # Save as YAML
        import yaml

        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        print(f"Saved sweep config: {config_file}")


def create_benchmark_dashboard():
    """Create a W&B dashboard for benchmark results."""

    if not WANDB_AVAILABLE:
        print("W&B not available - cannot create dashboard")
        return

    # This would create a custom W&B dashboard
    # In practice, you would set this up through the W&B web interface
    print("Dashboard template:")
    print("1. Go to your W&B project page")
    print("2. Create a new report")
    print("3. Add the following panels:")
    print("   - Leaderboard table (accuracy by dataset/model)")
    print("   - Performance trends over time")
    print("   - Hyperparameter correlation plots")
    print("   - Model comparison charts")
    print("   - Cross-subject generalization analysis")


# Artifact naming conventions
def get_artifact_name(
    artifact_type: str,
    dataset: str,
    model: str,
    method: str = "supervised",
    version: str = "latest",
) -> str:
    """Get standardized artifact name."""
    return f"{dataset}-{model}-{method}-{artifact_type}:{version}"


def get_model_registry_name(
    dataset: str,
    model: str,
    method: str = "supervised",
) -> str:
    """Get model registry name."""
    return f"eeg-models/{dataset}_{model}_{method}"


# Example usage functions
def example_wandb_setup():
    """Example of setting up W&B for a benchmark run."""

    if not WANDB_AVAILABLE:
        print("W&B not available")
        return

    # Configuration
    config = {
        "dataset": "bcic_iv_2a",
        "model": "eegnet",
        "learning_rate": 1e-3,
        "batch_size": 32,
        "epochs": 100,
    }

    # Tags for organization
    tags = ["bcic_iv_2a", "eegnet", "supervised", "within_subject"]

    # Set up run
    run = setup_wandb_run(
        config=config, tags=tags, group="baseline_experiments", job_type="benchmark"
    )

    # Log some dummy results
    results = {
        "test_accuracy": 0.85,
        "test_kappa": 0.78,
        "test_f1": 0.84,
        "train_time": 120.5,
        "inference_time": 0.05,
    }

    log_benchmark_results(results)

    # Finish run
    wandb.finish()

    print("Example W&B run completed")


if __name__ == "__main__":
    # Save sweep configurations
    save_sweep_configs()

    # Show dashboard instructions
    create_benchmark_dashboard()

    # Example setup
    example_wandb_setup()
