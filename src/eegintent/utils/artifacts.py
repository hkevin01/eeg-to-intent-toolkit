"""Artifact registry system for reproducible experiments."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch


class ArtifactRegistry:
    """Registry for experiment artifacts and reproducibility."""

    def __init__(self, base_dir: str | Path = "artifacts"):
        """Initialize artifact registry.

        Args:
            base_dir: Base directory for storing artifacts
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.base_dir / "checkpoints").mkdir(exist_ok=True)
        (self.base_dir / "configs").mkdir(exist_ok=True)
        (self.base_dir / "logs").mkdir(exist_ok=True)
        (self.base_dir / "metadata").mkdir(exist_ok=True)

    def create_experiment_id(self, prefix: str = "exp") -> str:
        """Create unique experiment ID."""
        timestamp = int(time.time())
        return f"{prefix}_{timestamp}"

    def save_config(self, config: dict[str, Any], experiment_id: str) -> Path:
        """Save experiment configuration.

        Args:
            config: Configuration dictionary
            experiment_id: Unique experiment identifier

        Returns:
            Path to saved config file
        """
        config_path = self.base_dir / "configs" / f"{experiment_id}.json"

        # Add metadata
        config_with_meta = {
            "experiment_id": experiment_id,
            "timestamp": time.time(),
            "config": config,
        }

        with config_path.open("w") as f:
            json.dump(config_with_meta, f, indent=2, default=str)

        return config_path

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        experiment_id: str,
        epoch: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save model checkpoint.

        Args:
            model: PyTorch model to save
            experiment_id: Unique experiment identifier
            epoch: Training epoch (optional)
            metadata: Additional metadata

        Returns:
            Path to saved checkpoint
        """
        if epoch is not None:
            filename = f"{experiment_id}_epoch_{epoch}.ckpt"
        else:
            filename = f"{experiment_id}_final.ckpt"

        checkpoint_path = self.base_dir / "checkpoints" / filename

        # Prepare checkpoint data
        checkpoint_data = {
            "experiment_id": experiment_id,
            "timestamp": time.time(),
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
        }

        if epoch is not None:
            checkpoint_data["epoch"] = epoch

        if metadata:
            checkpoint_data["metadata"] = metadata

        torch.save(checkpoint_data, checkpoint_path)
        return checkpoint_path

    def save_metrics(
        self,
        metrics: dict[str, Any],
        experiment_id: str,
        step: int | None = None,
    ) -> Path:
        """Save experiment metrics.

        Args:
            metrics: Metrics dictionary
            experiment_id: Unique experiment identifier
            step: Training step (optional)

        Returns:
            Path to saved metrics file
        """
        if step is not None:
            filename = f"{experiment_id}_step_{step}.json"
        else:
            filename = f"{experiment_id}_final_metrics.json"

        metrics_path = self.base_dir / "logs" / filename

        # Add metadata
        metrics_with_meta = {
            "experiment_id": experiment_id,
            "timestamp": time.time(),
            "metrics": metrics,
        }

        if step is not None:
            metrics_with_meta["step"] = step

        with metrics_path.open("w") as f:
            json.dump(metrics_with_meta, f, indent=2, default=str)

        return metrics_path

    def save_experiment_summary(
        self,
        experiment_id: str,
        config: dict[str, Any],
        final_metrics: dict[str, Any],
        notes: str = "",
    ) -> Path:
        """Save complete experiment summary.

        Args:
            experiment_id: Unique experiment identifier
            config: Experiment configuration
            final_metrics: Final experiment metrics
            notes: Additional notes about the experiment

        Returns:
            Path to saved summary file
        """
        summary_path = self.base_dir / "metadata" / f"{experiment_id}_summary.json"

        summary = {
            "experiment_id": experiment_id,
            "timestamp": time.time(),
            "config": config,
            "final_metrics": final_metrics,
            "notes": notes,
            "artifacts": {
                "config_file": f"configs/{experiment_id}.json",
                "checkpoint_files": list(
                    (self.base_dir / "checkpoints").glob(f"{experiment_id}*.ckpt")
                ),
                "log_files": list((self.base_dir / "logs").glob(f"{experiment_id}*.json")),
            },
        }

        # Convert Path objects to strings for JSON serialization
        summary["artifacts"]["checkpoint_files"] = [
            str(p.relative_to(self.base_dir)) for p in summary["artifacts"]["checkpoint_files"]
        ]
        summary["artifacts"]["log_files"] = [
            str(p.relative_to(self.base_dir)) for p in summary["artifacts"]["log_files"]
        ]

        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary_path

    def load_config(self, experiment_id: str) -> dict[str, Any]:
        """Load experiment configuration.

        Args:
            experiment_id: Unique experiment identifier

        Returns:
            Configuration dictionary
        """
        config_path = self.base_dir / "configs" / f"{experiment_id}.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with config_path.open() as f:
            data = json.load(f)

        return data["config"]

    def load_checkpoint(
        self,
        experiment_id: str,
        epoch: int | None = None,
    ) -> dict[str, Any]:
        """Load model checkpoint.

        Args:
            experiment_id: Unique experiment identifier
            epoch: Training epoch (None for final checkpoint)

        Returns:
            Checkpoint data dictionary
        """
        if epoch is not None:
            filename = f"{experiment_id}_epoch_{epoch}.ckpt"
        else:
            filename = f"{experiment_id}_final.ckpt"

        checkpoint_path = self.base_dir / "checkpoints" / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        return torch.load(checkpoint_path, map_location="cpu")

    def list_experiments(self) -> list[str]:
        """List all experiment IDs.

        Returns:
            List of experiment IDs
        """
        config_files = (self.base_dir / "configs").glob("*.json")
        experiment_ids = [f.stem for f in config_files]
        return sorted(experiment_ids)

    def get_experiment_summary(self, experiment_id: str) -> dict[str, Any]:
        """Get experiment summary.

        Args:
            experiment_id: Unique experiment identifier

        Returns:
            Experiment summary dictionary
        """
        summary_path = self.base_dir / "metadata" / f"{experiment_id}_summary.json"

        if summary_path.exists():
            with summary_path.open() as f:
                return json.load(f)
        else:
            # Generate summary from available artifacts
            return {
                "experiment_id": experiment_id,
                "config": self.load_config(experiment_id),
                "artifacts": self._scan_artifacts(experiment_id),
            }

    def _scan_artifacts(self, experiment_id: str) -> dict[str, list[str]]:
        """Scan for artifacts of an experiment.

        Args:
            experiment_id: Unique experiment identifier

        Returns:
            Dictionary of artifact file lists
        """
        artifacts = {
            "checkpoints": [],
            "logs": [],
        }

        # Scan checkpoints
        for ckpt_file in (self.base_dir / "checkpoints").glob(f"{experiment_id}*.ckpt"):
            artifacts["checkpoints"].append(str(ckpt_file.relative_to(self.base_dir)))

        # Scan logs
        for log_file in (self.base_dir / "logs").glob(f"{experiment_id}*.json"):
            artifacts["logs"].append(str(log_file.relative_to(self.base_dir)))

        return artifacts

    def cleanup_experiment(self, experiment_id: str) -> None:
        """Remove all artifacts for an experiment.

        Args:
            experiment_id: Unique experiment identifier
        """
        # Remove config
        config_path = self.base_dir / "configs" / f"{experiment_id}.json"
        if config_path.exists():
            config_path.unlink()

        # Remove checkpoints
        for ckpt_file in (self.base_dir / "checkpoints").glob(f"{experiment_id}*.ckpt"):
            ckpt_file.unlink()

        # Remove logs
        for log_file in (self.base_dir / "logs").glob(f"{experiment_id}*.json"):
            log_file.unlink()

        # Remove summary
        summary_path = self.base_dir / "metadata" / f"{experiment_id}_summary.json"
        if summary_path.exists():
            summary_path.unlink()

    def archive_experiment(self, experiment_id: str, archive_path: str | Path) -> Path:
        """Archive all artifacts for an experiment.

        Args:
            experiment_id: Unique experiment identifier
            archive_path: Path for the archive file (without extension)

        Returns:
            Path to created archive
        """
        archive_path = Path(archive_path)
        temp_dir = self.base_dir / f"temp_{experiment_id}"

        try:
            # Create temporary directory with experiment artifacts
            temp_dir.mkdir(exist_ok=True)

            # Copy config
            config_path = self.base_dir / "configs" / f"{experiment_id}.json"
            if config_path.exists():
                shutil.copy2(config_path, temp_dir)

            # Copy checkpoints
            ckpt_dir = temp_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            for ckpt_file in (self.base_dir / "checkpoints").glob(f"{experiment_id}*.ckpt"):
                shutil.copy2(ckpt_file, ckpt_dir)

            # Copy logs
            logs_dir = temp_dir / "logs"
            logs_dir.mkdir(exist_ok=True)
            for log_file in (self.base_dir / "logs").glob(f"{experiment_id}*.json"):
                shutil.copy2(log_file, logs_dir)

            # Copy summary
            summary_path = self.base_dir / "metadata" / f"{experiment_id}_summary.json"
            if summary_path.exists():
                shutil.copy2(summary_path, temp_dir)

            # Create archive
            archive_file = shutil.make_archive(str(archive_path), "zip", str(temp_dir))

            return Path(archive_file)

        finally:
            # Clean up temporary directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)


# Global registry instance
_global_registry: ArtifactRegistry | None = None


def get_global_registry() -> ArtifactRegistry:
    """Get the global artifact registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ArtifactRegistry()
    return _global_registry


def set_global_registry(registry: ArtifactRegistry) -> None:
    """Set the global artifact registry instance."""
    global _global_registry
    _global_registry = registry
