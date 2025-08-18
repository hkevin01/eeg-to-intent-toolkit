from typing import Optional

import typer

from .experiments.ablation import run_experiment
from .utils.logging import setup_logging

app = typer.Typer(help="EEG-to-Intent Toolkit CLI")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging")):
    """Initialize logging for commands."""
    setup_logging("DEBUG" if verbose else "INFO")


@app.command()
def prep(config: str = typer.Option(..., "--config", help="Path to data config YAML")):
    """Prepare datasets per config (fetch, preprocess, cache)."""
    typer.echo(f"Prep: using config {config}")


@app.command()
def train(config: str = typer.Option(..., "--config", help="Path to train config YAML")):
    """Train a model per config (Lightning)."""
    typer.echo(f"Train: using config {config}")


@app.command()
def eval(
    checkpoint: str = typer.Option(..., "--checkpoint", help="Path to model checkpoint"),
    protocol: str = typer.Option("within", "--protocol", help="Eval protocol: within|cross|loso"),
):
    """Evaluate a model checkpoint under a protocol."""
    typer.echo(f"Eval: ckpt={checkpoint} protocol={protocol}")


@app.command()
def pretrain(config: str = typer.Option(..., "--config", help="Path to SSL config YAML")):
    """Self-supervised pretraining per config."""
    typer.echo(f"Pretrain: using config {config}")


@app.command()
def personalize(
    checkpoint: str = typer.Option(..., "--checkpoint", help="Pretrained checkpoint"),
    subject: str = typer.Option(..., "--subject", help="Subject ID for calibration"),
):
    """Few-shot personalization for a subject."""
    typer.echo(f"Personalize: ckpt={checkpoint} subject={subject}")


@app.command()
def realtime(
    checkpoint: str = typer.Option(..., "--checkpoint", help="ONNX or TorchScript model"),
    lsl_stream: Optional[str] = typer.Option("EEG", "--lsl-stream", help="LSL stream name"),
):
    """Run real-time inference demo with LSL input and dashboard."""
    typer.echo(f"Realtime: ckpt={checkpoint} stream={lsl_stream}")


@app.command()
def ablate(
    n_channels: int = typer.Option(8, help="Number of EEG channels"),
    n_classes: int = typer.Option(2, help="Number of classes"),
    max_epochs: int = typer.Option(1, help="Max epochs"),
    batch_size: int = typer.Option(8, help="Batch size"),
    augment_noise_std: float = typer.Option(0.0, help="Gaussian noise std for augmentation"),
    use_personalization: bool = typer.Option(False, help="Enable FiLM/CBN personalization"),
    personalization_mode: str = typer.Option("film", help="film or cbn"),
    n_subjects: int = typer.Option(4, help="Number of subjects in simulation"),
):
    """Run a quick ablation with augmentation/personalization toggles."""
    cfg = {
        "n_channels": n_channels,
        "n_classes": n_classes,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "augment_noise_std": augment_noise_std,
        "use_personalization": use_personalization,
        "personalization_mode": personalization_mode,
        "n_subjects": n_subjects,
    }
    result = run_experiment(cfg)
    typer.echo(f"Ablation result: {result}")


if __name__ == "__main__":  # pragma: no cover
    app()
