import json

import typer

from .experiments.ablation import run_experiment
from .utils.logging import setup_logging

app = typer.Typer(help="EEG-to-Intent Toolkit CLI")


@app.callback()
def main(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose logging",
    )
):
    """Initialize logging for commands."""
    setup_logging("DEBUG" if verbose else "INFO")


@app.command()
def prep(
    config: str = typer.Option(
        ...,
        "--config",
        help="Path to data config YAML",
    )
):
    """Prepare datasets per config (fetch, preprocess, cache)."""
    typer.echo(f"Prep: using config {config}")


@app.command()
def train(
    config: str = typer.Option(
        ...,
        "--config",
        help="Path to train config YAML",
    )
):
    """Train a model per config (Lightning)."""
    typer.echo(f"Train: using config {config}")


@app.command(name="eval")
def evaluate(
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint",
        help="Path to model ckpt",
    ),
    protocol: str = typer.Option(
        "within",
        "--protocol",
        help="Eval protocol: within|cross|loso",
    ),
):
    """Evaluate a model checkpoint under a protocol."""
    typer.echo(f"Eval: ckpt={checkpoint} protocol={protocol}")


@app.command()
def pretrain(
    config: str = typer.Option(
        ...,
        "--config",
        help="Path to SSL config YAML",
    )
):
    """Self-supervised pretraining per config."""
    typer.echo(f"Pretrain: using config {config}")


@app.command()
def personalize(
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint",
        help="Pretrained checkpoint",
    ),
    subject: str = typer.Option(
        ...,
        "--subject",
        help="Subject ID for calibration",
    ),
):
    """Few-shot personalization for a subject."""
    typer.echo(f"Personalize: ckpt={checkpoint} subject={subject}")


@app.command()
def realtime(
    checkpoint: str = typer.Option(
        ...,
        "--checkpoint",
        help="ONNX or TorchScript model",
    ),
    lsl_stream: str | None = typer.Option(
        "EEG",
        "--lsl-stream",
        help="LSL stream name",
    ),
):
    """Run real-time inference demo with LSL input and dashboard."""
    typer.echo(f"Realtime: ckpt={checkpoint} stream={lsl_stream}")


@app.command()
def ablate(
    n_channels: int = typer.Option(8, help="Number of EEG channels"),
    n_classes: int = typer.Option(2, help="Number of classes"),
    max_epochs: int = typer.Option(1, help="Max epochs"),
    batch_size: int = typer.Option(8, help="Batch size"),
    config_json: str | None = typer.Option(
        None,
        "--config-json",
        help="Extra config JSON. Example: '{\"augment_noise_std\":0.1}'",
    ),
):
    """Run a quick ablation. Advanced toggles via --config-json."""
    cfg = {
        "n_channels": n_channels,
        "n_classes": n_classes,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
    }
    if config_json:
        try:
            extra = json.loads(config_json)
            if isinstance(extra, dict):
                cfg.update(extra)
        except json.JSONDecodeError:
            typer.echo("Invalid --config-json; must be valid JSON object.")
    result = run_experiment(cfg)
    typer.echo(f"Ablation result: {result}")


if __name__ == "__main__":  # pragma: no cover
    app()
