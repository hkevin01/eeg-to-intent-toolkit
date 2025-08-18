from typing import Optional

import typer

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


if __name__ == "__main__":  # pragma: no cover
    app()
