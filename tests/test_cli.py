from typer.testing import CliRunner

from eegintent.cli import app


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "EEG-to-Intent Toolkit CLI" in result.stdout
