"""Smoke tests for the user-facing Typer boundary."""

from __future__ import annotations

from typer.testing import CliRunner

from coding_agent import __version__
from coding_agent.cli import app

runner = CliRunner()


def test_version_option_prints_the_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_demo_exposes_the_complete_offline_agent_loop() -> None:
    result = runner.invoke(app, ["demo"])

    assert result.exit_code == 0
    assert "Running tool: echo" in result.stdout
    assert "Tool succeeded: echo" in result.stdout
    assert "State: acting -> completed_unverified" in result.stdout
    assert "Run ended with an unverified final response" in result.stdout
    assert "Offline loop completed successfully." in result.stdout
