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


def test_demo_exposes_real_read_only_repository_reconnaissance() -> None:
    result = runner.invoke(app, ["demo"])

    assert result.exit_code == 0
    assert "Running tool: list_files" in result.stdout
    assert "Running tool: search_text" in result.stdout
    assert "Running tool: read_file" in result.stdout
    assert (
        result.stdout.index("Running tool: list_files")
        < result.stdout.index("Running tool: search_text")
        < result.stdout.index("Running tool: read_file")
    )
    assert "Listed 4 of 4 discovered entries" in result.stdout
    assert "Found 1 matches and showed 1" in result.stdout
    assert "Read src/pricing.py lines 1-3 of 3" in result.stdout
    assert "State: acting -> completed_unverified" in result.stdout
    assert "Run ended with an unverified final response" in result.stdout
    assert "calculate_total subtracts the discount" in result.stdout
    assert "twice. Repository reconnaissance completed" in result.stdout
    assert "without modifying the workspace" in result.stdout
