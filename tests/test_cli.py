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


def test_demo_exposes_a_real_read_edit_verify_loop() -> None:
    result = runner.invoke(app, ["demo"])

    assert result.exit_code == 0
    assert "Running tool: list_files" in result.stdout
    assert "Running tool: search_text" in result.stdout
    assert "Running tool: read_file" in result.stdout
    assert "Running tool: replace_text" in result.stdout
    assert (
        result.stdout.index("Running tool: list_files")
        < result.stdout.index("Running tool: search_text")
        < result.stdout.index("Running tool: read_file")
        < result.stdout.index("Running tool: replace_text")
    )
    assert "Listed 4 of 4 discovered entries" in result.stdout
    assert "Found 1 matches and showed 1" in result.stdout
    assert "Read src/pricing.py lines 1-3 of 3" in result.stdout
    assert "Updated src/pricing.py (+1/-1" in result.stdout
    assert result.stdout.count("Running tool: read_file") == 2
    assert "State: acting -> completed_unverified" in result.stdout
    assert "Run ended with an unverified final response" in result.stdout
    assert "calculate_total now applies the discount exactly" in result.stdout
    assert "post-change read matches" in result.stdout
    assert "Repair report · post-read confirmed" in result.stdout
