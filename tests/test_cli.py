"""Smoke tests for the user-facing Typer boundary."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from coding_agent import __version__
from coding_agent.cli import app
from coding_agent.model import ScriptedModel
from coding_agent.models import ModelResponse

runner = CliRunner()


def test_version_option_prints_the_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_demo_exposes_a_real_read_edit_verify_loop() -> None:
    result = runner.invoke(app, ["demo"])

    assert result.exit_code == 0
    assert result.stdout.count("Running tool: update_plan") == 5
    assert result.stdout.count("Running tool: run_command") == 2
    assert "Running tool: list_files" in result.stdout
    assert "Running tool: search_text" in result.stdout
    assert "Running tool: read_file" in result.stdout
    assert "Running tool: replace_text" in result.stdout
    assert (
        result.stdout.index("Running tool: run_command")
        < result.stdout.index("Running tool: list_files")
        < result.stdout.index("Running tool: search_text")
        < result.stdout.index("Running tool: read_file")
        < result.stdout.index("Running tool: replace_text")
        < result.stdout.rindex("Running tool: run_command")
    )
    assert "Plan revision 1" in result.stdout
    assert "Plan revision 5" in result.stdout
    assert "Command exited 1" in result.stdout
    assert "Command exited 0" in result.stdout
    assert "Found 1 matches and showed 1" in result.stdout
    assert "Read src/pricing.py lines 1-3 of 3" in result.stdout
    assert "Updated src/pricing.py (+1/-1" in result.stdout
    assert result.stdout.count("Running tool: read_file") == 1
    assert "Recorded test evidence: failed" in result.stdout
    assert "Recorded test evidence: passed" in result.stdout
    assert "State: acting -> verifying" in result.stdout
    assert "State: verifying -> completed" in result.stdout
    assert "Run ended with current verification evidence" in result.stdout
    assert "calculate_total now applies the discount exactly" in result.stdout
    assert "post-change pytest run passed" in result.stdout
    assert "VERIFIED repair report" in result.stdout


def test_demo_returns_failure_when_the_runtime_cannot_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "coding_agent.cli._repository_demo_model",
        lambda: ScriptedModel([ModelResponse(content="No external verification was run.")]),
    )

    result = runner.invoke(app, ["demo"])

    assert result.exit_code == 1
    assert "UNVERIFIED repair report" in result.stdout
