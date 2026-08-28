"""Tests for private state path selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from coding_agent.state import default_state_paths


def test_explicit_state_directory_override_is_partitioned(tmp_path: Path) -> None:
    state = default_state_paths({"CODING_AGENT_STATE_DIR": str(tmp_path)})

    assert state.root == tmp_path.resolve()
    assert state.sessions == tmp_path.resolve() / "sessions"
    assert state.traces == tmp_path.resolve() / "traces"


def test_relative_state_directory_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="absolute"):
        default_state_paths({"CODING_AGENT_STATE_DIR": "relative/state"})


def test_default_state_directory_is_absolute_and_application_scoped() -> None:
    state = default_state_paths({})

    assert state.root.is_absolute()
    assert state.root.name == "coding-agent"
