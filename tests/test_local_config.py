"""Tests for the explicit, untracked local configuration boundary."""

from pathlib import Path

import pytest

from coding_agent.errors import CodedError
from coding_agent.local_config import LOCAL_CONFIG_FILENAME, load_local_environment


def test_loads_only_supported_keys_without_overriding_process_values(tmp_path: Path) -> None:
    tmp_path.joinpath(LOCAL_CONFIG_FILENAME).write_text(
        "\n".join(
            (
                "OPENAI_API_KEY=file-key",
                "OPENAI_BASE_URL=https://config.example/v1",
                "CODING_AGENT_MODEL=file-model",
                "CODING_AGENT_REASONING_EFFORT=none",
                "PATH=must-not-change",
                "UNRELATED_SETTING=must-not-load",
            )
        ),
        encoding="utf-8",
    )
    environment = {
        "OPENAI_API_KEY": "ambient-key",
        "PATH": "ambient-path",
    }

    loaded = load_local_environment(directory=tmp_path, environment=environment)

    assert loaded is True
    assert environment == {
        "OPENAI_API_KEY": "ambient-key",
        "OPENAI_BASE_URL": "https://config.example/v1",
        "CODING_AGENT_MODEL": "file-model",
        "CODING_AGENT_REASONING_EFFORT": "none",
        "PATH": "ambient-path",
    }


def test_missing_file_does_not_search_parent_directories(tmp_path: Path) -> None:
    tmp_path.joinpath(LOCAL_CONFIG_FILENAME).write_text(
        "OPENAI_API_KEY=parent-key\n",
        encoding="utf-8",
    )
    child = tmp_path / "child"
    child.mkdir()
    environment: dict[str, str] = {}

    loaded = load_local_environment(directory=child, environment=environment)

    assert loaded is False
    assert environment == {}


def test_values_are_not_interpolated_from_the_ambient_environment(tmp_path: Path) -> None:
    tmp_path.joinpath(LOCAL_CONFIG_FILENAME).write_text(
        "OPENAI_API_KEY=${AMBIENT_SECRET}\n",
        encoding="utf-8",
    )
    environment = {"AMBIENT_SECRET": "must-not-expand"}

    load_local_environment(directory=tmp_path, environment=environment)

    assert environment["OPENAI_API_KEY"] == "${AMBIENT_SECRET}"


def test_unreadable_utf8_is_a_sanitized_configuration_error(tmp_path: Path) -> None:
    tmp_path.joinpath(LOCAL_CONFIG_FILENAME).write_bytes(b"OPENAI_API_KEY=\xff\n")

    with pytest.raises(CodedError) as error:
        load_local_environment(directory=tmp_path, environment={})

    assert error.value.code == "local_config_unreadable"
    assert str(error.value) == "Could not read .env.local as a UTF-8 configuration file."
