"""Safety and integration tests for explicit workspace directory creation."""

from __future__ import annotations

from pathlib import Path

import pytest

from coding_agent.command import CommandPermissionMode
from coding_agent.models import ToolCall, ToolExecution
from coding_agent.mutation import MutationSession
from coding_agent.runtime import build_runtime
from coding_agent.tools import CreateDirectoryTool, ToolRegistry, WriteFileTool
from coding_agent.workspace import Workspace


def _execute(
    registry: ToolRegistry,
    name: str,
    arguments: dict[str, object],
) -> ToolExecution:
    return registry.execute(ToolCall(id=f"call-{name}", name=name, arguments=arguments))


def test_empty_workspace_can_create_one_package_then_write_its_initializer(
    tmp_path: Path,
) -> None:
    workspace = Workspace(tmp_path)
    registry = ToolRegistry(
        [
            CreateDirectoryTool(workspace),
            WriteFileTool(MutationSession(workspace)),
        ]
    )

    created = _execute(registry, "create_directory", {"path": "pkg"})
    written = _execute(
        registry,
        "write_file",
        {
            "path": "pkg/__init__.py",
            "content": '"""Example package."""\n',
            "expected_sha256": None,
        },
    )

    assert created.ok is True
    assert created.metadata == {
        "path": "pkg",
        "created": True,
        "changed": True,
        "change_kind": "create_directory",
    }
    assert created.control.invalidates_verification is True
    assert created.control.made_progress is True
    assert written.ok is True
    assert tmp_path.joinpath("pkg", "__init__.py").read_text(encoding="utf-8") == (
        '"""Example package."""\n'
    )


def test_create_directory_is_idempotent_for_an_existing_ordinary_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "pkg").mkdir()
    registry = ToolRegistry([CreateDirectoryTool(Workspace(tmp_path))])

    execution = _execute(registry, "create_directory", {"path": "pkg"})

    assert execution.ok is True
    assert execution.metadata["created"] is False
    assert execution.metadata["changed"] is False
    assert execution.metadata["change_kind"] == "noop"
    assert execution.control.invalidates_verification is False
    assert execution.control.made_progress is False
    assert "already exists" in (execution.output or "")


def test_create_directory_never_creates_missing_parent_levels(tmp_path: Path) -> None:
    registry = ToolRegistry([CreateDirectoryTool(Workspace(tmp_path))])

    execution = _execute(registry, "create_directory", {"path": "pkg/internal"})

    assert execution.ok is False
    assert execution.error_code == "not_found"
    assert not (tmp_path / "pkg").exists()


@pytest.mark.parametrize(
    ("path", "error_code"),
    [
        (".", "invalid_path"),
        ("../outside", "invalid_path"),
        (str(Path.cwd().anchor) or "/", "invalid_path"),
        (".git/objects", "path_ignored"),
        (".coding-agent", "path_ignored"),
        (".coding-agent-tmp-user", "path_ignored"),
        (".env", "sensitive_path"),
        ("CON", "invalid_path"),
    ],
)
def test_create_directory_reuses_workspace_mutation_policy(
    tmp_path: Path,
    path: str,
    error_code: str,
) -> None:
    registry = ToolRegistry([CreateDirectoryTool(Workspace(tmp_path))])

    execution = _execute(registry, "create_directory", {"path": path})

    assert execution.ok is False
    assert execution.error_code == error_code


def test_create_directory_applies_directory_gitignore_semantics(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("build/\n", encoding="utf-8")
    registry = ToolRegistry([CreateDirectoryTool(Workspace(tmp_path))])

    execution = _execute(registry, "create_directory", {"path": "build"})

    assert execution.ok is False
    assert execution.error_code == "path_ignored"
    assert not (tmp_path / "build").exists()


def test_create_directory_rejects_a_file_at_the_target(tmp_path: Path) -> None:
    (tmp_path / "pkg").write_text("not a directory", encoding="utf-8")
    registry = ToolRegistry([CreateDirectoryTool(Workspace(tmp_path))])

    execution = _execute(registry, "create_directory", {"path": "pkg"})

    assert execution.ok is False
    assert execution.error_code == "not_directory"
    assert str(tmp_path) not in (execution.error_message or "")


def test_create_directory_rejects_an_existing_directory_symlink(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    alias = tmp_path / "pkg"
    try:
        alias.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symbolic links are unavailable on this platform")
    registry = ToolRegistry([CreateDirectoryTool(Workspace(tmp_path))])

    execution = _execute(registry, "create_directory", {"path": "pkg"})

    assert execution.ok is False
    assert execution.error_code == "unsafe_directory_link"
    assert list(outside.iterdir()) == []


def test_create_directory_fails_closed_when_parent_is_swapped_to_a_link(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    parked = tmp_path / "parked"
    outside = tmp_path.parent / f"{tmp_path.name}-outside-parent"
    parent.mkdir()
    outside.mkdir()
    probe = tmp_path / "link-probe"
    try:
        probe.symlink_to(outside, target_is_directory=True)
        probe.unlink()
    except OSError:
        pytest.skip("directory symbolic links are unavailable on this platform")

    class RacyWorkspace(Workspace):
        def _before_mutation_commit(self, operation: str, path: Path) -> None:
            assert operation == "mkdir"
            assert path.name == "pkg"
            parent.rename(parked)
            parent.symlink_to(outside, target_is_directory=True)

    registry = ToolRegistry([CreateDirectoryTool(RacyWorkspace(tmp_path))])

    execution = _execute(registry, "create_directory", {"path": "parent/pkg"})

    assert execution.ok is False
    assert execution.error_code in {"invalid_path", "path_outside_workspace", "write_conflict"}
    assert not (outside / "pkg").exists()


def test_runtime_registers_directory_creation_for_safe_empty_projects(tmp_path: Path) -> None:
    runtime = build_runtime(
        tmp_path,
        permission_mode=CommandPermissionMode.SAFE,
        verification_commands=(),
    )

    execution = runtime.tools.execute(
        ToolCall(
            id="mkdir-runtime",
            name="create_directory",
            arguments={"path": "src"},
        )
    )

    assert execution.ok is True
    assert (tmp_path / "src").is_dir()
