"""Tests for the bounded project registry and safe project creation boundary."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from coding_agent.projects import ProjectRecord, ProjectRegistry, ProjectRegistryError
from coding_agent.state import StateStorageError

_NOW = datetime(2026, 8, 31, 8, 0, tzinfo=UTC)


def _clock(*values: datetime) -> Any:
    iterator = iter(values)
    return lambda: next(iterator)


def test_project_record_is_strict_bounded_and_timezone_aware(tmp_path: Path) -> None:
    valid = {
        "project_id": "p-demo",
        "display_name": "演示项目",
        "resolved_root": tmp_path.resolve(),
        "workspace_fingerprint": "a" * 64,
        "created_at": _NOW,
        "last_opened_at": _NOW,
    }
    record = ProjectRecord.model_validate(valid)
    assert record.display_name == "演示项目"

    for changes in (
        {"surprise": True},
        {"display_name": "  "},
        {"display_name": "bad\nname"},
        {"project_id": "../escape"},
        {"resolved_root": Path("relative")},
        {"workspace_fingerprint": "short"},
        {"created_at": _NOW.replace(tzinfo=None)},
        {"last_opened_at": _NOW - timedelta(seconds=1)},
        {"removed_at": _NOW.replace(tzinfo=None)},
        {"removed_at": _NOW - timedelta(seconds=1)},
    ):
        with pytest.raises((ValidationError, ValueError)):
            ProjectRecord.model_validate({**valid, **changes})


def test_register_round_trips_deduplicates_and_orders_by_recency(tmp_path: Path) -> None:
    state = tmp_path / "state"
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    registry = ProjectRegistry(
        state / "projects.json",
        clock=_clock(_NOW, _NOW + timedelta(seconds=1), _NOW + timedelta(seconds=2)),
    )

    first = registry.register(first_root, "第一个项目")
    reopened = registry.register(first_root.resolve(), "不会覆盖已有名称")
    second = registry.register(second_root)

    assert reopened.project_id == first.project_id
    assert reopened.display_name == "第一个项目"
    assert reopened.created_at == first.created_at
    assert reopened.last_opened_at > first.last_opened_at
    assert reopened.workspace_fingerprint == first.workspace_fingerprint
    assert registry.get(first.project_id) == reopened
    assert [project.project_id for project in registry.list()] == [
        second.project_id,
        first.project_id,
    ]
    payload = json.loads((state / "projects.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert len(payload["projects"]) == 2


def test_registry_loads_legacy_v1_records_without_removal_metadata(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    projects_file = tmp_path / "state" / "projects.json"
    registry = ProjectRegistry(projects_file, clock=lambda: _NOW)
    original = registry.register(root)
    payload = json.loads(projects_file.read_text(encoding="utf-8"))
    payload["projects"][0].pop("removed_at")
    projects_file.write_text(json.dumps(payload), encoding="utf-8")

    loaded = registry.list()

    assert len(loaded) == 1
    assert loaded[0].project_id == original.project_id
    assert loaded[0].removed_at is None


def test_mark_opened_updates_only_recency_and_rejects_unknown_ids(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    registry = ProjectRegistry(
        tmp_path / "state" / "projects.json",
        clock=_clock(_NOW, _NOW + timedelta(minutes=1)),
    )
    original = registry.register(root)

    opened = registry.mark_opened(original.project_id)

    assert opened.created_at == original.created_at
    assert opened.last_opened_at == _NOW + timedelta(minutes=1)
    with pytest.raises(ProjectRegistryError) as missing:
        registry.get("missing")
    assert missing.value.code == "project_not_found"
    with pytest.raises(ProjectRegistryError) as invalid:
        registry.mark_opened("../bad")
    assert invalid.value.code == "invalid_project_id"


def test_remove_hides_project_without_deleting_workspace_and_reopen_restores_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    sentinel = root / "keep.txt"
    sentinel.write_text("workspace data", encoding="utf-8")
    registry = ProjectRegistry(
        tmp_path / "state" / "projects.json",
        clock=_clock(
            _NOW,
            _NOW + timedelta(minutes=1),
            _NOW + timedelta(minutes=2),
        ),
    )
    original = registry.register(root, "保留名称")

    removed = registry.remove(original.project_id)

    assert removed.removed_at == _NOW + timedelta(minutes=1)
    assert registry.list() == ()
    assert sentinel.read_text(encoding="utf-8") == "workspace data"
    with pytest.raises(ProjectRegistryError) as hidden:
        registry.get(original.project_id)
    assert hidden.value.code == "project_not_found"
    with pytest.raises(ProjectRegistryError) as hidden_open:
        registry.mark_opened(original.project_id)
    assert hidden_open.value.code == "project_not_found"
    with pytest.raises(ProjectRegistryError) as already_removed:
        registry.remove(original.project_id)
    assert already_removed.value.code == "project_not_found"

    reopened = registry.register(root, "不会覆盖已有名称")

    assert reopened.project_id == original.project_id
    assert reopened.display_name == "保留名称"
    assert reopened.created_at == original.created_at
    assert reopened.last_opened_at == _NOW + timedelta(minutes=2)
    assert reopened.removed_at is None
    assert registry.list() == (reopened,)


def test_create_adds_only_one_new_leaf_and_never_overwrites(tmp_path: Path) -> None:
    parent = tmp_path / "projects"
    parent.mkdir()
    registry = ProjectRegistry(tmp_path / "state" / "projects.json", clock=lambda: _NOW)

    created = registry.create(parent, "fresh-project", "新项目")

    assert created.resolved_root == (parent / "fresh-project").resolve()
    assert created.resolved_root.is_dir()
    assert registry.get(created.project_id) == created

    sentinel = created.resolved_root / "keep.txt"
    sentinel.write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(ProjectRegistryError) as exists:
        registry.create(parent, "fresh-project")
    assert exists.value.code == "project_exists"
    assert sentinel.read_text(encoding="utf-8") == "do not overwrite"


@pytest.mark.parametrize(
    "directory_name",
    [
        "",
        " ",
        ".",
        "..",
        "nested/child",
        "nested\\child",
        "bad\nname",
        "NUL",
        "CON.txt",
        "bad:name",
        "tail.",
    ],
)
def test_create_rejects_non_leaf_or_nonportable_names(tmp_path: Path, directory_name: str) -> None:
    parent = tmp_path / "projects"
    parent.mkdir()
    registry = ProjectRegistry(tmp_path / "state" / "projects.json")

    with pytest.raises((ProjectRegistryError, ValueError)):
        registry.create(parent, directory_name)
    assert list(parent.iterdir()) == []


def test_create_requires_existing_parent_and_rolls_back_failed_registration(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    missing_parent = tmp_path / "missing"
    registry = ProjectRegistry(state / "projects.json")
    with pytest.raises(ProjectRegistryError) as invalid_parent:
        registry.create(missing_parent, "child")
    assert invalid_parent.value.code == "project_parent_invalid"

    parent = tmp_path / "parent"
    parent.mkdir()
    tiny = ProjectRegistry(state / "projects.json", max_bytes=1)
    with pytest.raises(ProjectRegistryError) as too_large:
        tiny.create(parent, "rolled-back")
    assert too_large.value.code == "state_too_large"
    assert not (parent / "rolled-back").exists()
    assert not (state / "projects.json").exists()


def test_registry_rejects_state_overlap_and_symlinked_project_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    nested = ProjectRegistry(workspace / ".state" / "projects.json")
    with pytest.raises(ProjectRegistryError) as overlap:
        nested.register(workspace)
    assert overlap.value.code == "project_state_overlap"

    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this platform")
    registry = ProjectRegistry(tmp_path / "external" / "projects.json")
    with pytest.raises(ProjectRegistryError) as unsafe:
        registry.register(alias)
    assert unsafe.value.code == "unsafe_state_path"


def test_registry_rejects_symlinked_state_file_and_non_private_hardlink(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    projects_file = tmp_path / "state" / "projects.json"
    registry = ProjectRegistry(projects_file, clock=lambda: _NOW)
    registry.register(root)
    original = tmp_path / "registry-copy.json"
    original.write_bytes(projects_file.read_bytes())
    projects_file.unlink()
    try:
        projects_file.symlink_to(original)
    except OSError:
        pytest.skip("file symlinks are unavailable on this platform")
    with pytest.raises(ProjectRegistryError) as linked:
        registry.list()
    assert linked.value.code == "unsafe_state_record"

    projects_file.unlink()
    os.link(original, projects_file)
    with pytest.raises(ProjectRegistryError) as hardlinked:
        registry.list()
    assert hardlinked.value.code == "unsafe_state_record"


@pytest.mark.parametrize(
    "raw",
    [
        b"not-json",
        b'{"schema_version":1,"schema_version":1,"projects":[]}',
        b'{"schema_version":"1","projects":[]}',
        b'{"schema_version":1,"projects":[],"extra":true}',
    ],
)
def test_registry_rejects_malformed_duplicate_or_non_strict_json(
    tmp_path: Path, raw: bytes
) -> None:
    path = tmp_path / "state" / "projects.json"
    path.parent.mkdir()
    path.write_bytes(raw)

    with pytest.raises(ProjectRegistryError):
        ProjectRegistry(path).list()


def test_registry_rejects_duplicate_entries_and_configured_limits(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    projects_file = tmp_path / "state" / "projects.json"
    registry = ProjectRegistry(projects_file, clock=lambda: _NOW)
    registry.register(first_root)
    raw = json.loads(projects_file.read_text(encoding="utf-8"))
    raw["projects"].append(raw["projects"][0])
    projects_file.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ProjectRegistryError) as duplicate:
        registry.list()
    assert duplicate.value.code == "project_registry_corrupt"

    projects_file.unlink()
    limited = ProjectRegistry(projects_file, max_projects=1, clock=lambda: _NOW)
    retained = limited.register(first_root)
    with pytest.raises(ProjectRegistryError) as full:
        limited.register(second_root)
    assert full.value.code == "project_limit"
    assert limited.get(retained.project_id).resolved_root == first_root.resolve()


def test_failed_atomic_replace_preserves_registry_and_removes_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    projects_file = tmp_path / "state" / "projects.json"
    registry = ProjectRegistry(projects_file, clock=lambda: _NOW)
    registry.register(first)
    original = projects_file.read_bytes()

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        del source, destination
        raise OSError("simulated failure")

    monkeypatch.setattr("coding_agent.state.os.replace", fail_replace)
    with pytest.raises(ProjectRegistryError) as failed:
        registry.register(second)
    assert failed.value.code == "state_io"
    assert projects_file.read_bytes() == original
    assert list(projects_file.parent.glob("*.tmp")) == []


def test_registry_constructor_rejects_relative_path_and_invalid_limits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        ProjectRegistry(Path("projects.json"))
    for invalid in (0, True, cast(Any, 1.5)):
        with pytest.raises(ValueError):
            ProjectRegistry(tmp_path / "projects.json", max_bytes=invalid)
    for invalid in (0, 10_001, True):
        with pytest.raises(ValueError):
            ProjectRegistry(tmp_path / "projects.json", max_projects=cast(Any, invalid))


def test_registry_rejects_a_state_directory_replaced_by_symlink(tmp_path: Path) -> None:
    state = tmp_path / "state"
    redirected = tmp_path / "redirected"
    redirected.mkdir()
    registry = ProjectRegistry(state / "projects.json")
    try:
        state.symlink_to(redirected, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this platform")

    root = tmp_path / "repo"
    root.mkdir()
    with pytest.raises((ProjectRegistryError, StateStorageError)):
        registry.register(root)
    assert not (redirected / "projects.json").exists()
