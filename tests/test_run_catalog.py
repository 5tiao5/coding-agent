"""Tests for strict, bounded, per-run sidebar metadata."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from coding_agent.run_catalog import (
    MAX_TASK_TITLE_LENGTH,
    RunCatalog,
    RunCatalogError,
    RunRecord,
    normalize_task_title,
)

_NOW = datetime(2026, 8, 31, 9, 0, tzinfo=UTC)
_FP_ONE = "1" * 64
_FP_TWO = "2" * 64


def _record(**changes: Any) -> RunRecord:
    values: dict[str, Any] = {
        "run_id": "run-1",
        "project_id": "p-demo",
        "workspace_fingerprint": _FP_ONE,
        "task_title": "修复价格计算",
        "created_at": _NOW,
    }
    values.update(changes)
    return RunRecord.model_validate(values)


def _clock(*values: datetime) -> Any:
    iterator = iter(values)
    return lambda: next(iterator)


def test_run_record_is_strict_bounded_single_line_and_timezone_aware() -> None:
    assert _record().task_title == "修复价格计算"
    for changes in (
        {"run_id": "../escape"},
        {"project_id": "UPPER"},
        {"workspace_fingerprint": "short"},
        {"workspace_fingerprint": "A" * 64},
        {"task_title": " "},
        {"task_title": "line one\nline two"},
        {"task_title": "x" * 241},
        {"created_at": _NOW.replace(tzinfo=None)},
        {"surprise": True},
    ):
        with pytest.raises((ValidationError, ValueError)):
            _record(**changes)


def test_task_title_normalization_is_single_line_bounded_and_never_padded() -> None:
    boundary_task = f"{'x' * (MAX_TASK_TITLE_LENGTH - 1)} trailing text"

    title = normalize_task_title(boundary_task)

    assert title == "x" * (MAX_TASK_TITLE_LENGTH - 1)
    assert len(title) == MAX_TASK_TITLE_LENGTH - 1
    assert normalize_task_title("  line one\nline two\t") == "line one line two"
    assert normalize_task_title("\u200b\x00") == "未命名任务"


def test_create_get_list_and_project_filter_are_durable_and_sorted(tmp_path: Path) -> None:
    catalog = RunCatalog(
        tmp_path / "state" / "runs",
        clock=_clock(_NOW, _NOW + timedelta(seconds=1), _NOW + timedelta(seconds=2)),
    )
    first = catalog.create(
        run_id="run-1",
        project_id="p-one",
        workspace_fingerprint=_FP_ONE,
        task_title="第一项",
    )
    second = catalog.create(
        run_id="run-2",
        project_id="p-two",
        workspace_fingerprint=_FP_TWO,
        task_title="第二项",
    )
    third = catalog.create(
        run_id="run-3",
        project_id="p-one",
        workspace_fingerprint=_FP_ONE,
        task_title="第三项",
    )

    assert catalog.get(first.run_id) == first
    assert [record.run_id for record in catalog.list()] == ["run-3", "run-2", "run-1"]
    assert [record.run_id for record in catalog.list(project_id="p-one", limit=1)] == ["run-3"]
    assert [
        record.run_id
        for record in catalog.list(
            project_id="p-one",
            workspace_fingerprint=_FP_ONE,
        )
    ] == ["run-3", "run-1"]
    assert catalog.list(project_id="p-one", workspace_fingerprint=_FP_TWO) == ()
    assert len(tuple((tmp_path / "state" / "runs").glob("*.json"))) == 3
    payload = json.loads((tmp_path / "state" / "runs" / "run-1.json").read_text("utf-8"))
    assert payload["schema_version"] == 1
    assert payload["run"]["task_title"] == "第一项"
    assert payload["run"]["workspace_fingerprint"] == _FP_ONE
    assert second.created_at > first.created_at
    assert third.created_at > second.created_at


def test_create_normalizes_schema_failures_to_a_stable_catalog_error(tmp_path: Path) -> None:
    catalog = RunCatalog(tmp_path / "runs")

    with pytest.raises(RunCatalogError) as raised:
        catalog.create(
            run_id="run-1",
            project_id="p-one",
            workspace_fingerprint=_FP_ONE,
            task_title="padded ",
        )

    assert raised.value.code == "run_record_invalid"
    assert raised.value.message == "run metadata could not be created"


def test_save_is_idempotent_but_rejects_conflicting_history(tmp_path: Path) -> None:
    catalog = RunCatalog(tmp_path / "runs")
    record = _record()
    first_path = catalog.save(record)

    assert catalog.save(record) == first_path
    with pytest.raises(RunCatalogError) as conflict:
        catalog.save(record.model_copy(update={"task_title": "different"}))
    assert conflict.value.code == "run_conflict"
    assert catalog.get(record.run_id) == record
    with pytest.raises(TypeError, match="RunRecord"):
        catalog.save(cast(Any, {"run_id": "run-2"}))


def test_catalog_rejects_missing_invalid_and_filename_mismatch(tmp_path: Path) -> None:
    catalog = RunCatalog(tmp_path / "runs")
    with pytest.raises(RunCatalogError) as missing:
        catalog.get("missing")
    assert missing.value.code == "run_not_found"
    with pytest.raises(RunCatalogError) as invalid:
        catalog.get("../bad")
    assert invalid.value.code == "invalid_run_id"

    catalog.save(_record())
    source = catalog.runs_dir / "run-1.json"
    alias = catalog.runs_dir / "alias.json"
    alias.write_bytes(source.read_bytes())
    with pytest.raises(RunCatalogError) as mismatch:
        catalog.get("alias")
    assert mismatch.value.code == "run_record_corrupt"


@pytest.mark.parametrize(
    "raw",
    [
        b"not-json",
        b'{"schema_version":1,"schema_version":1,"run":{}}',
        b'{"schema_version":"1","run":{}}',
        b'{"schema_version":1,"run":{},"extra":true}',
    ],
)
def test_catalog_rejects_malformed_duplicate_or_non_strict_json(tmp_path: Path, raw: bytes) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "run-1.json").write_bytes(raw)
    with pytest.raises(RunCatalogError):
        RunCatalog(runs).get("run-1")


def test_catalog_enforces_per_record_and_catalog_bounds(tmp_path: Path) -> None:
    tiny = RunCatalog(tmp_path / "tiny", max_record_bytes=1)
    with pytest.raises(RunCatalogError) as oversized:
        tiny.save(_record())
    assert oversized.value.code == "state_too_large"
    assert not tiny.runs_dir.joinpath("run-1.json").exists()

    limited = RunCatalog(tmp_path / "limited", max_records=1)
    limited.save(_record())
    with pytest.raises(RunCatalogError) as full:
        limited.save(_record(run_id="run-2"))
    assert full.value.code == "run_limit"


def test_catalog_bounds_directory_scanning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    for index in range(3):
        (runs / f"junk-{index}.txt").write_text("x", encoding="utf-8")
    monkeypatch.setattr("coding_agent.run_catalog._MIN_DIRECTORY_SCAN_LIMIT", 1)

    with pytest.raises(RunCatalogError) as too_many:
        RunCatalog(runs, max_records=1).list()
    assert too_many.value.code == "run_scan_limit"


def test_list_ignores_unrelated_entries_but_rejects_unsafe_run_records(tmp_path: Path) -> None:
    catalog = RunCatalog(tmp_path / "runs")
    catalog.save(_record())
    (catalog.runs_dir / "notes.txt").write_text("ignored", encoding="utf-8")
    (catalog.runs_dir / "BAD NAME.json").write_text("ignored", encoding="utf-8")
    (catalog.runs_dir / ".temporary.json").write_text("ignored", encoding="utf-8")
    assert [record.run_id for record in catalog.list()] == ["run-1"]

    source = catalog.runs_dir / "run-1.json"
    linked = catalog.runs_dir / "linked.json"
    try:
        linked.symlink_to(source)
    except OSError:
        pytest.skip("file symlinks are unavailable on this platform")
    with pytest.raises(RunCatalogError) as unsafe:
        catalog.list()
    assert unsafe.value.code == "unsafe_state_record"


def test_catalog_rejects_hardlinks_and_runs_directory_symlink_swap(tmp_path: Path) -> None:
    catalog = RunCatalog(tmp_path / "runs")
    catalog.save(_record())
    source = catalog.runs_dir / "run-1.json"
    alias = tmp_path / "outside.json"
    os.link(source, alias)
    with pytest.raises(RunCatalogError) as linked:
        catalog.get("run-1")
    assert linked.value.code == "unsafe_state_record"

    source.unlink()
    alias.unlink()
    catalog.runs_dir.rmdir()
    redirected = tmp_path / "redirected"
    redirected.mkdir()
    try:
        catalog.runs_dir.symlink_to(redirected, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this platform")
    with pytest.raises(RunCatalogError) as unsafe_dir:
        catalog.list()
    assert unsafe_dir.value.code == "unsafe_state_path"


def test_failed_atomic_save_removes_temp_and_preserves_existing_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = RunCatalog(tmp_path / "runs")
    first = _record()
    catalog.save(first)

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        del source, destination
        raise OSError("simulated failure")

    monkeypatch.setattr("coding_agent.state.os.replace", fail_replace)
    with pytest.raises(RunCatalogError) as failed:
        catalog.save(_record(run_id="run-2"))
    assert failed.value.code == "state_io"
    assert catalog.get(first.run_id) == first
    assert list(catalog.runs_dir.glob("*.tmp")) == []


def test_catalog_constructor_and_list_validate_boundaries(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        RunCatalog(Path("runs"))
    for invalid in (0, True, cast(Any, 1.5)):
        with pytest.raises(ValueError):
            RunCatalog(tmp_path / "runs", max_record_bytes=invalid)
    for invalid in (0, 100_001, True):
        with pytest.raises(ValueError):
            RunCatalog(tmp_path / "runs", max_records=cast(Any, invalid))

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ValueError, match="outside"):
        RunCatalog(workspace / ".state" / "runs", workspace_root=workspace)

    catalog = RunCatalog(tmp_path / "runs")
    assert catalog.list() == ()
    for invalid in (0, 1001, True):
        with pytest.raises(ValueError, match="limit"):
            catalog.list(limit=cast(Any, invalid))
    with pytest.raises(RunCatalogError) as bad_project:
        catalog.list(project_id="UPPER")
    assert bad_project.value.code == "invalid_project_id"
    with pytest.raises(RunCatalogError) as bad_fingerprint:
        catalog.list(workspace_fingerprint="not-a-fingerprint")
    assert bad_fingerprint.value.code == "invalid_workspace_fingerprint"


def test_catalog_rejects_non_directory_runs_path_and_invalid_clock(tmp_path: Path) -> None:
    runs_file = tmp_path / "runs"
    runs_file.write_text("occupied", encoding="utf-8")
    with pytest.raises(RunCatalogError) as invalid_dir:
        RunCatalog(runs_file).list()
    assert invalid_dir.value.code == "unsafe_state_dir"

    bad_clock = RunCatalog(tmp_path / "other", clock=lambda: _NOW.replace(tzinfo=None))
    with pytest.raises(RunCatalogError) as invalid_clock:
        bad_clock.create(
            run_id="run-1",
            project_id="p-demo",
            workspace_fingerprint=_FP_ONE,
            task_title="task",
        )
    assert invalid_clock.value.code == "clock_invalid"
