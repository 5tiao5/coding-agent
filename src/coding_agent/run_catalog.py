"""Bounded, one-file-per-run history metadata catalog."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Literal
from unicodedata import category

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from coding_agent.errors import CodedError
from coding_agent.run_id import require_run_id
from coding_agent.state import (
    StateStorageError,
    atomic_write_json_object,
    read_bounded_json_object,
    require_absolute_state_path,
    state_is_outside_workspace,
)

RUN_RECORD_SCHEMA_VERSION = 1
DEFAULT_MAX_RUN_RECORD_BYTES = 32 * 1024
DEFAULT_MAX_RUN_RECORDS = 10_000
MAX_TASK_TITLE_LENGTH = 240
MAX_MEMORY_SOURCE_RUNS = 6
_MIN_DIRECTORY_SCAN_LIMIT = 1_000
_CATALOG_WRITE_LOCK = Lock()


class RunCatalogError(CodedError):
    """A stable, presentation-safe run catalog failure."""


class RunRecord(BaseModel):
    """Small immutable metadata used to place a run in the project sidebar."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    run_id: str = Field(min_length=1, max_length=64)
    project_id: str = Field(min_length=1, max_length=64)
    workspace_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_title: str = Field(min_length=1, max_length=MAX_TASK_TITLE_LENGTH)
    created_at: datetime
    memory_requested: bool = False
    parent_run_id: str | None = Field(default=None, min_length=1, max_length=64)
    memory_source_run_ids: tuple[str, ...] = Field(
        default=(),
        max_length=MAX_MEMORY_SOURCE_RUNS,
    )

    @field_validator("run_id", "project_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        require_run_id(value)
        return value

    @field_validator("task_title")
    @classmethod
    def validate_task_title(cls, value: str) -> str:
        if value != value.strip() or not value:
            raise ValueError("task_title cannot be blank or padded")
        if any(category(character).startswith("C") for character in value):
            raise ValueError("task_title cannot contain control or formatting characters")
        return value

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("parent_run_id")
    @classmethod
    def validate_parent_run_id(cls, value: str | None) -> str | None:
        if value is not None:
            require_run_id(value)
        return value

    @field_validator("memory_source_run_ids")
    @classmethod
    def validate_memory_source_run_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for run_id in value:
            require_run_id(run_id)
        if len(set(value)) != len(value):
            raise ValueError("memory source run IDs must be unique")
        return value

    @model_validator(mode="after")
    def validate_memory_provenance(self) -> RunRecord:
        if self.run_id in self.memory_source_run_ids:
            raise ValueError("a run cannot use itself as a project-memory source")
        if self.parent_run_id == self.run_id:
            raise ValueError("a run cannot be its own parent")
        if self.parent_run_id is not None and self.parent_run_id not in self.memory_source_run_ids:
            raise ValueError("a parent run must be recorded as a project-memory source")
        if self.memory_source_run_ids and not self.memory_requested:
            expected = (self.parent_run_id,) if self.parent_run_id is not None else ()
            if self.memory_source_run_ids != expected:
                raise ValueError(
                    "memory sources require an explicit memory request unless they contain "
                    "only the parent run"
                )
        return self


class _RunRecordDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    run: RunRecord


class RunCatalog:
    """Persist immutable run metadata as independent bounded JSON records."""

    def __init__(
        self,
        runs_dir: Path,
        *,
        max_record_bytes: int = DEFAULT_MAX_RUN_RECORD_BYTES,
        max_records: int = DEFAULT_MAX_RUN_RECORDS,
        workspace_root: Path | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        resolved = require_absolute_state_path(runs_dir, kind="runs_dir")
        if type(max_record_bytes) is not int or max_record_bytes < 1:
            raise ValueError("max_record_bytes must be a positive integer")
        if type(max_records) is not int or not 1 <= max_records <= 100_000:
            raise ValueError("max_records must be between 1 and 100000")
        if workspace_root is not None:
            workspace = require_absolute_state_path(workspace_root, kind="workspace_root")
            if not workspace.is_dir():
                raise ValueError("workspace_root must be an existing directory")
            if not state_is_outside_workspace(resolved, workspace):
                raise ValueError("runs_dir must be outside the workspace")
        self._runs_dir = resolved
        self._max_record_bytes = max_record_bytes
        self._max_records = max_records
        self._clock = clock or _utc_now

    @property
    def runs_dir(self) -> Path:
        return self._runs_dir

    def create(
        self,
        *,
        run_id: str,
        project_id: str,
        workspace_fingerprint: str,
        task_title: str,
        memory_requested: bool = False,
        parent_run_id: str | None = None,
        memory_source_run_ids: tuple[str, ...] = (),
    ) -> RunRecord:
        """Construct and save one record using the catalog's timezone-aware clock."""

        try:
            record = RunRecord(
                run_id=run_id,
                project_id=project_id,
                workspace_fingerprint=workspace_fingerprint,
                task_title=task_title,
                created_at=_require_aware_timestamp(self._clock()),
                memory_requested=memory_requested,
                parent_run_id=parent_run_id,
                memory_source_run_ids=memory_source_run_ids,
            )
        except (ValidationError, TypeError, ValueError) as exc:
            raise RunCatalogError(
                "run_record_invalid",
                "run metadata could not be created",
            ) from exc
        self.save(record)
        return record

    def save(self, record: RunRecord) -> Path:
        """Atomically save one record; conflicting rewrites fail closed."""

        if not isinstance(record, RunRecord):
            raise TypeError("record must be a RunRecord")
        target = self._record_path(record.run_id)
        document = _RunRecordDocument(run=record)
        with _CATALOG_WRITE_LOCK:
            try:
                existing = self.get(record.run_id)
            except RunCatalogError as exc:
                if exc.code != "run_not_found":
                    raise
            else:
                if existing == record:
                    return target
                raise RunCatalogError(
                    "run_conflict",
                    "run metadata is immutable and cannot be overwritten",
                )
            if self._record_count() >= self._max_records:
                raise RunCatalogError(
                    "run_limit",
                    "run catalog reached its configured entry limit",
                    metadata={"max_records": self._max_records},
                )
            try:
                return atomic_write_json_object(
                    target,
                    document.model_dump(mode="json"),
                    max_bytes=self._max_record_bytes,
                )
            except StateStorageError as exc:
                raise RunCatalogError(exc.code, exc.message, metadata=exc.metadata) from exc

    def get(self, run_id: str) -> RunRecord:
        """Load and validate one immutable run record."""

        target = self._record_path(run_id)
        try:
            decoded = read_bounded_json_object(target, max_bytes=self._max_record_bytes)
            document = _RunRecordDocument.model_validate_json(
                json.dumps(decoded, allow_nan=False, separators=(",", ":")),
                strict=True,
            )
        except StateStorageError as exc:
            code = "run_not_found" if exc.code == "state_not_found" else exc.code
            message = "run is not recorded" if code == "run_not_found" else exc.message
            raise RunCatalogError(code, message, metadata=exc.metadata) from exc
        except (ValidationError, TypeError, ValueError) as exc:
            raise RunCatalogError(
                "run_record_corrupt",
                "run record does not satisfy its versioned schema",
            ) from exc
        if document.run.run_id != run_id:
            raise RunCatalogError(
                "run_record_corrupt",
                "run record ID does not match its filename",
            )
        return document.run

    def list(
        self,
        *,
        project_id: str | None = None,
        workspace_fingerprint: str | None = None,
        limit: int = 100,
    ) -> tuple[RunRecord, ...]:
        """List newest records, optionally restricted to one physical project identity."""

        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if project_id is not None:
            _require_identifier(project_id, field="project_id")
        if workspace_fingerprint is not None:
            _require_workspace_fingerprint(workspace_fingerprint)
        entries = self._record_entries()
        records: list[RunRecord] = []
        for entry in entries:
            record = self.get(entry.stem)
            if project_id is not None and record.project_id != project_id:
                continue
            if (
                workspace_fingerprint is not None
                and record.workspace_fingerprint != workspace_fingerprint
            ):
                continue
            records.append(record)
        records.sort(key=lambda record: (record.created_at, record.run_id), reverse=True)
        return tuple(records[:limit])

    def _record_path(self, run_id: str) -> Path:
        _require_identifier(run_id, field="run_id")
        target = self._runs_dir / f"{run_id}.json"
        if target.parent != self._runs_dir:
            raise RunCatalogError("invalid_run_id", "run ID does not map to the runs directory")
        return target

    def _record_count(self) -> int:
        return len(self._record_entries())

    def _record_entries(self) -> tuple[Path, ...]:
        try:
            self._runs_dir.lstat()
        except FileNotFoundError:
            return ()
        except OSError as exc:
            raise RunCatalogError("run_catalog_io", "run catalog is unavailable") from exc
        try:
            current = require_absolute_state_path(self._runs_dir, kind="runs_dir")
        except StateStorageError as exc:
            raise RunCatalogError(exc.code, exc.message, metadata=exc.metadata) from exc
        if current != self._runs_dir or not current.is_dir():
            raise RunCatalogError("unsafe_state_dir", "runs_dir must be a regular directory")
        try:
            candidates = current.iterdir()
        except OSError as exc:
            raise RunCatalogError("run_catalog_io", "run catalog could not be listed") from exc

        records: list[Path] = []
        scan_limit = max(self._max_records * 2, _MIN_DIRECTORY_SCAN_LIMIT)
        try:
            for scanned, candidate in enumerate(candidates, start=1):
                if scanned > scan_limit:
                    raise RunCatalogError(
                        "run_scan_limit",
                        "run catalog contains too many directory entries",
                        metadata={"max_entries": scan_limit},
                    )
                if candidate.suffix != ".json" or candidate.name.startswith("."):
                    continue
                try:
                    require_run_id(candidate.stem)
                except ValueError:
                    continue
                records.append(candidate)
                if len(records) > self._max_records:
                    raise RunCatalogError(
                        "run_limit",
                        "run catalog exceeds its configured entry limit",
                        metadata={"max_records": self._max_records},
                    )
        except OSError as exc:
            raise RunCatalogError("run_catalog_io", "run catalog could not be listed") from exc
        return tuple(records)


def _require_identifier(value: str, *, field: str) -> str:
    try:
        return require_run_id(value)
    except ValueError as exc:
        raise RunCatalogError(f"invalid_{field}", f"{field} is not valid") from exc


def _require_workspace_fingerprint(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RunCatalogError(
            "invalid_workspace_fingerprint",
            "workspace_fingerprint is not valid",
        )
    return value


def _require_aware_timestamp(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise RunCatalogError("clock_invalid", "run catalog clock must be timezone-aware")
    return value.astimezone(UTC)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def normalize_task_title(task: str) -> str:
    """Derive one safe, bounded sidebar title from an arbitrary task prompt."""

    without_controls = "".join(
        " " if category(character).startswith("C") else character for character in task
    )
    normalized = " ".join(without_controls.split())
    return normalized[:MAX_TASK_TITLE_LENGTH].rstrip() or "未命名任务"
