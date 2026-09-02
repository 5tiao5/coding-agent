"""Strict registry for user-selected repository workspaces."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self
from unicodedata import category
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from coding_agent.errors import CodedError
from coding_agent.run_id import require_run_id
from coding_agent.session import workspace_fingerprint
from coding_agent.state import (
    DEFAULT_MAX_STATE_JSON_BYTES,
    StateStorageError,
    atomic_write_json_object,
    read_bounded_json_object,
    require_absolute_state_path,
    state_is_outside_workspace,
)

PROJECTS_SCHEMA_VERSION = 1
DEFAULT_MAX_PROJECTS = 500
MAX_PROJECT_NAME_LENGTH = 120
MAX_RESOLVED_ROOT_LENGTH = 4096
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_WINDOWS_INVALID_NAME_CHARACTERS = frozenset('<>:"|?*')


class ProjectRegistryError(CodedError):
    """A stable, presentation-safe project registry failure."""


class ProjectRecord(BaseModel):
    """One immutable registration of a resolved local workspace."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    project_id: str = Field(min_length=1, max_length=64)
    display_name: str = Field(min_length=1, max_length=MAX_PROJECT_NAME_LENGTH)
    resolved_root: Path
    workspace_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime
    last_opened_at: datetime
    removed_at: datetime | None = None

    @field_validator("project_id")
    @classmethod
    def validate_project_id(cls, value: str) -> str:
        require_run_id(value)
        return value

    @field_validator("display_name")
    @classmethod
    def validate_display_name(cls, value: str) -> str:
        return _require_human_text(value, field="display_name")

    @field_validator("resolved_root")
    @classmethod
    def validate_resolved_root(cls, value: Path) -> Path:
        if not value.is_absolute() or len(str(value)) > MAX_RESOLVED_ROOT_LENGTH:
            raise ValueError("resolved_root must be a bounded absolute path")
        return value

    @field_validator("created_at", "last_opened_at")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("project timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("removed_at")
    @classmethod
    def validate_optional_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("project timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_timestamp_order(self) -> Self:
        if self.last_opened_at < self.created_at:
            raise ValueError("last_opened_at cannot precede created_at")
        if self.removed_at is not None and self.removed_at < self.last_opened_at:
            raise ValueError("removed_at cannot precede last_opened_at")
        return self


class _ProjectRegistryDocument(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    projects: tuple[ProjectRecord, ...] = ()


class ProjectRegistry:
    """Persist projects in one bounded JSON file outside registered workspaces."""

    def __init__(
        self,
        projects_file: Path,
        *,
        max_bytes: int = DEFAULT_MAX_STATE_JSON_BYTES,
        max_projects: int = DEFAULT_MAX_PROJECTS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        resolved = require_absolute_state_path(projects_file, kind="projects_file")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        if type(max_projects) is not int or not 1 <= max_projects <= 10_000:
            raise ValueError("max_projects must be between 1 and 10000")
        self._projects_file = resolved
        self._state_root = resolved.parent
        self._max_bytes = max_bytes
        self._max_projects = max_projects
        self._clock = clock or _utc_now

    @property
    def projects_file(self) -> Path:
        return self._projects_file

    def register(self, root: Path, display_name: str | None = None) -> ProjectRecord:
        """Register or reopen one existing directory, de-duplicated by identity/path."""

        resolved_root = self._validated_workspace_root(root)
        name = resolved_root.name if display_name is None else display_name
        if not name:
            name = str(resolved_root)
        name = _require_human_text(name, field="display_name")
        fingerprint = workspace_fingerprint(resolved_root)
        now = _require_aware_timestamp(self._clock())
        projects = list(self._load_document().projects)
        existing_index = _find_project(projects, resolved_root, fingerprint)
        if existing_index is not None:
            existing = projects[existing_index]
            reopened = ProjectRecord(
                project_id=existing.project_id,
                display_name=existing.display_name,
                resolved_root=resolved_root,
                workspace_fingerprint=fingerprint,
                created_at=existing.created_at,
                last_opened_at=max(existing.last_opened_at, now),
                removed_at=None,
            )
            projects[existing_index] = reopened
            self._save_projects(projects)
            return reopened

        if len(projects) >= self._max_projects:
            raise ProjectRegistryError(
                "project_limit",
                "project registry reached its configured entry limit",
                metadata={"max_projects": self._max_projects},
            )
        record = ProjectRecord(
            project_id=_new_project_id(projects),
            display_name=name,
            resolved_root=resolved_root,
            workspace_fingerprint=fingerprint,
            created_at=now,
            last_opened_at=now,
        )
        projects.append(record)
        self._save_projects(projects)
        return record

    def create(
        self,
        parent: Path,
        directory_name: str,
        display_name: str | None = None,
    ) -> ProjectRecord:
        """Create exactly one absent final directory and register it transactionally."""

        resolved_parent = require_absolute_state_path(parent, kind="project parent")
        if not resolved_parent.is_dir():
            raise ProjectRegistryError(
                "project_parent_invalid",
                "project parent must be an existing directory",
            )
        safe_name = _require_final_directory_name(directory_name)
        target = resolved_parent / safe_name
        if not state_is_outside_workspace(self._state_root, target):
            raise ProjectRegistryError(
                "project_state_overlap",
                "project directory must be disjoint from private application state",
            )
        try:
            target.mkdir(mode=0o700, exist_ok=False)
        except FileExistsError as exc:
            raise ProjectRegistryError(
                "project_exists",
                "project directory already exists and will not be overwritten",
            ) from exc
        except OSError as exc:
            raise ProjectRegistryError(
                "project_create_failed",
                "project directory could not be created",
            ) from exc

        try:
            return self.register(target, display_name=display_name)
        except Exception:
            # Only remove the exact empty leaf created by this call.  If another process
            # populated it, rmdir fails and preserves that external data.
            with suppress(OSError):
                target.rmdir()
            raise

    def get(self, project_id: str) -> ProjectRecord:
        """Return one registered project without mutating its last-opened timestamp."""

        _require_project_id(project_id)
        for project in self._load_document().projects:
            if project.project_id == project_id and project.removed_at is None:
                return project
        raise ProjectRegistryError("project_not_found", "project is not registered")

    def mark_opened(self, project_id: str) -> ProjectRecord:
        """Refresh one project's recency while preserving its stable identity."""

        _require_project_id(project_id)
        projects = list(self._load_document().projects)
        for index, project in enumerate(projects):
            if project.project_id != project_id or project.removed_at is not None:
                continue
            now = _require_aware_timestamp(self._clock())
            reopened = ProjectRecord(
                project_id=project.project_id,
                display_name=project.display_name,
                resolved_root=project.resolved_root,
                workspace_fingerprint=project.workspace_fingerprint,
                created_at=project.created_at,
                last_opened_at=max(project.last_opened_at, now),
                removed_at=None,
            )
            projects[index] = reopened
            self._save_projects(projects)
            return reopened
        raise ProjectRegistryError("project_not_found", "project is not registered")

    def remove(self, project_id: str) -> ProjectRecord:
        """Hide one project registration without deleting its workspace or history."""

        _require_project_id(project_id)
        projects = list(self._load_document().projects)
        for index, project in enumerate(projects):
            if project.project_id != project_id or project.removed_at is not None:
                continue
            now = _require_aware_timestamp(self._clock())
            removed = ProjectRecord(
                project_id=project.project_id,
                display_name=project.display_name,
                resolved_root=project.resolved_root,
                workspace_fingerprint=project.workspace_fingerprint,
                created_at=project.created_at,
                last_opened_at=project.last_opened_at,
                removed_at=max(project.last_opened_at, now),
            )
            projects[index] = removed
            self._save_projects(projects)
            return removed
        raise ProjectRegistryError("project_not_found", "project is not registered")

    def list(self) -> tuple[ProjectRecord, ...]:
        """List visible projects in most-recently-opened order."""

        return tuple(
            sorted(
                (
                    project
                    for project in self._load_document().projects
                    if project.removed_at is None
                ),
                key=lambda project: (project.last_opened_at, project.project_id),
                reverse=True,
            )
        )

    def _validated_workspace_root(self, root: Path) -> Path:
        try:
            resolved = require_absolute_state_path(root, kind="project root")
        except StateStorageError as exc:
            raise ProjectRegistryError(exc.code, exc.message, metadata=exc.metadata) from exc
        if not resolved.is_dir():
            raise ProjectRegistryError(
                "project_root_invalid",
                "project root must be an existing directory",
            )
        if not state_is_outside_workspace(self._state_root, resolved):
            raise ProjectRegistryError(
                "project_state_overlap",
                "private application state must be outside the project workspace",
            )
        return resolved

    def _load_document(self) -> _ProjectRegistryDocument:
        try:
            self._projects_file.lstat()
        except FileNotFoundError:
            return _ProjectRegistryDocument()
        except OSError as exc:
            raise ProjectRegistryError(
                "project_registry_io",
                "project registry is unavailable",
            ) from exc
        try:
            decoded = read_bounded_json_object(self._projects_file, max_bytes=self._max_bytes)
            document = _ProjectRegistryDocument.model_validate_json(
                json.dumps(decoded, allow_nan=False, separators=(",", ":")),
                strict=True,
            )
        except StateStorageError as exc:
            raise ProjectRegistryError(exc.code, exc.message, metadata=exc.metadata) from exc
        except (ValidationError, TypeError, ValueError) as exc:
            raise ProjectRegistryError(
                "project_registry_corrupt",
                "project registry does not satisfy its versioned schema",
            ) from exc
        if len(document.projects) > self._max_projects:
            raise ProjectRegistryError(
                "project_limit",
                "project registry exceeds its configured entry limit",
                metadata={"max_projects": self._max_projects},
            )
        _reject_duplicate_projects(document.projects)
        return document

    def _save_projects(self, projects: Sequence[ProjectRecord]) -> None:
        document = _ProjectRegistryDocument(projects=tuple(projects))
        try:
            atomic_write_json_object(
                self._projects_file,
                document.model_dump(mode="json"),
                max_bytes=self._max_bytes,
            )
        except StateStorageError as exc:
            raise ProjectRegistryError(exc.code, exc.message, metadata=exc.metadata) from exc


def _find_project(
    projects: Sequence[ProjectRecord],
    root: Path,
    fingerprint: str,
) -> int | None:
    normalized = os.path.normcase(str(root))
    for index, project in enumerate(projects):
        if project.workspace_fingerprint == fingerprint:
            return index
        if os.path.normcase(str(project.resolved_root)) == normalized:
            return index
    return None


def _reject_duplicate_projects(projects: Sequence[ProjectRecord]) -> None:
    ids: set[str] = set()
    fingerprints: set[str] = set()
    roots: set[str] = set()
    for project in projects:
        normalized_root = os.path.normcase(str(project.resolved_root))
        if (
            project.project_id in ids
            or project.workspace_fingerprint in fingerprints
            or normalized_root in roots
        ):
            raise ProjectRegistryError(
                "project_registry_corrupt",
                "project registry contains duplicate project identities",
            )
        ids.add(project.project_id)
        fingerprints.add(project.workspace_fingerprint)
        roots.add(normalized_root)


def _new_project_id(projects: Sequence[ProjectRecord]) -> str:
    existing = {project.project_id for project in projects}
    for _ in range(32):
        candidate = f"p-{uuid4().hex[:16]}"
        if candidate not in existing:
            return candidate
    raise ProjectRegistryError("project_id_exhausted", "a unique project ID could not be created")


def _require_project_id(project_id: str) -> str:
    try:
        return require_run_id(project_id)
    except ValueError as exc:
        raise ProjectRegistryError("invalid_project_id", "project ID is not valid") from exc


def _require_human_text(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if value != value.strip() or not value or len(value) > MAX_PROJECT_NAME_LENGTH:
        raise ValueError(
            f"{field} must be non-blank and at most {MAX_PROJECT_NAME_LENGTH} characters"
        )
    if any(category(character).startswith("C") for character in value):
        raise ValueError(f"{field} cannot contain control or formatting characters")
    return value


def _require_final_directory_name(value: str) -> str:
    name = _require_human_text(value, field="directory_name")
    if name in {".", ".."} or "/" in name or "\\" in name or Path(name).is_absolute():
        raise ProjectRegistryError(
            "invalid_directory_name",
            "directory_name must be one final path component",
        )
    reserved_stem = name.split(".", maxsplit=1)[0].upper()
    if (
        name.endswith((".", " "))
        or reserved_stem in _WINDOWS_RESERVED_NAMES
        or any(character in _WINDOWS_INVALID_NAME_CHARACTERS for character in name)
    ):
        raise ProjectRegistryError(
            "invalid_directory_name",
            "directory_name is not portable across supported systems",
        )
    return name


def _require_aware_timestamp(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ProjectRegistryError("clock_invalid", "project registry clock must be timezone-aware")
    return value.astimezone(UTC)


def _utc_now() -> datetime:
    return datetime.now(UTC)
