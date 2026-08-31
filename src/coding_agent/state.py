"""Private per-user state locations and hardened JSON persistence primitives.

The application state is deliberately separate from every repository the agent may
edit.  Stores in this package share the small helpers below so they all enforce the
same bounded-read, private-regular-file, no-symbolic-link, and atomic-replace rules.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from coding_agent.errors import CodedError

DEFAULT_MAX_STATE_JSON_BYTES = 1_000_000


class StateStorageError(CodedError):
    """A stable failure at the private state boundary."""


@dataclass(frozen=True, slots=True)
class StatePaths:
    root: Path

    @property
    def sessions(self) -> Path:
        return self.root / "sessions"

    @property
    def traces(self) -> Path:
        return self.root / "traces"

    @property
    def projects_file(self) -> Path:
        return self.root / "projects.json"

    @property
    def runs(self) -> Path:
        return self.root / "runs"


def default_state_paths(environment: dict[str, str] | None = None) -> StatePaths:
    """Resolve an explicit override or the native per-user state directory."""
    source = os.environ if environment is None else environment
    override = source.get("CODING_AGENT_STATE_DIR")
    if override:
        root = Path(override).expanduser()
        if not root.is_absolute():
            raise ValueError("CODING_AGENT_STATE_DIR must be an absolute path")
        return StatePaths(root.resolve(strict=False))

    if os.name == "nt":
        base = source.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
    else:
        base = source.get("XDG_STATE_HOME")
        root = Path(base).expanduser() if base else Path.home() / ".local" / "state"
    return StatePaths((root / "coding-agent").resolve(strict=False))


def require_absolute_state_path(path: Path, *, kind: str) -> Path:
    """Resolve one state path after rejecting existing link-like components."""

    raw = Path(path).expanduser()
    if not raw.is_absolute():
        raise ValueError(f"{kind} must be an absolute path")
    _reject_link_components(raw, kind=kind)
    try:
        return raw.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{kind} could not be resolved") from exc


def ensure_private_state_directory(directory: Path) -> None:
    """Create a private directory and fail closed if it is replaced by a link."""

    _reject_link_components(directory, kind="state directory")
    try:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise StateStorageError("state_io", "state directory could not be created") from exc
    if _is_link_like(directory) or not directory.is_dir():
        raise StateStorageError(
            "unsafe_state_dir",
            "state directory must be a regular directory and cannot be a symbolic link",
        )
    _reject_link_components(directory, kind="state directory")


def atomic_write_json_object(
    target: Path,
    payload: Mapping[str, object],
    *,
    max_bytes: int = DEFAULT_MAX_STATE_JSON_BYTES,
) -> Path:
    """Serialize and atomically replace one bounded private JSON object."""

    _require_positive_limit(max_bytes)
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise StateStorageError("state_invalid", "state record is not safe JSON") from exc
    if len(encoded) > max_bytes:
        raise StateStorageError(
            "state_too_large",
            "state record exceeds the configured size limit",
            metadata={"size_bytes": len(encoded), "max_bytes": max_bytes},
        )

    parent = target.parent
    ensure_private_state_directory(parent)
    _require_direct_child(target, parent)
    _reject_unsafe_record_if_present(target)
    temporary = parent / f".{target.name}.{uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise StateStorageError("state_io", "state record could not be saved") from exc
    return target


def read_bounded_json_object(
    target: Path,
    *,
    max_bytes: int = DEFAULT_MAX_STATE_JSON_BYTES,
) -> dict[str, Any]:
    """Read one strict JSON object without following links or allocating unbounded data."""

    _require_positive_limit(max_bytes)
    parent = target.parent
    _require_direct_child(target, parent)
    _reject_link_components(parent, kind="state directory")
    if not parent.is_dir():
        raise StateStorageError("state_not_found", "state directory does not exist")
    if _is_link_like(parent):
        raise StateStorageError("unsafe_state_dir", "state directory cannot be a symbolic link")
    _reject_unsafe_record_if_present(target)

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags)
    except FileNotFoundError as exc:
        raise StateStorageError("state_not_found", "state record does not exist") from exc
    except OSError as exc:
        raise StateStorageError("state_io", "state record could not be opened") from exc

    try:
        record_stat = os.fstat(descriptor)
        if not stat.S_ISREG(record_stat.st_mode) or record_stat.st_nlink != 1:
            raise StateStorageError(
                "unsafe_state_record",
                "state record must be one private regular file",
            )
        if record_stat.st_size > max_bytes:
            raise StateStorageError(
                "state_too_large",
                "state record exceeds the configured size limit",
                metadata={"size_bytes": record_stat.st_size, "max_bytes": max_bytes},
            )
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            raw = stream.read(max_bytes + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    if len(raw) > max_bytes:
        raise StateStorageError(
            "state_too_large",
            "state record exceeds the configured size limit",
            metadata={"max_bytes": max_bytes},
        )
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise StateStorageError("state_corrupt", "state record is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise StateStorageError("state_corrupt", "state record root must be a JSON object")
    return decoded


def state_is_outside_workspace(state_root: Path, workspace_root: Path) -> bool:
    """Return whether private state is disjoint from a repository workspace."""

    return not (
        state_root == workspace_root
        or state_root.is_relative_to(workspace_root)
        or workspace_root.is_relative_to(state_root)
    )


def _require_positive_limit(value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError("max_bytes must be a positive integer")


def _require_direct_child(target: Path, parent: Path) -> None:
    if target.parent != parent or not target.name:
        raise StateStorageError("unsafe_state_path", "state record must be a direct child")


def _reject_unsafe_record_if_present(target: Path) -> None:
    try:
        record_stat = target.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise StateStorageError("state_io", "state record could not be inspected") from exc
    if _is_link_like(target) or not stat.S_ISREG(record_stat.st_mode) or record_stat.st_nlink != 1:
        raise StateStorageError(
            "unsafe_state_record",
            "state record must be one private regular file and cannot be a symbolic link",
        )


def _reject_link_components(path: Path, *, kind: str) -> None:
    for component in (*reversed(path.parents), path):
        try:
            component.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise StateStorageError("state_io", f"{kind} could not be inspected") from exc
        if _is_link_like(component):
            raise StateStorageError(
                "unsafe_state_path",
                f"{kind} cannot contain symbolic links or reparse points",
            )


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    if os.name != "nt":
        return False
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")
