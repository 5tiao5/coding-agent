"""Rooted filesystem boundary shared by every local repository tool."""

from __future__ import annotations

import os
import secrets
import stat
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from itertools import islice
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal
from unicodedata import category

from pathspec import GitIgnoreSpec

from coding_agent.errors import CodedError

ExpectedPathKind = Literal["file", "directory", "any"]
IgnoreSignature = tuple[int, int, int, int, int] | None
FileIdentity = tuple[int, int]

_HARD_EXCLUDED_PARTS = frozenset({".git", ".coding-agent"})
_HARD_EXCLUDED_PREFIXES = (".coding-agent-tmp-",)
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "aux",
        "clock$",
        "con",
        "conin$",
        "conout$",
        "nul",
        "prn",
        *(f"com{number}" for number in range(1, 10)),
        *(f"com{number}" for number in "¹²³"),
        *(f"lpt{number}" for number in range(1, 10)),
        *(f"lpt{number}" for number in "¹²³"),
    }
)
_SENSITIVE_FILENAMES = frozenset(
    {
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
    }
)
_SENSITIVE_SUFFIXES = frozenset({".key", ".p12", ".pfx", ".pem"})
_MAX_RAW_DIRECTORY_ENTRIES = 20_000
_MAX_GITIGNORE_BYTES = 256_000
_MAX_GITIGNORE_PATTERNS = 10_000
_MAX_GITIGNORE_LINE_CHARS = 8_000
_ERROR_UNABLE_TO_MOVE_REPLACEMENT_2 = 1177


class WorkspaceError(CodedError):
    """Expected workspace-policy failure with a stable model-facing code."""


@dataclass(frozen=True, slots=True)
class WorkspacePath:
    """An allowed path with a physical target and a stable relative display name."""

    path: Path
    relative: str


@dataclass(frozen=True, slots=True)
class WorkspaceEntry:
    """One safe child returned by a non-recursive workspace scan."""

    path: Path
    relative: str
    is_directory: bool
    is_symlink: bool


@dataclass(frozen=True, slots=True)
class DirectoryScan:
    """A bounded directory scan with explicit incompleteness metadata."""

    entries: tuple[WorkspaceEntry, ...]
    examined: int
    skipped: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    """An authorized file version used as a compare-and-swap write precondition."""

    relative: str
    data: bytes | None
    sha256: str | None
    mode: int | None
    identity: FileIdentity | None = None


@dataclass(frozen=True, slots=True)
class WriteReceipt:
    """The stable outcome of one atomic workspace write."""

    relative: str
    before_sha256: str | None
    after_sha256: str
    bytes_written: int
    created: bool
    durability_uncertain: bool = False
    after_identity: FileIdentity | None = None


class Workspace:
    """Resolve and enumerate paths without allowing access outside one root."""

    def __init__(self, root: str | Path) -> None:
        try:
            resolved_root = Path(root).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise WorkspaceError("invalid_workspace", "workspace root is not accessible") from exc
        if not resolved_root.is_dir():
            raise WorkspaceError("invalid_workspace", "workspace root must be a directory")
        if any(part.casefold() in _HARD_EXCLUDED_PARTS for part in resolved_root.parts):
            raise WorkspaceError("invalid_workspace", "workspace root is inside excluded metadata")

        self._root = resolved_root
        self._ignore_cache: dict[str, tuple[IgnoreSignature, GitIgnoreSpec]] = {}
        self._ignore_spec_for(PurePosixPath("."))

    @property
    def root(self) -> Path:
        return self._root

    def contains(self, path: Path) -> bool:
        """Resolve a candidate and report whether its physical path is inside the root."""
        try:
            resolved = path.resolve(strict=False)
            resolved.relative_to(self._root)
        except (OSError, RuntimeError, ValueError):
            return False
        return True

    def resolve(
        self,
        user_path: str,
        *,
        expected: ExpectedPathKind = "any",
        allow_ignored: bool = False,
        allow_sensitive: bool = False,
    ) -> WorkspacePath:
        """Resolve an existing relative path and enforce all workspace policies."""
        if expected not in {"file", "directory", "any"}:
            raise ValueError(f"unsupported expected path kind: {expected}")
        relative = self._normalize_user_path(user_path)
        lexical = self._root.joinpath(*relative.parts)

        try:
            resolved = lexical.resolve(strict=True)
        except FileNotFoundError as exc:
            raise WorkspaceError("not_found", f"path not found: {relative.as_posix()}") from exc
        except PermissionError as exc:
            raise WorkspaceError(
                "permission_denied", f"permission denied: {relative.as_posix()}"
            ) from exc
        except (OSError, RuntimeError) as exc:
            raise WorkspaceError("io_error", f"cannot resolve path: {relative.as_posix()}") from exc

        if not self.contains(resolved):
            raise WorkspaceError(
                "path_outside_workspace",
                f"path resolves outside the workspace: {relative.as_posix()}",
            )

        if self._contains_directory_link(relative):
            raise WorkspaceError(
                "invalid_path",
                f"directory links cannot be traversed: {relative.as_posix()}",
            )

        self._enforce_policy(
            relative,
            resolved,
            allow_ignored=allow_ignored,
            allow_sensitive=allow_sensitive,
        )

        if expected == "file" and not resolved.is_file():
            raise WorkspaceError("not_file", f"path is not a file: {relative.as_posix()}")
        if expected == "directory" and not resolved.is_dir():
            raise WorkspaceError("not_directory", f"path is not a directory: {relative.as_posix()}")

        return WorkspacePath(path=resolved, relative=relative.as_posix())

    def children(
        self,
        directory: WorkspacePath,
        *,
        max_entries: int = 10_000,
        max_examined: int = _MAX_RAW_DIRECTORY_ENTRIES,
    ) -> DirectoryScan:
        """Return bounded safe children without following directory links.

        Results are name-sorted when the raw directory fits within ``max_examined``. At the
        raw limit the prefix is still bounded and explicitly marked truncated, but its members
        may reflect the operating system's enumeration order.
        """
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        if not 1 <= max_examined <= _MAX_RAW_DIRECTORY_ENTRIES:
            raise ValueError(f"max_examined must be between 1 and {_MAX_RAW_DIRECTORY_ENTRIES}")
        try:
            resolved_directory = directory.path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise WorkspaceError("io_error", "directory handle is not accessible") from exc
        if resolved_directory != directory.path or not self.contains(resolved_directory):
            raise WorkspaceError("path_outside_workspace", "directory handle is outside workspace")
        if not resolved_directory.is_dir():
            raise WorkspaceError("not_directory", "directory handle does not refer to a directory")
        try:
            rebound = self.resolve(directory.relative, expected="directory")
        except WorkspaceError as exc:
            raise WorkspaceError(
                "invalid_path", "directory handle does not match its relative path"
            ) from exc
        if rebound.path != resolved_directory:
            raise WorkspaceError(
                "invalid_path", "directory handle does not match its relative path"
            )

        try:
            with os.scandir(resolved_directory) as scan:
                selected = sorted(
                    islice(scan, max_examined),
                    key=lambda entry: (entry.name.casefold(), entry.name),
                )
        except PermissionError as exc:
            raise WorkspaceError(
                "permission_denied", f"permission denied: {directory.relative}"
            ) from exc
        except OSError as exc:
            raise WorkspaceError(
                "io_error", f"cannot list directory: {directory.relative}"
            ) from exc

        # Exact exhaustion cannot be distinguished without exceeding the hard raw budget.
        # Conservatively report an incomplete scan when the budget is fully consumed.
        truncated = len(selected) == max_examined
        safe_entries: list[WorkspaceEntry] = []
        skipped = 0
        for raw_entry in selected:
            relative = self._join_relative(directory.relative, raw_entry.name)
            try:
                normalized_relative = self._normalize_user_path(relative)
            except WorkspaceError:
                skipped += 1
                continue
            if normalized_relative.as_posix() != relative:
                skipped += 1
                continue
            lexical = Path(raw_entry.path)
            link_like = self._is_link_like(lexical)

            try:
                resolved = lexical.resolve(strict=True)
                is_directory = resolved.is_dir()
                is_file = resolved.is_file()
            except (OSError, RuntimeError):
                skipped += 1
                continue

            if not self.contains(resolved):
                skipped += 1
                continue
            if link_like and is_directory:
                skipped += 1
                continue
            if not (is_directory or is_file):
                skipped += 1
                continue

            relative_path = PurePosixPath(relative)
            if self._is_excluded(relative_path, resolved, is_directory=is_directory):
                continue
            resolved_relative = self._resolved_relative(resolved)
            if self._is_sensitive(relative_path) or self._is_sensitive(resolved_relative):
                continue

            if len(safe_entries) < max_entries:
                safe_entries.append(
                    WorkspaceEntry(
                        path=resolved,
                        relative=relative,
                        is_directory=is_directory,
                        is_symlink=link_like,
                    )
                )
            else:
                truncated = True

        return DirectoryScan(
            entries=tuple(safe_entries),
            examined=len(selected),
            skipped=skipped,
            truncated=truncated,
        )

    def read_bytes(self, target: WorkspacePath, *, max_bytes: int) -> bytes:
        """Read one previously authorized file through a revalidated OS handle.

        Authorization is repeated before opening, then the opened handle's physical target is
        checked before any bytes are consumed. This prevents a path swapped to an outside
        symlink between ``resolve`` and the tool's read from escaping the workspace.
        """
        if max_bytes < 1:
            raise ValueError("max_bytes must be at least 1")
        rebound = self.resolve(target.relative, expected="file")
        if rebound.path != target.path:
            raise WorkspaceError(
                "invalid_path", f"file changed after authorization: {target.relative}"
            )
        return self._read_approved_file(
            rebound.path,
            max_bytes=max_bytes,
            display_path=target.relative,
            too_large_code="file_too_large",
        )

    def snapshot_for_write(self, user_path: str, *, max_bytes: int) -> FileSnapshot:
        """Capture one authorized file version, or an authorized missing destination.

        The returned value is a compare-and-swap precondition rather than permission to write
        through its path. ``commit_bytes`` repeats policy and content checks immediately before
        changing the directory entry.
        """
        if max_bytes < 1:
            raise ValueError("max_bytes must be at least 1")

        relative, lexical = self._mutation_location(user_path)
        try:
            lexical_stat = lexical.lstat()
        except FileNotFoundError:
            return FileSnapshot(
                relative=relative.as_posix(),
                data=None,
                sha256=None,
                mode=None,
            )
        except PermissionError as exc:
            raise WorkspaceError(
                "permission_denied", f"permission denied: {relative.as_posix()}"
            ) from exc
        except OSError as exc:
            raise WorkspaceError("io_error", f"cannot inspect file: {relative.as_posix()}") from exc

        if self._is_link_like(lexical):
            raise WorkspaceError(
                "unsafe_file_link",
                f"filesystem links cannot be changed: {relative.as_posix()}",
            )
        if not stat.S_ISREG(lexical_stat.st_mode):
            raise WorkspaceError("not_file", f"path is not a file: {relative.as_posix()}")
        if lexical_stat.st_nlink > 1:
            raise WorkspaceError(
                "unsafe_file_link",
                f"files with multiple hard links cannot be changed: {relative.as_posix()}",
            )
        _require_no_windows_named_streams(lexical, relative.as_posix())

        target = self.resolve(relative.as_posix(), expected="file")
        data = self.read_bytes(target, max_bytes=max_bytes)

        # Bind the recorded mode to the same lexical target that was authorized. A swap after
        # the secure read is treated as a conflict by commit_bytes; a link is rejected here too.
        try:
            final_stat = lexical.lstat()
            final_path = lexical.resolve(strict=True)
        except FileNotFoundError as exc:
            raise WorkspaceError(
                "write_conflict", f"file changed while it was inspected: {relative.as_posix()}"
            ) from exc
        except (OSError, RuntimeError) as exc:
            raise WorkspaceError("io_error", f"cannot verify file: {relative.as_posix()}") from exc
        if (
            self._is_link_like(lexical)
            or not stat.S_ISREG(final_stat.st_mode)
            or final_stat.st_nlink > 1
            or final_path != target.path
        ):
            raise WorkspaceError(
                "write_conflict", f"file changed while it was inspected: {relative.as_posix()}"
            )

        identity = _stat_identity(final_stat)
        if identity is None:
            raise WorkspaceError(
                "unsupported_platform",
                f"stable file identity is unavailable: {relative.as_posix()}",
            )
        return FileSnapshot(
            relative=relative.as_posix(),
            data=data,
            sha256=sha256(data).hexdigest(),
            mode=stat.S_IMODE(final_stat.st_mode),
            identity=identity,
        )

    def commit_bytes(self, snapshot: FileSnapshot, new_data: bytes) -> WriteReceipt:
        """Atomically replace an unchanged snapshot, or create an unchanged missing target.

        The digest is revalidated immediately before the namespace operation. Portable
        filesystems do not provide a path-based compare-and-swap, so an uncooperative local
        writer can still race in the final validation-to-rename window; model-controlled paths
        and races injected before final validation remain fail-closed.
        """
        if not isinstance(new_data, bytes):
            raise TypeError("new_data must be bytes")
        self._validate_mutation_snapshot(snapshot)
        self._require_snapshot_current(snapshot)
        _, target = self._mutation_location(snapshot.relative)

        try:
            if os.name == "nt":
                durability_uncertain, after_identity = self._commit_bytes_windows(
                    snapshot, target, new_data
                )
            else:  # pragma: no cover - exercised by the POSIX CI job.
                durability_uncertain, after_identity = self._commit_bytes_posix(
                    snapshot, target, new_data
                )
        except WorkspaceError:
            raise
        except FileExistsError as exc:
            raise _mutation_conflict(
                snapshot.relative,
                expected_sha256=snapshot.sha256,
                current_sha256=None,
                message=f"target appeared before commit: {snapshot.relative}",
            ) from exc
        except FileNotFoundError as exc:
            raise _mutation_conflict(
                snapshot.relative,
                expected_sha256=snapshot.sha256,
                current_sha256=None,
                message=f"target changed before commit: {snapshot.relative}",
            ) from exc
        except PermissionError as exc:
            raise WorkspaceError(
                "permission_denied", f"permission denied: {snapshot.relative}"
            ) from exc
        except OSError as exc:
            raise WorkspaceError("io_error", f"cannot write file: {snapshot.relative}") from exc

        return WriteReceipt(
            relative=snapshot.relative,
            before_sha256=snapshot.sha256,
            after_sha256=sha256(new_data).hexdigest(),
            bytes_written=len(new_data),
            created=snapshot.data is None,
            durability_uncertain=durability_uncertain,
            after_identity=after_identity,
        )

    def remove_if_unchanged(
        self,
        relative: str,
        expected_sha256: str,
        *,
        expected_identity: FileIdentity | None = None,
    ) -> None:
        """Remove one ordinary file only when its current bytes match the expected digest."""
        if not _is_sha256(expected_sha256):
            raise ValueError("expected_sha256 must be a lowercase 64-character SHA-256 digest")

        current = self._snapshot_at_current_size(relative)
        if (
            current.data is None
            or not secrets.compare_digest(current.sha256 or "", expected_sha256)
            or (expected_identity is not None and current.identity != expected_identity)
        ):
            raise _mutation_conflict(
                current.relative,
                expected_sha256=expected_sha256,
                current_sha256=current.sha256,
                message=f"file changed before removal: {current.relative}",
            )
        _, target = self._mutation_location(current.relative)

        try:
            if os.name == "nt":
                parent_identity = _path_identity(target.parent)
                self._before_mutation_commit("remove", target)
                self._require_version_current(
                    current.relative, expected_sha256, expected_identity=expected_identity
                )
                if _path_identity(target.parent) != parent_identity:
                    raise WorkspaceError(
                        "write_conflict", f"parent changed before removal: {current.relative}"
                    )
                target.unlink()
            else:  # pragma: no cover - exercised by the POSIX CI job.
                parent_descriptor = _open_posix_directory(self._root, target.parent)
                try:
                    parent_identity = _descriptor_identity(parent_descriptor)
                    self._before_mutation_commit("remove", target)
                    self._require_version_current(
                        current.relative,
                        expected_sha256,
                        expected_identity=expected_identity,
                    )
                    if _path_identity(target.parent) != parent_identity:
                        raise WorkspaceError(
                            "write_conflict",
                            f"parent changed before removal: {current.relative}",
                        )
                    os.unlink(target.name, dir_fd=parent_descriptor)
                    with suppress(OSError):
                        # The namespace change has happened. Reporting a normal failure would
                        # incorrectly invite a retry against a file that is already gone.
                        os.fsync(parent_descriptor)
                finally:
                    with suppress(OSError):
                        os.close(parent_descriptor)
        except WorkspaceError:
            raise
        except FileNotFoundError as exc:
            raise _mutation_conflict(
                current.relative,
                expected_sha256=expected_sha256,
                current_sha256=None,
                message=f"target changed before removal: {current.relative}",
            ) from exc
        except PermissionError as exc:
            raise WorkspaceError(
                "permission_denied", f"permission denied: {current.relative}"
            ) from exc
        except OSError as exc:
            raise WorkspaceError("io_error", f"cannot remove file: {current.relative}") from exc

    def _mutation_location(self, user_path: str) -> tuple[PurePosixPath, Path]:
        relative = self._normalize_user_path(user_path)
        if relative == PurePosixPath("."):
            raise WorkspaceError("invalid_path", "the workspace root cannot be changed as a file")
        if relative.name.casefold() == ".gitignore":
            raise WorkspaceError("path_ignored", "repository ignore-policy files cannot be changed")
        if self._is_hard_excluded(relative):
            raise WorkspaceError("path_ignored", f"path is ignored: {relative.as_posix()}")
        if self._is_sensitive(relative):
            raise WorkspaceError(
                "sensitive_path", f"sensitive files cannot be changed: {relative.as_posix()}"
            )
        if self._matches_gitignore(relative, is_directory=False):
            raise WorkspaceError("path_ignored", f"path is ignored: {relative.as_posix()}")

        parent_relative = relative.parent
        parent = self.resolve(parent_relative.as_posix(), expected="directory")
        lexical = parent.path / relative.name
        if not self.contains(parent.path) or not self.contains(lexical):
            raise WorkspaceError(
                "path_outside_workspace",
                f"path resolves outside the workspace: {relative.as_posix()}",
            )
        return relative, lexical

    @staticmethod
    def _validate_mutation_snapshot(snapshot: FileSnapshot) -> None:
        if not isinstance(snapshot, FileSnapshot):
            raise TypeError("snapshot must be a FileSnapshot")
        if snapshot.data is None:
            if (
                snapshot.sha256 is not None
                or snapshot.mode is not None
                or snapshot.identity is not None
            ):
                raise WorkspaceError("invalid_snapshot", "missing snapshots cannot contain state")
            return
        if snapshot.sha256 is None or snapshot.mode is None or snapshot.identity is None:
            raise WorkspaceError(
                "invalid_snapshot", "existing snapshots require digest, mode, and identity"
            )
        digest = sha256(snapshot.data).hexdigest()
        if not secrets.compare_digest(digest, snapshot.sha256):
            raise WorkspaceError("invalid_snapshot", "snapshot content does not match its digest")

    def _snapshot_at_current_size(self, relative: str) -> FileSnapshot:
        _, target = self._mutation_location(relative)
        try:
            size = target.lstat().st_size
        except FileNotFoundError:
            size = 0
        except PermissionError as exc:
            raise WorkspaceError("permission_denied", f"permission denied: {relative}") from exc
        except OSError as exc:
            raise WorkspaceError("io_error", f"cannot inspect file: {relative}") from exc
        return self.snapshot_for_write(relative, max_bytes=max(1, size))

    def _require_snapshot_current(self, snapshot: FileSnapshot) -> None:
        try:
            current = self.snapshot_for_write(
                snapshot.relative,
                max_bytes=max(1, len(snapshot.data or b"")),
            )
        except WorkspaceError as exc:
            if exc.code == "file_too_large":
                raise _mutation_conflict(
                    snapshot.relative,
                    expected_sha256=snapshot.sha256,
                    current_sha256=None,
                    message=f"file changed before commit: {snapshot.relative}",
                ) from exc
            raise

        if snapshot.data is None:
            matches = current.data is None
        else:
            matches = (
                current.data is not None
                and current.mode == snapshot.mode
                and current.identity == snapshot.identity
                and current.sha256 is not None
                and snapshot.sha256 is not None
                and secrets.compare_digest(current.sha256, snapshot.sha256)
            )
        if not matches:
            raise _mutation_conflict(
                snapshot.relative,
                expected_sha256=snapshot.sha256,
                current_sha256=current.sha256,
                message=f"file changed before commit: {snapshot.relative}",
                expected_mode=snapshot.mode,
                current_mode=current.mode,
            )

    def _require_version_current(
        self,
        relative: str,
        expected_sha256: str,
        *,
        expected_identity: FileIdentity | None,
    ) -> None:
        try:
            current = self._snapshot_at_current_size(relative)
        except WorkspaceError as exc:
            if exc.code in {"file_too_large", "not_found"}:
                raise WorkspaceError(
                    "write_conflict", f"file changed before removal: {relative}"
                ) from exc
            raise
        if (
            current.data is None
            or not secrets.compare_digest(current.sha256 or "", expected_sha256)
            or (expected_identity is not None and current.identity != expected_identity)
        ):
            raise _mutation_conflict(
                relative,
                expected_sha256=expected_sha256,
                current_sha256=current.sha256,
                message=f"file changed before removal: {relative}",
            )

    def _commit_bytes_windows(
        self,
        snapshot: FileSnapshot,
        target: Path,
        new_data: bytes,
    ) -> tuple[bool, FileIdentity]:
        """Best-effort Windows atomic replace on a stable, non-reparse directory tree."""
        parent_identity = _path_identity(target.parent)
        temporary_path, descriptor = _open_windows_temporary(target.parent)
        temporary_exists = True
        try:
            try:
                _write_all(descriptor, new_data)
                os.fsync(descriptor)
                written_identity = _required_descriptor_identity(descriptor)
            finally:
                os.close(descriptor)
            self._before_mutation_commit("write", target)
            self._require_snapshot_current(snapshot)
            if _path_identity(target.parent) != parent_identity:
                raise WorkspaceError(
                    "write_conflict", f"parent changed before commit: {snapshot.relative}"
                )

            durability_uncertain = False
            if snapshot.data is None:
                # Unlike POSIX rename(), Windows rename is a no-clobber operation.
                os.rename(temporary_path, target)
            else:
                # os.replace() installs the temporary file's security descriptor and
                # attributes. ReplaceFileW instead carries the replaced file's DACL and
                # filesystem metadata forward while retaining an atomic namespace swap.
                backup_path = _unused_windows_backup_path(target.parent)
                # The helper now owns cleanup or preservation of both reserved paths.
                # This matters because ReplaceFileW can report a partial namespace move.
                temporary_exists = False
                durability_uncertain = _replace_file_windows(
                    target,
                    temporary_path,
                    backup_path,
                )
            temporary_exists = False
            return durability_uncertain, written_identity
        finally:
            if temporary_exists:
                _best_effort_unlink(temporary_path)

    def _commit_bytes_posix(
        self,
        snapshot: FileSnapshot,
        target: Path,
        new_data: bytes,
    ) -> tuple[bool, FileIdentity]:  # pragma: no cover - exercised by the POSIX CI job.
        parent_descriptor = _open_posix_directory(self._root, target.parent)
        try:
            temporary_name, descriptor = _open_posix_temporary(parent_descriptor)
        except BaseException:
            with suppress(OSError):
                os.close(parent_descriptor)
            raise
        temporary_exists = True
        durability_uncertain = False
        try:
            try:
                _write_all(descriptor, new_data)
                if snapshot.mode is not None:
                    os.chmod(descriptor, snapshot.mode & 0o777)
                os.fsync(descriptor)
                written_identity = _required_descriptor_identity(descriptor)
            finally:
                os.close(descriptor)

            parent_identity = _descriptor_identity(parent_descriptor)
            self._before_mutation_commit("write", target)
            self._require_snapshot_current(snapshot)
            if _path_identity(target.parent) != parent_identity:
                raise WorkspaceError(
                    "write_conflict", f"parent changed before commit: {snapshot.relative}"
                )

            if snapshot.data is None:
                os.link(
                    temporary_name,
                    target.name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                try:
                    os.unlink(temporary_name, dir_fd=parent_descriptor)
                    temporary_exists = False
                except OSError:
                    # The target is already committed. The finally block retries cleanup and
                    # the reserved temporary prefix remains inaccessible to tools meanwhile.
                    durability_uncertain = True
            else:
                os.replace(
                    temporary_name,
                    target.name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                )
                temporary_exists = False

            try:
                os.fsync(parent_descriptor)
            except OSError:
                durability_uncertain = True
            return durability_uncertain, written_identity
        finally:
            if temporary_exists:
                with suppress(OSError):
                    os.unlink(temporary_name, dir_fd=parent_descriptor)
            with suppress(OSError):
                os.close(parent_descriptor)

    def _before_mutation_commit(self, operation: str, path: Path) -> None:
        """Test seam immediately before final revalidation and namespace mutation."""
        del operation, path

    def _read_approved_file(
        self,
        path: Path,
        *,
        max_bytes: int,
        display_path: str,
        too_large_code: str,
    ) -> bytes:
        try:
            descriptor, anchored_descriptor = self._open_approved_descriptor(path)
        except FileNotFoundError as exc:
            raise WorkspaceError("not_found", f"path not found: {display_path}") from exc
        except PermissionError as exc:
            raise WorkspaceError("permission_denied", f"permission denied: {display_path}") from exc
        except OSError as exc:
            raise WorkspaceError("io_error", f"cannot open file: {display_path}") from exc

        try:
            opened_path = _opened_file_path(descriptor)
            opened_stat = os.fstat(descriptor)
            if not stat.S_ISREG(opened_stat.st_mode):
                raise WorkspaceError("not_file", f"path is not a regular file: {display_path}")
            if opened_stat.st_nlink > 1:
                raise WorkspaceError(
                    "unsafe_file_link",
                    f"files with multiple hard links cannot be read: {display_path}",
                )

            if opened_path is not None:
                try:
                    final_path = opened_path.resolve(strict=True)
                except (OSError, RuntimeError) as exc:
                    raise WorkspaceError(
                        "io_error", f"cannot verify opened file: {display_path}"
                    ) from exc
                if final_path != path or not self.contains(final_path):
                    raise WorkspaceError(
                        "path_outside_workspace",
                        f"opened file changed after authorization: {display_path}",
                    )
            elif not anchored_descriptor:
                raise WorkspaceError(
                    "unsupported_platform",
                    "opened file handle cannot be verified on this platform",
                )

            if opened_stat.st_size > max_bytes:
                raise WorkspaceError(
                    too_large_code,
                    f"file exceeds the {max_bytes}-byte limit: {display_path}",
                )

            chunks: list[bytes] = []
            remaining = max_bytes + 1
            while remaining:
                chunk = os.read(descriptor, min(remaining, 64 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            if len(data) > max_bytes:
                raise WorkspaceError(
                    too_large_code,
                    f"file grew beyond the {max_bytes}-byte limit: {display_path}",
                )
            return data
        except PermissionError as exc:
            raise WorkspaceError("permission_denied", f"permission denied: {display_path}") from exc
        except OSError as exc:
            raise WorkspaceError("io_error", f"cannot read file: {display_path}") from exc
        finally:
            os.close(descriptor)

    def _open_approved_descriptor(self, path: Path) -> tuple[int, bool]:
        """Open a file without allowing path-component link swaps on supported platforms."""
        if os.name != "nt":  # pragma: no cover - exercised on POSIX, not Windows.
            return _open_posix_descriptor(self._root, path), True
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
        return os.open(path, flags), False

    def _ignore_spec_for(self, directory: PurePosixPath) -> GitIgnoreSpec:
        cache_key = directory.as_posix()
        ignore_file = self._root.joinpath(*directory.parts) / ".gitignore"
        signature = self._ignore_signature(ignore_file)
        cached = self._ignore_cache.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[1]

        spec = self._load_gitignore(ignore_file)
        self._ignore_cache[cache_key] = (signature, spec)
        return spec

    def _load_gitignore(self, ignore_file: Path) -> GitIgnoreSpec:
        try:
            if self._is_link_like(ignore_file):
                raise WorkspaceError(
                    "invalid_workspace", ".gitignore files must not be filesystem links"
                )
            if not ignore_file.is_file():
                return GitIgnoreSpec.from_lines((), backend="simple")
            resolved_ignore_file = ignore_file.resolve(strict=True)
            if not self.contains(resolved_ignore_file):
                raise WorkspaceError("invalid_workspace", ".gitignore resolves outside workspace")
            data = self._read_approved_file(
                resolved_ignore_file,
                max_bytes=_MAX_GITIGNORE_BYTES,
                display_path=".gitignore",
                too_large_code="invalid_workspace",
            )
            if b"\x00" in data:
                raise WorkspaceError("invalid_workspace", ".gitignore contains binary data")
            lines = data.decode("utf-8-sig").splitlines()
            if len(lines) > _MAX_GITIGNORE_PATTERNS or any(
                len(line) > _MAX_GITIGNORE_LINE_CHARS for line in lines
            ):
                raise WorkspaceError(
                    "invalid_workspace", ".gitignore exceeds the pattern complexity limit"
                )
            policy_lines = [line.casefold() for line in lines] if os.name == "nt" else lines
            return GitIgnoreSpec.from_lines(policy_lines, backend="simple")
        except WorkspaceError:
            raise
        except (OSError, UnicodeError, ValueError) as exc:
            raise WorkspaceError(
                "invalid_workspace", ".gitignore could not be parsed as UTF-8"
            ) from exc

    @staticmethod
    def _ignore_signature(ignore_file: Path) -> IgnoreSignature:
        try:
            file_stat = ignore_file.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise WorkspaceError(
                "invalid_workspace", ".gitignore metadata is inaccessible"
            ) from exc
        return (
            file_stat.st_mode,
            file_stat.st_ino,
            file_stat.st_size,
            file_stat.st_mtime_ns,
            file_stat.st_ctime_ns,
        )

    def _normalize_user_path(self, user_path: str) -> PurePosixPath:
        if (
            not user_path
            or user_path.isspace()
            or any(_is_display_control(character) for character in user_path)
        ):
            raise WorkspaceError("invalid_path", "path must be a non-empty relative path")

        windows_path = PureWindowsPath(user_path)
        portable = user_path.replace("\\", "/")
        posix_path = PurePosixPath(portable)
        if windows_path.drive or windows_path.root or posix_path.is_absolute():
            raise WorkspaceError(
                "invalid_path", "absolute and drive-qualified paths are not allowed"
            )

        normalized_parts: list[str] = []
        for part in portable.split("/"):
            if not part or part == ".":
                continue
            if part == "..":
                raise WorkspaceError("invalid_path", "parent path segments are not allowed")
            if ":" in part:
                raise WorkspaceError("invalid_path", "colon characters are not allowed in paths")
            if any(character in '<>"|?*' for character in part):
                raise WorkspaceError(
                    "invalid_path", "portable paths cannot contain shell wildcards"
                )
            if part.endswith((" ", ".")):
                raise WorkspaceError(
                    "invalid_path", "path components cannot end with a space or dot"
                )
            reserved_candidate = part.split(".", maxsplit=1)[0].casefold()
            if reserved_candidate in _WINDOWS_RESERVED_NAMES:
                raise WorkspaceError("invalid_path", "reserved device names are not allowed")
            normalized_parts.append(part)

        if not normalized_parts:
            return PurePosixPath(".")
        return PurePosixPath(*normalized_parts)

    def _contains_directory_link(self, relative: PurePosixPath) -> bool:
        current = self._root
        for part in relative.parts:
            current /= part
            if self._is_link_like(current) and current.is_dir():
                return True
        return False

    def _enforce_policy(
        self,
        relative: PurePosixPath,
        resolved: Path,
        *,
        allow_ignored: bool,
        allow_sensitive: bool,
    ) -> None:
        resolved_relative = self._resolved_relative(resolved)
        is_directory = resolved.is_dir()
        if self._is_hard_excluded(relative) or self._is_hard_excluded(resolved_relative):
            raise WorkspaceError("path_ignored", f"path is ignored: {relative.as_posix()}")
        if not allow_ignored and (
            self._matches_gitignore(relative, is_directory=is_directory)
            or self._matches_gitignore(resolved_relative, is_directory=is_directory)
        ):
            raise WorkspaceError("path_ignored", f"path is ignored: {relative.as_posix()}")
        if not allow_sensitive and (
            self._is_sensitive(relative) or self._is_sensitive(resolved_relative)
        ):
            raise WorkspaceError(
                "sensitive_path", f"sensitive files cannot be read: {relative.as_posix()}"
            )

    def _is_excluded(
        self,
        relative: PurePosixPath,
        resolved: Path,
        *,
        is_directory: bool,
    ) -> bool:
        resolved_relative = self._resolved_relative(resolved)
        return (
            self._is_hard_excluded(relative)
            or self._is_hard_excluded(resolved_relative)
            or self._matches_gitignore(relative, is_directory=is_directory)
            or self._matches_gitignore(resolved_relative, is_directory=is_directory)
        )

    @staticmethod
    def _is_hard_excluded(relative: PurePosixPath) -> bool:
        return any(
            part.casefold() in _HARD_EXCLUDED_PARTS
            or any(
                part.casefold().startswith(prefix.casefold()) for prefix in _HARD_EXCLUDED_PREFIXES
            )
            for part in relative.parts
        )

    @staticmethod
    def _is_sensitive(relative: PurePosixPath) -> bool:
        for index, part in enumerate(relative.parts):
            lowered = part.casefold()
            is_example_file = lowered == ".env.example" and index == len(relative.parts) - 1
            if lowered == ".env" or (lowered.startswith(".env.") and not is_example_file):
                return True
            if lowered in _SENSITIVE_FILENAMES or Path(lowered).suffix in _SENSITIVE_SUFFIXES:
                return True
        return False

    def _matches_gitignore(self, relative: PurePosixPath, *, is_directory: bool) -> bool:
        if relative == PurePosixPath("."):
            return False

        inherited_specs: list[tuple[PurePosixPath, GitIgnoreSpec]] = []
        ignored = False
        for directory in self._ignore_directories(relative):
            if directory != PurePosixPath(".") and self._matches_ignore_specs(
                directory,
                is_directory=True,
                specs=inherited_specs,
            ):
                # Git never descends into an excluded parent, so rules inside that directory
                # cannot re-include one of its children.
                return True
            spec = self._ignore_spec_for(directory)
            inherited_specs.append((directory, spec))
            candidate = relative.relative_to(directory).as_posix()
            if is_directory:
                candidate += "/"
            if os.name == "nt":
                candidate = candidate.casefold()
            result = spec.check_file(candidate)
            if result.include is not None:
                ignored = result.include
        return ignored

    @staticmethod
    def _matches_ignore_specs(
        relative: PurePosixPath,
        *,
        is_directory: bool,
        specs: list[tuple[PurePosixPath, GitIgnoreSpec]],
    ) -> bool:
        ignored = False
        for base, spec in specs:
            candidate = relative.relative_to(base).as_posix()
            if is_directory:
                candidate += "/"
            if os.name == "nt":
                candidate = candidate.casefold()
            result = spec.check_file(candidate)
            if result.include is not None:
                ignored = result.include
        return ignored

    @staticmethod
    def _ignore_directories(relative: PurePosixPath) -> tuple[PurePosixPath, ...]:
        directories = [PurePosixPath(".")]
        current = PurePosixPath(".")
        for part in relative.parent.parts:
            current /= part
            directories.append(current)
        return tuple(directories)

    def _resolved_relative(self, resolved: Path) -> PurePosixPath:
        return PurePosixPath(resolved.relative_to(self._root).as_posix())

    @staticmethod
    def _join_relative(parent: str, child: str) -> str:
        if parent == ".":
            return PurePosixPath(child).as_posix()
        return (PurePosixPath(parent) / child).as_posix()

    @staticmethod
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


def _open_posix_descriptor(root: Path, path: Path) -> int:  # pragma: no cover
    """Anchor every POSIX path component to a no-follow directory descriptor."""
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
    if os.open not in os.supports_dir_fd or any(not hasattr(os, name) for name in required_flags):
        raise WorkspaceError(
            "unsupported_platform",
            "secure workspace file opening is unavailable on this platform",
        )

    relative = path.relative_to(root)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    no_follow_flag = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | directory_flag | no_follow_flag | getattr(os, "O_CLOEXEC", 0)
    file_flags = (
        os.O_RDONLY | no_follow_flag | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    directory_descriptor = os.open(root, directory_flags)
    try:
        for part in relative.parts[:-1]:
            next_descriptor = os.open(part, directory_flags, dir_fd=directory_descriptor)
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        return os.open(relative.name, file_flags, dir_fd=directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _opened_file_path(descriptor: int) -> Path | None:
    """Return the physical path bound to an open descriptor when the OS exposes it."""
    if os.name != "nt":  # pragma: no cover - POSIX safety comes from openat anchoring.
        return None

    import ctypes
    import msvcrt
    from ctypes import wintypes

    get_final_path = ctypes.WinDLL("kernel32", use_last_error=True).GetFinalPathNameByHandleW
    get_final_path.argtypes = [
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    get_final_path.restype = wintypes.DWORD
    buffer = ctypes.create_unicode_buffer(32_768)
    length = get_final_path(msvcrt.get_osfhandle(descriptor), buffer, len(buffer), 0)
    if length == 0 or length >= len(buffer):
        raise WorkspaceError("io_error", "cannot verify opened file handle")
    value = buffer.value
    unc_prefix = "\\\\?\\UNC\\"
    device_prefix = "\\\\?\\"
    if value.startswith(unc_prefix):
        value = "\\\\" + value[len(unc_prefix) :]
    elif value.startswith(device_prefix):
        value = value[len(device_prefix) :]
    return Path(value)


def _require_no_windows_named_streams(path: Path, display_path: str) -> None:
    """Reject NTFS streams that the byte-only journal cannot faithfully restore."""
    if os.name != "nt":  # pragma: no cover - Windows-specific filesystem metadata.
        return

    import ctypes
    from ctypes import wintypes

    class _StreamData(ctypes.Structure):
        _fields_ = [
            ("stream_size", ctypes.c_longlong),
            ("stream_name", wintypes.WCHAR * 296),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    find_first = kernel32.FindFirstStreamW
    find_first.argtypes = [wintypes.LPCWSTR, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    find_first.restype = wintypes.HANDLE
    find_next = kernel32.FindNextStreamW
    find_next.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    find_next.restype = wintypes.BOOL
    find_close = kernel32.FindClose
    find_close.argtypes = [wintypes.HANDLE]
    find_close.restype = wintypes.BOOL

    data = _StreamData()
    handle = find_first(str(path), 0, ctypes.byref(data), 0)
    invalid_handle = ctypes.c_void_p(-1).value
    if handle == invalid_handle:
        error = ctypes.get_last_error()
        if error in {2, 38}:  # No stream enumeration support or no streams.
            return
        raise WorkspaceError("io_error", f"cannot inspect file streams: {display_path}")

    has_named_stream = False
    try:
        while True:
            if data.stream_name.casefold() != "::$data":
                has_named_stream = True
                break
            if find_next(handle, ctypes.byref(data)):
                continue
            error = ctypes.get_last_error()
            if error != 38:  # ERROR_HANDLE_EOF
                raise WorkspaceError("io_error", f"cannot inspect file streams: {display_path}")
            break
    finally:
        find_close(handle)

    if has_named_stream:
        raise WorkspaceError(
            "unsafe_file_stream",
            f"files with named data streams cannot be changed: {display_path}",
        )


def _open_posix_directory(root: Path, directory: Path) -> int:  # pragma: no cover
    """Open an in-workspace directory through root-anchored, no-follow descriptors."""
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
    if os.open not in os.supports_dir_fd or any(not hasattr(os, name) for name in required_flags):
        raise WorkspaceError(
            "unsupported_platform",
            "secure workspace directory opening is unavailable on this platform",
        )

    try:
        relative = directory.relative_to(root)
    except ValueError as exc:
        raise WorkspaceError(
            "path_outside_workspace", "mutation parent is outside the workspace"
        ) from exc

    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(root, flags)
    try:
        for part in relative.parts:
            next_descriptor = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_posix_temporary(parent_descriptor: int) -> tuple[str, int]:  # pragma: no cover
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    for _ in range(32):
        name = f"{_HARD_EXCLUDED_PREFIXES[0]}{secrets.token_hex(16)}"
        try:
            return name, os.open(name, flags, 0o666, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
    raise WorkspaceError("io_error", "cannot allocate a unique mutation temporary file")


def _open_windows_temporary(parent: Path) -> tuple[Path, int]:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOINHERIT", 0)
    )
    for _ in range(32):
        path = parent / f"{_HARD_EXCLUDED_PREFIXES[0]}{secrets.token_hex(16)}"
        try:
            return path, os.open(path, flags, 0o666)
        except FileExistsError:
            continue
    raise WorkspaceError("io_error", "cannot allocate a unique mutation temporary file")


def _unused_windows_backup_path(parent: Path) -> Path:
    """Choose a reserved same-volume backup name without creating the file."""
    for _ in range(32):
        path = parent / f"{_HARD_EXCLUDED_PREFIXES[0]}backup-{secrets.token_hex(16)}"
        try:
            path.lstat()
        except FileNotFoundError:
            return path
    raise WorkspaceError("io_error", "cannot allocate a unique mutation backup name")


def _replace_file_windows(target: Path, replacement: Path, backup: Path) -> bool:
    """Replace a Windows file and recover the documented partial-failure state.

    Once called, this helper owns both reserved paths. A successful replacement returns
    whether the old-file backup could not be removed. On failure it either restores the
    original namespace and removes the replacement, or keeps every ambiguous artifact and
    reports a manual-recovery error instead of risking data loss.
    """
    if os.name != "nt":  # pragma: no cover - only called by the Windows commit path.
        raise WorkspaceError("unsupported_platform", "ReplaceFileW requires Windows")

    error = _invoke_replace_file_windows(target, replacement, backup)
    if error is None:
        return not _best_effort_unlink(backup)

    if error == _ERROR_UNABLE_TO_MOVE_REPLACEMENT_2:
        if not _path_entry_exists(backup):
            raise _windows_recovery_error(
                target,
                replacement,
                backup,
                error,
                "the original-file backup could not be located",
            )
        if _path_entry_exists(target):
            _best_effort_unlink(replacement)
            raise _windows_recovery_error(
                target,
                replacement,
                backup,
                error,
                "another entry occupies the target while the original remains in backup",
            )
        try:
            # Windows os.rename is no-clobber, so an unexpected creator wins safely.
            os.rename(backup, target)
        except OSError as exc:
            _best_effort_unlink(replacement)
            raise _windows_recovery_error(
                target,
                replacement,
                backup,
                error,
                "the original file could not be restored automatically",
            ) from exc

    if _path_entry_exists(backup):
        # Outside error 1177 Microsoft documents no backup artifact. Preserve any one we do
        # observe, because guessing which copy is authoritative would turn uncertainty into
        # data loss.
        _best_effort_unlink(replacement)
        raise _windows_recovery_error(
            target,
            replacement,
            backup,
            error,
            "an unexpected backup remains after replacement failed",
        )

    if not _best_effort_unlink(replacement):
        raise _windows_recovery_error(
            target,
            replacement,
            backup,
            error,
            "the uncommitted replacement could not be removed",
        )
    raise OSError(error, "ReplaceFileW failed after preserving the original file")


def _invoke_replace_file_windows(target: Path, replacement: Path, backup: Path) -> int | None:
    """Return None on success or the native ReplaceFileW error code on failure."""
    if os.name != "nt":  # pragma: no cover - only called by the Windows commit path.
        raise WorkspaceError("unsupported_platform", "ReplaceFileW requires Windows")

    import ctypes
    from ctypes import wintypes

    replace_file = ctypes.WinDLL("kernel32", use_last_error=True).ReplaceFileW
    replace_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.LPVOID,
    ]
    replace_file.restype = wintypes.BOOL
    if replace_file(str(target), str(replacement), str(backup), 0, None, None):
        return None
    return ctypes.get_last_error()


def _windows_recovery_error(
    target: Path,
    replacement: Path,
    backup: Path,
    error: int,
    detail: str,
) -> WorkspaceError:
    return WorkspaceError(
        "write_recovery_required",
        f"Windows replacement needs manual recovery for {target.name}: {detail}",
        metadata={
            "path": target.name,
            "backup_name": backup.name,
            "replacement_name": replacement.name,
            "windows_error": error,
            "recovery": "inspect_reserved_files_before_retry",
        },
    )


def _path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        # An inaccessible entry must be treated as present for data-preserving cleanup.
        return True
    return True


def _write_all(descriptor: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        try:
            written = os.write(descriptor, remaining)
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError("file write made no progress")
        remaining = remaining[written:]


def _path_identity(path: Path) -> tuple[int, int] | Path:
    path_stat = path.stat()
    if path_stat.st_ino:
        return path_stat.st_dev, path_stat.st_ino
    return path.resolve(strict=True)


def _descriptor_identity(descriptor: int) -> tuple[int, int]:
    descriptor_stat = os.fstat(descriptor)
    return descriptor_stat.st_dev, descriptor_stat.st_ino


def _required_descriptor_identity(descriptor: int) -> FileIdentity:
    identity = _stat_identity(os.fstat(descriptor))
    if identity is None:
        raise WorkspaceError("unsupported_platform", "stable file identity is unavailable")
    return identity


def _stat_identity(path_stat: os.stat_result) -> FileIdentity | None:
    if not path_stat.st_ino:
        return None
    return path_stat.st_dev, path_stat.st_ino


def _best_effort_unlink(path: Path) -> bool:
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return True
    except PermissionError:
        try:
            os.chmod(path, stat.S_IWRITE)
            path.unlink()
            return True
        except OSError:
            return False
    except OSError:
        return False


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _mutation_conflict(
    relative: str,
    *,
    expected_sha256: str | None,
    current_sha256: str | None,
    message: str,
    expected_mode: int | None = None,
    current_mode: int | None = None,
) -> WorkspaceError:
    metadata: dict[str, str | int | None] = {
        "path": relative,
        "expected_sha256": expected_sha256,
        "current_sha256": current_sha256,
        "recovery": "read_file_then_retry",
    }
    if expected_mode != current_mode:
        metadata["expected_mode"] = expected_mode
        metadata["current_mode"] = current_mode
    return WorkspaceError("write_conflict", message, metadata=metadata)


def _is_display_control(character: str) -> bool:
    return category(character) in {"Cc", "Cf", "Zl", "Zp"}
