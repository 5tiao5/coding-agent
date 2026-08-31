"""Rooted filesystem boundary shared by every local repository tool."""

from __future__ import annotations

import os
import secrets
import stat
from contextlib import suppress
from hashlib import sha256
from itertools import islice
from pathlib import Path, PurePosixPath

from coding_agent._workspace import native, posix, windows
from coding_agent._workspace.contracts import (
    DirectoryReceipt,
    DirectoryScan,
    ExpectedPathKind,
    FileIdentity,
    FileSnapshot,
    IgnoreSignature,
    WorkspaceEntry,
    WorkspaceError,
    WorkspacePath,
    WriteReceipt,
)
from coding_agent._workspace.policy import WorkspacePolicy

__all__ = (
    "DirectoryScan",
    "DirectoryReceipt",
    "ExpectedPathKind",
    "FileIdentity",
    "FileSnapshot",
    "IgnoreSignature",
    "Workspace",
    "WorkspaceEntry",
    "WorkspaceError",
    "WorkspacePath",
    "WriteReceipt",
)

_MAX_RAW_DIRECTORY_ENTRIES = 20_000


class Workspace:
    """Resolve and enumerate paths without allowing access outside one root."""

    def __init__(self, root: str | Path) -> None:
        try:
            resolved_root = Path(root).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise WorkspaceError("invalid_workspace", "workspace root is not accessible") from exc
        if not resolved_root.is_dir():
            raise WorkspaceError("invalid_workspace", "workspace root must be a directory")
        if any(part.casefold() in {".git", ".coding-agent"} for part in resolved_root.parts):
            raise WorkspaceError("invalid_workspace", "workspace root is inside excluded metadata")

        self._root = resolved_root
        self._policy = WorkspacePolicy(resolved_root, self._read_approved_file)
        self._policy.initialize()

    @property
    def root(self) -> Path:
        return self._root

    def contains(self, path: Path) -> bool:
        """Resolve a candidate and report whether its physical path is inside the root."""
        return self._policy.contains(path)

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
        relative = self._policy.normalize_user_path(user_path)
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

        if self._policy.contains_directory_link(relative):
            raise WorkspaceError(
                "invalid_path",
                f"directory links cannot be traversed: {relative.as_posix()}",
            )

        self._policy.enforce(
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
            relative = self._policy.join_relative(directory.relative, raw_entry.name)
            try:
                normalized_relative = self._policy.normalize_user_path(relative)
            except WorkspaceError:
                skipped += 1
                continue
            if normalized_relative.as_posix() != relative:
                skipped += 1
                continue
            lexical = Path(raw_entry.path)
            link_like = native.is_link_like(lexical)

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
            if self._policy.is_excluded(relative_path, resolved, is_directory=is_directory):
                continue
            resolved_relative = self._policy.resolved_relative(resolved)
            if self._policy.is_sensitive(relative_path) or self._policy.is_sensitive(
                resolved_relative
            ):
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

        if native.is_link_like(lexical):
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
        windows.require_no_named_streams(lexical, relative.as_posix())

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
            native.is_link_like(lexical)
            or not stat.S_ISREG(final_stat.st_mode)
            or final_stat.st_nlink > 1
            or final_path != target.path
        ):
            raise WorkspaceError(
                "write_conflict", f"file changed while it was inspected: {relative.as_posix()}"
            )

        identity = native.stat_identity(final_stat)
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
                durability_uncertain, after_identity = windows.commit_bytes(
                    snapshot,
                    target,
                    new_data,
                    before_mutation_commit=self._before_mutation_commit,
                    require_snapshot_current=self._require_snapshot_current,
                    invoke_replace_file=_invoke_replace_file_windows,
                )
            else:  # pragma: no cover - exercised by the POSIX CI job.
                durability_uncertain, after_identity = posix.commit_bytes(
                    self._root,
                    snapshot,
                    target,
                    new_data,
                    before_mutation_commit=self._before_mutation_commit,
                    require_snapshot_current=self._require_snapshot_current,
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
                parent_identity = native.path_identity(target.parent)
                self._before_mutation_commit("remove", target)
                self._require_version_current(
                    current.relative, expected_sha256, expected_identity=expected_identity
                )
                if native.path_identity(target.parent) != parent_identity:
                    raise WorkspaceError(
                        "write_conflict", f"parent changed before removal: {current.relative}"
                    )
                target.unlink()
            else:  # pragma: no cover - exercised by the POSIX CI job.
                parent_descriptor = posix.open_directory(self._root, target.parent)
                try:
                    parent_identity = native.descriptor_identity(parent_descriptor)
                    self._before_mutation_commit("remove", target)
                    self._require_version_current(
                        current.relative,
                        expected_sha256,
                        expected_identity=expected_identity,
                    )
                    if native.path_identity(target.parent) != parent_identity:
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

    def create_directory(self, user_path: str) -> DirectoryReceipt:
        """Create exactly one authorized directory without creating missing parents.

        The namespace operation is no-clobber and idempotent for an existing ordinary
        directory. Parent traversal, ignored/sensitive/internal paths, and link-like targets
        reuse the same fail-closed policy as file mutation.
        """

        relative, parent, target = self._directory_creation_location(user_path)
        existing = self._inspect_directory_target(relative, target)
        if existing is not None:
            return existing

        try:
            if os.name == "nt":
                parent_identity = native.path_identity(parent.path)
                self._before_mutation_commit("mkdir", target)
                rebound = self.resolve(parent.relative, expected="directory")
                if (
                    rebound.path != parent.path
                    or native.path_identity(parent.path) != parent_identity
                ):
                    raise WorkspaceError(
                        "write_conflict",
                        f"parent changed before directory creation: {relative.as_posix()}",
                    )
                os.mkdir(target)
            else:  # pragma: no cover - exercised by the POSIX CI job.
                if os.mkdir not in os.supports_dir_fd:
                    raise WorkspaceError(
                        "unsupported_platform",
                        "secure workspace directory creation is unavailable on this platform",
                    )
                parent_descriptor = posix.open_directory(self._root, parent.path)
                try:
                    parent_identity = native.descriptor_identity(parent_descriptor)
                    self._before_mutation_commit("mkdir", target)
                    rebound = self.resolve(parent.relative, expected="directory")
                    if (
                        rebound.path != parent.path
                        or native.path_identity(parent.path) != parent_identity
                    ):
                        raise WorkspaceError(
                            "write_conflict",
                            f"parent changed before directory creation: {relative.as_posix()}",
                        )
                    os.mkdir(target.name, dir_fd=parent_descriptor)
                    with suppress(OSError):
                        os.fsync(parent_descriptor)
                finally:
                    with suppress(OSError):
                        os.close(parent_descriptor)
        except WorkspaceError:
            raise
        except FileExistsError:
            raced = self._inspect_directory_target(relative, target)
            if raced is not None:
                return raced
            raise WorkspaceError(
                "write_conflict",
                f"directory target changed during creation: {relative.as_posix()}",
            ) from None
        except FileNotFoundError as exc:
            raise WorkspaceError(
                "write_conflict",
                f"parent changed before directory creation: {relative.as_posix()}",
            ) from exc
        except PermissionError as exc:
            raise WorkspaceError(
                "permission_denied", f"permission denied: {relative.as_posix()}"
            ) from exc
        except OSError as exc:
            raise WorkspaceError(
                "io_error", f"cannot create directory: {relative.as_posix()}"
            ) from exc

        # Resolve again so a link-like or outside replacement never receives a success receipt.
        created = self.resolve(relative.as_posix(), expected="directory")
        if created.path != target:
            raise WorkspaceError(
                "write_conflict",
                f"directory changed after creation: {relative.as_posix()}",
            )
        return DirectoryReceipt(relative=created.relative, created=True)

    def _directory_creation_location(
        self,
        user_path: str,
    ) -> tuple[PurePosixPath, WorkspacePath, Path]:
        relative = self._policy.normalize_user_path(user_path)
        if relative == PurePosixPath("."):
            raise WorkspaceError("invalid_path", "the workspace root cannot be created")
        self._policy.enforce_mutation(relative, is_directory=True)
        parent = self.resolve(relative.parent.as_posix(), expected="directory")
        target = parent.path / relative.name
        if not self.contains(parent.path):  # Defensive: ``resolve`` already establishes this.
            raise WorkspaceError(
                "path_outside_workspace",
                f"path resolves outside the workspace: {relative.as_posix()}",
            )
        return relative, parent, target

    def _inspect_directory_target(
        self,
        relative: PurePosixPath,
        target: Path,
    ) -> DirectoryReceipt | None:
        try:
            target_stat = target.lstat()
        except FileNotFoundError:
            return None
        except PermissionError as exc:
            raise WorkspaceError(
                "permission_denied", f"permission denied: {relative.as_posix()}"
            ) from exc
        except OSError as exc:
            raise WorkspaceError(
                "io_error", f"cannot inspect directory: {relative.as_posix()}"
            ) from exc

        if native.is_link_like(target):
            raise WorkspaceError(
                "unsafe_directory_link",
                f"directory links cannot be created or reused: {relative.as_posix()}",
            )
        if not stat.S_ISDIR(target_stat.st_mode):
            raise WorkspaceError("not_directory", f"path is not a directory: {relative.as_posix()}")
        approved = self.resolve(relative.as_posix(), expected="directory")
        if approved.path != target:
            raise WorkspaceError(
                "write_conflict", f"directory changed while inspected: {relative.as_posix()}"
            )
        return DirectoryReceipt(relative=approved.relative, created=False)

    def _mutation_location(self, user_path: str) -> tuple[PurePosixPath, Path]:
        relative = self._policy.normalize_user_path(user_path)
        if relative == PurePosixPath("."):
            raise WorkspaceError("invalid_path", "the workspace root cannot be changed as a file")
        self._policy.enforce_mutation(relative)

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
            opened_path = windows.opened_file_path(descriptor)
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
            return posix.open_descriptor(self._root, path), True
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
        return os.open(path, flags), False

    @staticmethod
    def _is_link_like(path: Path) -> bool:
        """Compatibility seam for platform-neutral reparse-point policy tests."""
        return native.is_link_like(path)


def _invoke_replace_file_windows(target: Path, replacement: Path, backup: Path) -> int | None:
    """Compatibility seam for tests that inject native ReplaceFileW outcomes."""
    return windows.invoke_replace_file(target, replacement, backup)


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
