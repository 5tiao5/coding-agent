"""Root-anchored POSIX file opening and atomic workspace replacement."""

from __future__ import annotations

import os
import secrets
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

from coding_agent._workspace.contracts import FileIdentity, FileSnapshot, WorkspaceError
from coding_agent._workspace.native import (
    RESERVED_TEMP_PREFIX,
    descriptor_identity,
    path_identity,
    required_descriptor_identity,
    write_all,
)

BeforeMutationCommit = Callable[[str, Path], None]
RequireSnapshotCurrent = Callable[[FileSnapshot], None]


def open_descriptor(root: Path, path: Path) -> int:  # pragma: no cover
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


def open_directory(root: Path, directory: Path) -> int:  # pragma: no cover
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


def commit_bytes(
    root: Path,
    snapshot: FileSnapshot,
    target: Path,
    new_data: bytes,
    *,
    before_mutation_commit: BeforeMutationCommit,
    require_snapshot_current: RequireSnapshotCurrent,
) -> tuple[bool, FileIdentity]:  # pragma: no cover - exercised by the POSIX CI job.
    """Commit bytes with the same fd-anchored ordering as the public workspace facade."""
    parent_descriptor = open_directory(root, target.parent)
    try:
        temporary_name, descriptor = _open_temporary(parent_descriptor)
    except BaseException:
        with suppress(OSError):
            os.close(parent_descriptor)
        raise
    temporary_exists = True
    durability_uncertain = False
    try:
        try:
            write_all(descriptor, new_data)
            if snapshot.mode is not None:
                os.chmod(descriptor, snapshot.mode & 0o777)
            os.fsync(descriptor)
            written_identity = required_descriptor_identity(descriptor)
        finally:
            os.close(descriptor)

        parent_identity = descriptor_identity(parent_descriptor)
        before_mutation_commit("write", target)
        require_snapshot_current(snapshot)
        if path_identity(target.parent) != parent_identity:
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


def _open_temporary(parent_descriptor: int) -> tuple[str, int]:  # pragma: no cover
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    for _ in range(32):
        name = f"{RESERVED_TEMP_PREFIX}{secrets.token_hex(16)}"
        try:
            return name, os.open(name, flags, 0o666, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
    raise WorkspaceError("io_error", "cannot allocate a unique mutation temporary file")
