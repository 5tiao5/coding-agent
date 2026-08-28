"""Small cross-platform filesystem primitives used by workspace backends."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from coding_agent._workspace.contracts import FileIdentity, WorkspaceError

RESERVED_TEMP_PREFIX = ".coding-agent-tmp-"


def is_link_like(path: Path) -> bool:
    """Report symbolic links and Windows reparse points without following them."""
    if path.is_symlink():
        return True
    if os.name != "nt":
        return False
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def path_identity(path: Path) -> FileIdentity | Path:
    path_stat = path.stat()
    if path_stat.st_ino:
        return path_stat.st_dev, path_stat.st_ino
    return path.resolve(strict=True)


def descriptor_identity(descriptor: int) -> FileIdentity:
    descriptor_stat = os.fstat(descriptor)
    return descriptor_stat.st_dev, descriptor_stat.st_ino


def required_descriptor_identity(descriptor: int) -> FileIdentity:
    identity = stat_identity(os.fstat(descriptor))
    if identity is None:
        raise WorkspaceError("unsupported_platform", "stable file identity is unavailable")
    return identity


def stat_identity(path_stat: os.stat_result) -> FileIdentity | None:
    if not path_stat.st_ino:
        return None
    return path_stat.st_dev, path_stat.st_ino


def write_all(descriptor: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        try:
            written = os.write(descriptor, remaining)
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError("file write made no progress")
        remaining = remaining[written:]


def path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        # An inaccessible entry must be treated as present for data-preserving cleanup.
        return True
    return True


def best_effort_unlink(path: Path) -> bool:
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
