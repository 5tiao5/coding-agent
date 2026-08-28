"""Windows handle verification and recoverable atomic file replacement."""

from __future__ import annotations

import os
import secrets
import sys
from collections.abc import Callable
from pathlib import Path

from coding_agent._workspace.contracts import FileIdentity, FileSnapshot, WorkspaceError
from coding_agent._workspace.native import (
    RESERVED_TEMP_PREFIX,
    best_effort_unlink,
    path_entry_exists,
    path_identity,
    required_descriptor_identity,
    write_all,
)

BeforeMutationCommit = Callable[[str, Path], None]
RequireSnapshotCurrent = Callable[[FileSnapshot], None]
InvokeReplaceFile = Callable[[Path, Path, Path], int | None]

_ERROR_UNABLE_TO_MOVE_REPLACEMENT_2 = 1177


def commit_bytes(
    snapshot: FileSnapshot,
    target: Path,
    new_data: bytes,
    *,
    before_mutation_commit: BeforeMutationCommit,
    require_snapshot_current: RequireSnapshotCurrent,
    invoke_replace_file: InvokeReplaceFile,
) -> tuple[bool, FileIdentity]:
    """Best-effort Windows atomic replace on a stable, non-reparse directory tree."""
    parent_identity = path_identity(target.parent)
    temporary_path, descriptor = _open_temporary(target.parent)
    temporary_exists = True
    try:
        try:
            write_all(descriptor, new_data)
            os.fsync(descriptor)
            written_identity = required_descriptor_identity(descriptor)
        finally:
            os.close(descriptor)
        before_mutation_commit("write", target)
        require_snapshot_current(snapshot)
        if path_identity(target.parent) != parent_identity:
            raise WorkspaceError(
                "write_conflict", f"parent changed before commit: {snapshot.relative}"
            )

        durability_uncertain = False
        if snapshot.data is None:
            # Unlike POSIX rename(), Windows rename is a no-clobber operation.
            os.rename(temporary_path, target)
        else:
            # ReplaceFileW carries the replaced file's DACL and metadata forward.
            backup_path = _unused_backup_path(target.parent)
            # The helper now owns cleanup or preservation of both reserved paths.
            temporary_exists = False
            durability_uncertain = _replace_file(
                target,
                temporary_path,
                backup_path,
                invoke_replace_file=invoke_replace_file,
            )
        temporary_exists = False
        return durability_uncertain, written_identity
    finally:
        if temporary_exists:
            best_effort_unlink(temporary_path)


def opened_file_path(descriptor: int) -> Path | None:
    """Return the physical path bound to an open descriptor when the OS exposes it."""
    if sys.platform != "win32" or os.name != "nt":  # pragma: no cover - POSIX path.
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


def require_no_named_streams(path: Path, display_path: str) -> None:
    """Reject NTFS streams that the byte-only journal cannot faithfully restore."""
    if sys.platform != "win32" or os.name != "nt":  # pragma: no cover - POSIX path.
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


def invoke_replace_file(target: Path, replacement: Path, backup: Path) -> int | None:
    """Return None on success or the native ReplaceFileW error code on failure."""
    if sys.platform != "win32" or os.name != "nt":  # pragma: no cover - POSIX path.
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


def _open_temporary(parent: Path) -> tuple[Path, int]:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOINHERIT", 0)
    )
    for _ in range(32):
        path = parent / f"{RESERVED_TEMP_PREFIX}{secrets.token_hex(16)}"
        try:
            return path, os.open(path, flags, 0o666)
        except FileExistsError:
            continue
    raise WorkspaceError("io_error", "cannot allocate a unique mutation temporary file")


def _unused_backup_path(parent: Path) -> Path:
    """Choose a reserved same-volume backup name without creating the file."""
    for _ in range(32):
        path = parent / f"{RESERVED_TEMP_PREFIX}backup-{secrets.token_hex(16)}"
        try:
            path.lstat()
        except FileNotFoundError:
            return path
    raise WorkspaceError("io_error", "cannot allocate a unique mutation backup name")


def _replace_file(
    target: Path,
    replacement: Path,
    backup: Path,
    *,
    invoke_replace_file: InvokeReplaceFile,
) -> bool:
    """Replace a Windows file and recover the documented partial-failure state."""
    if sys.platform != "win32" or os.name != "nt":  # pragma: no cover - POSIX path.
        raise WorkspaceError("unsupported_platform", "ReplaceFileW requires Windows")

    error = invoke_replace_file(target, replacement, backup)
    if error is None:
        return not best_effort_unlink(backup)

    if error == _ERROR_UNABLE_TO_MOVE_REPLACEMENT_2:
        if not path_entry_exists(backup):
            raise _recovery_error(
                target,
                replacement,
                backup,
                error,
                "the original-file backup could not be located",
            )
        if path_entry_exists(target):
            best_effort_unlink(replacement)
            raise _recovery_error(
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
            best_effort_unlink(replacement)
            raise _recovery_error(
                target,
                replacement,
                backup,
                error,
                "the original file could not be restored automatically",
            ) from exc

    if path_entry_exists(backup):
        # Outside error 1177 Microsoft documents no backup artifact. Preserve any one we do
        # observe, because guessing which copy is authoritative would risk data loss.
        best_effort_unlink(replacement)
        raise _recovery_error(
            target,
            replacement,
            backup,
            error,
            "an unexpected backup remains after replacement failed",
        )

    if not best_effort_unlink(replacement):
        raise _recovery_error(
            target,
            replacement,
            backup,
            error,
            "the uncommitted replacement could not be removed",
        )
    raise OSError(error, "ReplaceFileW failed after preserving the original file")


def _recovery_error(
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
