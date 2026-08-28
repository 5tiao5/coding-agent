"""Cross-process leases that serialize one original run and its resumes."""

from __future__ import annotations

import errno
import os
import stat
import sys
from pathlib import Path
from types import TracebackType

from coding_agent.errors import CodedError
from coding_agent.run_id import require_run_id


class RunLeaseError(CodedError):
    """A run lease could not be acquired safely."""


class RunLease:
    """Hold a non-blocking OS file lock for one run until the context exits."""

    def __init__(self, state_dir: Path, run_id: str) -> None:
        raw_state_dir = Path(state_dir)
        if not raw_state_dir.is_absolute():
            raise ValueError("lease state_dir must be absolute")
        if raw_state_dir.is_symlink():
            raise ValueError("lease state_dir cannot be a symbolic link")
        try:
            self._run_id = require_run_id(run_id)
        except ValueError as exc:
            raise RunLeaseError("invalid_run_id", "run_id is not safe for a lease") from exc
        self._state_dir = raw_state_dir.resolve(strict=False)
        self._descriptor: int | None = None

    @property
    def path(self) -> Path:
        return self._state_dir / f"{self._run_id}.lock"

    def acquire(self) -> None:
        if self._descriptor is not None:
            raise RunLeaseError("run_lease_reused", "run lease is already held")
        self._prepare_state_directory()
        path = self.path
        if path.is_symlink():
            raise RunLeaseError("unsafe_run_lease", "run lease path is not a regular file")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(path, flags, 0o600)
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
                raise RunLeaseError(
                    "unsafe_run_lease",
                    "run lease path is not a private regular file",
                )
            if file_stat.st_size == 0:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            _lock_descriptor(descriptor)
        except RunLeaseError:
            if descriptor is not None:
                os.close(descriptor)
            raise
        except OSError as exc:
            if descriptor is not None:
                os.close(descriptor)
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise RunLeaseError(
                    "run_already_active",
                    "another process is already running or resuming this run",
                ) from None
            raise RunLeaseError("run_lease_io", "run lease could not be acquired") from None
        self._descriptor = descriptor

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            _unlock_descriptor(descriptor)
        finally:
            os.close(descriptor)

    def __enter__(self) -> RunLease:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.release()

    def _prepare_state_directory(self) -> None:
        try:
            self._state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError:
            raise RunLeaseError(
                "run_lease_io", "run lease directory could not be created"
            ) from None
        if self._state_dir.is_symlink() or not self._state_dir.is_dir():
            raise RunLeaseError(
                "unsafe_run_lease",
                "run lease directory is not a regular directory",
            )


def _lock_descriptor(descriptor: int) -> None:
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        return
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_descriptor(descriptor: int) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN)
