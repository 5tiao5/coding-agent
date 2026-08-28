"""Platform process-tree ownership primitives for local command execution.

This module is deliberately private.  It establishes containment before untrusted
command code can run and exposes one small lifecycle contract to ``command.py``.
"""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol

import psutil

_WINDOWS_CREATE_SUSPENDED = 0x00000004
_WINDOWS_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_WINDOWS_JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
_WINDOWS_JOB_OBJECT_BASIC_PROCESS_ID_LIST = 3
_WINDOWS_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_WINDOWS_THREAD_SUSPEND_RESUME = 0x0002
_WINDOWS_MAX_TRACKED_PROCESS_IDS = 4_096


class ProcessContainment(Protocol):
    """Platform ownership capability for one complete command process tree."""

    def has_active_descendants(self, process: subprocess.Popen[bytes]) -> bool | None:
        """Report live owned processes other than the already-waited root."""

    def terminate_and_confirm(
        self,
        process: subprocess.Popen[bytes],
        grace_seconds: float,
    ) -> str | None:
        """Stop every owned process and return a safe failure reason, if any."""

    def close(self) -> None:
        """Release containment, applying its last-resort kill policy."""


class ProcessControlError(RuntimeError):
    """Containment could not be established before untrusted code was resumed."""


def start_contained_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> tuple[subprocess.Popen[bytes], ProcessContainment]:
    """Start a process only after an OS process-tree capability is established."""
    if os.name == "posix":  # pragma: no cover - exercised by POSIX CI.
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
            bufsize=0,
            start_new_session=True,
        )
        return process, _PosixProcessGroup(process.pid)

    if os.name != "nt":
        raise ProcessControlError(
            "command process containment is unavailable on this operating system"
        )

    job = _WindowsJobObject.create()
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
            bufsize=0,
            creationflags=(
                subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_NO_WINDOW
                | _WINDOWS_CREATE_SUSPENDED
            ),
        )
    except BaseException:
        job.close()
        raise

    try:
        attachment_failure = job.attach_and_resume(process)
    except (KeyboardInterrupt, SystemExit):
        _abort_windows_start(job, process)
        raise
    except Exception as exc:
        _abort_windows_start(job, process)
        raise ProcessControlError("Windows Job Object assignment failed unexpectedly") from exc

    if attachment_failure is not None:
        _abort_windows_start(job, process)
        raise ProcessControlError(attachment_failure)
    return process, job


class _PosixProcessGroup:  # pragma: no cover - exercised by POSIX CI.
    """Own a fresh POSIX session/process group rooted at one command PID."""

    def __init__(self, process_group_id: int) -> None:
        self._process_group_id = process_group_id

    def has_active_descendants(self, process: subprocess.Popen[bytes]) -> bool | None:
        del process
        active_processes = _posix_group_active_process_count(self._process_group_id)
        return None if active_processes is None else active_processes > 0

    def terminate_and_confirm(
        self,
        process: subprocess.Popen[bytes],
        grace_seconds: float,
    ) -> str | None:
        if not _signal_posix_group(self._process_group_id, signal.SIGTERM):
            return "POSIX command process group could not be signalled safely"
        if _wait_for_posix_group_exit(
            self._process_group_id,
            process,
            grace_seconds,
        ):
            return None

        if not _signal_posix_group(
            self._process_group_id,
            int(getattr(signal, "SIGKILL", 9)),
        ):
            return "POSIX command process group could not be killed safely"
        if _wait_for_posix_group_exit(
            self._process_group_id,
            process,
            grace_seconds,
        ):
            return None
        return "POSIX command process group termination could not be confirmed"

    def close(self) -> None:
        return


class _WindowsIoCounters(ctypes.Structure):
    _fields_ = [
        ("read_operations", ctypes.c_ulonglong),
        ("write_operations", ctypes.c_ulonglong),
        ("other_operations", ctypes.c_ulonglong),
        ("read_bytes", ctypes.c_ulonglong),
        ("write_bytes", ctypes.c_ulonglong),
        ("other_bytes", ctypes.c_ulonglong),
    ]


class _WindowsBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("per_process_user_time_limit", ctypes.c_longlong),
        ("per_job_user_time_limit", ctypes.c_longlong),
        ("limit_flags", ctypes.c_uint32),
        ("minimum_working_set_size", ctypes.c_size_t),
        ("maximum_working_set_size", ctypes.c_size_t),
        ("active_process_limit", ctypes.c_uint32),
        ("affinity", ctypes.c_size_t),
        ("priority_class", ctypes.c_uint32),
        ("scheduling_class", ctypes.c_uint32),
    ]


class _WindowsExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("basic_limit_information", _WindowsBasicLimitInformation),
        ("io_info", _WindowsIoCounters),
        ("process_memory_limit", ctypes.c_size_t),
        ("job_memory_limit", ctypes.c_size_t),
        ("peak_process_memory_used", ctypes.c_size_t),
        ("peak_job_memory_used", ctypes.c_size_t),
    ]


class _WindowsBasicAccountingInformation(ctypes.Structure):
    _fields_ = [
        ("total_user_time", ctypes.c_longlong),
        ("total_kernel_time", ctypes.c_longlong),
        ("this_period_total_user_time", ctypes.c_longlong),
        ("this_period_total_kernel_time", ctypes.c_longlong),
        ("total_page_fault_count", ctypes.c_uint32),
        ("total_processes", ctypes.c_uint32),
        ("active_processes", ctypes.c_uint32),
        ("total_terminated_processes", ctypes.c_uint32),
    ]


class _WindowsJobObject:
    """Windows process-tree capability with kill-on-close as a last resort."""

    def __init__(self, api: Any, handle: int) -> None:
        self._api = api
        self._handle: int | None = handle

    @classmethod
    def create(cls) -> _WindowsJobObject:
        api = _configured_windows_kernel32()
        handle = api.CreateJobObjectW(None, None)
        if not handle:
            raise ProcessControlError("Windows Job Object could not be created")

        job = cls(api, int(handle))
        limits = _WindowsExtendedLimitInformation()
        limits.basic_limit_information.limit_flags = _WINDOWS_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not api.SetInformationJobObject(
            ctypes.c_void_p(job._handle),
            _WINDOWS_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            job.close()
            raise ProcessControlError(
                "Windows Job Object kill-on-close policy could not be established"
            )
        return job

    def attach_and_resume(self, process: subprocess.Popen[bytes]) -> str | None:
        """Assign while suspended so descendants cannot escape before ownership."""
        if self._handle is None:
            _terminate_suspended_process(process)
            return "Windows Job Object was closed before command assignment"

        raw_process_handle = getattr(process, "_handle", None)
        if raw_process_handle is None or not self._api.AssignProcessToJobObject(
            ctypes.c_void_p(self._handle),
            ctypes.c_void_p(int(raw_process_handle)),
        ):
            cleanup_confirmed = _terminate_suspended_process(process)
            suffix = "" if cleanup_confirmed else "; suspended process cleanup was unconfirmed"
            return f"Windows Job Object assignment failed; the host may forbid nested jobs{suffix}"

        if _resume_windows_primary_thread(self._api, process):
            return None

        cleanup_failure = self.terminate_and_confirm(process, 1.0)
        suffix = "" if cleanup_failure is None else f"; {cleanup_failure}"
        return f"Windows suspended command could not be resumed safely{suffix}"

    def has_active_descendants(self, process: subprocess.Popen[bytes]) -> bool | None:
        process_ids = self._active_process_ids()
        if process_ids is None:
            return None
        for process_id in process_ids:
            if process_id == process.pid:
                continue
            try:
                status = psutil.Process(process_id).status()
            except psutil.NoSuchProcess:
                continue
            except (psutil.AccessDenied, OSError):
                return None
            if status != psutil.STATUS_ZOMBIE:
                return True
        return False

    def terminate_and_confirm(
        self,
        process: subprocess.Popen[bytes],
        grace_seconds: float,
    ) -> str | None:
        if self._handle is None:
            return "Windows Job Object was unavailable during command cleanup"

        active_processes = self._active_process_count()
        if active_processes is None:
            return "Windows Job Object membership could not be queried"
        if (
            active_processes > 0
            and not self._api.TerminateJobObject(
                ctypes.c_void_p(self._handle),
                1,
            )
            and self._active_process_count() != 0
        ):
            return "Windows Job Object could not terminate its process tree"

        deadline = time.monotonic() + grace_seconds
        while True:
            process.poll()
            active_processes = self._active_process_count()
            if active_processes == 0:
                return None
            if active_processes is None:
                return "Windows Job Object termination could not be queried"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "Windows Job Object process-tree termination could not be confirmed"
            time.sleep(min(0.01, remaining))

    def close(self) -> None:
        if self._handle is None:
            return
        # KILL_ON_JOB_CLOSE remains a best-effort safety net even when confirmation
        # failed.  The caller still reports CONTROL_FAILED because close is asynchronous.
        self._api.CloseHandle(ctypes.c_void_p(self._handle))
        self._handle = None

    def _active_process_count(self) -> int | None:
        if self._handle is None:
            return None
        accounting = _WindowsBasicAccountingInformation()
        returned_length = ctypes.c_uint32()
        if not self._api.QueryInformationJobObject(
            ctypes.c_void_p(self._handle),
            _WINDOWS_JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(accounting),
            ctypes.sizeof(accounting),
            ctypes.byref(returned_length),
        ):
            return None
        return int(accounting.active_processes)

    def _active_process_ids(self) -> tuple[int, ...] | None:
        if self._handle is None:
            return None
        pointer_size = ctypes.sizeof(ctypes.c_size_t)
        header_size = ctypes.sizeof(ctypes.c_uint32) * 2
        buffer_size = header_size + (_WINDOWS_MAX_TRACKED_PROCESS_IDS * pointer_size)
        buffer = ctypes.create_string_buffer(buffer_size)
        returned_length = ctypes.c_uint32()
        if not self._api.QueryInformationJobObject(
            ctypes.c_void_p(self._handle),
            _WINDOWS_JOB_OBJECT_BASIC_PROCESS_ID_LIST,
            buffer,
            buffer_size,
            ctypes.byref(returned_length),
        ):
            return None
        listed = ctypes.c_uint32.from_buffer(buffer, ctypes.sizeof(ctypes.c_uint32)).value
        if listed > _WINDOWS_MAX_TRACKED_PROCESS_IDS:
            return None
        process_id_array = (ctypes.c_size_t * listed).from_buffer(buffer, header_size)
        return tuple(int(process_id) for process_id in process_id_array)


def _configured_windows_kernel32() -> Any:
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        raise ProcessControlError("Windows Job Object API is unavailable")
    api = win_dll("kernel32", use_last_error=True)
    api.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    api.CreateJobObjectW.restype = ctypes.c_void_p
    api.SetInformationJobObject.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    api.SetInformationJobObject.restype = ctypes.c_int
    api.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    api.AssignProcessToJobObject.restype = ctypes.c_int
    api.TerminateJobObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    api.TerminateJobObject.restype = ctypes.c_int
    api.QueryInformationJobObject.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_void_p,
    ]
    api.QueryInformationJobObject.restype = ctypes.c_int
    api.OpenThread.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    api.OpenThread.restype = ctypes.c_void_p
    api.ResumeThread.argtypes = [ctypes.c_void_p]
    api.ResumeThread.restype = ctypes.c_uint32
    api.CloseHandle.argtypes = [ctypes.c_void_p]
    api.CloseHandle.restype = ctypes.c_int
    return api


def _resume_windows_primary_thread(api: Any, process: subprocess.Popen[bytes]) -> bool:
    try:
        threads = psutil.Process(process.pid).threads()
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return False
    if len(threads) != 1:
        return False

    thread_handle = api.OpenThread(
        _WINDOWS_THREAD_SUSPEND_RESUME,
        False,
        threads[0].id,
    )
    if not thread_handle:
        return False
    try:
        previous_suspend_count = int(api.ResumeThread(thread_handle))
        return previous_suspend_count == 1
    finally:
        api.CloseHandle(thread_handle)


def _terminate_suspended_process(process: subprocess.Popen[bytes]) -> bool:
    try:
        process.kill()
        process.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return process.poll() is not None


def _abort_windows_start(
    job: _WindowsJobObject,
    process: subprocess.Popen[bytes],
) -> None:
    with suppress(BaseException):
        job.terminate_and_confirm(process, 1.0)
    if process.poll() is None:
        _terminate_suspended_process(process)
    job.close()
    if process.stdout is not None:
        with suppress(OSError):
            process.stdout.close()


def _signal_posix_group(  # pragma: no cover - exercised by POSIX CI.
    process_group_id: int,
    requested_signal: int,
) -> bool:
    kill_process_group = getattr(os, "killpg", None)
    if kill_process_group is None:
        return False
    try:
        kill_process_group(process_group_id, requested_signal)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return False
    return True


def _posix_group_active_process_count(  # pragma: no cover - exercised by POSIX CI.
    process_group_id: int,
) -> int | None:
    """Report whether a POSIX process group still has at least one member."""
    kill_process_group = getattr(os, "killpg", None)
    if kill_process_group is None:
        return None
    try:
        kill_process_group(process_group_id, 0)
    except ProcessLookupError:
        return 0
    except (PermissionError, OSError):
        return None
    return 1


def _wait_for_posix_group_exit(  # pragma: no cover - exercised by POSIX CI.
    process_group_id: int,
    process: subprocess.Popen[bytes],
    timeout_seconds: float,
) -> bool:
    kill_process_group = getattr(os, "killpg", None)
    if kill_process_group is None:
        return False
    deadline = time.monotonic() + timeout_seconds
    while True:
        process.poll()
        try:
            kill_process_group(process_group_id, 0)
        except ProcessLookupError:
            return True
        except (PermissionError, OSError):
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.01, remaining))
