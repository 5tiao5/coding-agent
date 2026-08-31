"""Cross-platform, bounded local process execution for the command tool.

This module deliberately executes an argument vector without a model-selected shell.
It owns process lifetime, output capture, and environment hygiene; workspace policy and
model-facing rendering remain in the tool adapter.
"""

from __future__ import annotations

import locale
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path, PurePosixPath
from threading import Lock, Thread
from typing import Protocol

from coding_agent._command_process import (
    ProcessContainment,
    ProcessControlError,
    start_contained_process,
)
from coding_agent.errors import CodedError
from coding_agent.models import VerificationKind

_DEFAULT_MAX_OUTPUT_BYTES = 256_000
_DEFAULT_TERMINATION_GRACE_SECONDS = 1.0
_DESCENDANT_SETTLE_SECONDS = 0.05
_READ_CHUNK_BYTES = 16_384
_OUTPUT_OMISSION_TEMPLATE = "\n...[{omitted} output bytes omitted]...\n"

_SECRET_ENV_MARKERS = (
    "ACCESS_KEY",
    "API_KEY",
    "AUTHORIZATION",
    "CREDENTIAL",
    "GITHUB_TOKEN",
    "PASSWORD",
    "PASSWD",
    "PRIVATE_KEY",
    "SECRET",
    "TOKEN",
)
_SENSITIVE_ENV_NAMES = frozenset(
    {
        "AWS_SHARED_CREDENTIALS_FILE",
        "BOTO_CONFIG",
        "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE",
        "DATABASE_URL",
        "DB_URL",
        "DOCKER_AUTH_CONFIG",
        "DOCKER_CONFIG",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GPG_AGENT_INFO",
        "KUBECONFIG",
        "MONGODB_URI",
        "MYSQL_PWD",
        "NETRC",
        "PGPASSFILE",
        "PGPASSWORD",
        "PGSERVICEFILE",
        "REDIS_URL",
        "SSH_AGENT_PID",
        "SSH_AUTH_SOCK",
    }
)
_PROMPT_ENV_NAMES = frozenset(
    {
        "GIT_ASKPASS",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "SSH_ASKPASS",
        "SUDO_ASKPASS",
    }
)
_VERIFIER_ENV_NAMES = frozenset(
    {
        "APPDATA",
        "COMSPEC",
        "HOME",
        "HOMEDRIVE",
        "HOMEPATH",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "LOCALAPPDATA",
        "NUMBER_OF_PROCESSORS",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "PROGRAMDATA",
        "PYTHONIOENCODING",
        "PYTHONUTF8",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "SYSTEMROOT",
        "TEMP",
        "TERM",
        "TMP",
        "TMPDIR",
        "TZ",
        "USERPROFILE",
        "VIRTUAL_ENV",
        "WINDIR",
    }
)

_EXECUTABLE_SUFFIXES = (".exe", ".com", ".cmd", ".bat")
_PYTHON_EXECUTABLES = frozenset({"py", "pypy", "pypy3", "python", "python3"})
_DENIED_EXECUTABLES = frozenset(
    {
        "bash",
        "chown",
        "cmd",
        "del",
        "diskpart",
        "erase",
        "fish",
        "format",
        "halt",
        "kill",
        "killall",
        "ksh",
        "mkfs",
        "pkill",
        "poweroff",
        "powershell",
        "pwsh",
        "reboot",
        "reg",
        "rm",
        "rmdir",
        "runas",
        "sc",
        "sh",
        "shutdown",
        "shred",
        "su",
        "sudo",
        "takeown",
        "taskkill",
        "wsl",
        "zsh",
    }
)
_DENIED_GIT_SUBCOMMANDS = frozenset({"clean"})


class CommandError(CodedError):
    """Expected command-boundary failure safe to return to the model."""


class CommandClass(StrEnum):
    """Trusted policy classification derived from the argument vector."""

    VERIFIER = "verifier"
    READ_ONLY = "read_only"
    GENERAL = "general"


class CommandPermissionMode(StrEnum):
    """Whether unrecognized or potentially mutating commands may start."""

    SAFE = "safe"
    AUTO = "auto"


class CommandEnvironmentProfile(StrEnum):
    """How much ambient host state an already-authorized command may inherit."""

    SANITIZED = "sanitized"
    VERIFIER = "verifier"


class CommandStatus(StrEnum):
    """Observable process outcome; non-zero exit is still a completed observation."""

    EXITED = "exited"
    TIMED_OUT = "timed_out"
    CONTROL_FAILED = "control_failed"


@dataclass(frozen=True, slots=True)
class CommandClassification:
    command_class: CommandClass
    verification_kind: VerificationKind | None = None
    verification_label: str | None = None

    def __post_init__(self) -> None:
        if (self.verification_kind is None) != (self.verification_label is None):
            raise ValueError("verification kind and label must be set together")
        if self.command_class is CommandClass.VERIFIER and self.verification_kind is None:
            raise ValueError("verifier commands require verification facts")
        if self.command_class is not CommandClass.VERIFIER and self.verification_kind is not None:
            raise ValueError("only verifier commands may carry verification facts")


@dataclass(frozen=True, slots=True)
class VerificationCommandSpec:
    """One host-granted capability that may produce verification evidence.

    The model cannot broaden this capability: executable, arguments, working directory,
    evidence kind, and label are fixed before the run starts.
    """

    argv: tuple[str, ...]
    cwd: str
    kind: VerificationKind
    label: str
    workspace_executable_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError("verification command argv cannot be empty")
        if not Path(self.argv[0]).is_absolute():
            raise ValueError("verification command executable must be an absolute path")
        portable_cwd = PurePosixPath(self.cwd.replace("\\", "/"))
        if not self.cwd or portable_cwd.is_absolute() or ".." in portable_cwd.parts:
            raise ValueError("verification command cwd must be workspace-relative")
        if not self.label.strip() or len(self.label) > 120:
            raise ValueError("verification command label must contain 1-120 characters")
        if self.workspace_executable_sha256 is not None and (
            len(self.workspace_executable_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.workspace_executable_sha256
            )
        ):
            raise ValueError(
                "workspace executable SHA-256 must be 64 lowercase hexadecimal characters"
            )
        if any(
            ord(character) < 32 or ord(character) == 127
            for value in (*self.argv, self.cwd, self.label)
            for character in value
        ):
            raise ValueError("verification command fields must contain printable text")


@dataclass(frozen=True, slots=True)
class CommandRequest:
    argv: tuple[str, ...]
    cwd: Path
    timeout_seconds: float
    environment_profile: CommandEnvironmentProfile = CommandEnvironmentProfile.SANITIZED

    def __post_init__(self) -> None:
        if not isinstance(self.environment_profile, CommandEnvironmentProfile):
            raise ValueError("command environment profile must be host-owned")


@dataclass(frozen=True, slots=True)
class CommandResult:
    status: CommandStatus
    exit_code: int | None
    output: bytes
    total_output_bytes: int
    captured_output_bytes: int
    output_truncated: bool
    terminal_reason: str | None = None

    def __post_init__(self) -> None:
        if self.total_output_bytes < 0 or self.captured_output_bytes < 0:
            raise ValueError("output byte counts cannot be negative")
        if self.captured_output_bytes > self.total_output_bytes:
            raise ValueError("captured output cannot exceed total output")
        if self.status is CommandStatus.EXITED and self.exit_code is None:
            raise ValueError("exited commands require an exit code")
        if self.status is not CommandStatus.EXITED and self.exit_code is not None:
            raise ValueError("non-exited commands cannot carry an exit code")
        if (self.status is CommandStatus.CONTROL_FAILED) != (self.terminal_reason is not None):
            raise ValueError("control failures require exactly one terminal reason")


class CommandRunner(Protocol):
    """Seam used by the model-facing adapter and deterministic tests."""

    def run(self, request: CommandRequest) -> CommandResult:
        """Run one already-authorized command to a bounded terminal outcome."""


class CommandPolicy:
    """Small, explicit policy for command control and verification classification."""

    def __init__(
        self,
        mode: CommandPermissionMode = CommandPermissionMode.SAFE,
        *,
        verification_commands: Sequence[VerificationCommandSpec] = (),
    ) -> None:
        self._mode = CommandPermissionMode(mode)
        self._verification_commands = tuple(verification_commands)
        keys = tuple((spec.argv, spec.cwd) for spec in self._verification_commands)
        if len(keys) != len(set(keys)):
            raise ValueError("verification command argv/cwd capabilities must be unique")

    def classify(
        self,
        argv: Sequence[str],
        *,
        cwd: str = ".",
        workspace_root: Path | None = None,
        approved: bool = False,
    ) -> CommandClassification:
        if not argv:
            raise CommandError("invalid_command", "command argument vector cannot be empty")

        portable_executable = argv[0].replace("\\", "/").rsplit("/", maxsplit=1)[-1]
        if portable_executable.casefold().endswith((".bat", ".cmd")):
            raise CommandError(
                "command_denied",
                "batch commands require an implicit shell and are not allowed",
                metadata={"reason": "implicit_shell"},
            )

        normalized = _normalized_argv(argv)
        executable = normalized[0]
        arguments = normalized[1:]
        if executable in _DENIED_EXECUTABLES or executable.startswith("mkfs."):
            raise CommandError(
                "command_denied",
                "command is blocked by the local destructive-command policy",
                metadata={"reason": "destructive_executable"},
            )

        if executable == "git":
            subcommand = _git_subcommand(arguments)
            if subcommand in _DENIED_GIT_SUBCOMMANDS or (
                subcommand == "reset" and "--hard" in arguments
            ):
                raise CommandError(
                    "command_denied",
                    "command is blocked by the local destructive-command policy",
                    metadata={"reason": "destructive_git_operation"},
                )
        if _is_non_verifying_invocation(executable, arguments):
            return self._general_or_require_approval(approved=approved)

        requested_argv = tuple(argv)
        for spec in self._verification_commands:
            if spec.argv == requested_argv and spec.cwd == cwd:
                if _recognized_verification_kind(executable, arguments) is not spec.kind:
                    return self._general_or_require_approval(approved=approved)
                executable_path = Path(requested_argv[0])
                if spec.workspace_executable_sha256 is not None and not _matches_sha256(
                    executable_path, spec.workspace_executable_sha256
                ):
                    return self._general_or_require_approval(approved=approved)
                if (
                    workspace_root is not None
                    and _is_within_workspace(executable_path, workspace_root)
                    and spec.workspace_executable_sha256 is None
                ):
                    return self._general_or_require_approval(approved=approved)
                return CommandClassification(
                    command_class=CommandClass.VERIFIER,
                    verification_kind=spec.kind,
                    verification_label=spec.label,
                )
        return self._general_or_require_approval(approved=approved)

    def is_verification_call(
        self,
        argv: Sequence[str],
        *,
        cwd: str = ".",
        workspace_root: Path | None = None,
    ) -> bool:
        """Check an exact registered verifier without granting or requesting approval."""

        try:
            classification = self.classify(
                argv,
                cwd=cwd,
                workspace_root=workspace_root,
            )
        except (CommandError, OSError, RuntimeError, ValueError):
            return False
        return classification.command_class is CommandClass.VERIFIER

    def _general_or_require_approval(self, *, approved: bool) -> CommandClassification:
        if self._mode is CommandPermissionMode.SAFE and not approved:
            raise CommandError(
                "command_approval_required",
                "general command requires explicit approval in safe mode",
                metadata={"permission_mode": self._mode.value, "reason": "general_command"},
            )
        return CommandClassification(CommandClass.GENERAL)


class LocalCommandRunner:
    """Execute argv directly with bounded combined output and owned process lifetime."""

    def __init__(
        self,
        *,
        max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
        termination_grace_seconds: float = _DEFAULT_TERMINATION_GRACE_SECONDS,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        if max_output_bytes < 2:
            raise ValueError("max_output_bytes must be at least 2")
        if termination_grace_seconds <= 0:
            raise ValueError("termination_grace_seconds must be positive")
        self._max_output_bytes = max_output_bytes
        self._termination_grace_seconds = termination_grace_seconds
        self._environment = dict(os.environ if environment is None else environment)

    def run(self, request: CommandRequest) -> CommandResult:
        try:
            process, containment = self._start(request)
        except ProcessControlError as exc:
            return _empty_control_failure(str(exc))

        assert process.stdout is not None
        capture = _BoundedByteCapture(self._max_output_bytes)
        reader_errors: list[BaseException] = []
        reader = Thread(
            target=_drain_output,
            args=(process.stdout, capture, reader_errors),
            name="coding-agent-command-output",
            daemon=True,
        )
        reader.start()

        timed_out = False
        control_failed = False
        terminal_reason: str | None = None
        exit_code: int | None = None
        containment_attempted = False
        try:
            try:
                exit_code = process.wait(timeout=request.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True

            if not timed_out:
                active_descendants = _descendants_remain_after_root_exit(
                    containment,
                    process,
                    settle_seconds=min(
                        _DESCENDANT_SETTLE_SECONDS,
                        self._termination_grace_seconds,
                    ),
                )
                if active_descendants is None:
                    control_failed = True
                    terminal_reason = (
                        "command descendants could not be enumerated after the root process exited"
                    )
                elif active_descendants:
                    control_failed = True
                    terminal_reason = (
                        "command root process exited while descendant processes were still active"
                    )

            containment_failure = containment.terminate_and_confirm(
                process,
                self._termination_grace_seconds,
            )
            containment_attempted = True
            if containment_failure is not None:
                control_failed = True
                terminal_reason = containment_failure

            reader.join(timeout=self._termination_grace_seconds)
            if reader.is_alive():
                with suppress(OSError):
                    process.stdout.close()
                reader.join(timeout=self._termination_grace_seconds)
            if reader.is_alive():
                control_failed = True
                terminal_reason = "command output reader could not be stopped safely"
            if reader_errors and not timed_out:
                control_failed = True
                terminal_reason = "command output could not be captured reliably"
        except BaseException:
            if not containment_attempted:
                cleanup_interruptions = 0
                while not containment_attempted and cleanup_interruptions < 2:
                    try:
                        containment_failure = containment.terminate_and_confirm(
                            process,
                            self._termination_grace_seconds,
                        )
                    except BaseException:
                        cleanup_interruptions += 1
                    else:
                        containment_attempted = True
                if not containment_attempted:
                    containment_failure = (
                        "command process-tree cleanup was interrupted and could not be confirmed"
                    )
            reader.join(timeout=self._termination_grace_seconds)
            if reader.is_alive():
                with suppress(OSError):
                    process.stdout.close()
                reader.join(timeout=self._termination_grace_seconds)
            if containment_failure is None:
                raise
            control_failed = True
            terminal_reason = containment_failure
        finally:
            containment.close()
            if not reader.is_alive():
                with suppress(OSError):
                    process.stdout.close()

        output, captured_output_bytes = capture.render()
        if control_failed:
            return CommandResult(
                status=CommandStatus.CONTROL_FAILED,
                exit_code=None,
                output=output,
                total_output_bytes=capture.total_bytes,
                captured_output_bytes=captured_output_bytes,
                output_truncated=capture.truncated,
                terminal_reason=terminal_reason or "command process control failed",
            )
        if timed_out:
            return CommandResult(
                status=CommandStatus.TIMED_OUT,
                exit_code=None,
                output=output,
                total_output_bytes=capture.total_bytes,
                captured_output_bytes=captured_output_bytes,
                output_truncated=capture.truncated,
            )
        return CommandResult(
            status=CommandStatus.EXITED,
            exit_code=exit_code,
            output=output,
            total_output_bytes=capture.total_bytes,
            captured_output_bytes=captured_output_bytes,
            output_truncated=capture.truncated,
        )

    def _start(
        self,
        request: CommandRequest,
    ) -> tuple[subprocess.Popen[bytes], ProcessContainment]:
        environment = (
            _verification_environment(self._environment)
            if request.environment_profile is CommandEnvironmentProfile.VERIFIER
            else _sanitized_environment(self._environment)
        )
        try:
            return start_contained_process(
                request.argv,
                cwd=request.cwd,
                environment=environment,
            )
        except FileNotFoundError as exc:
            raise CommandError(
                "command_not_found",
                "command executable was not found",
                metadata={"executable": _safe_executable_name(request.argv[0])},
            ) from exc
        except PermissionError as exc:
            raise CommandError(
                "command_permission_denied",
                "command executable could not be launched with current permissions",
                metadata={"executable": _safe_executable_name(request.argv[0])},
            ) from exc
        except OSError as exc:
            raise CommandError(
                "command_launch_failed",
                "command process could not be launched",
                metadata={"executable": _safe_executable_name(request.argv[0])},
            ) from exc


def _empty_control_failure(reason: str) -> CommandResult:
    return CommandResult(
        status=CommandStatus.CONTROL_FAILED,
        exit_code=None,
        output=b"",
        total_output_bytes=0,
        captured_output_bytes=0,
        output_truncated=False,
        terminal_reason=reason,
    )


def decode_command_output(data: bytes) -> tuple[str, str]:
    """Decode typical developer-tool output deterministically without raising."""
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        preferred = locale.getpreferredencoding(False) or "utf-8"
        if preferred.casefold().replace("-", "") != "utf8":
            try:
                return data.decode(preferred), preferred
            except (LookupError, UnicodeDecodeError):
                pass
        return data.decode("utf-8", errors="replace"), "utf-8-replacement"


class _BoundedByteCapture:
    def __init__(self, max_bytes: int) -> None:
        self._head_limit = max_bytes // 2
        self._tail_limit = max_bytes - self._head_limit
        self._head = bytearray()
        self._tail = bytearray()
        self._lock = Lock()
        self._total_bytes = 0

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes

    @property
    def truncated(self) -> bool:
        with self._lock:
            return self._total_bytes > len(self._head) + len(self._tail)

    def append(self, chunk: bytes) -> None:
        if not chunk:
            return
        with self._lock:
            self._total_bytes += len(chunk)
            head_remaining = self._head_limit - len(self._head)
            if head_remaining > 0:
                self._head.extend(chunk[:head_remaining])
                chunk = chunk[head_remaining:]
            if not chunk:
                return
            if len(chunk) >= self._tail_limit:
                self._tail[:] = chunk[-self._tail_limit :]
                return
            overflow = len(self._tail) + len(chunk) - self._tail_limit
            if overflow > 0:
                del self._tail[:overflow]
            self._tail.extend(chunk)

    def render(self) -> tuple[bytes, int]:
        with self._lock:
            captured = len(self._head) + len(self._tail)
            if self._total_bytes <= captured:
                return bytes(self._head + self._tail), captured
            omitted = self._total_bytes - captured
            marker = _OUTPUT_OMISSION_TEMPLATE.format(omitted=omitted).encode("ascii")
            return bytes(self._head) + marker + bytes(self._tail), captured


def _drain_output(
    stream: object,
    capture: _BoundedByteCapture,
    errors: list[BaseException],
) -> None:
    try:
        while True:
            chunk = stream.read(_READ_CHUNK_BYTES)  # type: ignore[attr-defined]
            if not chunk:
                return
            capture.append(chunk)
    except (OSError, ValueError) as exc:
        errors.append(exc)


def _descendants_remain_after_root_exit(
    containment: ProcessContainment,
    process: subprocess.Popen[bytes],
    *,
    settle_seconds: float,
) -> bool | None:
    """Let launcher bookkeeping settle, then reject a still-live descendant tree."""
    deadline = time.monotonic() + settle_seconds
    while True:
        active_descendants = containment.has_active_descendants(process)
        if active_descendants is not True:
            return active_descendants
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        time.sleep(min(0.005, remaining))


def _sanitized_environment(source: Mapping[str, str]) -> dict[str, str]:
    environment = {
        key: value for key, value in source.items() if not _is_sensitive_environment_name(key)
    }
    environment.update(
        {
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
            "NO_COLOR": "1",
            "PAGER": "cat",
            "PIP_NO_INPUT": "1",
        }
    )
    return environment


def _verification_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Build a small host-owned environment for a command that can issue evidence."""

    sanitized = _sanitized_environment(source)
    environment = {
        key: value for key, value in sanitized.items() if key.upper() in _VERIFIER_ENV_NAMES
    }
    environment.update(
        {
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
            "NO_COLOR": "1",
            "PAGER": "cat",
            "PIP_NO_INPUT": "1",
        }
    )
    return environment


def _is_sensitive_environment_name(name: str) -> bool:
    normalized = name.upper()
    return (
        normalized in _PROMPT_ENV_NAMES
        or normalized in _SENSITIVE_ENV_NAMES
        or any(
            normalized == marker
            or normalized.startswith(marker + "_")
            or normalized.endswith("_" + marker)
            or f"_{marker}_" in normalized
            for marker in _SECRET_ENV_MARKERS
        )
    )


def _normalized_argv(argv: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(argv)
    executable = _normalized_executable(normalized[0])
    if executable == "uv" and len(normalized) >= 3 and normalized[1].casefold() == "run":
        return (
            _normalized_executable(normalized[2]),
            *normalized[3:],
        )
    return (executable, *normalized[1:])


def _normalized_executable(value: str) -> str:
    name = value.replace("\\", "/").rsplit("/", maxsplit=1)[-1].casefold()
    for suffix in _EXECUTABLE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _safe_executable_name(value: str) -> str:
    normalized = _normalized_executable(value)
    return normalized[:120] or "unknown"


def _git_subcommand(arguments: Sequence[str]) -> str | None:
    value_options = frozenset({"-c", "-C", "--exec-path", "--git-dir", "--work-tree"})
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in value_options:
            index += 2
            continue
        if argument.startswith(("--exec-path=", "--git-dir=", "--work-tree=")):
            index += 1
            continue
        if argument.startswith("-"):
            index += 1
            continue
        return argument
    return None


def _is_non_verifying_invocation(executable: str, arguments: Sequence[str]) -> bool:
    if executable in _PYTHON_EXECUTABLES and "-m" in arguments:
        module_index = arguments.index("-m")
        if module_index + 1 < len(arguments):
            executable = _normalized_executable(arguments[module_index + 1])
            arguments = arguments[module_index + 2 :]

    common = {"--help", "--version", "-h", "help", "version"}
    if any(argument in common for argument in arguments):
        return True
    if executable in _PYTHON_EXECUTABLES and arguments == ("-V",):
        return True
    if executable == "pytest" and any(
        argument
        in {
            "--cache-show",
            "--co",
            "--collect-only",
            "--fixtures",
            "--fixtures-per-test",
            "--markers",
            "--setup-only",
            "--setup-plan",
        }
        for argument in arguments
    ):
        return True
    return executable == "cargo" and "--no-run" in arguments


def _recognized_verification_kind(
    executable: str,
    arguments: Sequence[str],
) -> VerificationKind | None:
    """Accept only bounded, recognizable verifier entry points.

    Exact host registration remains the authority.  This semantic check is a second lock:
    generic interpreters, workspace scripts, and arbitrary executables cannot become evidence
    merely because a host accidentally registered their argv.
    """
    if executable in _PYTHON_EXECUTABLES:
        if "-c" in arguments or "-m" not in arguments:
            return None
        module_index = arguments.index("-m")
        if module_index + 1 >= len(arguments):
            return None
        executable = _normalized_executable(arguments[module_index + 1])
        arguments = arguments[module_index + 2 :]

    if executable in {"pytest", "unittest"}:
        return VerificationKind.TEST
    if executable in {"mypy", "pyright"}:
        return VerificationKind.CHECK
    if executable == "ruff":
        if arguments[:1] == ("check",) or (arguments[:1] == ("format",) and "--check" in arguments):
            return VerificationKind.CHECK
        return None
    if executable == "cargo":
        return _kind_for_named_action(
            arguments,
            tests={"test"},
            builds={"build"},
            checks={"check", "clippy", "fmt"},
        )
    if executable == "go":
        return _kind_for_named_action(
            arguments,
            tests={"test"},
            builds={"build"},
            checks={"vet"},
        )
    if executable in {"npm", "pnpm", "yarn"}:
        action_arguments = arguments[1:] if arguments[:1] == ("run",) else arguments
        return _kind_for_named_action(
            action_arguments,
            tests={"test"},
            builds={"build"},
            checks={"check", "lint", "typecheck"},
        )
    if executable in {"make", "ninja"}:
        return _kind_for_named_action(
            arguments,
            tests={"test", "tests"},
            builds={"build", "compile"},
            checks={"check", "lint", "typecheck"},
        )
    return None


def _kind_for_named_action(
    arguments: Sequence[str],
    *,
    tests: set[str],
    builds: set[str],
    checks: set[str],
) -> VerificationKind | None:
    action = next((value.casefold() for value in arguments if not value.startswith("-")), None)
    if action in tests:
        return VerificationKind.TEST
    if action in builds:
        return VerificationKind.BUILD
    if action in checks:
        return VerificationKind.CHECK
    return None


def _is_within_workspace(executable: Path, workspace_root: Path) -> bool:
    try:
        absolute_executable = Path(os.path.abspath(executable))
        absolute_root = Path(os.path.abspath(workspace_root))
        resolved_executable = executable.resolve(strict=False)
        resolved_root = workspace_root.resolve(strict=True)
    except (OSError, RuntimeError):
        return True
    return (
        absolute_executable == absolute_root
        or absolute_root in absolute_executable.parents
        or resolved_executable == resolved_root
        or resolved_root in resolved_executable.parents
    )


def executable_sha256(executable: Path) -> str:
    """Hash one host-selected executable without trusting its workspace path."""
    digest = sha256()
    try:
        with executable.resolve(strict=True).open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except (OSError, RuntimeError) as exc:
        raise ValueError("verification executable could not be hashed") from exc
    return digest.hexdigest()


def _matches_sha256(executable: Path, expected: str) -> bool:
    try:
        return executable_sha256(executable) == expected
    except ValueError:
        return False
