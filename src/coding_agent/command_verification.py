"""Typed command-verification capabilities and their semantic lock.

Exact argv/cwd registration is necessary but not sufficient for trusted evidence.  This
module owns the second lock: only recognizable verifier entry points, plus the deliberately
narrow project-declared Python module smoke test, can be classified as verification.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from coding_agent.models import VerificationKind

_EXECUTABLE_SUFFIXES = (".exe", ".com", ".cmd", ".bat")
_PYTHON_EXECUTABLE_PATTERN = re.compile(r"^(?:py|(?:python|pypy)(?:3(?:\.\d+)?)?)$")
_PYTHON_MODULE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


@dataclass(frozen=True, slots=True)
class VerificationCommandSpec:
    """One exact host-granted capability that may produce verification evidence.

    ``python_module`` opts into a separate typed smoke-test form.  It freezes the module and
    requires exactly ``python -B -m <module>``; it is not an escape hatch for arbitrary argv.
    """

    argv: tuple[str, ...]
    cwd: str
    kind: VerificationKind
    label: str
    workspace_executable_sha256: str | None = None
    python_module: str | None = None

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
        if self.python_module is not None:
            if _PYTHON_MODULE_PATTERN.fullmatch(self.python_module) is None:
                raise ValueError("python-module verifier must name one importable module")
            executable = normalized_executable(self.argv[0])
            if (
                self.kind is not VerificationKind.CHECK
                or not _is_python_executable(executable)
                or self.argv[1:] != ("-B", "-m", self.python_module)
            ):
                raise ValueError(
                    "python-module verifier must be an exact typed smoke-test capability"
                )
        if any(
            ord(character) < 32 or ord(character) == 127
            for value in (*self.argv, self.cwd, self.label)
            for character in value
        ):
            raise ValueError("verification command fields must contain printable text")


def normalized_argv(argv: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(argv)
    executable = normalized_executable(normalized[0])
    if executable == "uv" and len(normalized) >= 3 and normalized[1].casefold() == "run":
        return (normalized_executable(normalized[2]), *normalized[3:])
    return (executable, *normalized[1:])


def normalized_executable(value: str) -> str:
    name = value.replace("\\", "/").rsplit("/", maxsplit=1)[-1].casefold()
    for suffix in _EXECUTABLE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _is_python_executable(executable: str) -> bool:
    return _PYTHON_EXECUTABLE_PATTERN.fullmatch(executable) is not None


def is_non_verifying_invocation(executable: str, arguments: Sequence[str]) -> bool:
    """Reject help, discovery, and collection-only modes even when exactly registered."""

    if _is_python_executable(executable) and "-m" in arguments:
        module_index = arguments.index("-m")
        if module_index + 1 < len(arguments):
            executable = normalized_executable(arguments[module_index + 1])
            arguments = arguments[module_index + 2 :]

    common = {"--help", "--version", "-h", "help", "version"}
    if any(argument in common for argument in arguments):
        return True
    if _is_python_executable(executable) and arguments == ("-V",):
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


def verification_kind_for_spec(
    spec: VerificationCommandSpec,
    executable: str,
    arguments: Sequence[str],
) -> VerificationKind | None:
    """Apply the semantic lock selected by a host-created capability."""

    if spec.python_module is not None:
        if _is_python_executable(executable) and tuple(arguments) == (
            "-B",
            "-m",
            spec.python_module,
        ):
            return VerificationKind.CHECK
        return None
    return _recognized_verification_kind(executable, arguments)


def _recognized_verification_kind(
    executable: str,
    arguments: Sequence[str],
) -> VerificationKind | None:
    if _is_python_executable(executable):
        if "-c" in arguments or "-m" not in arguments:
            return None
        module_index = arguments.index("-m")
        if module_index + 1 >= len(arguments):
            return None
        executable = normalized_executable(arguments[module_index + 1])
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


__all__ = [
    "VerificationCommandSpec",
    "is_non_verifying_invocation",
    "normalized_argv",
    "normalized_executable",
    "verification_kind_for_spec",
]
