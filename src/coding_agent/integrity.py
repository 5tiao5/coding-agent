"""Host-owned integrity snapshots for project verification policy.

The Agent never supplies any value in this module.  Configuration bytes, the selected
interpreter, and protected workspace paths are captured by the local host and reduced to
bounded, deterministic hashes.  Re-checking returns safe reason codes rather than file
contents or raw operating-system errors.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Protocol

from coding_agent._workspace.native import is_link_like
from coding_agent.errors import CodedError

MAX_CONFIG_BYTES = 65_536
MAX_INTERPRETER_BYTES = 268_435_456
MAX_PROTECTED_FILE_BYTES = 16_777_216
MAX_PROTECTED_TOTAL_BYTES = 67_108_864
MAX_PROTECTED_ENTRIES = 10_000


class IntegrityCaptureError(CodedError):
    """A policy input could not be captured without weakening the trust boundary."""


class ProtectedPathKind(StrEnum):
    FILE = "file"
    DIRECTORY = "directory"


@dataclass(frozen=True, slots=True)
class ProtectedPath:
    """One canonical workspace-relative file or directory-prefix declaration."""

    path: str
    kind: ProtectedPathKind


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """Content-free identity of one protected path at a point in time."""

    path: str
    state: str


@dataclass(frozen=True, slots=True)
class ProtectedManifest:
    """Stable, bounded manifest that observes additions, deletions, and type changes."""

    entries: tuple[ManifestEntry, ...]

    @property
    def sha256(self) -> str:
        return _canonical_sha256(
            [{"path": entry.path, "state": entry.state} for entry in self.entries]
        )


@dataclass(frozen=True, slots=True)
class InterpreterBinding:
    """Exact host-selected interpreter identity used to build verifier argv values."""

    invocation_path: Path
    resolved_path: Path
    sha256: str
    explicitly_configured: bool
    link_target: str | None

    def fingerprint_payload(self) -> dict[str, object]:
        return {
            "invocation_path": str(self.invocation_path),
            "resolved_path": str(self.resolved_path),
            "sha256": self.sha256,
            "explicitly_configured": self.explicitly_configured,
            "link_target": self.link_target,
        }


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """Safe result of comparing current inputs with the run-start trust snapshot."""

    intact: bool
    expected_policy_fingerprint: str
    current_policy_fingerprint: str | None
    violations: tuple[str, ...]


class IntegrityPolicy(Protocol):
    """Structural seam implemented by the resolved project policy."""

    @property
    def configured(self) -> bool: ...

    @property
    def config_path(self) -> Path: ...

    @property
    def config_sha256(self) -> str | None: ...

    @property
    def interpreter(self) -> InterpreterBinding | None: ...

    @property
    def protected_paths(self) -> tuple[ProtectedPath, ...]: ...

    @property
    def protected_manifest(self) -> ProtectedManifest: ...

    @property
    def definition_sha256(self) -> str: ...

    @property
    def policy_fingerprint(self) -> str: ...


def read_configuration(path: Path) -> tuple[bytes, str] | None:
    """Read one regular, non-linked bounded policy file and return its digest."""

    parent = path.parent
    try:
        parent_stat = parent.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise IntegrityCaptureError(
            "project_config_unreadable",
            "Project verification configuration is unavailable.",
        ) from exc
    if is_link_like(parent) or not stat.S_ISDIR(parent_stat.st_mode):
        raise IntegrityCaptureError(
            "project_config_unsafe",
            "Project verification configuration must have a regular local parent directory.",
        )

    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise IntegrityCaptureError(
            "project_config_unreadable",
            "Project verification configuration is unavailable.",
        ) from exc
    if is_link_like(path) or not stat.S_ISREG(path_stat.st_mode):
        raise IntegrityCaptureError(
            "project_config_unsafe",
            "Project verification configuration must be a regular non-linked file.",
        )
    if path_stat.st_size > MAX_CONFIG_BYTES:
        raise IntegrityCaptureError(
            "project_config_too_large",
            "Project verification configuration exceeds its size limit.",
        )
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise IntegrityCaptureError(
            "project_config_unreadable",
            "Project verification configuration could not be read.",
        ) from exc
    if len(content) > MAX_CONFIG_BYTES:
        raise IntegrityCaptureError(
            "project_config_too_large",
            "Project verification configuration exceeds its size limit.",
        )
    return content, hashlib.sha256(content).hexdigest()


def bind_interpreter(path: Path, *, explicitly_configured: bool) -> InterpreterBinding:
    """Bind an invocation path while preserving POSIX virtual-environment launchers."""

    invocation = Path(os.path.abspath(path))
    try:
        invocation.lstat()
        resolved = invocation.resolve(strict=True)
        resolved_stat = resolved.stat()
    except (OSError, RuntimeError) as exc:
        raise IntegrityCaptureError(
            "project_interpreter_unavailable",
            "The selected project interpreter is unavailable.",
        ) from exc
    if not stat.S_ISREG(resolved_stat.st_mode):
        raise IntegrityCaptureError(
            "project_interpreter_invalid",
            "The selected project interpreter must resolve to a regular file.",
        )
    if os.name != "nt" and not os.access(invocation, os.X_OK):
        raise IntegrityCaptureError(
            "project_interpreter_invalid",
            "The selected project interpreter is not executable.",
        )
    digest = _hash_regular_file(
        resolved,
        max_bytes=MAX_INTERPRETER_BYTES,
        unavailable_code="project_interpreter_unavailable",
        too_large_code="project_interpreter_too_large",
        description="project interpreter",
    )
    link_target: str | None = None
    if invocation.is_symlink():
        try:
            link_target = os.readlink(invocation)
        except OSError as exc:
            raise IntegrityCaptureError(
                "project_interpreter_unavailable",
                "The selected project interpreter is unavailable.",
            ) from exc
    elif is_link_like(invocation):
        # Windows reparse points do not always expose a portable textual target.  The
        # resolved destination remains part of the binding and detects retargeting.
        link_target = str(resolved)
    return InterpreterBinding(
        invocation_path=invocation,
        resolved_path=resolved,
        sha256=digest,
        explicitly_configured=explicitly_configured,
        link_target=link_target,
    )


def capture_protected_manifest(
    root: Path,
    protected_paths: tuple[ProtectedPath, ...],
) -> ProtectedManifest:
    """Capture all declared paths without following filesystem links."""

    try:
        resolved_root = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise IntegrityCaptureError(
            "protected_manifest_unavailable",
            "Protected workspace paths are unavailable.",
        ) from exc
    entries: list[ManifestEntry] = []
    budget = _ManifestBudget()
    for protected in protected_paths:
        candidate = resolved_root.joinpath(*PurePosixPath(protected.path.rstrip("/")).parts)
        _require_no_link_ancestor(resolved_root, candidate)
        if protected.kind is ProtectedPathKind.FILE:
            _capture_file_declaration(candidate, protected.path, entries, budget)
        else:
            _capture_directory_declaration(
                resolved_root,
                candidate,
                protected.path,
                entries,
                budget,
            )
    entries.sort(key=lambda entry: (entry.path.casefold(), entry.path))
    return ProtectedManifest(tuple(entries))


def compute_policy_fingerprint(
    *,
    configured: bool,
    definition_sha256: str,
    config_sha256: str | None,
    interpreter: InterpreterBinding | None,
    protected_manifest: ProtectedManifest,
) -> str:
    """Hash canonical policy identities without embedding source or configuration text."""

    return _canonical_sha256(
        {
            "configured": configured,
            "definition_sha256": definition_sha256,
            "config_sha256": config_sha256,
            "interpreter": (None if interpreter is None else interpreter.fingerprint_payload()),
            "protected_manifest_sha256": protected_manifest.sha256,
        }
    )


def check_integrity(policy: IntegrityPolicy) -> IntegrityReport:
    """Re-capture policy inputs and report drift without leaking bytes or OS errors."""

    violations: list[str] = []
    try:
        current_config = read_configuration(policy.config_path)
        current_configured = current_config is not None
        current_config_sha256 = None if current_config is None else current_config[1]
        if current_configured != policy.configured or current_config_sha256 != policy.config_sha256:
            violations.append("configuration_changed")

        current_interpreter: InterpreterBinding | None = None
        if policy.interpreter is not None:
            try:
                current_interpreter = bind_interpreter(
                    policy.interpreter.invocation_path,
                    explicitly_configured=policy.interpreter.explicitly_configured,
                )
            except IntegrityCaptureError:
                violations.append("interpreter_changed")
            else:
                if current_interpreter != policy.interpreter:
                    violations.append("interpreter_changed")

        try:
            current_manifest = capture_protected_manifest(
                policy.config_path.parents[1], policy.protected_paths
            )
        except IntegrityCaptureError:
            violations.append("protected_manifest_unavailable")
            current_manifest = None
        else:
            if current_manifest != policy.protected_manifest:
                violations.append("protected_manifest_changed")

        current_fingerprint = (
            None
            if current_manifest is None
            or (policy.interpreter is not None and current_interpreter is None)
            else compute_policy_fingerprint(
                configured=current_configured,
                definition_sha256=policy.definition_sha256,
                config_sha256=current_config_sha256,
                interpreter=current_interpreter,
                protected_manifest=current_manifest,
            )
        )
    except (IntegrityCaptureError, OSError, RuntimeError, ValueError):
        # Expected filesystem races and access failures stay at the trust boundary.
        return IntegrityReport(
            intact=False,
            expected_policy_fingerprint=policy.policy_fingerprint,
            current_policy_fingerprint=None,
            violations=("integrity_unavailable",),
        )

    stable_violations = tuple(dict.fromkeys(violations))
    if current_fingerprint != policy.policy_fingerprint and not stable_violations:
        stable_violations = ("policy_fingerprint_changed",)
    return IntegrityReport(
        intact=not stable_violations and current_fingerprint == policy.policy_fingerprint,
        expected_policy_fingerprint=policy.policy_fingerprint,
        current_policy_fingerprint=current_fingerprint,
        violations=stable_violations,
    )


@dataclass(slots=True)
class _ManifestBudget:
    entries: int = 0
    bytes_hashed: int = 0

    def add_entry(self) -> None:
        self.entries += 1
        if self.entries > MAX_PROTECTED_ENTRIES:
            raise IntegrityCaptureError(
                "protected_manifest_too_large",
                "Protected workspace paths exceed the entry limit.",
            )

    def add_bytes(self, size: int) -> None:
        self.bytes_hashed += size
        if self.bytes_hashed > MAX_PROTECTED_TOTAL_BYTES:
            raise IntegrityCaptureError(
                "protected_manifest_too_large",
                "Protected workspace paths exceed the byte limit.",
            )


def _capture_file_declaration(
    candidate: Path,
    display: str,
    entries: list[ManifestEntry],
    budget: _ManifestBudget,
) -> None:
    try:
        candidate_stat = candidate.lstat()
    except FileNotFoundError:
        _append_manifest_entry(entries, budget, display, "missing")
        return
    except OSError as exc:
        raise IntegrityCaptureError(
            "protected_manifest_unavailable",
            "A protected workspace path is unavailable.",
        ) from exc
    if is_link_like(candidate):
        raise IntegrityCaptureError(
            "protected_path_unsafe",
            "Protected workspace paths cannot contain filesystem links.",
        )
    if not stat.S_ISREG(candidate_stat.st_mode):
        raise IntegrityCaptureError(
            "protected_path_type_invalid",
            "A protected file declaration must identify a regular file or an absent path.",
        )
    digest = _hash_manifest_file(candidate, candidate_stat.st_size, budget)
    _append_manifest_entry(entries, budget, display, f"file:{digest}")


def _capture_directory_declaration(
    root: Path,
    candidate: Path,
    display: str,
    entries: list[ManifestEntry],
    budget: _ManifestBudget,
) -> None:
    try:
        candidate_stat = candidate.lstat()
    except FileNotFoundError:
        _append_manifest_entry(entries, budget, display, "missing")
        return
    except OSError as exc:
        raise IntegrityCaptureError(
            "protected_manifest_unavailable",
            "A protected workspace path is unavailable.",
        ) from exc
    if is_link_like(candidate):
        raise IntegrityCaptureError(
            "protected_path_unsafe",
            "Protected workspace paths cannot contain filesystem links.",
        )
    if not stat.S_ISDIR(candidate_stat.st_mode):
        raise IntegrityCaptureError(
            "protected_path_type_invalid",
            "A protected directory declaration must identify a directory or an absent path.",
        )
    _append_manifest_entry(entries, budget, display, "directory")
    pending = [candidate]
    while pending:
        directory = pending.pop()
        try:
            children = sorted(
                directory.iterdir(), key=lambda path: (path.name.casefold(), path.name)
            )
        except OSError as exc:
            raise IntegrityCaptureError(
                "protected_manifest_unavailable",
                "A protected workspace path is unavailable.",
            ) from exc
        child_directories: list[Path] = []
        for child in children:
            relative = child.relative_to(root).as_posix()
            try:
                child_stat = child.lstat()
            except OSError as exc:
                raise IntegrityCaptureError(
                    "protected_manifest_unavailable",
                    "A protected workspace path is unavailable.",
                ) from exc
            if is_link_like(child):
                raise IntegrityCaptureError(
                    "protected_path_unsafe",
                    "Protected workspace paths cannot contain filesystem links.",
                )
            if stat.S_ISDIR(child_stat.st_mode):
                _append_manifest_entry(entries, budget, relative + "/", "directory")
                child_directories.append(child)
            elif stat.S_ISREG(child_stat.st_mode):
                digest = _hash_manifest_file(child, child_stat.st_size, budget)
                _append_manifest_entry(entries, budget, relative, f"file:{digest}")
            else:
                raise IntegrityCaptureError(
                    "protected_path_type_invalid",
                    "Protected workspace paths must contain only regular files and directories.",
                )
        pending.extend(reversed(child_directories))


def _hash_manifest_file(path: Path, size: int, budget: _ManifestBudget) -> str:
    if size > MAX_PROTECTED_FILE_BYTES:
        raise IntegrityCaptureError(
            "protected_manifest_too_large",
            "A protected workspace file exceeds the byte limit.",
        )
    budget.add_bytes(size)
    return _hash_regular_file(
        path,
        max_bytes=MAX_PROTECTED_FILE_BYTES,
        unavailable_code="protected_manifest_unavailable",
        too_large_code="protected_manifest_too_large",
        description="protected workspace file",
    )


def _append_manifest_entry(
    entries: list[ManifestEntry],
    budget: _ManifestBudget,
    path: str,
    state: str,
) -> None:
    budget.add_entry()
    entries.append(ManifestEntry(path=path, state=state))


def _require_no_link_ancestor(root: Path, candidate: Path) -> None:
    current = root
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:  # pragma: no cover - protected paths are validated earlier.
        raise IntegrityCaptureError(
            "protected_path_unsafe",
            "Protected workspace paths must remain inside the project root.",
        ) from exc
    for part in relative.parts:
        current /= part
        try:
            current.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise IntegrityCaptureError(
                "protected_manifest_unavailable",
                "A protected workspace path is unavailable.",
            ) from exc
        if is_link_like(current):
            raise IntegrityCaptureError(
                "protected_path_unsafe",
                "Protected workspace paths cannot contain filesystem links.",
            )


def _hash_regular_file(
    path: Path,
    *,
    max_bytes: int,
    unavailable_code: str,
    too_large_code: str,
    description: str,
) -> str:
    try:
        path_stat = path.stat()
        if not stat.S_ISREG(path_stat.st_mode):
            raise OSError("not a regular file")
        if path_stat.st_size > max_bytes:
            raise IntegrityCaptureError(
                too_large_code,
                f"The selected {description} exceeds its size limit.",
            )
        digest = hashlib.sha256()
        total = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise IntegrityCaptureError(
                        too_large_code,
                        f"The selected {description} exceeds its size limit.",
                    )
                digest.update(chunk)
        return digest.hexdigest()
    except IntegrityCaptureError:
        raise
    except OSError as exc:
        raise IntegrityCaptureError(
            unavailable_code,
            f"The selected {description} is unavailable.",
        ) from exc


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "IntegrityCaptureError",
    "IntegrityReport",
    "InterpreterBinding",
    "ManifestEntry",
    "ProtectedManifest",
    "ProtectedPath",
    "ProtectedPathKind",
    "bind_interpreter",
    "capture_protected_manifest",
    "check_integrity",
    "compute_policy_fingerprint",
    "read_configuration",
]
