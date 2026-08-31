"""Strict project-owned declarations compiled into host-owned verification policy.

Only ``<root>/.coding-agent/project.toml`` is considered.  The file is never searched
through parent directories, and the schema deliberately exposes typed verifier forms
instead of arbitrary command lines or shell fragments.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final

from coding_agent._workspace.native import is_link_like
from coding_agent.errors import CodedError
from coding_agent.integrity import (
    IntegrityCaptureError,
    InterpreterBinding,
    ProtectedManifest,
    ProtectedPath,
    ProtectedPathKind,
    bind_interpreter,
    capture_protected_manifest,
    check_integrity,
    compute_policy_fingerprint,
    read_configuration,
)
from coding_agent.models import VerificationKind

PROJECT_CONFIG_RELATIVE: Final = PurePosixPath(".coding-agent/project.toml")
_MAX_VERIFIERS = 8
_MAX_PROTECTED_PATHS = 128
_MAX_SCOPES_PER_VERIFIER = 16
_MAX_REQUIRED_SCOPES = 64
_MAX_MODULE_CHARS = 200
_TOKEN_PATTERN = re.compile(r"^[a-z][a-z0-9_.:-]{0,63}$")
_MODULE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_GLOB_CHARACTERS = frozenset("*?[]")
_WINDOWS_RESERVED_STEMS = frozenset(
    {"aux", "clock$", "con", "conin$", "conout$", "nul", "prn"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"com{index}" for index in "¹²³"}
    | {f"lpt{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in "¹²³"}
)


class ProjectConfigError(CodedError):
    """Expected project-policy failure with a stable, content-free message."""


class VerifierType(StrEnum):
    PYTEST = "pytest"
    PYTHON_MODULE = "python-module"


@dataclass(frozen=True, slots=True)
class ResolvedVerifier:
    """Typed verifier compiled to an exact no-shell argv capability."""

    label: str
    verifier_type: VerifierType
    cwd: str
    scopes: tuple[str, ...]
    required: bool
    kind: VerificationKind
    argv: tuple[str, ...]
    module: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedProjectPolicy:
    """Immutable input for runtime verification and completion wiring."""

    root: Path
    config_path: Path
    configured: bool
    config_sha256: str | None
    interpreter: InterpreterBinding | None
    verifiers: tuple[ResolvedVerifier, ...]
    protected_paths: tuple[ProtectedPath, ...]
    protected_manifest: ProtectedManifest
    required_labels: tuple[str, ...]
    required_scopes: tuple[str, ...]
    definition_sha256: str
    policy_fingerprint: str

    @property
    def target_runtime_eligible(self) -> bool:
        """Only an explicitly selected interpreter is eligible for task validation."""

        return bool(self.interpreter and self.interpreter.explicitly_configured)

    @property
    def target_runtime_id(self) -> str:
        if self.interpreter is None:
            return "unconfigured-python"
        source = "project" if self.interpreter.explicitly_configured else "host-fallback"
        return f"{source}-python:{self.interpreter.sha256[:12]}"


def load_project_policy(
    root: Path,
    *,
    host_python: Path | None = None,
) -> ResolvedProjectPolicy:
    """Load exactly one project config, or return an explicit unconfigured policy."""

    resolved_root = _resolve_root(root)
    config_path = resolved_root.joinpath(*PROJECT_CONFIG_RELATIVE.parts)
    try:
        captured_config = read_configuration(config_path)
    except IntegrityCaptureError as exc:
        raise ProjectConfigError(exc.code, exc.message) from exc
    if captured_config is None:
        policy = _unconfigured_policy(resolved_root, config_path)
        if not check_integrity(policy).intact:
            raise ProjectConfigError(
                "project_policy_changed_during_load",
                "Project verification policy changed while it was being loaded.",
            )
        return policy

    content, config_sha256 = captured_config
    document = _parse_document(content)
    interpreter = _resolve_interpreter(
        resolved_root,
        document.python_executable,
        host_python=host_python,
    )
    verifiers = _resolve_verifiers(resolved_root, document.verifiers, interpreter)
    protected_paths = _resolve_protected_paths(document.protected_paths)
    try:
        protected_manifest = capture_protected_manifest(resolved_root, protected_paths)
    except IntegrityCaptureError as exc:
        raise ProjectConfigError(exc.code, exc.message) from exc
    required_labels = tuple(verifier.label for verifier in verifiers if verifier.required)
    if not required_labels:
        raise ProjectConfigError(
            "project_config_invalid",
            "Project verification configuration requires at least one required verifier.",
        )
    required_scopes = _resolve_required_scopes(
        document.required_scopes,
        verifiers,
        protected=bool(protected_paths),
    )
    definition_sha256 = _definition_sha256(
        interpreter=interpreter,
        verifiers=verifiers,
        protected_paths=protected_paths,
        required_scopes=required_scopes,
    )
    policy_fingerprint = compute_policy_fingerprint(
        configured=True,
        definition_sha256=definition_sha256,
        config_sha256=config_sha256,
        interpreter=interpreter,
        protected_manifest=protected_manifest,
    )
    policy = ResolvedProjectPolicy(
        root=resolved_root,
        config_path=config_path,
        configured=True,
        config_sha256=config_sha256,
        interpreter=interpreter,
        verifiers=verifiers,
        protected_paths=protected_paths,
        protected_manifest=protected_manifest,
        required_labels=required_labels,
        required_scopes=required_scopes,
        definition_sha256=definition_sha256,
        policy_fingerprint=policy_fingerprint,
    )
    if not check_integrity(policy).intact:
        raise ProjectConfigError(
            "project_policy_changed_during_load",
            "Project verification policy changed while it was being loaded.",
        )
    return policy


@dataclass(frozen=True, slots=True)
class _VerifierDeclaration:
    label: str
    verifier_type: VerifierType
    cwd: str
    scopes: tuple[str, ...]
    required: bool
    module: str | None


@dataclass(frozen=True, slots=True)
class _ConfigDocument:
    python_executable: str | None
    verifiers: tuple[_VerifierDeclaration, ...]
    protected_paths: tuple[str, ...]
    required_scopes: tuple[str, ...] | None


def _parse_document(content: bytes) -> _ConfigDocument:
    try:
        text = content.decode("utf-8")
        raw = tomllib.loads(text)
    except (UnicodeError, tomllib.TOMLDecodeError, RecursionError, ValueError) as exc:
        raise ProjectConfigError(
            "project_config_invalid",
            "Project verification configuration is not valid UTF-8 TOML.",
        ) from exc
    top = _require_table(raw)
    _reject_unknown(
        top,
        {"schema_version", "protected_paths", "python", "verifiers", "completion"},
    )
    schema = top.get("schema_version")
    if isinstance(schema, bool) or not isinstance(schema, int) or schema != 1:
        raise ProjectConfigError(
            "project_config_unsupported",
            "Project verification configuration must declare schema_version = 1.",
        )

    python_executable: str | None = None
    if "python" in top:
        python_table = _require_table(top["python"])
        _reject_unknown(python_table, {"executable"})
        if "executable" in python_table:
            python_executable = _require_string(python_table["executable"])

    protected_paths = _string_sequence(
        top.get("protected_paths", []),
        maximum=_MAX_PROTECTED_PATHS,
        allow_empty=True,
    )
    raw_verifiers = top.get("verifiers")
    if not isinstance(raw_verifiers, list) or not raw_verifiers:
        raise ProjectConfigError(
            "project_config_invalid",
            "Project verification configuration requires a non-empty verifiers array.",
        )
    if len(raw_verifiers) > _MAX_VERIFIERS:
        raise ProjectConfigError(
            "project_config_too_large",
            "Project verification configuration exceeds the verifier limit.",
        )
    verifiers = tuple(_parse_verifier(item) for item in raw_verifiers)

    required_scopes: tuple[str, ...] | None = None
    if "completion" in top:
        completion = _require_table(top["completion"])
        _reject_unknown(completion, {"required_scopes"})
        if "required_scopes" in completion:
            required_scopes = _scope_sequence(
                completion["required_scopes"],
                maximum=_MAX_REQUIRED_SCOPES,
            )
    return _ConfigDocument(
        python_executable=python_executable,
        verifiers=verifiers,
        protected_paths=protected_paths,
        required_scopes=required_scopes,
    )


def _parse_verifier(value: object) -> _VerifierDeclaration:
    table = _require_table(value)
    _reject_unknown(table, {"label", "type", "cwd", "scopes", "required", "module"})
    label = _safe_label(_require_string(table.get("label")))
    raw_type = _require_string(table.get("type"))
    try:
        verifier_type = VerifierType(raw_type)
    except ValueError as exc:
        raise ProjectConfigError(
            "project_config_invalid",
            "Verifier type must be pytest or python-module.",
        ) from exc
    cwd = _safe_relative_path(_require_string(table.get("cwd", ".")), allow_dot=True)
    scopes = _scope_sequence(table.get("scopes"), maximum=_MAX_SCOPES_PER_VERIFIER)
    required = table.get("required", True)
    if type(required) is not bool:
        raise ProjectConfigError(
            "project_config_invalid",
            "Verifier required must be a boolean.",
        )
    module: str | None = None
    if verifier_type is VerifierType.PYTHON_MODULE:
        module = _safe_module(_require_string(table.get("module")))
    elif "module" in table:
        raise ProjectConfigError(
            "project_config_invalid",
            "Only python-module verifiers may declare a module.",
        )
    return _VerifierDeclaration(
        label=label,
        verifier_type=verifier_type,
        cwd=cwd,
        scopes=scopes,
        required=required,
        module=module,
    )


def _resolve_interpreter(
    root: Path,
    configured: str | None,
    *,
    host_python: Path | None,
) -> InterpreterBinding:
    explicitly_configured = configured is not None
    if configured is None:
        candidate = Path(sys.executable) if host_python is None else host_python
    else:
        candidate = _interpreter_candidate(root, configured)
    try:
        return bind_interpreter(candidate, explicitly_configured=explicitly_configured)
    except IntegrityCaptureError as exc:
        raise ProjectConfigError(exc.code, exc.message) from exc


def _interpreter_candidate(root: Path, value: str) -> Path:
    if value != value.strip() or not value or _contains_control(value):
        raise ProjectConfigError(
            "project_interpreter_invalid",
            "The configured interpreter path is invalid.",
        )
    native = Path(value)
    windows = PureWindowsPath(value)
    posix = PurePosixPath(value.replace("\\", "/"))
    if native.is_absolute():
        return native
    if windows.drive or windows.root or posix.is_absolute():
        raise ProjectConfigError(
            "project_interpreter_invalid",
            "The configured interpreter must use a native absolute path or a safe relative path.",
        )
    canonical = _safe_relative_path(value, allow_dot=False)
    candidate = root.joinpath(*PurePosixPath(canonical).parts)
    _reject_linked_parent(root, candidate, include_leaf=False, code="project_interpreter_invalid")
    return candidate


def _resolve_verifiers(
    root: Path,
    declarations: tuple[_VerifierDeclaration, ...],
    interpreter: InterpreterBinding,
) -> tuple[ResolvedVerifier, ...]:
    labels = tuple(item.label for item in declarations)
    if len({label.casefold() for label in labels}) != len(labels):
        raise ProjectConfigError(
            "project_config_invalid",
            "Verifier labels must be unique under portable case matching.",
        )
    resolved: list[ResolvedVerifier] = []
    executable = str(interpreter.invocation_path)
    for item in declarations:
        cwd_path = root if item.cwd == "." else root.joinpath(*PurePosixPath(item.cwd).parts)
        _reject_linked_parent(root, cwd_path, include_leaf=True, code="project_config_invalid")
        try:
            if not cwd_path.is_dir():
                raise OSError("cwd is not a directory")
            if not cwd_path.resolve(strict=True).is_relative_to(root):
                raise OSError("cwd escapes root")
        except (OSError, RuntimeError) as exc:
            raise ProjectConfigError(
                "project_config_invalid",
                "Verifier working directories must be existing project directories.",
            ) from exc
        argv: tuple[str, ...]
        if item.verifier_type is VerifierType.PYTEST:
            argv = (
                executable,
                "-I",
                "-B",
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
            )
            kind = VerificationKind.TEST
        else:
            argv = (executable, "-B", "-m", item.module or "")
            kind = VerificationKind.CHECK
        resolved.append(
            ResolvedVerifier(
                label=item.label,
                verifier_type=item.verifier_type,
                cwd=item.cwd,
                scopes=item.scopes,
                required=item.required,
                kind=kind,
                argv=argv,
                module=item.module,
            )
        )
    return tuple(resolved)


def _resolve_protected_paths(values: Sequence[str]) -> tuple[ProtectedPath, ...]:
    resolved: list[ProtectedPath] = []
    for value in values:
        if value != value.strip() or not value or _contains_control(value):
            raise ProjectConfigError(
                "protected_path_invalid",
                "Protected paths must be canonical project-relative paths.",
            )
        if any(character in value for character in _GLOB_CHARACTERS):
            raise ProjectConfigError(
                "protected_path_invalid",
                "Protected paths do not support glob syntax.",
            )
        is_directory = value.endswith("/")
        raw_path = value[:-1] if is_directory else value
        canonical = _safe_relative_path(raw_path, allow_dot=False)
        resolved.append(
            ProtectedPath(
                path=canonical + ("/" if is_directory else ""),
                kind=(ProtectedPathKind.DIRECTORY if is_directory else ProtectedPathKind.FILE),
            )
        )
    portable = tuple(item.path.casefold() for item in resolved)
    if len(set(portable)) != len(portable):
        raise ProjectConfigError(
            "protected_path_invalid",
            "Protected paths must be unique under portable case matching.",
        )
    for index, item in enumerate(resolved):
        for other in resolved[index + 1 :]:
            directory = item if item.kind is ProtectedPathKind.DIRECTORY else other
            nested = other if directory is item else item
            if directory.kind is ProtectedPathKind.DIRECTORY and nested.path.casefold().startswith(
                directory.path.casefold()
            ):
                raise ProjectConfigError(
                    "protected_path_invalid",
                    "Protected path declarations cannot overlap.",
                )
    return tuple(resolved)


def _resolve_required_scopes(
    configured: tuple[str, ...] | None,
    verifiers: tuple[ResolvedVerifier, ...],
    *,
    protected: bool,
) -> tuple[str, ...]:
    available = tuple(dict.fromkeys(scope for verifier in verifiers for scope in verifier.scopes))
    required = (
        tuple(
            dict.fromkeys(
                scope for verifier in verifiers if verifier.required for scope in verifier.scopes
            )
        )
        if configured is None
        else configured
    )
    if protected and "integrity:protected" not in required:
        required = (*required, "integrity:protected")
    permitted = set(available)
    if protected:
        permitted.add("integrity:protected")
    if not required or any(scope not in permitted for scope in required):
        raise ProjectConfigError(
            "project_config_invalid",
            "Completion scopes must be non-empty and covered by configured verifier policy.",
        )
    return required


def _unconfigured_policy(root: Path, config_path: Path) -> ResolvedProjectPolicy:
    protected_manifest = ProtectedManifest(())
    definition_sha256 = _canonical_sha256({"configured": False, "schema_version": 1})
    fingerprint = compute_policy_fingerprint(
        configured=False,
        definition_sha256=definition_sha256,
        config_sha256=None,
        interpreter=None,
        protected_manifest=protected_manifest,
    )
    return ResolvedProjectPolicy(
        root=root,
        config_path=config_path,
        configured=False,
        config_sha256=None,
        interpreter=None,
        verifiers=(),
        protected_paths=(),
        protected_manifest=protected_manifest,
        required_labels=(),
        required_scopes=(),
        definition_sha256=definition_sha256,
        policy_fingerprint=fingerprint,
    )


def _definition_sha256(
    *,
    interpreter: InterpreterBinding,
    verifiers: tuple[ResolvedVerifier, ...],
    protected_paths: tuple[ProtectedPath, ...],
    required_scopes: tuple[str, ...],
) -> str:
    return _canonical_sha256(
        {
            "schema_version": 1,
            "runtime_explicit": interpreter.explicitly_configured,
            "verifiers": [
                {
                    "label": verifier.label,
                    "type": verifier.verifier_type.value,
                    "cwd": verifier.cwd,
                    "scopes": list(verifier.scopes),
                    "required": verifier.required,
                    "kind": verifier.kind.value,
                    "argv": list(verifier.argv),
                    "module": verifier.module,
                }
                for verifier in verifiers
            ],
            "protected_paths": [
                {"path": item.path, "kind": item.kind.value} for item in protected_paths
            ],
            "required_scopes": list(required_scopes),
        }
    )


def _resolve_root(root: Path) -> Path:
    try:
        resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ProjectConfigError(
            "project_root_invalid",
            "Project root is unavailable.",
        ) from exc
    if not resolved.is_dir():
        raise ProjectConfigError("project_root_invalid", "Project root must be a directory.")
    return resolved


def _require_table(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ProjectConfigError(
            "project_config_invalid",
            "Project verification configuration contains an invalid table.",
        )
    return value


def _reject_unknown(table: Mapping[str, object], allowed: set[str]) -> None:
    if any(key not in allowed for key in table):
        raise ProjectConfigError(
            "project_config_unknown_field",
            "Project verification configuration contains unknown fields.",
        )


def _require_string(value: object) -> str:
    if not isinstance(value, str):
        raise ProjectConfigError(
            "project_config_invalid",
            "Project verification configuration contains an invalid text field.",
        )
    return value


def _string_sequence(value: object, *, maximum: int, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ProjectConfigError(
            "project_config_invalid",
            "Project verification configuration contains an invalid text array.",
        )
    if len(value) > maximum:
        raise ProjectConfigError(
            "project_config_too_large",
            "Project verification configuration exceeds an array limit.",
        )
    if not allow_empty and not value:
        raise ProjectConfigError(
            "project_config_invalid",
            "Project verification configuration requires a non-empty array.",
        )
    return tuple(value)


def _scope_sequence(value: object, *, maximum: int) -> tuple[str, ...]:
    scopes = _string_sequence(value, maximum=maximum, allow_empty=False)
    if len(set(scopes)) != len(scopes) or any(
        _TOKEN_PATTERN.fullmatch(scope) is None for scope in scopes
    ):
        raise ProjectConfigError(
            "project_config_invalid",
            "Verification scopes must be unique canonical lowercase tokens.",
        )
    return scopes


def _safe_label(value: str) -> str:
    if value != value.strip() or not value or len(value) > 120 or _contains_control(value):
        raise ProjectConfigError(
            "project_config_invalid",
            "Verifier labels must be canonical printable text of at most 120 characters.",
        )
    return value


def _safe_module(value: str) -> str:
    if len(value) > _MAX_MODULE_CHARS or _MODULE_PATTERN.fullmatch(value) is None:
        raise ProjectConfigError(
            "project_config_invalid",
            "Python module verifiers require a canonical dotted module name.",
        )
    return value


def _safe_relative_path(value: str, *, allow_dot: bool) -> str:
    if value != value.strip() or not value or _contains_control(value):
        raise ProjectConfigError(
            "project_config_invalid",
            "Project policy paths must be canonical relative paths.",
        )
    portable = value.replace("\\", "/")
    windows = PureWindowsPath(value)
    posix = PurePosixPath(portable)
    if windows.drive or windows.root or posix.is_absolute() or ":" in portable:
        raise ProjectConfigError(
            "project_config_invalid",
            "Project policy paths must remain relative to the project root.",
        )
    raw_parts = portable.split("/")
    if any(not part or part in {".", ".."} for part in raw_parts):
        if allow_dot and portable == ".":
            return "."
        raise ProjectConfigError(
            "project_config_invalid",
            "Project policy paths must be canonical and cannot contain parent segments.",
        )
    if any(any(character in '<>"|*?[]' for character in part) for part in raw_parts):
        raise ProjectConfigError(
            "project_config_invalid",
            "Project policy paths cannot contain wildcard or device syntax.",
        )
    if any(part.endswith((" ", ".")) for part in raw_parts):
        raise ProjectConfigError(
            "project_config_invalid",
            "Project policy path components must be portable.",
        )
    if any(
        part.split(".", maxsplit=1)[0].casefold() in _WINDOWS_RESERVED_STEMS for part in raw_parts
    ):
        raise ProjectConfigError(
            "project_config_invalid",
            "Project policy path components must be portable.",
        )
    return PurePosixPath(*raw_parts).as_posix()


def _reject_linked_parent(
    root: Path,
    candidate: Path,
    *,
    include_leaf: bool,
    code: str,
) -> None:
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ProjectConfigError(
            code, "Project policy paths must remain inside the project root."
        ) from exc
    current = root
    parts = relative.parts if include_leaf else relative.parts[:-1]
    for part in parts:
        current /= part
        try:
            current.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise ProjectConfigError(code, "A project policy path is unavailable.") from exc
        if is_link_like(current):
            raise ProjectConfigError(code, "Project policy paths cannot traverse filesystem links.")


def _contains_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


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
    "PROJECT_CONFIG_RELATIVE",
    "ProjectConfigError",
    "ResolvedProjectPolicy",
    "ResolvedVerifier",
    "VerifierType",
    "load_project_policy",
]
