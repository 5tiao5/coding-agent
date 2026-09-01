"""Bounded activity facts projected from the public runtime event stream.

This module is the disclosure boundary for expandable dashboard cards. It accepts
only already-public event fields, reconstructs a small allowlisted view, and never
serializes arbitrary command metadata or provider call identifiers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

from coding_agent._presentation_safety import (
    contains_credential_like_value,
    sanitize_public_label,
)
from coding_agent.events import RunEvent

ActivityFactFormat = Literal["text", "code", "status"]
ActivityState = Literal["started", "finished"]

_MAX_FACTS = 8
_MAX_LABEL_CHARS = 32
_MAX_VALUE_CHARS = 240
_MAX_SCOPE_ITEMS = 8
_MAX_VISIBLE_NUMBER = 1_000_000_000_000
_VERIFICATION_STATUSES = {
    "verified",
    "checks_only",
    "missing",
    "failed",
    "stale",
    "unverified",
}
_SAFE_SCOPE_PATTERN = re.compile(r"^[a-z][a-z0-9_.:-]{0,63}$")
_SAFE_EXECUTABLE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,119}$")
_VERIFICATION_KINDS = frozenset({"test", "check", "build"})


@dataclass(frozen=True, slots=True)
class ActivityFact:
    """One bounded, explicitly projected fact safe for an activity disclosure."""

    label: str
    value: str
    format: ActivityFactFormat


def activity_id(run_id: str, data: Mapping[str, object]) -> str | None:
    """Derive a stable per-run correlation id without exposing provider call ids."""

    call_id = data.get("call_id")
    if not isinstance(call_id, str) or not call_id.strip():
        return None
    digest = sha256(f"{run_id}\0{call_id}".encode()).hexdigest()[:16]
    return f"act_{digest}"


def public_verification_kind(value: object) -> str | None:
    """Accept only verifier kinds defined by the public runtime contract."""

    return value if isinstance(value, str) and value in _VERIFICATION_KINDS else None


def tool_activity_facts(
    tool_name: str,
    data: Mapping[str, object],
    *,
    finished: bool,
) -> tuple[tuple[ActivityFact, ...], bool]:
    """Project only explicitly approved command facts; arbitrary metadata stays private."""

    if tool_name != "run_command":
        return (), True

    facts: list[ActivityFact] = []
    complete = True
    invocation = _nested_mapping(data, "public_invocation")
    if invocation is None:
        complete = False
    else:
        # A persisted trace can be edited independently of the runtime that produced
        # it. Never replay a command line, even if a newer producer marked it public.
        if "display_command" in invocation:
            complete = False
        raw_executable = _mapping_string(invocation, "executable")
        executable = _safe_executable(raw_executable)
        if raw_executable is not None and executable is None:
            complete = False
        argument_count = _mapping_int(invocation, "argument_count", maximum=64)
        command = executable
        command_complete = executable is not None and argument_count == 0
        if command is not None and argument_count:
            command += f" ({argument_count} argument(s) hidden by safety policy)"
        if command is None:
            complete = False
        else:
            complete &= _append_fact(facts, "Command", command, format="code")
            complete &= command_complete

        raw_verification_label = invocation.get("verification_label")
        verification_label, label_complete = sanitize_public_label(raw_verification_label)
        if raw_verification_label is not None:
            complete &= label_complete
        raw_verification_kind = invocation.get("verification_kind")
        verification_kind = public_verification_kind(raw_verification_kind)
        if raw_verification_kind is not None and verification_kind is None:
            complete = False
        if verification_label or verification_kind:
            verification = " / ".join(
                part for part in (verification_kind, verification_label) if part
            )
            complete &= _append_fact(facts, "Verification", verification, format="text")

    if not finished:
        return tuple(facts), complete

    metadata = _nested_mapping(data, "metadata")
    if metadata is None:
        return tuple(facts), False

    command_class = _mapping_string(metadata, "command_class")
    if command_class in {"verifier", "read_only", "general"}:
        complete &= _append_fact(facts, "Category", command_class, format="text")

    raw_cwd = _mapping_string(metadata, "cwd")
    cwd = _safe_workspace_relative(raw_cwd)
    if raw_cwd is not None and cwd is None:
        complete = False
    complete &= _append_fact(facts, "Working directory", cwd, format="code")

    status = _mapping_string(metadata, "status")
    exit_code = _mapping_int(metadata, "exit_code", minimum=-1_000_000, maximum=1_000_000)
    result: str | None = None
    if status == "exited":
        result = f"Exited with code {exit_code}" if exit_code is not None else "Exited"
    elif status == "timed_out" or _mapping_bool(metadata, "timed_out") is True:
        result = "Timed out"
    elif status == "control_failed":
        result = "Process control failed"
    elif status == "integrity_failed":
        result = "Trusted verification rejected"
    elif _data_bool(data, "ok") is False:
        error_code = _data_string(data, "error_code")
        status_label, status_complete = _status_label(error_code)
        if error_code is not None:
            complete &= status_complete
        result = f"Failed ({status_label})" if status_label else "Failed"
    complete &= _append_fact(facts, "Result", result, format="status")

    captured_bytes = _mapping_int(metadata, "captured_output_bytes")
    total_bytes = _mapping_int(metadata, "total_output_bytes")
    output: str | None = None
    if captured_bytes is not None and total_bytes is not None:
        output = f"Captured {captured_bytes} of {total_bytes} byte(s)"
    elif (
        output_chars := _data_int(data, "output_chars")
    ) is not None and 0 <= output_chars <= _MAX_VISIBLE_NUMBER:
        output = f"{output_chars} output character(s)"
    truncated = _data_bool(data, "truncated") is True or (
        captured_bytes is not None and total_bytes is not None and captured_bytes < total_bytes
    )
    if output is not None and truncated:
        output += "; output truncated"
    complete &= _append_fact(facts, "Output", output, format="text")
    if truncated:
        complete = False
    return tuple(facts), complete


def verification_recorded_facts(
    event: RunEvent,
    data: Mapping[str, object],
) -> tuple[tuple[ActivityFact, ...], bool]:
    """Project one trusted verifier result and its host-owned scopes."""

    facts: list[ActivityFact] = []
    complete = True
    label, label_complete = sanitize_public_label(data.get("label"))
    kind = public_verification_kind(data.get("kind"))
    passed = _data_bool(data, "passed")
    epoch = _data_int(data, "epoch")
    complete &= label_complete
    complete &= _append_fact(facts, "Verification", label, format="text")
    complete &= _append_fact(facts, "Kind", kind, format="text")
    if label is None or kind is None or passed is None:
        complete = False
    if passed is not None:
        complete &= _append_fact(
            facts,
            "Result",
            "Passed" if passed else "Failed",
            format="status",
        )
    complete &= _append_fact(facts, "Step", str(event.step), format="text")
    if epoch is not None and epoch >= 0:
        complete &= _append_fact(facts, "Workspace revision", str(epoch), format="text")
    if "scopes" not in data:
        complete = False
    complete &= _append_scope_fact(facts, data, "scopes", "Scopes")
    if _data_bool(data, "scopes_truncated") is True:
        complete = False
    return tuple(facts), complete


def verification_gate_facts(
    data: Mapping[str, object],
) -> tuple[tuple[ActivityFact, ...], bool]:
    """Project the completion gate and its bounded scope coverage."""

    facts: list[ActivityFact] = []
    complete = True
    status = _data_string(data, "status")
    if status not in _VERIFICATION_STATUSES:
        status = None
        complete = False
    complete &= _append_fact(facts, "Gate", status, format="status")
    scope_keys = ("required_scopes", "passed_scopes", "missing_scopes")
    if not any(key in data for key in scope_keys):
        complete = False
    complete &= _append_scope_fact(facts, data, "required_scopes", "Required scopes")
    complete &= _append_scope_fact(facts, data, "passed_scopes", "Passed scopes")
    complete &= _append_scope_fact(facts, data, "missing_scopes", "Missing scopes")
    return tuple(facts), complete


def _activity_fact(
    label: str,
    value: str,
    *,
    format: ActivityFactFormat,
) -> tuple[ActivityFact | None, bool]:
    clean_label = _clean_text(label, limit=_MAX_LABEL_CHARS)
    printable = "".join(" " if not character.isprintable() else character for character in value)
    collapsed = " ".join(printable.split())
    if not clean_label or not collapsed:
        return None, False
    if len(collapsed) > _MAX_VALUE_CHARS or contains_credential_like_value(collapsed):
        return None, False
    complete = collapsed == value
    return ActivityFact(label=clean_label, value=collapsed, format=format), complete


def _append_fact(
    facts: list[ActivityFact],
    label: str,
    value: str | None,
    *,
    format: ActivityFactFormat,
) -> bool:
    if value is None:
        return True
    if len(facts) >= _MAX_FACTS:
        return False
    fact, complete = _activity_fact(label, value, format=format)
    if fact is not None:
        facts.append(fact)
    return complete


def _safe_scopes(
    data: Mapping[str, object],
    key: str,
) -> tuple[tuple[str, ...] | None, bool]:
    raw = data.get(key)
    if raw is None:
        return None, True
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes):
        return None, False
    scopes: list[str] = []
    complete = len(raw) <= _MAX_SCOPE_ITEMS
    for candidate in raw[:_MAX_SCOPE_ITEMS]:
        if not isinstance(candidate, str) or _SAFE_SCOPE_PATTERN.fullmatch(candidate) is None:
            complete = False
            continue
        if candidate not in scopes:
            scopes.append(candidate)
    return tuple(scopes), complete


def _append_scope_fact(
    facts: list[ActivityFact],
    data: Mapping[str, object],
    key: str,
    label: str,
) -> bool:
    scopes, complete = _safe_scopes(data, key)
    if scopes is None:
        return complete
    value = ", ".join(scopes) if scopes else "None"
    return _append_fact(facts, label, value, format="text") and complete


def _safe_executable(value: str | None) -> str | None:
    if (
        value is None
        or _SAFE_EXECUTABLE_PATTERN.fullmatch(value) is None
        or contains_credential_like_value(value)
    ):
        return None
    return value


def _nested_mapping(data: Mapping[str, object], key: str) -> Mapping[object, object] | None:
    value = data.get(key)
    return value if isinstance(value, Mapping) else None


def _mapping_string(data: Mapping[object, object], key: str) -> str | None:
    value = data.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _mapping_bool(data: Mapping[object, object], key: str) -> bool | None:
    value = data.get(key)
    return value if isinstance(value, bool) else None


def _mapping_int(
    data: Mapping[object, object],
    key: str,
    *,
    minimum: int = 0,
    maximum: int = _MAX_VISIBLE_NUMBER,
) -> int | None:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if minimum <= value <= maximum else None


def _safe_workspace_relative(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.replace("\\", "/")
    printable = "".join(
        " " if not character.isprintable() else character for character in normalized
    )
    collapsed = " ".join(printable.split())
    if (
        not collapsed
        or collapsed.startswith("/")
        or re.match(r"^[A-Za-z]:", collapsed) is not None
        or ".." in collapsed.split("/")
    ):
        return None
    # Return the untruncated normalized source.  `_activity_fact` owns the public
    # value bound and will either preserve it completely or omit it while marking
    # the disclosure incomplete.
    return normalized


def _data_string(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _data_bool(data: Mapping[str, object], key: str) -> bool | None:
    value = data.get(key)
    return value if isinstance(value, bool) else None


def _data_int(data: Mapping[str, object], key: str) -> int | None:
    value = data.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _clean_text(value: str, *, limit: int) -> str:
    printable = "".join(" " if not character.isprintable() else character for character in value)
    collapsed = " ".join(printable.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: max(0, limit - 3)]}..."


def _status_label(value: object) -> tuple[str | None, bool]:
    if not isinstance(value, str):
        return None, False
    printable = "".join(" " if not character.isprintable() else character for character in value)
    collapsed = " ".join(printable.split())
    if not collapsed or len(collapsed) > 80 or contains_credential_like_value(collapsed):
        return None, False
    cleaned = collapsed.replace("_", " ").replace("-", " ").upper()
    return (cleaned or None), collapsed == value


__all__ = [
    "ActivityFact",
    "ActivityFactFormat",
    "ActivityState",
    "activity_id",
    "public_verification_kind",
    "tool_activity_facts",
    "verification_gate_facts",
    "verification_recorded_facts",
]
