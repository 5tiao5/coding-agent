"""Bounded activity facts projected from the public runtime event stream.

This module is the disclosure boundary for expandable dashboard cards. It accepts
only already-public event fields, reconstructs a small allowlisted view, and never
serializes arbitrary command metadata or provider call identifiers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

from coding_agent._presentation_safety import (
    contains_credential_like_value,
    redact_command_argv,
    redact_credential_values,
    sanitize_public_label,
)
from coding_agent.events import RunEvent

ActivityFactFormat = Literal["text", "code", "pre", "status"]
ActivityState = Literal["started", "finished"]

_MAX_FACTS = 12
_MAX_LABEL_CHARS = 32
_MAX_VALUE_CHARS = 240
_MAX_OUTPUT_VALUE_CHARS = 16_000
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
    command_redacted = False
    invocation_cwd_fact: str | None = None
    invocation = _nested_mapping(data, "public_invocation")
    if invocation is None:
        complete = False
    else:
        command, command_redacted, command_complete = _public_command(invocation)
        if command is None:
            complete = False
        else:
            complete &= _append_fact(facts, "Command", command, format="pre")
            complete &= command_complete

        invocation_cwd = _mapping_string(invocation, "cwd")
        safe_invocation_cwd, cwd_replay_redacted = _safe_audit_text(invocation_cwd)
        if invocation_cwd is None or safe_invocation_cwd is None:
            complete = False
        if cwd_replay_redacted:
            command_redacted = True
            if _mapping_bool(invocation, "credentials_redacted") is not True:
                complete = False
        timeout = _mapping_number(invocation, "timeout_seconds", minimum=1, maximum=300)
        invocation_cwd_fact = safe_invocation_cwd
        complete &= _append_fact(
            facts,
            "Working directory",
            invocation_cwd_fact,
            format="code",
        )
        if timeout is None:
            complete = False
        else:
            complete &= _append_fact(
                facts,
                "Timeout",
                f"{timeout:g} second(s)",
                format="text",
            )

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
    if invocation_cwd_fact is None:
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
    source_truncated = _data_bool(data, "truncated") is True or (
        captured_bytes is not None and total_bytes is not None and captured_bytes < total_bytes
    )
    output_status: str | None = None
    if captured_bytes is not None and total_bytes is not None:
        output_status = f"Captured {captured_bytes} of {total_bytes} byte(s)"
    elif (
        output_chars := _data_int(data, "output_chars")
    ) is not None and 0 <= output_chars <= _MAX_VISIBLE_NUMBER:
        output_status = f"{output_chars} output character(s)"
    if output_status is not None and source_truncated:
        output_status += "; runtime capture truncated"

    public_output = _nested_mapping(data, "public_output")
    captured_text: str | None = None
    observed_text: str | None = None
    output_redacted = False
    captured_projection_truncated = False
    observed_projection_truncated = False
    observation_truncated = False
    if public_output is None:
        if _data_bool(data, "ok") is not False:
            complete = False
    else:
        declared_redacted = _mapping_bool(public_output, "credentials_redacted") is True
        raw_captured = public_output.get("captured_text")
        if isinstance(raw_captured, str):
            captured_text, replay_redacted = redact_credential_values(raw_captured)
            output_redacted = declared_redacted or replay_redacted
            if replay_redacted and not declared_redacted:
                complete = False
        else:
            complete = False
        raw_observed = public_output.get("observed_text")
        if isinstance(raw_observed, str):
            observed_text, observed_redacted = redact_credential_values(raw_observed)
            output_redacted |= observed_redacted
            if observed_redacted and not declared_redacted:
                complete = False
        captured_projection_truncated = (
            _mapping_bool(public_output, "captured_projection_truncated") is True
        )
        observed_projection_truncated = (
            _mapping_bool(public_output, "observed_projection_truncated") is True
        )
        observation_truncated = _mapping_bool(public_output, "observation_truncated") is True
    projection_truncated = captured_projection_truncated or observed_projection_truncated
    if output_status is not None and projection_truncated:
        output_status += "; audit projection truncated"
    if output_status is not None and observation_truncated:
        output_status += "; Agent observation compacted"
    complete &= _append_fact(facts, "Output status", output_status, format="text")
    complete &= _append_fact(facts, "Captured output", captured_text, format="pre")
    complete &= _append_fact(facts, "Agent observation", observed_text, format="pre")
    if command_redacted or output_redacted:
        complete &= _append_fact(
            facts,
            "Redaction",
            "Credential-like values redacted",
            format="status",
        )
        complete = False
    if source_truncated or projection_truncated:
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
    sanitized, credentials_redacted = redact_credential_values(value)
    if format == "pre":
        normalized = sanitized.replace("\r\n", "\n").replace("\r", "\n")
        prepared = "".join(
            character if character in {"\n", "\t"} or character.isprintable() else " "
            for character in normalized
        )
        limit = _MAX_OUTPUT_VALUE_CHARS
    else:
        printable = "".join(
            " " if not character.isprintable() else character for character in sanitized
        )
        prepared = " ".join(printable.split())
        limit = _MAX_VALUE_CHARS
    if not clean_label or not prepared:
        return None, False
    complete = prepared == value and not credentials_redacted
    if len(prepared) > limit:
        prepared = prepared[:limit]
        complete = False
    return ActivityFact(label=clean_label, value=prepared, format=format), complete


def _public_command(
    invocation: Mapping[object, object],
) -> tuple[str | None, bool, bool]:
    """Rebuild a stable display string from token boundaries in the public trace."""

    raw_argv = invocation.get("argv")
    if isinstance(raw_argv, Sequence) and not isinstance(raw_argv, str | bytes):
        if not 1 <= len(raw_argv) <= 64 or any(not isinstance(token, str) for token in raw_argv):
            return None, False, False
        argv = tuple(token for token in raw_argv if isinstance(token, str))
        if sum(len(token) for token in argv) > 16_000 or any(
            _contains_display_control(token) for token in argv
        ):
            return None, False, False
        visible, replay_redacted = redact_command_argv(argv)
        declared_redacted = _mapping_bool(invocation, "credentials_redacted") is True
        argument_count = _mapping_int(invocation, "argument_count", maximum=64)
        complete = argument_count == len(visible) - 1
        if replay_redacted and not declared_redacted:
            complete = False
        return (
            " ".join(_quote_command_token(token) for token in visible),
            (declared_redacted or replay_redacted),
            complete,
        )

    # Backwards compatibility for traces written before argv disclosure existed.
    raw_executable = _mapping_string(invocation, "executable")
    executable = _safe_executable(raw_executable)
    if executable is None:
        return None, False, False
    argument_count = _mapping_int(invocation, "argument_count", maximum=64)
    if argument_count:
        return f"{executable} ({argument_count} argument(s) unavailable in old trace)", False, False
    return executable, False, False


def _quote_command_token(value: str) -> str:
    if value and re.fullmatch(r"[^\s\"']+", value) is not None:
        return value
    return json.dumps(value, ensure_ascii=False)


def _append_fact(
    facts: list[ActivityFact],
    label: str,
    value: str | None,
    *,
    format: ActivityFactFormat,
) -> bool:
    if value is None:
        return True
    if format == "pre" and not value.strip():
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


def _mapping_number(
    data: Mapping[object, object],
    key: str,
    *,
    minimum: float,
    maximum: float,
) -> float | None:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if minimum <= numeric <= maximum else None


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


def _safe_audit_text(value: str | None) -> tuple[str | None, bool]:
    if value is None or len(value) > 1_000:
        return None, False
    normalized = value.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    printable = "".join(
        " " if not character.isprintable() else character for character in normalized
    )
    collapsed = " ".join(printable.split())
    if not collapsed:
        return None, False
    return redact_credential_values(collapsed)


def _contains_display_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


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
