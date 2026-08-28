"""Versioned, bounded checkpoints for explicit session resume.

The store is deliberately passive: loading a checkpoint returns data only.  It never
executes or replays a tool call, and verification evidence is intentionally outside the
persisted schema so a resumed run must establish fresh evidence.
"""

from __future__ import annotations

import json
import os
import re
import stat
from collections.abc import Sequence
from contextlib import suppress
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from coding_agent.errors import CodedError
from coding_agent.models import ChatMessage, FrozenModel, MessageRole, StopReason
from coding_agent.run_id import require_run_id

SESSION_SCHEMA_VERSION = 2
DEFAULT_MAX_CHECKPOINT_BYTES = 2_000_000

_FORBIDDEN_FIELD_NAMES = {
    "accesstoken",
    "analysis",
    "apikey",
    "authorization",
    "chainofthought",
    "env",
    "environment",
    "hiddenreasoning",
    "hiddenthoughts",
    "password",
    "reasoning",
    "reasoningcontent",
    "refreshtoken",
    "secret",
    "token",
    "verificationevidence",
    "verificationreport",
}
_MESSAGE_FIELD_NAMES = {"role", "content", "tool_calls", "tool_call_id", "tool_name"}
_TOOL_CALL_FIELD_NAMES = {"id", "name", "arguments"}
_CREDENTIAL_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|password|secret)\s*[:=]\s*"
        r"[\"']?[^\s\"']{8,}"
    ),
)


class SessionError(CodedError):
    """Stable checkpoint failure that does not expose raw filesystem details."""


class SessionBoundary(StrEnum):
    """Stable points at which a canonical transcript may be persisted."""

    READY_FOR_MODEL = "ready_for_model"
    TERMINAL = "terminal"


class SessionCheckpoint(FrozenModel):
    """Provider-neutral state captured only between complete model/tool turns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2] = 2
    run_id: str = Field(min_length=1, max_length=64)
    workspace_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    task: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)
    messages: tuple[ChatMessage, ...] = Field(min_length=2)
    completed_steps: int = Field(ge=0)
    completed_tool_calls: int = Field(ge=0)
    stop_boundary: SessionBoundary
    stop_reason: StopReason | None = None

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        _require_safe_run_id(value)
        return value

    @field_validator("task", "system_prompt")
    @classmethod
    def reject_blank_anchors(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("checkpoint anchors cannot be blank")
        return value

    @model_validator(mode="after")
    def validate_stable_transcript(self) -> Self:
        system, task = self.messages[:2]
        if system.role is not MessageRole.SYSTEM or task.role is not MessageRole.USER:
            raise ValueError("checkpoint must start with system and original user messages")
        if system.content != self.system_prompt or task.content != self.task:
            raise ValueError("checkpoint anchors must exactly match system_prompt and task")

        completed_steps, completed_tool_calls = _validate_closed_turns(self.messages)
        if self.completed_steps != completed_steps:
            raise ValueError("completed_steps does not match the canonical transcript")
        if self.completed_tool_calls != completed_tool_calls:
            raise ValueError("completed_tool_calls does not match the canonical transcript")

        last = self.messages[-1]
        if self.stop_boundary is SessionBoundary.READY_FOR_MODEL:
            if self.stop_reason is not None:
                raise ValueError("ready_for_model checkpoints cannot contain a stop reason")
            if last.role not in {MessageRole.USER, MessageRole.TOOL}:
                raise ValueError("ready_for_model checkpoints must end before a model request")
        else:
            if self.stop_reason is None:
                raise ValueError("terminal checkpoints require an explicit stop reason")
            final_response = last.role is MessageRole.ASSISTANT and not last.tool_calls
            if (self.stop_reason is StopReason.FINAL_RESPONSE) != final_response:
                raise ValueError("final_response stop reason must match a final assistant message")
        return self


class LoadedSession(FrozenModel):
    """A passive load result with explicit resume safety labels."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    checkpoint: SessionCheckpoint
    verification_evidence_restored: Literal[False] = False
    requires_reverification: Literal[True] = True
    auto_replay_tool_calls: Literal[False] = False


class SessionStore:
    """Persist checkpoints atomically in a caller-selected external state directory."""

    def __init__(
        self,
        state_dir: Path,
        *,
        max_checkpoint_bytes: int = DEFAULT_MAX_CHECKPOINT_BYTES,
        workspace_root: Path | None = None,
    ) -> None:
        raw_state_dir = Path(state_dir)
        if not raw_state_dir.is_absolute():
            raise ValueError("state_dir must be an absolute path outside the workspace")
        if max_checkpoint_bytes < 1:
            raise ValueError("max_checkpoint_bytes must be at least 1")
        if raw_state_dir.is_symlink():
            raise ValueError("state_dir cannot be a symbolic link")

        self._state_dir = raw_state_dir.resolve(strict=False)
        self._max_checkpoint_bytes = max_checkpoint_bytes
        self._workspace_fingerprint: str | None = None
        if workspace_root is not None:
            try:
                resolved_workspace = Path(workspace_root).resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise ValueError("workspace_root must identify an accessible directory") from exc
            if not resolved_workspace.is_dir():
                raise ValueError("workspace_root must identify an accessible directory")
            if self._state_dir == resolved_workspace or self._state_dir.is_relative_to(
                resolved_workspace
            ):
                raise ValueError("state_dir must be outside the workspace")
            self._workspace_fingerprint = workspace_fingerprint(resolved_workspace)

    @property
    def state_dir(self) -> Path:
        return self._state_dir

    @property
    def max_checkpoint_bytes(self) -> int:
        return self._max_checkpoint_bytes

    @property
    def workspace_fingerprint(self) -> str | None:
        """Opaque identity required when this store is bound to one workspace."""
        return self._workspace_fingerprint

    def save(self, checkpoint: SessionCheckpoint) -> Path:
        """Atomically replace one checkpoint after validating its safe JSON payload."""

        self._require_matching_workspace(checkpoint)

        payload_object = checkpoint.model_dump(mode="json", exclude_none=True)
        _reject_unsafe_payload(payload_object)
        payload = checkpoint.model_dump_json(exclude_none=True).encode("utf-8")
        if len(payload) > self._max_checkpoint_bytes:
            raise SessionError(
                "checkpoint_too_large",
                "checkpoint exceeds the configured size limit",
                metadata={
                    "size_bytes": len(payload),
                    "max_bytes": self._max_checkpoint_bytes,
                },
            )

        target = self._checkpoint_path(checkpoint.run_id)
        self._prepare_state_directory()
        temporary = self._state_dir / f".{target.name}.{uuid4().hex}.tmp"
        descriptor: int | None = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            descriptor = os.open(temporary, flags, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except OSError as exc:
            if descriptor is not None:
                os.close(descriptor)
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
            raise SessionError("checkpoint_io", "checkpoint could not be saved") from exc

        return target

    def load(self, run_id: str) -> LoadedSession:
        """Load inert checkpoint data; never replay calls or restore verification evidence."""

        target = self._checkpoint_path(run_id)
        raw = self._read_bounded(target)
        try:
            text = raw.decode("utf-8")
            decoded = json.loads(text, object_pairs_hook=_unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
            raise SessionError("checkpoint_corrupt", "checkpoint is not valid JSON") from exc
        if not isinstance(decoded, dict):
            raise SessionError("checkpoint_corrupt", "checkpoint root must be a JSON object")

        version = decoded.get("schema_version")
        if type(version) is not int or version != SESSION_SCHEMA_VERSION:
            raise SessionError(
                "checkpoint_version",
                "checkpoint schema version is not supported",
                metadata={"supported_version": SESSION_SCHEMA_VERSION},
            )
        _reject_unsafe_payload(decoded)
        _reject_unknown_nested_fields(decoded)
        try:
            checkpoint = SessionCheckpoint.model_validate(decoded)
        except ValidationError as exc:
            raise SessionError(
                "checkpoint_corrupt",
                "checkpoint does not satisfy the session schema",
            ) from exc
        if checkpoint.run_id != run_id:
            raise SessionError(
                "checkpoint_corrupt",
                "checkpoint run ID does not match its filename",
            )
        self._require_matching_workspace(checkpoint)
        return LoadedSession(checkpoint=checkpoint)

    def _require_matching_workspace(self, checkpoint: SessionCheckpoint) -> None:
        expected = self._workspace_fingerprint
        if expected is not None and checkpoint.workspace_fingerprint != expected:
            raise SessionError(
                "checkpoint_workspace_mismatch",
                "checkpoint belongs to a different workspace",
            )

    def _checkpoint_path(self, run_id: str) -> Path:
        _require_safe_run_id(run_id)
        target = self._state_dir / f"{run_id}.json"
        if target.parent != self._state_dir:
            raise SessionError("invalid_run_id", "run_id does not map to the state directory")
        return target

    def _prepare_state_directory(self) -> None:
        try:
            self._state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as exc:
            raise SessionError("checkpoint_io", "state directory could not be created") from exc
        if self._state_dir.is_symlink() or not self._state_dir.is_dir():
            raise SessionError("unsafe_state_dir", "state directory is not a regular directory")

    def _read_bounded(self, target: Path) -> bytes:
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        if target.is_symlink():
            raise SessionError("unsafe_checkpoint", "checkpoint cannot be a symbolic link")
        try:
            descriptor = os.open(target, flags)
        except FileNotFoundError as exc:
            raise SessionError("checkpoint_not_found", "checkpoint does not exist") from exc
        except OSError as exc:
            raise SessionError("checkpoint_io", "checkpoint could not be opened") from exc

        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
                raise SessionError("unsafe_checkpoint", "checkpoint is not a regular private file")
            if file_stat.st_size > self._max_checkpoint_bytes:
                raise SessionError(
                    "checkpoint_too_large",
                    "checkpoint exceeds the configured size limit",
                    metadata={
                        "size_bytes": file_stat.st_size,
                        "max_bytes": self._max_checkpoint_bytes,
                    },
                )
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = -1
                payload = stream.read(self._max_checkpoint_bytes + 1)
        finally:
            if descriptor >= 0:
                os.close(descriptor)

        if len(payload) > self._max_checkpoint_bytes:
            raise SessionError(
                "checkpoint_too_large",
                "checkpoint exceeds the configured size limit",
                metadata={"max_bytes": self._max_checkpoint_bytes},
            )
        return payload


def _validate_closed_turns(messages: Sequence[ChatMessage]) -> tuple[int, int]:
    cursor = 2
    completed_steps = 0
    completed_tool_calls = 0
    seen_call_ids: set[str] = set()
    while cursor < len(messages):
        assistant = messages[cursor]
        if assistant.role is not MessageRole.ASSISTANT:
            raise ValueError("canonical transcript contains an orphan tool or unexpected role")
        completed_steps += 1
        cursor += 1

        if not assistant.tool_calls:
            if cursor != len(messages):
                raise ValueError("a final assistant message must end the canonical transcript")
            continue

        for call in assistant.tool_calls:
            if call.id in seen_call_ids:
                raise ValueError("canonical transcript contains a duplicate tool call id")
            seen_call_ids.add(call.id)
            if cursor >= len(messages):
                raise ValueError("checkpoint cannot contain pending tool calls")
            result = messages[cursor]
            if result.role is not MessageRole.TOOL:
                raise ValueError("checkpoint cannot split an assistant/tool block")
            if result.tool_call_id != call.id or result.tool_name != call.name:
                raise ValueError("tool result does not match its canonical tool call")
            completed_tool_calls += 1
            cursor += 1
    return completed_steps, completed_tool_calls


def _require_safe_run_id(run_id: str) -> None:
    try:
        require_run_id(run_id)
    except ValueError:
        raise SessionError(
            "invalid_run_id",
            "run_id must be a lowercase safe 1-64 character file identifier",
        ) from None


def workspace_fingerprint(root: Path) -> str:
    """Bind a checkpoint to one resolved directory without persisting its raw path."""
    try:
        resolved = Path(root).resolve(strict=True)
        identity = resolved.stat()
    except (OSError, RuntimeError) as exc:
        raise ValueError("workspace root is not accessible") from exc
    if not resolved.is_dir():
        raise ValueError("workspace root must be a directory")
    payload = json.dumps(
        {
            "device": identity.st_dev,
            "inode": identity.st_ino,
            "path": os.path.normcase(str(resolved)),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_unsafe_payload(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise SessionError("unsafe_checkpoint", "checkpoint contains a non-string field")
            normalized = "".join(character for character in key.casefold() if character.isalnum())
            if normalized in _FORBIDDEN_FIELD_NAMES:
                raise SessionError(
                    "unsafe_checkpoint",
                    "checkpoint contains a forbidden secret or private-reasoning field",
                    metadata={"field": key},
                )
            _reject_unsafe_payload(nested)
        return
    if isinstance(value, list | tuple):
        for nested in value:
            _reject_unsafe_payload(nested)
        return
    if isinstance(value, str) and any(pattern.search(value) for pattern in _CREDENTIAL_PATTERNS):
        raise SessionError(
            "unsafe_checkpoint",
            "checkpoint contains text matching a credential pattern",
        )


def _reject_unknown_nested_fields(payload: dict[str, object]) -> None:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict):
            continue
        unknown_message_fields = set(message).difference(_MESSAGE_FIELD_NAMES)
        if unknown_message_fields:
            raise SessionError(
                "checkpoint_corrupt",
                "checkpoint message contains fields outside the versioned schema",
            )
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if isinstance(call, dict) and set(call).difference(_TOOL_CALL_FIELD_NAMES):
                raise SessionError(
                    "checkpoint_corrupt",
                    "checkpoint tool call contains fields outside the versioned schema",
                )
