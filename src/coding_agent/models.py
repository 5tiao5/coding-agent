"""Provider-neutral domain models used by the agent core."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    """Immutable base model for values passed across architectural boundaries."""

    model_config = ConfigDict(frozen=True)


class AgentState(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    ACTING = "acting"
    OBSERVING = "observing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    COMPLETED_UNVERIFIED = "completed_unverified"
    FAILED = "failed"


class StopReason(StrEnum):
    FINAL_RESPONSE = "final_response"
    MAX_STEPS = "max_steps"
    TOOL_LIMIT = "tool_limit"
    MODEL_ERROR = "model_error"
    USER_INTERRUPTED = "user_interrupted"
    COMMAND_CONTROL_FAILED = "command_control_failed"
    CONTEXT_LIMIT = "context_limit"
    REPEATED_TOOL_CALL = "repeated_tool_call"


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class VerificationSignal(StrEnum):
    """A trusted verification outcome produced by project-owned tool logic."""

    PASSED = "passed"
    FAILED = "failed"


class VerificationKind(StrEnum):
    """A project-recognized class of external verification."""

    TEST = "test"
    BUILD = "build"
    CHECK = "check"


class VerificationStatus(StrEnum):
    """Why a terminal answer is or is not backed by current evidence."""

    VERIFIED = "verified"
    MISSING = "missing"
    FAILED = "failed"
    STALE = "stale"


class ToolCall(FrozenModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(FrozenModel):
    role: MessageRole
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    tool_name: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> ChatMessage:
        if self.role in {MessageRole.SYSTEM, MessageRole.USER} and self.content is None:
            raise ValueError(f"{self.role.value} messages require content")
        if self.role is MessageRole.ASSISTANT and not (self.content or self.tool_calls):
            raise ValueError("assistant messages require content or tool calls")
        if self.role is MessageRole.TOOL:
            if self.content is None or not self.tool_call_id or not self.tool_name:
                raise ValueError("tool messages require content, tool_call_id, and tool_name")
        elif self.tool_call_id is not None or self.tool_name is not None:
            raise ValueError("only tool messages may set tool_call_id or tool_name")
        if self.role is not MessageRole.ASSISTANT and self.tool_calls:
            raise ValueError("only assistant messages may contain tool calls")
        return self


class ToolSpec(FrozenModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    input_schema: dict[str, Any]


ToolMetadataValue = str | int | float | bool | None


class ToolControlFacts(FrozenModel):
    """Runtime control facts that are never exposed as model-authored context."""

    invalidates_verification: bool = False
    made_progress: bool = False
    verification: VerificationSignal | None = None
    verification_kind: VerificationKind | None = None
    verification_label: str | None = Field(default=None, min_length=1, max_length=120)
    terminal_stop: bool = False
    terminal_reason: str | None = Field(default=None, min_length=1, max_length=500)

    @model_validator(mode="after")
    def validate_verification_label(self) -> ToolControlFacts:
        verification_fields = (
            self.verification,
            self.verification_kind,
            self.verification_label,
        )
        if any(value is None for value in verification_fields) and any(
            value is not None for value in verification_fields
        ):
            raise ValueError("verification, verification_kind, and label must be set together")
        if self.terminal_stop != (self.terminal_reason is not None):
            raise ValueError("terminal_stop and terminal_reason must be set together")
        return self


class ToolOutput(FrozenModel):
    """A tool result split into model content and a safe observable summary."""

    content: str
    summary: str = Field(min_length=1, max_length=500)
    metadata: dict[str, ToolMetadataValue] = Field(default_factory=dict)
    truncated: bool = False
    control: ToolControlFacts = Field(default_factory=ToolControlFacts)


class ModelResponse(FrozenModel):
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()

    @model_validator(mode="after")
    def require_output(self) -> ModelResponse:
        if not (self.content or self.tool_calls):
            raise ValueError("model response requires content or tool calls")
        return self


class ToolExecution(FrozenModel):
    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    ok: bool
    output: str | None = None
    summary: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict[str, ToolMetadataValue] = Field(default_factory=dict)
    truncated: bool = False
    control: ToolControlFacts = Field(default_factory=ToolControlFacts)
    duration_ms: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def validate_outcome(self) -> ToolExecution:
        if self.ok and (self.error_code is not None or self.error_message is not None):
            raise ValueError("successful tool executions cannot contain an error")
        if self.ok and (self.output is None or not self.summary):
            raise ValueError("successful tool executions require output and summary")
        if not self.ok and (not self.error_code or not self.error_message):
            raise ValueError("failed tool executions require an error code and message")
        if not self.ok and (self.output is not None or self.summary is not None or self.truncated):
            raise ValueError("failed tool executions cannot contain output, summary, or truncation")
        return self

    def as_message_content(self) -> str:
        # Timing and the sanitized UI summary are observability data, not model context.
        return self.model_dump_json(
            exclude={"control", "duration_ms", "summary"},
            exclude_defaults=True,
            exclude_none=True,
        )


class AgentResult(FrozenModel):
    run_id: str = Field(min_length=1)
    state: AgentState
    stop_reason: StopReason
    steps: int = Field(ge=0)
    final_text: str | None = None
    error: str | None = None
    messages: tuple[ChatMessage, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_terminal_result(self) -> AgentResult:
        completed_states = {AgentState.COMPLETED, AgentState.COMPLETED_UNVERIFIED}
        if self.state in completed_states:
            if self.stop_reason is not StopReason.FINAL_RESPONSE:
                raise ValueError("completed results require a final_response stop reason")
            if not self.final_text or self.error is not None:
                raise ValueError("completed results require final_text and no error")
        elif self.state is AgentState.FAILED:
            if self.stop_reason is StopReason.FINAL_RESPONSE:
                raise ValueError("failed results cannot use the final_response stop reason")
            if not self.error or self.final_text is not None:
                raise ValueError("failed results require an error and no final_text")
        else:
            raise ValueError("agent results must contain a terminal state")
        return self
