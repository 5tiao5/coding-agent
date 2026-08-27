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


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


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
    error_code: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> ToolExecution:
        if self.ok and (self.error_code is not None or self.error_message is not None):
            raise ValueError("successful tool executions cannot contain an error")
        if self.ok and self.output is None:
            raise ValueError("successful tool executions require output")
        if not self.ok and (not self.error_code or not self.error_message):
            raise ValueError("failed tool executions require an error code and message")
        if not self.ok and self.output is not None:
            raise ValueError("failed tool executions cannot contain output")
        return self

    def as_message_content(self) -> str:
        return self.model_dump_json(exclude_none=True)


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
