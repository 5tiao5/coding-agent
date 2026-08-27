"""Validation tests for provider-neutral values crossing core boundaries."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from coding_agent.models import (
    AgentResult,
    AgentState,
    ChatMessage,
    MessageRole,
    ModelResponse,
    StopReason,
    ToolCall,
    ToolExecution,
)


@pytest.mark.parametrize(
    ("message", "error_text"),
    [
        (lambda: ChatMessage(role=MessageRole.SYSTEM), "system messages require content"),
        (lambda: ChatMessage(role=MessageRole.USER), "user messages require content"),
        (
            lambda: ChatMessage(role=MessageRole.ASSISTANT),
            "assistant messages require content or tool calls",
        ),
        (
            lambda: ChatMessage(role=MessageRole.TOOL, content="result"),
            "tool messages require content, tool_call_id, and tool_name",
        ),
        (
            lambda: ChatMessage(
                role=MessageRole.USER,
                content="task",
                tool_calls=(ToolCall(id="call-1", name="echo"),),
            ),
            "only assistant messages may contain tool calls",
        ),
        (
            lambda: ChatMessage(
                role=MessageRole.ASSISTANT,
                content="done",
                tool_call_id="call-1",
            ),
            "only tool messages may set tool_call_id or tool_name",
        ),
    ],
    ids=[
        "system-without-content",
        "user-without-content",
        "empty-assistant",
        "incomplete-tool-message",
        "tool-call-on-user",
        "tool-id-on-assistant",
    ],
)
def test_chat_message_rejects_role_shape_mismatches(
    message: object,
    error_text: str,
) -> None:
    with pytest.raises(ValidationError, match=error_text):
        message()  # type: ignore[operator]


def test_assistant_and_tool_messages_accept_their_valid_shapes() -> None:
    call = ToolCall(id="call-1", name="echo", arguments={"text": "hello"})

    assistant = ChatMessage(role=MessageRole.ASSISTANT, tool_calls=(call,))
    tool = ChatMessage(
        role=MessageRole.TOOL,
        content='{"ok":true,"output":"hello"}',
        tool_call_id=call.id,
        tool_name=call.name,
    )

    assert assistant.tool_calls == (call,)
    assert tool.tool_call_id == call.id
    assert tool.tool_name == call.name


def test_core_models_require_non_empty_identifiers_and_outputs() -> None:
    with pytest.raises(ValidationError):
        ToolCall(id="", name="echo")
    with pytest.raises(ValidationError):
        ToolCall(id="call-1", name="")
    with pytest.raises(ValidationError, match="model response requires content or tool calls"):
        ModelResponse()


def test_tool_execution_enforces_success_and_failure_invariants() -> None:
    success = ToolExecution(call_id="call-1", tool_name="echo", ok=True, output="hello")
    empty_success = ToolExecution(call_id="call-empty", tool_name="echo", ok=True, output="")
    failure = ToolExecution(
        call_id="call-2",
        tool_name="echo",
        ok=False,
        error_code="invalid_arguments",
        error_message="text is required",
    )

    assert success.as_message_content() == (
        '{"call_id":"call-1","tool_name":"echo","ok":true,"output":"hello"}'
    )
    assert empty_success.output == ""
    assert "invalid_arguments" in failure.as_message_content()

    with pytest.raises(ValidationError, match="successful tool executions cannot contain an error"):
        ToolExecution(
            call_id="call-3",
            tool_name="echo",
            ok=True,
            error_code="impossible",
        )
    with pytest.raises(
        ValidationError,
        match="failed tool executions require an error code and message",
    ):
        ToolExecution(call_id="call-4", tool_name="echo", ok=False)


@pytest.mark.parametrize(
    "execution",
    [
        lambda: ToolExecution(call_id="call-1", tool_name="echo", ok=True),
        lambda: ToolExecution(
            call_id="call-1",
            tool_name="echo",
            ok=True,
            output="hello",
            error_message="success cannot also be an error",
        ),
        lambda: ToolExecution(
            call_id="call-1",
            tool_name="echo",
            ok=False,
            output="partial output",
            error_code="tool_error",
            error_message="failed",
        ),
    ],
    ids=["success-without-output", "success-with-error-message", "failure-with-output"],
)
def test_tool_execution_rejects_ambiguous_outcomes(execution: object) -> None:
    with pytest.raises(ValidationError):
        execution()  # type: ignore[operator]


def _result_messages() -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(role=MessageRole.SYSTEM, content="instructions"),
        ChatMessage(role=MessageRole.USER, content="task"),
    )


def test_agent_result_accepts_coherent_terminal_outcomes() -> None:
    final = AgentResult(
        run_id="run-final",
        state=AgentState.COMPLETED_UNVERIFIED,
        stop_reason=StopReason.FINAL_RESPONSE,
        steps=1,
        final_text="done",
        messages=_result_messages(),
    )
    failed = AgentResult(
        run_id="run-failed",
        state=AgentState.FAILED,
        stop_reason=StopReason.MAX_STEPS,
        steps=2,
        error="step budget exhausted",
        messages=_result_messages(),
    )

    assert final.final_text == "done"
    assert failed.error == "step budget exhausted"


@pytest.mark.parametrize(
    "result",
    [
        lambda: AgentResult(
            run_id="run-1",
            state=AgentState.FAILED,
            stop_reason=StopReason.FINAL_RESPONSE,
            steps=1,
            final_text="done",
            messages=_result_messages(),
        ),
        lambda: AgentResult(
            run_id="run-2",
            state=AgentState.COMPLETED_UNVERIFIED,
            stop_reason=StopReason.FINAL_RESPONSE,
            steps=1,
            messages=_result_messages(),
        ),
        lambda: AgentResult(
            run_id="run-3",
            state=AgentState.COMPLETED_UNVERIFIED,
            stop_reason=StopReason.FINAL_RESPONSE,
            steps=1,
            final_text="done",
            error="contradictory error",
            messages=_result_messages(),
        ),
        lambda: AgentResult(
            run_id="run-4",
            state=AgentState.COMPLETED_UNVERIFIED,
            stop_reason=StopReason.MODEL_ERROR,
            steps=1,
            error="provider failed",
            messages=_result_messages(),
        ),
        lambda: AgentResult(
            run_id="run-5",
            state=AgentState.FAILED,
            stop_reason=StopReason.MAX_STEPS,
            steps=2,
            messages=_result_messages(),
        ),
        lambda: AgentResult(
            run_id="run-6",
            state=AgentState.FAILED,
            stop_reason=StopReason.USER_INTERRUPTED,
            steps=1,
            final_text="should not coexist",
            error="interrupted",
            messages=_result_messages(),
        ),
        lambda: AgentResult(
            run_id="",
            state=AgentState.COMPLETED_UNVERIFIED,
            stop_reason=StopReason.FINAL_RESPONSE,
            steps=1,
            final_text="done",
            messages=_result_messages(),
        ),
        lambda: AgentResult(
            run_id="run-8",
            state=AgentState.COMPLETED_UNVERIFIED,
            stop_reason=StopReason.FINAL_RESPONSE,
            steps=1,
            final_text="done",
            messages=(ChatMessage(role=MessageRole.USER, content="task"),),
        ),
    ],
    ids=[
        "final-response-with-failed-state",
        "final-response-without-text",
        "final-response-with-error",
        "failure-with-completed-state",
        "failure-without-error",
        "failure-with-final-text",
        "empty-run-id",
        "missing-system-message",
    ],
)
def test_agent_result_rejects_incoherent_terminal_outcomes(result: object) -> None:
    with pytest.raises(ValidationError):
        result()  # type: ignore[operator]


def test_boundary_models_are_immutable() -> None:
    message = ChatMessage(role=MessageRole.USER, content="original")

    with pytest.raises(ValidationError, match="frozen"):
        message.content = "changed"
