from __future__ import annotations

from typing import cast

import pytest

from coding_agent import agent_protocol as protocol
from coding_agent.models import ChatMessage, MessageRole, ToolCall, ToolExecution
from coding_agent.tooling import ToolDispatcher


def _base_messages() -> list[ChatMessage]:
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="system"),
        ChatMessage(role=MessageRole.USER, content="task"),
    ]


def _rejected_block(call_id: str) -> tuple[ChatMessage, ChatMessage]:
    call = ToolCall(id=call_id, name="read_file")
    execution = ToolExecution(
        call_id=call.id,
        tool_name=call.name,
        ok=False,
        error_code=protocol.TOOL_BATCH_REJECTED_ERROR_CODE,
        error_message="rejected",
    )
    return (
        ChatMessage(role=MessageRole.ASSISTANT, tool_calls=(call,)),
        ChatMessage(
            role=MessageRole.TOOL,
            content=execution.as_message_content(),
            tool_call_id=call.id,
            tool_name=call.name,
        ),
    )


def test_runtime_limits_replace_the_previous_complete_host_block() -> None:
    first = protocol.with_runtime_limits(
        "base prompt",
        max_model_turns=2,
        max_calls_per_turn=3,
        max_total_tool_calls=4,
    )

    updated = protocol.with_runtime_limits(
        first,
        max_model_turns=5,
        max_calls_per_turn=6,
        max_total_tool_calls=7,
    )

    assert updated.count(protocol.RUNTIME_LIMITS_MARKER) == 1
    assert updated.count(protocol.RUNTIME_LIMITS_END_MARKER) == 1
    assert "Maximum model turns: 2." not in updated
    assert "Maximum model turns: 5." in updated
    assert updated.endswith(protocol.RUNTIME_LIMITS_END_MARKER)


def test_protocol_correction_only_changes_the_transient_system_view() -> None:
    messages = _base_messages()

    corrected = protocol.with_protocol_correction(messages)

    assert messages[0].content == "system"
    assert corrected[0].content == f"system{protocol.PROTOCOL_CORRECTION_INSTRUCTION}"
    assert corrected[1:] == tuple(messages[1:])


def test_cancelled_calls_receive_valid_failure_results() -> None:
    messages = _base_messages()
    calls = (
        ToolCall(id="call-1", name="read_file"),
        ToolCall(id="call-2", name="run_command"),
    )

    protocol.append_cancelled_tool_results(
        messages,
        calls,
        error_code="tool_call_cancelled",
        error_message="cancelled by host",
    )

    assert len(messages) == 4
    for call, message in zip(calls, messages[2:], strict=True):
        assert message.role is MessageRole.TOOL
        assert message.tool_call_id == call.id
        assert message.tool_name == call.name
        assert message.content is not None
        execution = ToolExecution.model_validate_json(message.content)
        assert execution.call_id == call.id
        assert execution.tool_name == call.name
        assert execution.ok is False
        assert execution.error_code == "tool_call_cancelled"
        assert execution.error_message == "cancelled by host"


class _Classifier:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome

    def is_verification_call(self, call: ToolCall) -> bool:
        del call
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return cast(bool, self.outcome)


@pytest.mark.parametrize("error", [TypeError("bad call"), ValueError("bad call")])
def test_verification_classifier_fails_closed_on_expected_errors(error: Exception) -> None:
    call = ToolCall(id="verify", name="run_command")
    dispatcher = cast(ToolDispatcher, _Classifier(error))

    assert protocol.is_verification_call(dispatcher, call) is False


def test_verification_classifier_requires_the_exact_true_singleton() -> None:
    call = ToolCall(id="verify", name="run_command")

    assert protocol.is_verification_call(cast(ToolDispatcher, object()), call) is False
    assert protocol.is_verification_call(cast(ToolDispatcher, _Classifier(1)), call) is False
    assert protocol.is_verification_call(cast(ToolDispatcher, _Classifier(True)), call) is True


def test_trailing_rejection_parser_counts_complete_blocks_and_resets() -> None:
    messages = _base_messages()
    messages.extend(_rejected_block("call-1"))
    messages.extend(_rejected_block("call-2"))

    assert protocol.trailing_tool_batch_rejections(messages) == 2

    messages.append(ChatMessage(role=MessageRole.ASSISTANT, content="done"))
    assert protocol.trailing_tool_batch_rejections(messages) == 0

    malformed = _base_messages()
    call = ToolCall(id="bad", name="read_file")
    malformed.extend(
        (
            ChatMessage(role=MessageRole.ASSISTANT, tool_calls=(call,)),
            ChatMessage(
                role=MessageRole.TOOL,
                content="{not-json",
                tool_call_id=call.id,
                tool_name=call.name,
            ),
        )
    )
    assert protocol.trailing_tool_batch_rejections(malformed) == 0


def test_tool_finished_view_only_previews_allowlisted_mutations() -> None:
    call = ToolCall(id="change", name="replace_text")
    raw_execution = ToolExecution(
        call_id=call.id,
        tool_name=call.name,
        ok=True,
        output="bounded diff",
        summary="Changed one file",
        metadata={"path": "target.py"},
        duration_ms=12.5,
    )
    fitted_execution = raw_execution.model_copy(update={"output": "short view"})

    event_data = protocol.tool_finished_event_data(call, raw_execution, fitted_execution)

    assert event_data == {
        "call_id": "change",
        "tool_name": "replace_text",
        "ok": True,
        "error_code": None,
        "duration_ms": 12.5,
        "output_chars": len("bounded diff"),
        "truncated": False,
        "summary": "Changed one file",
        "observation_chars": len(fitted_execution.as_message_content()),
        "observation_truncated": True,
        "metadata": {"path": "target.py"},
        "preview": "bounded diff",
    }

    private_call = call.model_copy(update={"name": "read_file"})
    private_execution = raw_execution.model_copy(update={"tool_name": "read_file"})
    private_data = protocol.tool_finished_event_data(
        private_call,
        private_execution,
        private_execution,
    )
    assert "preview" not in private_data


def test_protocol_correction_retry_payload_preserves_the_event_contract() -> None:
    assert protocol.protocol_correction_retry_event_data(
        attempt=2,
        max_attempts=3,
        prepared_context_chars=123,
    ) == {
        "attempt": 2,
        "next_attempt": 3,
        "max_attempts": 3,
        "delay_seconds": 0.0,
        "error_code": "model_response_invalid",
        "retry_kind": "protocol_correction",
        "instruction_chars": len(protocol.PROTOCOL_CORRECTION_INSTRUCTION),
        "prepared_context_chars": 123,
    }
