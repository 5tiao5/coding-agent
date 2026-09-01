from __future__ import annotations

import json
from typing import cast

import pytest

from coding_agent import agent_protocol as protocol
from coding_agent.completion import (
    CompletionContract,
    TargetRuntime,
    VerificationCheck,
    VerificationProfile,
)
from coding_agent.models import (
    ChatMessage,
    MessageRole,
    ToolCall,
    ToolControlFacts,
    ToolExecution,
    VerificationKind,
    VerificationSignal,
)
from coding_agent.tooling import ToolDispatcher
from coding_agent.verification import VerificationLedger


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


def test_command_event_exposes_only_a_safe_invocation_summary() -> None:
    secret = "TEST_PRIVATE_API_KEY_SENTINEL"
    call = ToolCall(
        id="command",
        name="run_command",
        arguments={
            "argv": [r"C:\Python\python.exe", "script.py", "--api-key", secret],
            "environment": {"OPENAI_API_KEY": secret},
        },
    )
    execution = ToolExecution(
        call_id=call.id,
        tool_name=call.name,
        ok=True,
        output=f"private stdout {secret}",
        summary="Command exited 0 in .",
        metadata={
            "status": "exited",
            "exit_code": 0,
            "cwd": ".",
            "command_class": "general",
            "total_output_bytes": 123,
            "captured_output_bytes": 123,
            "argv": secret,
            "environment": secret,
        },
        control=ToolControlFacts(
            verification=VerificationSignal.PASSED,
            verification_kind=VerificationKind.TEST,
            verification_label="pytest",
        ),
    )

    event_data = protocol.tool_finished_event_data(call, execution, execution)

    assert event_data["public_invocation"] == {
        "executable": "python",
        "argument_count": 3,
    }
    assert event_data["metadata"] == {
        "status": "exited",
        "command_class": "general",
        "cwd": ".",
        "exit_code": 0,
        "total_output_bytes": 123,
        "captured_output_bytes": 123,
    }
    serialized = json.dumps(event_data)
    assert secret not in serialized
    assert "private stdout" not in serialized
    assert "OPENAI_API_KEY" not in serialized


def test_registered_verifier_exposes_identity_without_model_authored_arguments() -> None:
    private_argument = "UNKNOWN_POSITIONAL_VERIFIER_SECRET_SENTINEL"
    call = ToolCall(
        id="verify",
        name="run_command",
        arguments={
            "argv": [
                r"C:\Users\student\.venv\Scripts\python.exe",
                "-I",
                "-B",
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                private_argument,
            ]
        },
    )
    execution = ToolExecution(
        call_id=call.id,
        tool_name=call.name,
        ok=True,
        output="1 passed",
        summary="Command exited 0 in .",
        control=ToolControlFacts(
            verification=VerificationSignal.PASSED,
            verification_kind=VerificationKind.TEST,
            verification_label="pytest",
        ),
    )

    event_data = protocol.tool_finished_event_data(
        call,
        execution,
        execution,
        verification_call=True,
    )

    assert event_data["public_invocation"] == {
        "executable": "python",
        "argument_count": 8,
        "verification_label": "pytest",
        "verification_kind": "test",
    }
    serialized = json.dumps(event_data)
    assert "C:\\Users" not in serialized
    assert private_argument not in serialized
    assert "no:cacheprovider" not in serialized


def test_registered_verifier_never_serializes_any_raw_argument_vector() -> None:
    secrets = (
        "TEST_PRIVATE_VERIFIER_TOKEN_SENTINEL",
        "UNMARKED_POSITIONAL_SECRET_7f3b2a1c",
    )
    argument_sets = tuple(["python", "-m", "pytest", marker] for marker in secrets)
    for index, argv in enumerate(argument_sets):
        call = ToolCall(
            id=f"verify-{index}",
            name="run_command",
            arguments={"argv": argv},
        )
        execution = ToolExecution(
            call_id=call.id,
            tool_name=call.name,
            ok=True,
            output="passed",
            summary="Command exited 0 in .",
        )

        event_data = protocol.tool_finished_event_data(
            call,
            execution,
            execution,
            verification_call=True,
        )

        invocation = event_data["public_invocation"]
        assert isinstance(invocation, dict)
        assert "display_command" not in invocation
        serialized = json.dumps(event_data)
        assert all(secret not in serialized for secret in secrets)
        assert "pytest" not in serialized


def test_unsafe_host_verifier_label_is_omitted_as_a_pair() -> None:
    unsafe_label = "pytest key=sk-proj-1234567890abcdef"
    call = ToolCall(
        id="verify-private-label",
        name="run_command",
        arguments={"argv": ["python", "-m", "pytest", "-q"]},
    )
    execution = ToolExecution(
        call_id=call.id,
        tool_name=call.name,
        ok=True,
        output="passed",
        summary="Command exited 0 in .",
        control=ToolControlFacts(
            verification=VerificationSignal.PASSED,
            verification_kind=VerificationKind.TEST,
            verification_label=unsafe_label,
        ),
    )

    event_data = protocol.tool_finished_event_data(
        call,
        execution,
        execution,
        verification_call=True,
    )

    invocation = event_data["public_invocation"]
    assert isinstance(invocation, dict)
    assert "verification_label" not in invocation
    assert "verification_kind" not in invocation
    assert unsafe_label not in json.dumps(event_data)


@pytest.mark.parametrize(
    "credential",
    [
        "sk-test1234",
        "ghp_1234567890abcdef",
        "github_pat_1234567890abcdef",
        "AKIA1234567890ABCDEF",
        "ASIA1234567890ABCDEF",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature1234567890",
        "key=value",
        "FOO=value",
    ],
)
def test_public_verifier_label_rejects_credential_shapes(credential: str) -> None:
    assert protocol.public_verifier_label(f"pytest {credential}") is None


def test_public_verifier_label_does_not_reject_a_benign_field_name() -> None:
    assert protocol.public_verifier_label("API_KEY format check") == "API_KEY format check"


def test_final_verification_view_redacts_private_labels_without_changing_completion() -> None:
    unsafe_label = "pytest token=ghp_1234567890abcdef"
    ledger = VerificationLedger()
    ledger.observe(
        ToolExecution(
            call_id="verify-private-label",
            tool_name="run_command",
            ok=True,
            output="passed",
            summary="Command exited 0 in .",
            control=ToolControlFacts(
                verification=VerificationSignal.PASSED,
                verification_kind=VerificationKind.TEST,
                verification_label=unsafe_label,
            ),
        ),
        step=1,
    )
    profile = VerificationProfile(
        checks=(VerificationCheck(unsafe_label, VerificationKind.TEST, ("tests",)),),
        required_labels=(unsafe_label,),
        target_runtime=TargetRuntime("configured-python", True),
    )

    view = protocol.final_verification_view(
        ledger,
        profile,
        CompletionContract(required_scopes=("tests",)),
    )

    assert view.verified is True
    assert view.event_data["labels_redacted"] is True
    assert view.event_data["evidence_count"] == 0
    assert view.event_data["evidence"] == []
    for field in (
        "evidence_labels",
        "required_labels",
        "missing_labels",
        "unexpected_labels",
        "mismatched_labels",
    ):
        assert view.event_data[field] == []
    assert unsafe_label not in json.dumps(view.event_data)


def test_verification_scope_view_is_profile_owned_and_bounded() -> None:
    scopes = tuple(f"scope:{index}" for index in range(18))
    profile = VerificationProfile(
        checks=(VerificationCheck("pytest", VerificationKind.TEST, scopes),),
        required_labels=("pytest",),
        target_runtime=TargetRuntime("configured-python", True),
    )

    event_data = protocol.verification_scope_event_data(
        profile,
        label="pytest",
        kind=VerificationKind.TEST,
    )

    assert event_data == {
        "scopes": list(scopes[:16]),
        "scopes_truncated": True,
    }
    assert protocol.verification_scope_event_data(
        profile,
        label="pytest",
        kind=VerificationKind.BUILD,
    ) == {"scopes": [], "scopes_truncated": False}


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
