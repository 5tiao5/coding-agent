from __future__ import annotations

import json
from typing import cast

import pytest

from coding_agent import agent_protocol as protocol
from coding_agent._presentation_safety import (
    redact_command_argv,
    redact_credential_values,
)
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


def test_command_event_exposes_auditable_invocation_and_redacted_output() -> None:
    secret = "sk-test12345678"
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
        "argv": [
            r"C:\Python\python.exe",
            "script.py",
            "--api-key",
            "[REDACTED]",
        ],
        "credentials_redacted": True,
        "cwd": ".",
        "timeout_seconds": 120.0,
    }
    assert event_data["public_output"] == {
        "captured_text": "private stdout [REDACTED]",
        "captured_projection_truncated": False,
        "observation_truncated": False,
        "credentials_redacted": True,
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
    assert "private stdout" in serialized
    assert "OPENAI_API_KEY" not in serialized


def test_registered_verifier_exposes_identity_and_non_sensitive_arguments() -> None:
    positional_argument = "route-cost-case"
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
                positional_argument,
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
        "argv": [
            r"C:\Users\student\.venv\Scripts\python.exe",
            "-I",
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            positional_argument,
        ],
        "credentials_redacted": False,
        "cwd": ".",
        "timeout_seconds": 120.0,
        "verification_label": "pytest",
        "verification_kind": "test",
    }
    serialized = json.dumps(event_data)
    assert "C:\\\\Users" in serialized
    assert positional_argument in serialized
    assert "no:cacheprovider" in serialized


def test_registered_verifier_redacts_only_credential_values_from_argv() -> None:
    secret = "ghp_1234567890abcdef"
    call = ToolCall(
        id="verify-secret",
        name="run_command",
        arguments={
            "argv": [
                "python",
                "-m",
                "pytest",
                "route-cost-case",
                "--token",
                secret,
                "seed=42",
                "x=1",
            ]
        },
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
    assert invocation["argv"] == [
        "python",
        "-m",
        "pytest",
        "route-cost-case",
        "--token",
        "[REDACTED]",
        "seed=42",
        "x=1",
    ]
    serialized = json.dumps(event_data)
    assert secret not in serialized
    assert "route-cost-case" in serialized
    assert "seed=42" in serialized
    assert "x=1" in serialized


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Authorization: Bearer abcdefghijk", "Authorization: [REDACTED]"),
        ("OPENAI_API_KEY=randomvalue", "OPENAI_API_KEY=[REDACTED]"),
        ('{"client_secret": "randomvalue"}', '{"client_secret": "[REDACTED]"}'),
        ("seed=42", "seed=42"),
        ("x=1", "x=1"),
    ],
)
def test_credential_redaction_is_value_scoped(source: str, expected: str) -> None:
    redacted, changed = redact_credential_values(source)
    assert redacted == expected
    assert changed is (source != expected)


def test_command_redaction_preserves_benign_flags_and_assignments() -> None:
    argv, changed = redact_command_argv(
        (
            "python",
            "--max-steps",
            "50",
            "--tokenizer",
            "wordpiece",
            "seed=42",
            "x=1",
            "--openai-api-key",
            "randomvalue",
            "--client-secret=another-random-value",
        )
    )
    assert argv == (
        "python",
        "--max-steps",
        "50",
        "--tokenizer",
        "wordpiece",
        "seed=42",
        "x=1",
        "--openai-api-key",
        "[REDACTED]",
        "--client-secret=[REDACTED]",
    )
    assert changed is True


def test_started_command_event_discloses_effective_execution_context() -> None:
    call = ToolCall(
        id="start-audit",
        name="run_command",
        arguments={
            "argv": ["python", "-m", "pytest", "tests/unit case.py"],
            "cwd": "packages/core",
            "timeout_seconds": 45,
            "environment": {"OPENAI_API_KEY": "must-not-enter-event"},
        },
    )

    data = protocol.tool_started_event_data(call)

    assert data["public_invocation"] == {
        "executable": "python",
        "argument_count": 3,
        "argv": ["python", "-m", "pytest", "tests/unit case.py"],
        "credentials_redacted": False,
        "cwd": "packages/core",
        "timeout_seconds": 45.0,
    }
    assert "must-not-enter-event" not in json.dumps(data)


def test_command_event_distinguishes_tool_capture_from_model_observation() -> None:
    call = ToolCall(id="bounded-output", name="run_command", arguments={"argv": ["pytest"]})
    raw = ToolExecution(
        call_id=call.id,
        tool_name=call.name,
        ok=True,
        output="Status: exited\nOutput:\nfirst line\nsecond line",
        summary="Command exited 0 in .",
    )
    observed = raw.model_copy(update={"output": "Status: exited\nOutput:\nfirst line"})

    data = protocol.tool_finished_event_data(call, raw, observed)

    assert data["public_output"] == {
        "captured_text": "first line\nsecond line",
        "captured_projection_truncated": False,
        "observation_truncated": True,
        "credentials_redacted": False,
        "observed_text": "first line",
        "observed_projection_truncated": False,
    }


def test_command_output_redacts_a_sensitive_argv_value_when_printed_bare() -> None:
    secret = "randomvalue"
    call = ToolCall(
        id="echo-token",
        name="run_command",
        arguments={"argv": ["credential-check", "--token", secret]},
    )
    execution = ToolExecution(
        call_id=call.id,
        tool_name=call.name,
        ok=True,
        output=f"Status: exited\nOutput:\ntoken accepted: {secret}",
        summary="Command exited 0 in .",
    )

    data = protocol.tool_finished_event_data(call, execution, execution)

    assert data["public_output"] == {
        "captured_text": "token accepted: [REDACTED]",
        "captured_projection_truncated": False,
        "observation_truncated": False,
        "credentials_redacted": True,
    }
    assert secret not in json.dumps(data)


def test_command_output_redacts_an_assigned_sensitive_argv_value() -> None:
    secret = "another-random-value"
    call = ToolCall(
        id="echo-assigned-token",
        name="run_command",
        arguments={"argv": ["credential-check", f"--client-secret={secret}"]},
    )
    execution = ToolExecution(
        call_id=call.id,
        tool_name=call.name,
        ok=True,
        output=f"Status: exited\nOutput:\ncredential accepted: {secret}",
        summary="Command exited 0 in .",
    )

    data = protocol.tool_finished_event_data(call, execution, execution)

    assert data["public_output"] == {
        "captured_text": "credential accepted: [REDACTED]",
        "captured_projection_truncated": False,
        "observation_truncated": False,
        "credentials_redacted": True,
    }
    assert secret not in json.dumps(data)


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
