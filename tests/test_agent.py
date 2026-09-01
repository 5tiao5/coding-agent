"""Behaviour tests for the project-owned agent loop."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel, Field, TypeAdapter

from coding_agent.agent import AgentRunner
from coding_agent.agent_protocol import EARLY_FINAL_CORRECTION_MARKER
from coding_agent.budget import BudgetPolicy
from coding_agent.cancellation import CancellationSource
from coding_agent.completion import (
    CompletionContract,
    TargetRuntime,
    VerificationCheck,
    VerificationProfile,
)
from coding_agent.context import ContextManager, context_char_count
from coding_agent.events import EventKind, MemoryEventSink
from coding_agent.model import (
    RecoverableModelResponseError,
    RetryableModelError,
    ScriptedModel,
)
from coding_agent.models import (
    AgentState,
    ChatMessage,
    MessageRole,
    ModelResponse,
    StopReason,
    ToolCall,
    ToolControlFacts,
    ToolSpec,
    VerificationKind,
    VerificationSignal,
)
from coding_agent.run_memory import RunMemory
from coding_agent.session import SessionBoundary, SessionStore
from coding_agent.tools import BaseTool, ToolOutput, ToolRegistry

_TOOL_PAYLOAD_ADAPTER = TypeAdapter(dict[str, object])


def test_importing_agent_does_not_load_concrete_tool_adapters() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import coding_agent.agent; "
                "forbidden={'coding_agent.tools', 'coding_agent.tools.filesystem', "
                "'coding_agent.tools.mutation', 'coding_agent.mutation', "
                "'coding_agent.workspace'}; "
                "loaded=sorted(forbidden.intersection(sys.modules)); "
                "print(','.join(loaded)); raise SystemExit(bool(loaded))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert probe.returncode == 0, probe.stdout + probe.stderr


class EchoArgs(BaseModel):
    text: str = Field(min_length=1)
    uppercase: bool = False


class EchoTool(BaseTool[EchoArgs]):
    name = "echo"
    description = "Return the supplied text."
    args_model = EchoArgs

    def __init__(self) -> None:
        self.calls: list[EchoArgs] = []

    def run(self, arguments: EchoArgs) -> ToolOutput:
        self.calls.append(arguments)
        content = arguments.text.upper() if arguments.uppercase else arguments.text
        return ToolOutput(content=content, summary="Echoed test text")


class CancellingEchoTool(EchoTool):
    """Request host cancellation after one deterministic tool execution."""

    def __init__(self, source: CancellationSource) -> None:
        super().__init__()
        self._source = source

    def run(self, arguments: EchoArgs) -> ToolOutput:
        output = super().run(arguments)
        self._source.request_cancellation()
        return output


class OutcomeModel:
    """Return responses or raise failures in a deterministic request sequence."""

    def __init__(self, outcomes: Sequence[ModelResponse | BaseException]) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[tuple[tuple[ChatMessage, ...], tuple[ToolSpec, ...]]] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> ModelResponse:
        self.requests.append((tuple(messages), tuple(tools)))
        if not self._outcomes:
            raise RuntimeError("outcome model has no response remaining")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _tool_payload(model: ScriptedModel, request_index: int = 1) -> dict[str, object]:
    message = model.requests[request_index].messages[-1]
    assert message.role is MessageRole.TOOL
    assert message.content is not None
    return _TOOL_PAYLOAD_ADAPTER.validate_json(message.content)


def test_scripted_model_completes_a_multi_turn_tool_loop() -> None:
    tool = EchoTool()
    model = ScriptedModel(
        [
            ModelResponse(
                content="I will inspect the value.",
                tool_calls=(
                    ToolCall(
                        id="echo-1",
                        name="echo",
                        arguments={"text": "hello", "uppercase": True},
                    ),
                ),
            ),
            ModelResponse(content="The tool returned HELLO."),
        ]
    )
    events = MemoryEventSink()

    result = AgentRunner(
        model,
        ToolRegistry([tool]),
        event_sink=events,
        max_steps=3,
    ).run("Echo hello in uppercase")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert result.steps == 2
    assert result.final_text == "The tool returned HELLO."
    assert [call.model_dump() for call in tool.calls] == [{"text": "hello", "uppercase": True}]
    assert len(model.requests) == 2
    assert [spec.name for spec in model.requests[0].tools] == ["echo"]

    second_request = model.requests[1].messages
    assert [message.role for message in second_request] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]
    assert second_request[-1].tool_call_id == "echo-1"
    assert second_request[-1].tool_name == "echo"
    assert _tool_payload(model) == {
        "call_id": "echo-1",
        "tool_name": "echo",
        "ok": True,
        "output": "HELLO",
    }

    assert [event.kind for event in events.events] == [
        EventKind.RUN_STARTED,
        EventKind.STATE_CHANGED,
        EventKind.STATE_CHANGED,
        EventKind.MODEL_REQUESTED,
        EventKind.MODEL_RESPONDED,
        EventKind.STATE_CHANGED,
        EventKind.TOOL_STARTED,
        EventKind.TOOL_FINISHED,
        EventKind.STATE_CHANGED,
        EventKind.MODEL_REQUESTED,
        EventKind.MODEL_RESPONDED,
        EventKind.STATE_CHANGED,
        EventKind.VERIFICATION_EVALUATED,
        EventKind.STATE_CHANGED,
        EventKind.RUN_FINISHED,
    ]
    assert [event.step for event in events.events] == [
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
    ]
    assert {event.run_id for event in events.events} == {result.run_id}
    assert events.events[-1].data == {
        "verified": False,
        "status": "missing",
        "epoch": 0,
        "invalidation_count": 0,
        "evidence_count": 0,
        "evidence_labels": [],
        "evidence": [],
    }


def test_host_can_inject_a_safe_run_id_before_starting_the_runner() -> None:
    events = MemoryEventSink()
    result = AgentRunner(
        ScriptedModel([ModelResponse(content="Done under a host-owned run ID.")]),
        ToolRegistry(),
        event_sink=events,
    ).run("Use the leased run ID", run_id="host-run")

    assert result.run_id == "host-run"
    assert {event.run_id for event in events.events} == {"host-run"}

    invalid_model = ScriptedModel([ModelResponse(content="must not be requested")])
    with pytest.raises(ValueError, match="run_id must be"):
        AgentRunner(invalid_model, ToolRegistry()).run("Reject the ID", run_id="UPPERCASE")
    assert invalid_model.requests == []


def test_unknown_tool_failure_is_returned_to_the_model_as_an_observation() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(ToolCall(id="missing-1", name="not_registered", arguments={}),)
            ),
            ModelResponse(content="I could not use that tool."),
        ]
    )
    events = MemoryEventSink()

    result = AgentRunner(model, ToolRegistry(), event_sink=events).run("Use a missing tool")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert _tool_payload(model) == {
        "call_id": "missing-1",
        "tool_name": "not_registered",
        "ok": False,
        "error_code": "unknown_tool",
        "error_message": "unknown tool: not_registered",
    }
    finished = [event for event in events.events if event.kind is EventKind.TOOL_FINISHED]
    assert len(finished) == 1
    assert isinstance(finished[0].data["duration_ms"], float)
    event_data = dict(finished[0].data)
    event_data.pop("duration_ms")
    assert event_data == {
        "call_id": "missing-1",
        "tool_name": "not_registered",
        "ok": False,
        "error_code": "unknown_tool",
        "output_chars": 0,
        "truncated": False,
        "summary": None,
        "observation_chars": len(result.messages[3].content or ""),
        "observation_truncated": False,
    }


def test_invalid_tool_arguments_are_reported_without_invoking_the_tool() -> None:
    tool = EchoTool()
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="echo-invalid",
                        name="echo",
                        arguments={"uppercase": True},
                    ),
                )
            ),
            ModelResponse(content="The arguments were invalid."),
        ]
    )

    result = AgentRunner(model, ToolRegistry([tool])).run("Call echo incorrectly")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert tool.calls == []
    payload = _tool_payload(model)
    assert payload["call_id"] == "echo-invalid"
    assert payload["tool_name"] == "echo"
    assert payload["ok"] is False
    assert payload["error_code"] == "invalid_arguments"
    assert "text" in str(payload["error_message"])


def test_model_error_becomes_a_terminal_failed_result() -> None:
    model = ScriptedModel([])
    events = MemoryEventSink()

    result = AgentRunner(model, ToolRegistry(), event_sink=events).run("Do something")

    assert result.state is AgentState.FAILED
    assert result.stop_reason is StopReason.MODEL_ERROR
    assert result.steps == 1
    assert result.final_text is None
    assert result.error == "Model request failed: scripted model has no response remaining"
    assert len(result.messages) == 2
    assert len(model.requests) == 1
    assert [event.kind for event in events.events] == [
        EventKind.RUN_STARTED,
        EventKind.STATE_CHANGED,
        EventKind.STATE_CHANGED,
        EventKind.MODEL_REQUESTED,
        EventKind.STATE_CHANGED,
        EventKind.RUN_FAILED,
    ]
    assert events.events[-1].data == {"stop_reason": StopReason.MODEL_ERROR.value}


def test_retryable_model_failures_use_bounded_exponential_backoff() -> None:
    model = OutcomeModel(
        [
            RetryableModelError("provider_busy", "provider is temporarily unavailable"),
            RetryableModelError("provider_busy", "provider is temporarily unavailable"),
            ModelResponse(content="Recovered after transient failures."),
        ]
    )
    events = MemoryEventSink()
    delays: list[float] = []

    result = AgentRunner(
        model,
        ToolRegistry(),
        event_sink=events,
        max_model_retries=2,
        model_retry_base_delay_seconds=0.25,
        model_retry_sleeper=delays.append,
    ).run("Retry a transient model request")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert len(model.requests) == 3
    assert delays == [0.25, 0.5]
    model_events = [
        event
        for event in events.events
        if event.kind
        in {EventKind.MODEL_REQUESTED, EventKind.MODEL_RETRYING, EventKind.MODEL_RESPONDED}
    ]
    assert [event.kind for event in model_events] == [
        EventKind.MODEL_REQUESTED,
        EventKind.MODEL_RETRYING,
        EventKind.MODEL_REQUESTED,
        EventKind.MODEL_RETRYING,
        EventKind.MODEL_REQUESTED,
        EventKind.MODEL_RESPONDED,
    ]
    assert [event.data for event in model_events if event.kind is EventKind.MODEL_RETRYING] == [
        {
            "attempt": 1,
            "next_attempt": 2,
            "max_attempts": 3,
            "delay_seconds": 0.25,
            "error_code": "model_request_transient",
            "retry_kind": "transport_backoff",
        },
        {
            "attempt": 2,
            "next_attempt": 3,
            "max_attempts": 3,
            "delay_seconds": 0.5,
            "error_code": "model_request_transient",
            "retry_kind": "transport_backoff",
        },
    ]


def test_invalid_model_arguments_request_a_bounded_sanitized_protocol_correction() -> None:
    secret = "TEST_PRIVATE_MALFORMED_RESPONSE_SENTINEL"
    model = OutcomeModel(
        [
            RecoverableModelResponseError(secret, secret),
            ModelResponse(content="Recovered with a valid response."),
        ]
    )
    events = MemoryEventSink()
    delays: list[float] = []
    context_limit = 10_000

    result = AgentRunner(
        model,
        ToolRegistry(),
        event_sink=events,
        max_model_retries=2,
        model_retry_sleeper=delays.append,
        context_manager=ContextManager(max_chars=context_limit),
    ).run("Recover a malformed function call")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.steps == 1
    assert len(model.requests) == 2
    assert delays == []
    first_system = model.requests[0][0][0].content or ""
    corrected_system = model.requests[1][0][0].content or ""
    canonical_system = result.messages[0].content or ""
    assert "[CODING_AGENT_PROTOCOL_CORRECTION]" not in first_system
    assert corrected_system.count("[CODING_AGENT_PROTOCOL_CORRECTION]") == 1
    assert "arguments field of every function call is a valid JSON object" in corrected_system
    assert "not an array or scalar" in corrected_system
    assert "[CODING_AGENT_PROTOCOL_CORRECTION]" not in canonical_system
    assert secret not in corrected_system
    assert secret not in str(events.events)
    assert secret not in str(result)
    retry = [event for event in events.events if event.kind is EventKind.MODEL_RETRYING]
    assert len(retry) == 1
    assert retry[0].message == "Requesting a corrected model protocol response"
    assert retry[0].data == {
        "attempt": 1,
        "next_attempt": 2,
        "max_attempts": 3,
        "delay_seconds": 0.0,
        "error_code": "model_response_invalid",
        "retry_kind": "protocol_correction",
        "instruction_chars": len(corrected_system) - len(first_system),
        "prepared_context_chars": retry[0].data["prepared_context_chars"],
    }
    assert isinstance(retry[0].data["prepared_context_chars"], int)
    assert retry[0].data["prepared_context_chars"] <= context_limit
    assert EventKind.TOOL_STARTED not in [event.kind for event in events.events]
    assert EventKind.TOOL_FINISHED not in [event.kind for event in events.events]


def test_protocol_correction_refuses_to_exceed_the_context_budget() -> None:
    task = "Bound the protocol correction context"
    probe = OutcomeModel(
        [
            RecoverableModelResponseError("invalid", "invalid"),
            ModelResponse(content="Recovered."),
        ]
    )
    probe_result = AgentRunner(
        probe,
        ToolRegistry(),
        context_manager=ContextManager(max_chars=10_000),
    ).run(task)
    assert probe_result.state is AgentState.COMPLETED_UNVERIFIED
    initial_chars = context_char_count(probe.requests[0][0])
    corrected_chars = context_char_count(probe.requests[1][0])
    assert corrected_chars > initial_chars

    secret = "TEST_PRIVATE_CONTEXT_PROTOCOL_SENTINEL"
    model = OutcomeModel([RecoverableModelResponseError(secret, secret)])
    events = MemoryEventSink()
    result = AgentRunner(
        model,
        ToolRegistry(),
        event_sink=events,
        context_manager=ContextManager(max_chars=corrected_chars - 1),
    ).run(task)

    assert result.state is AgentState.FAILED
    assert result.stop_reason is StopReason.CONTEXT_LIMIT
    assert result.error is not None
    assert result.error.startswith("Model response recovery context could not be prepared:")
    assert len(model.requests) == 1
    assert len(result.messages) == 2
    assert "[CODING_AGENT_PROTOCOL_CORRECTION]" not in (result.messages[0].content or "")
    assert EventKind.MODEL_RETRYING not in [event.kind for event in events.events]
    assert EventKind.TOOL_STARTED not in [event.kind for event in events.events]
    assert secret not in str(events.events)
    assert secret not in str(result)


def test_protocol_correction_does_not_spend_a_step_or_tool_budget() -> None:
    tool = EchoTool()
    model = OutcomeModel(
        [
            RecoverableModelResponseError("private-code", "private malformed response"),
            ModelResponse(
                tool_calls=(ToolCall(id="echo-once", name="echo", arguments={"text": "recovered"}),)
            ),
            ModelResponse(content="Finished after one real tool call."),
        ]
    )
    events = MemoryEventSink()

    result = AgentRunner(
        model,
        ToolRegistry([tool]),
        event_sink=events,
        max_steps=2,
        max_model_retries=1,
    ).run("Recover, then use the sole tool budget")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.steps == 2
    assert [call.text for call in tool.calls] == ["recovered"]
    assert len(model.requests) == 3
    assert [event.kind for event in events.events].count(EventKind.MODEL_REQUESTED) == 3
    assert [event.kind for event in events.events].count(EventKind.MODEL_RETRYING) == 1
    assert [event.kind for event in events.events].count(EventKind.TOOL_STARTED) == 1
    assert [message.role for message in result.messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    ]
    assert "[CODING_AGENT_PROTOCOL_CORRECTION]" not in str(result.messages)


def test_exhausted_protocol_corrections_fail_without_tools_or_private_response_data() -> None:
    secret = "TEST_PRIVATE_PROTOCOL_FAILURE_SENTINEL"
    model = OutcomeModel([RecoverableModelResponseError(secret, secret) for _ in range(3)])
    events = MemoryEventSink()
    delays: list[float] = []

    result = AgentRunner(
        model,
        ToolRegistry(),
        event_sink=events,
        max_model_retries=2,
        model_retry_sleeper=delays.append,
    ).run("Exhaust protocol correction attempts")

    assert result.state is AgentState.FAILED
    assert result.stop_reason is StopReason.MODEL_ERROR
    assert result.steps == 1
    assert result.error == (
        "Model returned invalid tool-call arguments after protocol recovery attempts"
    )
    assert len(model.requests) == 3
    assert delays == []
    assert len(result.messages) == 2
    assert all(
        (request[0].content or "").count("[CODING_AGENT_PROTOCOL_CORRECTION]") == 1
        for request, _ in model.requests[1:]
    )
    kinds = [event.kind for event in events.events]
    assert kinds.count(EventKind.MODEL_RETRYING) == 2
    assert EventKind.MODEL_RESPONDED not in kinds
    assert EventKind.TOOL_STARTED not in kinds
    assert EventKind.TOOL_FINISHED not in kinds
    assert secret not in str(events.events)
    assert secret not in str(result)


def test_exhausted_retryable_model_failure_is_terminal() -> None:
    model = OutcomeModel(
        [
            RetryableModelError(
                "TEST_PRIVATE_RETRY_CODE_SENTINEL",
                "TEST_PRIVATE_RETRY_MESSAGE_SENTINEL",
            )
            for _ in range(3)
        ]
    )
    events = MemoryEventSink()
    delays: list[float] = []

    result = AgentRunner(
        model,
        ToolRegistry(),
        event_sink=events,
        max_model_retries=2,
        model_retry_base_delay_seconds=1,
        model_retry_sleeper=delays.append,
    ).run("Exhaust the model retry budget")

    assert result.state is AgentState.FAILED
    assert result.stop_reason is StopReason.MODEL_ERROR
    assert result.error == "Model request failed after transient retries"
    assert len(model.requests) == 3
    assert delays == [1.0, 2.0]
    assert [event.kind for event in events.events].count(EventKind.MODEL_RETRYING) == 2
    assert EventKind.MODEL_RESPONDED not in [event.kind for event in events.events]
    assert "TEST_PRIVATE_RETRY" not in str(events.events)
    assert "TEST_PRIVATE_RETRY" not in str(result)


def test_keyboard_interrupt_during_model_retry_delay_stops_without_another_request() -> None:
    model = OutcomeModel([RetryableModelError("provider_busy", "transient request failure")])
    events = MemoryEventSink()

    def interrupt_delay(_: float) -> None:
        raise KeyboardInterrupt

    result = AgentRunner(
        model,
        ToolRegistry(),
        event_sink=events,
        model_retry_sleeper=interrupt_delay,
    ).run("Interrupt model retry backoff")

    assert result.state is AgentState.FAILED
    assert result.stop_reason is StopReason.USER_INTERRUPTED
    assert result.error == "Run interrupted by user during model retry delay"
    assert len(model.requests) == 1
    assert [event.kind for event in events.events].count(EventKind.MODEL_RETRYING) == 1


def test_keyboard_interrupt_during_model_request_is_not_retried() -> None:
    model = OutcomeModel([KeyboardInterrupt()])
    events = MemoryEventSink()
    delays: list[float] = []

    result = AgentRunner(
        model,
        ToolRegistry(),
        event_sink=events,
        model_retry_sleeper=delays.append,
    ).run("Interrupt the model request")

    assert result.state is AgentState.FAILED
    assert result.stop_reason is StopReason.USER_INTERRUPTED
    assert result.error == "Run interrupted by user"
    assert len(model.requests) == 1
    assert delays == []
    assert EventKind.MODEL_RETRYING not in [event.kind for event in events.events]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_model_retries": -1}, "max_model_retries"),
        ({"max_model_retries": True}, "max_model_retries"),
        ({"model_retry_base_delay_seconds": float("nan")}, "model_retry_base_delay_seconds"),
    ],
)
def test_model_retry_configuration_is_bounded(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AgentRunner(ScriptedModel([]), ToolRegistry(), **kwargs)  # type: ignore[arg-type]


def test_max_steps_stops_a_model_that_only_requests_tools() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(ToolCall(id="echo-1", name="echo", arguments={"text": "one"}),)
            ),
            ModelResponse(
                tool_calls=(ToolCall(id="echo-2", name="echo", arguments={"text": "two"}),)
            ),
        ]
    )
    events = MemoryEventSink()

    result = AgentRunner(
        model,
        ToolRegistry([EchoTool()]),
        event_sink=events,
        max_steps=2,
    ).run("Never finish")

    assert result.state is AgentState.FAILED
    assert result.stop_reason is StopReason.MAX_STEPS
    assert result.steps == 2
    assert result.final_text is None
    assert result.error == "Maximum step count reached: 2"
    assert len(model.requests) == 2
    assert [message.role for message in result.messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]
    assert events.events[-2].kind is EventKind.STATE_CHANGED
    assert events.events[-2].data["current"] == AgentState.FAILED.value
    assert events.events[-1].kind is EventKind.RUN_FAILED
    assert events.events[-1].data == {"stop_reason": StopReason.MAX_STEPS.value}
    assert EventKind.RUN_FINISHED not in [event.kind for event in events.events]


class InterruptingTool(BaseTool[EchoArgs]):
    name = "interrupt"
    description = "Simulate a user interrupt during local tool execution."
    args_model = EchoArgs

    def run(self, arguments: EchoArgs) -> ToolOutput:
        del arguments
        raise KeyboardInterrupt


class ControlArgs(BaseModel):
    passed: bool = True


class CommandProbeArgs(BaseModel):
    argv: list[str] = Field(min_length=1)


class CommandProbeTool(BaseTool[CommandProbeArgs]):
    name = "run_command"
    description = "Return a safe command observation for event-boundary tests."
    args_model = CommandProbeArgs

    def __init__(self, *, verification_label: str | None = None) -> None:
        self._verification_label = verification_label

    def _is_verification(self, arguments: CommandProbeArgs) -> bool:
        del arguments
        return self._verification_label is not None

    def run(self, arguments: CommandProbeArgs) -> ToolOutput:
        del arguments
        control = ToolControlFacts(invalidates_verification=True)
        if self._verification_label is not None:
            control = ToolControlFacts(
                verification=VerificationSignal.PASSED,
                verification_kind=VerificationKind.TEST,
                verification_label=self._verification_label,
            )
        return ToolOutput(
            content="command passed",
            summary="Command exited 0 in .",
            metadata={
                "status": "exited",
                "exit_code": 0,
                "cwd": ".",
                "command_class": (
                    "verifier" if self._verification_label is not None else "general"
                ),
            },
            control=control,
        )


class MutationMarkerTool(BaseTool[ControlArgs]):
    name = "mutation_marker"
    description = "Mark a trusted test mutation."
    args_model = ControlArgs

    def run(self, arguments: ControlArgs) -> ToolOutput:
        del arguments
        return ToolOutput(
            content="changed",
            summary="Marked a mutation",
            control=ToolControlFacts(invalidates_verification=True),
        )


class MemoryWriteTool(BaseTool[ControlArgs]):
    name = "write_file"
    description = "Record a deterministic mutation fact without touching the test workspace."
    args_model = ControlArgs

    def __init__(self) -> None:
        self.calls = 0

    def run(self, arguments: ControlArgs) -> ToolOutput:
        del arguments
        self.calls += 1
        return ToolOutput(
            content="created src/first.py",
            summary="Created first task file",
            metadata={
                "path": "src/first.py",
                "change_id": "chg_0001_first",
                "changed": True,
                "change_kind": "create",
                "before_sha256": None,
                "after_sha256": "a" * 64,
                "added_lines": 1,
                "removed_lines": 0,
                "mutation_revision": 1,
            },
            control=ToolControlFacts(invalidates_verification=True, made_progress=True),
        )


class VerificationMarkerTool(BaseTool[ControlArgs]):
    name = "verification_marker"
    description = "Return trusted test verification evidence."
    args_model = ControlArgs

    def _is_verification(self, arguments: ControlArgs) -> bool:
        del arguments
        return True

    def run(self, arguments: ControlArgs) -> ToolOutput:
        signal = VerificationSignal.PASSED if arguments.passed else VerificationSignal.FAILED
        return ToolOutput(
            content=f"verification {signal.value}",
            summary=f"Verification {signal.value}",
            control=ToolControlFacts(
                verification=signal,
                verification_kind=VerificationKind.TEST,
                verification_label="test-suite",
            ),
        )


class TerminalMarkerTool(BaseTool[ControlArgs]):
    name = "terminal_marker"
    description = "Return a trusted terminal stop fact."
    args_model = ControlArgs

    def run(self, arguments: ControlArgs) -> ToolOutput:
        del arguments
        return ToolOutput(
            content="unsafe child state",
            summary="Could not guarantee child cleanup",
            control=ToolControlFacts(
                invalidates_verification=True,
                terminal_stop=True,
                terminal_reason="Command process cleanup could not be guaranteed",
            ),
        )


class LargeOutputTool(BaseTool[ControlArgs]):
    name = "large_output"
    description = "Return a large deterministic test observation."
    args_model = ControlArgs

    def run(self, arguments: ControlArgs) -> ToolOutput:
        del arguments
        return ToolOutput(content="LARGE_OUTPUT_" * 1000, summary="Returned large output")


@pytest.mark.parametrize(
    ("verification_label", "positional_argument"),
    [
        (None, "ordinary-case"),
        ("pytest", "verifier-case"),
    ],
)
def test_agent_events_disclose_non_sensitive_command_argv(
    verification_label: str | None,
    positional_argument: str,
) -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="command-private-argv",
                        name="run_command",
                        arguments={"argv": ["python", "-m", "pytest", positional_argument]},
                    ),
                )
            ),
            ModelResponse(content="Command finished."),
        ]
    )
    events = MemoryEventSink()

    AgentRunner(
        model,
        ToolRegistry([CommandProbeTool(verification_label=verification_label)]),
        event_sink=events,
    ).run("Run one command and publish its auditable argument vector")

    serialized = json.dumps(
        [event.model_dump(mode="json") for event in events.events],
        ensure_ascii=False,
    )
    assert positional_argument in serialized
    finished = next(event for event in events.events if event.kind is EventKind.TOOL_FINISHED)
    assert finished.data["public_invocation"] == {
        "executable": "python",
        "argument_count": 3,
        "argv": ["python", "-m", "pytest", positional_argument],
        "credentials_redacted": False,
        "cwd": ".",
        "timeout_seconds": 120.0,
        **(
            {
                "verification_label": verification_label,
                "verification_kind": "test",
            }
            if verification_label is not None
            else {}
        ),
    }


def test_agent_redacts_credential_like_verifier_label_from_every_event() -> None:
    private_label = "pytest token=ghp_1234567890abcdef"
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="command-private-label",
                        name="run_command",
                        arguments={"argv": ["python", "-m", "pytest"]},
                    ),
                )
            ),
            ModelResponse(content="Trusted check passed."),
        ]
    )
    events = MemoryEventSink()

    result = AgentRunner(
        model,
        ToolRegistry([CommandProbeTool(verification_label=private_label)]),
        event_sink=events,
    ).run("Run a verifier whose private host label must not enter the trace")

    assert result.state is AgentState.COMPLETED
    serialized = json.dumps(
        [event.model_dump(mode="json") for event in events.events],
        ensure_ascii=False,
    )
    assert private_label not in serialized
    assert "ghp_1234567890abcdef" not in serialized
    recorded = next(
        event for event in events.events if event.kind is EventKind.VERIFICATION_RECORDED
    )
    assert "label" not in recorded.data
    assert recorded.data["labels_redacted"] is True
    for kind in (
        EventKind.VERIFICATION_EVALUATED,
        EventKind.RUN_FINISHED,
    ):
        terminal = next(event for event in events.events if event.kind is kind)
        assert terminal.data["labels_redacted"] is True
        assert terminal.data["evidence_labels"] == []
        assert terminal.data["evidence"] == []


def test_current_trusted_verification_evidence_completes_the_run() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(ToolCall(id="mutate-1", name="mutation_marker", arguments={}),)
            ),
            ModelResponse(
                tool_calls=(ToolCall(id="verify-1", name="verification_marker", arguments={}),)
            ),
            ModelResponse(content="The current test evidence passed."),
        ]
    )
    events = MemoryEventSink()
    registry = ToolRegistry([MutationMarkerTool(), VerificationMarkerTool()])

    result = AgentRunner(model, registry, event_sink=events).run("Change and verify")

    assert result.state is AgentState.COMPLETED
    assert events.events[-1].data["verified"] is True
    assert events.events[-1].data["status"] == "verified"
    assert [event.kind for event in events.events].count(EventKind.VERIFICATION_INVALIDATED) == 1
    recorded = [event for event in events.events if event.kind is EventKind.VERIFICATION_RECORDED]
    assert len(recorded) == 1
    assert recorded[0].data == {
        "call_id": "verify-1",
        "epoch": 1,
        "kind": "test",
        "label": "test-suite",
        "passed": True,
        "scopes": [],
        "scopes_truncated": False,
    }
    assert all("verification_label" not in (message.content or "") for message in result.messages)


def test_completion_contract_distinguishes_passing_checks_from_task_validation() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(ToolCall(id="verify-1", name="verification_marker", arguments={}),)
            ),
            ModelResponse(content="The registered test check passed."),
        ]
    )
    events = MemoryEventSink()
    profile = VerificationProfile(
        checks=(
            VerificationCheck(
                label="test-suite",
                kind=VerificationKind.TEST,
                scopes=("tests",),
            ),
        ),
        required_labels=("test-suite",),
        target_runtime=TargetRuntime(
            runtime_id="configured-python",
            eligible_for_task_validation=True,
        ),
    )

    result = AgentRunner(
        model,
        ToolRegistry([VerificationMarkerTool()]),
        event_sink=events,
        verification_profile=profile,
        completion_contract=CompletionContract(required_scopes=("tests", "types")),
    ).run("Run the available check but require type coverage too")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    evaluated = next(
        event for event in events.events if event.kind is EventKind.VERIFICATION_EVALUATED
    )
    assert evaluated.data["verified"] is False
    assert evaluated.data["status"] == "checks_only"
    assert evaluated.data["checks_passed"] is True
    assert evaluated.data["task_validated"] is False
    assert evaluated.data["completion_status"] == "checks_only"
    assert evaluated.data["required_labels"] == ["test-suite"]
    assert evaluated.data["missing_labels"] == []
    assert evaluated.data["required_scopes"] == ["tests", "types"]
    assert evaluated.data["passed_scopes"] == ["tests"]
    assert evaluated.data["missing_scopes"] == ["types"]
    assert evaluated.data["target_runtime_id"] == "configured-python"
    assert events.events[-1].data == evaluated.data
    recorded = next(
        event for event in events.events if event.kind is EventKind.VERIFICATION_RECORDED
    )
    assert recorded.data["scopes"] == ["tests"]
    assert recorded.data["scopes_truncated"] is False


def test_completion_contract_validates_only_when_every_requirement_is_satisfied() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(ToolCall(id="verify-1", name="verification_marker", arguments={}),)
            ),
            ModelResponse(content="The task contract is satisfied."),
        ]
    )
    events = MemoryEventSink()
    profile = VerificationProfile(
        checks=(VerificationCheck("test-suite", VerificationKind.TEST, ("tests",)),),
        required_labels=("test-suite",),
        target_runtime=TargetRuntime("configured-python", True),
    )

    result = AgentRunner(
        model,
        ToolRegistry([VerificationMarkerTool()]),
        event_sink=events,
        verification_profile=profile,
        completion_contract=CompletionContract(required_scopes=("tests",)),
    ).run("Verify the complete task contract")

    assert result.state is AgentState.COMPLETED
    assert events.events[-1].data["verified"] is True
    assert events.events[-1].data["status"] == "verified"
    assert events.events[-1].data["checks_passed"] is True
    assert events.events[-1].data["task_validated"] is True
    assert events.events[-1].data["completion_status"] == "validated"


def test_completion_profile_and_contract_must_be_configured_together() -> None:
    profile = VerificationProfile(
        checks=(VerificationCheck("pytest", VerificationKind.TEST, ("tests",)),),
        required_labels=("pytest",),
        target_runtime=TargetRuntime("configured-python", True),
    )

    with pytest.raises(ValueError, match="must be provided together"):
        AgentRunner(
            ScriptedModel([ModelResponse(content="unused")]),
            ToolRegistry(),
            verification_profile=profile,
        )
    with pytest.raises(ValueError, match="must be provided together"):
        AgentRunner(
            ScriptedModel([ModelResponse(content="unused")]),
            ToolRegistry(),
            completion_contract=CompletionContract(required_scopes=("tests",)),
        )


def test_mutation_after_passed_verification_makes_the_evidence_stale() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(ToolCall(id="verify-1", name="verification_marker", arguments={}),)
            ),
            ModelResponse(
                tool_calls=(ToolCall(id="mutate-1", name="mutation_marker", arguments={}),)
            ),
            ModelResponse(content="I changed the result after testing."),
        ]
    )
    events = MemoryEventSink()

    result = AgentRunner(
        model,
        ToolRegistry([MutationMarkerTool(), VerificationMarkerTool()]),
        event_sink=events,
    ).run("Verify, then change")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert events.events[-1].data["status"] == "stale"


def test_failed_rerun_replaces_passed_evidence_for_the_same_verifier() -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=(ToolCall(id="verify-pass", name="verification_marker"),)),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="verify-fail",
                        name="verification_marker",
                        arguments={"passed": False},
                    ),
                )
            ),
            ModelResponse(content="The latest test run failed."),
        ]
    )
    events = MemoryEventSink()

    result = AgentRunner(
        model,
        ToolRegistry([VerificationMarkerTool()]),
        event_sink=events,
    ).run("Rerun the same verifier")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert events.events[-1].data["status"] == "failed"
    assert events.events[-1].data["epoch"] == 0
    assert events.events[-1].data["invalidation_count"] == 0
    recorded = [event for event in events.events if event.kind is EventKind.VERIFICATION_RECORDED]
    assert [event.data["passed"] for event in recorded] == [True, False]


def test_terminal_control_fact_stops_with_a_terminal_checkpoint(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(ToolCall(id="terminal-1", name="terminal_marker", arguments={}),)
            )
        ]
    )

    store = SessionStore((tmp_path / "state").resolve())
    result = AgentRunner(
        model,
        ToolRegistry([TerminalMarkerTool()]),
        session_store=store,
    ).run("Trigger a terminal command-control failure")

    assert result.state is AgentState.FAILED
    assert result.stop_reason is StopReason.COMMAND_CONTROL_FAILED
    assert result.error == "Command process cleanup could not be guaranteed"
    assert result.messages[-1].role is MessageRole.TOOL
    checkpoint = store.load(result.run_id).checkpoint
    assert checkpoint.stop_boundary is SessionBoundary.TERMINAL
    assert checkpoint.stop_reason is StopReason.COMMAND_CONTROL_FAILED
    assert checkpoint.completed_tool_calls == 1


def test_verification_ledger_is_fresh_for_each_runner_invocation() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(ToolCall(id="verify-1", name="verification_marker", arguments={}),)
            ),
            ModelResponse(content="First run verified."),
            ModelResponse(content="Second run has no evidence."),
        ]
    )
    runner = AgentRunner(model, ToolRegistry([VerificationMarkerTool()]))

    first = runner.run("First task")
    second = runner.run("Second task")

    assert first.state is AgentState.COMPLETED
    assert second.state is AgentState.COMPLETED_UNVERIFIED


def test_run_memory_is_fresh_for_each_runner_invocation() -> None:
    memory = RunMemory()
    shared_plan = memory.plan_state
    write = MemoryWriteTool()
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=(ToolCall(id="write-first", name="write_file"),)),
            ModelResponse(content="First task finished."),
            ModelResponse(content="Second task needs no tools."),
        ]
    )
    runner = AgentRunner(model, ToolRegistry([write]), run_memory=memory)

    first = runner.run("First task")
    assert first.state is AgentState.COMPLETED_UNVERIFIED
    assert memory.snapshot().file_changes[0].path == "src/first.py"

    second = runner.run("Second task")

    assert second.state is AgentState.COMPLETED_UNVERIFIED
    assert write.calls == 1
    assert memory.plan_state is shared_plan
    assert memory.snapshot().revision == 0
    assert [message.role for message in model.requests[2].messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
    ]


def test_early_final_correction_consumes_a_turn_and_survives_resume(tmp_path: Path) -> None:
    profile = VerificationProfile(
        checks=(VerificationCheck("test-suite", VerificationKind.TEST, ("tests",)),),
        required_labels=("test-suite",),
        target_runtime=TargetRuntime("configured-python", True),
    )
    contract = CompletionContract(required_scopes=("tests",))
    store = SessionStore((tmp_path / "state").resolve())
    initial = AgentRunner(
        ScriptedModel(
            [
                ModelResponse(tool_calls=(ToolCall(id="write-before-final", name="write_file"),)),
                ModelResponse(content="Done before verification."),
            ]
        ),
        ToolRegistry([MemoryWriteTool(), VerificationMarkerTool()]),
        session_store=store,
        max_steps=4,
        verification_profile=profile,
        completion_contract=contract,
    ).run("Change, then verify before finishing")

    assert initial.state is AgentState.FAILED
    checkpoint = store.load(initial.run_id)
    assert checkpoint.checkpoint.stop_boundary is SessionBoundary.READY_FOR_MODEL
    assert checkpoint.checkpoint.completed_steps == 2
    assert checkpoint.checkpoint.messages[-1].role is MessageRole.USER
    assert EARLY_FINAL_CORRECTION_MARKER in (checkpoint.checkpoint.messages[-1].content or "")

    resumed_model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(ToolCall(id="verify-after-resume", name="verification_marker"),)
            ),
            ModelResponse(content="Now verified and complete."),
        ]
    )
    resumed = AgentRunner(
        resumed_model,
        ToolRegistry([MemoryWriteTool(), VerificationMarkerTool()]),
        session_store=store,
        max_steps=4,
        verification_profile=profile,
        completion_contract=contract,
    ).resume(checkpoint)

    assert resumed.state is AgentState.COMPLETED
    assert resumed.steps == 4
    assert "verification calls only" in (resumed_model.requests[0].messages[0].content or "")
    assert (
        sum(
            EARLY_FINAL_CORRECTION_MARKER in (message.content or "") for message in resumed.messages
        )
        == 1
    )


def test_early_final_correction_requires_a_workspace_mutation() -> None:
    profile = VerificationProfile(
        checks=(VerificationCheck("test-suite", VerificationKind.TEST, ("tests",)),),
        required_labels=("test-suite",),
        target_runtime=TargetRuntime("configured-python", True),
    )
    model = ScriptedModel([ModelResponse(content="No workspace changes were needed.")])

    result = AgentRunner(
        model,
        ToolRegistry([VerificationMarkerTool()]),
        max_steps=4,
        verification_profile=profile,
        completion_contract=CompletionContract(required_scopes=("tests",)),
    ).run("Answer without changing files")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert len(model.requests) == 1
    assert all(
        EARLY_FINAL_CORRECTION_MARKER not in (message.content or "") for message in result.messages
    )


def test_runner_compacts_only_the_model_view_and_keeps_canonical_history() -> None:
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=(ToolCall(id="large-1", name="large_output", arguments={}),)),
            ModelResponse(content="The large observation was summarized safely."),
        ]
    )
    events = MemoryEventSink()

    result = AgentRunner(
        model,
        ToolRegistry([LargeOutputTool()]),
        event_sink=events,
        context_manager=ContextManager(max_chars=1_500),
    ).run("Observe a large output")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert "LARGE_OUTPUT_" in (result.messages[3].content or "")
    assert all(
        "LARGE_OUTPUT_" not in (message.content or "") for message in model.requests[1].messages
    )
    compacted = [event for event in events.events if event.kind is EventKind.CONTEXT_COMPACTED]
    assert len(compacted) == 1
    assert compacted[0].data["compacted_blocks"] == 1
    assert compacted[0].data["prepared_chars"] <= 1_500


def test_context_anchor_overflow_stops_before_calling_the_model() -> None:
    model = ScriptedModel([ModelResponse(content="must not be requested")])

    result = AgentRunner(
        model,
        ToolRegistry(),
        context_manager=ContextManager(max_chars=1),
    ).run("Task cannot be truncated")

    assert result.state is AgentState.FAILED
    assert result.stop_reason is StopReason.CONTEXT_LIMIT
    assert result.error is not None
    assert "context_anchor_exceeds_budget" in result.error
    assert model.requests == []


def test_third_identical_tool_observation_stops_a_no_progress_loop() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(ToolCall(id=f"echo-{index}", name="echo", arguments={"text": "same"}),)
            )
            for index in range(1, 4)
        ]
    )
    tool = EchoTool()

    result = AgentRunner(model, ToolRegistry([tool]), max_steps=5).run("Repeat forever")

    assert result.state is AgentState.FAILED
    assert result.stop_reason is StopReason.REPEATED_TOOL_CALL
    assert result.steps == 3
    assert len(tool.calls) == 3
    assert result.error == "The same tool call produced the same observation 3 consecutive times"


def test_repeated_call_stop_closes_and_terminally_checkpoints_the_batch(
    tmp_path: Path,
) -> None:
    calls = tuple(
        ToolCall(id=f"echo-{index}", name="echo", arguments={"text": "same"})
        for index in range(1, 5)
    )
    tool = EchoTool()
    store = SessionStore((tmp_path / "state").resolve())

    result = AgentRunner(
        ScriptedModel([ModelResponse(tool_calls=calls)]),
        ToolRegistry([tool]),
        session_store=store,
    ).run("Repeat four times in one batch")

    assert result.state is AgentState.FAILED
    assert result.stop_reason is StopReason.REPEATED_TOOL_CALL
    assert len(tool.calls) == 3
    tool_messages = result.messages[3:]
    assert [message.tool_call_id for message in tool_messages] == [call.id for call in calls]
    cancelled = _TOOL_PAYLOAD_ADAPTER.validate_json(tool_messages[-1].content or "")
    assert cancelled["ok"] is False
    assert cancelled["error_code"] == "tool_call_cancelled"
    ContextManager(max_chars=100_000).prepare(result.messages)
    checkpoint = store.load(result.run_id).checkpoint
    assert checkpoint.stop_boundary is SessionBoundary.TERMINAL
    assert checkpoint.stop_reason is StopReason.REPEATED_TOOL_CALL
    assert checkpoint.completed_tool_calls == 4


def test_keyboard_interrupt_during_tool_execution_is_a_structured_failure() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(id="interrupt-1", name="interrupt", arguments={"text": "stop"}),
                )
            )
        ]
    )
    events = MemoryEventSink()

    result = AgentRunner(
        model,
        ToolRegistry([InterruptingTool()]),
        event_sink=events,
    ).run("Interrupt the tool")

    assert result.state is AgentState.FAILED
    assert result.stop_reason is StopReason.USER_INTERRUPTED
    assert result.steps == 1
    assert result.error == "Run interrupted by user during tool execution"
    assert [message.role for message in result.messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]
    assert [event.kind for event in events.events[-3:]] == [
        EventKind.TOOL_STARTED,
        EventKind.STATE_CHANGED,
        EventKind.RUN_FAILED,
    ]
    assert EventKind.TOOL_FINISHED not in [event.kind for event in events.events]
    assert events.events[-1].data == {"stop_reason": StopReason.USER_INTERRUPTED.value}


def test_pre_requested_host_cancellation_stops_before_model_and_saves_resume_point(
    tmp_path: Path,
) -> None:
    source = CancellationSource()
    assert source.request_cancellation() is True
    model = ScriptedModel([ModelResponse(content="must not be requested")])
    store = SessionStore((tmp_path / "state").resolve())

    result = AgentRunner(
        model,
        ToolRegistry(),
        session_store=store,
        cancellation_token=source.token,
    ).run("Stop before spending a model request", run_id="host-cancel-before-model")

    assert result.state is AgentState.FAILED
    assert result.stop_reason is StopReason.USER_INTERRUPTED
    assert result.steps == 1
    assert model.requests == []
    checkpoint = store.load(result.run_id).checkpoint
    assert checkpoint.stop_boundary is SessionBoundary.READY_FOR_MODEL
    assert checkpoint.stop_reason is None
    assert checkpoint.completed_steps == 0
    assert checkpoint.completed_tool_calls == 0


def test_cancellation_during_blocking_model_discards_late_final_and_keeps_resume_point(
    tmp_path: Path,
) -> None:
    source = CancellationSource()

    class LateFinalModel:
        def __init__(self) -> None:
            self.calls = 0

        def complete(
            self,
            messages: Sequence[ChatMessage],
            tools: Sequence[ToolSpec],
        ) -> ModelResponse:
            del messages, tools
            self.calls += 1
            source.request_cancellation()
            return ModelResponse(content="late final must not be committed")

    model = LateFinalModel()
    store = SessionStore((tmp_path / "state").resolve())

    result = AgentRunner(
        model,
        ToolRegistry(),
        session_store=store,
        cancellation_token=source.token,
    ).run("Do not accept a final after shutdown", run_id="host-cancel-late-final")

    assert result.state is AgentState.FAILED
    assert result.stop_reason is StopReason.USER_INTERRUPTED
    assert result.final_text is None
    assert model.calls == 1
    assert [message.role for message in result.messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
    ]
    checkpoint = store.load(result.run_id).checkpoint
    assert checkpoint.stop_boundary is SessionBoundary.READY_FOR_MODEL
    assert checkpoint.completed_steps == 0


def test_host_cancellation_between_tool_calls_closes_batch_and_saves_resume_point(
    tmp_path: Path,
) -> None:
    source = CancellationSource()
    tool = CancellingEchoTool(source)
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(id="echo-first", name="echo", arguments={"text": "first"}),
                    ToolCall(id="echo-second", name="echo", arguments={"text": "second"}),
                )
            )
        ]
    )
    store = SessionStore((tmp_path / "state").resolve())

    result = AgentRunner(
        model,
        ToolRegistry([tool]),
        session_store=store,
        cancellation_token=source.token,
    ).run("Stop between calls", run_id="host-cancel-between-tools")

    assert result.state is AgentState.FAILED
    assert result.stop_reason is StopReason.USER_INTERRUPTED
    assert [call.text for call in tool.calls] == ["first"]
    tool_messages = [message for message in result.messages if message.role is MessageRole.TOOL]
    assert [message.tool_call_id for message in tool_messages] == ["echo-first", "echo-second"]
    cancelled = _TOOL_PAYLOAD_ADAPTER.validate_json(tool_messages[-1].content or "")
    assert cancelled["ok"] is False
    assert cancelled["error_code"] == "tool_call_cancelled"
    checkpoint = store.load(result.run_id).checkpoint
    assert checkpoint.stop_boundary is SessionBoundary.READY_FOR_MODEL
    assert checkpoint.stop_reason is None
    assert checkpoint.completed_steps == 1
    assert checkpoint.completed_tool_calls == 2
    ContextManager(max_chars=100_000).prepare(checkpoint.messages)


def test_host_cancellation_wakes_model_retry_backoff_before_another_request(
    tmp_path: Path,
) -> None:
    source = CancellationSource()

    class CancellingRetryModel:
        def __init__(self) -> None:
            self.calls = 0

        def complete(
            self,
            messages: Sequence[ChatMessage],
            tools: Sequence[ToolSpec],
        ) -> ModelResponse:
            del messages, tools
            self.calls += 1
            source.request_cancellation()
            raise RetryableModelError("provider_busy", "temporary provider failure")

    model = CancellingRetryModel()
    store = SessionStore((tmp_path / "state").resolve())

    result = AgentRunner(
        model,
        ToolRegistry(),
        session_store=store,
        cancellation_token=source.token,
        max_model_retries=2,
        model_retry_base_delay_seconds=60,
    ).run("Cancel the retry", run_id="host-cancel-retry")

    assert result.state is AgentState.FAILED
    assert result.stop_reason is StopReason.USER_INTERRUPTED
    assert model.calls == 1
    checkpoint = store.load(result.run_id).checkpoint
    assert checkpoint.stop_boundary is SessionBoundary.READY_FOR_MODEL
    assert checkpoint.completed_steps == 0


def test_per_step_tool_call_limit_rejects_atomically_then_accepts_a_smaller_retry() -> None:
    tool = EchoTool()
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(id="echo-1", name="echo", arguments={"text": "one"}),
                    ToolCall(id="echo-2", name="echo", arguments={"text": "two"}),
                    ToolCall(id="echo-3", name="echo", arguments={"text": "three"}),
                )
            ),
            ModelResponse(
                tool_calls=(ToolCall(id="echo-4", name="echo", arguments={"text": "recovered"}),)
            ),
            ModelResponse(content="Recovered with a smaller batch."),
        ]
    )
    events = MemoryEventSink()

    result = AgentRunner(
        model,
        ToolRegistry([tool]),
        event_sink=events,
        max_steps=3,
        max_tool_calls_per_step=2,
    ).run("Request too many tools at once")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert result.steps == 3
    assert [call.text for call in tool.calls] == ["recovered"]
    assert [message.role for message in result.messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.TOOL,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    ]
    retry_messages = model.requests[1].messages
    rejected_payloads = [
        _TOOL_PAYLOAD_ADAPTER.validate_json(message.content or "")
        for message in retry_messages[-3:]
    ]
    assert [payload["error_code"] for payload in rejected_payloads] == [
        "tool_batch_rejected",
        "tool_batch_rejected",
        "tool_batch_rejected",
    ]
    assert [event.kind for event in events.events].count(EventKind.TOOL_STARTED) == 1
    rejected = [event for event in events.events if event.kind is EventKind.TOOL_BATCH_REJECTED]
    assert [event.data for event in rejected] == [
        {
            "requested_calls": 3,
            "max_calls_per_turn": 2,
            "rejection_count": 1,
            "max_rejections": 3,
        }
    ]


def test_third_consecutive_over_limit_batch_trips_the_tool_limit_circuit() -> None:
    tool = EchoTool()
    outcomes = [
        ModelResponse(
            tool_calls=tuple(
                ToolCall(
                    id=f"batch-{batch}-{index}",
                    name="echo",
                    arguments={"text": f"{batch}-{index}"},
                )
                for index in range(3)
            )
        )
        for batch in range(3)
    ]
    events = MemoryEventSink()

    result = AgentRunner(
        ScriptedModel(outcomes),
        ToolRegistry([tool]),
        event_sink=events,
        max_steps=4,
        max_tool_calls_per_step=2,
    ).run("Keep exceeding the per-turn tool budget")

    assert result.state is AgentState.FAILED
    assert result.stop_reason is StopReason.TOOL_LIMIT
    assert result.steps == 3
    assert result.error == (
        "Model exceeded the per-step tool call limit 3 consecutive times: 3 > 2"
    )
    assert tool.calls == []
    assert EventKind.TOOL_STARTED not in [event.kind for event in events.events]
    rejected = [event for event in events.events if event.kind is EventKind.TOOL_BATCH_REJECTED]
    assert [event.data["rejection_count"] for event in rejected] == [1, 2, 3]
    assert all(
        event.data
        == {
            "requested_calls": 3,
            "max_calls_per_turn": 2,
            "rejection_count": index,
            "max_rejections": 3,
        }
        for index, event in enumerate(rejected, start=1)
    )


def test_runtime_limits_are_visible_to_the_model_and_run_started_event() -> None:
    model = ScriptedModel([ModelResponse(content="Done within budget.")])
    events = MemoryEventSink()

    result = AgentRunner(
        model,
        ToolRegistry(),
        event_sink=events,
        max_steps=7,
        max_tool_calls_per_step=3,
    ).run("Inspect the runtime limits")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    system_message = model.requests[0].messages[0]
    assert system_message.role is MessageRole.SYSTEM
    assert system_message.content is not None
    assert "Maximum model turns: 7." in system_message.content
    assert "Maximum tool calls in one model response: 3." in system_message.content
    assert "Maximum accepted tool calls across the run: 14." in system_message.content
    assert "split independent calls across turns" in system_message.content
    assert events.events[0].kind is EventKind.RUN_STARTED
    assert events.events[0].data == {
        "task_chars": len("Inspect the runtime limits"),
        "limits": {
            "max_model_turns": 7,
            "max_calls_per_turn": 3,
            "max_total_tool_calls": 14,
        },
    }


def test_total_tool_budget_scales_with_the_selected_model_turn_limit() -> None:
    events = MemoryEventSink()

    result = AgentRunner(
        ScriptedModel([ModelResponse(content="No tools needed.")]),
        ToolRegistry(),
        event_sink=events,
        max_steps=50,
    ).run("Inspect a long-run budget")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert events.events[0].data["limits"] == {
        "max_model_turns": 50,
        "max_calls_per_turn": 8,
        "max_total_tool_calls": 100,
    }


def test_reserved_turns_allow_verification_and_a_final_response() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id=f"work-{index}",
                        name="echo",
                        arguments={"text": str(index)},
                    ),
                )
            )
            for index in range(3)
        ]
        + [
            ModelResponse(
                tool_calls=(
                    ToolCall(id="verify-reserved", name="verification_marker", arguments={}),
                )
            ),
            ModelResponse(content="Verified and complete."),
        ]
    )
    profile = VerificationProfile(
        checks=(VerificationCheck("test-suite", VerificationKind.TEST, ("tests",)),),
        required_labels=("test-suite",),
        target_runtime=TargetRuntime("configured-python", True),
    )

    result = AgentRunner(
        model,
        ToolRegistry([EchoTool(), VerificationMarkerTool()]),
        max_steps=5,
        max_tool_calls_per_step=1,
        verification_profile=profile,
        completion_contract=CompletionContract(required_scopes=("tests",)),
    ).run("Use the work budget, then verify")

    assert result.state is AgentState.COMPLETED
    assert result.steps == 5
    verification_prompt = model.requests[3].messages[0].content or ""
    final_prompt = model.requests[4].messages[0].content or ""
    assert "[CODING_AGENT_CLOSEOUT]" in verification_prompt
    assert "verification calls only" in verification_prompt
    assert "[CODING_AGENT_CLOSEOUT]" in final_prompt
    assert "Do not call tools" in final_prompt


def test_reserved_verification_turn_rejects_more_work_without_executing_it() -> None:
    echo = EchoTool()
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id=f"work-{index}",
                        name="echo",
                        arguments={"text": str(index)},
                    ),
                )
            )
            for index in range(4)
        ]
        + [ModelResponse(content="I could not verify within the reserved closeout.")]
    )
    profile = VerificationProfile(
        checks=(VerificationCheck("test-suite", VerificationKind.TEST, ("tests",)),),
        required_labels=("test-suite",),
        target_runtime=TargetRuntime("configured-python", True),
    )
    events = MemoryEventSink()

    result = AgentRunner(
        model,
        ToolRegistry([echo, VerificationMarkerTool()]),
        event_sink=events,
        max_steps=5,
        max_tool_calls_per_step=1,
        verification_profile=profile,
        completion_contract=CompletionContract(required_scopes=("tests",)),
    ).run("Do not consume the verification reserve with more edits")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert len(echo.calls) == 3
    rejected = [event for event in events.events if event.kind is EventKind.TOOL_BATCH_REJECTED]
    assert len(rejected) == 1
    assert rejected[0].data["reason"] == "verification_turn_requires_verifier"


def test_model_facing_tool_observations_share_one_aggregate_turn_budget() -> None:
    calls = tuple(
        ToolCall(id=f"large-{index}", name="large_output", arguments={}) for index in range(4)
    )
    model = ScriptedModel(
        [ModelResponse(tool_calls=calls), ModelResponse(content="Observed bounded results.")]
    )
    events = MemoryEventSink()

    result = AgentRunner(
        model,
        ToolRegistry([LargeOutputTool()]),
        event_sink=events,
        max_repeated_tool_results=5,
    ).run("Return several large observations")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    finished = [event for event in events.events if event.kind is EventKind.TOOL_FINISHED]
    assert len(finished) == 4
    assert sum(int(event.data["observation_chars"]) for event in finished) < 48_000
    assert all(event.data["observation_truncated"] is True for event in finished)
    visible_results = [
        message.content or ""
        for message in model.requests[1].messages
        if message.role is MessageRole.TOOL
    ]
    assert all("[observation truncated]" in content for content in visible_results)


def test_duplicate_tool_call_id_is_rejected_before_a_second_side_effect() -> None:
    tool = EchoTool()
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(ToolCall(id="echo-reused", name="echo", arguments={"text": "first"}),)
            ),
            ModelResponse(
                tool_calls=(ToolCall(id="echo-reused", name="echo", arguments={"text": "second"}),)
            ),
        ]
    )

    result = AgentRunner(model, ToolRegistry([tool])).run("Never repeat a call ID")

    assert result.state is AgentState.FAILED
    assert result.stop_reason is StopReason.MODEL_ERROR
    assert result.error == "Model returned a duplicate tool call ID"
    assert [call.text for call in tool.calls] == ["first"]
    assert result.messages[-1].role is MessageRole.TOOL
    assert result.messages[-1].tool_call_id == "echo-reused"


def test_total_tool_call_limit_rejects_a_batch_without_partial_execution() -> None:
    tool = EchoTool()
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(id="echo-1", name="echo", arguments={"text": "one"}),
                    ToolCall(id="echo-2", name="echo", arguments={"text": "two"}),
                )
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(id="echo-3", name="echo", arguments={"text": "three"}),
                    ToolCall(id="echo-4", name="echo", arguments={"text": "four"}),
                )
            ),
            ModelResponse(content="Stopped after receiving the budget feedback."),
        ]
    )
    events = MemoryEventSink()

    result = AgentRunner(
        model,
        ToolRegistry([tool]),
        event_sink=events,
        budget_policy=BudgetPolicy(
            max_model_turns=3,
            max_calls_per_turn=2,
            average_calls_per_turn=1,
            verification_turn_reserve=0,
            verification_call_reserve=0,
        ),
    ).run("Exceed the cumulative tool budget")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.stop_reason is StopReason.FINAL_RESPONSE
    assert result.steps == 3
    assert [call.text for call in tool.calls] == ["one", "two"]
    assert [message.role for message in result.messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.TOOL,
        MessageRole.ASSISTANT,
    ]
    assert [event.kind for event in events.events].count(EventKind.TOOL_STARTED) == 2
    assert [event.kind for event in events.events].count(EventKind.TOOL_FINISHED) == 2
    rejected = [event for event in events.events if event.kind is EventKind.TOOL_BATCH_REJECTED]
    assert len(rejected) == 1
    assert rejected[0].data["reason"] == "total_tool_calls_exhausted"
