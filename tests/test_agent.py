"""Behaviour tests for the project-owned agent loop."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence

import pytest
from pydantic import BaseModel, Field, TypeAdapter

from coding_agent.agent import AgentRunner
from coding_agent.context import ContextManager
from coding_agent.events import EventKind, MemoryEventSink
from coding_agent.model import RetryableModelError, ScriptedModel
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
        },
        {
            "attempt": 2,
            "next_attempt": 3,
            "max_attempts": 3,
            "delay_seconds": 0.5,
            "error_code": "model_request_transient",
        },
    ]


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


class VerificationMarkerTool(BaseTool[ControlArgs]):
    name = "verification_marker"
    description = "Return trusted test verification evidence."
    args_model = ControlArgs

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
    }
    assert all("verification_label" not in (message.content or "") for message in result.messages)


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


def test_terminal_control_fact_stops_the_run_after_recording_the_observation() -> None:
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(ToolCall(id="terminal-1", name="terminal_marker", arguments={}),)
            )
        ]
    )

    result = AgentRunner(model, ToolRegistry([TerminalMarkerTool()])).run(
        "Trigger a terminal command-control failure"
    )

    assert result.state is AgentState.FAILED
    assert result.stop_reason is StopReason.COMMAND_CONTROL_FAILED
    assert result.error == "Command process cleanup could not be guaranteed"
    assert result.messages[-1].role is MessageRole.TOOL


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
        context_manager=ContextManager(max_chars=1_000),
    ).run("Observe a large output")

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert "LARGE_OUTPUT_" in (result.messages[3].content or "")
    assert all(
        "LARGE_OUTPUT_" not in (message.content or "") for message in model.requests[1].messages
    )
    compacted = [event for event in events.events if event.kind is EventKind.CONTEXT_COMPACTED]
    assert len(compacted) == 1
    assert compacted[0].data["compacted_blocks"] == 1
    assert compacted[0].data["prepared_chars"] <= 1_000


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


def test_repeated_call_stop_closes_the_remaining_tool_batch_without_executing_it() -> None:
    calls = tuple(
        ToolCall(id=f"echo-{index}", name="echo", arguments={"text": "same"})
        for index in range(1, 5)
    )
    tool = EchoTool()

    result = AgentRunner(
        ScriptedModel([ModelResponse(tool_calls=calls)]),
        ToolRegistry([tool]),
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


def test_per_step_tool_call_limit_rejects_the_whole_batch_before_execution() -> None:
    tool = EchoTool()
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(id="echo-1", name="echo", arguments={"text": "one"}),
                    ToolCall(id="echo-2", name="echo", arguments={"text": "two"}),
                    ToolCall(id="echo-3", name="echo", arguments={"text": "three"}),
                )
            )
        ]
    )
    events = MemoryEventSink()

    result = AgentRunner(
        model,
        ToolRegistry([tool]),
        event_sink=events,
        max_tool_calls_per_step=2,
    ).run("Request too many tools at once")

    assert result.state is AgentState.FAILED
    assert result.stop_reason is StopReason.TOOL_LIMIT
    assert result.steps == 1
    assert result.error == "Model exceeded the per-step tool call limit: 3 > 2"
    assert tool.calls == []
    assert [message.role for message in result.messages] == [
        MessageRole.SYSTEM,
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.TOOL,
        MessageRole.TOOL,
    ]
    assert EventKind.TOOL_STARTED not in [event.kind for event in events.events]
    assert events.events[-1].data == {"stop_reason": StopReason.TOOL_LIMIT.value}


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


def test_total_tool_call_limit_counts_successful_batches_across_steps() -> None:
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
        ]
    )
    events = MemoryEventSink()

    result = AgentRunner(
        model,
        ToolRegistry([tool]),
        event_sink=events,
        max_tool_calls_per_step=2,
        max_total_tool_calls=3,
    ).run("Exceed the cumulative tool budget")

    assert result.state is AgentState.FAILED
    assert result.stop_reason is StopReason.TOOL_LIMIT
    assert result.steps == 2
    assert result.error == "Model exceeded the total tool call limit: 4 > 3"
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
    ]
    assert [event.kind for event in events.events].count(EventKind.TOOL_STARTED) == 2
    assert [event.kind for event in events.events].count(EventKind.TOOL_FINISHED) == 2
    assert events.events[-1].data == {"stop_reason": StopReason.TOOL_LIMIT.value}
