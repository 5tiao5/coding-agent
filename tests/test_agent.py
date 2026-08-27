"""Behaviour tests for the project-owned agent loop."""

from __future__ import annotations

from pydantic import BaseModel, Field, TypeAdapter

from coding_agent.agent import AgentRunner
from coding_agent.events import EventKind, MemoryEventSink
from coding_agent.model import ScriptedModel
from coding_agent.models import (
    AgentState,
    MessageRole,
    ModelResponse,
    StopReason,
    ToolCall,
)
from coding_agent.tools import BaseTool, ToolOutput, ToolRegistry

_TOOL_PAYLOAD_ADAPTER = TypeAdapter(dict[str, object])


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
        EventKind.RUN_FINISHED,
    ]
    assert [event.step for event in events.events] == [0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    assert {event.run_id for event in events.events} == {result.run_id}
    assert events.events[-1].data == {"verified": False}


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
    ]
    assert EventKind.TOOL_STARTED not in [event.kind for event in events.events]
    assert events.events[-1].data == {"stop_reason": StopReason.TOOL_LIMIT.value}


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
    ]
    assert [event.kind for event in events.events].count(EventKind.TOOL_STARTED) == 2
    assert [event.kind for event in events.events].count(EventKind.TOOL_FINISHED) == 2
    assert events.events[-1].data == {"stop_reason": StopReason.TOOL_LIMIT.value}
