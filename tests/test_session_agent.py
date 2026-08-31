"""Integration tests for passive AgentRunner checkpoint and resume."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from coding_agent.agent import AgentRunner
from coding_agent.events import EventKind, MemoryEventSink
from coding_agent.model import ScriptedModel
from coding_agent.models import (
    AgentState,
    MessageRole,
    ModelResponse,
    StopReason,
    ToolCall,
    ToolControlFacts,
    ToolOutput,
    VerificationKind,
    VerificationSignal,
)
from coding_agent.session import SessionBoundary, SessionStore
from coding_agent.tools import BaseTool, ToolRegistry


class VerifyArguments(BaseModel):
    pass


class CountingVerifyTool(BaseTool[VerifyArguments]):
    name = "verify_fixture"
    description = "Return trusted verification evidence for a checkpoint fixture."
    args_model = VerifyArguments

    def __init__(self) -> None:
        self.calls = 0

    def run(self, arguments: VerifyArguments) -> ToolOutput:
        del arguments
        self.calls += 1
        return ToolOutput(
            content="fixture passed",
            summary="Fixture verification passed",
            control=ToolControlFacts(
                verification=VerificationSignal.PASSED,
                verification_kind=VerificationKind.TEST,
                verification_label="fixture",
            ),
        )


def test_initial_checkpoint_makes_a_first_model_failure_resumable(tmp_path: Path) -> None:
    store = SessionStore((tmp_path / "state").resolve())

    failed = AgentRunner(
        ScriptedModel([]),
        ToolRegistry(),
        session_store=store,
    ).run("Survive a provider outage")

    assert failed.state is AgentState.FAILED
    loaded = store.load(failed.run_id)
    assert loaded.checkpoint.stop_boundary is SessionBoundary.READY_FOR_MODEL
    assert loaded.checkpoint.completed_steps == 0
    assert loaded.checkpoint.completed_tool_calls == 0


def test_resume_never_replays_tools_and_requires_fresh_verification(tmp_path: Path) -> None:
    store = SessionStore((tmp_path / "state").resolve())
    initial_tool = CountingVerifyTool()
    initial = AgentRunner(
        ScriptedModel(
            [
                ModelResponse(
                    tool_calls=(ToolCall(id="verify-1", name="verify_fixture", arguments={}),)
                )
            ]
        ),
        ToolRegistry([initial_tool]),
        session_store=store,
        max_steps=1,
    ).run("Verify once, then checkpoint")

    assert initial.state is AgentState.FAILED
    assert initial_tool.calls == 1
    loaded = store.load(initial.run_id)
    resumed_tool = CountingVerifyTool()
    events = MemoryEventSink()
    resumed_model = ScriptedModel([ModelResponse(content="Resume without rerunning tests.")])

    resumed = AgentRunner(
        resumed_model,
        ToolRegistry([resumed_tool]),
        event_sink=events,
    ).resume(loaded)

    assert resumed.run_id == initial.run_id
    assert resumed.state is AgentState.COMPLETED_UNVERIFIED
    assert resumed_tool.calls == 0
    assert len(resumed_model.requests) == 1
    resumed_messages = resumed_model.requests[0].messages
    assert resumed_messages[1:] == loaded.checkpoint.messages[1:]
    assert resumed_messages[0].content is not None
    assert "Maximum model turns: 20." in resumed_messages[0].content
    assert "Maximum model turns: 1." not in resumed_messages[0].content
    assert events.events[0].kind is EventKind.RUN_RESUMED
    assert events.events[0].data["requires_reverification"] is True
    assert EventKind.VERIFICATION_RECORDED not in [event.kind for event in events.events]


def test_terminal_checkpoint_cannot_be_resumed(tmp_path: Path) -> None:
    store = SessionStore((tmp_path / "state").resolve())
    events = MemoryEventSink()
    result = AgentRunner(
        ScriptedModel([ModelResponse(content="Done without tools.")]),
        ToolRegistry(),
        session_store=store,
        event_sink=events,
    ).run("Finish and save")
    loaded = store.load(result.run_id)

    assert events.events[-2].kind is EventKind.SESSION_CHECKPOINTED
    assert events.events[-1].kind is EventKind.RUN_FINISHED

    runner = AgentRunner(ScriptedModel([]), ToolRegistry())

    try:
        runner.resume(loaded)
    except ValueError as exc:
        assert str(exc) == "only ready_for_model checkpoints can be resumed"
    else:  # pragma: no cover - explicit failure keeps the assertion message clear.
        raise AssertionError("terminal checkpoint was resumed")


def test_resume_can_establish_fresh_verification_and_replace_the_checkpoint(
    tmp_path: Path,
) -> None:
    store = SessionStore((tmp_path / "state").resolve())
    initial_tool = CountingVerifyTool()
    initial = AgentRunner(
        ScriptedModel(
            [ModelResponse(tool_calls=(ToolCall(id="verify-old", name="verify_fixture"),))]
        ),
        ToolRegistry([initial_tool]),
        session_store=store,
        max_steps=1,
    ).run("Checkpoint, resume, and verify again")
    loaded = store.load(initial.run_id)
    resumed_tool = CountingVerifyTool()
    resumed_model = ScriptedModel(
        [
            ModelResponse(tool_calls=(ToolCall(id="verify-fresh", name="verify_fixture"),)),
            ModelResponse(content="Fresh verification passed after resume."),
        ]
    )

    resumed = AgentRunner(
        resumed_model,
        ToolRegistry([resumed_tool]),
        session_store=store,
    ).resume(loaded)

    assert initial_tool.calls == 1
    assert resumed_tool.calls == 1
    assert resumed.run_id == initial.run_id
    assert resumed.state is AgentState.COMPLETED
    assert len(resumed_model.requests) == 2
    terminal = store.load(resumed.run_id).checkpoint
    assert terminal.stop_boundary is SessionBoundary.TERMINAL
    assert terminal.messages == resumed.messages
    assert terminal.completed_steps == 3
    assert terminal.completed_tool_calls == 2


def test_rejected_batch_checkpoint_resumes_with_feedback_and_current_limits(
    tmp_path: Path,
) -> None:
    store = SessionStore((tmp_path / "state").resolve())
    rejected_calls = tuple(
        ToolCall(id=f"rejected-{index}", name="verify_fixture", arguments={}) for index in range(2)
    )
    initial_tool = CountingVerifyTool()
    initial = AgentRunner(
        ScriptedModel([ModelResponse(tool_calls=rejected_calls)]),
        ToolRegistry([initial_tool]),
        session_store=store,
        max_steps=2,
        max_tool_calls_per_step=1,
        max_total_tool_calls=2,
    ).run("Checkpoint an over-limit batch")

    assert initial.state is AgentState.FAILED
    assert initial_tool.calls == 0
    loaded = store.load(initial.run_id)
    checkpoint = loaded.checkpoint
    assert checkpoint.stop_boundary is SessionBoundary.READY_FOR_MODEL
    assert checkpoint.completed_steps == 1
    assert checkpoint.completed_tool_calls == 0
    assert [message.role for message in checkpoint.messages[-3:]] == [
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
        MessageRole.TOOL,
    ]
    assert all(
        message.content is not None and "tool_batch_rejected" in message.content
        for message in checkpoint.messages[-2:]
    )

    resumed_tool = CountingVerifyTool()
    resumed_model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(ToolCall(id="verify-retry", name="verify_fixture", arguments={}),)
            ),
            ModelResponse(content="Recovered after the smaller retry."),
        ]
    )
    events = MemoryEventSink()
    resumed = AgentRunner(
        resumed_model,
        ToolRegistry([resumed_tool]),
        session_store=store,
        event_sink=events,
        max_steps=4,
        max_tool_calls_per_step=2,
        max_total_tool_calls=5,
    ).resume(loaded)

    assert resumed.state is AgentState.COMPLETED
    assert resumed_tool.calls == 1
    visible_prompt = resumed_model.requests[0].messages[0].content
    assert visible_prompt is not None
    assert "Maximum model turns: 4." in visible_prompt
    assert "Maximum tool calls in one model response: 2." in visible_prompt
    assert "Maximum accepted tool calls across the run: 5." in visible_prompt
    assert "Maximum model turns: 2." not in visible_prompt
    assert events.events[0].kind is EventKind.RUN_RESUMED
    assert events.events[0].data["limits"] == {
        "max_model_turns": 4,
        "max_calls_per_turn": 2,
        "max_total_tool_calls": 5,
    }
    terminal = store.load(resumed.run_id).checkpoint
    assert terminal.completed_tool_calls == 1
    assert terminal.system_prompt == visible_prompt
    assert terminal.messages[0].content == visible_prompt


def test_consecutive_rejection_circuit_survives_checkpoint_resume(tmp_path: Path) -> None:
    store = SessionStore((tmp_path / "state").resolve())

    def rejected_batch(prefix: str) -> ModelResponse:
        return ModelResponse(
            tool_calls=tuple(
                ToolCall(
                    id=f"{prefix}-{index}",
                    name="verify_fixture",
                    arguments={},
                )
                for index in range(2)
            )
        )

    initial_tool = CountingVerifyTool()
    initial = AgentRunner(
        ScriptedModel([rejected_batch("first"), rejected_batch("second")]),
        ToolRegistry([initial_tool]),
        session_store=store,
        max_steps=2,
        max_tool_calls_per_step=1,
    ).run("Preserve the rejection circuit")

    assert initial.stop_reason is StopReason.MAX_STEPS
    assert initial_tool.calls == 0
    loaded = store.load(initial.run_id)
    assert loaded.checkpoint.completed_steps == 2
    assert loaded.checkpoint.completed_tool_calls == 0

    resumed_tool = CountingVerifyTool()
    events = MemoryEventSink()
    resumed = AgentRunner(
        ScriptedModel([rejected_batch("third")]),
        ToolRegistry([resumed_tool]),
        session_store=store,
        event_sink=events,
        max_steps=3,
        max_tool_calls_per_step=1,
    ).resume(loaded)

    assert resumed.state is AgentState.FAILED
    assert resumed.stop_reason is StopReason.TOOL_LIMIT
    assert resumed.steps == 3
    assert resumed_tool.calls == 0
    rejected_event = next(
        event for event in events.events if event.kind is EventKind.TOOL_BATCH_REJECTED
    )
    assert rejected_event.data == {
        "requested_calls": 2,
        "max_calls_per_turn": 1,
        "rejection_count": 3,
        "max_rejections": 3,
    }
