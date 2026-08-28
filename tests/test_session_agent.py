"""Integration tests for passive AgentRunner checkpoint and resume."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from coding_agent.agent import AgentRunner
from coding_agent.events import EventKind, MemoryEventSink
from coding_agent.model import ScriptedModel
from coding_agent.models import (
    AgentState,
    ModelResponse,
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
    assert resumed_model.requests[0].messages == loaded.checkpoint.messages
    assert events.events[0].kind is EventKind.RUN_RESUMED
    assert events.events[0].data["requires_reverification"] is True
    assert EventKind.VERIFICATION_RECORDED not in [event.kind for event in events.events]


def test_terminal_checkpoint_cannot_be_resumed(tmp_path: Path) -> None:
    store = SessionStore((tmp_path / "state").resolve())
    result = AgentRunner(
        ScriptedModel([ModelResponse(content="Done without tools.")]),
        ToolRegistry(),
        session_store=store,
    ).run("Finish and save")
    loaded = store.load(result.run_id)

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
