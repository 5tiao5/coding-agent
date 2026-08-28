"""Tests for deterministic repeated no-progress call detection."""

from __future__ import annotations

import pytest

from coding_agent.models import ToolCall, ToolControlFacts, ToolExecution
from coding_agent.stopping import RepeatedToolCallGuard


def _call(call_id: str = "call-1", *, value: str = "same") -> ToolCall:
    return ToolCall(id=call_id, name="read_file", arguments={"path": value})


def _execution(
    call_id: str = "call-1",
    *,
    output: str = "same output",
    made_progress: bool = False,
) -> ToolExecution:
    return ToolExecution(
        call_id=call_id,
        tool_name="read_file",
        ok=True,
        output=output,
        summary="Read a file",
        control=ToolControlFacts(made_progress=made_progress),
    )


def test_guard_rejects_an_unsafe_threshold() -> None:
    with pytest.raises(ValueError, match="max_identical must be at least 2"):
        RepeatedToolCallGuard(max_identical=1)


def test_third_identical_request_and_outcome_stops() -> None:
    guard = RepeatedToolCallGuard(max_identical=3)

    first = guard.observe(_call("call-1"), _execution("call-1"))
    second = guard.observe(_call("call-2"), _execution("call-2"))
    third = guard.observe(_call("call-3"), _execution("call-3"))

    assert first.streak == 1 and first.should_stop is False
    assert second.streak == 2 and second.should_stop is False
    assert third.streak == 3 and third.should_stop is True


def test_changed_arguments_or_outcomes_reset_the_streak() -> None:
    guard = RepeatedToolCallGuard(max_identical=3)
    guard.observe(_call(), _execution())

    changed_argument = guard.observe(_call(value="other"), _execution())
    changed_output = guard.observe(_call(value="other"), _execution(output="new output"))

    assert changed_argument.streak == 1
    assert changed_output.streak == 1


def test_real_progress_resets_without_counting_the_progress_call() -> None:
    guard = RepeatedToolCallGuard(max_identical=3)
    guard.observe(_call(), _execution())
    guard.observe(_call("call-2"), _execution("call-2"))

    progress = guard.observe(
        ToolCall(id="write", name="replace_text", arguments={"path": "file.py"}),
        ToolExecution(
            call_id="write",
            tool_name="replace_text",
            ok=True,
            output="changed",
            summary="Changed file",
            control=ToolControlFacts(
                invalidates_verification=True,
                made_progress=True,
            ),
        ),
    )
    after = guard.observe(_call("call-3"), _execution("call-3"))

    assert progress.streak == 0
    assert after.streak == 1
