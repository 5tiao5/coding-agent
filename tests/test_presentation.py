"""Tests for the passive, shared terminal result presenter."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from coding_agent.models import (
    AgentResult,
    AgentState,
    ChatMessage,
    MessageRole,
    StopReason,
)
from coding_agent.presentation import print_agent_response, safe_terminal_text


def _console(stream: StringIO) -> Console:
    return Console(file=stream, force_terminal=False, color_system=None, width=100)


def _messages() -> tuple[ChatMessage, ...]:
    return (
        ChatMessage(role=MessageRole.SYSTEM, content="system"),
        ChatMessage(role=MessageRole.USER, content="task"),
    )


def test_completed_response_is_sanitized_and_can_show_the_run_id() -> None:
    stream = StringIO()
    result = AgentResult(
        run_id="run-1",
        state=AgentState.COMPLETED,
        stop_reason=StopReason.FINAL_RESPONSE,
        steps=1,
        final_text="Fixed\x00 safely",
        messages=_messages(),
    )

    print_agent_response(result, console=_console(stream), show_run_id=True)

    output = stream.getvalue()
    assert "AGENT RESPONSE" in output
    assert "Fixed  safely" in output
    assert "Run ID: run-1" in output
    assert "\x00" not in output


def test_failed_response_uses_the_stopped_card_without_a_run_id() -> None:
    stream = StringIO()
    result = AgentResult(
        run_id="run-2",
        state=AgentState.FAILED,
        stop_reason=StopReason.MODEL_ERROR,
        steps=0,
        error="Provider failed",
        messages=_messages(),
    )

    print_agent_response(result, console=_console(stream))

    output = stream.getvalue()
    assert "RUN STOPPED" in output
    assert "Provider failed" in output
    assert "Run ID:" not in output


def test_safe_terminal_text_bounds_long_values() -> None:
    stream = StringIO()

    rendered = safe_terminal_text("x" * 100, console=_console(stream), limit=40)

    assert len(rendered) <= 40
    assert rendered.endswith("...[output truncated]...")


def test_safe_terminal_text_handles_tiny_and_invalid_limits() -> None:
    stream = StringIO()
    console = _console(stream)

    assert len(safe_terminal_text("long", console=console, limit=2)) == 2
    try:
        safe_terminal_text("long", console=console, limit=0)
    except ValueError as exc:
        assert str(exc) == "limit must be positive"
    else:  # pragma: no cover - defensive assertion branch.
        raise AssertionError("a non-positive limit must be rejected")
