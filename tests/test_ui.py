"""Tests for terminal-safe event rendering helpers."""

from __future__ import annotations

from io import BytesIO, StringIO, TextIOWrapper

from rich.console import Console

from coding_agent.events import EventKind, RunEvent
from coding_agent.ui import ConsoleEventSink, console_safe


def test_console_safe_replaces_characters_unsupported_by_the_console_encoding() -> None:
    ascii_stream = TextIOWrapper(BytesIO(), encoding="ascii")
    console = Console(file=ascii_stream, force_terminal=False, color_system=None)

    assert console_safe("plain Ω text", console) == "plain ? text"


def test_console_event_sink_renders_every_structured_event_kind() -> None:
    stream = StringIO()
    sink = ConsoleEventSink(
        Console(file=stream, force_terminal=False, color_system=None, width=120)
    )

    for kind in EventKind:
        sink.emit(RunEvent(run_id="ui-event-coverage", kind=kind, message=kind.value))

    rendered = stream.getvalue()
    assert all(kind.value in rendered for kind in EventKind)
