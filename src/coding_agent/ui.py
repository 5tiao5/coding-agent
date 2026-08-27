"""Minimal event renderer; the richer live UI arrives after the core is stable."""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

from coding_agent.events import EventKind, RunEvent

_STYLE: dict[EventKind, tuple[str, str]] = {
    EventKind.RUN_STARTED: ("RUN", "bold cyan"),
    EventKind.STATE_CHANGED: ("...", "dim"),
    EventKind.MODEL_REQUESTED: ("MODEL", "cyan"),
    EventKind.MODEL_RESPONDED: ("MODEL", "cyan"),
    EventKind.TOOL_STARTED: ("TOOL", "yellow"),
    EventKind.TOOL_FINISHED: ("OK", "green"),
    EventKind.RUN_FINISHED: ("DONE", "bold green"),
    EventKind.RUN_FAILED: ("FAIL", "bold red"),
}


class ConsoleEventSink:
    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    def emit(self, event: RunEvent) -> None:
        symbol, style = _STYLE[event.kind]
        if event.kind is EventKind.TOOL_FINISHED and not event.data.get("ok", True):
            symbol, style = "FAIL", "red"
        line = Text()
        line.append(f"[{symbol}] ", style=style)
        if event.step:
            line.append(f"[{event.step}] ", style="dim")
        line.append(console_safe(event.message, self._console), style=style)
        self._console.print(line)


def console_safe(text: str, console: Console) -> str:
    """Replace characters unsupported by a legacy terminal instead of crashing a run."""
    encoding = getattr(console.file, "encoding", None)
    if not encoding:
        return text
    return text.encode(encoding, errors="replace").decode(encoding)
