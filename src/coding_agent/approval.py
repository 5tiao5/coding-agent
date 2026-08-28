"""Host-owned command approval decisions for interactive safe mode."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from coding_agent.ui import console_safe


@dataclass(frozen=True, slots=True)
class CommandApprovalRequest:
    """One exact ordinary command awaiting a human decision."""

    argv: tuple[str, ...]
    cwd: str


class CommandApprover(Protocol):
    def approve(self, request: CommandApprovalRequest) -> bool:
        """Return a host decision for this exact request."""


class ConsoleCommandApprover:
    """Render an exact command and ask the terminal user; denial is the default."""

    def __init__(
        self,
        console: Console | None = None,
        *,
        confirm: Callable[[str], bool] | None = None,
        before_prompt: Callable[[], None] | None = None,
        input_is_terminal: Callable[[], bool] | None = None,
    ) -> None:
        self._console = console or Console()
        self._confirm = confirm or self._ask
        self._before_prompt = before_prompt
        self._input_is_terminal = input_is_terminal or _stdin_is_terminal

    def approve(self, request: CommandApprovalRequest) -> bool:
        if self._before_prompt is not None:
            self._before_prompt()
        command = _display_argv(request.argv)
        details = Table.grid(padding=(0, 1))
        details.add_column(style="bold", width=18)
        details.add_column()
        details.add_row(
            "Command",
            Text(console_safe(command, self._console)),
        )
        details.add_row(
            "Working directory",
            Text(console_safe(request.cwd, self._console)),
        )
        self._console.print(
            Panel(
                details,
                title="Approval required",
                border_style="yellow",
                expand=False,
            )
        )
        return bool(self._confirm("Run this command?"))

    def _ask(self, prompt: str) -> bool:
        if not self._console.is_terminal or not self._input_is_terminal():
            return False
        return Confirm.ask(prompt, console=self._console, default=False)


def _display_argv(argv: tuple[str, ...]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def _stdin_is_terminal() -> bool:
    try:
        return bool(sys.stdin.isatty())
    except (AttributeError, OSError):
        return False
