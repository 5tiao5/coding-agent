"""Small, passive helpers for presenting a completed Agent run."""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from coding_agent.models import AgentResult, AgentState
from coding_agent.ui import console_safe


def print_agent_response(
    result: AgentResult,
    *,
    console: Console,
    show_run_id: bool = False,
) -> None:
    """Render only model prose and a resumable run identifier.

    Verification is intentionally absent here. The caller prints the runtime-owned
    final card after this function, so model prose can never visually outrank evidence.
    """

    if result.final_text:
        console.print(
            Panel(
                Text(safe_terminal_text(result.final_text, console=console)),
                title=" AGENT RESPONSE ",
                border_style="green" if result.state is AgentState.COMPLETED else "yellow",
                box=box.ASCII,
            )
        )
    if result.error:
        console.print(
            Panel(
                Text(safe_terminal_text(result.error, console=console)),
                title=" RUN STOPPED ",
                border_style="red",
                box=box.ASCII,
            )
        )
    if show_run_id:
        console.print(
            Text.assemble(
                ("Run ID: ", "dim"),
                (result.run_id, "cyan"),
                (f"  |  inspect with: coding-agent inspect {result.run_id}", "dim"),
            )
        )


def safe_terminal_text(value: str, *, console: Console, limit: int = 12_000) -> str:
    """Bound control-free model or error text for the destination console."""

    if limit < 1:
        raise ValueError("limit must be positive")
    normalized = "".join(
        character if character.isprintable() or character == "\n" else " " for character in value
    )
    if len(normalized) > limit:
        suffix = "\n...[output truncated]..."
        if limit <= len(suffix):
            normalized = suffix[-limit:]
        else:
            normalized = f"{normalized[: limit - len(suffix)]}{suffix}"
    return console_safe(normalized, console)
