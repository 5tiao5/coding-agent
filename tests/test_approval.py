"""Tests for the human-owned command approval boundary."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from coding_agent.approval import CommandApprovalRequest, ConsoleCommandApprover


def test_console_approver_displays_exact_request_and_uses_injected_decision() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=100)
    prompts: list[str] = []

    def approve(prompt: str) -> bool:
        prompts.append(prompt)
        return True

    approver = ConsoleCommandApprover(console, confirm=approve)

    decision = approver.approve(
        CommandApprovalRequest(argv=("python", "-m", "pytest", "tests/a b.py"), cwd="src")
    )

    assert decision is True
    assert prompts == ["Run this command?"]
    rendered = output.getvalue()
    assert "Approval required" in rendered
    assert "python" in rendered
    assert "tests/a b.py" in rendered
    assert "src" in rendered


def test_console_approver_pauses_live_ui_and_treats_model_text_as_plain_text() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=100)
    calls: list[str] = []

    def confirm(_: str) -> bool:
        calls.append("confirm")
        return True

    approver = ConsoleCommandApprover(
        console,
        before_prompt=lambda: calls.append("pause"),
        confirm=confirm,
    )

    assert approver.approve(
        CommandApprovalRequest(argv=("tool", "[/bold][red]literal"), cwd="[/panel]")
    )
    assert calls == ["pause", "confirm"]
    rendered = output.getvalue()
    assert "[/bold][red]literal" in rendered
    assert "[/panel]" in rendered


def test_console_approver_defaults_to_denial_when_input_is_not_interactive() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None)

    approved = ConsoleCommandApprover(console).approve(
        CommandApprovalRequest(argv=("python", "script.py"), cwd=".")
    )

    assert approved is False


def test_console_approver_denies_piped_input_even_when_output_is_a_tty() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=True, color_system=None)
    approver = ConsoleCommandApprover(console, input_is_terminal=lambda: False)

    approved = approver.approve(CommandApprovalRequest(argv=("python", "script.py"), cwd="."))

    assert approved is False
    assert "Approval required" in output.getvalue()
