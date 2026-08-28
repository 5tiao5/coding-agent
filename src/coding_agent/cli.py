"""Command-line entry point."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import typer
from rich.console import Console
from rich.panel import Panel

from coding_agent import __version__
from coding_agent.agent import AgentRunner
from coding_agent.model import ScriptedModel
from coding_agent.models import ModelResponse, ToolCall
from coding_agent.mutation import MutationSession
from coding_agent.tools import (
    ListFilesTool,
    ReadFileTool,
    ReplaceTextTool,
    SearchTextTool,
    ToolRegistry,
    UndoChangeTool,
    WriteFileTool,
)
from coding_agent.ui import ConsoleEventSink, console_safe
from coding_agent.workspace import Workspace

app = typer.Typer(
    name="coding-agent",
    help="A small, observable coding agent built from first principles.",
    no_args_is_help=True,
)
console = Console()

_DEMO_SOURCE_BEFORE = (
    b"def calculate_total(subtotal: int, discount: int) -> int:\n"
    b"    discounted = subtotal - discount\n"
    b"    return discounted - discount\n"
)
_DEMO_SOURCE_AFTER = (
    b"def calculate_total(subtotal: int, discount: int) -> int:\n"
    b"    discounted = subtotal - discount\n"
    b"    return discounted\n"
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the package version and exit.",
    ),
) -> None:
    """Run the coding agent."""
    del version


@app.command()
def demo() -> None:
    """Run a deterministic offline read-edit-verify loop against a temporary repository."""
    with TemporaryDirectory(prefix="coding-agent-demo-") as temporary_directory:
        demo_root = Path(temporary_directory)
        _write_demo_project(demo_root)
        workspace = Workspace(demo_root)
        mutation_session = MutationSession(workspace)
        model = _repository_demo_model()
        tools = ToolRegistry(
            [
                ListFilesTool(workspace),
                ReadFileTool(workspace),
                SearchTextTool(workspace),
                WriteFileTool(mutation_session),
                ReplaceTextTool(mutation_session),
                UndoChangeTool(mutation_session),
            ]
        )
        runner = AgentRunner(model, tools, event_sink=ConsoleEventSink(console), max_steps=7)
        result = runner.run(
            "Inspect this repository, fix the duplicate discount in calculate_total, "
            "and read the file again to verify the edit."
        )
        demo_verified = demo_root.joinpath("src/pricing.py").read_bytes() == _DEMO_SOURCE_AFTER

    if result.final_text:
        final_text = console_safe(result.final_text, console)
        title = "Repair report · post-read confirmed" if demo_verified else "Repair report"
        console.print(Panel.fit(final_text, title=title, border_style="green"))
    if result.error:
        error = console_safe(result.error, console)
        console.print(Panel.fit(error, title="Failed", border_style="red"))
        raise typer.Exit(code=1)


def _repository_demo_model() -> ScriptedModel:
    return ScriptedModel(
        [
            ModelResponse(
                content="I will map the repository before choosing a file.",
                tool_calls=(
                    ToolCall(
                        id="demo-list-1",
                        name="list_files",
                        arguments={"path": ".", "max_depth": 3, "limit": 50},
                    ),
                ),
            ),
            ModelResponse(
                content="The repository is small; I will locate the target symbol.",
                tool_calls=(
                    ToolCall(
                        id="demo-search-1",
                        name="search_text",
                        arguments={"query": "def calculate_total", "path": "src"},
                    ),
                ),
            ),
            ModelResponse(
                content="I found the definition and will read its local context.",
                tool_calls=(
                    ToolCall(
                        id="demo-read-1",
                        name="read_file",
                        arguments={"path": "src/pricing.py", "start_line": 1, "line_count": 40},
                    ),
                ),
            ),
            ModelResponse(
                content="The second subtraction is the defect; I will apply one exact edit.",
                tool_calls=(
                    ToolCall(
                        id="demo-replace-1",
                        name="replace_text",
                        arguments={
                            "path": "src/pricing.py",
                            "old_text": "return discounted - discount",
                            "new_text": "return discounted",
                            "expected_sha256": sha256(_DEMO_SOURCE_BEFORE).hexdigest(),
                        },
                    ),
                ),
            ),
            ModelResponse(
                content="The diff was applied; I will re-read the file as post-change evidence.",
                tool_calls=(
                    ToolCall(
                        id="demo-read-2",
                        name="read_file",
                        arguments={"path": "src/pricing.py", "start_line": 1, "line_count": 40},
                    ),
                ),
            ),
            ModelResponse(
                content=(
                    "Fixed src/pricing.py: calculate_total now applies the discount exactly once. "
                    "The post-change read matches the intended implementation."
                )
            ),
        ]
    )


def _write_demo_project(root: Path) -> None:
    source = root / "src" / "pricing.py"
    test = root / "tests" / "test_pricing.py"
    source.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    source.write_bytes(_DEMO_SOURCE_BEFORE)
    test.write_text(
        "from src.pricing import calculate_total\n\n"
        "def test_discount_is_applied_once() -> None:\n"
        "    assert calculate_total(100, 15) == 85\n",
        encoding="utf-8",
    )
