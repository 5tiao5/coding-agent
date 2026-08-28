"""Command-line entry point."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import typer
from rich.console import Console
from rich.panel import Panel

from coding_agent import __version__
from coding_agent.agent import AgentRunner
from coding_agent.model import ScriptedModel
from coding_agent.models import ModelResponse, ToolCall
from coding_agent.tools import ListFilesTool, ReadFileTool, SearchTextTool, ToolRegistry
from coding_agent.ui import ConsoleEventSink, console_safe
from coding_agent.workspace import Workspace

app = typer.Typer(
    name="coding-agent",
    help="A small, observable coding agent built from first principles.",
    no_args_is_help=True,
)
console = Console()


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
    """Run deterministic, offline repository reconnaissance through the real agent loop."""
    with TemporaryDirectory(prefix="coding-agent-demo-") as temporary_directory:
        demo_root = Path(temporary_directory)
        _write_demo_project(demo_root)
        workspace = Workspace(demo_root)
        model = _repository_demo_model()
        tools = ToolRegistry(
            [
                ListFilesTool(workspace),
                ReadFileTool(workspace),
                SearchTextTool(workspace),
            ]
        )
        runner = AgentRunner(model, tools, event_sink=ConsoleEventSink(console), max_steps=5)
        result = runner.run(
            "Inspect this repository and identify the defect in calculate_total "
            "without editing files."
        )

    if result.final_text:
        final_text = console_safe(result.final_text, console)
        console.print(Panel.fit(final_text, title="Reconnaissance report", border_style="yellow"))
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
                content=(
                    "The defect is in src/pricing.py: calculate_total subtracts the discount "
                    "twice. Repository reconnaissance completed without modifying the workspace."
                )
            ),
        ]
    )


def _write_demo_project(root: Path) -> None:
    source = root / "src" / "pricing.py"
    test = root / "tests" / "test_pricing.py"
    source.parent.mkdir(parents=True)
    test.parent.mkdir(parents=True)
    source.write_text(
        "def calculate_total(subtotal: int, discount: int) -> int:\n"
        "    discounted = subtotal - discount\n"
        "    return discounted - discount\n",
        encoding="utf-8",
    )
    test.write_text(
        "from src.pricing import calculate_total\n\n"
        "def test_discount_is_applied_once() -> None:\n"
        "    assert calculate_total(100, 15) == 85\n",
        encoding="utf-8",
    )
