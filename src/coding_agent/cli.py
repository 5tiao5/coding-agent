"""Command-line entry point."""

from __future__ import annotations

import typer
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel

from coding_agent import __version__
from coding_agent.agent import AgentRunner
from coding_agent.model import ScriptedModel
from coding_agent.models import ModelResponse, ToolCall
from coding_agent.tools import BaseTool, ToolRegistry
from coding_agent.ui import ConsoleEventSink, console_safe

app = typer.Typer(
    name="coding-agent",
    help="A small, observable coding agent built from first principles.",
    no_args_is_help=True,
)
console = Console()


class EchoArguments(BaseModel):
    text: str = Field(min_length=1)


class EchoTool(BaseTool[EchoArguments]):
    name = "echo"
    description = "Echo text locally for the offline agent-loop demo."
    args_model = EchoArguments

    def run(self, arguments: EchoArguments) -> str:
        return arguments.text


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
    """Run a deterministic offline model/tool/observation loop."""
    model = ScriptedModel(
        [
            ModelResponse(
                content="I will verify the local tool boundary.",
                tool_calls=(
                    ToolCall(
                        id="demo-call-1",
                        name="echo",
                        arguments={"text": "local tool execution confirmed"},
                    ),
                ),
            ),
            ModelResponse(content="Offline loop completed successfully."),
        ]
    )
    tools = ToolRegistry([EchoTool()])
    runner = AgentRunner(model, tools, event_sink=ConsoleEventSink(console), max_steps=4)
    result = runner.run("Complete the deterministic offline loop demo.")

    if result.final_text:
        final_text = console_safe(result.final_text, console)
        console.print(Panel.fit(final_text, title="Final response", border_style="yellow"))
    if result.error:
        error = console_safe(result.error, console)
        console.print(Panel.fit(error, title="Failed", border_style="red"))
        raise typer.Exit(code=1)
