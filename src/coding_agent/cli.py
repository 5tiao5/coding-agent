"""User-facing commands for deterministic demos and real repository runs."""

from __future__ import annotations

import io
import os
from contextlib import suppress
from pathlib import Path
from typing import NoReturn
from uuid import uuid4

import typer
from prompt_toolkit import PromptSession
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from coding_agent import __version__
from coding_agent.agent import AgentRunner
from coding_agent.approval import ConsoleCommandApprover
from coding_agent.command import CommandPermissionMode
from coding_agent.dashboard import DashboardEventSink
from coding_agent.demo import run_demo
from coding_agent.errors import CodedError
from coding_agent.events import BestEffortEventSink, CompositeEventSink, EventKind, RunEvent
from coding_agent.lease import RunLease
from coding_agent.models import AgentResult, AgentState
from coding_agent.openai_model import create_openai_responses_model
from coding_agent.presentation import print_agent_response, safe_terminal_text
from coding_agent.runtime import build_runtime, system_prompt_for
from coding_agent.session import LoadedSession, SessionBoundary, SessionStore
from coding_agent.state import StatePaths, default_state_paths
from coding_agent.trace import (
    JsonlEventSink,
    TraceError,
    TraceRunStatus,
    TraceStore,
    TraceSummary,
    summarize_events,
)

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
    """Run the offline failing-test, repair, and verified-test scenario."""
    try:
        result = run_demo(console=console)
    except Exception as exc:  # noqa: BLE001 - sanitize the user-facing CLI boundary.
        _abort("Demo failed", _public_exception(exc), code=1)
    if result.state is not AgentState.COMPLETED:
        raise typer.Exit(code=1)


@app.command("run")
def run_task(
    task: str | None = typer.Argument(
        None,
        help="Repository task. Omit it to enter the task interactively.",
    ),
    root: Path = typer.Option(
        Path("."),
        "--root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Repository root exposed to local tools.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        envvar="CODING_AGENT_MODEL",
        show_envvar=True,
        help=(
            "Responses API model name; stateless tool turns must not include provider "
            "reasoning state, and no implicit model is selected."
        ),
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        envvar="OPENAI_BASE_URL",
        show_envvar=True,
        help="Optional OpenAI-compatible Responses API base URL.",
    ),
    mode: CommandPermissionMode = typer.Option(
        CommandPermissionMode.SAFE,
        "--mode",
        case_sensitive=False,
        help="safe asks before ordinary commands; auto runs them without asking.",
    ),
    state_dir: Path | None = typer.Option(
        None,
        "--state-dir",
        help="Private state root; defaults to the per-user OS state directory.",
    ),
    max_steps: int = typer.Option(
        20,
        "--max-steps",
        min=1,
        max=100,
        help="Maximum model turns for the complete run.",
    ),
    model_timeout: float = typer.Option(
        120.0,
        "--model-timeout",
        min=1.0,
        max=600.0,
        help="Timeout for each model request in seconds.",
    ),
    plain: bool = typer.Option(
        False,
        "--plain",
        help="Use stable append-only output instead of an in-place live display.",
    ),
) -> None:
    """Run a real model-controlled task inside one repository."""
    resolved_root = root.resolve(strict=True)
    selected_model = _required_model(model)
    selected_task = _task_text(task)
    paths = _resolve_state_paths(state_dir, workspace=resolved_root)
    _require_api_key()
    run_id = uuid4().hex
    try:
        with RunLease(paths.root / "leases", run_id):
            result = _execute_live_run(
                run_id=run_id,
                task=selected_task,
                root=resolved_root,
                model_name=selected_model,
                base_url=base_url,
                mode=mode,
                paths=paths,
                max_steps=max_steps,
                model_timeout=model_timeout,
                plain=plain,
            )
    except (CodedError, TraceError, ValueError, OSError) as exc:
        _abort("Run could not start or continue", _public_exception(exc), code=1)
    raise typer.Exit(code=_result_exit_code(result))


@app.command("resume")
def resume_task(
    run_id: str = typer.Argument(help="Run ID of a ready checkpoint."),
    root: Path = typer.Option(
        ...,
        "--root",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        help="Original repository root; required and fingerprint-checked.",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        envvar="CODING_AGENT_MODEL",
        show_envvar=True,
        help=(
            "Responses API model name; stateless tool turns must not include provider "
            "reasoning state."
        ),
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        envvar="OPENAI_BASE_URL",
        show_envvar=True,
        help="Optional OpenAI-compatible Responses API base URL.",
    ),
    mode: CommandPermissionMode = typer.Option(
        CommandPermissionMode.SAFE,
        "--mode",
        case_sensitive=False,
        help="Command permission mode for the resumed process.",
    ),
    state_dir: Path | None = typer.Option(None, "--state-dir", help="Private state root."),
    max_steps: int = typer.Option(
        20,
        "--max-steps",
        min=1,
        max=100,
        help="Maximum cumulative model turns, including completed turns.",
    ),
    model_timeout: float = typer.Option(
        120.0,
        "--model-timeout",
        min=1.0,
        max=600.0,
        help="Timeout for each model request in seconds.",
    ),
    plain: bool = typer.Option(False, "--plain", help="Disable the in-place live display."),
) -> None:
    """Resume a passive checkpoint without replaying completed tool calls."""
    resolved_root = root.resolve(strict=True)
    selected_model = _required_model(model)
    paths = _resolve_state_paths(state_dir, workspace=resolved_root)
    _require_api_key()
    try:
        with RunLease(paths.root / "leases", run_id):
            session_store = SessionStore(paths.sessions, workspace_root=resolved_root)
            loaded = session_store.load(run_id)
            if loaded.checkpoint.stop_boundary is not SessionBoundary.READY_FOR_MODEL:
                raise ValueError("terminal checkpoints cannot be resumed")
            trace_summary = TraceStore(paths.traces).summarize(run_id)
            if trace_summary.status is TraceRunStatus.COMPLETED:
                raise ValueError("trace shows that this run already completed")
            result = _execute_live_run(
                run_id=run_id,
                task=loaded.checkpoint.task,
                root=resolved_root,
                model_name=selected_model,
                base_url=base_url,
                mode=mode,
                paths=paths,
                max_steps=max_steps,
                model_timeout=model_timeout,
                plain=plain,
                session_store=session_store,
                loaded=loaded,
            )
    except (CodedError, TraceError, ValueError, OSError) as exc:
        _abort("Resume refused", _public_exception(exc), code=1)
    raise typer.Exit(code=_result_exit_code(result))


@app.command("inspect")
def inspect_run(
    run_id: str = typer.Argument(help="Run ID whose validated trace should be inspected."),
    state_dir: Path | None = typer.Option(None, "--state-dir", help="Private state root."),
    timeline_limit: int = typer.Option(
        40,
        "--timeline-limit",
        min=1,
        max=200,
        help="Maximum visible events from the latest run segment.",
    ),
) -> None:
    """Render a validated, read-only summary of one JSONL event trace."""
    paths = _resolve_state_paths(state_dir)
    store = TraceStore(paths.traces)
    try:
        events = store.read(run_id)
    except TraceError as exc:
        _abort("Trace could not be inspected", _public_exception(exc), code=1)
    summary = summarize_events(events)
    _print_trace_summary(summary)
    latest = _latest_trace_segment(events)
    replay_output = io.StringIO()
    replay_console = Console(file=replay_output, color_system=None, width=120)
    replay = DashboardEventSink(
        replay_console,
        live=False,
        task_label=f"Trace {run_id[:12]}",
        max_timeline=timeline_limit,
    )
    for event in latest:
        replay.emit(event)
    replay.close()
    console.print(replay.render())
    if replay.snapshot.terminal:
        console.print(replay.render_final_card())
    if len(latest) > timeline_limit:
        console.print(
            Text(
                f"Showing the latest {timeline_limit} projected events from "
                f"a {len(latest)}-event segment.",
                style="dim",
            )
        )


@app.command("runs")
def list_runs(
    state_dir: Path | None = typer.Option(None, "--state-dir", help="Private state root."),
    limit: int = typer.Option(20, "--limit", min=1, max=100, help="Maximum runs to list."),
) -> None:
    """List recent trace files without contacting a model or repository."""
    paths = _resolve_state_paths(state_dir)
    try:
        runs = TraceStore(paths.traces).list_runs(limit=limit)
    except TraceError as exc:
        _abort("Runs could not be listed", _public_exception(exc), code=1)
    table = Table(title="RECENT RUNS", box=box.ASCII, show_lines=False)
    table.add_column("RUN ID", style="cyan", no_wrap=True)
    table.add_column("MODIFIED", no_wrap=True)
    table.add_column("SIZE", justify="right")
    for item in runs:
        table.add_row(
            item.run_id,
            item.modified_at.astimezone().strftime("%Y-%m-%d %H:%M"),
            _format_bytes(item.size_bytes),
        )
    if not runs:
        table.add_row("-", "No traces recorded", "-")
    console.print(table)


def _execute_live_run(
    *,
    run_id: str,
    task: str,
    root: Path,
    model_name: str,
    base_url: str | None,
    mode: CommandPermissionMode,
    paths: StatePaths,
    max_steps: int,
    model_timeout: float,
    plain: bool,
    session_store: SessionStore | None = None,
    loaded: LoadedSession | None = None,
) -> AgentResult:
    dashboard = DashboardEventSink(
        console,
        live=False if plain else None,
        task_label=f"Repository task in {root.name}",
        auto_final_card=False,
    )
    approver = ConsoleCommandApprover(
        console,
        before_prompt=lambda: _close_dashboard(dashboard),
    )
    runtime = build_runtime(root, permission_mode=mode, approver=approver)
    store = session_store or SessionStore(paths.sessions, workspace_root=root)
    model = create_openai_responses_model(
        model=model_name,
        base_url=base_url,
        timeout_seconds=model_timeout,
        max_retries=0,
    )
    trace = JsonlEventSink(paths.traces)
    runner = AgentRunner(
        model,
        runtime.tools,
        event_sink=CompositeEventSink(trace, BestEffortEventSink(dashboard)),
        max_steps=max_steps,
        session_store=store,
    )
    try:
        if loaded is None:
            result = runner.run(
                task,
                system_prompt=system_prompt_for(runtime.verification_commands),
                run_id=run_id,
            )
        else:
            if loaded.checkpoint.run_id != run_id:
                raise ValueError("loaded checkpoint does not match the leased run ID")
            result = runner.resume(loaded)
    finally:
        _close_dashboard(dashboard)

    _present_live_result(result, dashboard)
    return result


def _present_live_result(result: AgentResult, dashboard: DashboardEventSink) -> None:
    """Keep terminal presentation passive even after the runtime has completed."""

    with suppress(Exception):  # Presentation cannot rewrite the runtime outcome.
        print_agent_response(result, console=console, show_run_id=True)
    with suppress(Exception):  # Presentation cannot rewrite the runtime outcome.
        dashboard.print_final_card()


def _required_model(value: str | None) -> str:
    if value is None or not value.strip():
        raise typer.BadParameter(
            "provide --model or set CODING_AGENT_MODEL",
            param_hint="--model",
        )
    return value.strip()


def _task_text(value: str | None) -> str:
    if value is None:
        value = _prompt_task()
    normalized = value.strip()
    if not normalized:
        raise typer.BadParameter("task cannot be blank", param_hint="TASK")
    return normalized


def _prompt_task() -> str:
    session: PromptSession[str] = PromptSession()
    try:
        return session.prompt("Task > ")
    except (EOFError, KeyboardInterrupt):
        console.print(Text("Task entry cancelled.", style="yellow"))
        raise typer.Exit(code=130) from None


def _require_api_key() -> None:
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise typer.BadParameter(
            "set OPENAI_API_KEY in the environment; secrets are not accepted as CLI arguments",
            param_hint="OPENAI_API_KEY",
        )


def _resolve_state_paths(
    value: Path | None,
    *,
    workspace: Path | None = None,
) -> StatePaths:
    if value is None:
        try:
            paths = default_state_paths()
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="CODING_AGENT_STATE_DIR") from None
    else:
        paths = StatePaths(value.expanduser().resolve(strict=False))
    if workspace is not None:
        resolved_workspace = workspace.resolve(strict=True)
        if paths.root == resolved_workspace or paths.root.is_relative_to(resolved_workspace):
            raise typer.BadParameter(
                "state directory must remain outside the repository",
                param_hint="--state-dir",
            )
    return paths


def _print_trace_summary(summary: TraceSummary) -> None:
    table = Table(title="TRACE SUMMARY", box=box.ASCII, show_header=False)
    table.add_column("FIELD", style="dim")
    table.add_column("VALUE")
    table.add_row("Run ID", summary.run_id)
    table.add_row("Latest status", summary.status.value.upper())
    table.add_row("Latest verification", (summary.verification_status or "not evaluated").upper())
    table.add_row("Lifetime events", str(summary.event_count))
    table.add_row("Lifetime model requests", str(summary.model_requests))
    table.add_row("Lifetime tool calls", str(summary.tool_calls))
    table.add_row("Lifetime tool failures", str(summary.tool_failures))
    table.add_row("Maximum step", str(summary.max_step))
    table.add_row("Wall time", f"{summary.duration_ms / 1_000:.2f} s")
    console.print(table)


def _latest_trace_segment(events: tuple[RunEvent, ...]) -> tuple[RunEvent, ...]:
    starts = [
        index
        for index, event in enumerate(events)
        if event.kind in {EventKind.RUN_STARTED, EventKind.RUN_RESUMED}
    ]
    return events[starts[-1] :] if starts else events


def _result_exit_code(result: AgentResult) -> int:
    if result.state is AgentState.COMPLETED:
        return 0
    if result.state is AgentState.COMPLETED_UNVERIFIED:
        return 3
    return 1


def _abort(title: str, message: str, *, code: int) -> NoReturn:
    console.print(
        Panel(
            Text(safe_terminal_text(message, console=console)),
            title=f" {title.upper()} ",
            border_style="red",
            box=box.ASCII,
        )
    )
    raise typer.Exit(code=code)


def _public_exception(exc: BaseException) -> str:
    if isinstance(exc, CodedError):
        return f"{exc.message} [{exc.code}]"
    if isinstance(exc, TraceError):
        return str(exc)
    if isinstance(exc, OSError):
        return "A local filesystem operation failed. The workspace may already have changed."
    return str(exc) or type(exc).__name__


def _format_bytes(size: int) -> str:
    if size < 1_024:
        return f"{size} B"
    if size < 1_024 * 1_024:
        return f"{size / 1_024:.1f} KiB"
    return f"{size / (1_024 * 1_024):.1f} MiB"


def _close_dashboard(dashboard: DashboardEventSink) -> None:
    with suppress(Exception):
        dashboard.close()
