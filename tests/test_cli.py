"""Smoke tests for the user-facing Typer boundary."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from coding_agent import __version__
from coding_agent.cli import app
from coding_agent.demo import (
    DEMO_SOURCE_BEFORE,
    DEMO_SOURCE_PATH,
    DEMO_VERIFICATION_ARGV,
    demo_verification_command,
    run_demo,
    run_repository_demo,
    write_demo_project,
)
from coding_agent.events import EventKind, MemoryEventSink, RunEvent
from coding_agent.lease import RunLease, RunLeaseError
from coding_agent.model import ScriptedModel
from coding_agent.models import (
    AgentResult,
    AgentState,
    ChatMessage,
    MessageRole,
    ModelResponse,
    StopReason,
    ToolCall,
    VerificationKind,
)
from coding_agent.openai_model import ReasoningEffort
from coding_agent.runtime import default_pytest_verifier
from coding_agent.session import SessionBoundary, SessionCheckpoint, SessionStore
from coding_agent.state import StatePaths
from coding_agent.trace import TraceStore

runner = CliRunner()


def test_version_option_prints_the_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_demo_exposes_a_real_read_edit_verify_loop() -> None:
    result = runner.invoke(app, ["demo"])

    assert result.exit_code == 0
    assert result.stdout.count("Running update_plan") == 5
    assert result.stdout.count("Running run_command") == 2
    assert "Running list_files" in result.stdout
    assert "Running search_text" in result.stdout
    assert "Running read_file" in result.stdout
    assert "Running replace_text" in result.stdout
    assert (
        result.stdout.index("Running run_command")
        < result.stdout.index("Running list_files")
        < result.stdout.index("Running search_text")
        < result.stdout.index("Running read_file")
        < result.stdout.index("Running replace_text")
        < result.stdout.rindex("Running run_command")
    )
    assert "Plan revision 1" in result.stdout
    assert "Plan revision 5" in result.stdout
    assert "Command exited 1" in result.stdout
    assert "Command exited 0" in result.stdout
    assert "Found 1 matches and showed 1" in result.stdout
    assert "Read src/pricing.py lines 1-3 of" in result.stdout
    assert "Updated src/pricing.py (+1/-1" in result.stdout
    assert "- return discounted - discount" in result.stdout
    assert "+ return discounted" in result.stdout
    assert result.stdout.count("Running read_file") == 1
    assert "Failing evidence recorded" in result.stdout
    assert "Passing evidence recorded" in result.stdout
    assert "FINAL RESULT" in result.stdout
    assert "Current trusted verification evidence passed" in result.stdout
    assert "Evidence: demo pytest" in result.stdout
    assert "calculate_total now applies the discount exactly" in result.stdout
    assert "post-change pytest run passed" in result.stdout
    assert "VERIFIED" in result.stdout
    assert result.stdout.rfind("AGENT RESPONSE") < result.stdout.rfind("FINAL RESULT")
    assert "\x1b" not in result.stdout


def test_demo_returns_failure_when_the_runtime_cannot_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "coding_agent.demo.repository_demo_model",
        lambda: ScriptedModel([ModelResponse(content="No external verification was run.")]),
    )

    result = runner.invoke(app, ["demo"])

    assert result.exit_code == 1
    assert "FINAL RESULT" in result.stdout
    assert "UNVERIFIED" in result.stdout
    assert "No current trusted verification evidence" in result.stdout
    assert result.stdout.rfind("AGENT RESPONSE") < result.stdout.rfind("FINAL RESULT")


def test_demo_fixture_and_verifier_are_explicit(tmp_path: Path) -> None:
    write_demo_project(tmp_path)
    verifier = demo_verification_command()

    assert tmp_path.joinpath(DEMO_SOURCE_PATH).read_bytes() == DEMO_SOURCE_BEFORE
    assert "assert calculate_total(100, 15) == 85" in tmp_path.joinpath(
        "tests/test_pricing.py"
    ).read_text(encoding="utf-8")
    assert verifier.argv == DEMO_VERIFICATION_ARGV
    assert verifier.cwd == "."
    assert verifier.kind is VerificationKind.TEST
    assert verifier.label == "demo pytest"


def test_repository_demo_accepts_an_injected_model_and_event_sink(tmp_path: Path) -> None:
    events = MemoryEventSink()
    outcome = run_repository_demo(
        tmp_path,
        event_sink=events,
        model=ScriptedModel([ModelResponse(content="No repository action was needed.")]),
    )

    assert outcome.result.state is AgentState.COMPLETED_UNVERIFIED
    assert outcome.verified is False
    assert outcome.source_matches_expected is False
    assert tmp_path.joinpath(DEMO_SOURCE_PATH).read_bytes() == DEMO_SOURCE_BEFORE
    assert events.events[-1].kind is EventKind.RUN_FINISHED
    assert events.events[-1].data["status"] == "missing"


def test_packaged_demo_uses_the_stable_dashboard_renderer() -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, color_system=None, width=100)

    result = run_demo(console=console)

    output = stream.getvalue()
    assert result.state is AgentState.COMPLETED
    assert "Running update_plan" in output
    assert "- [in_progress] reproduce: Run the failing pricing test" in output
    assert "FINAL RESULT" in output
    assert "VERIFIED" in output
    assert "Evidence: demo pytest" in output
    assert output.rfind("AGENT RESPONSE") < output.rfind("FINAL RESULT")
    assert "\x1b" not in output


def test_real_run_persists_terminal_session_and_inspectable_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    state = tmp_path / "state"
    _write_passing_project(workspace)
    verifier = default_pytest_verifier(workspace)
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="verify-1",
                        name="run_command",
                        arguments={"argv": list(verifier.argv), "cwd": "."},
                    ),
                )
            ),
            ModelResponse(content="The repository test passed."),
        ]
    )
    _install_live_model(monkeypatch, model)

    result = runner.invoke(
        app,
        [
            "run",
            "Verify the repository",
            "--root",
            str(workspace),
            "--model",
            "gpt-test",
            "--state-dir",
            str(state),
            "--plain",
        ],
    )

    assert result.exit_code == 0
    assert "FINAL RESULT" in result.stdout
    assert "VERIFIED" in result.stdout
    assert "The repository test passed" in result.stdout
    assert (
        result.stdout.rfind("AGENT RESPONSE")
        < result.stdout.rfind("Run ID:")
        < result.stdout.rfind("FINAL RESULT")
    )
    run_id = _run_id_from(result.stdout)
    paths = StatePaths(state.resolve())
    checkpoint = SessionStore(paths.sessions, workspace_root=workspace).load(run_id).checkpoint
    trace_store = TraceStore(paths.traces)
    events = trace_store.read(run_id)
    assert checkpoint.stop_boundary is SessionBoundary.TERMINAL
    assert checkpoint.workspace_fingerprint is not None
    assert events[-1].kind is EventKind.RUN_FINISHED
    assert "Verify the repository" not in trace_store.path_for(run_id).read_text(encoding="utf-8")

    inspect_result = runner.invoke(
        app,
        ["inspect", run_id, "--state-dir", str(state)],
    )
    assert inspect_result.exit_code == 0
    assert "TRACE SUMMARY" in inspect_result.stdout
    assert "Lifetime tool calls" in inspect_result.stdout
    assert "FINAL RESULT" in inspect_result.stdout
    assert "VERIFIED" in inspect_result.stdout

    runs_result = runner.invoke(app, ["runs", "--state-dir", str(state)])
    assert runs_result.exit_code == 0
    assert run_id in runs_result.stdout


def test_unverified_live_run_has_a_distinct_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _install_live_model(
        monkeypatch,
        ScriptedModel([ModelResponse(content="No trusted verifier was run.")]),
    )

    result = runner.invoke(
        app,
        [
            "run",
            "Only inspect the empty repository",
            "--root",
            str(workspace),
            "--model",
            "gpt-test",
            "--state-dir",
            str(tmp_path / "state"),
            "--plain",
        ],
    )

    assert result.exit_code == 3
    assert "UNVERIFIED" in result.stdout
    assert "No trusted verifier was run" in result.stdout
    assert (
        result.stdout.rfind("AGENT RESPONSE")
        < result.stdout.rfind("Run ID:")
        < result.stdout.rfind("FINAL RESULT")
    )


def test_live_run_passes_explicit_reasoning_effort_to_the_model_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    captured: dict[str, object] = {}

    def model_factory(**kwargs: object) -> ScriptedModel:
        captured.update(kwargs)
        return ScriptedModel([ModelResponse(content="Configuration received.")])

    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setattr("coding_agent.cli.create_openai_responses_model", model_factory)

    result = runner.invoke(
        app,
        [
            "run",
            "Check provider configuration",
            "--root",
            str(workspace),
            "--model",
            "deepseek-v4-flash",
            "--base-url",
            "https://api.deepseek.com",
            "--reasoning-effort",
            "none",
            "--state-dir",
            str(tmp_path / "state"),
            "--plain",
        ],
    )

    assert result.exit_code == 3
    assert captured["model"] == "deepseek-v4-flash"
    assert captured["base_url"] == "https://api.deepseek.com"
    assert captured["reasoning_effort"] is ReasoningEffort.NONE


@pytest.mark.parametrize(
    "target",
    [
        "coding_agent.cli.print_agent_response",
        "coding_agent.dashboard.DashboardEventSink.print_final_card",
    ],
)
def test_presentation_failure_does_not_change_the_agent_exit_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _install_live_model(
        monkeypatch,
        ScriptedModel([ModelResponse(content="No trusted verifier was run.")]),
    )

    def fail_presentation(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("renderer failed")

    monkeypatch.setattr(target, fail_presentation)
    result = runner.invoke(
        app,
        [
            "run",
            "Only inspect",
            "--root",
            str(workspace),
            "--model",
            "gpt-test",
            "--state-dir",
            str(tmp_path / "state"),
            "--plain",
        ],
    )

    assert result.exit_code == 3
    assert result.exception is not None
    assert not isinstance(result.exception, RuntimeError)


def test_new_cli_run_holds_its_preallocated_run_lease_for_the_full_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = tmp_path / "state"
    observed: dict[str, object] = {}

    def observe_lease(**kwargs: object) -> AgentResult:
        run_id = kwargs["run_id"]
        paths = kwargs["paths"]
        assert isinstance(run_id, str)
        assert isinstance(paths, StatePaths)
        competing = RunLease(paths.root / "leases", run_id)
        with pytest.raises(RunLeaseError) as raised:
            competing.acquire()
        assert raised.value.code == "run_already_active"
        observed.update(run_id=run_id, paths=paths)
        return AgentResult(
            run_id=run_id,
            state=AgentState.COMPLETED_UNVERIFIED,
            stop_reason=StopReason.FINAL_RESPONSE,
            steps=1,
            final_text="Lease observed.",
            messages=(
                ChatMessage(role=MessageRole.SYSTEM, content="system"),
                ChatMessage(role=MessageRole.USER, content="task"),
                ChatMessage(role=MessageRole.ASSISTANT, content="Lease observed."),
            ),
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setattr("coding_agent.cli._execute_live_run", observe_lease)

    result = runner.invoke(
        app,
        [
            "run",
            "Hold the run lease",
            "--root",
            str(workspace),
            "--model",
            "gpt-test",
            "--state-dir",
            str(state),
            "--plain",
        ],
    )

    assert result.exit_code == 3
    run_id = observed["run_id"]
    paths = observed["paths"]
    assert isinstance(run_id, str)
    assert isinstance(paths, StatePaths)
    assert re.fullmatch(r"[0-9a-f]{32}", run_id)
    with RunLease(paths.root / "leases", run_id):
        pass


def test_resume_process_reports_a_stable_error_while_the_original_run_holds_the_lease(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = (tmp_path / "state").resolve()
    run_id = "active-run"
    environment = os.environ.copy()
    environment.update(
        OPENAI_API_KEY="test-only-key",
        PYTHONIOENCODING="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "coding_agent",
        "resume",
        run_id,
        "--root",
        str(workspace),
        "--model",
        "gpt-test",
        "--state-dir",
        str(state),
        "--plain",
    ]

    with RunLease(state / "leases", run_id):
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 1, output
    assert "another process is already running or resuming this run" in output
    assert "run_already_active" in output


def test_safe_noninteractive_run_denies_an_ordinary_command_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    marker = "must-not-exist.txt"
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="ordinary-1",
                        name="run_command",
                        arguments={
                            "argv": [
                                sys.executable,
                                "-c",
                                f"from pathlib import Path; Path({marker!r}).write_text('x')",
                            ],
                            "cwd": ".",
                        },
                    ),
                )
            ),
            ModelResponse(content="The command was denied."),
        ]
    )
    _install_live_model(monkeypatch, model)

    result = runner.invoke(
        app,
        [
            "run",
            "Request one ordinary command",
            "--root",
            str(workspace),
            "--model",
            "gpt-test",
            "--state-dir",
            str(tmp_path / "state"),
            "--plain",
        ],
    )

    assert result.exit_code == 3
    assert "APPROVAL REQUIRED" in result.stdout.upper()
    assert "COMMAND DENIED" in result.stdout.upper()
    assert not workspace.joinpath(marker).exists()


def test_resume_reverifies_without_replaying_and_rejects_the_wrong_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    wrong_workspace = tmp_path / "wrong-workspace"
    state = tmp_path / "state"
    _write_passing_project(workspace)
    wrong_workspace.mkdir()
    verifier = default_pytest_verifier(workspace)
    initial_model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="verify-old",
                        name="run_command",
                        arguments={"argv": list(verifier.argv), "cwd": "."},
                    ),
                )
            )
        ]
    )
    _install_live_model(monkeypatch, initial_model)
    initial = runner.invoke(
        app,
        [
            "run",
            "Verify, checkpoint, and resume",
            "--root",
            str(workspace),
            "--model",
            "gpt-test",
            "--state-dir",
            str(state),
            "--max-steps",
            "1",
            "--plain",
        ],
    )
    assert initial.exit_code == 1
    run_id = _run_id_from(initial.stdout)

    calls = 0

    def must_not_create_model(**kwargs: object) -> ScriptedModel:
        nonlocal calls
        del kwargs
        calls += 1
        raise AssertionError("model must not be created for a mismatched workspace")

    monkeypatch.setattr("coding_agent.cli.create_openai_responses_model", must_not_create_model)
    refused = runner.invoke(
        app,
        [
            "resume",
            run_id,
            "--root",
            str(wrong_workspace),
            "--model",
            "gpt-test",
            "--state-dir",
            str(state),
            "--plain",
        ],
    )
    assert refused.exit_code == 1
    assert "different workspace" in refused.stdout
    assert calls == 0

    fresh_model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="verify-fresh",
                        name="run_command",
                        arguments={"argv": list(verifier.argv), "cwd": "."},
                    ),
                )
            ),
            ModelResponse(content="Fresh verification passed after resume."),
        ]
    )
    resume_configuration: dict[str, object] = {}

    def resume_model_factory(**kwargs: object) -> ScriptedModel:
        resume_configuration.update(kwargs)
        return fresh_model

    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setattr(
        "coding_agent.cli.create_openai_responses_model",
        resume_model_factory,
    )
    resumed = runner.invoke(
        app,
        [
            "resume",
            run_id,
            "--root",
            str(workspace),
            "--model",
            "gpt-test",
            "--reasoning-effort",
            "none",
            "--state-dir",
            str(state),
            "--plain",
        ],
    )

    assert resumed.exit_code == 0
    assert "Session resumed" in resumed.stdout
    assert "Fresh verification passed after resume" in resumed.stdout
    assert "VERIFIED" in resumed.stdout
    assert len(fresh_model.requests) == 2
    assert resume_configuration["reasoning_effort"] is ReasoningEffort.NONE


def test_live_cli_requires_explicit_model_and_environment_only_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    no_model = runner.invoke(app, ["run", "Task", "--root", str(workspace)])
    assert no_model.exit_code == 2
    assert "CODING_AGENT_MODEL" in no_model.stderr

    no_key = runner.invoke(
        app,
        ["run", "Task", "--root", str(workspace), "--model", "gpt-test"],
    )
    assert no_key.exit_code == 2
    assert "OPENAI_API_KEY" in no_key.stderr
    assert "api-key" not in no_key.stderr

    invalid_reasoning = runner.invoke(
        app,
        ["run", "Task", "--reasoning-effort", "turbo"],
    )
    assert invalid_reasoning.exit_code == 2
    assert "turbo" in invalid_reasoning.stderr


def test_live_cli_help_discloses_the_stateless_reasoning_limit() -> None:
    run_help = runner.invoke(app, ["run", "--help"], terminal_width=160)
    resume_help = runner.invoke(app, ["resume", "--help"], terminal_width=160)

    assert run_help.exit_code == 0
    assert resume_help.exit_code == 0
    for output in (run_help.stdout, resume_help.stdout):
        assert all(word in output for word in ("stateless", "provider", "reasoning", "state"))
        assert "--reasoning-effort" in output
        assert "use" in output and "none" in output


def test_resume_refuses_a_stale_ready_checkpoint_when_trace_already_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    paths = StatePaths((tmp_path / "state").resolve())
    session_store = SessionStore(paths.sessions, workspace_root=workspace)
    assert session_store.workspace_fingerprint is not None
    run_id = "stale-run"
    task = "Do not run twice"
    prompt = "system"
    session_store.save(
        SessionCheckpoint(
            run_id=run_id,
            workspace_fingerprint=session_store.workspace_fingerprint,
            task=task,
            system_prompt=prompt,
            messages=(
                ChatMessage(role=MessageRole.SYSTEM, content=prompt),
                ChatMessage(role=MessageRole.USER, content=task),
            ),
            completed_steps=0,
            completed_tool_calls=0,
            stop_boundary=SessionBoundary.READY_FOR_MODEL,
        )
    )
    trace = TraceStore(paths.traces)
    trace.append(RunEvent(run_id=run_id, kind=EventKind.RUN_STARTED, message="started"))
    trace.append(
        RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_FINISHED,
            message="completed",
            data={"verified": False, "status": "missing"},
        )
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    model_calls = 0

    def forbidden_model(**kwargs: object) -> ScriptedModel:
        nonlocal model_calls
        del kwargs
        model_calls += 1
        raise AssertionError("completed trace must be rejected before model creation")

    monkeypatch.setattr("coding_agent.cli.create_openai_responses_model", forbidden_model)

    result = runner.invoke(
        app,
        [
            "resume",
            run_id,
            "--root",
            str(workspace),
            "--model",
            "gpt-test",
            "--state-dir",
            str(paths.root),
            "--plain",
        ],
    )

    assert result.exit_code == 1
    assert "already completed" in result.stdout
    assert model_calls == 0


def _write_passing_project(root: Path) -> None:
    tests = root / "tests"
    tests.mkdir(parents=True)
    tests.joinpath("test_ok.py").write_text(
        "def test_ok() -> None:\n    assert 2 + 2 == 4\n",
        encoding="utf-8",
    )


def _install_live_model(
    monkeypatch: pytest.MonkeyPatch,
    model: ScriptedModel,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setattr(
        "coding_agent.cli.create_openai_responses_model",
        lambda **_: model,
    )


def _run_id_from(output: str) -> str:
    matched = re.search(r"Run ID:\s+([a-f0-9]{32})", output)
    assert matched is not None
    return matched.group(1)
