import json
import sys
from pathlib import Path

import pytest

from coding_agent import application
from coding_agent.application import RepositoryRunSpec, execute_repository_run
from coding_agent.cancellation import CancellationSource
from coding_agent.command import CommandPermissionMode
from coding_agent.events import EventKind, MemoryEventSink
from coding_agent.model import ScriptedModel
from coding_agent.models import AgentState, ModelResponse, StopReason, ToolCall
from coding_agent.session import LoadedSession, SessionBoundary, SessionError, SessionStore
from coding_agent.state import StatePaths
from coding_agent.trace import TraceRunStatus, TraceStore


def _spec(tmp_path: Path) -> RepositoryRunSpec:
    root = tmp_path / "repository"
    root.mkdir()
    return RepositoryRunSpec(
        run_id="shared-application-run",
        task="Inspect the repository and report the result",
        root=root,
        model_name="test-model",
        base_url=None,
        reasoning_effort=None,
        permission_mode=CommandPermissionMode.SAFE,
        paths=StatePaths((tmp_path / "state").resolve()),
        max_steps=3,
        model_timeout=10,
    )


def test_repository_application_service_drives_trace_and_presentation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    model = ScriptedModel([ModelResponse(content="Inspection complete")])
    monkeypatch.setattr(application, "create_openai_responses_model", lambda **_kwargs: model)
    presentation = MemoryEventSink()

    result = execute_repository_run(spec, event_sink=presentation)

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert presentation.events[-1].kind is EventKind.RUN_FINISHED
    assert presentation.events[-1].data["verified"] is False
    assert presentation.events[-1].data["checks_passed"] is False
    assert presentation.events[-1].data["task_validated"] is False
    assert presentation.events[-1].data["completion_status"] == "missing"
    assert presentation.events[-1].data["required_labels"] == ["pytest"]
    assert presentation.events[-1].data["missing_labels"] == ["pytest"]
    assert presentation.events[-1].data["required_scopes"] == ["tests"]
    assert presentation.events[-1].data["missing_scopes"] == ["tests"]
    assert presentation.events[-1].data["target_runtime_id"] == "unconfigured-python"
    trace = TraceStore(spec.paths.traces)
    assert trace.summarize(spec.run_id).status is TraceRunStatus.COMPLETED
    assert trace.read(spec.run_id) == tuple(presentation.events)


def test_repository_application_service_disables_a_broken_presentation_sink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    model = ScriptedModel([ModelResponse(content="Inspection complete")])
    monkeypatch.setattr(application, "create_openai_responses_model", lambda **_kwargs: model)

    class BrokenSink:
        def emit(self, _event: object) -> None:
            raise RuntimeError("renderer failed")

    result = execute_repository_run(spec, event_sink=BrokenSink())

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    events = TraceStore(spec.paths.traces).read(spec.run_id)
    assert events[-1].kind is EventKind.RUN_FINISHED


def test_unconfigured_repository_pytest_pass_is_checks_only(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec.root.joinpath("test_public.py").write_text(
        "def test_public():\n    assert True\n",
        encoding="utf-8",
    )
    presentation = MemoryEventSink()
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="pytest",
                        name="run_command",
                        arguments={
                            "argv": [
                                str(Path(sys.executable).absolute()),
                                "-I",
                                "-B",
                                "-m",
                                "pytest",
                                "-q",
                                "-p",
                                "no:cacheprovider",
                            ],
                            "cwd": ".",
                            "timeout_seconds": 20,
                        },
                    ),
                )
            ),
            ModelResponse(content="The available check passed."),
        ]
    )

    result = execute_repository_run(
        spec,
        event_sink=presentation,
        model_factory=lambda **_kwargs: model,
    )

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert presentation.events[-1].data["checks_passed"] is True
    assert presentation.events[-1].data["task_validated"] is False
    assert presentation.events[-1].data["completion_status"] == "checks_only"
    assert presentation.events[-1].data["target_runtime_eligible"] is False


def test_repository_application_passes_host_cancellation_to_a_resumable_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    model = ScriptedModel([ModelResponse(content="must not be requested")])
    monkeypatch.setattr(application, "create_openai_responses_model", lambda **_kwargs: model)
    source = CancellationSource()
    source.request_cancellation()

    result = execute_repository_run(spec, cancellation_token=source.token)

    assert result.state is AgentState.FAILED
    assert result.stop_reason is StopReason.USER_INTERRUPTED
    assert model.requests == []
    events = TraceStore(spec.paths.traces).read(spec.run_id)
    assert events[-1].kind is EventKind.RUN_FAILED
    assert events[-1].data == {"stop_reason": StopReason.USER_INTERRUPTED.value}
    checkpoint = (
        SessionStore(
            spec.paths.sessions,
            workspace_root=spec.root,
        )
        .load(spec.run_id)
        .checkpoint
    )
    assert checkpoint.stop_boundary is SessionBoundary.READY_FOR_MODEL


def test_repository_resume_accepts_the_same_policy_anchor(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    first = execute_repository_run(
        spec,
        model_factory=lambda **_kwargs: ScriptedModel([]),
    )
    assert first.state is AgentState.FAILED
    loaded = SessionStore(spec.paths.sessions, workspace_root=spec.root).load(spec.run_id)

    resumed = execute_repository_run(
        spec,
        loaded=loaded,
        model_factory=lambda **_kwargs: ScriptedModel(
            [ModelResponse(content="Recovered under the same verification policy.")]
        ),
    )

    assert resumed.state is AgentState.COMPLETED_UNVERIFIED


def test_repository_resume_rejects_configured_policy_manifest_drift(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    tests = spec.root / "tests"
    tests.mkdir()
    protected = tests / "test_public.py"
    protected.write_text("def test_public():\n    assert True\n", encoding="utf-8")
    metadata = spec.root / ".coding-agent"
    metadata.mkdir()
    metadata.joinpath("project.toml").write_text(
        "\n".join(
            (
                "schema_version = 1",
                'protected_paths = ["tests/"]',
                "[python]",
                "executable = " + json.dumps(str(Path(sys.executable).absolute())),
                "[[verifiers]]",
                'label = "pytest"',
                'type = "pytest"',
                'scopes = ["tests"]',
                "required = true",
                "",
            )
        ),
        encoding="utf-8",
    )
    first = execute_repository_run(
        spec,
        model_factory=lambda **_kwargs: ScriptedModel([]),
    )
    assert first.state is AgentState.FAILED
    loaded = SessionStore(spec.paths.sessions, workspace_root=spec.root).load(spec.run_id)
    protected.write_text("def test_public():\n    assert False\n", encoding="utf-8")
    factory_called = False

    def forbidden_factory(**_kwargs: object) -> ScriptedModel:
        nonlocal factory_called
        factory_called = True
        return ScriptedModel([ModelResponse(content="Must not resume")])

    with pytest.raises(SessionError, match="does not match") as raised:
        execute_repository_run(
            spec,
            loaded=loaded,
            model_factory=forbidden_factory,
        )

    assert raised.value.code == "checkpoint_policy_mismatch"
    assert factory_called is False


def test_resume_policy_guard_rejects_wrong_run_and_invalid_or_legacy_anchors(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    execute_repository_run(
        spec,
        model_factory=lambda **_kwargs: ScriptedModel([]),
    )
    loaded = SessionStore(spec.paths.sessions, workspace_root=spec.root).load(spec.run_id)

    with pytest.raises(ValueError, match="leased run ID"):
        application._require_resume_policy(
            loaded,
            run_id="different-run",
            policy_fingerprint="0" * 64,
            configured=False,
        )

    prompt_without_anchor = loaded.checkpoint.system_prompt.split(
        "\nVerification policy fingerprint:", maxsplit=1
    )[0]

    def with_prompt(prompt: str) -> LoadedSession:
        messages = list(loaded.checkpoint.messages)
        messages[0] = messages[0].model_copy(update={"content": prompt})
        checkpoint = loaded.checkpoint.model_copy(
            update={"system_prompt": prompt, "messages": tuple(messages)}
        )
        return loaded.model_copy(update={"checkpoint": checkpoint})

    invalid = with_prompt(prompt_without_anchor + "\nVerification policy fingerprint: not-a-sha256")
    with pytest.raises(SessionError, match="identity is invalid") as invalid_error:
        application._require_resume_policy(
            invalid,
            run_id=spec.run_id,
            policy_fingerprint="0" * 64,
            configured=True,
        )
    assert invalid_error.value.code == "checkpoint_policy_mismatch"

    legacy = with_prompt(prompt_without_anchor)
    with pytest.raises(SessionError, match="predates") as legacy_error:
        application._require_resume_policy(
            legacy,
            run_id=spec.run_id,
            policy_fingerprint="0" * 64,
            configured=True,
        )
    assert legacy_error.value.code == "checkpoint_policy_mismatch"
