from pathlib import Path

import pytest

from coding_agent import application
from coding_agent.application import RepositoryRunSpec, execute_repository_run
from coding_agent.command import CommandPermissionMode
from coding_agent.events import EventKind, MemoryEventSink
from coding_agent.model import ScriptedModel
from coding_agent.models import AgentState, ModelResponse
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
    assert presentation.events[-1].data["target_runtime_id"] == "configured-python"
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
