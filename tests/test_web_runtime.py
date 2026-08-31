"""Production Web runtime wiring tests."""

from __future__ import annotations

from pathlib import Path
from threading import Event

import pytest

from coding_agent.application import RepositoryRunSpec
from coding_agent.cancellation import CancellationToken
from coding_agent.command import CommandPermissionMode
from coding_agent.events import EventSink
from coding_agent.models import AgentResult, AgentState, ChatMessage, MessageRole, StopReason
from coding_agent.state import StatePaths
from coding_agent.web.runtime import WebRepositoryConfig, create_repository_service


def _interrupted_result(spec: RepositoryRunSpec) -> AgentResult:
    return AgentResult(
        run_id=spec.run_id,
        state=AgentState.FAILED,
        stop_reason=StopReason.USER_INTERRUPTED,
        steps=1,
        error="Run interrupted by host",
        messages=(
            ChatMessage(role=MessageRole.SYSTEM, content="system"),
            ChatMessage(role=MessageRole.USER, content=spec.task),
        ),
    )


def test_repository_service_shutdown_reaches_application_cancellation_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    started = Event()
    tokens: list[CancellationToken] = []

    def fake_execute(
        spec: RepositoryRunSpec,
        *,
        event_sink: EventSink,
        approver: object,
        cancellation_token: CancellationToken,
    ) -> AgentResult:
        del event_sink, approver
        tokens.append(cancellation_token)
        started.set()
        assert cancellation_token.wait(timeout=2)
        return _interrupted_result(spec)

    monkeypatch.setattr("coding_agent.web.runtime.execute_repository_run", fake_execute)
    service = create_repository_service(
        WebRepositoryConfig(
            root=root,
            model_name="test-model",
            base_url=None,
            reasoning_effort=None,
            permission_mode=CommandPermissionMode.SAFE,
            paths=StatePaths((tmp_path / "state").resolve()),
            max_steps=8,
            model_timeout=10,
        )
    )

    assert service.start("cancel fixed-root run")["status"] == "running"
    assert started.wait(timeout=2)
    assert service.shutdown(timeout=0.01) is True
    assert len(tokens) == 1
    assert tokens[0].is_cancellation_requested is True
    assert service.state()["status"] == "failed"
