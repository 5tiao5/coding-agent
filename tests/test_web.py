"""Local Web API tests with an injected synchronous runner."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Timer, current_thread

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from coding_agent.errors import CodedError
from coding_agent.events import EventKind, EventSink, RunEvent
from coding_agent.models import (
    AgentResult,
    AgentState,
    ChatMessage,
    MessageRole,
    StopReason,
)
from coding_agent.web.app import DEFAULT_SHUTDOWN_DRAIN_SECONDS, create_app
from coding_agent.web.service import WebRunService, WebServiceClosedError


def _test_client(app: FastAPI) -> TestClient:
    return TestClient(app, base_url="http://localhost")


def _messages(task: str, final_text: str | None = None) -> tuple[ChatMessage, ...]:
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="system"),
        ChatMessage(role=MessageRole.USER, content=task),
    ]
    if final_text is not None:
        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=final_text))
    return tuple(messages)


def _completed_result(
    run_id: str, task: str, *, final_text: str = "Repair complete"
) -> AgentResult:
    return AgentResult(
        run_id=run_id,
        state=AgentState.COMPLETED,
        stop_reason=StopReason.FINAL_RESPONSE,
        steps=2,
        final_text=final_text,
        messages=_messages(task, final_text),
    )


def _emit_completed_story(run_id: str, sink: EventSink) -> None:
    sink.emit(
        RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_STARTED,
            message="raw event message must not reach the browser",
            data={
                "private": "raw event data must not reach the browser",
                "limits": {
                    "max_model_turns": 20,
                    "max_calls_per_turn": 8,
                    "max_total_tool_calls": 40,
                    "private": "must-not-leak",
                },
            },
        )
    )
    sink.emit(
        RunEvent(
            run_id=run_id,
            kind=EventKind.TOOL_STARTED,
            message="tool start",
            step=1,
            data={"call_id": "edit-1", "tool_name": "replace_text"},
        )
    )
    sink.emit(
        RunEvent(
            run_id=run_id,
            kind=EventKind.TOOL_FINISHED,
            message="tool finish",
            step=1,
            data={
                "call_id": "edit-1",
                "tool_name": "replace_text",
                "ok": True,
                "summary": "Updated src/example.py (+1/-1)",
                "duration_ms": 12.5,
                "preview": (
                    "Diff preview:\n--- a/src/example.py\n+++ b/src/example.py\n"
                    "@@ -1 +1 @@\n-old\n+new"
                ),
                "metadata": {
                    "path": "src/example.py",
                    "changed": True,
                    "added_lines": 1,
                    "removed_lines": 1,
                    "mutation_revision": 1,
                    "change_kind": "update",
                    "private_sha256": "must-not-leak",
                },
            },
        )
    )
    sink.emit(
        RunEvent(
            run_id=run_id,
            kind=EventKind.VERIFICATION_RECORDED,
            message="verification",
            step=2,
            data={"passed": True, "kind": "test", "label": "pytest", "epoch": 1},
        )
    )
    terminal = {
        "verified": True,
        "status": "verified",
        "epoch": 1,
        "invalidation_count": 1,
        "evidence_labels": ["pytest"],
        "evidence": [
            {
                "label": "pytest",
                "kind": "test",
                "passed": True,
                "step": 2,
                "epoch": 1,
                "private": "must-not-leak",
            }
        ],
    }
    sink.emit(
        RunEvent(
            run_id=run_id,
            kind=EventKind.VERIFICATION_EVALUATED,
            message="verified",
            step=2,
            data=terminal,
        )
    )
    sink.emit(
        RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_FINISHED,
            message="finished",
            step=2,
            data=terminal,
        )
    )


def _wait_for(service: WebRunService) -> None:
    assert service.wait(timeout=2), "Web runner did not finish"


def test_api_returns_only_the_whitelisted_dashboard_projection(tmp_path: Path) -> None:
    def runner(run_id: str, task: str, event_sink: EventSink) -> AgentResult:
        _emit_completed_story(run_id, event_sink)
        return _completed_result(run_id, task)

    assets = tmp_path / "assets"
    assets.mkdir()
    assets.joinpath("index.html").write_text("<h1>Local agent</h1>", encoding="utf-8")
    assets.joinpath("app.js").write_text("console.log('ok')", encoding="utf-8")
    service = WebRunService(runner, max_timeline=20)
    client = _test_client(create_app(service, static_dir=assets))

    response = client.post("/api/runs", json={"task": "  Repair the example  "})
    assert response.status_code == 202
    _wait_for(service)
    state_response = client.get("/api/state")

    assert state_response.status_code == 200
    state = state_response.json()
    assert set(state) == {"status", "run_id", "task", "final_text", "error", "snapshot"}
    assert state["status"] == "completed"
    assert state["task"] == "Repair the example"
    assert state["final_text"] == "Repair complete"
    assert state["error"] is None
    snapshot = state["snapshot"]
    assert set(snapshot) == {
        "task_label",
        "phase",
        "current_step",
        "limits",
        "tools_started",
        "tools_finished",
        "tools_failed",
        "active_tools",
        "verification_status",
        "verification_labels",
        "verification_evidence",
        "verification_epoch",
        "invalidation_count",
        "changed_files",
        "outcome",
        "plan_lines",
        "timeline",
        "latest_change",
    }
    assert snapshot["task_label"] == "Repair the example"
    assert snapshot["phase"] == "COMPLETED"
    assert snapshot["current_step"] == 2
    assert snapshot["limits"] == {
        "max_model_turns": 20,
        "max_calls_per_turn": 8,
        "max_total_tool_calls": 40,
    }
    assert snapshot["tools_started"] == 1
    assert snapshot["tools_finished"] == 1
    assert snapshot["verification_status"] == "verified"
    assert snapshot["verification_labels"] == ["pytest"]
    assert snapshot["verification_evidence"] == [
        {"label": "pytest", "kind": "test", "passed": True, "step": 2, "epoch": 1}
    ]
    assert snapshot["verification_epoch"] == 1
    assert snapshot["invalidation_count"] == 1
    assert snapshot["changed_files"] == [
        {
            "path": "src/example.py",
            "added_lines": 1,
            "removed_lines": 1,
            "revision": 1,
            "change_kind": "update",
        }
    ]
    assert snapshot["outcome"] == "VERIFIED"
    assert snapshot["plan_lines"] == []
    assert snapshot["latest_change"]["preview"][-2:] == ["-old", "+new"]
    serialized = state_response.text
    assert "raw event message" not in serialized
    assert "raw event data" not in serialized
    assert "private_sha256" not in serialized
    assert "must-not-leak" not in serialized
    assert client.get("/").text == "<h1>Local agent</h1>"
    assert "console.log" in client.get("/static/app.js").text


def test_api_projects_a_sanitized_model_retry_while_the_run_is_active() -> None:
    retry_visible = Event()
    release = Event()

    def runner(run_id: str, task: str, event_sink: EventSink) -> AgentResult:
        event_sink.emit(RunEvent(run_id=run_id, kind=EventKind.RUN_STARTED, message="started"))
        event_sink.emit(
            RunEvent(
                run_id=run_id,
                kind=EventKind.MODEL_RETRYING,
                message="TEST_PRIVATE_PROVIDER_MESSAGE_SENTINEL",
                step=1,
                data={
                    "attempt": 1,
                    "next_attempt": 2,
                    "max_attempts": 3,
                    "delay_seconds": 0.5,
                    "error_code": "model_request_transient",
                    "provider_body": "TEST_PRIVATE_PROVIDER_BODY_SENTINEL",
                },
            )
        )
        retry_visible.set()
        assert release.wait(timeout=2)
        event_sink.emit(
            RunEvent(
                run_id=run_id,
                kind=EventKind.RUN_FAILED,
                message="failed",
                step=1,
                data={"stop_reason": StopReason.MODEL_ERROR.value},
            )
        )
        return AgentResult(
            run_id=run_id,
            state=AgentState.FAILED,
            stop_reason=StopReason.MODEL_ERROR,
            steps=1,
            error="Model request failed after transient retries",
            messages=_messages(task),
        )

    service = WebRunService(runner)
    client = _test_client(create_app(service))

    assert client.post("/api/runs", json={"task": "retry safely"}).status_code == 202
    assert retry_visible.wait(timeout=2)
    response = client.get("/api/state")
    snapshot = response.json()["snapshot"]

    assert response.json()["status"] == "running"
    assert snapshot["phase"] == "RETRYING"
    retry_entry = snapshot["timeline"][-1]
    assert retry_entry["step"] == 1
    assert retry_entry["category"] == "MODEL"
    assert retry_entry["headline"] == "Transient model failure; retry scheduled"
    assert retry_entry["detail"] == ("Attempt 2 of 3 · after 0.5s · MODEL REQUEST TRANSIENT")
    assert retry_entry["level"] == "warning"
    assert retry_entry["offset_seconds"] >= 0
    assert retry_entry["duration_ms"] is None
    assert retry_entry["preview"] == []
    assert "TEST_PRIVATE_PROVIDER" not in response.text
    release.set()
    _wait_for(service)


def test_api_projects_tool_batch_rejection_as_a_bounded_retry() -> None:
    rejection_visible = Event()
    release = Event()

    def runner(run_id: str, task: str, event_sink: EventSink) -> AgentResult:
        event_sink.emit(
            RunEvent(
                run_id=run_id,
                kind=EventKind.RUN_STARTED,
                message="started",
                data={
                    "limits": {
                        "max_model_turns": 20,
                        "max_calls_per_turn": 8,
                        "max_total_tool_calls": 40,
                    }
                },
            )
        )
        event_sink.emit(
            RunEvent(
                run_id=run_id,
                kind=EventKind.TOOL_BATCH_REJECTED,
                message="TEST_PRIVATE_TOOL_ARGUMENTS",
                step=2,
                data={
                    "requested_calls": 9,
                    "max_calls_per_turn": 8,
                    "rejection_count": 1,
                    "max_rejections": 3,
                    "raw_arguments": "TEST_PRIVATE_TOOL_ARGUMENTS",
                },
            )
        )
        rejection_visible.set()
        assert release.wait(timeout=2)
        event_sink.emit(
            RunEvent(
                run_id=run_id,
                kind=EventKind.RUN_FAILED,
                message="failed",
                step=2,
                data={"stop_reason": StopReason.MODEL_ERROR.value},
            )
        )
        return AgentResult(
            run_id=run_id,
            state=AgentState.FAILED,
            stop_reason=StopReason.MODEL_ERROR,
            steps=2,
            messages=_messages(task),
        )

    service = WebRunService(runner)
    client = _test_client(create_app(service))

    assert client.post("/api/runs", json={"task": "split the batch"}).status_code == 202
    assert rejection_visible.wait(timeout=2)
    response = client.get("/api/state")
    snapshot = response.json()["snapshot"]
    entry = snapshot["timeline"][-1]

    assert response.json()["status"] == "running"
    assert snapshot["phase"] == "REPLANNING"
    assert snapshot["limits"] == {
        "max_model_turns": 20,
        "max_calls_per_turn": 8,
        "max_total_tool_calls": 40,
    }
    assert snapshot["tools_started"] == snapshot["tools_finished"] == 0
    assert snapshot["tools_failed"] == 0
    assert entry["category"] == "MODEL"
    assert entry["level"] == "warning"
    assert entry["headline"] == "Tool batch too large; split retry requested"
    assert entry["detail"] == (
        "Requested 9 tool calls; per-turn limit 8; split retry requested; rejection 1 of 3"
    )
    assert "TEST_PRIVATE_TOOL_ARGUMENTS" not in response.text
    release.set()
    _wait_for(service)


def test_second_concurrent_run_is_rejected_with_conflict() -> None:
    started = Event()
    release = Event()

    def runner(run_id: str, task: str, event_sink: EventSink) -> AgentResult:
        event_sink.emit(RunEvent(run_id=run_id, kind=EventKind.RUN_STARTED, message="started"))
        started.set()
        assert release.wait(timeout=2)
        event_sink.emit(
            RunEvent(
                run_id=run_id,
                kind=EventKind.RUN_FINISHED,
                message="finished",
                data={"verified": True, "status": "verified"},
            )
        )
        return _completed_result(run_id, task)

    service = WebRunService(runner)
    client = _test_client(create_app(service))

    first = client.post("/api/runs", json={"task": "first"})
    assert first.status_code == 202
    assert started.wait(timeout=2)
    second = client.post("/api/runs", json={"task": "second"})

    assert second.status_code == 409
    assert second.json()["detail"] == "已有任务正在运行"
    assert client.get("/api/state").json()["task"] == "first"
    release.set()
    _wait_for(service)


def test_runner_exception_becomes_failed_state() -> None:
    secret = "TEST_RUNNER_SECRET_SENTINEL"

    def runner(run_id: str, task: str, event_sink: EventSink) -> AgentResult:
        del run_id, task, event_sink
        raise RuntimeError(f"provider exploded with {secret}")

    service = WebRunService(runner)
    client = _test_client(create_app(service))

    response = client.post("/api/runs", json={"task": "trigger failure"})
    assert response.status_code == 202
    _wait_for(service)
    state = client.get("/api/state").json()

    assert state["status"] == "failed"
    assert state["final_text"] is None
    assert state["error"] == "本地 Agent 运行失败（RuntimeError）。"
    assert secret not in client.get("/api/state").text
    assert state["snapshot"]["task_label"] == "trigger failure"


def test_model_setup_error_becomes_actionable_without_exposing_exception_text() -> None:
    def runner(run_id: str, task: str, event_sink: EventSink) -> AgentResult:
        del run_id, task, event_sink
        raise CodedError(
            "openai_client_configuration",
            "configuration included TEST_PRIVATE_API_KEY",
        )

    service = WebRunService(runner)
    client = _test_client(create_app(service))

    assert client.post("/api/runs", json={"task": "configure model"}).status_code == 202
    _wait_for(service)
    response = client.get("/api/state")

    assert response.json()["error"] == (
        "模型客户端配置失败。请检查 API Key、接口地址和相关环境变量。"
    )
    assert "TEST_PRIVATE" not in response.text


def test_failed_agent_result_maps_stop_reason_without_exposing_raw_error() -> None:
    secret = "TEST_RESULT_SECRET_SENTINEL"

    def runner(run_id: str, task: str, event_sink: EventSink) -> AgentResult:
        del event_sink
        return AgentResult(
            run_id=run_id,
            state=AgentState.FAILED,
            stop_reason=StopReason.MODEL_ERROR,
            steps=1,
            error=f"provider failure included {secret}",
            messages=_messages(task),
        )

    service = WebRunService(runner)
    client = _test_client(create_app(service))

    assert client.post("/api/runs", json={"task": "trigger result failure"}).status_code == 202
    _wait_for(service)
    response = client.get("/api/state")

    assert response.json()["status"] == "failed"
    assert response.json()["error"] == "模型请求失败。请检查模型配置或稍后重试。"
    assert secret not in response.text


@pytest.mark.parametrize(
    ("private_error", "expected"),
    [
        (
            "Model returned invalid tool-call arguments after protocol recovery attempts",
            "模型返回的工具参数格式无效，自动纠正重试后仍未恢复。",
        ),
        (
            "Model request failed after transient retries",
            "模型服务暂时不可用，自动重试后仍未恢复。请稍后重试。",
        ),
        (
            "Model request failed: OpenAI model request failed",
            "模型请求被服务端拒绝。请检查 API Key、模型名称、接口地址和账户权限。",
        ),
        (
            "Model request failed: OpenAI returned an invalid response: "
            "TEST_PRIVATE_PROVIDER_RESPONSE",
            "模型返回了无法识别的响应格式，请重试或更换兼容模型。",
        ),
    ],
)
def test_model_failures_map_to_specific_safe_public_reasons(
    private_error: str,
    expected: str,
) -> None:
    def runner(run_id: str, task: str, event_sink: EventSink) -> AgentResult:
        del event_sink
        return AgentResult(
            run_id=run_id,
            state=AgentState.FAILED,
            stop_reason=StopReason.MODEL_ERROR,
            steps=2,
            error=private_error,
            messages=_messages(task),
        )

    service = WebRunService(runner)
    client = _test_client(create_app(service))

    assert client.post("/api/runs", json={"task": "explain model failure"}).status_code == 202
    _wait_for(service)
    response = client.get("/api/state")

    assert response.json()["error"] == expected
    assert "TEST_PRIVATE" not in response.text


@pytest.mark.parametrize(
    ("private_error", "expected"),
    [
        (
            "Model exceeded the per-step tool call limit 3 consecutive times: 9 > 8",
            "模型连续提交过大的工具批次（本轮 9，上限 8）。",
        ),
        (
            "Model exceeded the total tool call limit: 41 > 40",
            "模型达到工具调用总上限（累计请求 41，上限 40）。",
        ),
        (
            "Model exceeded the total tool call limit: SECRET > 40",
            "任务已达到设定的工具调用上限。",
        ),
    ],
)
def test_tool_limit_error_exposes_only_exact_agent_owned_counts(
    private_error: str,
    expected: str,
) -> None:
    def runner(run_id: str, task: str, event_sink: EventSink) -> AgentResult:
        del event_sink
        return AgentResult(
            run_id=run_id,
            state=AgentState.FAILED,
            stop_reason=StopReason.TOOL_LIMIT,
            steps=3,
            error=private_error,
            messages=_messages(task),
        )

    service = WebRunService(runner)
    client = _test_client(create_app(service))

    assert client.post("/api/runs", json={"task": "respect tool limits"}).status_code == 202
    _wait_for(service)
    response = client.get("/api/state")

    assert response.json()["error"] == expected
    if "SECRET" in private_error:
        assert "SECRET" not in response.text


def test_app_factory_defaults_to_the_packaged_static_directory() -> None:
    def runner(run_id: str, task: str, event_sink: EventSink) -> AgentResult:
        del event_sink
        return _completed_result(run_id, task)

    client = _test_client(create_app(WebRunService(runner)))

    assert client.get("/").status_code == 200
    for stylesheet in (
        "styles.css",
        "conversation.css",
        "inspector.css",
        "responsive.css",
    ):
        assert client.get(f"/static/{stylesheet}").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/_metrics.js").status_code == 200
    assert client.get("/static/_workbench.js").status_code == 200
    assert client.get("/static/locale-zh.js").status_code == 200
    assert client.get("/static/favicon.svg").status_code == 200


def test_api_rejects_blank_or_unexpected_run_fields() -> None:
    def runner(run_id: str, task: str, event_sink: EventSink) -> AgentResult:
        del event_sink
        return _completed_result(run_id, task)

    client = _test_client(create_app(WebRunService(runner)))

    assert client.post("/api/runs", json={"task": "   "}).status_code == 422
    assert (
        client.post(
            "/api/runs",
            json={"task": "valid", "api_key": "must-not-be-accepted"},
        ).status_code
        == 422
    )


def test_local_app_rejects_untrusted_hosts_and_sets_browser_security_headers() -> None:
    def runner(run_id: str, task: str, event_sink: EventSink) -> AgentResult:
        del event_sink
        return _completed_result(run_id, task)

    client = _test_client(create_app(WebRunService(runner)))

    rejected = client.get("/api/state", headers={"Host": "attacker.example"})
    accepted = client.get("/api/state")

    assert rejected.status_code == 400
    assert accepted.headers["content-security-policy"].startswith("default-src 'self'")
    assert accepted.headers["referrer-policy"] == "no-referrer"
    assert accepted.headers["x-content-type-options"] == "nosniff"
    assert accepted.headers["x-frame-options"] == "DENY"
    assert accepted.headers["cache-control"] == "no-store"


def test_metadata_exposes_only_server_owned_display_settings() -> None:
    def runner(run_id: str, task: str, event_sink: EventSink) -> AgentResult:
        del event_sink
        return _completed_result(run_id, task)

    client = _test_client(
        create_app(
            WebRunService(runner),
            workspace_label="demo-fixture",
            runtime_label="Offline deterministic demo",
            default_task="Repair the pricing helper",
            task_locked=True,
        )
    )

    response = client.get("/api/meta")

    assert response.status_code == 200
    assert response.json() == {
        "workspace": "demo-fixture",
        "runtime": "Offline deterministic demo",
        "task_locked": True,
        "default_task": "Repair the pricing helper",
        "native_folder_picker_available": False,
    }
    assert client.post("/api/folders/pick", json={}).status_code == 404
    assert client.post("/api/runs", json={"task": "Run an unrelated task"}).status_code == 422


def test_locked_metadata_requires_a_default_task() -> None:
    def runner(run_id: str, task: str, event_sink: EventSink) -> AgentResult:
        del event_sink
        return _completed_result(run_id, task)

    with pytest.raises(ValueError, match="locked Web task requires default_task"):
        create_app(WebRunService(runner), task_locked=True)


def test_app_shutdown_drains_the_active_run_and_closes_the_service() -> None:
    started = Event()
    release = Event()
    finished = Event()

    def runner(run_id: str, task: str, event_sink: EventSink) -> AgentResult:
        event_sink.emit(RunEvent(run_id=run_id, kind=EventKind.RUN_STARTED, message="started"))
        started.set()
        assert release.wait(timeout=2)
        finished.set()
        return _completed_result(run_id, task)

    service = WebRunService(runner)
    release_timer = Timer(0.05, release.set)
    with _test_client(create_app(service)) as client:
        assert client.post("/api/runs", json={"task": "finish before shutdown"}).status_code == 202
        assert started.wait(timeout=2)
        release_timer.start()

    release_timer.join(timeout=1)
    assert finished.is_set()
    assert service.state()["status"] == "completed"
    with pytest.raises(WebServiceClosedError, match="正在关闭"):
        service.start("must not start")


def test_shutdown_timeout_closes_admission_and_leaves_a_daemon_fallback() -> None:
    started = Event()
    release = Event()
    worker_daemon: list[bool] = []

    def runner(run_id: str, task: str, event_sink: EventSink) -> AgentResult:
        del event_sink
        worker_daemon.append(current_thread().daemon)
        started.set()
        assert release.wait(timeout=2)
        return _completed_result(run_id, task)

    service = WebRunService(runner)
    service.start("outlive the drain window")
    assert started.wait(timeout=2)

    assert service.shutdown(timeout=0) is False
    assert worker_daemon == [True]
    with pytest.raises(WebServiceClosedError, match="正在关闭"):
        service.start("must not start after shutdown")

    release.set()
    assert service.wait(timeout=2)
    assert service.state()["status"] == "completed"


def test_app_lifespan_uses_a_finite_shutdown_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_timeouts: list[float | None] = []

    def runner(run_id: str, task: str, event_sink: EventSink) -> AgentResult:
        del event_sink
        return _completed_result(run_id, task)

    service = WebRunService(runner)

    def fake_shutdown(timeout: float | None = None) -> bool:
        captured_timeouts.append(timeout)
        return True

    monkeypatch.setattr(service, "shutdown", fake_shutdown)

    with _test_client(create_app(service)):
        pass

    assert captured_timeouts == [DEFAULT_SHUTDOWN_DRAIN_SECONDS]
    assert DEFAULT_SHUTDOWN_DRAIN_SECONDS == 5.0


def test_worker_start_failure_rolls_back_running_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RefusingThread:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def is_alive(self) -> bool:
            return False

        def start(self) -> None:
            raise RuntimeError("thread quota exhausted")

        def join(self, timeout: float | None = None) -> None:
            del timeout

    def runner(run_id: str, task: str, event_sink: EventSink) -> AgentResult:
        del event_sink
        return _completed_result(run_id, task)

    monkeypatch.setattr("coding_agent.web.service.Thread", RefusingThread)
    service = WebRunService(runner)
    client = _test_client(create_app(service))

    response = client.post("/api/runs", json={"task": "start a worker"})

    assert response.status_code == 503
    assert response.json()["detail"] == "本地 Agent 后台任务启动失败。"
    state = client.get("/api/state").json()
    assert state["status"] == "failed"
    assert state["error"] == "本地 Agent 后台任务启动失败。"
