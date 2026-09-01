"""Project workbench API tests with a deterministic repository runner."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread

import pytest
from fastapi.testclient import TestClient

from coding_agent.application import RepositoryRunSpec
from coding_agent.cancellation import CancellationToken
from coding_agent.cli import _ensure_run_catalog_entry
from coding_agent.command import CommandPermissionMode
from coding_agent.events import EventKind, EventSink, RunEvent
from coding_agent.models import (
    AgentResult,
    AgentState,
    ChatMessage,
    MessageRole,
    StopReason,
)
from coding_agent.run_catalog import MAX_TASK_TITLE_LENGTH, RunCatalog
from coding_agent.session import (
    SessionBoundary,
    SessionCheckpoint,
    SessionStore,
    workspace_fingerprint,
)
from coding_agent.state import StatePaths
from coding_agent.trace import TraceStore
from coding_agent.web.app import create_app
from coding_agent.web.native_picker import (
    NativeFolderPicker,
    NativeFolderPickerBusyError,
    NativeFolderPickerUnavailableError,
)
from coding_agent.web.workbench import WebWorkbench, WebWorkbenchConfig


def _workbench(tmp_path: Path) -> WebWorkbench:
    return WebWorkbench(
        WebWorkbenchConfig(
            model_name="test-model",
            base_url=None,
            reasoning_effort=None,
            permission_mode=CommandPermissionMode.SAFE,
            paths=StatePaths((tmp_path / "state").resolve()),
            max_steps=8,
            model_timeout=10.0,
        )
    )


def _client(
    workbench: WebWorkbench,
    *,
    native_folder_picker: NativeFolderPicker | None = None,
) -> TestClient:
    return TestClient(
        create_app(
            workbench,
            workbench=workbench,
            native_folder_picker=native_folder_picker,
        ),
        base_url="http://localhost",
    )


class _FakePicker:
    def __init__(
        self,
        choose: Callable[[], Path | None],
        *,
        available: bool = True,
    ) -> None:
        self._choose = choose
        self.available = available
        self.calls = 0

    def pick_directory(self) -> Path | None:
        self.calls += 1
        return self._choose()


def _token(client: TestClient) -> str:
    value = client.get("/api/meta").json()["control_token"]
    assert isinstance(value, str) and len(value) >= 32
    return value


def _headers(token: str, *, origin: str | None = None) -> dict[str, str]:
    result = {"X-Coding-Agent-Token": token}
    if origin is not None:
        result["Origin"] = origin
    return result


def _completed_result(run_id: str, task: str) -> AgentResult:
    final_text = "Repository task complete"
    return AgentResult(
        run_id=run_id,
        state=AgentState.COMPLETED,
        stop_reason=StopReason.FINAL_RESPONSE,
        steps=1,
        final_text=final_text,
        messages=(
            ChatMessage(role=MessageRole.SYSTEM, content="system"),
            ChatMessage(role=MessageRole.USER, content=task),
            ChatMessage(role=MessageRole.ASSISTANT, content=final_text),
        ),
    )


def _save_terminal_checkpoint(spec: RepositoryRunSpec, result: AgentResult) -> None:
    store = SessionStore(spec.paths.sessions, workspace_root=spec.root)
    system_prompt = result.messages[0].content
    assert system_prompt is not None
    store.save(
        SessionCheckpoint(
            run_id=result.run_id,
            workspace_fingerprint=store.workspace_fingerprint,
            task=spec.task,
            system_prompt=system_prompt,
            messages=result.messages,
            completed_steps=result.steps,
            completed_tool_calls=0,
            completed_work_tool_calls=0,
            completed_verification_tool_calls=0,
            stop_boundary=SessionBoundary.TERMINAL,
            stop_reason=StopReason.FINAL_RESPONSE,
        )
    )


def _interrupted_result(run_id: str, task: str) -> AgentResult:
    return AgentResult(
        run_id=run_id,
        state=AgentState.FAILED,
        stop_reason=StopReason.USER_INTERRUPTED,
        steps=1,
        error="Run interrupted by host",
        messages=(
            ChatMessage(role=MessageRole.SYSTEM, content="system"),
            ChatMessage(role=MessageRole.USER, content=task),
        ),
    )


def test_workbench_shutdown_passes_cooperative_cancellation_to_repository_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    started = Event()
    received_tokens: list[CancellationToken] = []

    def fake_execute(
        spec: RepositoryRunSpec,
        *,
        event_sink: EventSink,
        approver: object,
        cancellation_token: CancellationToken,
    ) -> AgentResult:
        del event_sink, approver
        received_tokens.append(cancellation_token)
        started.set()
        assert cancellation_token.wait(timeout=2)
        return _interrupted_result(spec.run_id, spec.task)

    monkeypatch.setattr("coding_agent.web.workbench.execute_repository_run", fake_execute)
    workbench = _workbench(tmp_path)
    workbench.register_project(root=workspace)

    assert workbench.start("cooperatively stop this run")["status"] == "running"
    assert started.wait(timeout=2)
    assert workbench.shutdown(timeout=0.01) is True

    assert len(received_tokens) == 1
    assert received_tokens[0].is_cancellation_requested is True
    state = workbench.state()
    assert state["status"] == "failed"
    assert state["error"] == "任务已由用户中断。"


def test_native_folder_picker_is_token_protected_and_distinguishes_cancel(
    tmp_path: Path,
) -> None:
    selected_root = tmp_path / "picked"
    selected_root.mkdir()
    choices: list[Path | None] = [selected_root.resolve(), None]
    picker = _FakePicker(lambda: choices.pop(0))
    workbench = _workbench(tmp_path)
    client = _client(workbench, native_folder_picker=picker)
    token = _token(client)

    metadata = client.get("/api/meta").json()
    assert metadata["native_folder_picker_available"] is True
    assert client.post("/api/folders/pick", json={}).status_code == 403
    assert (
        client.post(
            "/api/folders/pick",
            json={},
            headers=_headers(token, origin="https://attacker.example"),
        ).status_code
        == 403
    )
    assert picker.calls == 0

    selected = client.post(
        "/api/folders/pick",
        json={},
        headers=_headers(token, origin="http://localhost"),
    )
    cancelled = client.post(
        "/api/folders/pick",
        json={},
        headers=_headers(token),
    )

    assert selected.status_code == 200
    assert selected.json() == {"status": "selected", "path": str(selected_root.resolve())}
    assert cancelled.status_code == 200
    assert cancelled.json() == {"status": "cancelled", "path": None}
    assert client.get("/api/projects").json()["projects"] == []
    assert picker.calls == 2
    assert (
        client.post(
            "/api/projects",
            json={"root": str(selected_root)},
            headers=_headers(token),
        ).status_code
        == 201
    )
    assert (
        client.post(
            "/api/folders/pick",
            json={"unexpected": True},
            headers=_headers(token),
        ).status_code
        == 422
    )


def test_native_folder_picker_failures_are_stable_and_path_free(tmp_path: Path) -> None:
    workbench = _workbench(tmp_path)
    recovered_root = tmp_path / "recovered"
    recovered_root.mkdir()

    unavailable = _FakePicker(lambda: tmp_path, available=False)
    unavailable_client = _client(workbench, native_folder_picker=unavailable)
    unavailable_response = unavailable_client.post(
        "/api/folders/pick",
        json={},
        headers=_headers(_token(unavailable_client)),
    )
    assert unavailable_response.status_code == 503
    assert unavailable_response.json() == {"detail": "本机文件夹选择器暂时不可用"}
    assert unavailable.calls == 0

    def busy() -> Path | None:
        raise NativeFolderPickerBusyError("PRIVATE_BUSY_DETAIL")

    busy_picker = _FakePicker(busy)
    busy_client = _client(workbench, native_folder_picker=busy_picker)
    busy_response = busy_client.post(
        "/api/folders/pick",
        json={},
        headers=_headers(_token(busy_client)),
    )
    assert busy_response.status_code == 409
    assert busy_response.json() == {"detail": "已有文件夹选择窗口正在等待操作"}
    assert "PRIVATE_BUSY_DETAIL" not in busy_response.text

    def fail() -> Path | None:
        raise RuntimeError(f"PRIVATE_PATH: {tmp_path}")

    failed_picker = _FakePicker(fail)
    failed_client = _client(workbench, native_folder_picker=failed_picker)
    failed_response = failed_client.post(
        "/api/folders/pick",
        json={},
        headers=_headers(_token(failed_client)),
    )
    assert failed_response.status_code == 503
    assert failed_response.json() == {"detail": "本机文件夹选择器暂时不可用"}
    assert "PRIVATE_PATH" not in failed_response.text
    assert str(tmp_path) not in failed_response.text

    def native_fail() -> Path | None:
        raise NativeFolderPickerUnavailableError("PRIVATE_NATIVE_DETAIL")

    native_failed_picker = _FakePicker(native_fail)
    native_failed_client = _client(workbench, native_folder_picker=native_failed_picker)
    native_failed_response = native_failed_client.post(
        "/api/folders/pick",
        json={},
        headers=_headers(_token(native_failed_client)),
    )
    assert native_failed_response.status_code == 503
    assert native_failed_response.json() == {"detail": "本机文件夹选择器暂时不可用"}
    assert "PRIVATE_NATIVE_DETAIL" not in native_failed_response.text
    assert (
        native_failed_client.post(
            "/api/projects",
            json={"root": str(recovered_root)},
            headers=_headers(_token(native_failed_client)),
        ).status_code
        == 201
    )


def test_native_folder_picker_is_not_opened_during_an_active_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_started = Event()
    release_run = Event()

    def fake_execute(
        spec: RepositoryRunSpec,
        *,
        event_sink: EventSink,
        approver: object,
        cancellation_token: CancellationToken,
    ) -> AgentResult:
        del event_sink, approver, cancellation_token
        run_started.set()
        assert release_run.wait(timeout=2)
        return _completed_result(spec.run_id, spec.task)

    monkeypatch.setattr("coding_agent.web.workbench.execute_repository_run", fake_execute)
    picker = _FakePicker(lambda: workspace)
    workbench = _workbench(tmp_path)
    client = _client(workbench, native_folder_picker=picker)
    token = _token(client)
    assert (
        client.post(
            "/api/projects",
            json={"root": str(workspace)},
            headers=_headers(token),
        ).status_code
        == 201
    )
    assert (
        client.post(
            "/api/runs",
            json={"task": "hold the sole worker"},
            headers=_headers(token),
        ).status_code
        == 202
    )
    assert run_started.wait(timeout=1)

    response = client.post(
        "/api/folders/pick",
        json={},
        headers=_headers(token),
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "任务运行期间不能登记或切换项目"}
    assert picker.calls == 0
    release_run.set()
    assert workbench.wait(timeout=2)


def test_folder_dialog_reservation_rejects_every_racing_navigation(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    third = tmp_path / "third"
    picked = tmp_path / "picked"
    for directory in (first, second, third, picked):
        directory.mkdir()
    dialog_open = Event()
    release_dialog = Event()

    def choose() -> Path:
        dialog_open.set()
        assert release_dialog.wait(timeout=3)
        return picked.resolve()

    picker = _FakePicker(choose)
    workbench = _workbench(tmp_path)
    client = _client(workbench, native_folder_picker=picker)
    token = _token(client)
    first_record = client.post(
        "/api/projects",
        json={"root": str(first)},
        headers=_headers(token),
    ).json()
    second_record = client.post(
        "/api/projects",
        json={"root": str(second)},
        headers=_headers(token),
    ).json()
    first_response: dict[str, object] = {}

    def request_picker() -> None:
        response = client.post(
            "/api/folders/pick",
            json={},
            headers=_headers(token),
        )
        first_response["status_code"] = response.status_code
        first_response["body"] = response.json()

    request_thread = Thread(target=request_picker)
    request_thread.start()
    assert dialog_open.wait(timeout=1)

    blocked_run = client.post(
        "/api/runs",
        json={"task": "must not capture a racing project"},
        headers=_headers(token),
    )
    blocked_register = client.post(
        "/api/projects",
        json={"root": str(third)},
        headers=_headers(token),
    )
    blocked_select = client.post(
        f"/api/projects/{first_record['project_id']}/select",
        json={},
        headers=_headers(token),
    )
    blocked_picker = client.post(
        "/api/folders/pick",
        json={},
        headers=_headers(token),
    )

    assert blocked_run.status_code == 409
    assert blocked_run.json() == {"detail": "正在选择项目目录，请稍后再启动任务"}
    assert blocked_register.status_code == 409
    assert blocked_select.status_code == 409
    assert blocked_picker.status_code == 409
    assert blocked_picker.json() == {"detail": "正在选择项目目录，请稍后再试"}
    assert picker.calls == 1
    assert client.get("/api/state").json()["status"] == "idle"
    projects = client.get("/api/projects").json()
    assert projects["active_project_id"] == second_record["project_id"]
    assert {record["root"] for record in projects["projects"]} == {
        str(first.resolve()),
        str(second.resolve()),
    }
    assert client.get(f"/api/projects/{first_record['project_id']}/runs").json()["runs"] == []
    assert client.get(f"/api/projects/{second_record['project_id']}/runs").json()["runs"] == []

    release_dialog.set()
    request_thread.join(timeout=2)
    assert not request_thread.is_alive()
    assert first_response == {
        "status_code": 200,
        "body": {"status": "selected", "path": str(picked.resolve())},
    }
    assert (
        client.post(
            f"/api/projects/{first_record['project_id']}/select",
            json={},
            headers=_headers(token),
        ).status_code
        == 200
    )


def test_project_registration_creation_selection_and_control_token(tmp_path: Path) -> None:
    first = tmp_path / "first"
    first.mkdir()
    created = tmp_path / "new-project"
    workbench = _workbench(tmp_path)
    client = _client(workbench)
    token = _token(client)

    assert client.post("/api/projects", json={"root": str(first)}).status_code == 403
    assert (
        client.post(
            "/api/projects",
            json={"root": str(first)},
            headers=_headers(token, origin="https://attacker.example"),
        ).status_code
        == 403
    )

    registered = client.post(
        "/api/projects",
        json={"root": str(first), "display_name": "第一个项目", "create": False},
        headers=_headers(token, origin="http://localhost"),
    )
    assert registered.status_code == 201, registered.text
    first_record = registered.json()
    assert first_record["display_name"] == "第一个项目"
    assert Path(first_record["root"]) == first.resolve()

    created_response = client.post(
        "/api/projects",
        json={"root": str(created), "create": True},
        headers=_headers(token),
    )
    assert created_response.status_code == 201, created_response.text
    assert created.is_dir()
    assert (
        client.get("/api/projects").json()["active_project_id"]
        == (created_response.json()["project_id"])
    )

    selected = client.post(
        f"/api/projects/{first_record['project_id']}/select",
        json={},
        headers=_headers(token),
    )
    assert selected.status_code == 200
    assert client.get("/api/meta").json()["workspace"] == "第一个项目"
    assert (
        client.post(
            f"/api/projects/{first_record['project_id']}/select",
            json={"root": str(created)},
            headers=_headers(token),
        ).status_code
        == 422
    )


def test_run_requires_an_active_project_and_never_accepts_a_raw_root(tmp_path: Path) -> None:
    client = _client(_workbench(tmp_path))
    token = _token(client)

    missing = client.post(
        "/api/runs",
        json={"task": "inspect the project"},
        headers=_headers(token),
    )
    injected = client.post(
        "/api/runs",
        json={"task": "inspect", "root": str(tmp_path)},
        headers=_headers(token),
    )

    assert missing.status_code == 409
    assert missing.json()["detail"] == "请先选择一个项目"
    assert injected.status_code == 422


@pytest.mark.parametrize("task", ["inspect\x00project", "inspect\u202eproject"])
def test_run_rejects_control_and_formatting_characters(
    tmp_path: Path,
    task: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = _client(_workbench(tmp_path))
    token = _token(client)
    assert (
        client.post(
            "/api/projects",
            json={"root": str(workspace)},
            headers=_headers(token),
        ).status_code
        == 201
    )

    response = client.post(
        "/api/runs",
        json={"task": task},
        headers=_headers(token),
    )

    assert response.status_code == 422


def test_replaced_project_directory_must_be_explicitly_registered_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workbench = _workbench(tmp_path)
    client = _client(workbench)
    token = _token(client)
    record = client.post(
        "/api/projects",
        json={"root": str(workspace)},
        headers=_headers(token),
    ).json()

    monkeypatch.setattr(
        "coding_agent.web.workbench.workspace_fingerprint",
        lambda root: "mismatched-directory-identity",
    )

    refused_select = client.post(
        f"/api/projects/{record['project_id']}/select",
        json={},
        headers=_headers(token),
    )
    refused_run = client.post(
        "/api/runs",
        json={"task": "must not cross the replaced identity"},
        headers=_headers(token),
    )

    assert refused_select.status_code == 422
    assert "重新登记" in refused_select.json()["detail"]
    assert refused_run.status_code == 409
    assert "重新登记" in refused_run.json()["detail"]


def test_recreated_directory_keeps_project_slot_but_filters_old_identity_runs(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    paths = StatePaths((tmp_path / "state").resolve())
    workbench = _workbench(tmp_path)
    client = _client(workbench)
    token = _token(client)
    registered = client.post(
        "/api/projects",
        json={"root": str(workspace)},
        headers=_headers(token),
    ).json()
    project_id = registered["project_id"]
    original_fingerprint = workspace_fingerprint(workspace)
    catalog = RunCatalog(paths.runs)
    catalog.create(
        run_id="old-identity-run",
        project_id=project_id,
        workspace_fingerprint=original_fingerprint,
        task_title="Old physical directory",
    )
    TraceStore(paths.traces).append(
        RunEvent(
            run_id="old-identity-run",
            kind=EventKind.RUN_STARTED,
            message="Old identity started",
        )
    )

    workspace.rmdir()
    replacement_fingerprint = original_fingerprint
    for index in range(32):
        (tmp_path / f"identity-consumer-{index}").mkdir()
        workspace.mkdir()
        replacement_fingerprint = workspace_fingerprint(workspace)
        if replacement_fingerprint != original_fingerprint:
            break
        workspace.rmdir()
    assert replacement_fingerprint != original_fingerprint

    reopened = client.post(
        "/api/projects",
        json={"root": str(workspace)},
        headers=_headers(token),
    )
    assert reopened.status_code == 201, reopened.text
    assert reopened.json()["project_id"] == project_id
    catalog.create(
        run_id="current-identity-run",
        project_id=project_id,
        workspace_fingerprint=replacement_fingerprint,
        task_title="Current physical directory",
    )

    visible = client.get(f"/api/projects/{project_id}/runs")

    assert visible.status_code == 200, visible.text
    assert [run["run_id"] for run in visible.json()["runs"]] == ["current-identity-run"]
    assert client.get("/api/history/old-identity-run").status_code == 404
    assert {record.run_id for record in catalog.list(project_id=project_id)} == {
        "old-identity-run",
        "current-identity-run",
    }


def test_run_context_is_immutable_and_history_is_a_whitelisted_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    started = Event()
    release = Event()
    captured: list[RepositoryRunSpec] = []

    def fake_execute(
        spec: RepositoryRunSpec,
        *,
        event_sink: EventSink,
        approver: object,
        cancellation_token: CancellationToken,
    ) -> AgentResult:
        del approver, cancellation_token
        captured.append(spec)
        trace = TraceStore(spec.paths.traces)

        def emit(event: RunEvent) -> None:
            trace.append(event)
            event_sink.emit(event)

        emit(
            RunEvent(
                run_id=spec.run_id,
                kind=EventKind.RUN_STARTED,
                message="PRIVATE_RAW_TRACE_MESSAGE",
                data={
                    "private": "PRIVATE_RAW_TRACE_DATA",
                    "limits": {
                        "max_model_turns": 8,
                        "max_calls_per_turn": 8,
                        "max_total_tool_calls": 40,
                    },
                },
            )
        )
        started.set()
        assert release.wait(timeout=3)
        invocation = {
            "executable": "python",
            "argument_count": 3,
            "argv": ["python", "-m", "pytest", "-q"],
            "credentials_redacted": False,
            "cwd": ".",
            "timeout_seconds": 120.0,
            "verification_label": "pytest",
            "verification_kind": "test",
            "private": "PRIVATE INVOCATION",
        }
        emit(
            RunEvent(
                run_id=spec.run_id,
                kind=EventKind.TOOL_STARTED,
                message="PRIVATE COMMAND START",
                step=1,
                data={
                    "call_id": "provider-command-call",
                    "tool_name": "run_command",
                    "public_invocation": invocation,
                },
            )
        )
        emit(
            RunEvent(
                run_id=spec.run_id,
                kind=EventKind.TOOL_FINISHED,
                message="PRIVATE COMMAND FINISH",
                step=1,
                data={
                    "call_id": "provider-command-call",
                    "tool_name": "run_command",
                    "ok": True,
                    "summary": "Command exited 0 in .",
                    "truncated": False,
                    "public_invocation": invocation,
                    "metadata": {
                        "command_class": "verifier",
                        "cwd": ".",
                        "status": "exited",
                        "exit_code": 0,
                        "captured_output_bytes": 10,
                        "total_output_bytes": 10,
                        "private": "PRIVATE COMMAND METADATA",
                    },
                    "public_output": {
                        "captured_text": "1 passed",
                        "captured_projection_truncated": False,
                        "observation_truncated": False,
                        "credentials_redacted": False,
                    },
                },
            )
        )
        expanded_diff = [
            "--- a/src/example.py",
            "+++ b/src/example.py",
            "@@ -1,30 +1,30 @@",
            *(f"+visible line {index}" for index in range(30)),
        ]
        emit(
            RunEvent(
                run_id=spec.run_id,
                kind=EventKind.TOOL_FINISHED,
                message="PRIVATE_MUTATION_EVENT",
                step=1,
                data={
                    "tool_name": "write_file",
                    "ok": True,
                    "summary": "Updated src/example.py",
                    "preview": "Diff preview:\n" + "\n".join(expanded_diff),
                    "truncated": False,
                    "metadata": {
                        "diff_complete": True,
                        "private": "PRIVATE_MUTATION_METADATA",
                    },
                },
            )
        )
        terminal = {
            "verified": True,
            "status": "verified",
            "epoch": 0,
            "invalidation_count": 0,
            "evidence_labels": [],
            "evidence": [],
        }
        emit(
            RunEvent(
                run_id=spec.run_id,
                kind=EventKind.VERIFICATION_EVALUATED,
                message="verified",
                step=1,
                data=terminal,
            )
        )
        result = _completed_result(spec.run_id, spec.task)
        _save_terminal_checkpoint(spec, result)
        emit(
            RunEvent(
                run_id=spec.run_id,
                kind=EventKind.RUN_FINISHED,
                message="finished",
                step=1,
                data=terminal,
            )
        )
        return result

    monkeypatch.setattr("coding_agent.web.workbench.execute_repository_run", fake_execute)
    workbench = _workbench(tmp_path)
    client = _client(workbench)
    token = _token(client)
    first_record = client.post(
        "/api/projects",
        json={"root": str(first)},
        headers=_headers(token),
    ).json()

    started_response = client.post(
        "/api/runs",
        json={"task": "  Repair\nthis project  "},
        headers=_headers(token),
    )
    assert started_response.status_code == 202
    run_id = started_response.json()["run_id"]
    assert started.wait(timeout=2)
    active_runs = client.get(f"/api/projects/{first_record['project_id']}/runs")
    assert active_runs.json()["runs"][0]["status"] == "running"

    blocked = client.post(
        "/api/projects",
        json={"root": str(second)},
        headers=_headers(token),
    )
    assert blocked.status_code == 409
    release.set()
    assert workbench.wait(timeout=3)
    assert captured[0].root == first.resolve()
    live_state = client.get("/api/state").json()

    runs_response = client.get(f"/api/projects/{first_record['project_id']}/runs")
    assert runs_response.status_code == 200
    assert runs_response.json()["runs"] == [
        {
            "run_id": run_id,
            "project_id": first_record["project_id"],
            "task": "Repair this project",
            "status": "completed",
            "created_at": runs_response.json()["runs"][0]["created_at"],
            "completed_at": runs_response.json()["runs"][0]["completed_at"],
            "final_text": None,
            "error": None,
        }
    ]

    history = client.get(f"/api/history/{run_id}")
    assert history.status_code == 200, history.text
    assert history.json()["snapshot"]["phase"] == "COMPLETED"
    assert history.json()["snapshot"]["task_label"] == "Repair this project"
    assert history.json()["snapshot"]["limits"] == {
        "max_model_turns": 8,
        "max_calls_per_turn": 8,
        "max_total_tool_calls": 40,
    }
    assert "PRIVATE_RAW_TRACE" not in history.text
    assert "PRIVATE_MUTATION" not in history.text
    assert history.json()["run"]["final_text"] == "Repository task complete"
    live_command_entries = [
        entry
        for entry in live_state["snapshot"]["timeline"]
        if entry["activity_state"] in {"started", "finished"} and "run_command" in entry["headline"]
    ]
    history_command_entries = [
        entry
        for entry in history.json()["snapshot"]["timeline"]
        if entry["activity_state"] in {"started", "finished"}
        and "run_command" in entry["headline"]
        and entry["category"] == "TOOL"
    ]
    assert history_command_entries == live_command_entries
    assert len(history_command_entries) == 2
    assert history_command_entries[0]["activity_id"] == history_command_entries[1]["activity_id"]
    assert history_command_entries[0]["activity_id"].startswith("act_")
    assert history_command_entries[1]["facts"][0] == {
        "label": "Command",
        "value": "python -m pytest -q",
        "format": "pre",
    }
    assert history_command_entries[1]["facts_complete"] is True
    latest_change = history.json()["snapshot"]["latest_change"]
    assert len(latest_change["preview"]) == 6
    assert len(latest_change["expanded_preview"]) == 33
    assert latest_change["expanded_preview"][-1] == "+visible line 29"
    assert latest_change["expanded_preview_complete"] is True
    assert all("expanded_preview" not in entry for entry in history.json()["snapshot"]["timeline"])
    assert latest_change["activity_id"] is None
    assert latest_change["activity_state"] == "finished"
    assert latest_change["facts"] == []

    restarted_history = _client(_workbench(tmp_path)).get(f"/api/history/{run_id}")
    assert restarted_history.status_code == 200, restarted_history.text
    assert restarted_history.json()["run"]["final_text"] == "Repository task complete"
    assert restarted_history.json()["snapshot"]["latest_change"] == latest_change
    assert (
        restarted_history.json()["snapshot"]["timeline"] == history.json()["snapshot"]["timeline"]
    )
    assert "PRIVATE_RAW_TRACE" not in restarted_history.text
    assert "PRIVATE COMMAND" not in restarted_history.text
    assert "provider-command-call" not in restarted_history.text

    selected_second = client.post(
        "/api/projects",
        json={"root": str(second)},
        headers=_headers(token),
    )
    assert selected_second.status_code == 201
    assert client.get("/api/state").json()["status"] == "idle"


def test_history_final_reply_is_bounded_and_workspace_bound(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    foreign_workspace = tmp_path / "foreign"
    workspace.mkdir()
    foreign_workspace.mkdir()
    paths = StatePaths((tmp_path / "state").resolve())
    client = _client(_workbench(tmp_path))
    token = _token(client)
    project = client.post(
        "/api/projects",
        json={"root": str(workspace)},
        headers=_headers(token),
    ).json()
    run_id = "bounded-final-run"
    RunCatalog(paths.runs).create(
        run_id=run_id,
        project_id=project["project_id"],
        workspace_fingerprint=workspace_fingerprint(workspace),
        task_title="Bound final history",
    )
    terminal = {
        "verified": True,
        "status": "verified",
        "epoch": 0,
        "invalidation_count": 0,
        "evidence_labels": [],
        "evidence": [],
    }
    trace = TraceStore(paths.traces)
    trace.append(RunEvent(run_id=run_id, kind=EventKind.RUN_STARTED, message="started"))
    trace.append(
        RunEvent(
            run_id=run_id,
            kind=EventKind.VERIFICATION_EVALUATED,
            message="verified",
            step=1,
            data=terminal,
        )
    )
    trace.append(
        RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_FINISHED,
            message="finished",
            step=1,
            data=terminal,
        )
    )

    final_text = "前言\r\n\u202e" + ("中" * 20_000)
    messages = (
        ChatMessage(role=MessageRole.SYSTEM, content="system"),
        ChatMessage(role=MessageRole.USER, content="Bound final history"),
        ChatMessage(role=MessageRole.ASSISTANT, content=final_text),
    )
    bound_store = SessionStore(paths.sessions, workspace_root=workspace)
    bound_store.save(
        SessionCheckpoint(
            run_id=run_id,
            workspace_fingerprint=bound_store.workspace_fingerprint,
            task="Bound final history",
            system_prompt="system",
            messages=messages,
            completed_steps=1,
            completed_tool_calls=0,
            stop_boundary=SessionBoundary.TERMINAL,
            stop_reason=StopReason.FINAL_RESPONSE,
        )
    )

    history = client.get(f"/api/history/{run_id}")
    exposed = history.json()["run"]["final_text"]
    assert history.status_code == 200
    assert isinstance(exposed, str)
    assert len(exposed) <= 16_000
    assert exposed.startswith("前言\n\N{REPLACEMENT CHARACTER}")
    assert exposed.endswith("[最终回复过长，历史回放已截断]")
    assert "\u202e" not in exposed
    assert (
        client.get(f"/api/projects/{project['project_id']}/runs").json()["runs"][0]["final_text"]
        is None
    )

    bound_store.save(
        SessionCheckpoint(
            run_id=run_id,
            workspace_fingerprint=bound_store.workspace_fingerprint,
            task="Bound final history",
            system_prompt="system",
            messages=messages[:2],
            completed_steps=0,
            completed_tool_calls=0,
            stop_boundary=SessionBoundary.READY_FOR_MODEL,
        )
    )
    assert client.get(f"/api/history/{run_id}").json()["run"]["final_text"] is None

    foreign_final = "FOREIGN_PRIVATE_FINAL"
    SessionStore(paths.sessions).save(
        SessionCheckpoint(
            run_id=run_id,
            workspace_fingerprint=workspace_fingerprint(foreign_workspace),
            task="Bound final history",
            system_prompt="system",
            messages=(
                *messages[:2],
                ChatMessage(role=MessageRole.ASSISTANT, content=foreign_final),
            ),
            completed_steps=1,
            completed_tool_calls=0,
            stop_boundary=SessionBoundary.TERMINAL,
            stop_reason=StopReason.FINAL_RESPONSE,
        )
    )
    mismatched = client.get(f"/api/history/{run_id}")
    assert mismatched.status_code == 200
    assert mismatched.json()["run"]["final_text"] is None
    assert foreign_final not in mismatched.text


def test_noncurrent_nonterminal_trace_is_reported_as_interrupted(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    paths = StatePaths((tmp_path / "state").resolve())
    workbench = _workbench(tmp_path)
    client = _client(workbench)
    token = _token(client)
    project = client.post(
        "/api/projects",
        json={"root": str(workspace)},
        headers=_headers(token),
    ).json()
    run_id = "stale-running-run"
    RunCatalog(paths.runs).create(
        run_id=run_id,
        project_id=project["project_id"],
        workspace_fingerprint=workspace_fingerprint(workspace),
        task_title="Interrupted by a previous process",
    )
    TraceStore(paths.traces).append(
        RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_STARTED,
            message="Previous process started",
        )
    )

    runs = client.get(f"/api/projects/{project['project_id']}/runs")

    assert runs.status_code == 200, runs.text
    assert runs.json()["runs"][0]["status"] == "interrupted"


def test_pretrace_host_failure_is_safely_persisted_and_projected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def fail_before_first_event(*args: object, **kwargs: object) -> RepositoryRunSpec:
        del args, kwargs
        raise RuntimeError("PRIVATE_PROVIDER_OR_FILESYSTEM_DETAIL")

    monkeypatch.setattr(
        "coding_agent.web.workbench.RepositoryRunSpec",
        fail_before_first_event,
    )
    workbench = _workbench(tmp_path)
    client = _client(workbench)
    token = _token(client)
    project = client.post(
        "/api/projects",
        json={"root": str(workspace)},
        headers=_headers(token),
    ).json()
    started = client.post(
        "/api/runs",
        json={"task": "Trigger a host setup failure"},
        headers=_headers(token),
    )
    assert started.status_code == 202, started.text
    run_id = started.json()["run_id"]
    assert workbench.wait(timeout=3)

    state = client.get("/api/state")
    events = TraceStore((tmp_path / "state" / "traces").resolve()).read(run_id)
    history = client.get(f"/api/history/{run_id}")
    runs = client.get(f"/api/projects/{project['project_id']}/runs")

    assert state.json()["status"] == "failed"
    assert state.json()["snapshot"]["phase"] == "FAILED"
    assert [event.kind for event in events] == [EventKind.RUN_STARTED, EventKind.RUN_FAILED]
    assert events[-1].data == {"stop_reason": "runtime_setup_failed"}
    assert history.status_code == 200, history.text
    assert history.json()["snapshot"]["phase"] == "FAILED"
    assert history.json()["snapshot"]["limits"] is None
    assert runs.json()["runs"][0]["status"] == "failed"
    assert "PRIVATE_PROVIDER_OR_FILESYSTEM_DETAIL" not in state.text
    assert "PRIVATE_PROVIDER_OR_FILESYSTEM_DETAIL" not in history.text
    assert "PRIVATE_PROVIDER_OR_FILESYSTEM_DETAIL" not in "".join(
        event.model_dump_json() for event in events
    )


def test_cli_catalog_binding_is_idempotent_for_resume(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    paths = StatePaths((tmp_path / "state").resolve())

    _ensure_run_catalog_entry(
        paths=paths,
        root=workspace.resolve(),
        run_id="run-1",
        task="A task with\nline breaks",
    )
    _ensure_run_catalog_entry(
        paths=paths,
        root=workspace.resolve(),
        run_id="run-1",
        task="A task with\nline breaks",
    )

    records = RunCatalog(paths.runs).list()
    assert len(records) == 1
    assert records[0].task_title == "A task with line breaks"


def test_cli_catalog_binding_handles_a_title_boundary_space(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    paths = StatePaths((tmp_path / "state").resolve())
    boundary_task = f"{'x' * (MAX_TASK_TITLE_LENGTH - 1)} trailing task detail"
    _ensure_run_catalog_entry(
        paths=paths,
        root=workspace.resolve(),
        run_id="run-1",
        task=boundary_task,
    )

    record = RunCatalog(paths.runs).get("run-1")
    assert record.task_title == "x" * (MAX_TASK_TITLE_LENGTH - 1)


def test_long_task_whose_title_cut_lands_on_space_starts_normally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    paths = StatePaths((tmp_path / "state").resolve())
    boundary_task = f"{'x' * (MAX_TASK_TITLE_LENGTH - 1)} trailing task detail"
    received_tasks: list[str] = []

    def fake_execute(
        spec: RepositoryRunSpec,
        *,
        event_sink: EventSink,
        approver: object,
        cancellation_token: CancellationToken,
    ) -> AgentResult:
        del event_sink, approver, cancellation_token
        received_tasks.append(spec.task)
        return _completed_result(spec.run_id, spec.task)

    monkeypatch.setattr("coding_agent.web.workbench.execute_repository_run", fake_execute)
    workbench = _workbench(tmp_path)
    client = _client(workbench)
    token = _token(client)
    project = client.post(
        "/api/projects",
        json={"root": str(workspace)},
        headers=_headers(token),
    ).json()

    started = client.post(
        "/api/runs",
        json={"task": boundary_task},
        headers=_headers(token),
    )

    assert started.status_code == 202, started.text
    assert workbench.wait(timeout=3)
    assert client.get("/api/state").json()["status"] == "completed"
    assert received_tasks == [boundary_task]
    records = RunCatalog(paths.runs).list(project_id=project["project_id"])
    assert len(records) == 1
    assert records[0].task_title == "x" * (MAX_TASK_TITLE_LENGTH - 1)


def test_project_history_distinguishes_completed_without_verification(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    paths = StatePaths((tmp_path / "state").resolve())
    workbench = _workbench(tmp_path)
    client = _client(workbench)
    token = _token(client)
    project = client.post(
        "/api/projects",
        json={"root": str(workspace)},
        headers=_headers(token),
    ).json()
    run_id = "unverified-1"
    RunCatalog(paths.runs).create(
        run_id=run_id,
        project_id=project["project_id"],
        workspace_fingerprint=workspace_fingerprint(workspace),
        task_title="Inspect only",
    )
    trace = TraceStore(paths.traces)
    trace.append(
        RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_STARTED,
            message="started",
        )
    )
    report = {
        "verified": False,
        "status": "missing",
        "epoch": 0,
        "invalidation_count": 0,
        "evidence_labels": [],
        "evidence": [],
    }
    trace.append(
        RunEvent(
            run_id=run_id,
            kind=EventKind.VERIFICATION_EVALUATED,
            message="missing verification",
            step=1,
            data=report,
        )
    )
    trace.append(
        RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_FINISHED,
            message="finished",
            step=1,
            data=report,
        )
    )

    runs = client.get(f"/api/projects/{project['project_id']}/runs").json()["runs"]
    history_without_checkpoint = client.get(f"/api/history/{run_id}")

    assert runs[0]["status"] == "completed_unverified"
    assert history_without_checkpoint.status_code == 200
    assert history_without_checkpoint.json()["run"]["status"] == "completed_unverified"
    assert history_without_checkpoint.json()["run"]["final_text"] is None
    assert history_without_checkpoint.json()["snapshot"]["phase"] == "COMPLETED"

    paths.sessions.mkdir(parents=True)
    (paths.sessions / f"{run_id}.json").write_text(
        '{"private":"CORRUPT_PRIVATE_CHECKPOINT"}',
        encoding="utf-8",
    )
    history_with_corrupt_checkpoint = client.get(f"/api/history/{run_id}")
    assert history_with_corrupt_checkpoint.status_code == 200
    assert history_with_corrupt_checkpoint.json()["run"]["final_text"] is None
    assert "CORRUPT_PRIVATE_CHECKPOINT" not in history_with_corrupt_checkpoint.text


def test_cli_catalog_failure_warns_without_revoking_explicit_root_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    def fail_registration(self: object, root: Path) -> object:
        del self, root
        raise OSError("PRIVATE_PATH_OR_OS_DETAIL")

    monkeypatch.setattr("coding_agent.cli.ProjectRegistry.register", fail_registration)

    _ensure_run_catalog_entry(
        paths=StatePaths((tmp_path / "state").resolve()),
        root=workspace.resolve(),
        run_id="run-1",
        task="The explicit CLI task must still run",
    )

    warning = capsys.readouterr().err
    assert "Agent run will continue" in warning
    assert "PRIVATE_PATH_OR_OS_DETAIL" not in warning
