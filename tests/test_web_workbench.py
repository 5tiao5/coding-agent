"""Project workbench API tests with a deterministic repository runner."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread

import pytest
from fastapi.testclient import TestClient

from coding_agent.agent_protocol import EARLY_FINAL_CORRECTION
from coding_agent.application import RepositoryRunSpec
from coding_agent.cancellation import CancellationToken
from coding_agent.cli import _ensure_run_catalog_entry
from coding_agent.command import CommandPermissionMode
from coding_agent.events import EventKind, EventSink, RunEvent
from coding_agent.lease import RunLease
from coding_agent.models import (
    AgentResult,
    AgentState,
    ChatMessage,
    MessageRole,
    StopReason,
)
from coding_agent.project_memory import ProjectMemoryStore
from coding_agent.run_catalog import MAX_TASK_TITLE_LENGTH, RunCatalog
from coding_agent.run_memory import RunMemorySnapshot
from coding_agent.session import (
    LoadedSession,
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


def _save_ready_checkpoint(
    *,
    paths: StatePaths,
    workspace: Path,
    run_id: str,
    task: str,
    completed_steps: int = 0,
) -> None:
    store = SessionStore(paths.sessions, workspace_root=workspace)
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="system"),
        ChatMessage(role=MessageRole.USER, content=task),
    ]
    for step in range(completed_steps):
        messages.extend(
            (
                ChatMessage(role=MessageRole.ASSISTANT, content=f"premature final {step}"),
                ChatMessage(role=MessageRole.USER, content=EARLY_FINAL_CORRECTION),
            )
        )
    store.save(
        SessionCheckpoint(
            run_id=run_id,
            workspace_fingerprint=store.workspace_fingerprint,
            task=task,
            system_prompt="system",
            messages=tuple(messages),
            completed_steps=completed_steps,
            completed_tool_calls=0,
            stop_boundary=SessionBoundary.READY_FOR_MODEL,
        )
    )


def _save_interrupted_run(
    *,
    paths: StatePaths,
    workspace: Path,
    project_id: str,
    run_id: str,
    task: str,
    completed_steps: int = 0,
) -> None:
    RunCatalog(paths.runs).create(
        run_id=run_id,
        project_id=project_id,
        workspace_fingerprint=workspace_fingerprint(workspace),
        task_title=task,
    )
    _save_ready_checkpoint(
        paths=paths,
        workspace=workspace,
        run_id=run_id,
        task=task,
        completed_steps=completed_steps,
    )
    trace = TraceStore(paths.traces)
    trace.append(RunEvent(run_id=run_id, kind=EventKind.RUN_STARTED, message="started"))
    trace.append(
        RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_FAILED,
            message="interrupted before completion",
            step=1,
            data={"stop_reason": StopReason.USER_INTERRUPTED.value},
        )
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


def test_project_removal_hides_sidebar_entry_preserves_disk_and_restores_history(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    sentinel = first / "keep.txt"
    sentinel.write_text("do not delete", encoding="utf-8")
    paths = StatePaths((tmp_path / "state").resolve())
    workbench = _workbench(tmp_path)
    client = _client(workbench)
    token = _token(client)
    first_record = client.post(
        "/api/projects",
        json={"root": str(first), "display_name": "待移除项目"},
        headers=_headers(token),
    ).json()
    RunCatalog(paths.runs).create(
        run_id="retained-run",
        project_id=first_record["project_id"],
        workspace_fingerprint=workspace_fingerprint(first),
        task_title="保留的历史任务",
    )
    second_record = client.post(
        "/api/projects",
        json={"root": str(second)},
        headers=_headers(token),
    ).json()

    endpoint = f"/api/projects/{first_record['project_id']}"
    assert client.delete(endpoint).status_code == 403
    assert (
        client.delete(
            endpoint,
            headers=_headers(token, origin="https://attacker.example"),
        ).status_code
        == 403
    )
    removed = client.delete(endpoint, headers=_headers(token))

    assert removed.status_code == 200
    assert removed.json() == {
        "project_id": first_record["project_id"],
        "removed_from_sidebar": True,
        "workspace_deleted": False,
        "history_preserved": True,
    }
    listed = client.get("/api/projects").json()
    assert listed["active_project_id"] == second_record["project_id"]
    assert [project["project_id"] for project in listed["projects"]] == [
        second_record["project_id"]
    ]
    assert sentinel.read_text(encoding="utf-8") == "do not delete"
    assert client.get(f"{endpoint}/runs").status_code == 404
    assert client.delete(endpoint, headers=_headers(token)).status_code == 404

    reopened = client.post(
        "/api/projects",
        json={"root": str(first)},
        headers=_headers(token),
    ).json()

    assert reopened["project_id"] == first_record["project_id"]
    runs = client.get(f"{endpoint}/runs").json()["runs"]
    assert [run["run_id"] for run in runs] == ["retained-run"]
    assert runs[0]["task"] == "保留的历史任务"

    removed_active = client.delete(endpoint, headers=_headers(token))

    assert removed_active.status_code == 200
    assert client.get("/api/projects").json()["active_project_id"] is None
    assert client.get("/api/meta").json()["workspace"] == "请选择项目"
    assert client.get("/api/state").json()["status"] == "idle"
    refused = client.post(
        "/api/runs",
        json={"task": "must choose a project"},
        headers=_headers(token),
    )
    assert refused.status_code == 409
    assert refused.json() == {"detail": "请先选择一个项目"}


def test_project_removal_is_rejected_while_a_run_is_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    started = Event()
    release = Event()

    def fake_execute(
        spec: RepositoryRunSpec,
        *,
        event_sink: EventSink,
        approver: object,
        cancellation_token: CancellationToken,
    ) -> AgentResult:
        del event_sink, approver, cancellation_token
        started.set()
        assert release.wait(timeout=2)
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
    assert (
        client.post(
            "/api/runs",
            json={"task": "hold the project open"},
            headers=_headers(token),
        ).status_code
        == 202
    )
    assert started.wait(timeout=1)

    blocked = client.delete(
        f"/api/projects/{project['project_id']}",
        headers=_headers(token),
    )

    assert blocked.status_code == 409
    assert client.get("/api/projects").json()["active_project_id"] == project["project_id"]
    release.set()
    assert workbench.wait(timeout=2)


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
            "parent_run_id": None,
            "project_id": first_record["project_id"],
            "task": "Repair this project",
            "status": "completed",
            "created_at": runs_response.json()["runs"][0]["created_at"],
            "completed_at": runs_response.json()["runs"][0]["completed_at"],
            "final_text": None,
            "error": None,
            "resume_available": False,
            "resume_reason": "任务已经完成",
            "continuation": {
                "kind": "follow_up",
                "available": True,
                "reason": None,
            },
            "memory_context": {
                "requested": True,
                "applied": False,
                "source_run_ids": [],
                "sources": [],
                "error": None,
            },
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


def test_history_replays_workspace_changes_across_all_resume_segments(tmp_path: Path) -> None:
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
    run_id = "multi-segment-change-ledger"
    RunCatalog(paths.runs).create(
        run_id=run_id,
        project_id=project["project_id"],
        workspace_fingerprint=workspace_fingerprint(workspace),
        task_title="Preserve every mutation across resume segments",
    )
    trace = TraceStore(paths.traces)
    events = (
        RunEvent(run_id=run_id, kind=EventKind.RUN_STARTED, message="started"),
        RunEvent(
            run_id=run_id,
            kind=EventKind.TOOL_FINISHED,
            message="first mutation",
            step=1,
            data={
                "tool_name": "replace_text",
                "ok": True,
                "summary": "Updated routeforge/domain.py (+2/-1)",
                "preview": (
                    "Diff preview:\n--- a/routeforge/domain.py\n"
                    "+++ b/routeforge/domain.py\n-old\n+new"
                ),
                "metadata": {
                    "path": "routeforge/domain.py",
                    "changed": True,
                    "added_lines": 2,
                    "removed_lines": 1,
                    "mutation_revision": 1,
                    "change_kind": "update",
                    "diff_complete": True,
                },
            },
        ),
        RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_FAILED,
            message="paused",
            step=1,
            data={"stop_reason": StopReason.USER_INTERRUPTED.value},
        ),
        RunEvent(run_id=run_id, kind=EventKind.RUN_RESUMED, message="resumed", step=1),
        RunEvent(
            run_id=run_id,
            kind=EventKind.TOOL_FINISHED,
            message="second mutation",
            step=2,
            data={
                "tool_name": "write_file",
                "ok": True,
                "summary": "Created routeforge/search.py (+3/-0)",
                "preview": (
                    "Diff preview:\n--- /dev/null\n+++ b/routeforge/search.py\n"
                    "+def search():\n+    return []"
                ),
                "metadata": {
                    "path": "routeforge/search.py",
                    "changed": True,
                    "added_lines": 3,
                    "removed_lines": 0,
                    "mutation_revision": 2,
                    "change_kind": "create",
                    "diff_complete": True,
                },
            },
        ),
        RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_FINISHED,
            message="finished",
            step=2,
            data={"verified": True, "status": "verified"},
        ),
    )
    for event in events:
        trace.append(event)

    history = client.get(f"/api/history/{run_id}")
    assert history.status_code == 200, history.text
    snapshot = history.json()["snapshot"]
    assert [change["detail"] for change in snapshot["workspace_changes"]] == [
        "Updated routeforge/domain.py (+2/-1)",
        "Created routeforge/search.py (+3/-0)",
    ]
    assert snapshot["workspace_changes_complete"] is True
    assert snapshot["omitted_change_count"] == 0
    assert snapshot["latest_change"] == snapshot["workspace_changes"][-1]
    assert snapshot["changed_files"] == [
        {
            "path": "routeforge/domain.py",
            "added_lines": 2,
            "removed_lines": 1,
            "revision": 1,
            "change_kind": "update",
        },
        {
            "path": "routeforge/search.py",
            "added_lines": 3,
            "removed_lines": 0,
            "revision": 2,
            "change_kind": "create",
        },
    ]
    assert all(
        "routeforge/domain.py" not in "\n".join(entry["preview"]) for entry in snapshot["timeline"]
    )

    restarted = _client(_workbench(tmp_path)).get(f"/api/history/{run_id}")
    assert restarted.status_code == 200
    assert restarted.json()["snapshot"]["workspace_changes"] == snapshot["workspace_changes"]
    assert restarted.json()["snapshot"]["changed_files"] == snapshot["changed_files"]


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


def test_web_resume_reuses_run_identity_checkpoint_and_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    run_id = "resume-web-run"
    task = "Continue the interrupted repository task"
    _save_interrupted_run(
        paths=paths,
        workspace=workspace,
        project_id=project["project_id"],
        run_id=run_id,
        task=task,
    )

    before = client.get(f"/api/projects/{project['project_id']}/runs").json()["runs"][0]
    assert before["status"] == "interrupted"
    assert before["completed_at"] is None
    assert before["resume_available"] is True
    assert before["resume_reason"] is None

    started = Event()
    release = Event()
    received: list[tuple[RepositoryRunSpec, LoadedSession, SessionStore]] = []

    def fake_resume(
        spec: RepositoryRunSpec,
        *,
        event_sink: EventSink,
        approver: object,
        session_store: SessionStore,
        loaded: LoadedSession,
        cancellation_token: CancellationToken,
    ) -> AgentResult:
        del approver, cancellation_token
        received.append((spec, loaded, session_store))
        trace = TraceStore(spec.paths.traces)

        def emit(event: RunEvent) -> None:
            trace.append(event)
            event_sink.emit(event)

        emit(
            RunEvent(
                run_id=spec.run_id,
                kind=EventKind.RUN_RESUMED,
                message="resumed",
                step=loaded.checkpoint.completed_steps,
            )
        )
        started.set()
        assert release.wait(timeout=3)
        result = _completed_result(spec.run_id, spec.task)
        _save_terminal_checkpoint(spec, result)
        emit(
            RunEvent(
                run_id=spec.run_id,
                kind=EventKind.VERIFICATION_EVALUATED,
                message="verified",
                step=result.steps,
                data={"verified": True, "status": "verified"},
            )
        )
        emit(
            RunEvent(
                run_id=spec.run_id,
                kind=EventKind.RUN_FINISHED,
                message="finished",
                step=result.steps,
                data={"verified": True, "status": "verified"},
            )
        )
        return result

    monkeypatch.setattr("coding_agent.web.workbench.execute_repository_run", fake_resume)

    forbidden = client.post(f"/api/runs/{run_id}/resume", json={})
    assert forbidden.status_code == 403
    resumed = client.post(
        f"/api/runs/{run_id}/resume",
        json={},
        headers=_headers(token),
    )

    assert resumed.status_code == 202, resumed.text
    assert resumed.json()["run_id"] == run_id
    assert resumed.json()["task"] == task
    assert started.wait(timeout=3)
    during = client.get(f"/api/projects/{project['project_id']}/runs").json()["runs"][0]
    assert during["status"] == "running"
    assert during["resume_available"] is False
    assert during["resume_reason"] == "已有任务正在运行"

    release.set()
    assert workbench.wait(timeout=3)
    assert len(received) == 1
    spec, loaded, store = received[0]
    assert spec.run_id == run_id
    assert spec.task == task
    assert loaded.checkpoint.run_id == run_id
    assert loaded.checkpoint.stop_boundary is SessionBoundary.READY_FOR_MODEL
    assert store.workspace_fingerprint == workspace_fingerprint(workspace)
    assert [record.run_id for record in RunCatalog(paths.runs).list()] == [run_id]
    assert [event.kind for event in TraceStore(paths.traces).read(run_id)] == [
        EventKind.RUN_STARTED,
        EventKind.RUN_FAILED,
        EventKind.RUN_RESUMED,
        EventKind.VERIFICATION_EVALUATED,
        EventKind.RUN_FINISHED,
    ]
    after = client.get(f"/api/history/{run_id}").json()["run"]
    assert after["status"] == "completed"
    assert after["resume_available"] is False
    assert after["resume_reason"] == "任务已经完成"


def test_web_resume_requires_a_larger_model_turn_ceiling_after_budget_exhaustion(
    tmp_path: Path,
) -> None:
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
    run_id = "exhausted-web-run"
    _save_interrupted_run(
        paths=paths,
        workspace=workspace,
        project_id=project["project_id"],
        run_id=run_id,
        task="Continue with a larger cumulative model-turn budget",
        completed_steps=8,
    )

    run = client.get(f"/api/projects/{project['project_id']}/runs").json()["runs"][0]
    assert run["resume_available"] is False
    assert "--max-steps" in run["resume_reason"]

    refused = client.post(
        f"/api/runs/{run_id}/resume",
        json={},
        headers=_headers(token),
    )
    assert refused.status_code == 409
    assert "--max-steps" in refused.json()["detail"]


def test_web_resume_refuses_completed_terminal_and_foreign_project_runs(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    paths = StatePaths((tmp_path / "state").resolve())
    workbench = _workbench(tmp_path)
    client = _client(workbench)
    token = _token(client)
    first_project = client.post(
        "/api/projects",
        json={"root": str(first)},
        headers=_headers(token),
    ).json()
    run_id = "completed-ready-run"
    task = "A stale ready checkpoint must not replay"
    _save_interrupted_run(
        paths=paths,
        workspace=first,
        project_id=first_project["project_id"],
        run_id=run_id,
        task=task,
    )
    TraceStore(paths.traces).append(
        RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_FINISHED,
            message="already completed",
            step=1,
            data={"verified": False, "status": "missing"},
        )
    )

    completed = client.get(f"/api/projects/{first_project['project_id']}/runs").json()["runs"][0]
    assert completed["resume_available"] is False
    assert completed["resume_reason"] == "任务已经完成"
    assert completed["continuation"] == {
        "kind": "none",
        "available": False,
        "reason": "该历史没有可用的项目记忆摘要",
    }
    refused = client.post(
        f"/api/runs/{run_id}/resume",
        json={},
        headers=_headers(token),
    )
    assert refused.status_code == 409
    assert "已经完成" in refused.json()["detail"]
    follow_up_refused = client.post(
        "/api/runs",
        json={"task": "Continue safely", "parent_run_id": run_id},
        headers=_headers(token),
    )
    assert follow_up_refused.status_code == 409
    assert "项目记忆摘要不可用" in follow_up_refused.json()["detail"]

    terminal_run_id = "terminal-checkpoint-run"
    terminal_task = "A terminal checkpoint cannot resume"
    _save_interrupted_run(
        paths=paths,
        workspace=first,
        project_id=first_project["project_id"],
        run_id=terminal_run_id,
        task=terminal_task,
    )
    terminal_store = SessionStore(paths.sessions, workspace_root=first)
    terminal_store.save(
        SessionCheckpoint(
            run_id=terminal_run_id,
            workspace_fingerprint=terminal_store.workspace_fingerprint,
            task=terminal_task,
            system_prompt="system",
            messages=(
                ChatMessage(role=MessageRole.SYSTEM, content="system"),
                ChatMessage(role=MessageRole.USER, content=terminal_task),
                ChatMessage(role=MessageRole.ASSISTANT, content="done"),
            ),
            completed_steps=1,
            completed_tool_calls=0,
            stop_boundary=SessionBoundary.TERMINAL,
            stop_reason=StopReason.FINAL_RESPONSE,
        )
    )
    records = client.get(f"/api/projects/{first_project['project_id']}/runs").json()["runs"]
    terminal_record = next(item for item in records if item["run_id"] == terminal_run_id)
    assert terminal_record["resume_available"] is False
    assert terminal_record["resume_reason"] == "任务已经结束"
    terminal_refused = client.post(
        f"/api/runs/{terminal_run_id}/resume",
        json={},
        headers=_headers(token),
    )
    assert terminal_refused.status_code == 409
    assert "已经结束" in terminal_refused.json()["detail"]

    foreign_run_id = "foreign-project-run"
    _save_interrupted_run(
        paths=paths,
        workspace=first,
        project_id=first_project["project_id"],
        run_id=foreign_run_id,
        task="Continue only from the original project",
    )
    client.post(
        "/api/projects",
        json={"root": str(second)},
        headers=_headers(token),
    )
    foreign = client.post(
        f"/api/runs/{foreign_run_id}/resume",
        json={},
        headers=_headers(token),
    )
    assert foreign.status_code == 404
    assert "不属于当前项目" in foreign.json()["detail"]


def test_web_follow_up_rejects_a_stale_noncompleted_parent_memory(
    tmp_path: Path,
) -> None:
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
    run_id = "stale-memory-parent"
    task = "A stale failed memory must not seed a follow-up"
    _save_interrupted_run(
        paths=paths,
        workspace=workspace,
        project_id=project["project_id"],
        run_id=run_id,
        task=task,
    )
    TraceStore(paths.traces).append(
        RunEvent(
            run_id=run_id,
            kind=EventKind.RUN_FINISHED,
            message="later marked complete",
            step=1,
            data={"verified": True},
        )
    )
    memory = ProjectMemoryStore(
        paths.project_memories,
        project_id=project["project_id"],
        workspace_root=workspace,
        workspace_fingerprint_value=workspace_fingerprint(workspace),
    )
    memory.remember_run(
        run_id=run_id,
        task_goal=task,
        final_status="failed",
        final_summary="旧的失败摘要",
        run_memory=RunMemorySnapshot(revision=0),
    )

    listed = client.get(f"/api/projects/{project['project_id']}/runs").json()["runs"][0]
    assert listed["continuation"] == {
        "kind": "none",
        "available": False,
        "reason": "该历史的项目记忆摘要记录为失败，不能继续对话",
    }
    refused = client.post(
        "/api/runs",
        json={"task": "Continue only with a completed parent", "parent_run_id": run_id},
        headers=_headers(token),
    )
    assert refused.status_code == 409
    assert "项目记忆摘要记录为失败" in refused.json()["detail"]

    memory.remember_run(
        run_id=run_id,
        task_goal=task,
        final_status="completed_unverified",
        final_summary="任务已结束，但仍需要补充验证。",
        run_memory=RunMemorySnapshot(revision=0),
    )
    upgraded = client.get(f"/api/projects/{project['project_id']}/runs").json()["runs"][0]
    assert upgraded["continuation"] == {
        "kind": "follow_up",
        "available": True,
        "reason": None,
    }


def test_web_resume_honors_existing_run_lease_without_mutating_trace(tmp_path: Path) -> None:
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
    run_id = "leased-resume-run"
    _save_interrupted_run(
        paths=paths,
        workspace=workspace,
        project_id=project["project_id"],
        run_id=run_id,
        task="Do not race another process",
    )
    before = TraceStore(paths.traces).read(run_id)

    with RunLease(paths.root / "leases", run_id):
        response = client.post(
            f"/api/runs/{run_id}/resume",
            json={},
            headers=_headers(token),
        )
        assert response.status_code == 202
        assert workbench.wait(timeout=3)

    state = client.get("/api/state")
    assert state.json()["status"] == "failed"
    assert state.json()["error"].endswith("[run_already_active]")
    assert str(paths.root) not in state.text
    assert TraceStore(paths.traces).read(run_id) == before


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


def test_project_memory_is_injected_with_persisted_sources_and_can_be_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    paths = StatePaths((tmp_path / "state").resolve())
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
        started = RunEvent(
            run_id=spec.run_id,
            kind=EventKind.RUN_STARTED,
            message="started",
        )
        trace.append(started)
        event_sink.emit(started)
        result = _completed_result(spec.run_id, spec.task)
        _save_terminal_checkpoint(spec, result)
        finished = RunEvent(
            run_id=spec.run_id,
            kind=EventKind.RUN_FINISHED,
            message="finished",
            step=1,
            data={"verified": True},
        )
        trace.append(finished)
        event_sink.emit(finished)
        return result

    monkeypatch.setattr("coding_agent.web.workbench.execute_repository_run", fake_execute)
    workbench = _workbench(tmp_path)
    client = _client(workbench)
    token = _token(client)
    project = client.post(
        "/api/projects",
        json={"root": str(workspace)},
        headers=_headers(token),
    ).json()

    first = client.post(
        "/api/runs",
        json={"task": "Implement the route-cost calculation", "use_project_memory": True},
        headers=_headers(token),
    )
    assert first.status_code == 202, first.text
    assert first.json()["memory_context"] == {
        "requested": True,
        "applied": False,
        "source_run_ids": [],
        "sources": [],
        "error": None,
    }
    first_run_id = first.json()["run_id"]
    assert workbench.wait(timeout=3)

    second = client.post(
        "/api/runs",
        json={"task": "Extend the route display", "use_project_memory": True},
        headers=_headers(token),
    )
    assert second.status_code == 202, second.text
    second_context = second.json()["memory_context"]
    assert second_context["requested"] is True
    assert second_context["applied"] is True
    assert second_context["source_run_ids"] == [first_run_id]
    assert second_context["sources"][0]["task"] == "Implement the route-cost calculation"
    second_run_id = second.json()["run_id"]
    assert workbench.wait(timeout=3)

    assert captured[0].project_memory_context is None
    assert captured[1].project_memory_context is not None
    assert "Implement the route-cost calculation" in captured[1].project_memory_context
    second_record = RunCatalog(paths.runs).get(second_run_id)
    assert second_record.memory_requested is True
    assert second_record.memory_source_run_ids == (first_run_id,)

    history = client.get(f"/api/history/{second_run_id}")
    assert history.status_code == 200
    assert history.json()["run"]["memory_context"] == second_context

    disabled = client.post(
        "/api/runs",
        json={"task": "Run without historical context", "use_project_memory": False},
        headers=_headers(token),
    )
    assert disabled.status_code == 202, disabled.text
    assert disabled.json()["memory_context"]["requested"] is False
    disabled_run_id = disabled.json()["run_id"]
    assert workbench.wait(timeout=3)
    assert captured[2].project_memory_context is None
    assert RunCatalog(paths.runs).get(disabled_run_id).memory_source_run_ids == ()

    follow_up = client.post(
        "/api/runs",
        json={
            "task": "Continue by explaining and refining the route display",
            "use_project_memory": False,
            "parent_run_id": first_run_id,
        },
        headers=_headers(token),
    )
    assert follow_up.status_code == 202, follow_up.text
    follow_up_run_id = follow_up.json()["run_id"]
    assert follow_up_run_id != first_run_id
    assert follow_up.json()["memory_context"] == {
        "requested": False,
        "applied": True,
        "source_run_ids": [first_run_id],
        "sources": [
            {
                "run_id": first_run_id,
                "task": "Implement the route-cost calculation",
                "completed_at": follow_up.json()["memory_context"]["sources"][0]["completed_at"],
            }
        ],
        "error": None,
    }
    assert workbench.wait(timeout=3)
    assert captured[3].project_memory_context is not None
    assert "Implement the route-cost calculation" in captured[3].project_memory_context
    follow_up_record = RunCatalog(paths.runs).get(follow_up_run_id)
    assert follow_up_record.parent_run_id == first_run_id
    assert follow_up_record.memory_requested is False
    assert follow_up_record.memory_source_run_ids == (first_run_id,)

    original_history = client.get(f"/api/history/{first_run_id}")
    assert original_history.status_code == 200
    assert original_history.json()["run"]["parent_run_id"] is None
    assert original_history.json()["run"]["status"] == "completed_unverified"

    restarted = _client(_workbench(tmp_path))
    restarted_token = _token(restarted)
    assert (
        restarted.post(
            f"/api/projects/{project['project_id']}/select",
            json={},
            headers=_headers(restarted_token),
        ).status_code
        == 200
    )
    replay = restarted.get(f"/api/history/{second_run_id}")
    assert replay.status_code == 200
    assert replay.json()["run"]["memory_context"] == second_context
