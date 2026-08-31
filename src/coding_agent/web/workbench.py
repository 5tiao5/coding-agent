"""Project-aware host layer for the optional local Web frontend.

The browser chooses only a previously registered project ID.  Absolute paths remain
server-owned after registration, and every background run captures an immutable
``ProjectContext`` before the shared Agent application layer is entered.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import TypedDict

from coding_agent.application import RepositoryRunSpec, execute_repository_run
from coding_agent.cancellation import CancellationSource, CancellationToken
from coding_agent.command import CommandPermissionMode
from coding_agent.dashboard import DashboardProjection
from coding_agent.events import EventKind, EventSink, RunEvent
from coding_agent.lease import RunLease
from coding_agent.models import AgentResult
from coding_agent.openai_model import ReasoningEffort
from coding_agent.projects import ProjectRecord, ProjectRegistry, ProjectRegistryError
from coding_agent.run_catalog import (
    RunCatalog,
    RunCatalogError,
    RunRecord,
    normalize_task_title,
)
from coding_agent.session import workspace_fingerprint
from coding_agent.state import StatePaths
from coding_agent.trace import TraceError, TraceNotFoundError, TraceRunStatus, TraceStore
from coding_agent.web.service import (
    SnapshotPayload,
    WebRunService,
    WebRunStatePayload,
    _snapshot_payload,
)

_HISTORY_TIMELINE_LIMIT = 100


class ProjectPayload(TypedDict):
    project_id: str
    display_name: str
    root: str
    created_at: str
    last_opened_at: str


class ProjectListPayload(TypedDict):
    active_project_id: str | None
    projects: list[ProjectPayload]


class RunCatalogPayload(TypedDict):
    run_id: str
    project_id: str
    task: str
    status: str
    created_at: str
    completed_at: str | None
    final_text: None
    error: None


class ProjectRunsPayload(TypedDict):
    project_id: str
    runs: list[RunCatalogPayload]


class HistoryPayload(TypedDict):
    run: RunCatalogPayload
    snapshot: SnapshotPayload


class WorkbenchError(RuntimeError):
    """Base class for sanitized host-layer failures."""


class WorkbenchBusyError(WorkbenchError):
    """A project mutation was attempted while the sole worker was occupied."""


class NoActiveProjectError(WorkbenchError):
    """A run was requested before the user selected a registered project."""


class WorkbenchInputError(WorkbenchError):
    """A user-provided project path or identifier is invalid."""


class WorkbenchNotFoundError(WorkbenchError):
    """A project or history record is not available."""


@dataclass(frozen=True, slots=True)
class ProjectContext:
    """Immutable project facts captured for exactly one Agent run."""

    project_id: str
    display_name: str
    root: Path
    workspace_fingerprint: str


@dataclass(frozen=True, slots=True)
class WebWorkbenchConfig:
    """Server-owned settings shared by every project opened in this process."""

    model_name: str
    base_url: str | None
    reasoning_effort: ReasoningEffort | None
    permission_mode: CommandPermissionMode
    paths: StatePaths
    max_steps: int
    model_timeout: float


class WebWorkbench:
    """Own one active project and one replaceable, globally serial run worker."""

    def __init__(
        self,
        config: WebWorkbenchConfig,
        *,
        project_registry: ProjectRegistry | None = None,
        run_catalog: RunCatalog | None = None,
        trace_store: TraceStore | None = None,
    ) -> None:
        self._config = config
        self._registry = project_registry or ProjectRegistry(config.paths.projects_file)
        self._catalog = run_catalog or RunCatalog(config.paths.runs)
        self._traces = trace_store or TraceStore(config.paths.traces)
        self._lock = RLock()
        self._active: ProjectContext | None = None
        self._service = self._new_service()
        self._closed = False
        self._navigation_reserved = False

    @property
    def workspace_label(self) -> str:
        with self._lock:
            return self._active.display_name if self._active is not None else "请选择项目"

    def projects(self) -> ProjectListPayload:
        with self._lock:
            active_id = self._active.project_id if self._active is not None else None
        try:
            records = self._registry.list()
        except ProjectRegistryError as exc:
            raise WorkbenchError("项目列表暂时不可用") from exc
        return {
            "active_project_id": active_id,
            "projects": [_project_payload(record) for record in records],
        }

    def register_project(
        self,
        *,
        root: str | Path,
        display_name: str | None = None,
        create: bool = False,
    ) -> ProjectPayload:
        """Register one existing root or create exactly its absent final directory."""

        with self._lock:
            self._require_project_mutation_allowed()
            try:
                requested = Path(root).expanduser()
                if not requested.is_absolute():
                    raise WorkbenchInputError("项目根目录必须是绝对路径")
                if create:
                    record = self._registry.create(
                        requested.parent,
                        requested.name,
                        display_name=display_name,
                    )
                else:
                    record = self._registry.register(
                        requested,
                        display_name=display_name,
                    )
            except WorkbenchInputError:
                raise
            except (ProjectRegistryError, OSError, ValueError) as exc:
                raise _project_error(exc) from exc
            self._activate_locked(record)
            return _project_payload(record)

    def select_project(self, project_id: str) -> ProjectPayload:
        """Select one registered root after the current worker has fully stopped."""

        with self._lock:
            self._require_project_mutation_allowed()
            try:
                known = self._registry.get(project_id)
                if workspace_fingerprint(known.resolved_root) != known.workspace_fingerprint:
                    raise WorkbenchInputError("项目目录身份已经变化，请重新登记后再打开")
                record = self._registry.mark_opened(project_id)
            except WorkbenchInputError:
                raise
            except (ProjectRegistryError, OSError, ValueError) as exc:
                raise _project_error(exc) from exc
            self._activate_locked(record)
            return _project_payload(record)

    def project_runs(self, project_id: str) -> ProjectRunsPayload:
        try:
            project = self._registry.get(project_id)
            records = self._catalog.list(
                project_id=project_id,
                workspace_fingerprint=project.workspace_fingerprint,
                limit=100,
            )
        except ProjectRegistryError as exc:
            raise _project_error(exc) from exc
        except (RunCatalogError, ValueError) as exc:
            raise _run_error(exc) from exc
        return {
            "project_id": project_id,
            "runs": [self._run_payload(record) for record in records],
        }

    def history(self, run_id: str) -> HistoryPayload:
        """Replay one validated trace exclusively through the dashboard whitelist."""

        try:
            record = self._catalog.get(run_id)
        except (RunCatalogError, ValueError) as exc:
            raise _run_error(exc) from exc
        try:
            project = self._registry.get(record.project_id)
        except ProjectRegistryError as exc:
            raise _project_error(exc) from exc
        if record.workspace_fingerprint != project.workspace_fingerprint:
            raise WorkbenchNotFoundError("该任务属于项目目录的旧身份，不能作为当前历史回放")
        try:
            events = self._traces.read(run_id)
        except TraceNotFoundError as exc:
            raise WorkbenchNotFoundError("该任务尚无可回放的运行轨迹") from exc
        except (TraceError, OSError, ValueError) as exc:
            raise WorkbenchError("该任务的运行轨迹暂时不可用") from exc

        segment = _latest_trace_segment(events)
        projection = DashboardProjection(
            task_label=record.task_title,
            max_timeline=_HISTORY_TIMELINE_LIMIT,
        )
        for event in segment:
            projection.apply(event)
        return {
            "run": self._run_payload(record),
            "snapshot": _snapshot_payload(projection.snapshot),
        }

    def start(self, task: str) -> WebRunStatePayload:
        with self._lock:
            if self._closed:
                raise WorkbenchError("本地项目服务正在关闭")
            if self._navigation_reserved:
                raise WorkbenchBusyError("正在选择项目目录，请稍后再启动任务")
            if self._active is None:
                raise NoActiveProjectError("请先选择一个项目")
            _require_context_identity(self._active)
            return self._service.start(task)

    @contextmanager
    def reserve_navigation(self) -> Iterator[None]:
        """Atomically reserve project navigation while a native dialog is open."""

        with self._lock:
            self._require_project_mutation_allowed()
            self._navigation_reserved = True
        try:
            yield
        finally:
            with self._lock:
                self._navigation_reserved = False

    def state(self) -> WebRunStatePayload:
        with self._lock:
            return self._service.state()

    def wait(self, timeout: float | None = None) -> bool:
        with self._lock:
            service = self._service
        return service.wait(timeout)

    def shutdown(self, timeout: float | None = None) -> bool:
        with self._lock:
            self._closed = True
            service = self._service
        return service.shutdown(timeout)

    def _new_service(self) -> WebRunService:
        return WebRunService(
            self._run_repository,
            cancellation_source=CancellationSource(),
        )

    def _activate_locked(self, record: ProjectRecord) -> None:
        if self._active is not None and self._active.project_id == record.project_id:
            self._active = _project_context(record)
            return
        previous = self._service
        if not previous.shutdown(timeout=0):
            raise WorkbenchBusyError("任务运行期间不能切换项目")
        self._active = _project_context(record)
        self._service = self._new_service()

    def _require_project_mutation_allowed(self) -> None:
        if self._closed:
            raise WorkbenchBusyError("本地项目服务正在关闭")
        if self._navigation_reserved:
            raise WorkbenchBusyError("正在选择项目目录，请稍后再试")
        state = self._service.state()
        if state["status"] == "running" or not self._service.wait(timeout=0):
            raise WorkbenchBusyError("任务运行期间不能登记或切换项目")

    def _run_repository(
        self,
        run_id: str,
        task: str,
        event_sink: EventSink,
        cancellation_token: CancellationToken,
    ) -> AgentResult:
        with self._lock:
            if self._active is None:  # Selection cannot change while status is running.
                raise NoActiveProjectError("请先选择一个项目")
            context = self._active
            _require_context_identity(context)

        self._catalog.create(
            run_id=run_id,
            project_id=context.project_id,
            workspace_fingerprint=context.workspace_fingerprint,
            task_title=normalize_task_title(task),
        )
        config = self._config
        try:
            spec = RepositoryRunSpec(
                run_id=run_id,
                task=task,
                root=context.root,
                model_name=config.model_name,
                base_url=config.base_url,
                reasoning_effort=config.reasoning_effort,
                permission_mode=config.permission_mode,
                paths=config.paths,
                max_steps=config.max_steps,
                model_timeout=config.model_timeout,
            )
            with RunLease(config.paths.root / "leases", run_id):
                # Safe Web mode remains fail-closed until an approval broker is added.
                return execute_repository_run(
                    spec,
                    event_sink=event_sink,
                    approver=None,
                    cancellation_token=cancellation_token,
                )
        except BaseException:
            self._record_host_failure(run_id, event_sink)
            raise

    def _record_host_failure(self, run_id: str, event_sink: EventSink) -> None:
        """Best-effort terminate a cataloged run without serializing its exception."""

        try:
            events = self._traces.read(run_id)
        except TraceNotFoundError:
            events = ()
        except (TraceError, OSError, ValueError):
            return

        latest = _latest_trace_segment(events)
        if any(event.kind in {EventKind.RUN_FINISHED, EventKind.RUN_FAILED} for event in latest):
            return

        synthetic: list[RunEvent] = []
        if not events:
            synthetic.append(
                RunEvent(
                    run_id=run_id,
                    kind=EventKind.RUN_STARTED,
                    message="Workspace run started",
                )
            )
        synthetic.append(
            RunEvent(
                run_id=run_id,
                kind=EventKind.RUN_FAILED,
                message="Workspace run failed before normal completion",
                step=max((event.step for event in latest), default=0),
                data={"stop_reason": "runtime_setup_failed"},
            )
        )
        for event in synthetic:
            with suppress(TraceError, OSError, ValueError):
                self._traces.append(event)
            with suppress(Exception):
                event_sink.emit(event)

    def _run_payload(self, record: RunRecord) -> RunCatalogPayload:
        status = "pending"
        completed_at: str | None = None
        try:
            summary = self._traces.summarize(record.run_id)
        except TraceNotFoundError:
            pass
        except TraceError:
            status = "unavailable"
        else:
            status = summary.status.value
            if summary.status is TraceRunStatus.RUNNING:
                live = self._service.state()
                if live["status"] != "running" or live["run_id"] != record.run_id:
                    status = "interrupted"
            if summary.status is TraceRunStatus.COMPLETED and summary.verified is not True:
                status = "completed_unverified"
            if summary.status in {TraceRunStatus.COMPLETED, TraceRunStatus.FAILED}:
                completed_at = summary.last_event_at.isoformat()
        return {
            "run_id": record.run_id,
            "project_id": record.project_id,
            "task": record.task_title,
            "status": status,
            "created_at": record.created_at.isoformat(),
            "completed_at": completed_at,
            # Canonical messages are deliberately absent from JSONL traces.  History
            # exposes only projected audit facts, never checkpoint conversation data.
            "final_text": None,
            "error": None,
        }


def _project_context(record: ProjectRecord) -> ProjectContext:
    return ProjectContext(
        project_id=record.project_id,
        display_name=record.display_name,
        root=record.resolved_root,
        workspace_fingerprint=record.workspace_fingerprint,
    )


def _project_payload(record: ProjectRecord) -> ProjectPayload:
    return {
        "project_id": record.project_id,
        "display_name": record.display_name,
        "root": str(record.resolved_root),
        "created_at": record.created_at.isoformat(),
        "last_opened_at": record.last_opened_at.isoformat(),
    }


def _require_context_identity(context: ProjectContext) -> None:
    try:
        current = workspace_fingerprint(context.root)
    except ValueError as exc:
        raise WorkbenchInputError("项目目录已经不可访问，请重新登记") from exc
    if current != context.workspace_fingerprint:
        raise WorkbenchInputError("项目目录身份已经变化，请重新登记后再运行")


def _latest_trace_segment(events: tuple[RunEvent, ...]) -> tuple[RunEvent, ...]:
    starts = [
        index
        for index, event in enumerate(events)
        if event.kind in {EventKind.RUN_STARTED, EventKind.RUN_RESUMED}
    ]
    return events[starts[-1] :] if starts else events


def _project_error(exc: Exception) -> WorkbenchError:
    if isinstance(exc, ProjectRegistryError):
        if exc.code in {"project_not_found", "invalid_project_id"}:
            return WorkbenchNotFoundError("项目不存在或未登记")
        if exc.code in {
            "project_parent_invalid",
            "project_root_invalid",
            "project_state_overlap",
            "project_exists",
            "invalid_directory_name",
            "project_create_failed",
        }:
            return WorkbenchInputError(exc.message)
    if isinstance(exc, (OSError, ValueError)):
        return WorkbenchInputError("项目目录无效或不可访问")
    return WorkbenchError("项目注册表暂时不可用")


def _run_error(exc: Exception) -> WorkbenchError:
    if isinstance(exc, RunCatalogError) and exc.code in {
        "run_not_found",
        "invalid_run_id",
        "invalid_project_id",
    }:
        return WorkbenchNotFoundError("任务历史不存在")
    if isinstance(exc, ValueError):
        return WorkbenchInputError("任务或项目标识无效")
    return WorkbenchError("任务历史暂时不可用")


__all__ = [
    "HistoryPayload",
    "NoActiveProjectError",
    "ProjectContext",
    "ProjectListPayload",
    "ProjectPayload",
    "ProjectRunsPayload",
    "WebWorkbench",
    "WebWorkbenchConfig",
    "WorkbenchBusyError",
    "WorkbenchError",
    "WorkbenchInputError",
    "WorkbenchNotFoundError",
]
