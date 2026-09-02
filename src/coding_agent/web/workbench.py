"""Project-aware host layer for the optional local Web frontend.

The browser chooses only a previously registered project ID.  Absolute paths remain
server-owned after registration, and every background run captures an immutable
``ProjectContext`` before the shared Agent application layer is entered.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Literal, TypedDict

from coding_agent.application import RepositoryRunSpec, execute_repository_run
from coding_agent.cancellation import CancellationSource, CancellationToken
from coding_agent.command import CommandPermissionMode
from coding_agent.dashboard import MAX_EXPANDED_MUTATION_PREVIEW_LINES, DashboardProjection
from coding_agent.events import EventKind, EventSink, RunEvent
from coding_agent.lease import RunLease
from coding_agent.models import AgentResult
from coding_agent.openai_model import ReasoningEffort
from coding_agent.project_memory import ProjectMemoryError
from coding_agent.projects import ProjectRecord, ProjectRegistry, ProjectRegistryError
from coding_agent.run_catalog import (
    RunCatalog,
    RunCatalogError,
    RunRecord,
    normalize_task_title,
)
from coding_agent.session import (
    LoadedSession,
    SessionBoundary,
    SessionError,
    SessionStore,
    workspace_fingerprint,
)
from coding_agent.state import StatePaths
from coding_agent.trace import TraceError, TraceNotFoundError, TraceRunStatus, TraceStore
from coding_agent.web.project_memory import (
    ProjectMemoryCoordinator,
    ProjectMemoryIndex,
    ProjectMemoryLaunch,
    ProjectMemoryScope,
)
from coding_agent.web.service import (
    ProjectMemoryContextPayload,
    SnapshotPayload,
    WebRunService,
    WebRunStatePayload,
    _snapshot_payload,
)

_HISTORY_TIMELINE_LIMIT = 100
_HISTORY_FINAL_TEXT_LIMIT = 16_000
_HISTORY_FINAL_TEXT_TRUNCATION_MARKER = "\n\n[最终回复过长，历史回放已截断]"


class ProjectPayload(TypedDict):
    project_id: str
    display_name: str
    root: str
    created_at: str
    last_opened_at: str


class ProjectListPayload(TypedDict):
    active_project_id: str | None
    projects: list[ProjectPayload]


class ProjectRemovalPayload(TypedDict):
    project_id: str
    removed_from_sidebar: Literal[True]
    workspace_deleted: Literal[False]
    history_preserved: Literal[True]


class RunCatalogPayload(TypedDict):
    run_id: str
    parent_run_id: str | None
    project_id: str
    task: str
    status: str
    created_at: str
    completed_at: str | None
    final_text: str | None
    error: None
    resume_available: bool
    resume_reason: str | None
    continuation: RunContinuationPayload
    memory_context: ProjectMemoryContextPayload


class RunContinuationPayload(TypedDict):
    """One explicit UI action; follow-up and checkpoint resume never overlap."""

    kind: Literal["resume", "follow_up", "none"]
    available: bool
    reason: str | None


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


class WorkbenchResumeError(WorkbenchError):
    """An existing run cannot be continued from its persisted state."""


class WorkbenchFollowUpError(WorkbenchError):
    """A terminal run cannot safely seed a new follow-up run."""


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
        self._project_memory = ProjectMemoryCoordinator(config.paths, self._catalog)
        self._lock = RLock()
        self._active: ProjectContext | None = None
        self._service = self._new_service()
        self._closed = False
        self._navigation_reserved = False
        self._pending_memory: dict[str, ProjectMemoryLaunch] = {}
        self._pending_parent_run_ids: dict[str, str] = {}
        self._current_memory: tuple[str, ProjectMemoryContextPayload] | None = None

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

    def remove_project(self, project_id: str) -> ProjectRemovalPayload:
        """Hide one project while preserving its workspace and private run history."""

        with self._lock:
            self._require_project_mutation_allowed()
            try:
                removed = self._registry.remove(project_id)
            except (ProjectRegistryError, OSError, ValueError) as exc:
                raise _project_error(exc) from exc
            if self._active is not None and self._active.project_id == removed.project_id:
                previous = self._service
                if not previous.shutdown(timeout=0):  # Guarded by the admission check above.
                    raise WorkbenchBusyError("任务运行期间不能移除项目")
                self._active = None
                self._service = self._new_service()
                self._pending_memory.clear()
                self._pending_parent_run_ids.clear()
                self._current_memory = None
            return {
                "project_id": removed.project_id,
                "removed_from_sidebar": True,
                "workspace_deleted": False,
                "history_preserved": True,
            }

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
        memory_index = self._project_memory.load_index(_project_memory_scope(project))
        return {
            "project_id": project_id,
            "runs": [
                self._run_payload(
                    record,
                    project=project,
                    memory_index=memory_index,
                )
                for record in records
            ],
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
            expanded_mutation_preview_lines=MAX_EXPANDED_MUTATION_PREVIEW_LINES,
        )
        for event in segment:
            projection.apply(event)
        # Status and the ordinary activity timeline intentionally describe the latest
        # resume segment. File mutations are a run-wide audit ledger, so collect them
        # from every segment that shares this immutable run ID.
        change_projection = DashboardProjection(
            task_label=record.task_title,
            max_timeline=1,
            expanded_mutation_preview_lines=MAX_EXPANDED_MUTATION_PREVIEW_LINES,
        )
        for event in events:
            change_projection.apply(event)
        change_snapshot = change_projection.snapshot
        snapshot = replace(
            projection.snapshot,
            changed_files=change_snapshot.changed_files,
            workspace_changes=change_snapshot.workspace_changes,
            workspace_changes_complete=change_snapshot.workspace_changes_complete,
            omitted_change_count=change_snapshot.omitted_change_count,
            latest_change=change_snapshot.latest_change,
        )
        run = self._run_payload(
            record,
            project=project,
            memory_index=self._project_memory.load_index(_project_memory_scope(project)),
        )
        if run["status"] in {"completed", "completed_unverified"}:
            run["final_text"] = self._history_final_text(
                run_id=record.run_id,
                workspace_root=project.resolved_root,
            )
        return {
            "run": run,
            "snapshot": _snapshot_payload(snapshot),
        }

    def start(
        self,
        task: str,
        *,
        use_project_memory: bool = True,
        parent_run_id: str | None = None,
    ) -> WebRunStatePayload:
        with self._lock:
            if self._closed:
                raise WorkbenchError("本地项目服务正在关闭")
            if self._navigation_reserved:
                raise WorkbenchBusyError("正在选择项目目录，请稍后再启动任务")
            if self._active is None:
                raise NoActiveProjectError("请先选择一个项目")
            context = self._active
            _require_context_identity(context)
            if parent_run_id is not None:
                self._require_follow_up_parent(parent_run_id, context=context)
            try:
                launch = self._project_memory.prepare_launch(
                    _context_memory_scope(context),
                    task,
                    requested=use_project_memory,
                    parent_run_id=parent_run_id,
                )
            except ProjectMemoryError as exc:
                if parent_run_id is not None:
                    raise WorkbenchFollowUpError(
                        "父任务的项目记忆摘要不可用，不能静默丢失上下文"
                    ) from exc
                raise WorkbenchError("项目记忆暂时不可用") from exc
            state = self._service.start(task)
            run_id = state["run_id"]
            if run_id is None:  # Defensive: a successful start always owns an ID.
                raise WorkbenchError("本地 Agent 后台任务未返回任务标识")
            self._pending_memory[run_id] = launch
            if parent_run_id is not None:
                self._pending_parent_run_ids[run_id] = parent_run_id
            self._current_memory = (run_id, launch.payload)
            return self._project_memory.attach_to_state(state, launch.payload)

    def resume(self, run_id: str) -> WebRunStatePayload:
        """Continue one interrupted checkpoint without changing its run identity."""

        with self._lock:
            if self._closed:
                raise WorkbenchError("本地项目服务正在关闭")
            if self._navigation_reserved:
                raise WorkbenchBusyError("正在选择项目目录，请稍后再继续任务")
            if self._active is None:
                raise NoActiveProjectError("请先选择一个项目")
            context = self._active
            _require_context_identity(context)
            record = self._resume_record(run_id, context=context)
            loaded, _ = self._load_resume_candidate(record, context=context)
            try:
                project = self._registry.get(record.project_id)
            except ProjectRegistryError as exc:
                raise _project_error(exc) from exc
            memory_scope = _project_memory_scope(project)
            memory_payload = self._project_memory.payload_for_record(
                record,
                scope=memory_scope,
                index=self._project_memory.load_index(memory_scope),
            )
            state = self._service.resume(run_id, loaded.checkpoint.task)
            self._current_memory = (run_id, memory_payload)
            return self._project_memory.attach_to_state(state, memory_payload)

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
            state = self._service.state()
            current = self._current_memory
            if current is None or current[0] != state["run_id"]:
                return state
            return self._project_memory.attach_to_state(state, current[1])

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
            resume_runner=self._resume_repository,
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
        self._pending_memory.clear()
        self._pending_parent_run_ids.clear()
        self._current_memory = None

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
            launch = self._pending_memory.pop(
                run_id,
                self._project_memory.empty_launch(requested=False),
            )
            parent_run_id = self._pending_parent_run_ids.pop(run_id, None)

        self._catalog.create(
            run_id=run_id,
            project_id=context.project_id,
            workspace_fingerprint=context.workspace_fingerprint,
            task_title=normalize_task_title(task),
            parent_run_id=parent_run_id,
            memory_requested=launch.payload["requested"],
            memory_source_run_ids=tuple(launch.payload["source_run_ids"]),
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
                project_memory_context=launch.context_text,
            )
            with RunLease(config.paths.root / "leases", run_id):
                # Safe Web mode remains fail-closed until an approval broker is added.
                result = execute_repository_run(
                    spec,
                    event_sink=event_sink,
                    approver=None,
                    cancellation_token=cancellation_token,
                )
                self._project_memory.remember_run(
                    _context_memory_scope(context),
                    task=task,
                    result=result,
                )
                return result
        except BaseException:
            self._record_host_failure(run_id, event_sink)
            raise

    def _resume_repository(
        self,
        run_id: str,
        task: str,
        event_sink: EventSink,
        cancellation_token: CancellationToken,
    ) -> AgentResult:
        """Reload and resume under the lease; preflight state is never trusted across threads."""

        del task  # The canonical task comes exclusively from the workspace-bound checkpoint.
        with self._lock:
            if self._active is None:  # Selection cannot change while status is running.
                raise NoActiveProjectError("请先选择一个项目")
            context = self._active
            _require_context_identity(context)
            record = self._resume_record(run_id, context=context)

        config = self._config
        # A competing process must fail without appending a synthetic failure to its trace.
        with RunLease(config.paths.root / "leases", run_id):
            try:
                loaded, store = self._load_resume_candidate(record, context=context)
                spec = RepositoryRunSpec(
                    run_id=run_id,
                    task=loaded.checkpoint.task,
                    root=context.root,
                    model_name=config.model_name,
                    base_url=config.base_url,
                    reasoning_effort=config.reasoning_effort,
                    permission_mode=config.permission_mode,
                    paths=config.paths,
                    max_steps=config.max_steps,
                    model_timeout=config.model_timeout,
                )
                result = execute_repository_run(
                    spec,
                    event_sink=event_sink,
                    approver=None,
                    session_store=store,
                    loaded=loaded,
                    cancellation_token=cancellation_token,
                )
                self._project_memory.remember_run(
                    _context_memory_scope(context),
                    task=loaded.checkpoint.task,
                    result=result,
                )
                return result
            except BaseException:
                self._record_host_failure(run_id, event_sink)
                raise

    def _resume_record(self, run_id: str, *, context: ProjectContext) -> RunRecord:
        try:
            record = self._catalog.get(run_id)
        except (RunCatalogError, ValueError) as exc:
            raise _run_error(exc) from exc
        if (
            record.project_id != context.project_id
            or record.workspace_fingerprint != context.workspace_fingerprint
        ):
            raise WorkbenchNotFoundError("该任务不属于当前项目，不能继续")
        return record

    def _require_follow_up_parent(self, run_id: str, *, context: ProjectContext) -> RunRecord:
        """Require one completed, workspace-bound run with an allowlisted memory entry."""

        record = self._resume_record(run_id, context=context)
        try:
            summary = self._traces.summarize(record.run_id)
        except TraceNotFoundError as exc:
            raise WorkbenchFollowUpError("父任务没有可用的运行轨迹") from exc
        except (TraceError, OSError, ValueError) as exc:
            raise WorkbenchFollowUpError("父任务的运行轨迹暂时不可用") from exc
        if summary.status is not TraceRunStatus.COMPLETED:
            raise WorkbenchFollowUpError("只有已经完成的任务才能作为继续对话的父任务")
        index = self._project_memory.load_index(_context_memory_scope(context))
        memory_issue = _follow_up_memory_issue(record.run_id, index)
        if memory_issue == "missing":
            raise WorkbenchFollowUpError("父任务的项目记忆摘要不可用，不能静默丢失上下文")
        if memory_issue == "failed":
            raise WorkbenchFollowUpError("父任务的项目记忆摘要记录为失败，不能作为继续对话上下文")
        return record

    def _load_resume_candidate(
        self,
        record: RunRecord,
        *,
        context: ProjectContext,
    ) -> tuple[LoadedSession, SessionStore]:
        store = SessionStore(self._config.paths.sessions, workspace_root=context.root)
        try:
            loaded = store.load(record.run_id)
        except SessionError as exc:
            if exc.code in {"checkpoint_not_found", "invalid_run_id"}:
                raise WorkbenchResumeError("该任务没有可继续的检查点") from exc
            if exc.code == "checkpoint_workspace_mismatch":
                raise WorkbenchNotFoundError("该任务的检查点不属于当前项目") from exc
            raise WorkbenchResumeError("该任务的继续检查点不可用") from exc
        except (OSError, ValueError) as exc:
            raise WorkbenchResumeError("该任务的继续检查点不可用") from exc
        if loaded.checkpoint.stop_boundary is not SessionBoundary.READY_FOR_MODEL:
            raise WorkbenchResumeError("该任务已经结束，不能继续")
        if loaded.checkpoint.completed_steps >= self._config.max_steps:
            raise WorkbenchResumeError(
                "当前模型轮次上限已耗尽；请用更大的 --max-steps 重启 Web 后再继续"
            )
        try:
            trace_summary = self._traces.summarize(record.run_id)
        except TraceNotFoundError as exc:
            raise WorkbenchResumeError("该任务没有可继续的运行轨迹") from exc
        except (TraceError, OSError, ValueError) as exc:
            raise WorkbenchResumeError("该任务的运行轨迹不可用") from exc
        if trace_summary.status is TraceRunStatus.COMPLETED:
            raise WorkbenchResumeError("运行轨迹显示该任务已经完成，不能继续")
        return loaded, store

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

    def _run_payload(
        self,
        record: RunRecord,
        *,
        project: ProjectRecord,
        memory_index: ProjectMemoryIndex,
    ) -> RunCatalogPayload:
        status = "pending"
        completed_at: str | None = None
        trace_status: TraceRunStatus | None = None
        try:
            summary = self._traces.summarize(record.run_id)
        except TraceNotFoundError:
            pass
        except TraceError:
            status = "unavailable"
        else:
            trace_status = summary.status
            status = summary.status.value
            live = self._service.state()
            if live["status"] == "running" and live["run_id"] == record.run_id:
                status = "running"
            elif summary.status is TraceRunStatus.RUNNING:
                status = "interrupted"
            if summary.status is TraceRunStatus.COMPLETED and summary.verified is not True:
                status = "completed_unverified"
            if summary.status in {TraceRunStatus.COMPLETED, TraceRunStatus.FAILED}:
                completed_at = summary.last_event_at.isoformat()
        resume_available, resume_reason = self._resume_probe(
            record,
            project=project,
            trace_status=trace_status,
        )
        if resume_available and status == "failed":
            status = "interrupted"
            completed_at = None
        continuation: RunContinuationPayload
        if resume_available:
            continuation = {
                "kind": "resume",
                "available": True,
                "reason": None,
            }
        elif status in {"completed", "completed_unverified"}:
            follow_up_available, follow_up_reason = self._follow_up_probe(
                record,
                project=project,
                trace_status=trace_status,
                memory_index=memory_index,
            )
            continuation = {
                "kind": "follow_up" if follow_up_available else "none",
                "available": follow_up_available,
                "reason": follow_up_reason,
            }
        else:
            continuation = {
                "kind": "none",
                "available": False,
                "reason": resume_reason,
            }
        return {
            "run_id": record.run_id,
            "parent_run_id": record.parent_run_id,
            "project_id": record.project_id,
            "task": record.task_title,
            "status": status,
            "created_at": record.created_at.isoformat(),
            "completed_at": completed_at,
            # Lists stay metadata-only. A single history detail may replace this with
            # the bounded final reply from its workspace-bound terminal checkpoint.
            "final_text": None,
            "error": None,
            "resume_available": resume_available,
            "resume_reason": resume_reason,
            "continuation": continuation,
            "memory_context": self._project_memory.payload_for_record(
                record,
                scope=_project_memory_scope(project),
                index=memory_index,
            ),
        }

    def _resume_probe(
        self,
        record: RunRecord,
        *,
        project: ProjectRecord,
        trace_status: TraceRunStatus | None,
    ) -> tuple[bool, str | None]:
        """Return a conservative, presentation-safe resume capability snapshot."""

        unavailable_reason = self._continuation_environment_reason(record, project=project)
        if unavailable_reason is not None:
            return False, unavailable_reason
        if trace_status is None:
            return False, "运行轨迹不可用"
        if trace_status is TraceRunStatus.COMPLETED:
            return False, "任务已经完成"
        try:
            loaded = SessionStore(
                self._config.paths.sessions,
                workspace_root=project.resolved_root,
            ).load(record.run_id)
        except (OSError, SessionError, ValueError):
            return False, "没有可用的继续检查点"
        if loaded.checkpoint.stop_boundary is not SessionBoundary.READY_FOR_MODEL:
            return False, "任务已经结束"
        if loaded.checkpoint.completed_steps >= self._config.max_steps:
            return False, "当前轮次上限已耗尽；请用更大的 --max-steps 重启 Web"
        return True, None

    def _follow_up_probe(
        self,
        record: RunRecord,
        *,
        project: ProjectRecord,
        trace_status: TraceRunStatus | None,
        memory_index: ProjectMemoryIndex,
    ) -> tuple[bool, str | None]:
        """Return whether a terminal run can seed an explicit new-run follow-up."""

        unavailable_reason = self._continuation_environment_reason(record, project=project)
        if unavailable_reason is not None:
            return False, unavailable_reason
        if trace_status is not TraceRunStatus.COMPLETED:
            return False, "只有已经完成的任务才能继续对话"
        memory_issue = _follow_up_memory_issue(record.run_id, memory_index)
        if memory_issue == "missing":
            return False, "该历史没有可用的项目记忆摘要"
        if memory_issue == "failed":
            return False, "该历史的项目记忆摘要记录为失败，不能继续对话"
        return True, None

    def _continuation_environment_reason(
        self,
        record: RunRecord,
        *,
        project: ProjectRecord,
    ) -> str | None:
        """Check shared host and workspace preconditions for either continuation kind."""

        with self._lock:
            active = self._active
            live = self._service.state()
            if self._closed:
                return "本地项目服务正在关闭"
            if self._navigation_reserved:
                return "正在选择项目目录"
            if live["status"] == "running":
                return "已有任务正在运行"
            if active is None or active.project_id != record.project_id:
                return "请先打开该任务所属项目"
            if active.workspace_fingerprint != record.workspace_fingerprint:
                return "任务不属于当前项目身份"
        if record.workspace_fingerprint != project.workspace_fingerprint:
            return "项目目录身份已经变化"
        try:
            if workspace_fingerprint(project.resolved_root) != project.workspace_fingerprint:
                return "项目目录身份已经变化"
        except ValueError:
            return "项目目录当前不可访问"
        return None

    def _history_final_text(self, *, run_id: str, workspace_root: Path) -> str | None:
        """Project one terminal reply without exposing the canonical transcript."""

        try:
            value = SessionStore(
                self._config.paths.sessions,
                workspace_root=workspace_root,
            ).load_terminal_final_text(run_id)
        except (OSError, SessionError, ValueError):
            return None
        return _bounded_history_final_text(value)


def _project_context(record: ProjectRecord) -> ProjectContext:
    return ProjectContext(
        project_id=record.project_id,
        display_name=record.display_name,
        root=record.resolved_root,
        workspace_fingerprint=record.workspace_fingerprint,
    )


def _context_memory_scope(context: ProjectContext) -> ProjectMemoryScope:
    return ProjectMemoryScope(
        project_id=context.project_id,
        workspace_root=context.root,
        workspace_fingerprint=context.workspace_fingerprint,
    )


def _project_memory_scope(record: ProjectRecord) -> ProjectMemoryScope:
    return ProjectMemoryScope(
        project_id=record.project_id,
        workspace_root=record.resolved_root,
        workspace_fingerprint=record.workspace_fingerprint,
    )


def _follow_up_memory_issue(
    run_id: str,
    index: ProjectMemoryIndex,
) -> Literal["missing", "failed"] | None:
    entry = next((entry for entry in index.entries if entry.run_id == run_id), None)
    if entry is None:
        return "missing"
    if entry.final_status == "failed":
        return "failed"
    return None


def _project_payload(record: ProjectRecord) -> ProjectPayload:
    return {
        "project_id": record.project_id,
        "display_name": record.display_name,
        "root": str(record.resolved_root),
        "created_at": record.created_at.isoformat(),
        "last_opened_at": record.last_opened_at.isoformat(),
    }


def _bounded_history_final_text(value: str | None) -> str | None:
    """Keep Markdown layout while removing invisible controls and bounding the response."""

    if value is None:
        return None
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    display_safe = "".join(
        character
        if character in {"\n", "\t"} or character.isprintable()
        else "\N{REPLACEMENT CHARACTER}"
        for character in normalized
    ).strip()
    if not display_safe:
        return None
    if len(display_safe) <= _HISTORY_FINAL_TEXT_LIMIT:
        return display_safe
    retained = _HISTORY_FINAL_TEXT_LIMIT - len(_HISTORY_FINAL_TEXT_TRUNCATION_MARKER)
    return display_safe[:retained].rstrip() + _HISTORY_FINAL_TEXT_TRUNCATION_MARKER


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
    "ProjectRemovalPayload",
    "ProjectRunsPayload",
    "WebWorkbench",
    "WebWorkbenchConfig",
    "WorkbenchBusyError",
    "WorkbenchError",
    "WorkbenchInputError",
    "WorkbenchNotFoundError",
    "WorkbenchResumeError",
]
