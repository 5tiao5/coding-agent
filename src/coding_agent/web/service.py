"""Thread-safe bridge from a synchronous Agent run to a local Web frontend."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isfinite
from threading import RLock, Thread
from typing import Literal, Protocol, TypedDict, cast, overload
from uuid import uuid4

from coding_agent.cancellation import CancellationSource, CancellationToken
from coding_agent.dashboard import DashboardProjection, DashboardSnapshot, TimelineEntry
from coding_agent.errors import CodedError
from coding_agent.events import EventSink, RunEvent
from coding_agent.models import AgentResult, AgentState
from coding_agent.public_errors import public_coded_error, public_result_error

WebRunStatus = Literal["idle", "running", "completed", "completed_unverified", "failed"]
DEFAULT_CANCELLATION_GRACE_SECONDS = 0.25
MAX_CANCELLATION_GRACE_SECONDS = 1.0


class TimelinePayload(TypedDict):
    """Whitelisted presentation fields for one projected timeline entry."""

    step: int
    category: str
    headline: str
    detail: str | None
    level: str
    offset_seconds: float
    duration_ms: float | None
    preview: list[str]


class ChangedFilePayload(TypedDict):
    """Whitelisted mutation summary for one workspace-relative file."""

    path: str
    added_lines: int
    removed_lines: int
    revision: int
    change_kind: str


class VerificationEvidencePayload(TypedDict):
    """Whitelisted trusted verifier fact bound to one workspace revision."""

    label: str
    kind: str
    passed: bool
    step: int
    epoch: int


class RunLimitsPayload(TypedDict):
    """Whitelisted immutable budgets selected by the host for one run."""

    max_model_turns: int
    max_calls_per_turn: int
    max_total_tool_calls: int


class SnapshotPayload(TypedDict):
    """Browser-safe subset of :class:`DashboardSnapshot`."""

    task_label: str
    phase: str
    current_step: int
    limits: RunLimitsPayload | None
    tools_started: int
    tools_finished: int
    tools_failed: int
    active_tools: list[str]
    verification_status: str
    verification_labels: list[str]
    verification_evidence: list[VerificationEvidencePayload]
    verification_epoch: int
    invalidation_count: int
    changed_files: list[ChangedFilePayload]
    outcome: str
    plan_lines: list[str]
    timeline: list[TimelinePayload]
    latest_change: TimelinePayload | None


class WebRunStatePayload(TypedDict):
    """Stable JSON contract returned by the local Web API."""

    status: WebRunStatus
    run_id: str | None
    task: str | None
    final_text: str | None
    error: str | None
    snapshot: SnapshotPayload | None


class WebRunner(Protocol):
    """One injected synchronous run callable owned by the application layer."""

    def __call__(self, run_id: str, task: str, event_sink: EventSink) -> AgentResult: ...


class CancellableWebRunner(Protocol):
    """Opt-in runner that cooperatively observes a host-owned cancellation token."""

    def __call__(
        self,
        run_id: str,
        task: str,
        event_sink: EventSink,
        cancellation_token: CancellationToken,
    ) -> AgentResult: ...


class RunAlreadyActiveError(RuntimeError):
    """A second run was requested while the sole worker was occupied."""


class WebServiceClosedError(RuntimeError):
    """A run was requested after the host began graceful shutdown."""


class WebRunStartError(RuntimeError):
    """The operating system refused to start the background worker."""


class _ProjectionSink:
    """Fold events into the service without retaining raw event records."""

    def __init__(self, service: WebRunService, run_id: str) -> None:
        self._service = service
        self._run_id = run_id

    def emit(self, event: RunEvent) -> None:
        self._service._apply_event(self._run_id, event)


class WebRunService:
    """Own at most one background runner and expose immutable Web state snapshots.

    The injected runner remains synchronous. Only the small projection sink crosses the
    worker boundary, so browser reads never receive canonical messages or raw ``RunEvent``
    values.
    """

    @overload
    def __init__(
        self,
        runner: WebRunner | Callable[[str, str, EventSink], AgentResult],
        *,
        max_timeline: int = 10,
        cancellation_source: None = None,
        cancellation_grace_seconds: float = DEFAULT_CANCELLATION_GRACE_SECONDS,
    ) -> None: ...

    @overload
    def __init__(
        self,
        runner: CancellableWebRunner
        | Callable[[str, str, EventSink, CancellationToken], AgentResult],
        *,
        max_timeline: int = 10,
        cancellation_source: CancellationSource,
        cancellation_grace_seconds: float = DEFAULT_CANCELLATION_GRACE_SECONDS,
    ) -> None: ...

    def __init__(
        self,
        runner: WebRunner
        | CancellableWebRunner
        | Callable[[str, str, EventSink], AgentResult]
        | Callable[[str, str, EventSink, CancellationToken], AgentResult],
        *,
        max_timeline: int = 10,
        cancellation_source: CancellationSource | None = None,
        cancellation_grace_seconds: float = DEFAULT_CANCELLATION_GRACE_SECONDS,
    ) -> None:
        if max_timeline < 1:
            raise ValueError("max_timeline must be at least 1")
        if (
            not isfinite(cancellation_grace_seconds)
            or cancellation_grace_seconds < 0
            or cancellation_grace_seconds > MAX_CANCELLATION_GRACE_SECONDS
        ):
            raise ValueError(
                f"cancellation_grace_seconds must be between 0 and {MAX_CANCELLATION_GRACE_SECONDS}"
            )
        self._runner = runner
        self._max_timeline = max_timeline
        self._cancellation_source = cancellation_source
        self._cancellation_grace_seconds = cancellation_grace_seconds
        self._lock = RLock()
        self._status: WebRunStatus = "idle"
        self._run_id: str | None = None
        self._task: str | None = None
        self._final_text: str | None = None
        self._error: str | None = None
        self._projection: DashboardProjection | None = None
        self._worker: Thread | None = None
        self._accepting_runs = True
        self._cancellation_requested = False

    def start(self, task: str) -> WebRunStatePayload:
        """Start one run immediately and reject overlap with the active worker."""
        normalized_task = task.strip()
        if not normalized_task:
            raise ValueError("task cannot be blank")

        with self._lock:
            if not self._accepting_runs:
                raise WebServiceClosedError("本地 Agent 服务正在关闭")
            if self._worker is not None and self._worker.is_alive():
                raise RunAlreadyActiveError("已有任务正在运行")
            run_id = uuid4().hex
            self._status = "running"
            self._run_id = run_id
            self._task = normalized_task
            self._final_text = None
            self._error = None
            self._projection = DashboardProjection(
                task_label=normalized_task,
                max_timeline=self._max_timeline,
            )
            self._cancellation_requested = False
            worker = Thread(
                target=self._run,
                args=(run_id, normalized_task),
                name=f"coding-agent-web-{run_id[:8]}",
                daemon=True,
            )
            self._worker = worker
            try:
                worker.start()
            except RuntimeError as exc:
                self._status = "failed"
                self._error = "本地 Agent 后台任务启动失败。"
                self._worker = None
                raise WebRunStartError(self._error) from exc
            return self._state_locked()

    def state(self) -> WebRunStatePayload:
        """Return a fresh, whitelisted state document for the browser."""
        with self._lock:
            return self._state_locked()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the current worker; intended for hosts and deterministic tests."""
        with self._lock:
            worker = self._worker
        if worker is None:
            return True
        worker.join(timeout)
        return not worker.is_alive()

    def shutdown(self, timeout: float | None = None) -> bool:
        """Close admission, drain, then cooperatively cancel an opted-in runner.

        A zero-second timeout is used by hosts as a non-blocking busy probe. It closes
        admission as before, but deliberately does not emit a cancellation request.
        Positive drain timeouts may request cancellation only when a source was supplied;
        the following cleanup wait is always short and bounded.
        """
        if timeout is not None and timeout < 0:
            raise ValueError("timeout cannot be negative")
        with self._lock:
            self._accepting_runs = False
            worker = self._worker
        if worker is None:
            return True

        worker.join(timeout)
        if not worker.is_alive():
            return True
        if timeout is None or timeout == 0:
            return False

        source: CancellationSource | None
        request_now = False
        with self._lock:
            source = self._cancellation_source
            if source is not None and not self._cancellation_requested:
                self._cancellation_requested = True
                request_now = True
        if source is None:
            return False
        if request_now:
            source.request_cancellation()
        worker.join(self._cancellation_grace_seconds)
        return not worker.is_alive()

    def _run(self, run_id: str, task: str) -> None:
        try:
            sink = _ProjectionSink(self, run_id)
            if self._cancellation_source is None:
                runner = cast(WebRunner, self._runner)
                result = runner(run_id, task, sink)
            else:
                cancellable_runner = cast(CancellableWebRunner, self._runner)
                result = cancellable_runner(
                    run_id,
                    task,
                    sink,
                    self._cancellation_source.token,
                )
            if result.run_id != run_id:
                raise ValueError("runner returned a result for a different run")
        except BaseException as exc:  # Worker failures must always leave a terminal Web state.
            with self._lock:
                if self._run_id == run_id:
                    self._status = "failed"
                    self._final_text = None
                    self._error = _public_worker_error(exc)
            return

        with self._lock:
            if self._run_id != run_id:
                return
            self._status = _result_status(result)
            self._final_text = result.final_text
            self._error = public_result_error(result) if result.state is AgentState.FAILED else None

    def _apply_event(self, expected_run_id: str, event: RunEvent) -> None:
        if event.run_id != expected_run_id:
            raise ValueError("runner emitted an event for a different run")
        with self._lock:
            if self._run_id != expected_run_id or self._status != "running":
                raise RuntimeError("run is no longer active")
            if self._projection is None:  # Defensive: start always installs it first.
                raise RuntimeError("run projection is unavailable")
            self._projection.apply(event)

    def _state_locked(self) -> WebRunStatePayload:
        snapshot = self._projection.snapshot if self._projection is not None else None
        return {
            "status": self._status,
            "run_id": self._run_id,
            "task": self._task,
            "final_text": self._final_text,
            "error": self._error,
            "snapshot": _snapshot_payload(snapshot) if snapshot is not None else None,
        }


def _result_status(result: AgentResult) -> WebRunStatus:
    if result.state is AgentState.COMPLETED:
        return "completed"
    if result.state is AgentState.COMPLETED_UNVERIFIED:
        return "completed_unverified"
    return "failed"


def _snapshot_payload(snapshot: DashboardSnapshot) -> SnapshotPayload:
    return {
        "task_label": snapshot.task_label,
        "phase": snapshot.phase,
        "current_step": snapshot.current_step,
        "limits": (
            {
                "max_model_turns": snapshot.limits.max_model_turns,
                "max_calls_per_turn": snapshot.limits.max_calls_per_turn,
                "max_total_tool_calls": snapshot.limits.max_total_tool_calls,
            }
            if snapshot.limits is not None
            else None
        ),
        "tools_started": snapshot.tools_started,
        "tools_finished": snapshot.tools_finished,
        "tools_failed": snapshot.tools_failed,
        "active_tools": list(snapshot.active_tools),
        "verification_status": snapshot.verification_status,
        "verification_labels": list(snapshot.verification_labels),
        "verification_evidence": [
            {
                "label": item.label,
                "kind": item.kind,
                "passed": item.passed,
                "step": item.step,
                "epoch": item.epoch,
            }
            for item in snapshot.verification_evidence
        ],
        "verification_epoch": snapshot.verification_epoch,
        "invalidation_count": snapshot.invalidation_count,
        "changed_files": [
            {
                "path": item.path,
                "added_lines": item.added_lines,
                "removed_lines": item.removed_lines,
                "revision": item.revision,
                "change_kind": item.change_kind,
            }
            for item in snapshot.changed_files
        ],
        "outcome": snapshot.outcome,
        "plan_lines": _plan_lines(snapshot),
        "timeline": [_timeline_payload(entry) for entry in snapshot.timeline],
        "latest_change": (
            _timeline_payload(snapshot.latest_change)
            if snapshot.latest_change is not None
            else None
        ),
    }


def _plan_lines(snapshot: DashboardSnapshot) -> list[str]:
    value: object = getattr(snapshot, "plan_lines", ())
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    return [line for line in value if isinstance(line, str)]


def _timeline_payload(entry: TimelineEntry) -> TimelinePayload:
    return {
        "step": entry.step,
        "category": entry.category,
        "headline": entry.headline,
        "detail": entry.detail,
        "level": entry.level,
        "offset_seconds": entry.offset_seconds,
        "duration_ms": entry.duration_ms,
        "preview": list(entry.preview),
    }


def _public_worker_error(exc: BaseException) -> str:
    if isinstance(exc, CodedError):
        return public_coded_error(exc)
    if isinstance(exc, OSError):
        return "Agent 运行时发生了本地文件系统错误。"
    return f"本地 Agent 运行失败（{type(exc).__name__}）。"
