"""Rich dashboard projected exclusively from the public runtime event stream.

The dashboard is deliberately a passive projection: it does not receive messages,
tool outputs, or agent internals.  That keeps presentation concerns outside the
runtime and makes the same sink usable for a live terminal and deterministic
non-interactive recordings.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from types import TracebackType
from typing import Literal

from rich import box
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from coding_agent._dashboard_activity import (
    ActivityFact,
    ActivityState,
    activity_id,
    public_verification_kind,
    tool_activity_facts,
    verification_gate_facts,
    verification_recorded_facts,
)
from coding_agent._dashboard_evidence import (
    ChangedFile,
    VerificationEvidenceItem,
    changed_file_from_event,
    changed_file_label,
    verification_evidence_items,
    verification_evidence_label,
)
from coding_agent._presentation_safety import sanitize_public_label
from coding_agent.events import EventKind, RunEvent
from coding_agent.ui import console_safe

TimelineLevel = Literal["info", "success", "warning", "error"]

_TERMINAL_KINDS = {EventKind.RUN_FINISHED, EventKind.RUN_FAILED}
_START_KINDS = {EventKind.RUN_STARTED, EventKind.RUN_RESUMED}
_MUTATION_TOOLS = {"replace_text", "undo_change", "write_file"}
_NON_TTY_HIDDEN_KINDS = {EventKind.MODEL_REQUESTED, EventKind.MODEL_RESPONDED}
_VERIFICATION_STATUSES = {
    "verified",
    "checks_only",
    "missing",
    "failed",
    "stale",
    "unverified",
}
_NO_CURRENT_EVIDENCE_STATUSES = {"pending", "missing", "stale", "unverified"}
_MAX_PLAN_LINES = 8
_MAX_EVIDENCE_LABELS = 8
_MAX_CHANGED_FILES = 12
MAX_EXPANDED_MUTATION_PREVIEW_LINES = 80
_MAX_EXPANDED_DIFF_LINE_CHARS = 240
_MAX_VISIBLE_RUNTIME_LIMIT = 1_000_000
_LEVEL_STYLE: dict[TimelineLevel, str] = {
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red",
}
_LEVEL_MARK: dict[TimelineLevel, str] = {
    "info": "INFO",
    "success": "PASS",
    "warning": "WARN",
    "error": "FAIL",
}


@dataclass(frozen=True, slots=True)
class TimelineEntry:
    """One bounded, presentation-safe item in the visible activity timeline."""

    step: int
    category: str
    headline: str
    detail: str | None
    level: TimelineLevel
    offset_seconds: float
    duration_ms: float | None = None
    preview: tuple[str, ...] = ()
    activity_id: str | None = None
    activity_state: ActivityState | None = None
    facts: tuple[ActivityFact, ...] = ()
    facts_complete: bool = True
    expanded_preview: tuple[str, ...] = ()
    expanded_preview_complete: bool = True


@dataclass(frozen=True, slots=True)
class RunLimits:
    """Strictly whitelisted run budgets safe for presentation and trace replay."""

    max_model_turns: int
    max_calls_per_turn: int
    max_total_tool_calls: int


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    """Immutable view useful to renderers and deterministic tests."""

    run_id: str | None
    task_label: str
    phase: str
    current_step: int
    limits: RunLimits | None
    tools_started: int
    tools_finished: int
    tools_failed: int
    active_tools: tuple[str, ...]
    verification_status: str
    verification_labels: tuple[str, ...]
    verification_evidence: tuple[VerificationEvidenceItem, ...]
    verification_epoch: int
    invalidation_count: int
    changed_files: tuple[ChangedFile, ...]
    elapsed_seconds: float
    terminal: bool
    run_failed: bool
    stop_reason: str | None
    plan_lines: tuple[str, ...]
    timeline: tuple[TimelineEntry, ...]
    latest_change: TimelineEntry | None

    @property
    def outcome(self) -> str:
        if not self.terminal:
            return "RUNNING"
        if not self.run_failed and self.verification_status == "verified":
            return "VERIFIED"
        return "UNVERIFIED"


class DashboardProjection:
    """Fold :class:`RunEvent` values into a small UI-oriented state machine."""

    def __init__(
        self,
        *,
        task_label: str = "Coding task",
        max_timeline: int = 10,
        expanded_mutation_preview_lines: int = 0,
    ) -> None:
        if max_timeline < 1:
            raise ValueError("max_timeline must be at least 1")
        if not 0 <= expanded_mutation_preview_lines <= MAX_EXPANDED_MUTATION_PREVIEW_LINES:
            raise ValueError(
                "expanded_mutation_preview_lines must be between 0 and "
                f"{MAX_EXPANDED_MUTATION_PREVIEW_LINES}"
            )
        self._task_label = _clean_text(task_label, limit=80) or "Coding task"
        self._timeline: deque[TimelineEntry] = deque(maxlen=max_timeline)
        self._expanded_mutation_preview_lines = expanded_mutation_preview_lines
        self._run_id: str | None = None
        self._phase = "WAITING"
        self._current_step = 0
        self._limits: RunLimits | None = None
        self._tools_started = 0
        self._tools_finished = 0
        self._tools_failed = 0
        self._active_tools: dict[str, str] = {}
        self._verification_status = "pending"
        self._verification_labels: list[str] = []
        self._verification_evidence: dict[str, VerificationEvidenceItem] = {}
        self._verification_epoch = 0
        self._invalidation_count = 0
        self._changed_files: dict[str, ChangedFile] = {}
        self._started_at: datetime | None = None
        self._last_at: datetime | None = None
        self._terminal = False
        self._run_failed = False
        self._stop_reason: str | None = None
        self._plan_lines: tuple[str, ...] = ()
        self._latest_change: TimelineEntry | None = None

    @property
    def snapshot(self) -> DashboardSnapshot:
        return DashboardSnapshot(
            run_id=self._run_id,
            task_label=self._task_label,
            phase=self._phase,
            current_step=self._current_step,
            limits=self._limits,
            tools_started=self._tools_started,
            tools_finished=self._tools_finished,
            tools_failed=self._tools_failed,
            active_tools=tuple(self._active_tools.values()),
            verification_status=self._verification_status,
            verification_labels=tuple(self._verification_labels),
            verification_evidence=tuple(self._verification_evidence.values()),
            verification_epoch=self._verification_epoch,
            invalidation_count=self._invalidation_count,
            changed_files=tuple(self._changed_files.values()),
            elapsed_seconds=self._elapsed_seconds(),
            terminal=self._terminal,
            run_failed=self._run_failed,
            stop_reason=self._stop_reason,
            plan_lines=self._plan_lines,
            timeline=tuple(self._timeline),
            latest_change=self._latest_change,
        )

    def apply(self, event: RunEvent) -> TimelineEntry | None:
        """Apply an event and return a new visible timeline item, if any."""
        if self._run_id is None:
            self._run_id = event.run_id
            self._started_at = event.timestamp
        elif event.run_id != self._run_id:
            raise ValueError("a dashboard projection can only contain one run")
        if self._last_at is None or event.timestamp > self._last_at:
            self._last_at = event.timestamp
        self._current_step = max(self._current_step, event.step)

        entry = self._project_event(event)
        if entry is not None:
            self._timeline.append(entry)
        return entry

    def _project_event(self, event: RunEvent) -> TimelineEntry | None:  # noqa: C901
        kind = event.kind
        data = event.data
        if kind is EventKind.RUN_STARTED:
            self._phase = "PLANNING"
            self._limits = _run_limits(data)
            return self._entry(event, "RUN", "Task accepted", "Workspace run started")
        if kind is EventKind.RUN_RESUMED:
            self._phase = "RESUMED"
            self._limits = _run_limits(data)
            completed = _data_int(data, "completed_steps")
            detail = "Fresh verification is required"
            if completed is not None:
                detail = f"Resumed after {completed} completed step(s); {detail.lower()}"
            return self._entry(event, "RUN", "Session resumed", detail, level="warning")
        if kind is EventKind.STATE_CHANGED:
            current = _data_string(data, "current")
            if current:
                self._phase = _status_label(current)
            return None
        if kind is EventKind.MODEL_REQUESTED:
            self._phase = "DECIDING"
            return self._entry(event, "MODEL", "Selecting the next action", None)
        if kind is EventKind.MODEL_RETRYING:
            retry_kind = _data_string(data, "retry_kind")
            if retry_kind == "verification_closeout":
                self._phase = "VERIFYING"
                return self._entry(
                    event,
                    "VERIFY",
                    "Final response deferred; verification scheduled",
                    "Fresh verification is required",
                    level="warning",
                )
            self._phase = "RETRYING"
            next_attempt = _data_int(data, "next_attempt")
            max_attempts = _data_int(data, "max_attempts")
            delay_seconds = _data_number(data, "delay_seconds")
            detail_parts: list[str] = []
            if next_attempt is not None and max_attempts is not None:
                detail_parts.append(f"Attempt {next_attempt} of {max_attempts}")
            if delay_seconds is not None:
                detail_parts.append(f"after {delay_seconds:g}s")
            error_code = _data_string(data, "error_code")
            protocol_correction = retry_kind == "protocol_correction" or error_code in {
                "model_response_invalid",
                "openai_invalid_function_arguments",
            }
            if protocol_correction:
                detail_parts.append("MODEL RESPONSE INVALID")
            elif error_code == "model_request_transient":
                detail_parts.append("MODEL REQUEST TRANSIENT")
            else:
                detail_parts.append("MODEL REQUEST FAILURE")
            return self._entry(
                event,
                "MODEL",
                (
                    "Invalid model response; protocol correction scheduled"
                    if protocol_correction
                    else "Transient model failure; retry scheduled"
                ),
                " · ".join(detail_parts) or None,
                level="warning",
            )
        if kind is EventKind.MODEL_RESPONDED:
            tool_count = _data_int(data, "tool_count") or 0
            has_content = _data_bool(data, "has_content")
            if tool_count:
                detail = f"Prepared {tool_count} tool call(s)"
            elif has_content:
                detail = "Prepared a final response"
            else:
                detail = "Response received"
            return self._entry(event, "MODEL", "Action selected", detail)
        if kind is EventKind.TOOL_BATCH_REJECTED:
            self._phase = "REPLANNING"
            batch_detail = _tool_batch_rejection_detail(data)
            return self._entry(
                event,
                "MODEL",
                "Tool batch too large; split retry requested",
                batch_detail,
                level="warning",
            )
        if kind is EventKind.CONTEXT_COMPACTED:
            blocks = _data_int(data, "compacted_blocks")
            context_detail = (
                f"Summarized {blocks} older tool block(s)" if blocks is not None else None
            )
            return self._entry(event, "CONTEXT", "Context budget compacted", context_detail)
        if kind is EventKind.TOOL_STARTED:
            name = _data_string(data, "tool_name") or "tool"
            call_id = _data_string(data, "call_id") or f"step-{event.step}-{self._tools_started}"
            if call_id not in self._active_tools:
                self._tools_started += 1
            self._active_tools[call_id] = name
            self._phase = f"RUNNING {_status_label(name)}"
            facts, facts_complete = tool_activity_facts(name, data, finished=False)
            return self._entry(
                event,
                "TOOL",
                f"Running {name}",
                None,
                activity_id=activity_id(event.run_id, data),
                activity_state="started",
                facts=facts,
                facts_complete=facts_complete,
            )
        if kind is EventKind.TOOL_FINISHED:
            return self._tool_finished(event)
        if kind is EventKind.VERIFICATION_INVALIDATED:
            self._verification_status = "stale"
            self._verification_labels.clear()
            self._verification_evidence.clear()
            epoch = _data_int(data, "epoch")
            if epoch is not None:
                self._verification_epoch = max(self._verification_epoch, epoch)
            self._invalidation_count += 1
            return self._entry(
                event,
                "VERIFY",
                "Previous evidence invalidated",
                "A workspace change requires a fresh check",
                level="warning",
            )
        if kind is EventKind.VERIFICATION_RECORDED:
            return self._verification_recorded(event)
        if kind is EventKind.VERIFICATION_EVALUATED:
            self._phase = "VERIFYING"
            self._update_verification_report(data)
            level: TimelineLevel = (
                "success" if self._verification_status == "verified" else "warning"
            )
            facts, facts_complete = verification_gate_facts(data)
            return self._entry(
                event,
                "VERIFY",
                "Verification gate evaluated",
                _status_label(self._verification_status),
                level=level,
                facts=facts,
                facts_complete=facts_complete,
            )
        if kind is EventKind.SESSION_CHECKPOINTED:
            boundary = _data_string(data, "boundary")
            checkpoint_detail = _status_label(boundary) if boundary else None
            return self._entry(event, "SAVE", "Checkpoint saved", checkpoint_detail)
        if kind is EventKind.SESSION_CHECKPOINT_FAILED:
            return self._entry(
                event,
                "SAVE",
                "Checkpoint not saved",
                _error_detail(data),
                level="warning",
            )
        if kind is EventKind.RUN_FINISHED:
            self._terminal = True
            self._phase = "COMPLETED"
            self._active_tools.clear()
            self._update_verification_report(data)
            return None
        if kind is EventKind.RUN_FAILED:
            self._terminal = True
            self._run_failed = True
            self._phase = "FAILED"
            self._active_tools.clear()
            self._verification_labels.clear()
            self._verification_evidence.clear()
            self._stop_reason = _data_string(data, "stop_reason")
            if self._verification_status in {"pending", "verified"}:
                self._verification_status = "unverified"
            return None
        return None

    def _tool_finished(self, event: RunEvent) -> TimelineEntry:
        data = event.data
        name = _data_string(data, "tool_name") or "tool"
        call_id = _data_string(data, "call_id")
        if call_id:
            self._active_tools.pop(call_id, None)
        self._tools_finished += 1
        explicit_ok = _data_bool(data, "ok")
        ok = explicit_ok is not False
        level: TimelineLevel = "success" if ok else "error"
        if not ok:
            self._tools_failed += 1
        summary = _data_string(data, "summary")
        detail = summary if summary else (_error_detail(data) if not ok else "Completed")
        duration_ms = _data_number(data, "duration_ms")
        self._phase = "OBSERVING"
        preview = _preview_lines(
            data.get("preview"),
            max_lines=_MAX_PLAN_LINES if name == "update_plan" else 6,
        )
        facts, facts_complete = tool_activity_facts(name, data, finished=True)
        successful_mutation = name in _MUTATION_TOOLS and explicit_ok is True
        entry = self._entry(
            event,
            "TOOL",
            f"{name} {'completed' if ok else 'failed'}",
            detail,
            level=level,
            duration_ms=duration_ms,
            preview=preview,
            activity_id=activity_id(event.run_id, data),
            activity_state="finished",
            facts=facts,
            facts_complete=facts_complete,
        )
        if name == "update_plan" and explicit_ok is True:
            # The plan is durable presentation state, not merely a recent event.  An
            # explicitly successful update replaces the previous snapshot; a missing
            # or malformed safe preview clears it rather than showing a stale plan.
            self._plan_lines = preview
        if successful_mutation and entry.preview:
            latest_change = entry
            if self._expanded_mutation_preview_lines:
                expanded_preview, projection_complete = _expanded_mutation_preview(
                    data.get("preview"),
                    max_lines=self._expanded_mutation_preview_lines,
                )
                latest_change = replace(
                    entry,
                    expanded_preview=expanded_preview,
                    expanded_preview_complete=(
                        projection_complete and _mutation_preview_is_complete(data)
                    ),
                )
            self._latest_change = latest_change
        if name in _MUTATION_TOOLS and explicit_ok is True:
            self._record_changed_file(data)
        return entry

    def _record_changed_file(self, data: Mapping[str, object]) -> None:
        change = changed_file_from_event(data)
        if change is None:
            return
        existing = self._changed_files.get(change.path)
        if existing is None and len(self._changed_files) >= _MAX_CHANGED_FILES:
            return
        self._changed_files[change.path] = change

    def _verification_recorded(self, event: RunEvent) -> TimelineEntry:
        data = event.data
        passed = _data_bool(data, "passed") is True
        self._verification_status = "passed" if passed else "failed"
        label, _ = sanitize_public_label(data.get("label"), limit=100)
        if (
            label
            and label not in self._verification_labels
            and len(self._verification_labels) < _MAX_EVIDENCE_LABELS
        ):
            self._verification_labels.append(label)
        kind = public_verification_kind(data.get("kind"))
        epoch = _data_int(data, "epoch")
        if epoch is not None:
            self._verification_epoch = max(self._verification_epoch, epoch)
        if (
            label
            and kind
            and (
                label in self._verification_evidence
                or len(self._verification_evidence) < _MAX_EVIDENCE_LABELS
            )
        ):
            self._verification_evidence[label] = VerificationEvidenceItem(
                label=label,
                kind=kind,
                passed=passed,
                step=max(0, event.step),
                epoch=max(0, epoch or 0),
            )
        detail_parts = [part for part in (kind, label) if part]
        detail = " / ".join(detail_parts) if detail_parts else None
        facts, facts_complete = verification_recorded_facts(event, data)
        return self._entry(
            event,
            "VERIFY",
            "Passing evidence recorded" if passed else "Failing evidence recorded",
            detail,
            level="success" if passed else "error",
            activity_id=activity_id(event.run_id, data),
            facts=facts,
            facts_complete=facts_complete,
        )

    def _update_verification_status(self, data: Mapping[str, object]) -> None:
        verified = _data_bool(data, "verified")
        status = _data_string(data, "status")
        if verified is True:
            self._verification_status = "verified"
        elif status in _VERIFICATION_STATUSES:
            self._verification_status = status
        elif verified is False:
            self._verification_status = "unverified"

    def _update_verification_report(self, data: Mapping[str, object]) -> None:
        """Replace the projection with the report's current, bounded evidence set."""
        self._update_verification_status(data)
        epoch = _data_int(data, "epoch")
        if epoch is not None:
            self._verification_epoch = max(0, epoch)
        invalidations = _data_int(data, "invalidation_count")
        if invalidations is not None:
            self._invalidation_count = max(0, invalidations)
        labels = _verification_label_sequence(
            data.get("evidence_labels"),
            max_items=_MAX_EVIDENCE_LABELS,
        )
        if self._verification_status in _NO_CURRENT_EVIDENCE_STATUSES:
            labels = ()
        self._verification_labels = list(labels)
        evidence = verification_evidence_items(
            data.get("evidence"),
            max_items=_MAX_EVIDENCE_LABELS,
        )
        if evidence:
            self._verification_evidence = {item.label: item for item in evidence}
        elif self._verification_status in _NO_CURRENT_EVIDENCE_STATUSES:
            self._verification_evidence.clear()
        elif labels:
            self._verification_evidence = {
                label: item
                for label, item in self._verification_evidence.items()
                if label in labels
            }

    def _entry(
        self,
        event: RunEvent,
        category: str,
        headline: str,
        detail: str | None,
        *,
        level: TimelineLevel = "info",
        duration_ms: float | None = None,
        preview: tuple[str, ...] = (),
        activity_id: str | None = None,
        activity_state: ActivityState | None = None,
        facts: tuple[ActivityFact, ...] = (),
        facts_complete: bool = True,
    ) -> TimelineEntry:
        return TimelineEntry(
            step=event.step,
            category=category,
            headline=_clean_text(headline, limit=100),
            detail=_clean_text(detail, limit=180) if detail else None,
            level=level,
            offset_seconds=self._event_offset(event.timestamp),
            duration_ms=duration_ms,
            preview=preview,
            activity_id=activity_id,
            activity_state=activity_state,
            facts=facts,
            facts_complete=facts_complete,
        )

    def _event_offset(self, timestamp: datetime) -> float:
        if self._started_at is None:
            return 0.0
        return max(0.0, (timestamp - self._started_at).total_seconds())

    def _elapsed_seconds(self) -> float:
        if self._started_at is None or self._last_at is None:
            return 0.0
        return max(0.0, (self._last_at - self._started_at).total_seconds())


class DashboardEventSink:
    """EventSink-compatible Rich timeline with optional in-place TTY updates.

    ``live=None`` enables live rendering only when Rich identifies the destination
    as an interactive terminal.  Redirected output and recordings therefore get
    stable append-only lines and a final ASCII-boxed result card.
    """

    def __init__(
        self,
        console: Console | None = None,
        *,
        live: bool | None = None,
        task_label: str = "Coding task",
        max_timeline: int = 10,
        refresh_per_second: float = 8.0,
        auto_final_card: bool = True,
    ) -> None:
        if refresh_per_second <= 0:
            raise ValueError("refresh_per_second must be positive")
        self._console = console or Console()
        self._task_label = task_label
        self._max_timeline = max_timeline
        self._projection = DashboardProjection(
            task_label=task_label,
            max_timeline=max_timeline,
        )
        self._use_live = self._console.is_terminal if live is None else live
        self._refresh_per_second = refresh_per_second
        self._auto_final_card = auto_final_card
        self._live: Live | None = None
        self._final_card_printed = False

    @property
    def snapshot(self) -> DashboardSnapshot:
        return self._projection.snapshot

    @property
    def uses_live_rendering(self) -> bool:
        return self._use_live

    def emit(self, event: RunEvent) -> None:
        if self._starts_another_run(event):
            self.close()
            self._projection = DashboardProjection(
                task_label=self._task_label,
                max_timeline=self._max_timeline,
            )
            self._final_card_printed = False

        entry = self._projection.apply(event)
        if self._use_live:
            self._render_live()
        elif entry is not None and event.kind not in _NON_TTY_HIDDEN_KINDS:
            self._print_timeline_entry(entry)

        if event.kind in _TERMINAL_KINDS:
            if self._live is not None:
                self._live.stop()
                self._live = None
            if self._auto_final_card:
                self.print_final_card()

    def close(self) -> None:
        """Stop an active Live display without manufacturing a terminal outcome."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def __enter__(self) -> DashboardEventSink:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

    def render(self) -> RenderableType:
        """Render a complete point-in-time dashboard without side effects."""
        snapshot = self.snapshot
        header = Table.grid(expand=True, padding=(0, 1))
        header.add_column(style="dim", width=14)
        header.add_column(ratio=1)
        header.add_column(style="dim", width=12)
        header.add_column(justify="right")
        header.add_row(
            Text("TASK"),
            self._text(snapshot.task_label, "bold"),
            Text("PHASE"),
            self._text(snapshot.phase, _phase_style(snapshot.phase)),
        )
        header.add_row(
            Text("PROGRESS"),
            self._text(_progress_label(snapshot)),
            Text("ELAPSED"),
            Text(_format_duration(snapshot.elapsed_seconds * 1_000)),
        )

        timeline = Table(box=None, expand=True, pad_edge=False, show_header=True)
        timeline.add_column("STEP", style="dim", width=6)
        timeline.add_column("EVENT", width=10)
        timeline.add_column("DETAIL", ratio=1)
        timeline.add_column("TIME", style="dim", justify="right", width=10)
        if not snapshot.timeline:
            timeline.add_row("-", "WAIT", "Waiting for the first runtime event", "0 ms")
        else:
            for entry in snapshot.timeline:
                detail = Text()
                detail.append(self._safe(entry.headline), style="bold")
                if entry.detail:
                    detail.append(f" - {self._safe(entry.detail)}", style="dim")
                for preview_line in entry.preview:
                    detail.append(f"\n  | {self._safe(preview_line)}", style="dim")
                timeline.add_row(
                    str(entry.step) if entry.step else "-",
                    Text(entry.category, style=_LEVEL_STYLE[entry.level]),
                    detail,
                    _entry_time(entry),
                )

        verification = self._verification_text(snapshot)
        return Group(
            Panel(
                header,
                title=" CODING AGENT ",
                border_style="cyan",
                box=box.ASCII,
            ),
            Panel(
                self._plan_text(snapshot),
                title=" CURRENT PLAN ",
                border_style="cyan" if snapshot.plan_lines else "dim",
                box=box.ASCII,
            ),
            Panel(timeline, title=" ACTIVITY TIMELINE ", border_style="blue", box=box.ASCII),
            Panel(
                self._latest_change_text(snapshot),
                title=" LATEST CHANGE ",
                border_style="magenta" if snapshot.latest_change is not None else "dim",
                box=box.ASCII,
            ),
            Panel(
                verification,
                title=" VERIFICATION GATE ",
                border_style=_verification_style(snapshot.verification_status),
                box=box.ASCII,
            ),
        )

    def render_final_card(self) -> Panel:
        """Build an explicit trustworthy terminal outcome card."""
        snapshot = self.snapshot
        outcome = snapshot.outcome
        style = "bold green" if outcome == "VERIFIED" else "bold red"
        explanation = _outcome_explanation(snapshot)
        body = Text(justify="center")
        body.append(f"{outcome}\n", style=style)
        body.append(self._safe(explanation), style="bold")
        body.append(
            "\n\n"
            f"{snapshot.current_step} step(s) | "
            f"{snapshot.tools_finished}/{snapshot.tools_started} tool(s) finished | "
            f"{_format_duration(snapshot.elapsed_seconds * 1_000)}",
            style="dim",
        )
        if snapshot.changed_files:
            changes = ", ".join(changed_file_label(item) for item in snapshot.changed_files)
            body.append(f"\nChanges: {self._safe(changes)}", style="dim")
        if snapshot.verification_evidence:
            evidence = ", ".join(
                verification_evidence_label(item) for item in snapshot.verification_evidence
            )
            body.append(f"\nEvidence: {self._safe(evidence)}", style="dim")
        elif snapshot.verification_labels:
            labels = ", ".join(snapshot.verification_labels)
            body.append(f"\nEvidence: {self._safe(labels)}", style="dim")
        if snapshot.verification_status == "verified":
            body.append(
                f"\nFreshness: evidence matches workspace revision {snapshot.verification_epoch}",
                style="dim",
            )
        if snapshot.run_failed:
            body.append("\nRun state: FAILED", style="bold red")
            if snapshot.stop_reason:
                reason = self._safe(_status_label(snapshot.stop_reason))
                body.append(f"\nStop reason: {reason}", style="dim")
        return Panel(
            body,
            title=" FINAL RESULT ",
            border_style="green" if outcome == "VERIFIED" else "red",
            box=box.ASCII,
            padding=(1, 2),
        )

    def print_final_card(self) -> bool:
        """Print a terminal result once, allowing the host to choose its final position."""
        if not self.snapshot.terminal or self._final_card_printed:
            return False
        self._console.print(self.render_final_card())
        self._final_card_printed = True
        return True

    def _render_live(self) -> None:
        renderable = self.render()
        if self._live is None:
            self._live = Live(
                renderable,
                console=self._console,
                auto_refresh=False,
                refresh_per_second=self._refresh_per_second,
                transient=False,
                vertical_overflow="visible",
            )
            self._live.start(refresh=True)
        else:
            self._live.update(renderable, refresh=True)

    def _print_timeline_entry(self, entry: TimelineEntry) -> None:
        style = _LEVEL_STYLE[entry.level]
        line = Text()
        line.append(f"[{_LEVEL_MARK[entry.level]}] ", style=f"bold {style}")
        if entry.step:
            line.append(f"step {entry.step}  ", style="dim")
        line.append(f"{entry.category:<8} ", style=style)
        line.append(self._safe(entry.headline), style="bold")
        if entry.detail:
            line.append(f" - {self._safe(entry.detail)}", style="dim")
        if entry.duration_ms is not None:
            line.append(f" ({_format_duration(entry.duration_ms)})", style="dim")
        self._console.print(line)
        for preview_line in entry.preview:
            self._console.print(Text(f"          | {self._safe(preview_line)}", style="dim"))

    def _verification_text(self, snapshot: DashboardSnapshot) -> Text:
        status = _status_label(snapshot.verification_status)
        text = Text()
        text.append(status, style=f"bold {_verification_style(snapshot.verification_status)}")
        if snapshot.verification_labels:
            labels = ", ".join(snapshot.verification_labels)
            text.append(f"  Evidence: {self._safe(labels)}", style="dim")
        elif snapshot.verification_status == "pending":
            text.append("  Waiting for trusted test/build/check evidence", style="dim")
        return text

    def _latest_change_text(self, snapshot: DashboardSnapshot) -> Text:
        change = snapshot.latest_change
        if change is None:
            return Text("No workspace mutation recorded yet", style="dim")
        text = Text()
        text.append(self._safe(change.headline), style="bold")
        if change.detail:
            text.append(f" - {self._safe(change.detail)}", style="dim")
        for line in change.preview:
            text.append("\n")
            text.append(self._safe(line), style=_diff_line_style(line))
        return text

    def _plan_text(self, snapshot: DashboardSnapshot) -> Text:
        if not snapshot.plan_lines:
            return Text("No structured plan recorded yet", style="dim")
        text = Text()
        for index, line in enumerate(snapshot.plan_lines):
            if index:
                text.append("\n")
            text.append(self._safe(line), style="dim")
        return text

    def _safe(self, value: str) -> str:
        return console_safe(_clean_text(value, limit=600), self._console)

    def _text(self, value: str, style: str | None = None) -> Text:
        if style is None:
            return Text(self._safe(value))
        return Text(self._safe(value), style=style)

    def _starts_another_run(self, event: RunEvent) -> bool:
        snapshot = self.snapshot
        return (
            event.kind in _START_KINDS
            and snapshot.run_id is not None
            and (event.run_id != snapshot.run_id or snapshot.terminal)
        )


def _data_string(data: Mapping[str, object], key: str) -> str | None:
    value = data.get(key)
    if not isinstance(value, str):
        return None
    cleaned = _clean_text(value, limit=180)
    return cleaned or None


def _data_bool(data: Mapping[str, object], key: str) -> bool | None:
    value = data.get(key)
    return value if isinstance(value, bool) else None


def _data_int(data: Mapping[str, object], key: str) -> int | None:
    value = data.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _bounded_positive_int(data: Mapping[str, object], key: str) -> int | None:
    value = _data_int(data, key)
    if value is None or not 1 <= value <= _MAX_VISIBLE_RUNTIME_LIMIT:
        return None
    return value


def _run_limits(data: Mapping[str, object]) -> RunLimits | None:
    raw_limits = data.get("limits")
    if not isinstance(raw_limits, Mapping):
        return None
    max_model_turns = _bounded_positive_int(raw_limits, "max_model_turns")
    max_calls_per_turn = _bounded_positive_int(raw_limits, "max_calls_per_turn")
    max_total_tool_calls = _bounded_positive_int(raw_limits, "max_total_tool_calls")
    if None in (max_model_turns, max_calls_per_turn, max_total_tool_calls):
        return None
    assert max_model_turns is not None
    assert max_calls_per_turn is not None
    assert max_total_tool_calls is not None
    return RunLimits(
        max_model_turns=max_model_turns,
        max_calls_per_turn=max_calls_per_turn,
        max_total_tool_calls=max_total_tool_calls,
    )


def _tool_batch_rejection_detail(data: Mapping[str, object]) -> str | None:
    requested_calls = _bounded_positive_int(data, "requested_calls")
    max_calls_per_turn = _bounded_positive_int(data, "max_calls_per_turn")
    rejection_count = _bounded_positive_int(data, "rejection_count")
    max_rejections = _bounded_positive_int(data, "max_rejections")
    if (
        requested_calls is None
        or max_calls_per_turn is None
        or requested_calls <= max_calls_per_turn
    ):
        return None
    detail = (
        f"Requested {requested_calls} tool calls; per-turn limit {max_calls_per_turn}; "
        "split retry requested"
    )
    if (
        rejection_count is not None
        and max_rejections is not None
        and rejection_count <= max_rejections
    ):
        detail += f"; rejection {rejection_count} of {max_rejections}"
    return detail


def _data_number(data: Mapping[str, object], key: str) -> float | None:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    numeric = float(value)
    return numeric if numeric >= 0 and math.isfinite(numeric) else None


def _verification_label_sequence(
    value: object,
    *,
    max_items: int,
) -> tuple[str, ...]:
    """Read bounded, credential-free verifier labels from a terminal report."""

    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    result: list[str] = []
    for candidate in value[:max_items]:
        label, _ = sanitize_public_label(candidate, limit=100)
        if label and label not in result:
            result.append(label)
    return tuple(result)


def _error_detail(data: Mapping[str, object]) -> str | None:
    code = _data_string(data, "error_code") or _data_string(data, "stop_reason")
    return _status_label(code) if code else None


def _preview_lines(value: object, *, max_lines: int = 6) -> tuple[str, ...]:
    """Extract only bounded, explicitly presentable preview fields.

    Mappings are never stringified.  This avoids accidentally turning arbitrary
    event data into a raw JSON/debug dump while still supporting plan item and
    bounded-diff previews.
    """
    lines: list[str] = []

    def append_text(candidate: object, *, prefix: str = "") -> None:
        if len(lines) >= max_lines or not isinstance(candidate, str):
            return
        for raw_line in candidate.splitlines() or [candidate]:
            cleaned = _clean_text(raw_line, limit=140)
            if cleaned:
                lines.append(f"{prefix}{cleaned}")
            if len(lines) >= max_lines:
                break

    def append_preview_text(candidate: str, *, prefix: str = "") -> None:
        raw_lines = candidate.splitlines()
        marker = next(
            (
                index
                for index, raw_line in enumerate(raw_lines)
                if raw_line.strip().casefold() == "diff preview:"
            ),
            None,
        )
        if marker is None:
            append_text(candidate, prefix=prefix)
            return
        diff_lines = raw_lines[marker + 1 :]
        headers = [line for line in diff_lines if line.startswith(("--- ", "+++ ", "@@ "))]
        changes = [
            line
            for line in diff_lines
            if line.startswith(("-", "+")) and not line.startswith(("--- ", "+++ "))
        ]
        selected = [*headers[:3], *changes[: max(0, max_lines - 3)]]
        append_text("\n".join(selected or diff_lines), prefix=prefix)

    def visit(candidate: object) -> None:
        if len(lines) >= max_lines:
            return
        if isinstance(candidate, str):
            append_preview_text(candidate)
            return
        if isinstance(candidate, Mapping):
            for container_key in ("plan", "items", "lines", "changes"):
                nested = candidate.get(container_key)
                if isinstance(nested, Sequence) and not isinstance(nested, str | bytes):
                    visit(nested)
                    return
            status = candidate.get("status")
            prefix = f"[{_status_label(status)}] " if isinstance(status, str) else ""
            for text_key in ("step", "text", "summary", "path", "diff"):
                text_value = candidate.get(text_key)
                if isinstance(text_value, str):
                    append_preview_text(text_value, prefix=prefix)
                    return
            return
        if isinstance(candidate, Sequence) and not isinstance(candidate, str | bytes):
            for item in candidate:
                visit(item)
                if len(lines) >= max_lines:
                    break

    visit(value)
    return tuple(lines)


def _expanded_mutation_preview(
    value: object,
    *,
    max_lines: int,
) -> tuple[tuple[str, ...], bool]:
    """Preserve one mutation Diff's order while bounding its Web-only projection."""

    raw_lines: list[str] | None = None
    if isinstance(value, str):
        raw_lines = value.splitlines()
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
        raw_lines = []
        for item in value:
            if isinstance(item, str):
                raw_lines.extend(item.splitlines() or [item])
    if raw_lines is None:
        fallback = _preview_lines(value, max_lines=max_lines)
        # A structured fallback may itself have selected or shortened fields.
        # Keep it display-safe, but never claim that it represents the full Diff.
        return fallback, False

    marker = next(
        (
            index
            for index, raw_line in enumerate(raw_lines)
            if raw_line.strip().casefold() == "diff preview:"
        ),
        None,
    )
    candidates = raw_lines[marker + 1 :] if marker is not None else raw_lines
    complete = True
    safe_lines: list[str] = []
    for raw_line in candidates:
        safe_line, line_complete = _clean_preview_line(
            raw_line,
            limit=_MAX_EXPANDED_DIFF_LINE_CHARS,
        )
        safe_lines.append(safe_line)
        complete = complete and line_complete

    if len(safe_lines) <= max_lines:
        return tuple(safe_lines), complete

    head_count = (max_lines - 1) // 2
    tail_count = max_lines - head_count - 1
    omitted = len(safe_lines) - head_count - tail_count
    marker_line = f"...[{omitted} lines omitted from expanded Diff preview]..."
    tail = safe_lines[-tail_count:] if tail_count else []
    sampled = [
        *safe_lines[:head_count],
        marker_line,
        *tail,
    ]
    return tuple(sampled), False


def _mutation_preview_is_complete(data: Mapping[str, object]) -> bool:
    if _data_bool(data, "truncated") is True:
        return False
    metadata = data.get("metadata")
    if isinstance(metadata, Mapping) and _data_bool(metadata, "diff_complete") is False:
        return False
    preview = data.get("preview")
    if isinstance(preview, str):
        lowered = preview.casefold()
        markers = (
            "diff preview truncated",
            "lines omitted from diff preview",
        )
        if any(marker in lowered for marker in markers):
            return False
    return True


def _clean_preview_line(value: str, *, limit: int) -> tuple[str, bool]:
    visible = "".join(" " if not character.isprintable() else character for character in value)
    if len(visible) <= limit:
        return visible, True
    return f"{visible[: max(0, limit - 3)]}...", False


def _clean_text(value: str, *, limit: int) -> str:
    printable = "".join(" " if not char.isprintable() else char for char in value)
    collapsed = " ".join(printable.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: max(0, limit - 3)]}..."


def _status_label(value: object) -> str:
    if not isinstance(value, str):
        return "UNKNOWN"
    cleaned = _clean_text(value, limit=80).replace("_", " ").replace("-", " ")
    return cleaned.upper() or "UNKNOWN"


def _format_duration(duration_ms: float) -> str:
    if duration_ms < 1_000:
        return f"{duration_ms:.0f} ms"
    return f"{duration_ms / 1_000:.1f} s"


def _entry_time(entry: TimelineEntry) -> str:
    if entry.duration_ms is not None:
        return _format_duration(entry.duration_ms)
    return f"+{_format_duration(entry.offset_seconds * 1_000)}"


def _diff_line_style(line: str) -> str:
    if line.startswith("+++") or line.startswith("---"):
        return "bold magenta"
    if line.startswith("+"):
        return "green"
    if line.startswith("-"):
        return "red"
    if line.startswith("@@"):
        return "cyan"
    return "dim"


def _progress_label(snapshot: DashboardSnapshot) -> str:
    if snapshot.current_step == 0:
        step = "Waiting for step 1"
    elif snapshot.terminal:
        step = f"{snapshot.current_step} step(s) complete"
    else:
        step = f"Step {snapshot.current_step} active"
    return f"{step} | {snapshot.tools_finished}/{snapshot.tools_started} tool(s) finished"


def _phase_style(phase: str) -> str:
    if phase == "FAILED":
        return "bold red"
    if phase == "COMPLETED":
        return "bold green"
    return "bold cyan"


def _verification_style(status: str) -> str:
    if status in {"verified", "passed"}:
        return "green"
    if status == "failed":
        return "red"
    return "yellow"


def _outcome_explanation(snapshot: DashboardSnapshot) -> str:
    if snapshot.outcome == "VERIFIED":
        return "Current trusted verification evidence passed."
    if snapshot.run_failed:
        return "The run stopped before trustworthy completion."
    explanations = {
        "checks_only": (
            "Configured checks passed, but the task completion contract is incomplete."
        ),
        "missing": "No current trusted verification evidence was recorded.",
        "failed": "The latest trusted verification evidence failed.",
        "stale": "The workspace changed after the latest passing evidence.",
    }
    return explanations.get(
        snapshot.verification_status,
        "The final response is not backed by current trusted evidence.",
    )
