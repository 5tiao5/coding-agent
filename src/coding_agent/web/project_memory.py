"""Project-memory coordination for the local Web workbench.

The durable store lives in :mod:`coding_agent.project_memory`.  This module owns
the Web-specific orchestration around that store: workspace binding, context
selection, provenance projection, terminal-run persistence, and presentation-safe
text normalization.  Keeping those responsibilities here leaves the workbench in
charge of run lifecycle rather than project-memory policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from coding_agent._presentation_safety import redact_credential_values
from coding_agent.models import AgentResult, AgentState
from coding_agent.project_memory import (
    ProjectMemoryEntry,
    ProjectMemoryError,
    ProjectMemoryStore,
    ProjectTaskStatus,
)
from coding_agent.run_catalog import RunCatalog, RunCatalogError, RunRecord
from coding_agent.run_memory import RunMemorySnapshot
from coding_agent.session import SessionError, SessionStore
from coding_agent.state import StatePaths
from coding_agent.web.service import (
    ProjectMemoryContextPayload,
    ProjectMemorySourcePayload,
    WebRunStatePayload,
)


@dataclass(frozen=True, slots=True)
class ProjectMemoryScope:
    """Immutable project identity used at every memory persistence boundary."""

    project_id: str
    workspace_root: Path
    workspace_fingerprint: str


@dataclass(frozen=True, slots=True)
class ProjectMemoryIndex:
    """Opaque validated entries loaded once for a run-list projection."""

    entries: tuple[ProjectMemoryEntry, ...] = ()


@dataclass(frozen=True, slots=True)
class ProjectMemoryLaunch:
    """One immutable context selection captured before a worker starts."""

    context_text: str | None
    payload: ProjectMemoryContextPayload


class ProjectMemoryCoordinator:
    """Coordinate bounded project memory without affecting Agent run outcomes."""

    def __init__(self, paths: StatePaths, run_catalog: RunCatalog) -> None:
        self._paths = paths
        self._catalog = run_catalog

    def prepare_launch(
        self,
        scope: ProjectMemoryScope,
        task: str,
        *,
        requested: bool,
        parent_run_id: str | None = None,
    ) -> ProjectMemoryLaunch:
        """Select one bounded immutable context view before the worker starts.

        Ordinary project memory is optional and fails closed.  A parent run is an
        explicit follow-up dependency, so it is injected even when broader project
        memory is disabled and cannot silently disappear when unavailable.
        """

        if type(requested) is not bool:
            raise ValueError("use_project_memory must be a boolean")
        if not requested and parent_run_id is None:
            return self.empty_launch(requested=False)
        try:
            prepared = self._store(scope).prepare_context(
                task[:8_000],
                limit=6 if requested else 1,
                preferred_run_ids=(parent_run_id,) if parent_run_id is not None else (),
            )
        except (ProjectMemoryError, OSError, TypeError, ValueError) as exc:
            if parent_run_id is not None:
                if isinstance(exc, ProjectMemoryError):
                    raise
                raise ProjectMemoryError(
                    "project_memory_parent_unavailable",
                    "parent project-memory source is unavailable",
                ) from exc
            return self.empty_launch(
                requested=True,
                error="项目记忆暂时不可用，本轮未注入历史摘要",
            )
        if parent_run_id is not None:
            parent_entry = next(
                (entry for entry in prepared.entries if entry.run_id == parent_run_id),
                None,
            )
            if parent_entry is None:  # pragma: no cover - preferred-source invariant.
                raise ProjectMemoryError(
                    "project_memory_parent_unavailable",
                    "parent project-memory source is unavailable",
                )
            if parent_entry.final_status == "failed":
                raise ProjectMemoryError(
                    "project_memory_parent_failed",
                    "parent project-memory source records a failed task",
                )
        payload = ProjectMemoryContextPayload(
            requested=requested,
            applied=bool(prepared.text),
            source_run_ids=list(prepared.source_run_ids),
            sources=[_source_payload(entry) for entry in prepared.entries],
            error=None,
        )
        return ProjectMemoryLaunch(
            context_text=prepared.text or None,
            payload=payload,
        )

    def load_index(self, scope: ProjectMemoryScope) -> ProjectMemoryIndex:
        """Load validated entries for provenance display, failing closed to empty."""

        try:
            entries = self._store(scope).load().entries
        except (ProjectMemoryError, OSError, ValueError):
            return ProjectMemoryIndex()
        return ProjectMemoryIndex(entries=entries)

    def payload_for_record(
        self,
        record: RunRecord,
        *,
        scope: ProjectMemoryScope,
        index: ProjectMemoryIndex,
    ) -> ProjectMemoryContextPayload:
        """Rebuild auditable provenance without exposing durable memory text."""

        entries_by_run_id = {entry.run_id: entry for entry in index.entries}
        sources: list[ProjectMemorySourcePayload] = []
        for source_run_id in record.memory_source_run_ids:
            entry = entries_by_run_id.get(source_run_id)
            if entry is not None:
                sources.append(_source_payload(entry))
                continue
            try:
                source_record = self._catalog.get(source_run_id)
            except (RunCatalogError, ValueError):
                sources.append(
                    ProjectMemorySourcePayload(
                        run_id=source_run_id,
                        task="历史任务摘要已被清理",
                        completed_at=None,
                    )
                )
                continue
            if (
                source_record.project_id != scope.project_id
                or source_record.workspace_fingerprint != scope.workspace_fingerprint
            ):
                continue
            sources.append(
                ProjectMemorySourcePayload(
                    run_id=source_run_id,
                    task=source_record.task_title,
                    completed_at=None,
                )
            )
        return ProjectMemoryContextPayload(
            requested=record.memory_requested,
            applied=bool(record.memory_source_run_ids),
            source_run_ids=list(record.memory_source_run_ids),
            sources=sources,
            error=None,
        )

    def remember_run(
        self,
        scope: ProjectMemoryScope,
        *,
        task: str,
        result: AgentResult,
    ) -> None:
        """Best-effort persist allowlisted terminal facts without altering the result."""

        try:
            loaded = SessionStore(
                self._paths.sessions,
                workspace_root=scope.workspace_root,
            ).load(result.run_id)
            run_memory = loaded.checkpoint.run_memory
        except (OSError, SessionError, ValueError):
            run_memory = RunMemorySnapshot(revision=0)

        final_status: ProjectTaskStatus
        if result.state is AgentState.COMPLETED:
            final_status = "completed"
        elif result.state is AgentState.COMPLETED_UNVERIFIED:
            final_status = "completed_unverified"
        else:
            final_status = "failed"
        summary = _safe_memory_text(
            result.final_text or result.error or "任务结束，但没有可保存的自然语言摘要。",
            limit=8_000,
        )
        unresolved = (
            (_safe_memory_text(summary, limit=1_000),) if result.state is AgentState.FAILED else ()
        )
        try:
            self._store(scope).remember_run(
                run_id=result.run_id,
                task_goal=_safe_memory_text(task, limit=4_000),
                final_status=final_status,
                final_summary=summary,
                run_memory=run_memory,
                unresolved_issues=unresolved,
            )
        except (ProjectMemoryError, OSError, ValueError):
            return

    @staticmethod
    def empty_launch(
        *,
        requested: bool,
        error: str | None = None,
    ) -> ProjectMemoryLaunch:
        return ProjectMemoryLaunch(
            context_text=None,
            payload=_empty_payload(requested=requested, error=error),
        )

    @staticmethod
    def attach_to_state(
        state: WebRunStatePayload,
        payload: ProjectMemoryContextPayload,
    ) -> WebRunStatePayload:
        enriched = state.copy()
        enriched["memory_context"] = payload
        return enriched

    def _store(self, scope: ProjectMemoryScope) -> ProjectMemoryStore:
        return ProjectMemoryStore(
            self._paths.project_memories,
            project_id=scope.project_id,
            workspace_root=scope.workspace_root,
            workspace_fingerprint_value=scope.workspace_fingerprint,
        )


def _empty_payload(
    *,
    requested: bool,
    error: str | None = None,
) -> ProjectMemoryContextPayload:
    return ProjectMemoryContextPayload(
        requested=requested,
        applied=False,
        source_run_ids=[],
        sources=[],
        error=error,
    )


def _source_payload(entry: ProjectMemoryEntry) -> ProjectMemorySourcePayload:
    return ProjectMemorySourcePayload(
        run_id=entry.run_id,
        task=_safe_memory_text(entry.task_goal, limit=240),
        completed_at=entry.completed_at.isoformat(),
    )


def _safe_memory_text(value: str, *, limit: int) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    printable = "".join(
        character if character == "\n" or character.isprintable() else " "
        for character in normalized
    )
    redacted, _ = redact_credential_values(printable)
    safe = redacted.replace(
        "Verification policy fingerprint:",
        "Verification policy fingerprint [remembered text]:",
    ).strip()
    if not safe:
        safe = "任务结束，但没有可保存的文本摘要。"
    if len(safe) <= limit:
        return safe
    return safe[: max(1, limit - 1)].rstrip() + "…"
