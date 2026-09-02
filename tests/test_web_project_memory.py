"""Focused tests for Web-side project-memory coordination."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from coding_agent.models import VerificationKind
from coding_agent.project_memory import (
    ProjectMemoryEntry,
    ProjectMemoryError,
    ProjectMemoryStore,
    ProjectTaskStatus,
    ProjectVerification,
)
from coding_agent.run_catalog import RunCatalog
from coding_agent.session import workspace_fingerprint
from coding_agent.state import StatePaths
from coding_agent.web.project_memory import ProjectMemoryCoordinator, ProjectMemoryScope


def _entry(
    run_id: str,
    *,
    task: str,
    status: ProjectTaskStatus = "completed",
) -> ProjectMemoryEntry:
    from datetime import UTC, datetime

    return ProjectMemoryEntry(
        run_id=run_id,
        task_goal=task,
        final_status=status,
        final_summary=f"{task} 已完成。",
        verifications=(
            ProjectVerification(label="pytest", kind=VerificationKind.TEST, passed=True),
        ),
        completed_at=datetime(2026, 9, 1, 8, 0, tzinfo=UTC),
    )


def test_project_memory_coordinator_rejects_non_boolean_requests_and_fails_closed(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Deliberately overlap private state with the editable workspace.  The durable
    # store rejects this binding; the Web coordinator must convert it into a bounded,
    # auditable no-memory launch rather than failing the Agent run.
    paths = StatePaths((workspace / "state").resolve())
    coordinator = ProjectMemoryCoordinator(paths, RunCatalog(paths.runs))
    scope = ProjectMemoryScope(
        project_id="project-1",
        workspace_root=workspace.resolve(),
        workspace_fingerprint=workspace_fingerprint(workspace),
    )

    with pytest.raises(ValueError, match="must be a boolean"):
        coordinator.prepare_launch(scope, "task", requested=cast(Any, 1))

    launch = coordinator.prepare_launch(scope, "task", requested=True)
    assert launch.context_text is None
    assert launch.payload == {
        "requested": True,
        "applied": False,
        "source_run_ids": [],
        "sources": [],
        "error": "项目记忆暂时不可用，本轮未注入历史摘要",
    }
    assert coordinator.load_index(scope).entries == ()


@pytest.mark.parametrize("status", ["completed", "completed_unverified"])
def test_follow_up_terminal_parent_is_injected_when_broader_memory_is_disabled(
    tmp_path: Path,
    status: ProjectTaskStatus,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    paths = StatePaths((tmp_path / "state").resolve())
    scope = ProjectMemoryScope(
        project_id="project-1",
        workspace_root=workspace.resolve(),
        workspace_fingerprint=workspace_fingerprint(workspace),
    )
    store = ProjectMemoryStore(
        paths.project_memories,
        project_id=scope.project_id,
        workspace_root=scope.workspace_root,
        workspace_fingerprint_value=scope.workspace_fingerprint,
    )
    store.remember(_entry("parent", task="修复路线规划", status=status))
    store.remember(_entry("unrelated", task="整理其他文档"))
    coordinator = ProjectMemoryCoordinator(paths, RunCatalog(paths.runs))

    launch = coordinator.prepare_launch(
        scope,
        "继续改进路线演示",
        requested=False,
        parent_run_id="parent",
    )

    assert launch.context_text is not None
    assert '"run_id":"parent"' in launch.context_text
    assert '"run_id":"unrelated"' not in launch.context_text
    assert launch.payload == {
        "requested": False,
        "applied": True,
        "source_run_ids": ["parent"],
        "sources": [
            {
                "run_id": "parent",
                "task": "修复路线规划",
                "completed_at": "2026-09-01T08:00:00+00:00",
            }
        ],
        "error": None,
    }


def test_follow_up_parent_unavailability_is_never_silently_downgraded(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    paths = StatePaths((tmp_path / "state").resolve())
    coordinator = ProjectMemoryCoordinator(paths, RunCatalog(paths.runs))
    scope = ProjectMemoryScope(
        project_id="project-1",
        workspace_root=workspace.resolve(),
        workspace_fingerprint=workspace_fingerprint(workspace),
    )

    with pytest.raises(ProjectMemoryError) as missing:
        coordinator.prepare_launch(
            scope,
            "继续任务",
            requested=False,
            parent_run_id="missing-parent",
        )
    assert missing.value.code == "project_memory_preferred_source_missing"

    overlapping_paths = StatePaths((workspace / "state").resolve())
    overlapping = ProjectMemoryCoordinator(
        overlapping_paths,
        RunCatalog(overlapping_paths.runs),
    )
    with pytest.raises(ProjectMemoryError) as unsafe:
        overlapping.prepare_launch(
            scope,
            "继续任务",
            requested=True,
            parent_run_id="missing-parent",
        )
    assert unsafe.value.code == "project_memory_state_overlap"


def test_follow_up_parent_rejects_a_failed_memory_entry(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    paths = StatePaths((tmp_path / "state").resolve())
    scope = ProjectMemoryScope(
        project_id="project-1",
        workspace_root=workspace.resolve(),
        workspace_fingerprint=workspace_fingerprint(workspace),
    )
    store = ProjectMemoryStore(
        paths.project_memories,
        project_id=scope.project_id,
        workspace_root=scope.workspace_root,
        workspace_fingerprint_value=scope.workspace_fingerprint,
    )
    store.remember(_entry("stale-parent", task="未成功结束的旧任务", status="failed"))
    coordinator = ProjectMemoryCoordinator(paths, RunCatalog(paths.runs))

    with pytest.raises(ProjectMemoryError) as stale:
        coordinator.prepare_launch(
            scope,
            "继续任务",
            requested=False,
            parent_run_id="stale-parent",
        )

    assert stale.value.code == "project_memory_parent_failed"
