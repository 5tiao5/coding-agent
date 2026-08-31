"""Application adapters used by the optional local Web presentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from coding_agent.application import RepositoryRunSpec, execute_repository_run
from coding_agent.cancellation import CancellationSource, CancellationToken
from coding_agent.command import CommandPermissionMode
from coding_agent.demo import DEMO_TASK, run_repository_demo
from coding_agent.events import BestEffortEventSink, EventSink
from coding_agent.lease import RunLease
from coding_agent.models import AgentResult, AgentState
from coding_agent.openai_model import ReasoningEffort
from coding_agent.state import StatePaths
from coding_agent.web.service import WebRunService
from coding_agent.web.workbench import WebWorkbench, WebWorkbenchConfig


@dataclass(frozen=True, slots=True)
class WebRepositoryConfig:
    """Server-owned repository settings that the browser cannot override."""

    root: Path
    model_name: str
    base_url: str | None
    reasoning_effort: ReasoningEffort | None
    permission_mode: CommandPermissionMode
    paths: StatePaths
    max_steps: int
    model_timeout: float


def create_demo_service() -> WebRunService:
    """Build a deterministic service used for offline UI rehearsal."""

    def run_demo(run_id: str, task: str, event_sink: EventSink) -> AgentResult:
        if task != DEMO_TASK:
            raise ValueError("the offline demo task is fixed")
        with TemporaryDirectory(prefix="coding-agent-web-demo-") as temporary_directory:
            outcome = run_repository_demo(
                Path(temporary_directory),
                event_sink=BestEffortEventSink(event_sink),
                run_id=run_id,
            )
            if outcome.result.state is AgentState.COMPLETED and not outcome.source_matches_expected:
                raise RuntimeError("verified demo did not produce the expected repository state")
            return outcome.result

    return WebRunService(run_demo)


def create_repository_service(config: WebRepositoryConfig) -> WebRunService:
    """Build a one-worker service backed by the shared repository application layer."""

    def run_repository(
        run_id: str,
        task: str,
        event_sink: EventSink,
        cancellation_token: CancellationToken,
    ) -> AgentResult:
        spec = RepositoryRunSpec(
            run_id=run_id,
            task=task,
            root=config.root,
            model_name=config.model_name,
            base_url=config.base_url,
            reasoning_effort=config.reasoning_effort,
            permission_mode=config.permission_mode,
            paths=config.paths,
            max_steps=config.max_steps,
            model_timeout=config.model_timeout,
        )
        with RunLease(config.paths.root / "leases", run_id):
            # The first presentation milestone is deliberately fail-closed: safe-mode
            # ordinary commands are denied until a Web approval broker is implemented.
            return execute_repository_run(
                spec,
                event_sink=event_sink,
                approver=None,
                cancellation_token=cancellation_token,
            )

    return WebRunService(run_repository, cancellation_source=CancellationSource())


def create_workbench_service(
    config: WebWorkbenchConfig,
    *,
    initial_root: Path | None = None,
) -> WebWorkbench:
    """Build the project-aware host and optionally register one CLI-selected root."""

    workbench = WebWorkbench(config)
    if initial_root is not None:
        workbench.register_project(root=initial_root)
    return workbench
