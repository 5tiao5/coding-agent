"""Presentation-neutral application service for one repository agent run.

The CLI and optional local Web console both enter through this module.  Keeping
composition here prevents either presentation adapter from growing its own
version of the Agent loop, verification policy, session semantics, or trace
ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from coding_agent.agent import AgentRunner
from coding_agent.approval import CommandApprover
from coding_agent.command import CommandPermissionMode
from coding_agent.events import BestEffortEventSink, CompositeEventSink, EventSink
from coding_agent.model import ModelAdapter
from coding_agent.models import AgentResult
from coding_agent.openai_model import ReasoningEffort, create_openai_responses_model
from coding_agent.runtime import build_runtime, system_prompt_for
from coding_agent.session import LoadedSession, SessionStore
from coding_agent.state import StatePaths
from coding_agent.trace import JsonlEventSink


@dataclass(frozen=True, slots=True)
class RepositoryRunSpec:
    """Host-selected inputs shared by every repository-run presentation."""

    run_id: str
    task: str
    root: Path
    model_name: str
    base_url: str | None
    reasoning_effort: ReasoningEffort | None
    permission_mode: CommandPermissionMode
    paths: StatePaths
    max_steps: int
    model_timeout: float
    max_model_retries: int = 2


class ModelFactory(Protocol):
    def __call__(
        self,
        *,
        model: str,
        base_url: str | None,
        reasoning_effort: ReasoningEffort | None,
        timeout_seconds: float,
        max_retries: int,
    ) -> ModelAdapter: ...


def execute_repository_run(
    spec: RepositoryRunSpec,
    *,
    event_sink: EventSink | None = None,
    approver: CommandApprover | None = None,
    session_store: SessionStore | None = None,
    loaded: LoadedSession | None = None,
    model_factory: ModelFactory | None = None,
) -> AgentResult:
    """Execute or resume one run while preserving trace-first event ordering."""

    runtime = build_runtime(
        spec.root,
        permission_mode=spec.permission_mode,
        approver=approver,
    )
    store = session_store or SessionStore(spec.paths.sessions, workspace_root=spec.root)
    factory = model_factory or create_openai_responses_model
    model = factory(
        model=spec.model_name,
        base_url=spec.base_url,
        reasoning_effort=spec.reasoning_effort,
        timeout_seconds=spec.model_timeout,
        max_retries=0,
    )
    trace = JsonlEventSink(spec.paths.traces)
    # Durable audit facts remain authoritative.  A broken renderer is disabled
    # without rewriting the Agent outcome or suppressing later trace events.
    sinks: tuple[EventSink, ...] = (
        (trace,) if event_sink is None else (trace, BestEffortEventSink(event_sink))
    )
    runner = AgentRunner(
        model,
        runtime.tools,
        event_sink=CompositeEventSink(*sinks),
        max_steps=spec.max_steps,
        max_model_retries=spec.max_model_retries,
        session_store=store,
        run_memory=runtime.run_memory,
        verification_profile=runtime.verification_profile,
        completion_contract=runtime.completion_contract,
    )
    if loaded is None:
        return runner.run(
            spec.task,
            system_prompt=system_prompt_for(runtime.verification_commands),
            run_id=spec.run_id,
        )
    if loaded.checkpoint.run_id != spec.run_id:
        raise ValueError("loaded checkpoint does not match the leased run ID")
    return runner.resume(loaded)
