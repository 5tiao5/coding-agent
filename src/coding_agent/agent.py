"""The project-owned model/tool/observation loop."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

from coding_agent.context import ContextError, ContextManager
from coding_agent.events import EventKind, EventSink, NullEventSink, RunEvent
from coding_agent.model import ModelAdapter
from coding_agent.models import (
    AgentResult,
    AgentState,
    ChatMessage,
    MessageRole,
    StopReason,
    ToolCall,
    ToolExecution,
)
from coding_agent.session import (
    LoadedSession,
    SessionBoundary,
    SessionCheckpoint,
    SessionError,
    SessionStore,
)
from coding_agent.stopping import RepeatedToolCallGuard
from coding_agent.tooling import ToolDispatcher
from coding_agent.verification import VerificationLedger

DEFAULT_SYSTEM_PROMPT = """You are a coding agent. Use the available local tools when needed.
Keep plans and action summaries explicit, but never provide hidden reasoning or chain of thought.
After changing files, run a recognized test, build, or check before giving a concise final answer.
Runtime evidence, not your textual claim, determines whether the result is verified."""

_ALLOWED_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.CREATED: {AgentState.PLANNING},
    AgentState.PLANNING: {AgentState.ACTING},
    AgentState.ACTING: {
        AgentState.OBSERVING,
        AgentState.VERIFYING,
        AgentState.FAILED,
    },
    AgentState.OBSERVING: {AgentState.ACTING, AgentState.FAILED},
    AgentState.VERIFYING: {
        AgentState.COMPLETED,
        AgentState.COMPLETED_UNVERIFIED,
        AgentState.FAILED,
    },
}


class AgentRunner:
    def __init__(
        self,
        model: ModelAdapter,
        tools: ToolDispatcher,
        *,
        event_sink: EventSink | None = None,
        max_steps: int = 20,
        max_tool_calls_per_step: int = 8,
        max_total_tool_calls: int = 40,
        max_repeated_tool_results: int = 3,
        context_manager: ContextManager | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if max_tool_calls_per_step < 1 or max_total_tool_calls < 1:
            raise ValueError("tool call limits must be at least 1")
        if max_repeated_tool_results < 2:
            raise ValueError("max_repeated_tool_results must be at least 2")
        self._model = model
        self._tools = tools
        self._events = event_sink or NullEventSink()
        self._max_steps = max_steps
        self._max_tool_calls_per_step = max_tool_calls_per_step
        self._max_total_tool_calls = max_total_tool_calls
        self._max_repeated_tool_results = max_repeated_tool_results
        self._context = context_manager or ContextManager(max_chars=80_000)
        self._session_store = session_store

    def run(self, task: str, *, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> AgentResult:
        return self._run(task, system_prompt=system_prompt, resumed=None)

    def resume(self, loaded: LoadedSession) -> AgentResult:
        """Continue one passive ready-for-model checkpoint without replaying tools."""

        checkpoint = loaded.checkpoint
        if checkpoint.stop_boundary is not SessionBoundary.READY_FOR_MODEL:
            raise ValueError("only ready_for_model checkpoints can be resumed")
        return self._run(
            checkpoint.task,
            system_prompt=checkpoint.system_prompt,
            resumed=loaded,
        )

    def _run(
        self,
        task: str,
        *,
        system_prompt: str,
        resumed: LoadedSession | None,
    ) -> AgentResult:
        if not task.strip():
            raise ValueError("task cannot be empty")

        if resumed is None:
            run_id = uuid4().hex
            state = AgentState.CREATED
            messages: list[ChatMessage] = [
                ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
                ChatMessage(role=MessageRole.USER, content=task),
            ]
            # The full task already lives in conversation state; duplicating it into the
            # observable event stream would increase the chance of leaking source or secrets.
            self._emit(
                run_id,
                EventKind.RUN_STARTED,
                "Run started",
                data={"task_chars": len(task)},
            )
            state = self._transition(run_id, state, AgentState.PLANNING)
            total_tool_calls = 0
            first_step = 1
        else:
            checkpoint = resumed.checkpoint
            run_id = checkpoint.run_id
            state = AgentState.OBSERVING
            messages = list(checkpoint.messages)
            total_tool_calls = checkpoint.completed_tool_calls
            first_step = checkpoint.completed_steps + 1
            self._emit(
                run_id,
                EventKind.RUN_RESUMED,
                "Run resumed from a stable checkpoint; fresh verification is required",
                checkpoint.completed_steps,
                {
                    "completed_steps": checkpoint.completed_steps,
                    "completed_tool_calls": checkpoint.completed_tool_calls,
                    "requires_reverification": resumed.requires_reverification,
                    "auto_replay_tool_calls": resumed.auto_replay_tool_calls,
                },
            )
        verification = VerificationLedger()
        repetition = RepeatedToolCallGuard(max_identical=self._max_repeated_tool_results)

        for step in range(first_step, self._max_steps + 1):
            state = self._transition(run_id, state, AgentState.ACTING, step)
            tool_specs = tuple(self._tools.specs())
            try:
                prepared = self._context.prepare(messages, tool_specs)
            except ContextError as exc:
                return self._failure(
                    run_id,
                    state,
                    StopReason.CONTEXT_LIMIT,
                    step,
                    messages,
                    f"Model context could not be prepared: {exc}",
                )
            if prepared.metadata.compacted:
                self._emit(
                    run_id,
                    EventKind.CONTEXT_COMPACTED,
                    f"Compacted {prepared.metadata.compacted_blocks} older tool blocks",
                    step,
                    {
                        "original_chars": prepared.metadata.original,
                        "prepared_chars": prepared.metadata.prepared,
                        "compacted_blocks": prepared.metadata.compacted_blocks,
                    },
                )
            self._emit(run_id, EventKind.MODEL_REQUESTED, "Model requested", step)

            try:
                response = self._model.complete(prepared.model_view, tool_specs)
            except KeyboardInterrupt:
                return self._failure(
                    run_id,
                    state,
                    StopReason.USER_INTERRUPTED,
                    step,
                    messages,
                    "Run interrupted by user",
                )
            except Exception as exc:  # noqa: BLE001 - adapter errors become terminal run results.
                return self._failure(
                    run_id,
                    state,
                    StopReason.MODEL_ERROR,
                    step,
                    messages,
                    f"Model request failed: {exc}",
                )

            self._emit(
                run_id,
                EventKind.MODEL_RESPONDED,
                "Model responded",
                step,
                {"tool_count": len(response.tool_calls), "has_content": bool(response.content)},
            )
            messages.append(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=response.content,
                    tool_calls=response.tool_calls,
                )
            )

            if response.tool_calls:
                requested_calls = len(response.tool_calls)
                if requested_calls > self._max_tool_calls_per_step:
                    _append_cancelled_tool_results(
                        messages,
                        response.tool_calls,
                        error_code="tool_batch_rejected",
                        error_message="tool batch exceeded the per-step call limit",
                    )
                    return self._failure(
                        run_id,
                        state,
                        StopReason.TOOL_LIMIT,
                        step,
                        messages,
                        "Model exceeded the per-step tool call limit: "
                        f"{requested_calls} > {self._max_tool_calls_per_step}",
                    )
                if total_tool_calls + requested_calls > self._max_total_tool_calls:
                    _append_cancelled_tool_results(
                        messages,
                        response.tool_calls,
                        error_code="tool_batch_rejected",
                        error_message="tool batch exceeded the total call limit",
                    )
                    return self._failure(
                        run_id,
                        state,
                        StopReason.TOOL_LIMIT,
                        step,
                        messages,
                        "Model exceeded the total tool call limit: "
                        f"{total_tool_calls + requested_calls} > {self._max_total_tool_calls}",
                    )
                total_tool_calls += requested_calls
                state = self._transition(run_id, state, AgentState.OBSERVING, step)
                for call_index, call in enumerate(response.tool_calls):
                    self._emit(
                        run_id,
                        EventKind.TOOL_STARTED,
                        f"Running tool: {call.name}",
                        step,
                        {"call_id": call.id, "tool_name": call.name},
                    )
                    try:
                        execution = self._tools.execute(call)
                    except KeyboardInterrupt:
                        _append_cancelled_tool_results(
                            messages,
                            response.tool_calls[call_index:],
                            error_code="tool_call_cancelled",
                            error_message="run was interrupted during tool execution",
                        )
                        return self._failure(
                            run_id,
                            state,
                            StopReason.USER_INTERRUPTED,
                            step,
                            messages,
                            "Run interrupted by user during tool execution",
                        )
                    messages.append(
                        ChatMessage(
                            role=MessageRole.TOOL,
                            content=execution.as_message_content(),
                            tool_call_id=call.id,
                            tool_name=call.name,
                        )
                    )
                    self._emit(
                        run_id,
                        EventKind.TOOL_FINISHED,
                        f"Tool {'succeeded' if execution.ok else 'failed'}: {call.name}",
                        step,
                        {
                            "call_id": call.id,
                            "tool_name": call.name,
                            "ok": execution.ok,
                            "error_code": execution.error_code,
                            "duration_ms": execution.duration_ms,
                            "output_chars": len(execution.output or ""),
                            "truncated": execution.truncated,
                            "summary": execution.summary,
                        },
                    )
                    self._record_control_facts(run_id, execution, verification, step)
                    if execution.control.terminal_stop:
                        assert execution.control.terminal_reason is not None
                        _append_cancelled_tool_results(
                            messages,
                            response.tool_calls[call_index + 1 :],
                            error_code="tool_call_cancelled",
                            error_message="an earlier tool forced a terminal safety stop",
                        )
                        return self._failure(
                            run_id,
                            state,
                            StopReason.COMMAND_CONTROL_FAILED,
                            step,
                            messages,
                            execution.control.terminal_reason,
                        )
                    repeated = repetition.observe(call, execution)
                    if repeated.should_stop:
                        _append_cancelled_tool_results(
                            messages,
                            response.tool_calls[call_index + 1 :],
                            error_code="tool_call_cancelled",
                            error_message="the repeated-call stop policy cancelled this call",
                        )
                        return self._failure(
                            run_id,
                            state,
                            StopReason.REPEATED_TOOL_CALL,
                            step,
                            messages,
                            "The same tool call produced the same observation "
                            f"{repeated.streak} consecutive times",
                        )
                self._save_checkpoint(
                    run_id,
                    task,
                    system_prompt,
                    messages,
                    SessionBoundary.READY_FOR_MODEL,
                    step,
                )
                continue

            state = self._transition(run_id, state, AgentState.VERIFYING, step)
            verification_report = verification.report()
            self._emit(
                run_id,
                EventKind.VERIFICATION_EVALUATED,
                (
                    "Current verification evidence passed"
                    if verification_report.verified
                    else f"Verification evidence is {verification_report.status.value}"
                ),
                step,
                verification_report.event_data(),
            )
            terminal_state = (
                AgentState.COMPLETED
                if verification_report.verified
                else AgentState.COMPLETED_UNVERIFIED
            )
            state = self._transition(run_id, state, terminal_state, step)
            self._emit(
                run_id,
                EventKind.RUN_FINISHED,
                (
                    "Run ended with current verification evidence"
                    if verification_report.verified
                    else "Run ended with an unverified final response"
                ),
                step,
                verification_report.event_data(),
            )
            self._save_checkpoint(
                run_id,
                task,
                system_prompt,
                messages,
                SessionBoundary.TERMINAL,
                step,
                stop_reason=StopReason.FINAL_RESPONSE,
            )
            return AgentResult(
                run_id=run_id,
                state=state,
                stop_reason=StopReason.FINAL_RESPONSE,
                steps=step,
                final_text=response.content,
                messages=tuple(messages),
            )

        return self._failure(
            run_id,
            state,
            StopReason.MAX_STEPS,
            self._max_steps,
            messages,
            f"Maximum step count reached: {self._max_steps}",
        )

    def _save_checkpoint(
        self,
        run_id: str,
        task: str,
        system_prompt: str,
        messages: Sequence[ChatMessage],
        boundary: SessionBoundary,
        step: int,
        *,
        stop_reason: StopReason | None = None,
    ) -> None:
        if self._session_store is None:
            return
        try:
            checkpoint = SessionCheckpoint(
                run_id=run_id,
                task=task,
                system_prompt=system_prompt,
                messages=tuple(messages),
                completed_steps=sum(message.role is MessageRole.ASSISTANT for message in messages),
                completed_tool_calls=sum(message.role is MessageRole.TOOL for message in messages),
                stop_boundary=boundary,
                stop_reason=stop_reason,
            )
            self._session_store.save(checkpoint)
        except (SessionError, ValueError) as exc:
            error_code = exc.code if isinstance(exc, SessionError) else "checkpoint_invalid"
            self._emit(
                run_id,
                EventKind.SESSION_CHECKPOINT_FAILED,
                f"Session checkpoint was not saved: {error_code}",
                step,
                {"error_code": error_code, "boundary": boundary.value},
            )
            return
        self._emit(
            run_id,
            EventKind.SESSION_CHECKPOINTED,
            f"Session checkpoint saved at {boundary.value}",
            step,
            {"boundary": boundary.value},
        )

    def _record_control_facts(
        self,
        run_id: str,
        execution: ToolExecution,
        verification: VerificationLedger,
        step: int,
    ) -> None:
        facts = execution.control
        verification.observe(execution, step=step)
        if facts.invalidates_verification:
            self._emit(
                run_id,
                EventKind.VERIFICATION_INVALIDATED,
                "Previous verification evidence was invalidated",
                step,
                {"call_id": execution.call_id, "epoch": verification.epoch},
            )
        if facts.verification is not None:
            assert facts.verification_kind is not None
            assert facts.verification_label is not None
            self._emit(
                run_id,
                EventKind.VERIFICATION_RECORDED,
                f"Recorded {facts.verification_kind.value} evidence: {facts.verification.value}",
                step,
                {
                    "call_id": execution.call_id,
                    "epoch": verification.epoch,
                    "kind": facts.verification_kind.value,
                    "label": facts.verification_label,
                    "passed": facts.verification.value == "passed",
                },
            )

    def _transition(
        self,
        run_id: str,
        previous: AgentState,
        current: AgentState,
        step: int = 0,
    ) -> AgentState:
        allowed = _ALLOWED_TRANSITIONS.get(previous, set())
        if current not in allowed:
            raise RuntimeError(
                f"invalid agent state transition: {previous.value} -> {current.value}"
            )
        self._emit(
            run_id,
            EventKind.STATE_CHANGED,
            f"State: {previous.value} -> {current.value}",
            step,
            {"previous": previous.value, "current": current.value},
        )
        return current

    def _failure(
        self,
        run_id: str,
        state: AgentState,
        reason: StopReason,
        steps: int,
        messages: Sequence[ChatMessage],
        error: str,
    ) -> AgentResult:
        failed_state = self._transition(run_id, state, AgentState.FAILED, steps)
        self._emit(
            run_id,
            EventKind.RUN_FAILED,
            error,
            steps,
            {"stop_reason": reason.value},
        )
        return AgentResult(
            run_id=run_id,
            state=failed_state,
            stop_reason=reason,
            steps=steps,
            error=error,
            messages=tuple(messages),
        )

    def _emit(
        self,
        run_id: str,
        kind: EventKind,
        message: str,
        step: int = 0,
        data: dict[str, object] | None = None,
    ) -> None:
        self._events.emit(
            RunEvent(
                run_id=run_id,
                kind=kind,
                message=message,
                step=step,
                data=data or {},
            )
        )


def _append_cancelled_tool_results(
    messages: list[ChatMessage],
    calls: Sequence[ToolCall],
    *,
    error_code: str,
    error_message: str,
) -> None:
    for call in calls:
        execution = ToolExecution(
            call_id=call.id,
            tool_name=call.name,
            ok=False,
            error_code=error_code,
            error_message=error_message,
        )
        messages.append(
            ChatMessage(
                role=MessageRole.TOOL,
                content=execution.as_message_content(),
                tool_call_id=call.id,
                tool_name=call.name,
            )
        )
