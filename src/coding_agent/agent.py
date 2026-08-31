"""The project-owned model/tool/observation loop."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from math import isfinite
from time import sleep
from uuid import uuid4

from coding_agent.budget import BudgetPolicy, BudgetPurpose, BudgetUsage
from coding_agent.completion import (
    CompletionContract,
    VerificationProfile,
    evaluate_completion,
)
from coding_agent.context import ContextError, ContextManager
from coding_agent.events import EventKind, EventSink, NullEventSink, RunEvent
from coding_agent.model import (
    ModelAdapter,
    RecoverableModelResponseError,
    RetryableModelError,
)
from coding_agent.models import (
    AgentResult,
    AgentState,
    ChatMessage,
    MessageRole,
    StopReason,
    ToolCall,
    ToolExecution,
)
from coding_agent.run_id import require_run_id
from coding_agent.run_memory import RunMemory, RunMemoryError
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

_RUNTIME_LIMITS_MARKER = "[CODING_AGENT_RUNTIME_LIMITS]"
_RUNTIME_LIMITS_END_MARKER = "[/CODING_AGENT_RUNTIME_LIMITS]"
_MAX_CONSECUTIVE_TOOL_BATCH_REJECTIONS = 3
_TOOL_BATCH_REJECTED_ERROR_CODE = "tool_batch_rejected"
_PROTOCOL_CORRECTION_MARKER = "[CODING_AGENT_PROTOCOL_CORRECTION]"
_PROTOCOL_CORRECTION_INSTRUCTION = (
    f"\n\n{_PROTOCOL_CORRECTION_MARKER}\n"
    "The previous response was discarded. Return a fresh response and ensure the "
    "arguments field of every function call is a valid JSON object, not an array or scalar."
)
_CLOSEOUT_MARKER = "[CODING_AGENT_CLOSEOUT]"

_PRESENTATION_PREVIEW_TOOLS = frozenset(
    {"replace_text", "undo_change", "update_plan", "write_file"}
)

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
        max_total_tool_calls: int | None = None,
        budget_policy: BudgetPolicy | None = None,
        max_repeated_tool_results: int = 3,
        max_model_retries: int = 2,
        model_retry_base_delay_seconds: float = 0.5,
        model_retry_sleeper: Callable[[float], None] = sleep,
        context_manager: ContextManager | None = None,
        session_store: SessionStore | None = None,
        run_memory: RunMemory | None = None,
        verification_profile: VerificationProfile | None = None,
        completion_contract: CompletionContract | None = None,
    ) -> None:
        if (verification_profile is None) != (completion_contract is None):
            raise ValueError(
                "verification_profile and completion_contract must be provided together"
            )
        if budget_policy is not None and (
            max_steps != 20 or max_tool_calls_per_step != 8 or max_total_tool_calls is not None
        ):
            raise ValueError("budget_policy cannot be combined with legacy budget arguments")
        required_verifier_calls = (
            len(verification_profile.required_labels) if verification_profile is not None else 1
        )
        verifier_batch_width = (
            budget_policy.max_calls_per_turn
            if budget_policy is not None
            else max_tool_calls_per_step
        )
        required_verifier_turns = (
            required_verifier_calls + verifier_batch_width - 1
        ) // verifier_batch_width
        selected_budget = budget_policy or BudgetPolicy(
            max_model_turns=max_steps,
            max_calls_per_turn=max_tool_calls_per_step,
            average_calls_per_turn=min(2, max_tool_calls_per_step),
            verification_turn_reserve=max(2, required_verifier_turns + 1),
            verification_call_reserve=max(1, required_verifier_calls),
        )
        if (
            max_total_tool_calls is not None
            and max_total_tool_calls != selected_budget.max_total_tool_calls
        ):
            raise ValueError(
                "max_total_tool_calls is derived from max_steps; use BudgetPolicy "
                "to configure coupled budgets"
            )
        if max_repeated_tool_results < 2:
            raise ValueError("max_repeated_tool_results must be at least 2")
        if (
            isinstance(max_model_retries, bool)
            or not isinstance(max_model_retries, int)
            or not 0 <= max_model_retries <= 10
        ):
            raise ValueError("max_model_retries must be an integer between 0 and 10")
        if (
            isinstance(model_retry_base_delay_seconds, bool)
            or not isinstance(model_retry_base_delay_seconds, int | float)
            or not isfinite(model_retry_base_delay_seconds)
            or not 0 <= model_retry_base_delay_seconds <= 60
        ):
            raise ValueError("model_retry_base_delay_seconds must be between 0 and 60")
        if (
            verification_profile is not None
            and selected_budget.verification_call_reserve < required_verifier_calls
        ):
            raise ValueError(
                "budget_policy must reserve at least one call for every required verifier"
            )
        if (
            verification_profile is not None
            and selected_budget.verification_turn_reserve < required_verifier_turns + 1
        ):
            raise ValueError(
                "budget_policy must reserve verifier batches and one final-response turn"
            )
        self._model = model
        self._tools = tools
        self._events = event_sink or NullEventSink()
        self._budget = selected_budget
        self._max_repeated_tool_results = max_repeated_tool_results
        self._max_model_retries = max_model_retries
        self._model_retry_base_delay_seconds = float(model_retry_base_delay_seconds)
        self._model_retry_sleeper = model_retry_sleeper
        self._context = context_manager or ContextManager(max_chars=selected_budget.context_chars)
        self._session_store = session_store
        self._memory = run_memory or RunMemory(max_chars=selected_budget.memory_chars)
        self._verification_profile = verification_profile
        self._completion_contract = completion_contract

    def run(
        self,
        task: str,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        run_id: str | None = None,
    ) -> AgentResult:
        """Start a run, optionally using a host-owned ID acquired under an external lease."""

        selected_run_id = uuid4().hex if run_id is None else require_run_id(run_id)
        return self._run(
            task,
            system_prompt=system_prompt,
            resumed=None,
            run_id=selected_run_id,
        )

    def resume(self, loaded: LoadedSession) -> AgentResult:
        """Continue one passive ready-for-model checkpoint without replaying tools."""

        checkpoint = loaded.checkpoint
        if checkpoint.stop_boundary is not SessionBoundary.READY_FOR_MODEL:
            raise ValueError("only ready_for_model checkpoints can be resumed")
        return self._run(
            checkpoint.task,
            system_prompt=checkpoint.system_prompt,
            resumed=loaded,
            run_id=checkpoint.run_id,
        )

    def _run(
        self,
        task: str,
        *,
        system_prompt: str,
        resumed: LoadedSession | None,
        run_id: str,
    ) -> AgentResult:
        if not task.strip():
            raise ValueError("task cannot be empty")

        policy = self._budget
        system_prompt = _with_runtime_limits(
            system_prompt,
            max_model_turns=policy.max_model_turns,
            max_calls_per_turn=policy.max_calls_per_turn,
            max_total_tool_calls=policy.max_total_tool_calls,
        )
        limits = {
            "max_model_turns": policy.max_model_turns,
            "max_calls_per_turn": policy.max_calls_per_turn,
            "max_total_tool_calls": policy.max_total_tool_calls,
        }
        verification_required = self._verification_profile is not None

        if resumed is None:
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
                data={"task_chars": len(task), "limits": limits},
            )
            state = self._transition(run_id, state, AgentState.PLANNING)
            budget_state = policy.bind(
                BudgetUsage(),
                verification_required=verification_required,
            )
            first_step = 1
            # The initial system/user transcript is already a stable boundary. Saving
            # it makes a first-request provider failure resumable without replaying any
            # side effect.
            self._save_checkpoint(
                run_id,
                task,
                system_prompt,
                messages,
                SessionBoundary.READY_FOR_MODEL,
                0,
                budget_usage=budget_state.usage,
            )
        else:
            checkpoint = resumed.checkpoint
            state = AgentState.OBSERVING
            try:
                self._memory.restore(
                    checkpoint.run_memory,
                    mark_verification_stale=False,
                )
            except (RunMemoryError, ValueError) as exc:
                error_code = exc.code if isinstance(exc, RunMemoryError) else "invalid_snapshot"
                return self._failure(
                    run_id,
                    state,
                    StopReason.MODEL_ERROR,
                    checkpoint.completed_steps,
                    checkpoint.messages,
                    f"Run memory could not be restored: {error_code}",
                )
            messages = list(checkpoint.messages)
            messages[0] = messages[0].model_copy(update={"content": system_prompt})
            resumed_usage = BudgetUsage(
                model_turns=checkpoint.completed_steps,
                work_calls=getattr(
                    checkpoint,
                    "completed_work_tool_calls",
                    checkpoint.completed_tool_calls,
                ),
                verification_calls=getattr(
                    checkpoint,
                    "completed_verification_tool_calls",
                    0,
                ),
            )
            try:
                budget_state = policy.bind(
                    resumed_usage,
                    verification_required=verification_required,
                )
            except ValueError:
                reason = (
                    StopReason.MAX_STEPS
                    if resumed_usage.model_turns > policy.max_model_turns
                    else StopReason.TOOL_LIMIT
                )
                return self._failure(
                    run_id,
                    state,
                    reason,
                    checkpoint.completed_steps,
                    messages,
                    "Resume checkpoint exceeds the selected run budget; increase "
                    "--max-steps before resuming",
                )
            first_step = checkpoint.completed_steps + 1
            self._emit(
                run_id,
                EventKind.RUN_RESUMED,
                "Run resumed from a stable checkpoint; fresh verification is required",
                checkpoint.completed_steps,
                {
                    "completed_steps": checkpoint.completed_steps,
                    "completed_tool_calls": checkpoint.completed_tool_calls,
                    "completed_work_tool_calls": budget_state.usage.work_calls,
                    "completed_verification_tool_calls": (budget_state.usage.verification_calls),
                    "requires_reverification": resumed.requires_reverification,
                    "auto_replay_tool_calls": resumed.auto_replay_tool_calls,
                    "limits": limits,
                },
            )
        verification = VerificationLedger()
        repetition = RepeatedToolCallGuard(max_identical=self._max_repeated_tool_results)
        seen_tool_call_ids = {
            call.id
            for message in messages
            if message.role is MessageRole.ASSISTANT
            for call in message.tool_calls
        }
        consecutive_batch_rejections = _trailing_tool_batch_rejections(messages)
        closeout_mode = False

        for step in range(first_step, policy.max_model_turns + 1):
            turn_purpose: BudgetPurpose = (
                _closeout_turn_purpose(
                    budget_state.remaining_model_turns,
                    verification_required=_requires_current_verification(
                        verification,
                        self._verification_profile,
                        self._completion_contract,
                    ),
                )
                if closeout_mode
                else "work"
            )
            turn_admission = budget_state.admit_turn(turn_purpose)
            if not turn_admission.accepted:
                closeout_mode = True
                turn_purpose = _closeout_turn_purpose(
                    budget_state.remaining_model_turns,
                    verification_required=_requires_current_verification(
                        verification,
                        self._verification_profile,
                        self._completion_contract,
                    ),
                )
                turn_admission = budget_state.admit_turn(turn_purpose)
            if not turn_admission.accepted:
                return self._failure(
                    run_id,
                    state,
                    StopReason.MAX_STEPS,
                    max(0, step - 1),
                    messages,
                    f"Model turn budget is exhausted: {turn_admission.code}",
                )
            budget_state = budget_state.consume_turn(turn_purpose)
            state = self._transition(run_id, state, AgentState.ACTING, step)
            tool_specs = tuple(self._tools.specs())
            try:
                prepared = self._context.prepare(
                    messages,
                    tool_specs,
                    memory=self._memory.snapshot(),
                )
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
            response = None
            request_view = prepared.model_view
            if closeout_mode:
                request_view = _with_closeout_instruction(
                    request_view,
                    purpose=turn_purpose,
                    remaining_turns=budget_state.remaining_model_turns,
                )
            max_attempts = self._max_model_retries + 1
            for attempt in range(1, max_attempts + 1):
                self._emit(
                    run_id,
                    EventKind.MODEL_REQUESTED,
                    "Model requested",
                    step,
                    {"attempt": attempt, "max_attempts": max_attempts},
                )
                try:
                    response = self._model.complete(request_view, tool_specs)
                except KeyboardInterrupt:
                    return self._failure(
                        run_id,
                        state,
                        StopReason.USER_INTERRUPTED,
                        step,
                        messages,
                        "Run interrupted by user",
                    )
                except RetryableModelError:
                    if attempt == max_attempts:
                        return self._failure(
                            run_id,
                            state,
                            StopReason.MODEL_ERROR,
                            step,
                            messages,
                            "Model request failed after transient retries",
                        )
                    delay_seconds = min(
                        self._model_retry_base_delay_seconds * (2 ** (attempt - 1)),
                        60.0,
                    )
                    self._emit(
                        run_id,
                        EventKind.MODEL_RETRYING,
                        "Retrying model after a transient failure",
                        step,
                        {
                            "attempt": attempt,
                            "next_attempt": attempt + 1,
                            "max_attempts": max_attempts,
                            "delay_seconds": delay_seconds,
                            "error_code": "model_request_transient",
                            "retry_kind": "transport_backoff",
                        },
                    )
                    try:
                        self._model_retry_sleeper(delay_seconds)
                    except KeyboardInterrupt:
                        return self._failure(
                            run_id,
                            state,
                            StopReason.USER_INTERRUPTED,
                            step,
                            messages,
                            "Run interrupted by user during model retry delay",
                        )
                except RecoverableModelResponseError:
                    if attempt == max_attempts:
                        return self._failure(
                            run_id,
                            state,
                            StopReason.MODEL_ERROR,
                            step,
                            messages,
                            "Model returned invalid tool-call arguments after protocol "
                            "recovery attempts",
                        )
                    try:
                        corrected = self._context.prepare(
                            _with_protocol_correction(messages),
                            tool_specs,
                            memory=self._memory.snapshot(),
                        )
                    except ContextError as exc:
                        return self._failure(
                            run_id,
                            state,
                            StopReason.CONTEXT_LIMIT,
                            step,
                            messages,
                            f"Model response recovery context could not be prepared: {exc}",
                        )
                    request_view = corrected.model_view
                    if closeout_mode:
                        request_view = _with_closeout_instruction(
                            request_view,
                            purpose=turn_purpose,
                            remaining_turns=budget_state.remaining_model_turns,
                        )
                    self._emit(
                        run_id,
                        EventKind.MODEL_RETRYING,
                        "Requesting a corrected model protocol response",
                        step,
                        {
                            "attempt": attempt,
                            "next_attempt": attempt + 1,
                            "max_attempts": max_attempts,
                            "delay_seconds": 0.0,
                            "error_code": "model_response_invalid",
                            "retry_kind": "protocol_correction",
                            "instruction_chars": len(_PROTOCOL_CORRECTION_INSTRUCTION),
                            "prepared_context_chars": corrected.metadata.prepared,
                        },
                    )
                except Exception as exc:  # noqa: BLE001 - adapter errors terminate the run.
                    return self._failure(
                        run_id,
                        state,
                        StopReason.MODEL_ERROR,
                        step,
                        messages,
                        f"Model request failed: {exc}",
                    )
                else:
                    break
            assert response is not None

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
                response_call_ids = [call.id for call in response.tool_calls]
                duplicate_call_id = len(response_call_ids) != len(set(response_call_ids)) or any(
                    call_id in seen_tool_call_ids for call_id in response_call_ids
                )
                if duplicate_call_id:
                    _append_cancelled_tool_results(
                        messages,
                        response.tool_calls,
                        error_code="tool_batch_rejected",
                        error_message="tool call IDs must be unique across the run",
                    )
                    return self._failure(
                        run_id,
                        state,
                        StopReason.MODEL_ERROR,
                        step,
                        messages,
                        "Model returned a duplicate tool call ID",
                    )
                seen_tool_call_ids.update(response_call_ids)
                requested_calls = len(response.tool_calls)
                if requested_calls > policy.max_calls_per_turn:
                    consecutive_batch_rejections += 1
                    _append_cancelled_tool_results(
                        messages,
                        response.tool_calls,
                        error_code=_TOOL_BATCH_REJECTED_ERROR_CODE,
                        error_message=(
                            "tool batch exceeded the per-step call limit "
                            f"({requested_calls} requested, "
                            f"{policy.max_calls_per_turn} allowed); "
                            "retry with a smaller batch"
                        ),
                    )
                    self._emit(
                        run_id,
                        EventKind.TOOL_BATCH_REJECTED,
                        "Rejected an over-limit model tool batch before execution",
                        step,
                        {
                            "requested_calls": requested_calls,
                            "max_calls_per_turn": policy.max_calls_per_turn,
                            "rejection_count": consecutive_batch_rejections,
                            "max_rejections": _MAX_CONSECUTIVE_TOOL_BATCH_REJECTIONS,
                        },
                    )
                    if consecutive_batch_rejections >= _MAX_CONSECUTIVE_TOOL_BATCH_REJECTIONS:
                        return self._failure(
                            run_id,
                            state,
                            StopReason.TOOL_LIMIT,
                            step,
                            messages,
                            "Model exceeded the per-step tool call limit "
                            f"{consecutive_batch_rejections} consecutive times: "
                            f"{requested_calls} > {policy.max_calls_per_turn}",
                        )
                    state = self._transition(run_id, state, AgentState.OBSERVING, step)
                    self._save_checkpoint(
                        run_id,
                        task,
                        system_prompt,
                        messages,
                        SessionBoundary.READY_FOR_MODEL,
                        step,
                        budget_usage=budget_state.usage,
                    )
                    continue
                verification_calls = sum(
                    _is_verification_call(self._tools, call) for call in response.tool_calls
                )
                work_calls = requested_calls - verification_calls
                batch_admission = budget_state.admit_batch(
                    work_calls=work_calls,
                    verification_calls=verification_calls,
                )
                if turn_purpose == "final":
                    rejection_code = "final_turn_requires_response"
                elif turn_purpose == "verification" and work_calls:
                    rejection_code = "verification_turn_requires_verifier"
                elif not batch_admission.accepted:
                    rejection_code = batch_admission.code
                else:
                    rejection_code = None
                observation_turn = None
                if rejection_code is None:
                    try:
                        observation_turn = policy.observation_budget.begin(
                            messages[-1],
                            result_count=requested_calls,
                        )
                    except ValueError:
                        rejection_code = "observation_budget_exceeded"
                if rejection_code is not None:
                    consecutive_batch_rejections += 1
                    _append_cancelled_tool_results(
                        messages,
                        response.tool_calls,
                        error_code=_TOOL_BATCH_REJECTED_ERROR_CODE,
                        error_message=(
                            f"tool batch was not admitted ({rejection_code}); "
                            "return a final response or request an allowed smaller batch"
                        ),
                    )
                    self._emit(
                        run_id,
                        EventKind.TOOL_BATCH_REJECTED,
                        "Rejected a tool batch before execution",
                        step,
                        {
                            "requested_calls": requested_calls,
                            "max_calls_per_turn": policy.max_calls_per_turn,
                            "rejection_count": consecutive_batch_rejections,
                            "max_rejections": _MAX_CONSECUTIVE_TOOL_BATCH_REJECTIONS,
                            "reason": rejection_code,
                            "remaining_total_tool_calls": (budget_state.remaining_total_tool_calls),
                            "remaining_work_tool_calls": (budget_state.remaining_work_tool_calls),
                        },
                    )
                    if consecutive_batch_rejections >= _MAX_CONSECUTIVE_TOOL_BATCH_REJECTIONS:
                        return self._failure(
                            run_id,
                            state,
                            StopReason.TOOL_LIMIT,
                            step,
                            messages,
                            "Model submitted an inadmissible tool batch "
                            f"{consecutive_batch_rejections} consecutive times: "
                            f"{rejection_code}",
                        )
                    closeout_mode = True
                    state = self._transition(run_id, state, AgentState.OBSERVING, step)
                    self._save_checkpoint(
                        run_id,
                        task,
                        system_prompt,
                        messages,
                        SessionBoundary.READY_FOR_MODEL,
                        step,
                        budget_usage=budget_state.usage,
                    )
                    continue
                assert observation_turn is not None
                consecutive_batch_rejections = 0
                budget_state = budget_state.consume_batch(
                    work_calls=work_calls,
                    verification_calls=verification_calls,
                )
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
                        raw_execution = self._tools.execute(call)
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
                    execution = observation_turn.fit(
                        raw_execution,
                        pending=requested_calls - call_index - 1,
                    )
                    messages.append(
                        ChatMessage(
                            role=MessageRole.TOOL,
                            content=execution.as_message_content(),
                            tool_call_id=call.id,
                            tool_name=call.name,
                        )
                    )
                    event_data: dict[str, object] = {
                        "call_id": call.id,
                        "tool_name": call.name,
                        "ok": raw_execution.ok,
                        "error_code": raw_execution.error_code,
                        "duration_ms": raw_execution.duration_ms,
                        "output_chars": len(raw_execution.output or ""),
                        "truncated": raw_execution.truncated,
                        "summary": raw_execution.summary,
                        "observation_chars": len(execution.as_message_content()),
                        "observation_truncated": (
                            execution.as_message_content() != raw_execution.as_message_content()
                        ),
                    }
                    if raw_execution.metadata:
                        event_data["metadata"] = dict(raw_execution.metadata)
                    if (
                        raw_execution.ok
                        and raw_execution.output is not None
                        and call.name in _PRESENTATION_PREVIEW_TOOLS
                    ):
                        # Only explicit plans and bounded mutation diffs are presentation-safe.
                        # Read/search/command output stays in the private canonical transcript.
                        event_data["preview"] = raw_execution.output
                    self._emit(
                        run_id,
                        EventKind.TOOL_FINISHED,
                        f"Tool {'succeeded' if execution.ok else 'failed'}: {call.name}",
                        step,
                        event_data,
                    )
                    self._record_control_facts(run_id, raw_execution, verification, step)
                    self._memory.observe(call, raw_execution, step=step)
                    if raw_execution.control.terminal_stop:
                        assert raw_execution.control.terminal_reason is not None
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
                            raw_execution.control.terminal_reason,
                        )
                    repeated = repetition.observe(call, raw_execution)
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
                    budget_usage=budget_state.usage,
                )
                continue

            state = self._transition(run_id, state, AgentState.VERIFYING, step)
            verification_report = verification.report()
            if self._verification_profile is not None:
                assert self._completion_contract is not None
                completion_report = evaluate_completion(
                    self._verification_profile,
                    self._completion_contract,
                    verification_report,
                )
                verified = completion_report.task_validated
                verification_data = completion_report.event_data(verification_report)
                evaluation_message = (
                    "Task completion contract validated"
                    if verified
                    else f"Completion contract is {completion_report.completion_status.value}"
                )
            else:
                verified = verification_report.verified
                verification_data = verification_report.event_data()
                evaluation_message = (
                    "Current verification evidence passed"
                    if verified
                    else f"Verification evidence is {verification_report.status.value}"
                )
            self._emit(
                run_id,
                EventKind.VERIFICATION_EVALUATED,
                evaluation_message,
                step,
                verification_data,
            )
            terminal_state = AgentState.COMPLETED if verified else AgentState.COMPLETED_UNVERIFIED
            state = self._transition(run_id, state, terminal_state, step)
            self._save_checkpoint(
                run_id,
                task,
                system_prompt,
                messages,
                SessionBoundary.TERMINAL,
                step,
                budget_usage=budget_state.usage,
                stop_reason=StopReason.FINAL_RESPONSE,
            )
            # Keep the terminal event truly terminal. Renderers can now print one final
            # card without a later checkpoint event appearing beneath it or restarting
            # a live display.
            self._emit(
                run_id,
                EventKind.RUN_FINISHED,
                (
                    "Run ended with a validated completion contract"
                    if verified and self._verification_profile is not None
                    else (
                        "Run ended with current verification evidence"
                        if verified
                        else "Run ended with an unverified final response"
                    )
                ),
                step,
                verification_data,
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
            policy.max_model_turns,
            messages,
            f"Maximum step count reached: {policy.max_model_turns}",
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
        budget_usage: BudgetUsage,
        stop_reason: StopReason | None = None,
    ) -> None:
        if self._session_store is None:
            return
        try:
            checkpoint = SessionCheckpoint(
                run_id=run_id,
                workspace_fingerprint=self._session_store.workspace_fingerprint,
                task=task,
                system_prompt=system_prompt,
                messages=tuple(messages),
                completed_steps=sum(message.role is MessageRole.ASSISTANT for message in messages),
                completed_tool_calls=budget_usage.total_tool_calls,
                completed_work_tool_calls=budget_usage.work_calls,
                completed_verification_tool_calls=budget_usage.verification_calls,
                run_memory=self._memory.snapshot(),
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


def _with_runtime_limits(
    system_prompt: str,
    *,
    max_model_turns: int,
    max_calls_per_turn: int,
    max_total_tool_calls: int,
) -> str:
    """Replace the host-owned runtime budget block with current runner limits."""

    base = system_prompt
    marker_index = base.find(_RUNTIME_LIMITS_MARKER)
    if marker_index >= 0:
        end_index = base.find(_RUNTIME_LIMITS_END_MARKER, marker_index)
        if end_index >= 0:
            base = base[:marker_index] + base[end_index + len(_RUNTIME_LIMITS_END_MARKER) :]
    base = base.rstrip()
    return (
        f"{base}\n\n{_RUNTIME_LIMITS_MARKER}\n"
        "Runtime budgets enforced for this run:\n"
        f"- Maximum model turns: {max_model_turns}.\n"
        f"- Maximum tool calls in one model response: {max_calls_per_turn}. "
        "Never exceed this per-turn limit; split independent calls across turns.\n"
        f"- Maximum accepted tool calls across the run: {max_total_tool_calls}.\n"
        "The host reserves enough closing capacity for registered verification and one "
        "honest final response when a completion contract is active.\n"
        "An over-limit per-turn batch is rejected atomically: none of its calls execute. "
        "Use the returned tool errors to retry with a smaller batch.\n"
        f"{_RUNTIME_LIMITS_END_MARKER}"
    )


def _with_protocol_correction(messages: Sequence[ChatMessage]) -> tuple[ChatMessage, ...]:
    """Add one bounded, sanitized retry instruction to a transient request view only."""

    system = messages[0]
    assert system.role is MessageRole.SYSTEM
    assert system.content is not None
    content = f"{system.content.rstrip()}{_PROTOCOL_CORRECTION_INSTRUCTION}"
    return (system.model_copy(update={"content": content}), *messages[1:])


def _with_closeout_instruction(
    messages: Sequence[ChatMessage],
    *,
    purpose: BudgetPurpose,
    remaining_turns: int,
) -> tuple[ChatMessage, ...]:
    """Inject a host-owned closeout instruction into the request view only."""

    system = messages[0]
    assert system.role is MessageRole.SYSTEM
    assert system.content is not None
    if purpose == "verification":
        instruction = (
            "The work-turn budget is exhausted. This turn is reserved for recognized "
            "verification calls only; do not inspect or modify files. Request the smallest "
            "registered verification batch needed to validate the current workspace."
        )
    else:
        instruction = (
            "This is the reserved final-response turn. Do not call tools. Give a concise, "
            "honest summary and explicitly state any missing or failed verification."
        )
    content = (
        f"{system.content.rstrip()}\n\n{_CLOSEOUT_MARKER}\n"
        f"{instruction}\nRemaining model turns after this response: {remaining_turns}."
    )
    return (system.model_copy(update={"content": content}), *messages[1:])


def _requires_current_verification(
    ledger: VerificationLedger,
    profile: VerificationProfile | None,
    contract: CompletionContract | None,
) -> bool:
    if profile is None:
        return False
    assert contract is not None
    return not evaluate_completion(profile, contract, ledger.report()).task_validated


def _closeout_turn_purpose(
    remaining_turns: int,
    *,
    verification_required: bool,
) -> BudgetPurpose:
    if verification_required and remaining_turns > 1:
        return "verification"
    return "final"


def _is_verification_call(tools: ToolDispatcher, call: ToolCall) -> bool:
    classifier = getattr(tools, "is_verification_call", None)
    if classifier is None or not callable(classifier):
        return False
    try:
        return classifier(call) is True
    except (TypeError, ValueError):
        return False


def _trailing_tool_batch_rejections(messages: Sequence[ChatMessage]) -> int:
    """Recover the consecutive rejection streak from a stable transcript."""

    consecutive = 0
    cursor = 2
    while cursor < len(messages):
        assistant = messages[cursor]
        cursor += 1
        if assistant.role is not MessageRole.ASSISTANT or not assistant.tool_calls:
            consecutive = 0
            continue
        results = messages[cursor : cursor + len(assistant.tool_calls)]
        cursor += len(assistant.tool_calls)
        if len(results) == len(assistant.tool_calls) and all(
            _tool_result_error_code(result) == _TOOL_BATCH_REJECTED_ERROR_CODE for result in results
        ):
            consecutive += 1
        else:
            consecutive = 0
    return consecutive


def _tool_result_error_code(message: ChatMessage) -> str | None:
    if message.role is not MessageRole.TOOL or message.content is None:
        return None
    try:
        payload = json.loads(message.content)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("ok") is not False:
        return None
    error_code = payload.get("error_code")
    return error_code if isinstance(error_code, str) else None
