"""The project-owned model/tool/observation loop."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from time import sleep
from uuid import uuid4

from coding_agent import agent_protocol as protocol
from coding_agent.agent_protocol import DEFAULT_SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT
from coding_agent.budget import BudgetPolicy, BudgetPurpose, BudgetUsage
from coding_agent.cancellation import (
    CancellationToken,
    cancellation_requested,
    wait_for_retry_or_cancellation,
)
from coding_agent.completion import (
    CompletionContract,
    VerificationProfile,
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
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        if (verification_profile is None) != (completion_contract is None):
            raise ValueError(
                "verification_profile and completion_contract must be provided together"
            )
        selected_budget = protocol.resolve_budget_policy(
            budget_policy=budget_policy,
            max_steps=max_steps,
            max_tool_calls_per_step=max_tool_calls_per_step,
            max_total_tool_calls=max_total_tool_calls,
            verification_profile=verification_profile,
        )
        if max_repeated_tool_results < 2:
            raise ValueError("max_repeated_tool_results must be at least 2")
        retry_delay = protocol.validate_model_retry_settings(
            max_model_retries,
            model_retry_base_delay_seconds,
        )
        self._model = model
        self._tools = tools
        self._events = event_sink or NullEventSink()
        self._budget = selected_budget
        self._max_repeated_tool_results = max_repeated_tool_results
        self._max_model_retries = max_model_retries
        self._model_retry_base_delay_seconds = retry_delay
        self._model_retry_sleeper = model_retry_sleeper
        self._context = context_manager or ContextManager(max_chars=selected_budget.context_chars)
        self._session_store = session_store
        self._memory = run_memory or RunMemory(max_chars=selected_budget.memory_chars)
        self._verification_profile = verification_profile
        self._completion_contract = completion_contract
        self._cancellation_token = cancellation_token

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
        system_prompt = protocol.with_runtime_limits(
            system_prompt,
            max_model_turns=policy.max_model_turns,
            max_calls_per_turn=policy.max_calls_per_turn,
            max_total_tool_calls=policy.max_total_tool_calls,
        )
        limits = protocol.runtime_limits_event_data(policy)
        verification_required = self._verification_profile is not None

        if resumed is None:
            self._memory.reset()
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

        def interrupt_at_ready_boundary(
            error: str,
            step: int,
            *,
            pending_calls: Sequence[ToolCall] = (),
            tool_error_message: str = "the host requested cooperative cancellation",
        ) -> AgentResult:
            if pending_calls:
                protocol.append_cancelled_tool_results(
                    messages,
                    pending_calls,
                    error_code="tool_call_cancelled",
                    error_message=tool_error_message,
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
            return self._failure(
                run_id,
                state,
                StopReason.USER_INTERRUPTED,
                step,
                messages,
                error,
            )

        verification = VerificationLedger()
        repetition = RepeatedToolCallGuard(max_identical=self._max_repeated_tool_results)
        seen_tool_call_ids = {
            call.id
            for message in messages
            if message.role is MessageRole.ASSISTANT
            for call in message.tool_calls
        }
        consecutive_batch_rejections = protocol.trailing_tool_batch_rejections(messages)
        early_final_corrected = protocol.has_early_final_correction(messages)
        closeout_mode = early_final_corrected

        for step in range(first_step, policy.max_model_turns + 1):
            turn_purpose: BudgetPurpose = (
                protocol.closeout_turn_purpose(
                    budget_state.remaining_model_turns,
                    verification_required=protocol.requires_current_verification(
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
                turn_purpose = protocol.closeout_turn_purpose(
                    budget_state.remaining_model_turns,
                    verification_required=protocol.requires_current_verification(
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
                request_view = protocol.with_closeout_instruction(
                    request_view,
                    purpose=turn_purpose,
                    remaining_turns=budget_state.remaining_model_turns,
                )
            max_attempts = self._max_model_retries + 1
            for attempt in range(1, max_attempts + 1):
                if cancellation_requested(self._cancellation_token):
                    return interrupt_at_ready_boundary(
                        "Run interrupted by host before a model request",
                        step,
                    )
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
                    return interrupt_at_ready_boundary(
                        "Run interrupted by user",
                        step,
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
                        cancelled_during_delay = wait_for_retry_or_cancellation(
                            self._cancellation_token,
                            delay_seconds,
                            self._model_retry_sleeper,
                        )
                    except KeyboardInterrupt:
                        return interrupt_at_ready_boundary(
                            "Run interrupted by user during model retry delay",
                            step,
                        )
                    if cancelled_during_delay:
                        return interrupt_at_ready_boundary(
                            "Run interrupted by host during model retry delay",
                            step,
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
                            protocol.with_protocol_correction(messages),
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
                        request_view = protocol.with_closeout_instruction(
                            request_view,
                            purpose=turn_purpose,
                            remaining_turns=budget_state.remaining_model_turns,
                        )
                    self._emit(
                        run_id,
                        EventKind.MODEL_RETRYING,
                        "Requesting a corrected model protocol response",
                        step,
                        protocol.protocol_correction_retry_event_data(
                            attempt=attempt,
                            max_attempts=max_attempts,
                            prepared_context_chars=corrected.metadata.prepared,
                        ),
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

            # A model call is one atomic external operation. If shutdown arrived while
            # it was blocking, discard the not-yet-committed response and resume later
            # from the preceding stable transcript instead of accepting a late final.
            if cancellation_requested(self._cancellation_token):
                return interrupt_at_ready_boundary(
                    "Run interrupted by host after the model request completed",
                    step,
                )

            self._emit(
                run_id,
                EventKind.MODEL_RESPONDED,
                "Model responded",
                step,
                {"tool_count": len(response.tool_calls), "has_content": bool(response.content)},
            )
            if protocol.should_append_early_final_correction(
                response,
                memory=self._memory.snapshot(),
                ledger=verification,
                profile=self._verification_profile,
                contract=self._completion_contract,
                already_corrected=early_final_corrected,
                remaining_model_turns=budget_state.remaining_model_turns,
            ):
                protocol.append_early_final_correction(messages, response)
                early_final_corrected = True
                closeout_mode = True
                consecutive_batch_rejections = 0
                self._emit(
                    run_id,
                    EventKind.MODEL_RETRYING,
                    "Scheduling one host-required verification closeout turn",
                    step,
                    protocol.early_final_correction_event_data(
                        remaining_model_turns=budget_state.remaining_model_turns
                    ),
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
                    protocol.append_cancelled_tool_results(
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
                    protocol.append_cancelled_tool_results(
                        messages,
                        response.tool_calls,
                        error_code=protocol.TOOL_BATCH_REJECTED_ERROR_CODE,
                        error_message=protocol.over_limit_tool_batch_error_message(
                            requested_calls=requested_calls,
                            max_calls_per_turn=policy.max_calls_per_turn,
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
                            "max_rejections": (protocol.MAX_CONSECUTIVE_TOOL_BATCH_REJECTIONS),
                        },
                    )
                    if (
                        consecutive_batch_rejections
                        >= protocol.MAX_CONSECUTIVE_TOOL_BATCH_REJECTIONS
                    ):
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
                    protocol.is_verification_call(self._tools, call) for call in response.tool_calls
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
                    protocol.append_cancelled_tool_results(
                        messages,
                        response.tool_calls,
                        error_code=protocol.TOOL_BATCH_REJECTED_ERROR_CODE,
                        error_message=protocol.inadmissible_tool_batch_error_message(
                            rejection_code
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
                            "max_rejections": (protocol.MAX_CONSECUTIVE_TOOL_BATCH_REJECTIONS),
                            "reason": rejection_code,
                            "remaining_total_tool_calls": (budget_state.remaining_total_tool_calls),
                            "remaining_work_tool_calls": (budget_state.remaining_work_tool_calls),
                        },
                    )
                    if (
                        consecutive_batch_rejections
                        >= protocol.MAX_CONSECUTIVE_TOOL_BATCH_REJECTIONS
                    ):
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
                    if cancellation_requested(self._cancellation_token):
                        return interrupt_at_ready_boundary(
                            "Run interrupted by host between tool calls",
                            step,
                            pending_calls=response.tool_calls[call_index:],
                        )
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
                        return interrupt_at_ready_boundary(
                            "Run interrupted by user during tool execution",
                            step,
                            pending_calls=response.tool_calls[call_index:],
                            tool_error_message="run was interrupted during tool execution",
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
                    event_data = protocol.tool_finished_event_data(
                        call,
                        raw_execution,
                        execution,
                    )
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
                        protocol.append_cancelled_tool_results(
                            messages,
                            response.tool_calls[call_index + 1 :],
                            error_code="tool_call_cancelled",
                            error_message="an earlier tool forced a terminal safety stop",
                        )
                        self._save_checkpoint(
                            run_id,
                            task,
                            system_prompt,
                            messages,
                            SessionBoundary.TERMINAL,
                            step,
                            budget_usage=budget_state.usage,
                            stop_reason=StopReason.COMMAND_CONTROL_FAILED,
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
                        protocol.append_cancelled_tool_results(
                            messages,
                            response.tool_calls[call_index + 1 :],
                            error_code="tool_call_cancelled",
                            error_message="the repeated-call stop policy cancelled this call",
                        )
                        self._save_checkpoint(
                            run_id,
                            task,
                            system_prompt,
                            messages,
                            SessionBoundary.TERMINAL,
                            step,
                            budget_usage=budget_state.usage,
                            stop_reason=StopReason.REPEATED_TOOL_CALL,
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
            final_verification = protocol.final_verification_view(
                verification,
                self._verification_profile,
                self._completion_contract,
            )
            self._emit(
                run_id,
                EventKind.VERIFICATION_EVALUATED,
                final_verification.evaluation_message,
                step,
                final_verification.event_data,
            )
            terminal_state = (
                AgentState.COMPLETED
                if final_verification.verified
                else AgentState.COMPLETED_UNVERIFIED
            )
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
                final_verification.finished_message,
                step,
                final_verification.event_data,
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
        if not protocol.is_allowed_transition(previous, current):
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
