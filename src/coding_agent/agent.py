"""The project-owned model/tool/observation loop."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

from coding_agent.events import EventKind, EventSink, NullEventSink, RunEvent
from coding_agent.model import ModelAdapter
from coding_agent.models import (
    AgentResult,
    AgentState,
    ChatMessage,
    MessageRole,
    StopReason,
)
from coding_agent.tooling import ToolDispatcher

DEFAULT_SYSTEM_PROMPT = """You are a coding agent. Use the available local tools when needed.
Return a concise final answer only when the task is complete or you cannot make further progress."""

_ALLOWED_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.CREATED: {AgentState.PLANNING},
    AgentState.PLANNING: {AgentState.ACTING},
    AgentState.ACTING: {
        AgentState.OBSERVING,
        AgentState.COMPLETED_UNVERIFIED,
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
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if max_tool_calls_per_step < 1 or max_total_tool_calls < 1:
            raise ValueError("tool call limits must be at least 1")
        self._model = model
        self._tools = tools
        self._events = event_sink or NullEventSink()
        self._max_steps = max_steps
        self._max_tool_calls_per_step = max_tool_calls_per_step
        self._max_total_tool_calls = max_total_tool_calls

    def run(self, task: str, *, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> AgentResult:
        if not task.strip():
            raise ValueError("task cannot be empty")

        run_id = uuid4().hex
        state = AgentState.CREATED
        messages: list[ChatMessage] = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=task),
        ]
        # The full task already lives in conversation state; duplicating it into the
        # observable event stream would increase the chance of leaking source or secrets.
        self._emit(run_id, EventKind.RUN_STARTED, "Run started", data={"task_chars": len(task)})
        state = self._transition(run_id, state, AgentState.PLANNING)
        total_tool_calls = 0

        for step in range(1, self._max_steps + 1):
            state = self._transition(run_id, state, AgentState.ACTING, step)
            self._emit(run_id, EventKind.MODEL_REQUESTED, "Model requested", step)

            try:
                response = self._model.complete(tuple(messages), self._tools.specs())
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
                for call in response.tool_calls:
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
                continue

            state = self._transition(run_id, state, AgentState.COMPLETED_UNVERIFIED, step)
            self._emit(
                run_id,
                EventKind.RUN_FINISHED,
                "Run ended with an unverified final response",
                step,
                {"verified": False},
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
