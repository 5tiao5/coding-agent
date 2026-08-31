"""Coupled run budgets and aggregate bounds for model-facing observations.

The classes in this module deliberately do not know about ``AgentRunner``.  They
are immutable policy/state values plus a small per-turn observation accumulator,
which keeps admission decisions deterministic and straightforward to test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Self

from coding_agent.models import ChatMessage, MessageRole, ToolExecution

BudgetPurpose = Literal["work", "verification", "final"]

_OBSERVATION_TRUNCATION_MARKER = "\n...[observation truncated]...\n"
_BOUNDED_FAILURE_MESSAGE = "tool failed; details omitted by observation budget"
_MIN_MEMORY_CHARS = 4_096


@dataclass(frozen=True, slots=True)
class BudgetUsage:
    """Cumulative accepted work for one run, including restored checkpoints."""

    model_turns: int = 0
    work_calls: int = 0
    verification_calls: int = 0

    def __post_init__(self) -> None:
        _require_int("model_turns", self.model_turns, minimum=0)
        _require_int("work_calls", self.work_calls, minimum=0)
        _require_int("verification_calls", self.verification_calls, minimum=0)

    @property
    def total_tool_calls(self) -> int:
        return self.work_calls + self.verification_calls

    def add_turn(self) -> Self:
        return type(self)(
            model_turns=self.model_turns + 1,
            work_calls=self.work_calls,
            verification_calls=self.verification_calls,
        )

    def add_calls(self, *, work_calls: int = 0, verification_calls: int = 0) -> Self:
        _require_int("work_calls", work_calls, minimum=0)
        _require_int("verification_calls", verification_calls, minimum=0)
        return type(self)(
            model_turns=self.model_turns,
            work_calls=self.work_calls + work_calls,
            verification_calls=self.verification_calls + verification_calls,
        )


@dataclass(frozen=True, slots=True)
class BudgetAdmission:
    """One admission decision with remaining capacity after an accepted request."""

    accepted: bool
    code: str
    remaining_model_turns: int
    remaining_total_tool_calls: int
    remaining_work_tool_calls: int

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise ValueError("accepted must be a boolean")
        if not self.code:
            raise ValueError("admission code cannot be empty")
        _require_int("remaining_model_turns", self.remaining_model_turns, minimum=0)
        _require_int(
            "remaining_total_tool_calls",
            self.remaining_total_tool_calls,
            minimum=0,
        )
        _require_int(
            "remaining_work_tool_calls",
            self.remaining_work_tool_calls,
            minimum=0,
        )


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    """Host-selected run policy with one model-turn knob and derived call capacity.

    ``average_calls_per_turn`` couples the cumulative call ceiling to
    ``max_model_turns``.  ``max_calls_per_turn`` remains a burst/atomic-batch
    safety limit, and also supplies a floor so very short deterministic runs can
    still issue one useful batch.
    """

    max_model_turns: int = 20
    max_calls_per_turn: int = 8
    average_calls_per_turn: int = 2
    verification_turn_reserve: int = 2
    verification_call_reserve: int = 1
    context_chars: int = 80_000
    observation_chars_per_turn: int = 48_000
    memory_chars: int = 8_000

    def __post_init__(self) -> None:
        _require_int("max_model_turns", self.max_model_turns, minimum=1)
        _require_int("max_calls_per_turn", self.max_calls_per_turn, minimum=1)
        _require_int("average_calls_per_turn", self.average_calls_per_turn, minimum=1)
        _require_int(
            "verification_turn_reserve",
            self.verification_turn_reserve,
            minimum=0,
        )
        _require_int(
            "verification_call_reserve",
            self.verification_call_reserve,
            minimum=0,
        )
        _require_int("context_chars", self.context_chars, minimum=1)
        _require_int(
            "observation_chars_per_turn",
            self.observation_chars_per_turn,
            minimum=1,
        )
        _require_int("memory_chars", self.memory_chars, minimum=_MIN_MEMORY_CHARS)
        if self.average_calls_per_turn > self.max_calls_per_turn:
            raise ValueError("average_calls_per_turn cannot exceed max_calls_per_turn")
        if self.observation_chars_per_turn + self.memory_chars >= self.context_chars:
            raise ValueError(
                "observation and memory budgets must leave context capacity for anchors and tools"
            )

    @property
    def max_total_tool_calls(self) -> int:
        """Return the derived cumulative ceiling; it is not an independent knob."""

        return max(
            self.max_calls_per_turn,
            self.max_model_turns * self.average_calls_per_turn,
        )

    @property
    def observation_budget(self) -> ObservationBudget:
        return ObservationBudget(max_turn_chars=self.observation_chars_per_turn)

    def bind(
        self,
        usage: BudgetUsage | None = None,
        *,
        verification_required: bool = False,
    ) -> BudgetState:
        if type(verification_required) is not bool:
            raise ValueError("verification_required must be a boolean")
        selected = usage or BudgetUsage()
        if selected.model_turns > self.max_model_turns:
            raise ValueError("model-turn usage exceeds the selected budget policy")
        if selected.total_tool_calls > self.max_total_tool_calls:
            raise ValueError("tool-call usage exceeds the selected budget policy")
        return BudgetState(
            policy=self,
            usage=selected,
            verification_required=verification_required,
        )


@dataclass(frozen=True, slots=True)
class BudgetState:
    """A policy bound to cumulative usage and the current verification requirement."""

    policy: BudgetPolicy
    usage: BudgetUsage
    verification_required: bool

    def __post_init__(self) -> None:
        if type(self.verification_required) is not bool:
            raise ValueError("verification_required must be a boolean")
        if self.usage.model_turns > self.policy.max_model_turns:
            raise ValueError("model-turn usage exceeds the selected budget policy")
        if self.usage.total_tool_calls > self.policy.max_total_tool_calls:
            raise ValueError("tool-call usage exceeds the selected budget policy")

    @property
    def effective_verification_turn_reserve(self) -> int:
        if not self.verification_required:
            return 0
        # Always leave one work-capable turn for tiny explicit budgets.  From three
        # turns onward the default reserves a verifier-request turn and a final turn.
        return min(
            self.policy.verification_turn_reserve,
            max(0, self.policy.max_model_turns - 1),
        )

    @property
    def effective_verification_call_reserve(self) -> int:
        if not self.verification_required:
            return 0
        return min(
            self.policy.verification_call_reserve,
            max(0, self.policy.max_total_tool_calls - 1),
        )

    @property
    def max_work_model_turns(self) -> int:
        return self.policy.max_model_turns - self.effective_verification_turn_reserve

    @property
    def max_work_tool_calls(self) -> int:
        return self.policy.max_total_tool_calls - self.effective_verification_call_reserve

    @property
    def remaining_model_turns(self) -> int:
        return self.policy.max_model_turns - self.usage.model_turns

    @property
    def remaining_total_tool_calls(self) -> int:
        return self.policy.max_total_tool_calls - self.usage.total_tool_calls

    @property
    def remaining_work_tool_calls(self) -> int:
        lane_remaining = max(0, self.max_work_tool_calls - self.usage.work_calls)
        return min(lane_remaining, self.remaining_total_tool_calls)

    def admit_turn(self, purpose: BudgetPurpose = "work") -> BudgetAdmission:
        if purpose not in {"work", "verification", "final"}:
            raise ValueError("purpose must be work, verification, or final")
        if self.remaining_model_turns == 0:
            return self._admission(False, "model_turns_exhausted")
        if purpose == "work" and self.usage.model_turns >= self.max_work_model_turns:
            return self._admission(False, "verification_turns_reserved")
        if (
            purpose == "verification"
            and self.verification_required
            and self.remaining_model_turns <= 1
        ):
            return self._admission(False, "final_turn_reserved")
        return self._admission(True, "accepted", model_turns=1)

    def admit_batch(
        self,
        *,
        work_calls: int = 0,
        verification_calls: int = 0,
    ) -> BudgetAdmission:
        _require_int("work_calls", work_calls, minimum=0)
        _require_int("verification_calls", verification_calls, minimum=0)
        requested = work_calls + verification_calls
        if requested < 1:
            raise ValueError("a tool batch must contain at least one call")
        if requested > self.policy.max_calls_per_turn:
            return self._admission(False, "batch_too_large")
        if requested > self.remaining_total_tool_calls:
            return self._admission(False, "total_tool_calls_exhausted")
        if (
            self.verification_required
            and self.usage.work_calls + work_calls > self.max_work_tool_calls
        ):
            return self._admission(False, "verification_calls_reserved")
        return self._admission(
            True,
            "accepted",
            work_calls=work_calls,
            verification_calls=verification_calls,
        )

    def consume_turn(self, purpose: BudgetPurpose = "work") -> BudgetState:
        admission = self.admit_turn(purpose)
        _require_accepted(admission)
        return self.policy.bind(
            self.usage.add_turn(),
            verification_required=self.verification_required,
        )

    def consume_batch(
        self,
        *,
        work_calls: int = 0,
        verification_calls: int = 0,
    ) -> BudgetState:
        admission = self.admit_batch(
            work_calls=work_calls,
            verification_calls=verification_calls,
        )
        _require_accepted(admission)
        return self.policy.bind(
            self.usage.add_calls(
                work_calls=work_calls,
                verification_calls=verification_calls,
            ),
            verification_required=self.verification_required,
        )

    def _admission(
        self,
        accepted: bool,
        code: str,
        *,
        model_turns: int = 0,
        work_calls: int = 0,
        verification_calls: int = 0,
    ) -> BudgetAdmission:
        if accepted:
            remaining_turns = self.remaining_model_turns - model_turns
            remaining_total = self.remaining_total_tool_calls - work_calls - verification_calls
            remaining_work = min(
                max(0, self.max_work_tool_calls - self.usage.work_calls - work_calls),
                remaining_total,
            )
        else:
            remaining_turns = self.remaining_model_turns
            remaining_total = self.remaining_total_tool_calls
            remaining_work = self.remaining_work_tool_calls
        return BudgetAdmission(
            accepted=accepted,
            code=code,
            remaining_model_turns=remaining_turns,
            remaining_total_tool_calls=remaining_total,
            remaining_work_tool_calls=remaining_work,
        )


@dataclass(frozen=True, slots=True)
class ObservationBudget:
    """Aggregate cap for one assistant tool-call block and all of its results."""

    max_turn_chars: int = 48_000
    max_single_chars: int = 16_000
    min_result_chars: int = 512

    def __post_init__(self) -> None:
        _require_int("max_turn_chars", self.max_turn_chars, minimum=1)
        _require_int("max_single_chars", self.max_single_chars, minimum=1)
        _require_int("min_result_chars", self.min_result_chars, minimum=1)
        if self.min_result_chars > self.max_single_chars:
            raise ValueError("min_result_chars cannot exceed max_single_chars")
        if self.max_single_chars > self.max_turn_chars:
            raise ValueError("max_single_chars cannot exceed max_turn_chars")

    def begin(self, assistant: ChatMessage, *, result_count: int) -> ObservationTurn:
        _require_int("result_count", result_count, minimum=1)
        if assistant.role is not MessageRole.ASSISTANT:
            raise ValueError("an observation turn requires an assistant message")
        if len(assistant.tool_calls) != result_count:
            raise ValueError("result_count must match the assistant tool-call count")
        assistant_chars = _chat_message_chars(assistant)
        if assistant_chars + result_count * self.min_result_chars > self.max_turn_chars:
            raise ValueError("assistant and minimum observations exceed the turn budget")
        return ObservationTurn(
            self,
            assistant.tool_calls,
            assistant_chars=assistant_chars,
        )


class ObservationTurn:
    """Mutable accounting local to one closed assistant/tool-result block."""

    __slots__ = (
        "_assistant_chars",
        "_budget",
        "_calls",
        "_next_index",
        "_used_chars",
    )

    def __init__(
        self,
        budget: ObservationBudget,
        calls: tuple[object, ...],
        *,
        assistant_chars: int,
    ) -> None:
        self._budget = budget
        self._calls = calls
        self._assistant_chars = assistant_chars
        self._used_chars = assistant_chars
        self._next_index = 0

    @property
    def assistant_chars(self) -> int:
        return self._assistant_chars

    @property
    def used_chars(self) -> int:
        return self._used_chars

    @property
    def remaining_chars(self) -> int:
        return max(0, self._budget.max_turn_chars - self._used_chars)

    @property
    def remaining_results(self) -> int:
        return len(self._calls) - self._next_index

    @property
    def overflowed(self) -> bool:
        return self._used_chars > self._budget.max_turn_chars

    def fit(self, execution: ToolExecution, *, pending: int) -> ToolExecution:
        """Fit one result fairly while reserving minimum space for pending results."""

        _require_int("pending", pending, minimum=0)
        if self.remaining_results < 1:
            raise ValueError("all observation results have already been accounted")
        if pending != self.remaining_results - 1:
            raise ValueError("pending must equal the number of later unaccounted results")

        expected = self._calls[self._next_index]
        expected_id = getattr(expected, "id", None)
        expected_name = getattr(expected, "name", None)
        if execution.call_id != expected_id or execution.tool_name != expected_name:
            raise ValueError("tool execution does not match the next assistant tool call")

        fair_share = self.remaining_chars // self.remaining_results
        target = min(self._budget.max_single_chars, fair_share)
        fitted = _fit_execution(execution, target)
        self._used_chars += _execution_chars(fitted)
        self._next_index += 1
        return fitted


def _fit_execution(execution: ToolExecution, target: int) -> ToolExecution:
    if _execution_chars(execution) <= target:
        return execution
    if execution.ok:
        return _fit_success(execution, target)
    return _fit_failure(execution, target)


def _fit_success(execution: ToolExecution, target: int) -> ToolExecution:
    assert execution.output is not None
    for metadata in (execution.metadata, {}):
        base = execution.model_copy(update={"metadata": metadata, "output": "", "truncated": True})
        if _execution_chars(base) > target:
            continue

        marker_candidate = base.model_copy(update={"output": _OBSERVATION_TRUNCATION_MARKER})
        if _execution_chars(marker_candidate) > target:
            return base

        low = 0
        high = len(execution.output)
        best = marker_candidate
        while low <= high:
            preserved = (low + high) // 2
            candidate = base.model_copy(
                update={
                    "output": _head_tail_with_marker(execution.output, preserved),
                }
            )
            if _execution_chars(candidate) <= target:
                best = candidate
                low = preserved + 1
            else:
                high = preserved - 1
        return best

    # Call identifiers and names are protocol identity and cannot be shortened.  Return
    # the smallest valid success even if hostile identity lengths alone exceed target;
    # ``overflowed`` makes that exceptional condition observable to the caller.
    return execution.model_copy(update={"metadata": {}, "output": "", "truncated": True})


def _fit_failure(execution: ToolExecution, target: int) -> ToolExecution:
    for message in (_BOUNDED_FAILURE_MESSAGE, "tool failed", "x"):
        candidate = execution.model_copy(update={"metadata": {}, "error_message": message})
        if _execution_chars(candidate) <= target:
            return candidate
    # As above, preserving the provider call identity takes priority over a pathological
    # target overrun.  Failed executions cannot legally set ``truncated=True``.
    return execution.model_copy(update={"metadata": {}, "error_message": "x"})


def _head_tail_with_marker(text: str, preserved_chars: int) -> str:
    if preserved_chars <= 0:
        return _OBSERVATION_TRUNCATION_MARKER
    if preserved_chars >= len(text):
        return f"{text}{_OBSERVATION_TRUNCATION_MARKER}"
    head_chars = (preserved_chars + 1) // 2
    tail_chars = preserved_chars - head_chars
    tail = text[-tail_chars:] if tail_chars else ""
    return text[:head_chars] + _OBSERVATION_TRUNCATION_MARKER + tail


def _chat_message_chars(message: ChatMessage) -> int:
    payload = message.model_dump(mode="json", exclude_defaults=True, exclude_none=True)
    return len(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _execution_chars(execution: ToolExecution) -> int:
    return len(execution.as_message_content())


def _require_int(name: str, value: int, *, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def _require_accepted(admission: BudgetAdmission) -> None:
    if not admission.accepted:
        raise ValueError(f"budget admission rejected: {admission.code}")
