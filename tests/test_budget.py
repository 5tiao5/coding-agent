"""Tests for coupled run budgets and aggregate model-observation limits."""

from __future__ import annotations

from typing import Any, cast

import pytest

from coding_agent.budget import BudgetPolicy, BudgetUsage, ObservationBudget
from coding_agent.models import (
    ChatMessage,
    MessageRole,
    ToolCall,
    ToolControlFacts,
    ToolExecution,
    VerificationKind,
    VerificationSignal,
)


def _assistant(call_count: int) -> ChatMessage:
    return ChatMessage(
        role=MessageRole.ASSISTANT,
        tool_calls=tuple(
            ToolCall(id=f"call-{index}", name="read_file", arguments={"path": f"f{index}.py"})
            for index in range(call_count)
        ),
    )


def _success(
    index: int,
    *,
    output: str,
    metadata: dict[str, object] | None = None,
) -> ToolExecution:
    return ToolExecution(
        call_id=f"call-{index}",
        tool_name="read_file",
        ok=True,
        output=output,
        summary=f"Read file {index}",
        metadata=cast(Any, metadata or {}),
    )


def test_policy_couples_model_turns_to_the_derived_total_tool_budget() -> None:
    default = BudgetPolicy()
    expanded = BudgetPolicy(max_model_turns=100)
    tiny = BudgetPolicy(max_model_turns=1)

    assert default.max_total_tool_calls == 40
    assert expanded.max_total_tool_calls == 200
    assert tiny.max_total_tool_calls == tiny.max_calls_per_turn == 8


def test_policy_rejects_incoherent_or_non_integer_configuration() -> None:
    invalid: tuple[dict[str, object], ...] = (
        {"max_model_turns": 0},
        {"max_model_turns": True},
        {"max_calls_per_turn": 0},
        {"average_calls_per_turn": 9},
        {"verification_turn_reserve": -1},
        {"verification_call_reserve": -1},
        {"context_chars": 1_000, "observation_chars_per_turn": 900, "memory_chars": 100},
    )

    for values in invalid:
        with pytest.raises(ValueError):
            BudgetPolicy(**cast(Any, values))


def test_usage_is_immutable_validated_and_counts_both_call_lanes() -> None:
    usage = BudgetUsage(model_turns=3, work_calls=5, verification_calls=2)

    assert usage.total_tool_calls == 7
    assert usage.add_turn().model_turns == 4
    assert usage.add_calls(work_calls=2, verification_calls=1) == BudgetUsage(
        model_turns=3,
        work_calls=7,
        verification_calls=3,
    )
    assert usage == BudgetUsage(model_turns=3, work_calls=5, verification_calls=2)

    with pytest.raises(ValueError):
        BudgetUsage(model_turns=-1)
    with pytest.raises(ValueError):
        usage.add_calls(work_calls=True)


def test_verification_reserve_protects_one_call_and_two_terminal_turns() -> None:
    policy = BudgetPolicy()
    state = policy.bind(
        BudgetUsage(model_turns=18, work_calls=39),
        verification_required=True,
    )

    work_turn = state.admit_turn("work")
    verify_turn = state.admit_turn("verification")
    final_turn = state.admit_turn("final")
    work_call = state.admit_batch(work_calls=1)
    verify_call = state.admit_batch(verification_calls=1)

    assert work_turn.accepted is False
    assert work_turn.code == "verification_turns_reserved"
    assert verify_turn.accepted is True
    assert final_turn.accepted is True
    assert work_call.accepted is False
    assert work_call.code == "verification_calls_reserved"
    assert verify_call.accepted is True
    assert state.effective_verification_turn_reserve == 2
    assert state.effective_verification_call_reserve == 1
    assert state.remaining_model_turns == 2
    assert state.remaining_total_tool_calls == 1


def test_absent_verification_requirement_releases_both_reserves() -> None:
    policy = BudgetPolicy()
    state = policy.bind(
        BudgetUsage(model_turns=19, work_calls=39),
        verification_required=False,
    )

    assert state.admit_turn("work").accepted is True
    assert state.admit_batch(work_calls=1).accepted is True
    assert state.effective_verification_turn_reserve == 0
    assert state.effective_verification_call_reserve == 0


def test_small_turn_budgets_degrade_the_turn_reserve_but_keep_one_work_turn() -> None:
    one = BudgetPolicy(max_model_turns=1).bind(verification_required=True)
    two = BudgetPolicy(max_model_turns=2).bind(verification_required=True)

    assert one.effective_verification_turn_reserve == 0
    assert one.admit_turn("work").accepted is True
    assert two.effective_verification_turn_reserve == 1
    assert two.consume_turn("work").admit_turn("work").accepted is False
    assert two.consume_turn("work").admit_turn("final").accepted is True


def test_batch_admission_enforces_burst_and_cumulative_limits_atomically() -> None:
    policy = BudgetPolicy(max_model_turns=5, max_calls_per_turn=8, average_calls_per_turn=2)
    state = policy.bind(BudgetUsage(work_calls=3), verification_required=False)

    assert policy.max_total_tool_calls == 10
    assert state.admit_batch(work_calls=8).code == "total_tool_calls_exhausted"
    assert state.admit_batch(work_calls=9).code == "batch_too_large"
    accepted = state.admit_batch(work_calls=7)
    assert accepted.accepted is True
    assert accepted.remaining_total_tool_calls == 0
    assert state.consume_batch(work_calls=7).usage.total_tool_calls == 10


def test_binding_refuses_usage_that_already_exceeds_the_selected_policy() -> None:
    policy = BudgetPolicy(max_model_turns=5)

    with pytest.raises(ValueError, match="model-turn usage"):
        policy.bind(BudgetUsage(model_turns=6))
    with pytest.raises(ValueError, match="tool-call usage"):
        policy.bind(BudgetUsage(work_calls=11))


def test_observation_turn_fairly_bounds_eight_individual_16k_results() -> None:
    budget = ObservationBudget()
    turn = budget.begin(_assistant(8), result_count=8)
    fitted = [
        turn.fit(_success(index, output=str(index) * 16_000), pending=7 - index)
        for index in range(8)
    ]

    assert turn.used_chars <= budget.max_turn_chars
    assert turn.remaining_results == 0
    assert all(item.truncated for item in fitted)
    assert all(len(item.as_message_content()) <= budget.max_single_chars for item in fitted)
    assert (
        max(len(item.as_message_content()) for item in fitted)
        - min(len(item.as_message_content()) for item in fitted)
        <= 8
    )


def test_observation_fit_preserves_short_results_and_host_control_facts() -> None:
    assistant = _assistant(1)
    control = ToolControlFacts(
        verification=VerificationSignal.PASSED,
        verification_kind=VerificationKind.TEST,
        verification_label="pytest",
    )
    execution = ToolExecution(
        call_id="call-0",
        tool_name="read_file",
        ok=True,
        output="short",
        summary="Short result",
        control=control,
    )
    turn = ObservationBudget(max_turn_chars=2_000, max_single_chars=1_000).begin(
        assistant,
        result_count=1,
    )

    fitted = turn.fit(execution, pending=0)

    assert fitted == execution
    assert fitted.control == control
    assert fitted.summary == execution.summary


def test_oversized_success_drops_optional_metadata_before_breaking_the_limit() -> None:
    execution = _success(
        0,
        output="OUTPUT" * 500,
        metadata={"private_blob": "M" * 4_000, "count": 12},
    )
    turn = ObservationBudget(
        max_turn_chars=1_200,
        max_single_chars=700,
        min_result_chars=200,
    ).begin(_assistant(1), result_count=1)

    fitted = turn.fit(execution, pending=0)

    assert fitted.ok is True
    assert fitted.truncated is True
    assert fitted.metadata == {}
    assert fitted.output is not None
    assert "observation truncated" in fitted.output
    assert len(fitted.as_message_content()) <= 700


def test_oversized_failure_uses_a_stable_bounded_message_and_remains_valid() -> None:
    execution = ToolExecution(
        call_id="call-0",
        tool_name="read_file",
        ok=False,
        error_code="read_failed",
        error_message="PRIVATE_FAILURE_DETAIL_" * 500,
        metadata={"private_blob": "M" * 4_000},
    )
    turn = ObservationBudget(
        max_turn_chars=1_000,
        max_single_chars=500,
        min_result_chars=200,
    ).begin(_assistant(1), result_count=1)

    fitted = turn.fit(execution, pending=0)

    assert fitted.ok is False
    assert fitted.error_code == "read_failed"
    assert fitted.error_message == "tool failed; details omitted by observation budget"
    assert fitted.metadata == {}
    assert fitted.truncated is False
    assert "PRIVATE_FAILURE_DETAIL" not in fitted.as_message_content()
    assert len(fitted.as_message_content()) <= 500
    assert ToolExecution.model_validate(fitted.model_dump()) == fitted


def test_observation_turn_rejects_protocol_misuse_before_accounting_drifts() -> None:
    budget = ObservationBudget(max_turn_chars=2_000, max_single_chars=1_000)

    with pytest.raises(ValueError, match="tool-call count"):
        budget.begin(_assistant(2), result_count=1)
    with pytest.raises(ValueError, match="minimum observations"):
        ObservationBudget(
            max_turn_chars=800,
            max_single_chars=400,
            min_result_chars=300,
        ).begin(_assistant(3), result_count=3)

    turn = budget.begin(_assistant(2), result_count=2)
    with pytest.raises(ValueError, match="pending"):
        turn.fit(_success(0, output="ok"), pending=0)
