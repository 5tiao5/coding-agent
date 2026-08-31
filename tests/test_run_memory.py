"""Tests for bounded host-owned memory retained across long agent runs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from coding_agent.models import (
    ToolCall,
    ToolControlFacts,
    ToolExecution,
    ToolMetadataValue,
    VerificationKind,
    VerificationSignal,
)
from coding_agent.plan import PlanStatus
from coding_agent.run_memory import RunMemory, RunMemoryError, RunMemorySnapshot
from coding_agent.tools.base import ToolRegistry
from coding_agent.tools.plan import UpdatePlanTool


def _success(
    call: ToolCall,
    *,
    metadata: dict[str, ToolMetadataValue] | None = None,
    output: str = "private raw output",
    control: ToolControlFacts | None = None,
) -> ToolExecution:
    return ToolExecution(
        call_id=call.id,
        tool_name=call.name,
        ok=True,
        output=output,
        summary=f"Observed {call.name}",
        metadata=metadata or {},
        control=control or ToolControlFacts(),
    )


def _mutation(
    number: int,
    path: str,
    *,
    kind: str = "update",
    after: str | None = None,
) -> tuple[ToolCall, ToolExecution]:
    tool_name = "undo_change" if kind == "undo" else "replace_text"
    call = ToolCall(
        id=f"mutation-{number}",
        name=tool_name,
        arguments={
            "path": path,
            "old_text": "PRIVATE_OLD_TEXT",
            "new_text": "PRIVATE_NEW_TEXT",
        },
    )
    execution = _success(
        call,
        metadata={
            "path": path,
            "change_id": f"chg_{number:04d}",
            "changed": True,
            "change_kind": kind,
            "before_sha256": "a" * 64,
            "after_sha256": after if after is not None else "b" * 64,
            "added_lines": number,
            "removed_lines": number - 1,
            "mutation_revision": number,
            "private_blob": "MUST_NOT_SURVIVE",
        },
        output="PRIVATE_DIFF_MUST_NOT_SURVIVE",
        control=ToolControlFacts(invalidates_verification=True, made_progress=True),
    )
    return call, execution


def _failed_command(
    number: int,
    *,
    status: str = "exited",
    exit_code: int | None = 1,
) -> tuple[ToolCall, ToolExecution]:
    call = ToolCall(
        id=f"command-{number}",
        name="run_command",
        arguments={"argv": ["pytest", f"tests/test_{number}.py"], "cwd": "."},
    )
    execution = _success(
        call,
        metadata={
            "status": status,
            "exit_code": exit_code,
            "timed_out": status == "timed_out",
            "cwd": ".",
            "command_class": "general",
            "private_blob": "MUST_NOT_SURVIVE",
        },
        output="PRIVATE_STDERR_MUST_NOT_SURVIVE",
    )
    return call, execution


def _verification(
    call_id: str,
    signal: VerificationSignal,
    *,
    label: str = "pytest",
) -> tuple[ToolCall, ToolExecution]:
    call = ToolCall(id=call_id, name="run_command", arguments={"argv": ["pytest"], "cwd": "."})
    execution = _success(
        call,
        metadata={
            "status": "exited",
            "exit_code": 0 if signal is VerificationSignal.PASSED else 1,
            "cwd": ".",
        },
        control=ToolControlFacts(
            verification=signal,
            verification_kind=VerificationKind.TEST,
            verification_label=label,
        ),
    )
    return call, execution


def _large_failed_command(number: int, *, argument_chars: int) -> tuple[ToolCall, ToolExecution]:
    call = ToolCall(
        id=f"large-command-{number}",
        name="run_command",
        arguments={
            "argv": ["pytest", "X" * argument_chars, f"case-{number}"],
            "cwd": ".",
        },
    )
    return call, _success(
        call,
        metadata={"status": "exited", "exit_code": 1, "cwd": "."},
        output="RAW COMMAND OUTPUT MUST NEVER ENTER MEMORY",
    )


def test_memory_snapshots_the_shared_explicit_plan_without_reasoning() -> None:
    memory = RunMemory()
    call = ToolCall(
        id="plan-1",
        name="update_plan",
        arguments={
            "items": [
                {"id": "inspect", "step": "Inspect files", "status": "completed"},
                {"id": "edit", "step": "Apply fix", "status": "in_progress"},
            ]
        },
    )
    execution = ToolRegistry([UpdatePlanTool(memory.plan_state)]).execute(call)

    memory.observe(call, execution, step=1)
    snapshot = memory.snapshot()

    assert snapshot.plan.revision == 1
    assert [item.status for item in snapshot.plan.items] == [
        PlanStatus.COMPLETED,
        PlanStatus.IN_PROGRESS,
    ]
    assert snapshot.revision == 1
    assert "reasoning" not in snapshot.canonical_json()


def test_reset_clears_task_facts_without_replacing_shared_plan_state() -> None:
    memory = RunMemory()
    shared_plan = memory.plan_state
    plan_call = ToolCall(
        id="plan-reset",
        name="update_plan",
        arguments={"items": [{"id": "old", "step": "Old task", "status": "in_progress"}]},
    )
    plan_execution = ToolRegistry([UpdatePlanTool(shared_plan)]).execute(plan_call)
    memory.observe(plan_call, plan_execution, step=1)
    mutation_call, mutation_execution = _mutation(1, "src/old.py")
    memory.observe(mutation_call, mutation_execution, step=2)

    reset = memory.reset()

    assert memory.plan_state is shared_plan
    assert reset == RunMemorySnapshot(revision=0)
    assert shared_plan.snapshot().revision == 0
    next_execution = ToolRegistry([UpdatePlanTool(shared_plan)]).execute(
        ToolCall(
            id="plan-next",
            name="update_plan",
            arguments={"items": [{"id": "new", "step": "New task", "status": "in_progress"}]},
        )
    )
    assert next_execution.ok is True
    assert shared_plan.revision == 1


def test_file_changes_are_allowlisted_coalesced_by_path_and_lru_bounded() -> None:
    memory = RunMemory(max_file_changes=2)
    first = _mutation(1, "src/a.py")
    second = _mutation(2, "src/a.py")
    third = _mutation(3, "src/b.py")
    fourth = _mutation(4, "src/c.py")

    for step, (call, execution) in enumerate((first, second, third, fourth), 1):
        memory.observe(call, execution, step=step)

    snapshot = memory.snapshot()

    assert [fact.path for fact in snapshot.file_changes] == ["src/b.py", "src/c.py"]
    assert snapshot.file_changes[-1].last_change_id == "chg_0004"
    assert snapshot.file_changes[-1].last_step == 4
    assert snapshot.file_changes[-1].change_count == 1
    serialized = snapshot.canonical_json()
    assert "PRIVATE_DIFF_MUST_NOT_SURVIVE" not in serialized
    assert "PRIVATE_OLD_TEXT" not in serialized
    assert "PRIVATE_NEW_TEXT" not in serialized
    assert "MUST_NOT_SURVIVE" not in serialized

    coalesced = RunMemory(max_file_changes=2)
    coalesced.observe(*first, step=1)
    coalesced.observe(*second, step=2)
    only = coalesced.snapshot().file_changes
    assert len(only) == 1
    assert only[0].change_count == 2
    assert only[0].last_change_id == "chg_0002"


def test_only_failed_commands_are_retained_and_recent_failures_are_bounded() -> None:
    memory = RunMemory(max_failed_commands=2)
    successful_call = ToolCall(
        id="success",
        name="run_command",
        arguments={"argv": ["pytest", "-q"], "cwd": "."},
    )
    successful = _success(
        successful_call,
        metadata={"status": "exited", "exit_code": 0, "cwd": "."},
    )
    memory.observe(successful_call, successful, step=1)

    failures = (
        _failed_command(1),
        _failed_command(2, status="timed_out", exit_code=None),
        _failed_command(3),
    )
    for step, (call, execution) in enumerate(failures, 2):
        memory.observe(call, execution, step=step)

    snapshot = memory.snapshot()

    assert [fact.argv[-1] for fact in snapshot.failed_commands] == [
        "tests/test_2.py",
        "tests/test_3.py",
    ]
    assert snapshot.failed_commands[0].failure_kind == "timed_out"
    assert snapshot.failed_commands[1].failure_kind == "nonzero_exit"
    serialized = snapshot.canonical_json()
    assert "PRIVATE_STDERR_MUST_NOT_SURVIVE" not in serialized
    assert "MUST_NOT_SURVIVE" not in serialized


def test_tool_level_command_failure_retains_only_safe_code_and_invocation() -> None:
    memory = RunMemory()
    call = ToolCall(
        id="denied",
        name="run_command",
        arguments={"argv": ["git", "status"], "cwd": "."},
    )
    failure = ToolExecution(
        call_id=call.id,
        tool_name=call.name,
        ok=False,
        error_code="command_denied",
        error_message="PRIVATE FAILURE DETAIL MUST NOT SURVIVE",
    )

    memory.observe(call, failure, step=7)
    fact = memory.snapshot().failed_commands[0]

    assert fact.failure_kind == "tool_error"
    assert fact.error_code == "command_denied"
    assert fact.argv == ("git", "status")
    assert "PRIVATE FAILURE DETAIL" not in memory.snapshot().canonical_json()


def test_verification_facts_survive_invalidation_only_as_stale_history() -> None:
    memory = RunMemory()
    passed_call, passed = _verification("verify-1", VerificationSignal.PASSED)
    memory.observe(passed_call, passed, step=1)

    initial = memory.snapshot().verification_facts[0]
    assert initial.passed is True
    assert initial.stale is False

    invalidating_call = ToolCall(id="general", name="other", arguments={})
    invalidating = _success(
        invalidating_call,
        control=ToolControlFacts(invalidates_verification=True),
    )
    memory.observe(invalidating_call, invalidating, step=2)

    stale = memory.snapshot().verification_facts[0]
    assert stale.passed is True
    assert stale.stale is True

    failed_call, failed = _verification("verify-2", VerificationSignal.FAILED)
    memory.observe(failed_call, failed, step=3)
    current = memory.snapshot().verification_facts[0]
    assert current.passed is False
    assert current.stale is False
    assert current.step == 3

    assert memory.mark_verification_stale() is True
    assert memory.mark_verification_stale() is False
    assert memory.snapshot().verification_facts[0].stale is True


def test_restore_requires_pristine_memory_and_forces_old_verification_stale() -> None:
    original = RunMemory()
    call, execution = _verification("verify", VerificationSignal.PASSED)
    original.observe(call, execution, step=4)
    snapshot = original.snapshot()

    restored = RunMemory()
    restored.restore(snapshot, mark_verification_stale=True)

    restored_snapshot = restored.snapshot()
    assert restored_snapshot.verification_facts[0].stale is True
    assert restored_snapshot.revision == snapshot.revision + 1

    with pytest.raises(RunMemoryError) as repeated:
        restored.restore(snapshot)
    assert repeated.value.code == "run_memory_restore_conflict"


def test_snapshot_can_be_purely_demoted_for_passive_session_load() -> None:
    original = RunMemory()
    call, execution = _verification("verify", VerificationSignal.PASSED)
    original.observe(call, execution, step=4)
    snapshot = original.snapshot()

    stale = snapshot.with_stale_verification()

    assert snapshot.verification_facts[0].stale is False
    assert stale.verification_facts[0].stale is True
    assert stale.revision == snapshot.revision + 1
    assert stale.with_stale_verification() is stale


def test_snapshot_schema_is_strict_bounded_and_canonical_json_is_deterministic() -> None:
    first = RunMemory()
    second = RunMemory()
    observations = (_mutation(1, "src/a.py"), _failed_command(1))
    for memory in (first, second):
        for step, (call, execution) in enumerate(observations, 1):
            memory.observe(call, execution, step=step)

    assert first.snapshot().canonical_json() == second.snapshot().canonical_json()

    payload = first.snapshot().model_dump(mode="json")
    payload["raw_output"] = "forbidden"
    with pytest.raises(ValidationError):
        RunMemorySnapshot.model_validate(payload)


def test_character_budget_keeps_plan_and_file_facts_before_oversized_commands() -> None:
    memory = RunMemory(max_chars=4_096, max_failed_commands=32)
    plan_call = ToolCall(
        id="plan-budget",
        name="update_plan",
        arguments={
            "items": [
                {"id": "inspect", "step": "Inspect files", "status": "completed"},
                {"id": "verify", "step": "Run tests", "status": "in_progress"},
            ]
        },
    )
    plan_execution = ToolRegistry([UpdatePlanTool(memory.plan_state)]).execute(plan_call)
    memory.observe(plan_call, plan_execution, step=1)
    mutation_call, mutation_execution = _mutation(1, "src/important.py")
    memory.observe(mutation_call, mutation_execution, step=2)

    for number in range(8):
        call, execution = _large_failed_command(number, argument_chars=3_500)
        memory.observe(call, execution, step=number + 3)

    snapshot = memory.snapshot()

    assert memory.max_chars == 4_096
    assert len(snapshot.canonical_json()) <= memory.max_chars
    assert snapshot.plan.revision == 1
    assert [fact.path for fact in snapshot.file_changes] == ["src/important.py"]
    assert snapshot.failed_commands == ()
    assert "RAW COMMAND OUTPUT" not in snapshot.canonical_json()


def test_character_budget_retains_the_most_recent_commands_deterministically() -> None:
    memories = [RunMemory(max_chars=4_096, max_failed_commands=32) for _ in range(2)]

    for memory in memories:
        for number in range(7):
            call, execution = _large_failed_command(number, argument_chars=800)
            memory.observe(call, execution, step=number + 1)

    first = memories[0].snapshot()
    second = memories[1].snapshot()

    assert first.canonical_json() == second.canonical_json()
    assert len(first.canonical_json()) <= 4_096
    assert 1 <= len(first.failed_commands) < 7
    assert first.failed_commands[-1].argv[-1] == "case-6"
    assert all(fact.argv[-1] != "case-0" for fact in first.failed_commands)


def test_restore_rejects_snapshot_above_the_runtime_character_budget() -> None:
    source = RunMemory(max_chars=64_000, max_failed_commands=32)
    for number in range(8):
        call, execution = _large_failed_command(number, argument_chars=1_000)
        source.observe(call, execution, step=number + 1)
    snapshot = source.snapshot()
    assert len(snapshot.canonical_json()) > 4_096

    with pytest.raises(RunMemoryError) as raised:
        RunMemory(max_chars=4_096, max_failed_commands=32).restore(snapshot)

    assert raised.value.code == "run_memory_limit_mismatch"
    assert raised.value.metadata["max_chars"] == 4_096


def test_character_budget_has_safe_constructor_bounds() -> None:
    assert RunMemory().max_chars == 8_000
    for invalid in (4_095, 64_001, True):
        with pytest.raises(ValueError, match="max_chars"):
            RunMemory(max_chars=invalid)
