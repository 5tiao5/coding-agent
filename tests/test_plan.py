"""Tests for explicit, monotonic task-plan snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest
from pydantic import ValidationError

from coding_agent.models import ToolCall, ToolExecution
from coding_agent.plan import PlanError, PlanItem, PlanSnapshot, PlanState, PlanStatus
from coding_agent.tools.base import ToolRegistry
from coding_agent.tools.plan import UpdatePlanArguments, UpdatePlanTool


def _item(
    item_id: str,
    step: str,
    status: PlanStatus = PlanStatus.PENDING,
) -> PlanItem:
    return PlanItem(id=item_id, step=step, status=status)


def _execute(state: PlanState, items: Sequence[Mapping[str, object]]) -> ToolExecution:
    return ToolRegistry([UpdatePlanTool(state)]).execute(
        ToolCall(id="plan-call", name="update_plan", arguments={"items": items})
    )


def test_plan_item_accepts_only_short_ids_and_one_printable_explicit_step() -> None:
    item = _item("verify_1", "Run the focused test suite", PlanStatus.IN_PROGRESS)

    assert item.id == "verify_1"
    assert item.step == "Run the focused test suite"
    assert item.status is PlanStatus.IN_PROGRESS

    invalid_values = [
        {"id": "has spaces", "step": "Read files", "status": "pending"},
        {"id": "x" * 25, "step": "Read files", "status": "pending"},
        {"id": "read", "step": " padded", "status": "pending"},
        {"id": "read", "step": "two\nlines", "status": "pending"},
        {"id": "read", "step": "escape\x1b", "status": "pending"},
        {"id": "read", "step": "Read files", "status": "pending", "reasoning": "hidden"},
    ]
    for values in invalid_values:
        with pytest.raises(ValidationError):
            PlanItem.model_validate(values)


def test_plan_requires_one_to_eight_unique_items_and_one_active_step() -> None:
    state = PlanState()

    with pytest.raises(PlanError) as empty:
        state.update([])
    assert empty.value.code == "plan_size_invalid"

    with pytest.raises(PlanError) as too_many:
        state.update([_item(f"item-{index}", f"Step {index}") for index in range(9)])
    assert too_many.value.code == "plan_size_invalid"

    with pytest.raises(PlanError) as duplicate:
        state.update([_item("same", "First"), _item("same", "Second")])
    assert duplicate.value.code == "plan_duplicate_id"

    with pytest.raises(PlanError) as multiple_active:
        state.update(
            [
                _item("first", "First", PlanStatus.IN_PROGRESS),
                _item("second", "Second", PlanStatus.IN_PROGRESS),
            ]
        )
    assert multiple_active.value.code == "plan_multiple_in_progress"
    assert state.items == ()
    assert state.revision == 0


def test_revision_changes_only_for_a_genuinely_different_snapshot() -> None:
    state = PlanState()
    initial = (
        _item("inspect", "Inspect the repository", PlanStatus.IN_PROGRESS),
        _item("verify", "Run tests"),
    )

    first = state.update(initial)
    replay = state.update(tuple(initial))
    changed = state.update(
        (
            _item("inspect", "Inspect the repository", PlanStatus.COMPLETED),
            _item("verify", "Run tests", PlanStatus.IN_PROGRESS),
        )
    )

    assert first.changed is True
    assert first.revision == 1
    assert replay.changed is False
    assert replay.revision == 1
    assert changed.changed is True
    assert changed.revision == 2
    assert state.items == changed.items


@pytest.mark.parametrize(
    ("replacement", "code"),
    [
        ((_item("new", "New work"),), "plan_completed_item_removed"),
        (
            (_item("done", "Inspect", PlanStatus.PENDING),),
            "plan_completed_item_regressed",
        ),
        (
            (_item("done", "Rewrite history", PlanStatus.COMPLETED),),
            "plan_completed_item_rewritten",
        ),
    ],
    ids=["removed", "regressed", "rewritten"],
)
def test_completed_items_cannot_be_erased_or_rewritten(
    replacement: tuple[PlanItem, ...],
    code: str,
) -> None:
    state = PlanState()
    original = (_item("done", "Inspect", PlanStatus.COMPLETED),)
    state.update(original)

    with pytest.raises(PlanError) as raised:
        state.update(replacement)

    assert raised.value.code == code
    assert state.items == original
    assert state.revision == 1


def test_pending_items_may_be_refined_and_full_snapshot_order_is_significant() -> None:
    state = PlanState()
    state.update((_item("a", "First"), _item("b", "Second")))

    update = state.update((_item("b", "Refined second"), _item("a", "First")))

    assert update.changed is True
    assert update.revision == 2
    assert [item.id for item in update.items] == ["b", "a"]
    assert update.items[0].step == "Refined second"


def test_update_plan_tool_exposes_only_explicit_snapshot_and_scalar_counts() -> None:
    state = PlanState()
    registry = ToolRegistry([UpdatePlanTool(state)])
    execution = registry.execute(
        ToolCall(
            id="plan-1",
            name="update_plan",
            arguments={
                "items": [
                    {"id": "inspect", "step": "Inspect files", "status": "completed"},
                    {"id": "edit", "step": "Apply the fix", "status": "in_progress"},
                    {"id": "verify", "step": "Run tests", "status": "pending"},
                ]
            },
        )
    )

    assert execution.ok is True
    assert execution.output == (
        "- [completed] inspect: Inspect files\n"
        "- [in_progress] edit: Apply the fix\n"
        "- [pending] verify: Run tests"
    )
    assert execution.metadata == {
        "revision": 1,
        "item_count": 3,
        "pending_count": 1,
        "in_progress_count": 1,
        "completed_count": 1,
    }
    assert all(type(value) is int for value in execution.metadata.values())
    assert execution.control.made_progress is True
    assert execution.control.invalidates_verification is False
    assert execution.control.verification is None
    assert "reason" not in (execution.output or "").lower()


def test_idempotent_tool_replay_does_not_report_progress_or_advance_revision() -> None:
    state = PlanState()
    arguments = [
        {"id": "inspect", "step": "Inspect files", "status": "in_progress"},
        {"id": "verify", "step": "Run tests", "status": "pending"},
    ]

    first = _execute(state, arguments)
    replay = _execute(state, arguments)

    assert first.ok is True
    assert first.control.made_progress is True
    assert replay.ok is True
    assert replay.control.made_progress is False
    assert replay.metadata["revision"] == 1
    assert state.revision == 1


def test_invalid_snapshot_is_a_structured_tool_failure_and_leaves_state_unchanged() -> None:
    state = PlanState()
    execution = _execute(
        state,
        [
            {"id": "same", "step": "First", "status": "pending"},
            {"id": "same", "step": "Second", "status": "pending"},
        ],
    )

    assert execution.ok is False
    assert execution.error_code == "plan_duplicate_id"
    assert state.items == ()
    assert state.revision == 0


def test_tool_schema_rejects_hidden_reasoning_and_describes_a_full_snapshot() -> None:
    tool = UpdatePlanTool(PlanState())
    schema = tool.spec.input_schema
    item_schema = schema["$defs"]["PlanItem"]

    assert tool.name == "update_plan"
    assert "complete" in tool.description.lower()
    assert "reasoning" in tool.description.lower()
    assert schema["additionalProperties"] is False
    assert item_schema["additionalProperties"] is False
    assert set(item_schema["properties"]) == {"id", "step", "status"}
    with pytest.raises(ValidationError):
        UpdatePlanArguments.model_validate(
            {
                "items": [
                    {
                        "id": "inspect",
                        "step": "Inspect files",
                        "status": "pending",
                        "reasoning": "private chain of thought",
                    }
                ]
            }
        )


def test_plan_snapshot_restore_preserves_revision_and_completed_item_monotonicity() -> None:
    original = PlanState()
    original.update(
        (
            _item("inspect", "Inspect files", PlanStatus.IN_PROGRESS),
            _item("verify", "Run tests"),
        )
    )
    original.update(
        (
            _item("inspect", "Inspect files", PlanStatus.COMPLETED),
            _item("verify", "Run tests", PlanStatus.IN_PROGRESS),
        )
    )

    snapshot = original.snapshot()
    restored = PlanState()

    assert snapshot.revision == 2
    assert restored.restore(snapshot) == snapshot
    assert restored.snapshot() == snapshot

    with pytest.raises(PlanError) as regressed:
        restored.update(
            (
                _item("inspect", "Inspect files", PlanStatus.PENDING),
                _item("verify", "Run tests", PlanStatus.IN_PROGRESS),
            )
        )
    assert regressed.value.code == "plan_completed_item_regressed"

    with pytest.raises(PlanError) as repeated_restore:
        restored.restore(snapshot)
    assert repeated_restore.value.code == "plan_restore_conflict"


def test_plan_snapshot_is_bounded_strict_and_has_one_valid_empty_state() -> None:
    empty = PlanState().snapshot()

    assert empty == PlanSnapshot(revision=0, items=())
    assert PlanState().restore(empty) == empty

    invalid = [
        {"revision": 0, "items": [_item("inspect", "Inspect")]},
        {"revision": 1, "items": []},
        {
            "revision": 1,
            "items": [
                _item("first", "First", PlanStatus.IN_PROGRESS),
                _item("second", "Second", PlanStatus.IN_PROGRESS),
            ],
        },
        {
            "revision": 1,
            "items": [_item("same", "First"), _item("same", "Second")],
        },
        {"revision": 0, "items": [], "reasoning": "must not persist"},
    ]
    for payload in invalid:
        with pytest.raises(ValidationError):
            PlanSnapshot.model_validate(payload)
