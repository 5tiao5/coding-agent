"""Project-owned explicit task-plan state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import ConfigDict, Field, field_validator

from coding_agent.errors import CodedError
from coding_agent.models import FrozenModel


class PlanError(CodedError):
    """Expected rejection of an invalid plan transition."""


class PlanStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class PlanItem(FrozenModel):
    """One explicit, observable unit of work; never an implicit reasoning trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(
        min_length=1,
        max_length=24,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,23}$",
        description="Short stable identifier for this explicit step.",
    )
    step: str = Field(
        min_length=1,
        max_length=200,
        description="One concise, observable action; never hidden reasoning or chain of thought.",
    )
    status: PlanStatus

    @field_validator("step")
    @classmethod
    def validate_step(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("step cannot have leading or trailing whitespace")
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("step must be one printable line")
        return value


@dataclass(frozen=True, slots=True)
class PlanUpdate:
    """Result of replacing the complete explicit plan snapshot."""

    items: tuple[PlanItem, ...]
    revision: int
    changed: bool


class PlanState:
    """A bounded in-memory plan with monotonic completed items."""

    def __init__(self) -> None:
        self._items: tuple[PlanItem, ...] = ()
        self._revision = 0

    @property
    def items(self) -> tuple[PlanItem, ...]:
        return self._items

    @property
    def revision(self) -> int:
        return self._revision

    def update(self, items: Sequence[PlanItem]) -> PlanUpdate:
        """Replace the whole snapshot after validating its monotonic invariants."""

        candidate = tuple(items)
        if not 1 <= len(candidate) <= 8:
            raise PlanError("plan_size_invalid", "plan must contain between 1 and 8 items")

        ids = tuple(item.id for item in candidate)
        duplicate_id = next((item_id for item_id in ids if ids.count(item_id) > 1), None)
        if duplicate_id is not None:
            raise PlanError("plan_duplicate_id", f"plan item id is duplicated: {duplicate_id}")

        in_progress = sum(item.status is PlanStatus.IN_PROGRESS for item in candidate)
        if in_progress > 1:
            raise PlanError(
                "plan_multiple_in_progress",
                "plan may contain at most one in_progress item",
            )

        candidate_by_id = {item.id: item for item in candidate}
        for previous in self._items:
            if previous.status is not PlanStatus.COMPLETED:
                continue
            current = candidate_by_id.get(previous.id)
            if current is None:
                raise PlanError(
                    "plan_completed_item_removed",
                    f"completed plan item cannot be removed: {previous.id}",
                )
            if current.status is not PlanStatus.COMPLETED:
                raise PlanError(
                    "plan_completed_item_regressed",
                    f"completed plan item cannot regress: {previous.id}",
                )
            if current.step != previous.step:
                raise PlanError(
                    "plan_completed_item_rewritten",
                    f"completed plan item cannot be rewritten: {previous.id}",
                )

        if candidate == self._items:
            return PlanUpdate(items=self._items, revision=self._revision, changed=False)

        self._items = candidate
        self._revision += 1
        return PlanUpdate(items=self._items, revision=self._revision, changed=True)
