"""Model-facing tool for explicit structured plan snapshots."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

from coding_agent.models import ToolControlFacts, ToolOutput
from coding_agent.plan import PlanItem, PlanState, PlanStatus
from coding_agent.tools.base import BaseTool


class UpdatePlanArguments(BaseModel):
    """A complete replacement snapshot, not a request for model reasoning."""

    model_config = ConfigDict(extra="forbid")

    items: tuple[PlanItem, ...] = Field(
        min_length=1,
        max_length=8,
        description=(
            "Complete ordered plan snapshot with explicit steps and statuses only. "
            "Do not include analysis, hidden reasoning, or chain of thought."
        ),
    )


class UpdatePlanTool(BaseTool[UpdatePlanArguments]):
    name = "update_plan"
    description = (
        "Replace the complete explicit task plan with 1-8 concise steps and their statuses. "
        "Use at most one in_progress item. Never provide analysis or hidden reasoning."
    )
    args_model = UpdatePlanArguments
    output_budget_chars = 2_500

    def __init__(self, state: PlanState) -> None:
        self._state = state

    def run(self, arguments: UpdatePlanArguments) -> ToolOutput:
        update = self._state.update(arguments.items)
        counts = Counter(item.status for item in update.items)
        content = "\n".join(
            f"- [{item.status.value}] {item.id}: {item.step}" for item in update.items
        )
        pending = counts[PlanStatus.PENDING]
        in_progress = counts[PlanStatus.IN_PROGRESS]
        completed = counts[PlanStatus.COMPLETED]
        return ToolOutput(
            content=content,
            summary=(
                f"Plan revision {update.revision}: {pending} pending, "
                f"{in_progress} in progress, {completed} completed"
            ),
            metadata={
                "revision": update.revision,
                "item_count": len(update.items),
                "pending_count": pending,
                "in_progress_count": in_progress,
                "completed_count": completed,
            },
            control=ToolControlFacts(
                made_progress=update.changed,
                invalidates_verification=False,
            ),
        )
