"""Core tool-dispatch contract consumed by the agent runtime."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from coding_agent.models import ToolCall, ToolExecution, ToolSpec


class ToolDispatcher(Protocol):
    """The minimal tool boundary consumed by the agent loop."""

    def specs(self) -> Sequence[ToolSpec]:
        """Describe tools exposed to the model."""

    def execute(self, call: ToolCall) -> ToolExecution:
        """Execute one validated tool call."""
