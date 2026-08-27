"""Structured, provider-neutral events emitted by the agent runtime."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from pydantic import Field

from coding_agent.models import FrozenModel


class EventKind(StrEnum):
    RUN_STARTED = "run_started"
    STATE_CHANGED = "state_changed"
    MODEL_REQUESTED = "model_requested"
    MODEL_RESPONDED = "model_responded"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    RUN_FINISHED = "run_finished"
    RUN_FAILED = "run_failed"


class RunEvent(FrozenModel):
    run_id: str
    kind: EventKind
    message: str
    step: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    data: dict[str, Any] = Field(default_factory=dict)


class EventSink(Protocol):
    def emit(self, event: RunEvent) -> None:
        """Consume one runtime event."""


class NullEventSink:
    def emit(self, event: RunEvent) -> None:
        del event


class MemoryEventSink:
    """In-memory sink useful for deterministic tests and future replay support."""

    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def emit(self, event: RunEvent) -> None:
        self.events.append(event.model_copy(deep=True))


class CompositeEventSink:
    def __init__(self, *sinks: EventSink) -> None:
        self._sinks = sinks

    def emit(self, event: RunEvent) -> None:
        for sink in self._sinks:
            # A sink is an integration boundary. Give each sink its own snapshot so
            # one renderer cannot mutate what a later recorder observes.
            sink.emit(event.model_copy(deep=True))
