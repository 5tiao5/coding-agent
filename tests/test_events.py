"""Tests for event snapshot ownership across sink boundaries."""

from __future__ import annotations

from coding_agent.events import (
    BestEffortEventSink,
    CompositeEventSink,
    EventKind,
    MemoryEventSink,
    RunEvent,
)


class MutatingSink:
    def __init__(self) -> None:
        self.received: RunEvent | None = None

    def emit(self, event: RunEvent) -> None:
        self.received = event
        event.data["details"]["status"] = "mutated"


def test_composite_sink_isolates_each_consumer_and_the_source_event() -> None:
    mutating = MutatingSink()
    recording = MemoryEventSink()
    source = RunEvent(
        run_id="run-1",
        kind=EventKind.RUN_STARTED,
        message="started",
        data={"details": {"status": "original"}},
    )

    CompositeEventSink(mutating, recording).emit(source)

    assert mutating.received is not None
    assert mutating.received.data["details"]["status"] == "mutated"
    assert source.data["details"]["status"] == "original"
    assert recording.events[0].data["details"]["status"] == "original"
    assert mutating.received is not recording.events[0]


def test_memory_sink_records_a_deep_snapshot() -> None:
    recording = MemoryEventSink()
    source = RunEvent(
        run_id="run-1",
        kind=EventKind.RUN_STARTED,
        message="started",
        data={"details": {"status": "original"}},
    )

    recording.emit(source)
    source.data["details"]["status"] = "changed later"

    assert recording.events[0].data["details"]["status"] == "original"


def test_best_effort_sink_disables_a_broken_renderer_without_raising() -> None:
    class BrokenSink:
        def __init__(self) -> None:
            self.calls = 0

        def emit(self, event: RunEvent) -> None:
            del event
            self.calls += 1
            raise RuntimeError("terminal renderer failed")

    broken = BrokenSink()
    guarded = BestEffortEventSink(broken)
    event = RunEvent(run_id="run-1", kind=EventKind.RUN_STARTED, message="started")

    guarded.emit(event)
    guarded.emit(event)

    assert guarded.disabled is True
    assert guarded.failure_count == 1
    assert broken.calls == 1
