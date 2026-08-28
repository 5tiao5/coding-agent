"""Tests for bounded, strict, provider-neutral JSONL event traces."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from coding_agent.events import EventKind, EventSink, RunEvent
from coding_agent.trace import (
    InvalidRunIdError,
    JsonlEventSink,
    TraceCorruptError,
    TraceError,
    TraceNotFoundError,
    TraceRunStatus,
    TraceStore,
    TraceTooLargeError,
    TraceWriteError,
    default_trace_dir,
    summarize_events,
    validate_run_id,
)

_NOW = datetime(2026, 8, 28, 8, 0, tzinfo=UTC)


def _event(
    run_id: str = "run-1",
    *,
    kind: EventKind = EventKind.RUN_STARTED,
    message: str = "开始运行",
    step: int = 0,
    timestamp: datetime = _NOW,
    data: dict[str, Any] | None = None,
) -> RunEvent:
    return RunEvent(
        run_id=run_id,
        kind=kind,
        message=message,
        step=step,
        timestamp=timestamp,
        data=data or {},
    )


def _record_bytes(event: RunEvent, **changes: Any) -> bytes:
    payload = event.model_dump(mode="json")
    payload.update(changes)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_jsonl_sink_is_event_sink_and_round_trips_utf8_one_event_per_line(tmp_path: Path) -> None:
    sink: EventSink = JsonlEventSink(tmp_path)
    first = _event(data={"detail": {"语言": "中文"}})
    second = _event(
        kind=EventKind.MODEL_REQUESTED,
        message="请求模型",
        step=1,
        timestamp=_NOW + timedelta(seconds=1),
    )

    sink.emit(first)
    sink.emit(second)

    raw = (tmp_path / "run-1.jsonl").read_bytes()
    assert raw.count(b"\n") == 2
    assert "中文" in raw.decode("utf-8")
    assert JsonlEventSink(tmp_path).store.read("run-1") == (first, second)


def test_sink_accepts_only_run_events(tmp_path: Path) -> None:
    sink = JsonlEventSink(tmp_path)

    with pytest.raises(TypeError, match="RunEvent"):
        sink.emit(cast(Any, {"run_id": "run-1"}))


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        ".",
        "../escape",
        "a/b",
        "a\\b",
        " space",
        "Uppercase",
        "CON",
        "nul.txt",
        "demo.run",
        "a" * 65,
    ],
)
def test_run_id_rejects_unsafe_or_nonportable_filenames(run_id: str) -> None:
    with pytest.raises(InvalidRunIdError):
        validate_run_id(run_id)


def test_run_id_accepts_runtime_and_human_readable_ids() -> None:
    assert validate_run_id("b4e9a3ff1a2b4c5d") == "b4e9a3ff1a2b4c5d"
    assert validate_run_id("demo-run-02_alpha") == "demo-run-02_alpha"


def test_default_trace_dir_uses_shared_private_state_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("CODING_AGENT_STATE_DIR", str(tmp_path))

    assert default_trace_dir() == tmp_path.resolve() / "traces"


@pytest.mark.parametrize(
    ("event_limit", "trace_limit"),
    [
        (0, 10),
        (-1, 10),
        (True, 10),
        (cast(Any, 1.5), 10),
        (11, 10),
        (1, 0),
        (1, False),
        (1, cast(Any, "10")),
    ],
)
def test_store_rejects_invalid_resource_limits(
    tmp_path: Path, event_limit: int, trace_limit: int
) -> None:
    with pytest.raises(ValueError):
        TraceStore(tmp_path, max_event_bytes=event_limit, max_trace_bytes=trace_limit)


def test_append_rejects_oversized_or_unserializable_events_before_writing(
    tmp_path: Path,
) -> None:
    tiny = TraceStore(tmp_path / "tiny", max_event_bytes=1, max_trace_bytes=2)
    with pytest.raises(TraceTooLargeError, match="event is"):
        tiny.append(_event())
    assert not tiny.state_dir.exists()

    store = TraceStore(tmp_path / "invalid")
    with pytest.raises(TraceWriteError, match="not JSON serializable"):
        store.append(_event(data={"opaque": object()}))


def test_append_enforces_total_size_and_refuses_incomplete_existing_trace(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    store.append(_event())
    path = store.path_for("run-1")
    first_size = path.stat().st_size
    second = _event(kind=EventKind.MODEL_REQUESTED)
    second_size = len(_record_bytes(second))
    bounded = TraceStore(
        tmp_path,
        max_event_bytes=second_size,
        max_trace_bytes=first_size + second_size - 1,
    )
    with pytest.raises(TraceTooLargeError, match="would exceed"):
        bounded.append(second)

    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(TraceCorruptError, match="incomplete"):
        store.append(_event(kind=EventKind.MODEL_REQUESTED))


def test_append_refuses_non_regular_target_and_short_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory_target = tmp_path / "run-1.jsonl"
    directory_target.mkdir()
    with pytest.raises(TraceWriteError):
        TraceStore(tmp_path).append(_event())

    short_dir = tmp_path / "short"
    short_store = TraceStore(short_dir)

    def short_write(descriptor: int, content: bytes) -> int:
        del descriptor
        return len(content) - 1

    monkeypatch.setattr(os, "write", short_write)
    with pytest.raises(TraceWriteError, match="short trace write"):
        short_store.append(_event())


def test_trace_store_rejects_relative_state_and_hardlinked_records(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        TraceStore(Path("relative-traces"))

    store = TraceStore(tmp_path)
    store.append(_event())
    path = store.path_for("run-1")
    os.link(path, tmp_path / "trace-alias.jsonl")

    with pytest.raises(TraceCorruptError, match="private regular file"):
        store.read("run-1")
    with pytest.raises(TraceWriteError, match="private regular file"):
        store.append(_event(kind=EventKind.MODEL_REQUESTED))


def test_trace_store_rejects_a_state_directory_replaced_by_a_symlink(tmp_path: Path) -> None:
    state = tmp_path / "traces"
    target = tmp_path / "redirected"
    target.mkdir()
    store = TraceStore(state)
    try:
        state.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this platform")

    with pytest.raises(TraceWriteError, match="not a directory"):
        store.append(_event())
    assert not target.joinpath("run-1.jsonl").exists()


def test_read_reports_missing_empty_incomplete_and_blank_traces(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    with pytest.raises(TraceNotFoundError, match="no trace"):
        store.read("missing")

    path = store.path_for("run-1")
    _write(path, b"")
    with pytest.raises(TraceCorruptError, match="empty"):
        store.read("run-1")

    path.write_bytes(_record_bytes(_event()).rstrip(b"\n"))
    with pytest.raises(TraceCorruptError, match="incomplete"):
        store.read("run-1")

    path.write_bytes(_record_bytes(_event()) + b"\n")
    with pytest.raises(TraceCorruptError, match="blank record"):
        store.read("run-1")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"\xff\n", "invalid UTF-8"),
        (b"{not-json}\n", "invalid JSON"),
        (b'{"run_id":"run-1","run_id":"run-1"}\n', "duplicate JSON field"),
        (b'{"value":NaN}\n', "non-finite number"),
        (b"[]\n", "JSON object"),
    ],
)
def test_read_rejects_noncanonical_json(tmp_path: Path, content: bytes, message: str) -> None:
    store = TraceStore(tmp_path)
    _write(store.path_for("run-1"), content)

    with pytest.raises(TraceCorruptError, match=message):
        store.read("run-1")


def test_read_normalizes_json_recursion_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = TraceStore(tmp_path)
    _write(store.path_for("run-1"), b"{}\n")

    def raise_recursion(*args: object, **kwargs: object) -> object:
        raise RecursionError("malicious nesting")

    monkeypatch.setattr("coding_agent.trace.json.loads", raise_recursion)

    with pytest.raises(TraceCorruptError, match="invalid JSON"):
        store.read("run-1")


def test_read_rejects_unknown_missing_or_wrongly_typed_event_fields(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    path = store.path_for("run-1")

    path.write_bytes(_record_bytes(_event(), surprise=True))
    with pytest.raises(TraceCorruptError, match="unknown fields: surprise"):
        store.read("run-1")

    payload = _event().model_dump(mode="json")
    del payload["message"]
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(TraceCorruptError, match="missing fields: message"):
        store.read("run-1")

    path.write_bytes(_record_bytes(_event(), step="1"))
    with pytest.raises(TraceCorruptError, match="field types"):
        store.read("run-1")


def test_read_rejects_invalid_event_cross_run_and_naive_timestamp(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    path = store.path_for("run-1")

    path.write_bytes(_record_bytes(_event(), kind="future_kind"))
    with pytest.raises(TraceCorruptError, match="not a valid RunEvent"):
        store.read("run-1")

    path.write_bytes(_record_bytes(_event("other")))
    with pytest.raises(TraceCorruptError, match="belongs to run"):
        store.read("run-1")

    path.write_bytes(_record_bytes(_event(), timestamp="2026-08-28T08:00:00"))
    with pytest.raises(TraceCorruptError, match="timezone-naive"):
        store.read("run-1")


def test_read_enforces_whole_file_and_per_record_bounds(tmp_path: Path) -> None:
    path = tmp_path / "run-1.jsonl"
    _write(path, b"x" * 11)
    with pytest.raises(TraceTooLargeError, match="trace is"):
        TraceStore(tmp_path, max_event_bytes=5, max_trace_bytes=10).read("run-1")

    record = _record_bytes(_event())
    path.write_bytes(record)
    with pytest.raises(TraceTooLargeError, match="trace line"):
        TraceStore(
            tmp_path,
            max_event_bytes=len(record) - 1,
            max_trace_bytes=len(record),
        ).read("run-1")


def test_list_runs_is_bounded_sorted_and_ignores_non_trace_entries(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    store.append(_event("older"))
    store.append(_event("newer"))
    older = store.path_for("older")
    newer = store.path_for("newer")
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    (tmp_path / "bad name.jsonl").write_text("ignore", encoding="utf-8")
    (tmp_path / "directory.jsonl").mkdir()

    recent = store.list_runs(limit=1)

    assert [item.run_id for item in recent] == ["newer"]
    assert recent[0].size_bytes == newer.stat().st_size
    assert recent[0].modified_at == datetime.fromtimestamp(newer.stat().st_mtime, UTC)


def test_list_runs_handles_absent_or_invalid_state_paths(tmp_path: Path) -> None:
    absent = TraceStore(tmp_path / "absent")
    assert absent.list_runs() == ()
    for invalid_limit in (0, 1001, True):
        with pytest.raises(ValueError, match="limit"):
            absent.list_runs(limit=cast(Any, invalid_limit))

    state_file = tmp_path / "file"
    state_file.write_text("not a directory", encoding="utf-8")
    with pytest.raises(TraceError, match="not a directory"):
        TraceStore(state_file).list_runs()


def test_summary_reports_provider_neutral_runtime_facts_and_terminal_event() -> None:
    events = (
        _event(),
        _event(
            kind=EventKind.MODEL_REQUESTED,
            step=1,
            timestamp=_NOW + timedelta(seconds=1),
        ),
        _event(
            kind=EventKind.TOOL_STARTED,
            step=1,
            timestamp=_NOW + timedelta(seconds=2),
        ),
        _event(
            kind=EventKind.TOOL_FINISHED,
            step=1,
            timestamp=_NOW + timedelta(seconds=3),
            data={"ok": False},
        ),
        _event(
            kind=EventKind.VERIFICATION_EVALUATED,
            step=2,
            timestamp=_NOW + timedelta(seconds=4),
            data={"verified": True, "status": "verified"},
        ),
        _event(
            kind=EventKind.RUN_FINISHED,
            message="Verified completion",
            step=2,
            timestamp=_NOW + timedelta(seconds=5),
        ),
        _event(
            kind=EventKind.SESSION_CHECKPOINTED,
            step=2,
            timestamp=_NOW + timedelta(seconds=6),
        ),
    )

    summary = summarize_events(events)

    assert summary.status is TraceRunStatus.COMPLETED
    assert summary.event_count == 7
    assert summary.duration_ms == 6000
    assert summary.max_step == 2
    assert summary.model_requests == 1
    assert summary.tool_calls == 1
    assert summary.tool_failures == 1
    assert summary.verified is True
    assert summary.verification_status == "verified"
    assert summary.terminal_message == "Verified completion"


def test_store_summary_and_running_failed_or_untyped_verification_states(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    store.append(_event("persisted"))
    assert store.summarize("persisted").status is TraceRunStatus.RUNNING

    failed = (
        _event("failed", timestamp=_NOW + timedelta(seconds=2)),
        _event(
            "failed",
            kind=EventKind.VERIFICATION_EVALUATED,
            timestamp=_NOW + timedelta(seconds=1),
            data={"verified": "yes", "status": 42},
        ),
        _event("failed", kind=EventKind.RUN_FAILED, message="boom"),
    )
    summary = summarize_events(failed)
    assert summary.status is TraceRunStatus.FAILED
    assert summary.duration_ms == 0
    assert summary.verified is None
    assert summary.verification_status is None
    assert summary.terminal_message == "boom"


def test_summary_resume_segment_does_not_reuse_old_terminal_or_verification() -> None:
    events = (
        _event("resumed"),
        _event(
            "resumed",
            kind=EventKind.VERIFICATION_EVALUATED,
            data={"verified": True, "status": "verified"},
        ),
        _event("resumed", kind=EventKind.RUN_FINISHED, message="old completion"),
        _event(
            "resumed",
            kind=EventKind.RUN_RESUMED,
            message="fresh verification required",
            timestamp=_NOW + timedelta(seconds=1),
        ),
    )

    summary = summarize_events(events)

    assert summary.status is TraceRunStatus.RUNNING
    assert summary.verified is None
    assert summary.verification_status is None
    assert summary.terminal_message is None


def test_summary_rejects_empty_or_mixed_run_sequences() -> None:
    with pytest.raises(TraceCorruptError, match="empty"):
        summarize_events(())
    with pytest.raises(TraceCorruptError, match="multiple runs"):
        summarize_events((_event("one"), _event("two")))
