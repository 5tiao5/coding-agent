"""Durable, provider-neutral JSONL traces for observable agent runs.

The trace is deliberately narrower than a session checkpoint: it records only
``RunEvent`` values already emitted by the runtime.  It never serializes model
objects, prompts, hidden state, or provider-specific response payloads.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import Field, ValidationError

from coding_agent.events import EventKind, RunEvent
from coding_agent.models import FrozenModel
from coding_agent.run_id import require_run_id
from coding_agent.state import default_state_paths

DEFAULT_MAX_EVENT_BYTES = 256 * 1024
DEFAULT_MAX_TRACE_BYTES = 32 * 1024 * 1024

_TRACE_SUFFIX = ".jsonl"
_EVENT_FIELDS = frozenset(RunEvent.model_fields)
_APPEND_LOCK = Lock()


class TraceError(Exception):
    """Base class for trace persistence and validation failures."""


class InvalidRunIdError(TraceError):
    """A run id cannot be represented as one safe trace filename."""


class TraceNotFoundError(TraceError):
    """The requested run has no trace file."""


class TraceCorruptError(TraceError):
    """A trace is incomplete or does not contain canonical RunEvent records."""


class TraceTooLargeError(TraceError):
    """An event or trace exceeds its configured resource bound."""


class TraceWriteError(TraceError):
    """An event could not be durably appended to its trace."""


class TraceRunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TraceRunInfo(FrozenModel):
    """Cheap metadata used to enumerate recent traces without loading them."""

    run_id: str
    modified_at: datetime
    size_bytes: int = Field(ge=0)


class TraceSummary(FrozenModel):
    """Provider-neutral facts suitable for a CLI ``inspect`` view."""

    run_id: str
    status: TraceRunStatus
    event_count: int = Field(ge=1)
    first_event_at: datetime
    last_event_at: datetime
    duration_ms: float = Field(ge=0)
    max_step: int = Field(ge=0)
    model_requests: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    tool_failures: int = Field(ge=0)
    verified: bool | None = None
    verification_status: str | None = None
    terminal_message: str | None = None


def default_trace_dir() -> Path:
    """Return a conventional per-user state directory without creating it."""

    return default_state_paths().traces


def validate_run_id(run_id: str) -> str:
    """Validate and return a run id safe for direct use as a filename stem."""
    try:
        return require_run_id(run_id)
    except ValueError as exc:
        raise InvalidRunIdError(str(exc)) from None


class TraceStore:
    """Append, read, enumerate, and summarize bounded per-run JSONL traces."""

    def __init__(
        self,
        state_dir: str | Path | None = None,
        *,
        max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES,
        max_trace_bytes: int = DEFAULT_MAX_TRACE_BYTES,
    ) -> None:
        if (
            isinstance(max_event_bytes, bool)
            or not isinstance(max_event_bytes, int)
            or max_event_bytes <= 0
        ):
            raise ValueError("max_event_bytes must be a positive integer")
        if (
            isinstance(max_trace_bytes, bool)
            or not isinstance(max_trace_bytes, int)
            or max_trace_bytes <= 0
        ):
            raise ValueError("max_trace_bytes must be a positive integer")
        if max_event_bytes > max_trace_bytes:
            raise ValueError("max_event_bytes cannot exceed max_trace_bytes")
        raw_state_dir = Path(state_dir) if state_dir is not None else default_trace_dir()
        if not raw_state_dir.is_absolute():
            raise ValueError("trace state_dir must be absolute")
        if raw_state_dir.is_symlink():
            raise ValueError("trace state_dir cannot be a symbolic link")
        self._state_dir = raw_state_dir.resolve(strict=False)
        self._max_event_bytes = max_event_bytes
        self._max_trace_bytes = max_trace_bytes

    @property
    def state_dir(self) -> Path:
        return self._state_dir

    def path_for(self, run_id: str) -> Path:
        return self._state_dir / f"{validate_run_id(run_id)}{_TRACE_SUFFIX}"

    def append(self, event: RunEvent) -> None:
        """Durably append one event using one bounded ``O_APPEND`` write."""

        if not isinstance(event, RunEvent):
            raise TypeError("trace sinks accept RunEvent values only")
        validate_run_id(event.run_id)
        try:
            record = (
                json.dumps(
                    event.model_dump(mode="json"),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
        except Exception as exc:  # noqa: BLE001 - normalize arbitrary Any serialization failures.
            raise TraceWriteError(
                f"event for run {event.run_id!r} is not JSON serializable"
            ) from exc

        if len(record) > self._max_event_bytes:
            raise TraceTooLargeError(
                f"event is {len(record)} bytes; limit is {self._max_event_bytes} bytes"
            )
        self._ensure_state_dir()
        path = self.path_for(event.run_id)

        # A process-local lock prevents thread interleaving. O_APPEND plus a single
        # os.write keeps independent processes from overwriting one another's records.
        with _APPEND_LOCK:
            self._append_record(path, record)

    def read(self, run_id: str) -> tuple[RunEvent, ...]:
        """Load and strictly validate every event for one run."""

        path = self.path_for(run_id)
        raw = self._read_bounded(path)
        if not raw:
            raise TraceCorruptError(f"trace for run {run_id!r} is empty")
        if not raw.endswith(b"\n"):
            raise TraceCorruptError(f"trace for run {run_id!r} ends with an incomplete record")

        events: list[RunEvent] = []
        for line_number, raw_line in enumerate(raw[:-1].split(b"\n"), start=1):
            if not raw_line:
                raise TraceCorruptError(
                    f"trace for run {run_id!r} contains a blank record at line {line_number}"
                )
            if len(raw_line) + 1 > self._max_event_bytes:
                raise TraceTooLargeError(
                    f"trace line {line_number} is larger than {self._max_event_bytes} bytes"
                )
            events.append(_decode_event(raw_line, run_id=run_id, line_number=line_number))
        return tuple(events)

    def list_runs(self, *, limit: int = 20) -> tuple[TraceRunInfo, ...]:
        """Enumerate recent regular trace files, newest modification first."""

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer between 1 and 1000")
        if not self._state_dir.exists():
            return ()
        if not self._state_dir.is_dir():
            raise TraceError(f"trace state path is not a directory: {self._state_dir}")

        candidates: list[tuple[int, TraceRunInfo]] = []
        try:
            entries = self._state_dir.iterdir()
            for path in entries:
                if path.suffix != _TRACE_SUFFIX or path.is_symlink() or not path.is_file():
                    continue
                try:
                    run_id = validate_run_id(path.stem)
                except InvalidRunIdError:
                    continue
                file_stat = path.stat()
                if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
                    continue
                candidates.append(
                    (
                        file_stat.st_mtime_ns,
                        TraceRunInfo(
                            run_id=run_id,
                            modified_at=datetime.fromtimestamp(file_stat.st_mtime, UTC),
                            size_bytes=file_stat.st_size,
                        ),
                    )
                )
        except OSError as exc:
            raise TraceError(
                f"could not enumerate trace directory {self._state_dir}: {exc}"
            ) from exc

        candidates.sort(key=lambda item: (item[0], item[1].run_id), reverse=True)
        return tuple(info for _, info in candidates[:limit])

    def summarize(self, run_id: str) -> TraceSummary:
        return summarize_events(self.read(run_id))

    def _ensure_state_dir(self) -> None:
        try:
            self._state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as exc:
            raise TraceWriteError(
                f"could not create trace directory {self._state_dir}: {exc}"
            ) from exc
        if self._state_dir.is_symlink() or not self._state_dir.is_dir():
            raise TraceWriteError(f"trace state path is not a directory: {self._state_dir}")

    def _append_record(self, path: Path, record: bytes) -> None:
        flags = os.O_APPEND | os.O_CREAT | os.O_RDWR
        flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        if path.is_symlink():
            raise TraceWriteError(f"refusing to append through trace symlink: {path}")

        descriptor: int | None = None
        try:
            descriptor = os.open(path, flags, 0o600)
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
                raise TraceWriteError(f"trace path is not a private regular file: {path}")
            if file_stat.st_size > self._max_trace_bytes:
                raise TraceTooLargeError(
                    f"trace is {file_stat.st_size} bytes; limit is {self._max_trace_bytes} bytes"
                )
            if file_stat.st_size + len(record) > self._max_trace_bytes:
                raise TraceTooLargeError(
                    f"appending event would exceed the {self._max_trace_bytes}-byte trace limit"
                )
            if file_stat.st_size:
                os.lseek(descriptor, -1, os.SEEK_END)
                if os.read(descriptor, 1) != b"\n":
                    raise TraceCorruptError(f"refusing to append to incomplete trace: {path}")
            written = os.write(descriptor, record)
            if written != len(record):
                raise TraceWriteError(
                    f"short trace write for {path}: wrote {written} of {len(record)} bytes"
                )
            os.fsync(descriptor)
        except TraceError:
            raise
        except OSError as exc:
            raise TraceWriteError(f"could not append trace event to {path}: {exc}") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _read_bounded(self, path: Path) -> bytes:
        flags = os.O_RDONLY
        flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        if path.is_symlink():
            raise TraceCorruptError(f"refusing to read trace symlink: {path}")

        descriptor: int | None = None
        try:
            descriptor = os.open(path, flags)
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
                raise TraceCorruptError(f"trace path is not a private regular file: {path}")
            if file_stat.st_size > self._max_trace_bytes:
                raise TraceTooLargeError(
                    f"trace is {file_stat.st_size} bytes; limit is {self._max_trace_bytes} bytes"
                )
            raw = bytearray()
            while len(raw) <= self._max_trace_bytes:
                chunk = os.read(
                    descriptor,
                    min(64 * 1024, self._max_trace_bytes + 1 - len(raw)),
                )
                if not chunk:
                    return bytes(raw)
                raw.extend(chunk)
            raise TraceTooLargeError(
                f"trace grew beyond {self._max_trace_bytes} bytes while reading"
            )
        except FileNotFoundError as exc:
            raise TraceNotFoundError(f"no trace exists for run {path.stem!r}") from exc
        except TraceError:
            raise
        except OSError as exc:
            raise TraceError(f"could not read trace {path}: {exc}") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)


class JsonlEventSink:
    """An ``EventSink``-compatible adapter backed by a :class:`TraceStore`."""

    def __init__(
        self,
        state_dir: str | Path | None = None,
        *,
        max_event_bytes: int = DEFAULT_MAX_EVENT_BYTES,
        max_trace_bytes: int = DEFAULT_MAX_TRACE_BYTES,
    ) -> None:
        self._store = TraceStore(
            state_dir,
            max_event_bytes=max_event_bytes,
            max_trace_bytes=max_trace_bytes,
        )

    @property
    def store(self) -> TraceStore:
        return self._store

    def emit(self, event: RunEvent) -> None:
        self._store.append(event)


def summarize_events(events: Sequence[RunEvent]) -> TraceSummary:
    """Summarize already validated events without any model-provider assumptions."""

    if not events:
        raise TraceCorruptError("cannot summarize an empty event sequence")
    run_id = events[0].run_id
    if any(event.run_id != run_id for event in events):
        raise TraceCorruptError("cannot summarize events from multiple runs")

    latest_segment_start = max(
        (
            index
            for index, event in enumerate(events)
            if event.kind in {EventKind.RUN_STARTED, EventKind.RUN_RESUMED}
        ),
        default=0,
    )
    latest_segment = events[latest_segment_start:]
    terminal = next(
        (
            event
            for event in reversed(latest_segment)
            if event.kind in {EventKind.RUN_FINISHED, EventKind.RUN_FAILED}
        ),
        None,
    )
    if terminal is None:
        status = TraceRunStatus.RUNNING
    elif terminal.kind is EventKind.RUN_FAILED:
        status = TraceRunStatus.FAILED
    else:
        status = TraceRunStatus.COMPLETED

    verification = next(
        (
            event
            for event in reversed(latest_segment)
            if event.kind is EventKind.VERIFICATION_EVALUATED
        ),
        None,
    )
    verified_value = verification.data.get("verified") if verification else None
    verified = verified_value if isinstance(verified_value, bool) else None
    verification_status_value = verification.data.get("status") if verification else None
    verification_status = (
        verification_status_value if isinstance(verification_status_value, str) else None
    )
    duration_ms = max(
        0.0,
        (events[-1].timestamp - events[0].timestamp).total_seconds() * 1000,
    )

    return TraceSummary(
        run_id=run_id,
        status=status,
        event_count=len(events),
        first_event_at=events[0].timestamp,
        last_event_at=events[-1].timestamp,
        duration_ms=duration_ms,
        max_step=max(event.step for event in events),
        model_requests=sum(event.kind is EventKind.MODEL_REQUESTED for event in events),
        tool_calls=sum(event.kind is EventKind.TOOL_STARTED for event in events),
        tool_failures=sum(
            event.kind is EventKind.TOOL_FINISHED and event.data.get("ok") is False
            for event in events
        ),
        verified=verified,
        verification_status=verification_status,
        terminal_message=terminal.message if terminal else None,
    )


def _decode_event(raw_line: bytes, *, run_id: str, line_number: int) -> RunEvent:
    try:
        text = raw_line.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise TraceCorruptError(
            f"trace for run {run_id!r} has invalid UTF-8 at line {line_number}"
        ) from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_non_finite_number,
        )
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise TraceCorruptError(
            f"trace for run {run_id!r} has invalid JSON at line {line_number}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise TraceCorruptError(f"trace line {line_number} must contain one JSON object")

    fields = frozenset(payload)
    unknown = fields - _EVENT_FIELDS
    missing = _EVENT_FIELDS - fields
    if unknown or missing:
        details: list[str] = []
        if unknown:
            details.append(f"unknown fields: {', '.join(sorted(unknown))}")
        if missing:
            details.append(f"missing fields: {', '.join(sorted(missing))}")
        raise TraceCorruptError(f"trace line {line_number} has {'; '.join(details)}")
    if not _has_strict_event_field_types(payload):
        raise TraceCorruptError(f"trace line {line_number} has invalid RunEvent field types")

    try:
        event = RunEvent.model_validate(payload)
    except ValidationError as exc:
        raise TraceCorruptError(
            f"trace line {line_number} is not a valid RunEvent: {exc.errors(include_url=False)}"
        ) from exc
    if event.run_id != run_id:
        raise TraceCorruptError(
            f"trace line {line_number} belongs to run {event.run_id!r}, expected {run_id!r}"
        )
    if event.timestamp.utcoffset() is None:
        raise TraceCorruptError(f"trace line {line_number} has a timezone-naive timestamp")
    return event


def _has_strict_event_field_types(payload: dict[str, Any]) -> bool:
    step = payload.get("step")
    return (
        isinstance(payload.get("run_id"), str)
        and isinstance(payload.get("kind"), str)
        and isinstance(payload.get("message"), str)
        and isinstance(step, int)
        and not isinstance(step, bool)
        and isinstance(payload.get("timestamp"), str)
        and isinstance(payload.get("data"), dict)
    )


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_non_finite_number(value: str) -> Any:
    raise ValueError(f"non-finite number {value!r} is not valid trace JSON")
