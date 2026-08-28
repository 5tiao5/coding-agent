"""Deterministic stop policies for repeated no-progress tool observations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

from coding_agent.models import ToolCall, ToolExecution


@dataclass(frozen=True, slots=True)
class RepetitionObservation:
    streak: int
    should_stop: bool


class RepeatedToolCallGuard:
    """Stop only after the same request produces the same outcome repeatedly."""

    def __init__(self, *, max_identical: int = 3) -> None:
        if max_identical < 2:
            raise ValueError("max_identical must be at least 2")
        self._max_identical = max_identical
        self._last_pair: tuple[str, str] | None = None
        self._streak = 0

    def observe(self, call: ToolCall, execution: ToolExecution) -> RepetitionObservation:
        if execution.control.made_progress:
            self.reset()
            return RepetitionObservation(streak=0, should_stop=False)

        pair = (_call_fingerprint(call), _outcome_fingerprint(execution))
        if pair == self._last_pair:
            self._streak += 1
        else:
            self._last_pair = pair
            self._streak = 1
        return RepetitionObservation(
            streak=self._streak,
            should_stop=self._streak >= self._max_identical,
        )

    def reset(self) -> None:
        self._last_pair = None
        self._streak = 0


def _call_fingerprint(call: ToolCall) -> str:
    return _fingerprint({"name": call.name, "arguments": call.arguments})


def _outcome_fingerprint(execution: ToolExecution) -> str:
    return _fingerprint(
        execution.model_dump(
            mode="json",
            exclude={"call_id", "duration_ms", "summary"},
            exclude_defaults=True,
            exclude_none=True,
        )
    )


def _fingerprint(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_stable_fallback,
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def _stable_fallback(value: object) -> str:
    return f"<{type(value).__module__}.{type(value).__qualname__}>"
