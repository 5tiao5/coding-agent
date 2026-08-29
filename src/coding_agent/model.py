"""Model adapter boundary and a deterministic offline implementation."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

from coding_agent.errors import CodedError
from coding_agent.models import ChatMessage, ModelResponse, ToolSpec


class RetryableModelError(CodedError):
    """A sanitized transient model failure that the project-owned loop may retry."""


class ModelAdapter(Protocol):
    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> ModelResponse:
        """Return one provider-neutral model turn."""


@dataclass(frozen=True, slots=True)
class RecordedModelRequest:
    messages: tuple[ChatMessage, ...]
    tools: tuple[ToolSpec, ...]


class ScriptedModel:
    """Deterministic fake model that returns a predefined sequence of turns."""

    def __init__(self, responses: Iterable[ModelResponse]) -> None:
        self._responses = deque(responses)
        self.requests: list[RecordedModelRequest] = []

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> ModelResponse:
        self.requests.append(RecordedModelRequest(tuple(messages), tuple(tools)))
        if not self._responses:
            raise RuntimeError("scripted model has no response remaining")
        return self._responses.popleft()
