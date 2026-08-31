"""Typed local tool definition and dispatch owned by this project."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from time import perf_counter
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from coding_agent.errors import CodedError, ErrorMetadataValue
from coding_agent.models import ToolCall, ToolExecution, ToolOutput, ToolSpec
from coding_agent.tooling import ToolDispatcher as ToolDispatcher

ArgsT = TypeVar("ArgsT", bound=BaseModel)


class ToolError(CodedError):
    """Expected tool failure that can be reported to the model safely."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        metadata: Mapping[str, ErrorMetadataValue] | None = None,
    ) -> None:
        super().__init__(
            code.strip() or "tool_error",
            message.strip() or "tool failed",
            metadata=metadata,
        )


class BaseTool(ABC, Generic[ArgsT]):
    name: str
    description: str
    args_model: type[ArgsT]
    output_budget_chars: int | None = None

    @property
    def spec(self) -> ToolSpec:
        input_schema = self.args_model.model_json_schema()
        input_schema["additionalProperties"] = False
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=input_schema,
        )

    def invoke(self, arguments: dict[str, object]) -> ToolOutput:
        return self.run(self._validated_arguments(arguments))

    def _is_verification(self, arguments: ArgsT) -> bool:
        """Pure preflight hook; implementations must not execute or consume state."""

        del arguments
        return False

    def classifies_as_verification(self, arguments: dict[str, object]) -> bool:
        """Validate an inert call and fail closed at the classification boundary."""

        try:
            parsed = self._validated_arguments(arguments)
            classified = self._is_verification(parsed)
        except (CodedError, OSError, RuntimeError, ValueError):
            return False
        return classified is True

    def _validated_arguments(self, arguments: dict[str, object]) -> ArgsT:
        unexpected = sorted(set(arguments).difference(self.args_model.model_fields))
        if unexpected:
            joined = ", ".join(unexpected)
            raise ToolError("invalid_arguments", f"unexpected argument(s): {joined}")
        try:
            parsed = self.args_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolError("invalid_arguments", str(exc)) from exc
        return parsed

    @abstractmethod
    def run(self, arguments: ArgsT) -> ToolOutput:
        """Execute the tool locally and return bounded, observable output."""


class ToolRegistry:
    def __init__(
        self,
        tools: Iterable[BaseTool[Any]] = (),
        *,
        max_output_chars: int = 20_000,
    ) -> None:
        if max_output_chars < 1:
            raise ValueError("max_output_chars must be at least 1")
        self._tools: dict[str, BaseTool[Any]] = {}
        self._max_output_chars = max_output_chars
        for tool in tools:
            self.register(tool)

    def register(self, tool: BaseTool[Any]) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        if (
            tool.output_budget_chars is not None
            and tool.output_budget_chars > self._max_output_chars
        ):
            raise ValueError(
                f"tool {tool.name} output budget ({tool.output_budget_chars}) exceeds "
                f"registry budget ({self._max_output_chars})"
            )
        self._tools[tool.name] = tool

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(tool.spec for tool in self._tools.values())

    def is_verification_call(self, call: ToolCall) -> bool:
        """Classify without executing, approving, or consuming a tool call."""

        tool = self._tools.get(call.name)
        if tool is None:
            return False
        return tool.classifies_as_verification(call.arguments)

    def execute(self, call: ToolCall) -> ToolExecution:
        started = perf_counter()
        tool = self._tools.get(call.name)
        if tool is None:
            return self._failure(
                call,
                "unknown_tool",
                f"unknown tool: {call.name}",
                started,
            )

        try:
            result = tool.invoke(call.arguments)
            if not isinstance(result, ToolOutput):
                raise ToolError(
                    "invalid_output",
                    f"tool {call.name} returned {type(result).__name__}; expected ToolOutput",
                )
            summary = " ".join(result.summary.split())
            if not summary:
                raise ToolError(
                    "invalid_output",
                    f"tool {call.name} returned an empty summary",
                )
        except ValidationError as exc:
            return self._failure(
                call,
                "invalid_output",
                f"tool {call.name} produced invalid structured output: {exc}",
                started,
            )
        except CodedError as exc:
            return self._failure(
                call,
                exc.code,
                exc.message,
                started,
                metadata=exc.metadata,
            )
        except Exception as exc:  # noqa: BLE001 - tool failures become model observations.
            return self._failure(
                call,
                "tool_error",
                str(exc).strip() or type(exc).__name__,
                started,
            )

        output, hard_truncated = self._bounded(result.content)
        return ToolExecution(
            call_id=call.id,
            tool_name=call.name,
            ok=True,
            output=output,
            summary=summary,
            metadata=result.metadata,
            truncated=result.truncated or hard_truncated,
            control=result.control,
            duration_ms=self._elapsed_ms(started),
        )

    def _failure(
        self,
        call: ToolCall,
        code: str,
        message: str,
        started: float,
        *,
        metadata: Mapping[str, ErrorMetadataValue] | None = None,
    ) -> ToolExecution:
        bounded_message, _ = self._bounded(message)
        return ToolExecution(
            call_id=call.id,
            tool_name=call.name,
            ok=False,
            error_code=code,
            error_message=bounded_message,
            metadata=dict(metadata or {}),
            duration_ms=self._elapsed_ms(started),
        )

    def _bounded(self, text: str) -> tuple[str, bool]:
        if len(text) <= self._max_output_chars:
            return text, False
        suffix = "\n...[truncated]"
        if self._max_output_chars <= len(suffix):
            return text[: self._max_output_chars], True
        return text[: self._max_output_chars - len(suffix)] + suffix, True

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((perf_counter() - started) * 1000, 3)
