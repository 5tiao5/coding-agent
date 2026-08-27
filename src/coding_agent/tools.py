"""Typed local tool definition and dispatch owned by this project."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from coding_agent.models import ToolCall, ToolExecution, ToolSpec

ArgsT = TypeVar("ArgsT", bound=BaseModel)


class ToolError(Exception):
    """Expected tool failure that can be reported to the model safely."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code.strip() or "tool_error"
        self.message = message.strip() or "tool failed"


class BaseTool(ABC, Generic[ArgsT]):
    name: str
    description: str
    args_model: type[ArgsT]

    @property
    def spec(self) -> ToolSpec:
        input_schema = self.args_model.model_json_schema()
        input_schema["additionalProperties"] = False
        return ToolSpec(
            name=self.name,
            description=self.description,
            input_schema=input_schema,
        )

    def invoke(self, arguments: dict[str, object]) -> str:
        unexpected = sorted(set(arguments).difference(self.args_model.model_fields))
        if unexpected:
            joined = ", ".join(unexpected)
            raise ToolError("invalid_arguments", f"unexpected argument(s): {joined}")
        parsed = self.args_model.model_validate(arguments)
        return self.run(parsed)

    @abstractmethod
    def run(self, arguments: ArgsT) -> str:
        """Execute the tool locally and return bounded textual output."""


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
        self._tools[tool.name] = tool

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(tool.spec for tool in self._tools.values())

    def execute(self, call: ToolCall) -> ToolExecution:
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolExecution(
                call_id=call.id,
                tool_name=call.name,
                ok=False,
                error_code="unknown_tool",
                error_message=self._bounded(f"unknown tool: {call.name}"),
            )

        try:
            output = tool.invoke(call.arguments)
            if not isinstance(output, str):
                raise ToolError(
                    "invalid_output",
                    f"tool {call.name} returned {type(output).__name__}; expected str",
                )
        except ValidationError as exc:
            return ToolExecution(
                call_id=call.id,
                tool_name=call.name,
                ok=False,
                error_code="invalid_arguments",
                error_message=self._bounded(str(exc)),
            )
        except ToolError as exc:
            return ToolExecution(
                call_id=call.id,
                tool_name=call.name,
                ok=False,
                error_code=exc.code,
                error_message=self._bounded(exc.message or type(exc).__name__),
            )
        except Exception as exc:  # noqa: BLE001 - tool failures become model observations.
            return ToolExecution(
                call_id=call.id,
                tool_name=call.name,
                ok=False,
                error_code="tool_error",
                error_message=self._bounded(str(exc).strip() or type(exc).__name__),
            )

        return ToolExecution(
            call_id=call.id,
            tool_name=call.name,
            ok=True,
            output=self._bounded(output),
        )

    def _bounded(self, text: str) -> str:
        if len(text) <= self._max_output_chars:
            return text
        suffix = "\n...[truncated]"
        if self._max_output_chars <= len(suffix):
            return text[: self._max_output_chars]
        return text[: self._max_output_chars - len(suffix)] + suffix
