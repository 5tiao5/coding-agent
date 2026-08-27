"""Regression tests for local tool validation and error normalization."""

from __future__ import annotations

from pydantic import BaseModel

from coding_agent.models import ToolCall
from coding_agent.tools import BaseTool, ToolError, ToolOutput, ToolRegistry


class TextArgs(BaseModel):
    text: str


class TextTool(BaseTool[TextArgs]):
    name = "text"
    description = "Return text."
    args_model = TextArgs

    def run(self, arguments: TextArgs) -> ToolOutput:
        return ToolOutput(content=arguments.text, summary="Returned test text")


class NonStringTool(BaseTool[TextArgs]):
    name = "non_string"
    description = "Return an invalid non-string value."
    args_model = TextArgs

    def run(self, arguments: TextArgs) -> ToolOutput:
        del arguments
        return 42  # type: ignore[return-value]


class EmptyErrorTool(BaseTool[TextArgs]):
    name = "empty_error"
    description = "Raise an exception without a message."
    args_model = TextArgs

    def run(self, arguments: TextArgs) -> ToolOutput:
        del arguments
        raise RuntimeError


class EmptyCodeTool(BaseTool[TextArgs]):
    name = "empty_code"
    description = "Raise a declared tool error with an invalid empty code."
    args_model = TextArgs

    def run(self, arguments: TextArgs) -> ToolOutput:
        del arguments
        raise ToolError("", "oops")


class BlankSummaryTool(BaseTool[TextArgs]):
    name = "blank_summary"
    description = "Return an invalid blank observable summary."
    args_model = TextArgs

    def run(self, arguments: TextArgs) -> ToolOutput:
        return ToolOutput(content=arguments.text, summary="   ")


def test_tool_schema_and_runtime_both_reject_extra_arguments() -> None:
    registry = ToolRegistry([TextTool()])

    assert registry.specs()[0].input_schema["additionalProperties"] is False

    execution = registry.execute(
        ToolCall(
            id="text-extra",
            name="text",
            arguments={"text": "hello", "unexpected": "value"},
        )
    )

    assert execution.ok is False
    assert execution.output is None
    assert execution.error_code == "invalid_arguments"
    assert execution.error_message is not None
    assert "unexpected" in execution.error_message


def test_non_string_tool_output_becomes_an_invalid_output_result() -> None:
    execution = ToolRegistry([NonStringTool()]).execute(
        ToolCall(id="non-string-1", name="non_string", arguments={"text": "hello"})
    )

    assert execution.ok is False
    assert execution.output is None
    assert execution.error_code == "invalid_output"
    assert execution.error_message == "tool non_string returned int; expected ToolOutput"


def test_exception_without_a_message_still_has_a_structured_error_message() -> None:
    execution = ToolRegistry([EmptyErrorTool()]).execute(
        ToolCall(id="empty-error-1", name="empty_error", arguments={"text": "hello"})
    )

    assert execution.ok is False
    assert execution.output is None
    assert execution.error_code == "tool_error"
    assert execution.error_message == "RuntimeError"


def test_tool_output_is_truncated_to_the_configured_character_budget() -> None:
    registry = ToolRegistry([TextTool()], max_output_chars=20)

    execution = registry.execute(
        ToolCall(id="long-output-1", name="text", arguments={"text": "x" * 100})
    )

    assert execution.ok is True
    assert execution.output is not None
    assert len(execution.output) == 20
    assert execution.output.endswith("\n...[truncated]")
    assert execution.output == "x" * 5 + "\n...[truncated]"


def test_tiny_output_budget_still_enforces_the_hard_limit() -> None:
    execution = ToolRegistry([TextTool()], max_output_chars=5).execute(
        ToolCall(id="tiny-budget-1", name="text", arguments={"text": "abcdefghij"})
    )

    assert execution.ok is True
    assert execution.output == "abcde"
    assert execution.truncated is True


def test_empty_declared_tool_error_code_is_normalized_to_a_structured_failure() -> None:
    execution = ToolRegistry([EmptyCodeTool()]).execute(
        ToolCall(id="empty-code-1", name="empty_code", arguments={"text": "hello"})
    )

    assert execution.ok is False
    assert execution.output is None
    assert execution.error_code == "tool_error"
    assert execution.error_message == "oops"


def test_blank_tool_summary_becomes_an_invalid_output_result() -> None:
    execution = ToolRegistry([BlankSummaryTool()]).execute(
        ToolCall(id="blank-summary-1", name="blank_summary", arguments={"text": "hello"})
    )

    assert execution.ok is False
    assert execution.error_code == "invalid_output"
    assert execution.error_message == "tool blank_summary returned an empty summary"
