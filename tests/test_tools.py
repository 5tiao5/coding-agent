"""Regression tests for local tool validation and error normalization."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from coding_agent.errors import CodedError
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


class MissingWorkspacePathTool(BaseTool[TextArgs]):
    name = "missing_workspace_path"
    description = "Raise a stable error from another project-owned boundary."
    args_model = TextArgs

    def run(self, arguments: TextArgs) -> ToolOutput:
        del arguments
        raise CodedError(
            "not_found",
            "path not found: missing.py",
            metadata={"path": "missing.py", "recovery": "choose_another_path"},
        )


class RevisionConflictTool(BaseTool[TextArgs]):
    name = "revision_conflict"
    description = "Raise a recoverable write conflict with safe metadata."
    args_model = TextArgs

    def run(self, arguments: TextArgs) -> ToolOutput:
        del arguments
        raise ToolError(
            "revision_conflict",
            "file changed after it was read",
            metadata={
                "path": "src/example.py",
                "expected_sha256": "a" * 64,
                "current_sha256": "b" * 64,
                "actual_occurrences": 2,
                "recoverable": True,
                "detail": None,
            },
        )


class InvalidStructuredOutputTool(BaseTool[TextArgs]):
    name = "invalid_structured_output"
    description = "Raise validation while constructing a structured result."
    args_model = TextArgs

    def run(self, arguments: TextArgs) -> ToolOutput:
        del arguments
        return ToolOutput(content="hello", summary="")


class BudgetedTextTool(TextTool):
    name = "budgeted_text"
    output_budget_chars = 100


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


def test_ordinary_and_unknown_tools_never_claim_the_verification_reserve() -> None:
    registry = ToolRegistry([TextTool()])

    assert (
        registry.is_verification_call(
            ToolCall(id="ordinary", name="text", arguments={"text": "pytest passed"})
        )
        is False
    )
    assert (
        registry.is_verification_call(
            ToolCall(id="unknown", name="missing", arguments={"text": "pytest passed"})
        )
        is False
    )


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


def test_project_owned_coded_errors_keep_their_semantics_at_registry_boundary() -> None:
    execution = ToolRegistry([MissingWorkspacePathTool()]).execute(
        ToolCall(
            id="missing-workspace-path-1",
            name="missing_workspace_path",
            arguments={"text": "unused"},
        )
    )

    assert execution.ok is False
    assert execution.error_code == "not_found"
    assert execution.error_message == "path not found: missing.py"
    assert execution.metadata == {
        "path": "missing.py",
        "recovery": "choose_another_path",
    }


def test_tool_error_metadata_reaches_the_model_as_safe_structured_context() -> None:
    execution = ToolRegistry([RevisionConflictTool()]).execute(
        ToolCall(
            id="revision-conflict-1",
            name="revision_conflict",
            arguments={"text": "unused"},
        )
    )

    assert execution.ok is False
    assert execution.output is None
    assert execution.summary is None
    assert execution.truncated is False
    assert execution.error_code == "revision_conflict"
    assert execution.metadata == {
        "path": "src/example.py",
        "expected_sha256": "a" * 64,
        "current_sha256": "b" * 64,
        "actual_occurrences": 2,
        "recoverable": True,
        "detail": None,
    }
    assert '"metadata"' in execution.as_message_content()


def test_coded_error_rejects_non_scalar_or_non_finite_metadata() -> None:
    with pytest.raises(TypeError, match="non-empty strings"):
        CodedError(
            "unsafe_metadata",
            "bad metadata",
            metadata={"": "missing key"},
        )

    with pytest.raises(TypeError, match="safe scalar"):
        CodedError(
            "unsafe_metadata",
            "bad metadata",
            metadata={"nested": ["not", "safe"]},  # type: ignore[dict-item]
        )

    with pytest.raises(TypeError, match="finite"):
        CodedError("unsafe_metadata", "bad metadata", metadata={"ratio": float("nan")})


def test_output_model_validation_is_not_misreported_as_bad_arguments() -> None:
    execution = ToolRegistry([InvalidStructuredOutputTool()]).execute(
        ToolCall(
            id="invalid-structured-output-1",
            name="invalid_structured_output",
            arguments={"text": "unused"},
        )
    )

    assert execution.ok is False
    assert execution.error_code == "invalid_output"
    assert "structured output" in str(execution.error_message)


def test_registry_rejects_a_tool_whose_semantic_budget_exceeds_its_hard_cap() -> None:
    with pytest.raises(ValueError, match="output budget .* exceeds registry budget"):
        ToolRegistry([BudgetedTextTool()], max_output_chars=99)
