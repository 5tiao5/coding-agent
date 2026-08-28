"""Model-facing adapter for bounded local command execution."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints, field_validator

from coding_agent.approval import CommandApprovalRequest, CommandApprover
from coding_agent.command import (
    CommandClass,
    CommandClassification,
    CommandEnvironmentProfile,
    CommandError,
    CommandPolicy,
    CommandRequest,
    CommandResult,
    CommandRunner,
    CommandStatus,
    LocalCommandRunner,
    decode_command_output,
)
from coding_agent.models import ToolControlFacts, ToolOutput, VerificationSignal
from coding_agent.tools._rendering import (
    clip_at_escape_boundary,
    clip_with_ellipsis,
    is_display_control,
    render_visible_text,
    summarize_path,
)
from coding_agent.tools.base import BaseTool
from coding_agent.workspace import Workspace

CommandArgument = Annotated[str, StringConstraints(min_length=1, max_length=4_000)]

_MAX_ARGUMENTS = 64
_MAX_TOTAL_ARGUMENT_CHARS = 16_000
_OUTPUT_TRUNCATION_MARKER = "\n...[command output truncated]...\n"


class RunCommandArguments(BaseModel):
    argv: list[CommandArgument] = Field(
        min_length=1,
        max_length=_MAX_ARGUMENTS,
        description=(
            "Executable and arguments as a list. The command runs directly without a shell; "
            "request separate tool calls instead of pipes or shell operators."
        ),
    )
    cwd: str = Field(
        default=".",
        min_length=1,
        max_length=1_000,
        description="Existing workspace-relative working directory.",
    )
    timeout_seconds: float = Field(
        default=120.0,
        ge=1.0,
        le=300.0,
        description="Hard wall-clock timeout in seconds.",
    )

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, value: list[str]) -> list[str]:
        if value[0].isspace():
            raise ValueError("command executable cannot contain only whitespace")
        if sum(len(argument) for argument in value) > _MAX_TOTAL_ARGUMENT_CHARS:
            raise ValueError(
                f"command arguments cannot exceed {_MAX_TOTAL_ARGUMENT_CHARS} characters"
            )
        if any(is_display_control(character) for argument in value for character in argument):
            raise ValueError("command arguments must contain printable single-line text")
        return value


class RunCommandTool(BaseTool[RunCommandArguments]):
    name = "run_command"
    description = (
        "Run one local executable with explicit arguments inside a workspace-relative cwd. "
        "Uses no model-selected shell, enforces a timeout, terminates child processes, strips "
        "credential-like environment variables, and returns bounded combined stdout/stderr. "
        "A non-zero exit or timeout is an observed command result, not a tool transport error."
    )
    args_model = RunCommandArguments

    def __init__(
        self,
        workspace: Workspace,
        *,
        runner: CommandRunner | None = None,
        policy: CommandPolicy | None = None,
        approver: CommandApprover | None = None,
        max_output_chars: int = 16_000,
    ) -> None:
        minimum = 512
        if max_output_chars < minimum:
            raise ValueError("max_output_chars is too small for command metadata and output")
        self._workspace = workspace
        self._runner = runner or LocalCommandRunner()
        self._policy = policy or CommandPolicy()
        self._approver = approver
        self._max_output_chars = max_output_chars
        self.output_budget_chars = max_output_chars

    def run(self, arguments: RunCommandArguments) -> ToolOutput:
        argv = tuple(arguments.argv)
        cwd = self._workspace.resolve(arguments.cwd, expected="directory")
        try:
            classification = self._policy.classify(
                argv,
                cwd=cwd.relative,
                workspace_root=self._workspace.root,
            )
        except CommandError as exc:
            if exc.code != "command_approval_required" or self._approver is None:
                raise
            try:
                approved = self._approver.approve(
                    CommandApprovalRequest(argv=argv, cwd=cwd.relative)
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as approval_error:
                raise CommandError(
                    "command_approval_failed",
                    "command approval could not be obtained safely",
                ) from approval_error
            if not approved:
                raise CommandError(
                    "command_denied",
                    "command was denied by the user",
                    metadata={"reason": "user_denied"},
                ) from None
            classification = self._policy.classify(
                argv,
                cwd=cwd.relative,
                workspace_root=self._workspace.root,
                approved=True,
            )
        result = self._runner.run(
            CommandRequest(
                argv=argv,
                cwd=cwd.path,
                timeout_seconds=arguments.timeout_seconds,
                environment_profile=(
                    CommandEnvironmentProfile.VERIFIER
                    if classification.command_class is CommandClass.VERIFIER
                    else CommandEnvironmentProfile.SANITIZED
                ),
            )
        )
        return self._render_result(
            result,
            classification=classification,
            cwd=cwd.relative,
            timeout_seconds=arguments.timeout_seconds,
        )

    def _render_result(
        self,
        result: CommandResult,
        *,
        classification: CommandClassification,
        cwd: str,
        timeout_seconds: float,
    ) -> ToolOutput:
        decoded, encoding = decode_command_output(result.output)
        visible_output = _render_command_output(decoded)
        display_cwd = summarize_path(cwd)
        status_lines = _status_lines(result, cwd=display_cwd, timeout_seconds=timeout_seconds)
        prefix = "\n".join(status_lines) + "\nOutput:\n"
        display_output = visible_output or "(no output)"
        if len(prefix) >= self._max_output_chars:
            content = clip_at_escape_boundary(prefix, self._max_output_chars)
            display_truncated = True
        else:
            clipped_output, display_truncated = _clip_visible_head_tail(
                display_output,
                self._max_output_chars - len(prefix),
            )
            content = prefix + clipped_output
        control = _control_facts(result, classification)
        summary = _summary(result, cwd=display_cwd, timeout_seconds=timeout_seconds)
        return ToolOutput(
            content=content,
            summary=summary,
            metadata={
                "status": result.status.value,
                "exit_code": result.exit_code,
                "timed_out": result.status is CommandStatus.TIMED_OUT,
                "cwd": cwd,
                "command_class": classification.command_class.value,
                "output_encoding": encoding,
                "total_output_bytes": result.total_output_bytes,
                "captured_output_bytes": result.captured_output_bytes,
                "termination_failed": result.status is CommandStatus.CONTROL_FAILED,
            },
            truncated=result.output_truncated or display_truncated,
            control=control,
        )


def _status_lines(
    result: CommandResult,
    *,
    cwd: str,
    timeout_seconds: float,
) -> tuple[str, ...]:
    if result.status is CommandStatus.EXITED:
        return (
            "Status: exited",
            f"Exit code: {result.exit_code}",
            f"Working directory: {render_visible_text(cwd)}",
        )
    if result.status is CommandStatus.TIMED_OUT:
        return (
            f"Status: timed out after {timeout_seconds:g} seconds",
            "Exit code: unavailable",
            f"Working directory: {render_visible_text(cwd)}",
        )
    return (
        "Status: process control failed",
        "Exit code: unavailable",
        f"Working directory: {render_visible_text(cwd)}",
        "Safety stop: "
        + clip_with_ellipsis(
            render_visible_text(result.terminal_reason or "unknown control failure"), 240
        ),
    )


def _summary(result: CommandResult, *, cwd: str, timeout_seconds: float) -> str:
    safe_cwd = render_visible_text(cwd)
    if result.status is CommandStatus.EXITED:
        return f"Command exited {result.exit_code} in {safe_cwd}"
    if result.status is CommandStatus.TIMED_OUT:
        return f"Command timed out after {timeout_seconds:g}s in {safe_cwd}"
    return f"Command process control failed in {safe_cwd}; run must stop"


def _control_facts(
    result: CommandResult,
    classification: CommandClassification,
) -> ToolControlFacts:
    terminal = result.status is CommandStatus.CONTROL_FAILED
    verification: VerificationSignal | None = None
    if classification.command_class is CommandClass.VERIFIER:
        verification = (
            VerificationSignal.PASSED
            if result.status is CommandStatus.EXITED and result.exit_code == 0
            else VerificationSignal.FAILED
        )
    invalidates = terminal or classification.command_class is CommandClass.GENERAL
    return ToolControlFacts(
        invalidates_verification=invalidates,
        verification=verification,
        verification_kind=classification.verification_kind,
        verification_label=classification.verification_label,
        terminal_stop=terminal,
        terminal_reason=result.terminal_reason if terminal else None,
    )


def _render_command_output(text: str) -> str:
    normalized = text.replace("\r\n", "\n")
    return "\n".join(render_visible_text(line) for line in normalized.split("\n"))


def _clip_visible_head_tail(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", bool(text)
    if len(text) <= max_chars:
        return text, False
    if max_chars <= len(_OUTPUT_TRUNCATION_MARKER):
        return text[:max_chars], True

    tokens = _visible_tokens(text)
    available = max_chars - len(_OUTPUT_TRUNCATION_MARKER)
    head_budget = available // 2
    tail_budget = available - head_budget
    head = _take_tokens(tokens, head_budget)
    tail = _take_tokens(tuple(reversed(tokens)), tail_budget, reverse_result=True)
    return head + _OUTPUT_TRUNCATION_MARKER + tail, True


def _visible_tokens(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    index = 0
    escape_lengths = {"x": 4, "u": 6, "U": 10}
    while index < len(text):
        token_length = 1
        if text[index] == "\\" and index + 1 < len(text):
            candidate_length = escape_lengths.get(text[index + 1])
            if candidate_length is not None:
                candidate = text[index + 2 : index + candidate_length]
                if len(candidate) == candidate_length - 2 and all(
                    character in "0123456789abcdef" for character in candidate
                ):
                    token_length = candidate_length
        tokens.append(text[index : index + token_length])
        index += token_length
    return tuple(tokens)


def _take_tokens(
    tokens: tuple[str, ...],
    budget: int,
    *,
    reverse_result: bool = False,
) -> str:
    selected: list[str] = []
    used = 0
    for token in tokens:
        if used + len(token) > budget:
            break
        selected.append(token)
        used += len(token)
    if reverse_result:
        selected.reverse()
    return "".join(selected)
