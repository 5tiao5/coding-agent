"""Pure protocol views and transcript helpers used by the agent runner."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite

from coding_agent._presentation_safety import sanitize_public_label
from coding_agent.budget import BudgetPolicy, BudgetPurpose
from coding_agent.completion import (
    CompletionContract,
    CompletionStatus,
    VerificationProfile,
    evaluate_completion,
)
from coding_agent.models import (
    AgentState,
    ChatMessage,
    MessageRole,
    ModelResponse,
    ToolCall,
    ToolExecution,
    VerificationKind,
)
from coding_agent.run_memory import RunMemorySnapshot
from coding_agent.tooling import ToolDispatcher
from coding_agent.verification import VerificationLedger

DEFAULT_SYSTEM_PROMPT = """You are a coding agent. Use the available local tools when needed.
Keep plans and action summaries explicit, but never provide hidden reasoning or chain of thought.
After changing files, run a recognized test, build, or check before giving a concise final answer.
Runtime evidence, not your textual claim, determines whether the result is verified."""

RUNTIME_LIMITS_MARKER = "[CODING_AGENT_RUNTIME_LIMITS]"
RUNTIME_LIMITS_END_MARKER = "[/CODING_AGENT_RUNTIME_LIMITS]"
MAX_CONSECUTIVE_TOOL_BATCH_REJECTIONS = 3
TOOL_BATCH_REJECTED_ERROR_CODE = "tool_batch_rejected"
PROTOCOL_CORRECTION_MARKER = "[CODING_AGENT_PROTOCOL_CORRECTION]"
PROTOCOL_CORRECTION_INSTRUCTION = (
    f"\n\n{PROTOCOL_CORRECTION_MARKER}\n"
    "The previous response was discarded. Return a fresh response and ensure the "
    "arguments field of every function call is a valid JSON object, not an array or scalar."
)
CLOSEOUT_MARKER = "[CODING_AGENT_CLOSEOUT]"
EARLY_FINAL_CORRECTION_MARKER = "[CODING_AGENT_EARLY_FINAL_CORRECTION]"
EARLY_FINAL_CORRECTION = (
    f"{EARLY_FINAL_CORRECTION_MARKER}\n"
    "Host correction: workspace files changed, but the completion contract still lacks "
    "current verification. On the next turn, request only the smallest registered verifier "
    "batch needed for the current workspace. Do not inspect or modify files, and do not "
    "return another final answer before requesting those checks."
)

PRESENTATION_PREVIEW_TOOLS = frozenset({"replace_text", "undo_change", "update_plan", "write_file"})

_MAX_PUBLIC_COMMAND_ARGUMENTS = 64
_MAX_PUBLIC_COMMAND_ARGUMENT_CHARS = 16_000
_MAX_PUBLIC_VERIFICATION_SCOPES = 16
_EXECUTABLE_SUFFIXES = (".exe", ".com", ".cmd", ".bat")
_SENSITIVE_ARGUMENT_MARKERS = (
    "ACCESS_KEY",
    "API_KEY",
    "AUTHORIZATION",
    "CREDENTIAL",
    "PASSWORD",
    "PASSWD",
    "PRIVATE_KEY",
    "SECRET",
    "TOKEN",
)
_FINAL_VERIFICATION_LABEL_FIELDS = (
    "evidence_labels",
    "required_labels",
    "missing_labels",
    "unexpected_labels",
    "mismatched_labels",
)
_ALLOWED_TRANSITIONS: dict[AgentState, frozenset[AgentState]] = {
    AgentState.CREATED: frozenset({AgentState.PLANNING}),
    AgentState.PLANNING: frozenset({AgentState.ACTING}),
    AgentState.ACTING: frozenset({AgentState.OBSERVING, AgentState.VERIFYING, AgentState.FAILED}),
    AgentState.OBSERVING: frozenset({AgentState.ACTING, AgentState.FAILED}),
    AgentState.VERIFYING: frozenset(
        {AgentState.COMPLETED, AgentState.COMPLETED_UNVERIFIED, AgentState.FAILED}
    ),
}


@dataclass(frozen=True, slots=True)
class FinalVerificationView:
    """Presentation-safe terminal view of trusted completion evidence."""

    verified: bool
    event_data: dict[str, object]
    evaluation_message: str
    finished_message: str


def is_allowed_transition(previous: AgentState, current: AgentState) -> bool:
    """Validate one public agent lifecycle edge."""

    return current in _ALLOWED_TRANSITIONS.get(previous, frozenset())


def validate_model_retry_settings(max_retries: int, base_delay_seconds: float) -> float:
    """Validate bounded provider retry settings and normalize the delay."""

    if (
        isinstance(max_retries, bool)
        or not isinstance(max_retries, int)
        or not 0 <= max_retries <= 10
    ):
        raise ValueError("max_model_retries must be an integer between 0 and 10")
    if (
        isinstance(base_delay_seconds, bool)
        or not isinstance(base_delay_seconds, int | float)
        or not isfinite(base_delay_seconds)
        or not 0 <= base_delay_seconds <= 60
    ):
        raise ValueError("model_retry_base_delay_seconds must be between 0 and 60")
    return float(base_delay_seconds)


def resolve_budget_policy(
    *,
    budget_policy: BudgetPolicy | None,
    max_steps: int,
    max_tool_calls_per_step: int,
    max_total_tool_calls: int | None,
    verification_profile: VerificationProfile | None,
) -> BudgetPolicy:
    """Resolve legacy knobs and reject completion budgets that cannot close safely."""

    if budget_policy is not None and (
        max_steps != 20 or max_tool_calls_per_step != 8 or max_total_tool_calls is not None
    ):
        raise ValueError("budget_policy cannot be combined with legacy budget arguments")
    required_calls = (
        len(verification_profile.required_labels) if verification_profile is not None else 1
    )
    batch_width = (
        budget_policy.max_calls_per_turn if budget_policy is not None else max_tool_calls_per_step
    )
    required_turns = (required_calls + batch_width - 1) // batch_width
    selected = budget_policy or BudgetPolicy(
        max_model_turns=max_steps,
        max_calls_per_turn=max_tool_calls_per_step,
        average_calls_per_turn=min(2, max_tool_calls_per_step),
        verification_turn_reserve=max(2, required_turns + 1),
        verification_call_reserve=max(1, required_calls),
    )
    if max_total_tool_calls is not None and max_total_tool_calls != selected.max_total_tool_calls:
        raise ValueError(
            "max_total_tool_calls is derived from max_steps; use BudgetPolicy "
            "to configure coupled budgets"
        )
    if verification_profile is None:
        return selected
    if selected.verification_call_reserve < required_calls:
        raise ValueError("budget_policy must reserve at least one call for every required verifier")
    if selected.verification_turn_reserve < required_turns + 1:
        raise ValueError("budget_policy must reserve verifier batches and one final-response turn")
    return selected


def append_cancelled_tool_results(
    messages: list[ChatMessage],
    calls: Sequence[ToolCall],
    *,
    error_code: str,
    error_message: str,
) -> None:
    """Close every unexecuted call with a stable failure result."""

    for call in calls:
        execution = ToolExecution(
            call_id=call.id,
            tool_name=call.name,
            ok=False,
            error_code=error_code,
            error_message=error_message,
        )
        messages.append(
            ChatMessage(
                role=MessageRole.TOOL,
                content=execution.as_message_content(),
                tool_call_id=call.id,
                tool_name=call.name,
            )
        )


def runtime_limits_event_data(policy: BudgetPolicy) -> dict[str, int]:
    """Render runner limits for observable lifecycle events."""

    return {
        "max_model_turns": policy.max_model_turns,
        "max_calls_per_turn": policy.max_calls_per_turn,
        "max_total_tool_calls": policy.max_total_tool_calls,
    }


def with_runtime_limits(
    system_prompt: str,
    *,
    max_model_turns: int,
    max_calls_per_turn: int,
    max_total_tool_calls: int,
) -> str:
    """Replace the host-owned runtime budget block with current runner limits."""

    base = system_prompt
    marker_index = base.find(RUNTIME_LIMITS_MARKER)
    if marker_index >= 0:
        end_index = base.find(RUNTIME_LIMITS_END_MARKER, marker_index)
        if end_index >= 0:
            base = base[:marker_index] + base[end_index + len(RUNTIME_LIMITS_END_MARKER) :]
    base = base.rstrip()
    return (
        f"{base}\n\n{RUNTIME_LIMITS_MARKER}\n"
        "Runtime budgets enforced for this run:\n"
        f"- Maximum model turns: {max_model_turns}.\n"
        f"- Maximum tool calls in one model response: {max_calls_per_turn}. "
        "Never exceed this per-turn limit; split independent calls across turns.\n"
        f"- Maximum accepted tool calls across the run: {max_total_tool_calls}.\n"
        "The host reserves enough closing capacity for registered verification and one "
        "honest final response when a completion contract is active.\n"
        "An over-limit per-turn batch is rejected atomically: none of its calls execute. "
        "Use the returned tool errors to retry with a smaller batch.\n"
        f"{RUNTIME_LIMITS_END_MARKER}"
    )


def with_protocol_correction(messages: Sequence[ChatMessage]) -> tuple[ChatMessage, ...]:
    """Add one bounded, sanitized retry instruction to a transient request view only."""

    system = messages[0]
    assert system.role is MessageRole.SYSTEM
    assert system.content is not None
    content = f"{system.content.rstrip()}{PROTOCOL_CORRECTION_INSTRUCTION}"
    return (system.model_copy(update={"content": content}), *messages[1:])


def protocol_correction_retry_event_data(
    *,
    attempt: int,
    max_attempts: int,
    prepared_context_chars: int,
) -> dict[str, object]:
    """Build the stable retry event payload for one corrected request."""

    return {
        "attempt": attempt,
        "next_attempt": attempt + 1,
        "max_attempts": max_attempts,
        "delay_seconds": 0.0,
        "error_code": "model_response_invalid",
        "retry_kind": "protocol_correction",
        "instruction_chars": len(PROTOCOL_CORRECTION_INSTRUCTION),
        "prepared_context_chars": prepared_context_chars,
    }


def with_closeout_instruction(
    messages: Sequence[ChatMessage],
    *,
    purpose: BudgetPurpose,
    remaining_turns: int,
) -> tuple[ChatMessage, ...]:
    """Inject a host-owned closeout instruction into the request view only."""

    system = messages[0]
    assert system.role is MessageRole.SYSTEM
    assert system.content is not None
    if purpose == "verification":
        instruction = (
            "The work-turn budget is exhausted. This turn is reserved for recognized "
            "verification calls only; do not inspect or modify files. Request the smallest "
            "registered verification batch needed to validate the current workspace."
        )
    else:
        instruction = (
            "This is the reserved final-response turn. Do not call tools. Give a concise, "
            "honest summary and explicitly state any missing or failed verification."
        )
    content = (
        f"{system.content.rstrip()}\n\n{CLOSEOUT_MARKER}\n"
        f"{instruction}\nRemaining model turns after this response: {remaining_turns}."
    )
    return (system.model_copy(update={"content": content}), *messages[1:])


def append_early_final_correction(
    messages: list[ChatMessage],
    response: ModelResponse,
) -> None:
    """Close a premature textual answer with one provider-valid host user correction."""

    assert not response.tool_calls
    assert response.content is not None
    messages.extend(
        (
            ChatMessage(role=MessageRole.ASSISTANT, content=response.content),
            ChatMessage(role=MessageRole.USER, content=EARLY_FINAL_CORRECTION),
        )
    )


def early_final_correction_event_data(*, remaining_model_turns: int) -> dict[str, object]:
    """Describe the one host-scheduled verification closeout without model text."""

    return {
        "retry_kind": "verification_closeout",
        "remaining_model_turns": remaining_model_turns,
        "instruction_chars": len(EARLY_FINAL_CORRECTION),
    }


def is_early_final_correction(message: ChatMessage) -> bool:
    """Recognize only the exact bounded host correction used by this protocol."""

    return message.role is MessageRole.USER and message.content == EARLY_FINAL_CORRECTION


def has_early_final_correction(messages: Sequence[ChatMessage]) -> bool:
    """Recover the once-only correction flag from a canonical transcript."""

    return any(is_early_final_correction(message) for message in messages[2:])


def should_append_early_final_correction(
    response: ModelResponse,
    *,
    memory: RunMemorySnapshot,
    ledger: VerificationLedger,
    profile: VerificationProfile | None,
    contract: CompletionContract | None,
    already_corrected: bool,
    remaining_model_turns: int,
) -> bool:
    """Require one normal verifier turn after a changed workspace closes too early."""

    if (
        already_corrected
        or response.tool_calls
        or profile is None
        or contract is None
        or remaining_model_turns < 2
        or not memory.file_changes
    ):
        return False
    report = evaluate_completion(profile, contract, ledger.report())
    if report.completion_status not in {CompletionStatus.MISSING, CompletionStatus.STALE}:
        return False
    if contract.require_target_runtime and not profile.target_runtime.eligible_for_task_validation:
        return False
    available_scopes = {scope for check in profile.checks for scope in check.scopes}
    return all(scope in available_scopes for scope in contract.required_scopes)


def final_verification_view(
    ledger: VerificationLedger,
    profile: VerificationProfile | None,
    contract: CompletionContract | None,
) -> FinalVerificationView:
    """Evaluate trusted evidence and choose stable lifecycle copy in one pure helper."""

    verification = ledger.report()
    if profile is None:
        verified = verification.verified
        data = _public_final_verification_data(verification.event_data())
        evaluation_message = (
            "Current verification evidence passed"
            if verified
            else f"Verification evidence is {verification.status.value}"
        )
        finished_message = (
            "Run ended with current verification evidence"
            if verified
            else "Run ended with an unverified final response"
        )
        return FinalVerificationView(verified, data, evaluation_message, finished_message)

    assert contract is not None
    completion = evaluate_completion(profile, contract, verification)
    verified = completion.task_validated
    data = _public_final_verification_data(completion.event_data(verification))
    evaluation_message = (
        "Task completion contract validated"
        if verified
        else f"Completion contract is {completion.completion_status.value}"
    )
    finished_message = (
        "Run ended with a validated completion contract"
        if verified
        else "Run ended with an unverified final response"
    )
    return FinalVerificationView(verified, data, evaluation_message, finished_message)


def requires_current_verification(
    ledger: VerificationLedger,
    profile: VerificationProfile | None,
    contract: CompletionContract | None,
) -> bool:
    """Return whether closeout still needs current contract evidence."""

    if profile is None:
        return False
    assert contract is not None
    return not evaluate_completion(profile, contract, ledger.report()).task_validated


def closeout_turn_purpose(
    remaining_turns: int,
    *,
    verification_required: bool,
) -> BudgetPurpose:
    """Choose the only admissible kind of closeout turn."""

    if verification_required and remaining_turns > 1:
        return "verification"
    return "final"


def is_verification_call(tools: ToolDispatcher, call: ToolCall) -> bool:
    """Classify verifier calls fail-closed across dispatcher implementations."""

    classifier = getattr(tools, "is_verification_call", None)
    if classifier is None or not callable(classifier):
        return False
    try:
        return classifier(call) is True
    except (TypeError, ValueError):
        return False


def over_limit_tool_batch_error_message(
    *,
    requested_calls: int,
    max_calls_per_turn: int,
) -> str:
    """Describe an atomically rejected over-width tool batch."""

    return (
        "tool batch exceeded the per-step call limit "
        f"({requested_calls} requested, {max_calls_per_turn} allowed); "
        "retry with a smaller batch"
    )


def inadmissible_tool_batch_error_message(rejection_code: str) -> str:
    """Describe a batch rejected by closeout or aggregate-budget policy."""

    return (
        f"tool batch was not admitted ({rejection_code}); "
        "return a final response or request an allowed smaller batch"
    )


def trailing_tool_batch_rejections(messages: Sequence[ChatMessage]) -> int:
    """Recover the consecutive rejection streak from a stable transcript."""

    consecutive = 0
    cursor = 2
    while cursor < len(messages):
        assistant = messages[cursor]
        cursor += 1
        if assistant.role is not MessageRole.ASSISTANT or not assistant.tool_calls:
            consecutive = 0
            continue
        results = messages[cursor : cursor + len(assistant.tool_calls)]
        cursor += len(assistant.tool_calls)
        if len(results) == len(assistant.tool_calls) and all(
            _tool_result_error_code(result) == TOOL_BATCH_REJECTED_ERROR_CODE for result in results
        ):
            consecutive += 1
        else:
            consecutive = 0
    return consecutive


def tool_finished_event_data(
    call: ToolCall,
    raw_execution: ToolExecution,
    execution: ToolExecution,
    *,
    verification_call: bool = False,
) -> dict[str, object]:
    """Build the public tool-finished event view without exposing private output."""

    event_data: dict[str, object] = {
        "call_id": call.id,
        "tool_name": call.name,
        "ok": raw_execution.ok,
        "error_code": raw_execution.error_code,
        "duration_ms": raw_execution.duration_ms,
        "output_chars": len(raw_execution.output or ""),
        "truncated": raw_execution.truncated,
        "summary": raw_execution.summary,
        "observation_chars": len(execution.as_message_content()),
        "observation_truncated": execution.as_message_content()
        != raw_execution.as_message_content(),
    }
    if raw_execution.metadata:
        metadata = (
            _public_command_metadata(raw_execution.metadata)
            if call.name == "run_command"
            else dict(raw_execution.metadata)
        )
        if metadata:
            event_data["metadata"] = metadata
    public_invocation = _public_invocation(
        call,
        raw_execution,
        verification_call=verification_call,
    )
    if public_invocation is not None:
        event_data["public_invocation"] = public_invocation
    if (
        raw_execution.ok
        and raw_execution.output is not None
        and call.name in PRESENTATION_PREVIEW_TOOLS
    ):
        # Only explicit plans and bounded mutation diffs are presentation-safe.
        # Read/search/command output stays in the private canonical transcript.
        event_data["preview"] = raw_execution.output
    return event_data


def verification_scope_event_data(
    profile: VerificationProfile | None,
    *,
    label: str,
    kind: object,
) -> dict[str, object]:
    """Return bounded scopes owned by the host profile for one exact verifier."""

    scopes: tuple[str, ...] = ()
    if profile is not None:
        check = next(
            (
                candidate
                for candidate in profile.checks
                if candidate.label == label and candidate.kind is kind
            ),
            None,
        )
        if check is not None:
            scopes = check.scopes
    visible = scopes[:_MAX_PUBLIC_VERIFICATION_SCOPES]
    return {
        "scopes": list(visible),
        "scopes_truncated": len(visible) < len(scopes),
    }


def _public_invocation(
    call: ToolCall,
    execution: ToolExecution,
    *,
    verification_call: bool,
) -> dict[str, object] | None:
    """Project a command call without serializing its model-authored argument vector."""

    if call.name != "run_command":
        return None
    raw_argv = call.arguments.get("argv")
    if not isinstance(raw_argv, list) or not 1 <= len(raw_argv) <= _MAX_PUBLIC_COMMAND_ARGUMENTS:
        return None
    if any(not isinstance(token, str) for token in raw_argv):
        return None
    argv = tuple(raw_argv)
    if sum(len(token) for token in argv) > _MAX_PUBLIC_COMMAND_ARGUMENT_CHARS or any(
        _contains_display_control(token) for token in argv
    ):
        return None
    executable = _public_executable_name(argv[0])
    if executable is None:
        return None
    public: dict[str, object] = {
        "executable": executable,
        "argument_count": len(argv) - 1,
    }
    if verification_call is True:
        public.update(_public_verification_identity(execution))
    return public


def _public_verification_identity(execution: ToolExecution) -> dict[str, str]:
    """Expose only the paired verifier identity asserted by host-owned control facts."""

    label = execution.control.verification_label
    kind = execution.control.verification_kind
    if label is None or not isinstance(kind, VerificationKind):
        return {}
    normalized_label = public_verifier_label(label)
    if normalized_label is None:
        return {}
    return {
        "verification_label": normalized_label,
        "verification_kind": kind.value,
    }


def public_verifier_label(value: object) -> str | None:
    """Return one bounded verifier label only when it contains no credential shape."""

    label, _ = sanitize_public_label(value)
    return label


def _public_final_verification_data(data: Mapping[str, object]) -> dict[str, object]:
    """Remove private verifier labels from terminal evidence before event emission."""

    public = dict(data)
    labels_redacted = False
    for field in _FINAL_VERIFICATION_LABEL_FIELDS:
        if field not in public:
            continue
        labels, redacted = _public_label_sequence(public[field])
        public[field] = labels
        labels_redacted |= redacted

    raw_evidence = public.get("evidence")
    visible_evidence: list[dict[str, object]] = []
    if isinstance(raw_evidence, Sequence) and not isinstance(raw_evidence, str | bytes):
        for candidate in raw_evidence:
            if not isinstance(candidate, Mapping):
                labels_redacted = True
                continue
            label = public_verifier_label(candidate.get("label"))
            if label is None:
                labels_redacted = True
                continue
            visible_evidence.append(
                {
                    "label": label,
                    "kind": candidate.get("kind"),
                    "passed": candidate.get("passed"),
                    "step": candidate.get("step"),
                    "epoch": candidate.get("epoch"),
                }
            )
    elif raw_evidence is not None:
        labels_redacted = True
    if "evidence" in public:
        public["evidence"] = visible_evidence
    if "evidence_count" in public:
        public["evidence_count"] = len(visible_evidence)
    if labels_redacted:
        public["labels_redacted"] = True
    return public


def _public_label_sequence(value: object) -> tuple[list[str], bool]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return [], True
    labels: list[str] = []
    redacted = False
    for candidate in value:
        label = public_verifier_label(candidate)
        if label is None:
            redacted = True
            continue
        labels.append(label)
    return labels, redacted


def _public_executable_name(value: str) -> str | None:
    portable = value.replace("\\", "/").rstrip("/")
    candidate = portable.rsplit("/", maxsplit=1)[-1].strip()
    lowered = candidate.casefold()
    for suffix in _EXECUTABLE_SUFFIXES:
        if lowered.endswith(suffix):
            candidate = candidate[: -len(suffix)]
            lowered = lowered[: -len(suffix)]
            break
    if (
        not candidate
        or candidate in {".", ".."}
        or len(candidate) > 120
        or _contains_display_control(candidate)
        or _contains_sensitive_marker(candidate)
    ):
        return None
    return lowered


def _contains_sensitive_marker(value: str) -> bool:
    normalized = "_".join(
        part
        for part in "".join(
            character if character.isalnum() else "_" for character in value.upper()
        ).split("_")
        if part
    )
    padded = f"_{normalized}_"
    return any(f"_{marker}_" in padded for marker in _SENSITIVE_ARGUMENT_MARKERS)


def _looks_absolute_path(value: str) -> bool:
    return value.startswith(("/", "\\\\")) or (
        len(value) >= 3 and value[1] == ":" and value[2] in {"/", "\\"}
    )


def _contains_display_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _public_command_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    """Allow only stable command facts; arbitrary adapter metadata stays private."""

    public: dict[str, object] = {}
    string_values = {
        "status": {"exited", "timed_out", "control_failed", "integrity_failed"},
        "command_class": {"verifier", "read_only", "general"},
        "integrity_phase": {"before", "after"},
        "permission_mode": {"safe", "auto"},
        "reason": {
            "destructive_executable",
            "destructive_git_operation",
            "general_command",
            "implicit_shell",
            "user_denied",
        },
    }
    for key, allowed in string_values.items():
        value = metadata.get(key)
        if isinstance(value, str) and value in allowed:
            public[key] = value
    cwd = metadata.get("cwd")
    if isinstance(cwd, str) and _public_relative_path(cwd):
        public["cwd"] = cwd.replace("\\", "/")
    encoding = metadata.get("output_encoding")
    if (
        isinstance(encoding, str)
        and 1 <= len(encoding) <= 40
        and not _contains_display_control(encoding)
    ):
        public["output_encoding"] = encoding
    executable = metadata.get("executable")
    if isinstance(executable, str):
        safe_executable = _public_executable_name(executable)
        if safe_executable is not None:
            public["executable"] = safe_executable
    for key in ("exit_code", "total_output_bytes", "captured_output_bytes"):
        value = metadata.get(key)
        if (
            isinstance(value, int)
            and not isinstance(value, bool)
            and -(2**31) <= value <= 2**63 - 1
        ):
            public[key] = value
    for key in ("timed_out", "termination_failed", "integrity_intact"):
        value = metadata.get(key)
        if isinstance(value, bool):
            public[key] = value
    return public


def _public_relative_path(value: str) -> bool:
    if not value or len(value) > 1_000 or _contains_display_control(value):
        return False
    portable = value.replace("\\", "/")
    return not _looks_absolute_path(value) and ".." not in portable.split("/")


def _tool_result_error_code(message: ChatMessage) -> str | None:
    if message.role is not MessageRole.TOOL or message.content is None:
        return None
    try:
        payload = json.loads(message.content)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("ok") is not False:
        return None
    error_code = payload.get("error_code")
    return error_code if isinstance(error_code, str) else None
