"""Deterministic context preparation without mutating the canonical transcript."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import cast

from coding_agent.models import ChatMessage, MessageRole, ToolSpec


class ContextError(ValueError):
    """Stable, project-owned failure raised while preparing model context."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class ContextBudgetError(ContextError):
    """The configured budget cannot represent the required context."""


class ContextTranscriptError(ContextError):
    """The canonical transcript does not contain complete model/tool blocks."""


@dataclass(frozen=True, slots=True)
class ContextMetadata:
    """Character accounting for one prepared model view."""

    compacted: bool
    original: int
    prepared: int
    compacted_blocks: int


@dataclass(frozen=True, slots=True)
class PreparedContext:
    """The immutable transcript and the independently prepared model view."""

    canonical_transcript: tuple[ChatMessage, ...]
    model_view: tuple[ChatMessage, ...]
    metadata: ContextMetadata


@dataclass(frozen=True, slots=True)
class _ToolBlock:
    messages: tuple[ChatMessage, ...]


@dataclass(frozen=True, slots=True)
class _ToolFact:
    tool_name: str
    ok: bool | None
    error_code: str | None
    truncated: bool


class ContextManager:
    """Build a bounded model view while retaining a lossless canonical transcript.

    The budget is measured using :func:`context_char_count`, a canonical JSON
    representation that includes roles, tool calls, arguments, and tool-result
    identifiers as well as message content. Old history is compacted only at complete
    assistant/tool-block boundaries.
    """

    def __init__(self, *, max_chars: int) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be at least 1")
        self._max_chars = max_chars

    @property
    def max_chars(self) -> int:
        return self._max_chars

    def prepare(
        self,
        transcript: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
    ) -> PreparedContext:
        """Return a bounded model view without modifying ``transcript``."""

        canonical = tuple(message for message in transcript)
        anchors, blocks = _split_transcript(canonical)
        tool_chars = tool_spec_char_count(tools)
        original_chars = context_char_count(canonical) + tool_chars
        anchor_chars = context_char_count(anchors) + tool_chars
        if anchor_chars > self._max_chars:
            raise ContextBudgetError(
                "context_anchor_exceeds_budget",
                f"context anchors require {anchor_chars} characters; budget is {self._max_chars}",
            )

        if original_chars <= self._max_chars:
            model_view = tuple(message for message in canonical)
            return PreparedContext(
                canonical_transcript=canonical,
                model_view=model_view,
                metadata=ContextMetadata(
                    compacted=False,
                    original=original_chars,
                    prepared=original_chars,
                    compacted_blocks=0,
                ),
            )

        # Try the largest possible suffix first. Summary detail may be reduced, but a
        # newer complete block is never sacrificed merely to retain verbose old facts.
        for retained_count in range(len(blocks) - 1, -1, -1):
            compacted_count = len(blocks) - retained_count
            old_blocks = blocks[:compacted_count]
            recent_blocks = blocks[compacted_count:]
            recent_messages = tuple(
                message for block in recent_blocks for message in block.messages
            )
            for summary in _summary_variants(old_blocks):
                candidate = (*anchors, summary, *recent_messages)
                prepared_chars = context_char_count(candidate) + tool_chars
                if prepared_chars <= self._max_chars:
                    return PreparedContext(
                        canonical_transcript=canonical,
                        model_view=tuple(candidate),
                        metadata=ContextMetadata(
                            compacted=True,
                            original=original_chars,
                            prepared=prepared_chars,
                            compacted_blocks=compacted_count,
                        ),
                    )

        raise ContextBudgetError(
            "context_compaction_exceeds_budget",
            "context anchors fit, but the budget cannot represent compacted tool facts",
        )


def context_char_count(messages: Sequence[ChatMessage]) -> int:
    """Return the exact character count used by :class:`ContextManager`."""

    serialized = ",".join(_canonical_message(message) for message in messages)
    return len(f"[{serialized}]")


def tool_spec_char_count(tools: Sequence[ToolSpec]) -> int:
    """Return the schema characters that share the model request budget."""

    if not tools:
        return 0
    serialized = ",".join(
        json.dumps(
            tool.model_dump(mode="json", exclude_defaults=True, exclude_none=True),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        for tool in tools
    )
    return len(f"[{serialized}]")


def _canonical_message(message: ChatMessage) -> str:
    payload = message.model_dump(mode="json", exclude_defaults=True, exclude_none=True)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _split_transcript(
    transcript: tuple[ChatMessage, ...],
) -> tuple[tuple[ChatMessage, ChatMessage], tuple[_ToolBlock, ...]]:
    if len(transcript) < 2:
        raise ContextTranscriptError(
            "context_missing_anchors",
            "canonical transcript requires a leading system message and original user task",
        )
    system, task = transcript[:2]
    if system.role is not MessageRole.SYSTEM or task.role is not MessageRole.USER:
        raise ContextTranscriptError(
            "context_invalid_anchors",
            "first message must be system and second message must be the original user task",
        )

    blocks: list[_ToolBlock] = []
    cursor = 2
    while cursor < len(transcript):
        assistant = transcript[cursor]
        if assistant.role is MessageRole.TOOL:
            raise ContextTranscriptError(
                "context_orphan_tool_result",
                f"tool result at index {cursor} has no assistant tool-call block",
            )
        if assistant.role is not MessageRole.ASSISTANT:
            raise ContextTranscriptError(
                "context_unexpected_message",
                f"expected an assistant tool-call block at index {cursor}",
            )
        if not assistant.tool_calls:
            raise ContextTranscriptError(
                "context_standalone_assistant",
                f"assistant message at index {cursor} has no tool calls",
            )

        call_ids = tuple(call.id for call in assistant.tool_calls)
        if len(set(call_ids)) != len(call_ids):
            raise ContextTranscriptError(
                "context_duplicate_tool_call",
                f"assistant message at index {cursor} repeats a tool call id",
            )
        result_end = cursor + 1 + len(assistant.tool_calls)
        if result_end > len(transcript):
            raise ContextTranscriptError(
                "context_incomplete_tool_block",
                f"assistant tool-call block at index {cursor} is missing results",
            )

        results = transcript[cursor + 1 : result_end]
        for offset, (call, result) in enumerate(zip(assistant.tool_calls, results, strict=True), 1):
            if result.role is not MessageRole.TOOL:
                raise ContextTranscriptError(
                    "context_incomplete_tool_block",
                    f"expected tool result at index {cursor + offset}",
                )
            if result.tool_call_id != call.id or result.tool_name != call.name:
                raise ContextTranscriptError(
                    "context_mismatched_tool_result",
                    f"tool result at index {cursor + offset} does not match call {call.id}",
                )

        blocks.append(_ToolBlock(messages=(assistant, *results)))
        cursor = result_end

    return (system, task), tuple(blocks)


def _summary_variants(blocks: Sequence[_ToolBlock]) -> tuple[ChatMessage, ...]:
    facts = tuple(_extract_tool_fact(message) for block in blocks for message in block.messages[1:])
    successful = sum(fact.ok is True for fact in facts)
    failed = sum(fact.ok is False for fact in facts)
    unknown = len(facts) - successful - failed
    truncated = sum(fact.truncated for fact in facts)
    tools = Counter(fact.tool_name for fact in facts)
    errors = Counter(fact.error_code for fact in facts if fact.error_code is not None)

    detailed_payload: dict[str, object] = {
        "blocks": len(blocks),
        "calls": len(facts),
        "errors": dict(sorted(errors.items())),
        "failed": failed,
        "kind": "compacted_tool_facts",
        "ok": successful,
        "tools": dict(sorted(tools.items())),
        "truncated": truncated,
        "unknown": unknown,
    }
    detailed = json.dumps(
        detailed_payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    compact = (
        f"[tool-facts b={len(blocks)} c={len(facts)} o={successful} "
        f"f={failed} u={unknown} t={truncated}]"
    )
    variants = (
        ChatMessage(role=MessageRole.ASSISTANT, content=detailed),
        ChatMessage(role=MessageRole.ASSISTANT, content=compact),
    )
    return variants


def _extract_tool_fact(message: ChatMessage) -> _ToolFact:
    payload: dict[object, object] = {}
    try:
        decoded: object = json.loads(message.content or "")
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, dict):
        payload = cast(dict[object, object], decoded)

    raw_ok = payload.get("ok")
    ok = raw_ok if isinstance(raw_ok, bool) else None
    raw_error = payload.get("error_code")
    error_code = _safe_fact_token(raw_error) if isinstance(raw_error, str) else None
    raw_truncated = payload.get("truncated")
    truncated = raw_truncated if isinstance(raw_truncated, bool) else False
    return _ToolFact(
        tool_name=_safe_fact_token(message.tool_name or "unknown"),
        ok=ok,
        error_code=error_code,
        truncated=truncated,
    )


def _safe_fact_token(value: str) -> str:
    if 1 <= len(value) <= 64 and all(
        character.isascii() and (character.isalnum() or character in "._-") for character in value
    ):
        return value
    digest = sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()[:12]
    return f"sha256-{digest}"
