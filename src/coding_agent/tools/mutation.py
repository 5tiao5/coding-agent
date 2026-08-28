"""Bounded model-facing tools for auditable workspace text changes."""

from __future__ import annotations

from typing import Annotated, Literal
from unicodedata import category

from pydantic import BaseModel, Field, StringConstraints

from coding_agent.models import ToolOutput
from coding_agent.mutation import MutationResult, MutationSession
from coding_agent.tools.base import BaseTool

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

_DIFF_TRUNCATION_MARKER = "...[diff preview truncated; file change was applied completely]"
_SUMMARY_PATH_CHARS = 180


class WriteFileArguments(BaseModel):
    path: str = Field(
        min_length=1,
        max_length=1000,
        description="Workspace-relative file path; its parent directory must already exist.",
    )
    content: str = Field(
        max_length=1_000_000,
        description="Complete UTF-8 text content. A final newline is never added implicitly.",
    )
    expected_sha256: Sha256 | None = Field(
        ...,
        description=(
            "SHA-256 returned by read_file for an overwrite, or null only when the file "
            "must not exist. This is never an unchecked-write switch."
        ),
    )
    newline: Literal["preserve", "lf", "crlf"] = Field(
        default="preserve",
        description="Preserve the existing convention, or explicitly write LF/CRLF.",
    )


class ReplaceTextArguments(BaseModel):
    path: str = Field(
        min_length=1,
        max_length=1000,
        description="Workspace-relative UTF-8 file path.",
    )
    old_text: str = Field(
        min_length=1,
        max_length=250_000,
        description="Exact literal text to replace; regular expressions are not interpreted.",
    )
    new_text: str = Field(
        max_length=250_000,
        description="Exact replacement text.",
    )
    expected_sha256: Sha256 = Field(description="Current raw-byte SHA-256 returned by read_file.")
    expected_occurrences: int = Field(
        default=1,
        ge=1,
        le=1000,
        description="Required number of non-overlapping exact matches.",
    )


class UndoChangeArguments(BaseModel):
    change_id: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Change identifier returned by write_file or replace_text.",
    )


class WriteFileTool(BaseTool[WriteFileArguments]):
    name = "write_file"
    description = (
        "Create or rewrite one UTF-8 workspace file with a SHA-256 stale-write check, atomic "
        "file replacement, a bounded diff preview, and session undo. Never creates parents."
    )
    args_model = WriteFileArguments

    def __init__(self, session: MutationSession, *, max_output_chars: int = 16_000) -> None:
        _validate_output_budget(max_output_chars)
        self._session = session
        self._max_output_chars = max_output_chars
        self.output_budget_chars = max_output_chars

    def run(self, arguments: WriteFileArguments) -> ToolOutput:
        result = self._session.write_text(
            path=arguments.path,
            content=arguments.content,
            expected_sha256=arguments.expected_sha256,
            newline=arguments.newline,
        )
        return _render_result(result, self._session.revision, self._max_output_chars)


class ReplaceTextTool(BaseTool[ReplaceTextArguments]):
    name = "replace_text"
    description = (
        "Replace an exact counted text fragment in one UTF-8 file. Requires read_file's "
        "SHA-256, preserves BOM and uniform line endings, and rejects stale or ambiguous edits."
    )
    args_model = ReplaceTextArguments

    def __init__(self, session: MutationSession, *, max_output_chars: int = 16_000) -> None:
        _validate_output_budget(max_output_chars)
        self._session = session
        self._max_output_chars = max_output_chars
        self.output_budget_chars = max_output_chars

    def run(self, arguments: ReplaceTextArguments) -> ToolOutput:
        result = self._session.replace_text(
            path=arguments.path,
            old_text=arguments.old_text,
            new_text=arguments.new_text,
            expected_sha256=arguments.expected_sha256,
            expected_occurrences=arguments.expected_occurrences,
        )
        return _render_result(result, self._session.revision, self._max_output_chars)


class UndoChangeTool(BaseTool[UndoChangeArguments]):
    name = "undo_change"
    description = (
        "Undo the latest session change by id if the current file still has the exact "
        "post-change SHA-256 and file identity. Refuses later or external replacements."
    )
    args_model = UndoChangeArguments

    def __init__(self, session: MutationSession, *, max_output_chars: int = 16_000) -> None:
        _validate_output_budget(max_output_chars)
        self._session = session
        self._max_output_chars = max_output_chars
        self.output_budget_chars = max_output_chars

    def run(self, arguments: UndoChangeArguments) -> ToolOutput:
        result = self._session.undo_change(arguments.change_id)
        return _render_result(result, self._session.revision, self._max_output_chars)


def _render_result(result: MutationResult, revision: int, max_chars: int) -> ToolOutput:
    if result.idempotent_replay:
        headline = f"Change {result.change_id} was already in the requested state."
    elif not result.changed:
        headline = f"No change was needed for {result.path}; bytes already match."
    elif result.change_kind == "undo":
        headline = f"Change {result.change_id} was undone completely."
    else:
        headline = f"Change {result.change_id} was applied completely."

    diff_complete = result.diff_complete
    truncation_reasons = (
        [result.diff_truncation_reason] if result.diff_truncation_reason is not None else []
    )
    if result.diff:
        visible_diff = "\n".join(_visible_line(line) for line in result.diff.split("\n"))
        content = f"{headline}\nDiff preview:\n{visible_diff}"
    else:
        content = headline
    if len(content) > max_chars:
        keep = max_chars - len(_DIFF_TRUNCATION_MARKER) - 1
        content = _safe_visible_prefix(content, keep) + "\n" + _DIFF_TRUNCATION_MARKER
        diff_complete = False
        truncation_reasons.append("diff_output_limit")

    action = {
        "create": "Created",
        "update": "Updated",
        "undo": "Undid",
        "noop": "Skipped unchanged",
        "replay": "Replayed",
    }[result.change_kind]
    summary = (
        f"{action} {_summary_path(result.path)} "
        f"(+{result.added_lines}/-{result.removed_lines}, change {result.change_id or 'none'})"
    )
    return ToolOutput(
        content=content,
        summary=summary,
        metadata={
            "path": result.path,
            "change_id": result.change_id,
            "changed": result.changed,
            "change_kind": result.change_kind,
            "before_sha256": result.before_sha256,
            "after_sha256": result.after_sha256,
            "before_bytes": result.before_bytes,
            "after_bytes": result.after_bytes,
            "added_lines": result.added_lines,
            "removed_lines": result.removed_lines,
            "occurrences_replaced": result.occurrences_replaced,
            "newline": result.newline,
            "before_newline": result.before_newline,
            "before_ends_with_newline": result.before_ends_with_newline,
            "ends_with_newline": result.ends_with_newline,
            "utf8_bom": result.utf8_bom,
            "idempotent_replay": result.idempotent_replay,
            "mutation_revision": revision,
            "diff_complete": diff_complete,
            "truncation_reason": ",".join(truncation_reasons) or None,
            "durability_uncertain": result.durability_uncertain,
        },
        truncated=not diff_complete,
    )


def _visible_line(text: str) -> str:
    rendered: list[str] = []
    for character in text:
        if category(character) not in {"Cc", "Cf", "Zl", "Zp"}:
            rendered.append(character)
            continue
        codepoint = ord(character)
        if codepoint <= 0xFF:
            rendered.append(f"\\x{codepoint:02x}")
        elif codepoint <= 0xFFFF:
            rendered.append(f"\\u{codepoint:04x}")
        else:
            rendered.append(f"\\U{codepoint:08x}")
    return "".join(rendered)


def _summary_path(path: str) -> str:
    if len(path) <= _SUMMARY_PATH_CHARS:
        return path
    return "..." + path[-(_SUMMARY_PATH_CHARS - 3) :]


def _safe_visible_prefix(text: str, max_chars: int) -> str:
    """Clip text without splitting one of our rendered control escape tokens."""
    index = 0
    while index < len(text):
        token_length = 1
        if text[index] == "\\" and index + 1 < len(text):
            widths = {"x": 4, "u": 6, "U": 10}
            candidate_length = widths.get(text[index + 1])
            if candidate_length is not None:
                candidate = text[index + 2 : index + candidate_length]
                if len(candidate) == candidate_length - 2 and all(
                    character in "0123456789abcdef" for character in candidate
                ):
                    token_length = candidate_length
        if index + token_length > max_chars:
            break
        index += token_length
    return text[:index]


def _validate_output_budget(max_output_chars: int) -> None:
    minimum = len(_DIFF_TRUNCATION_MARKER) + 40
    if max_output_chars < minimum:
        raise ValueError("max_output_chars is too small for a mutation result and marker")
