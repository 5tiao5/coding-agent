"""Pure, bounded unified-diff preview construction."""

from __future__ import annotations

import difflib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DiffPreview:
    """A rendered diff and its independently computed completeness metadata."""

    content: str
    added_lines: int
    removed_lines: int
    complete: bool = True
    truncation_reason: str | None = None


def build_diff_preview(
    path: str,
    before: str,
    after: str,
    *,
    before_exists: bool,
    after_exists: bool,
    before_newline: str,
    after_newline: str,
    max_lines: int,
) -> DiffPreview:
    """Build a bounded unified-diff preview without touching workspace or session state."""
    before_lines = _logical_lines(before)
    after_lines = _logical_lines(after)
    from_file = f"a/{path}" if before_exists else "/dev/null"
    to_file = f"b/{path}" if after_exists else "/dev/null"
    complete = True
    truncation_reason: str | None = None

    if len(before_lines) + len(after_lines) <= max_lines:
        rendered = list(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=from_file,
                tofile=to_file,
                lineterm="",
                n=3,
            )
        )
        added = sum(line.startswith("+") and not line.startswith("+++") for line in rendered)
        removed = sum(line.startswith("-") and not line.startswith("---") for line in rendered)
    else:
        rendered, added, removed, complete = _bounded_large_diff(
            before_lines,
            after_lines,
            from_file=from_file,
            to_file=to_file,
            max_lines=max_lines,
        )
        if not complete:
            truncation_reason = "diff_line_budget"

    if not rendered and (
        before_exists != after_exists or before != after or before_newline != after_newline
    ):
        rendered = [f"--- {from_file}", f"+++ {to_file}"]
        shared_line = next(iter(before_lines or after_lines), None)
        if shared_line is not None:
            rendered.extend(["@@ -1 +1 @@", f" {shared_line}"])

    if before_newline != after_newline:
        rendered.append(f"\\ newline style: {before_newline} -> {after_newline}")
    if before_exists and before and not before.endswith("\n"):
        rendered.append("\\ original file has no newline at end")
    if after_exists and after and not after.endswith("\n"):
        rendered.append("\\ new file has no newline at end")
    if not rendered and before_exists != after_exists:
        rendered = [f"--- {from_file}", f"+++ {to_file}"]

    return DiffPreview(
        content="\n".join(rendered),
        added_lines=added,
        removed_lines=removed,
        complete=complete,
        truncation_reason=truncation_reason,
    )


def _bounded_large_diff(
    before_lines: list[str],
    after_lines: list[str],
    *,
    from_file: str,
    to_file: str,
    max_lines: int,
) -> tuple[list[str], int, int, bool]:
    """Build one exact changed-span hunk, truncating only a very large middle span."""
    prefix = 0
    shared_limit = min(len(before_lines), len(after_lines))
    while prefix < shared_limit and before_lines[prefix] == after_lines[prefix]:
        prefix += 1

    suffix = 0
    while (
        suffix < len(before_lines) - prefix
        and suffix < len(after_lines) - prefix
        and before_lines[-(suffix + 1)] == after_lines[-(suffix + 1)]
    ):
        suffix += 1

    before_middle_end = len(before_lines) - suffix if suffix else len(before_lines)
    after_middle_end = len(after_lines) - suffix if suffix else len(after_lines)
    before_middle = before_lines[prefix:before_middle_end]
    after_middle = after_lines[prefix:after_middle_end]
    context_before = before_lines[max(0, prefix - 3) : prefix]
    context_after_start = len(before_lines) - suffix
    context_after = before_lines[context_after_start : context_after_start + min(3, suffix)]

    changed_budget = max(2, max_lines - len(context_before) - len(context_after))
    complete = len(before_middle) + len(after_middle) <= changed_budget
    if complete:
        rendered_before = [f"-{line}" for line in before_middle]
        rendered_after = [f"+{line}" for line in after_middle]
    else:
        if before_middle and after_middle:
            before_budget = max(1, changed_budget // 2)
            after_budget = max(1, changed_budget - before_budget)
        elif before_middle:
            before_budget, after_budget = changed_budget, 0
        else:
            before_budget, after_budget = 0, changed_budget
        rendered_before = _sample_changed_lines(
            before_middle, prefix="-", budget=before_budget, label="removed"
        )
        rendered_after = _sample_changed_lines(
            after_middle, prefix="+", budget=after_budget, label="added"
        )

    old_count = len(context_before) + len(before_middle) + len(context_after)
    new_count = len(context_before) + len(after_middle) + len(context_after)
    old_start = max(0, prefix - len(context_before)) + (1 if old_count else 0)
    new_start = max(0, prefix - len(context_before)) + (1 if new_count else 0)
    hunk = (
        f"@@ -{_format_hunk_range(old_start, old_count)} "
        f"+{_format_hunk_range(new_start, new_count)} @@"
    )
    rendered = [
        f"--- {from_file}",
        f"+++ {to_file}",
        hunk,
        *(f" {line}" for line in context_before),
        *rendered_before,
        *rendered_after,
        *(f" {line}" for line in context_after),
    ]
    return rendered, len(after_middle), len(before_middle), complete


def _sample_changed_lines(lines: list[str], *, prefix: str, budget: int, label: str) -> list[str]:
    if not lines or budget <= 0:
        return []
    if len(lines) <= budget:
        return [f"{prefix}{line}" for line in lines]
    if budget == 1:
        return [f"...[{len(lines)} {label} lines omitted from diff preview]..."]
    head = budget // 2
    tail = budget - head
    omitted = len(lines) - head - tail
    return [
        *(f"{prefix}{line}" for line in lines[:head]),
        f"...[{omitted} {label} lines omitted from diff preview]...",
        *(f"{prefix}{line}" for line in lines[-tail:]),
    ]


def _format_hunk_range(start: int, count: int) -> str:
    return str(start) if count == 1 else f"{start},{count}"


def _logical_lines(text: str) -> list[str]:
    """Split only logical LF so every other Unicode separator remains escapable data."""
    if not text:
        return []
    lines = text.split("\n")
    return lines[:-1] if text.endswith("\n") else lines
