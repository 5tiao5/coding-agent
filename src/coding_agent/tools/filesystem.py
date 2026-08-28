"""Bounded, gitignore-aware tools for inspecting a local code workspace."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from coding_agent.errors import CodedError
from coding_agent.models import ToolOutput
from coding_agent.text import TextDocument, TextDocumentError, decode_utf8_document
from coding_agent.tools._rendering import (
    clip_with_ellipsis,
    is_display_control,
    render_visible_text,
    summarize_path,
)
from coding_agent.tools.base import BaseTool, ToolError
from coding_agent.workspace import Workspace, WorkspaceError, WorkspacePath

_OUTPUT_TRUNCATION_MARKER = "...[output truncated; narrow the path or request]"
_EXPECTED_UNSEARCHABLE_CODES = frozenset({"binary_file", "unsupported_encoding"})


@dataclass(frozen=True, slots=True)
class _CollectedFiles:
    files: tuple[WorkspacePath, ...]
    direct_file: bool
    skipped: int
    skipped_subtrees: int
    aliases_skipped: int
    examined: int
    walk_limited: bool


class ListFilesArguments(BaseModel):
    path: str = Field(
        default=".",
        min_length=1,
        max_length=1000,
        description="Workspace-relative directory to list.",
    )
    max_depth: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Maximum number of directory levels to include.",
    )
    limit: int = Field(
        default=200,
        ge=1,
        le=1000,
        description="Maximum number of entries to return.",
    )


class ReadFileArguments(BaseModel):
    path: str = Field(
        min_length=1,
        max_length=1000,
        description="Workspace-relative file to read.",
    )
    start_line: int = Field(default=1, ge=1, description="First one-based line to return.")
    line_count: int = Field(
        default=200,
        ge=1,
        le=500,
        description="Maximum number of lines to return.",
    )


class SearchTextArguments(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=500,
        description="Literal text to find; regular expressions are not interpreted.",
    )
    path: str = Field(
        default=".",
        min_length=1,
        max_length=1000,
        description="Workspace-relative file or directory to search.",
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether letter case must match exactly.",
    )
    max_results: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of matching lines to return.",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        if value.isspace():
            raise ValueError("query cannot contain only whitespace")
        if any(is_display_control(character) for character in value):
            raise ValueError("query must be a single printable line")
        return value


class ListFilesTool(BaseTool[ListFilesArguments]):
    name = "list_files"
    description = (
        "List a name-sorted, bounded workspace tree. Respects .gitignore, reports incomplete "
        "scans, and never traverses linked directories."
    )
    args_model = ListFilesArguments

    def __init__(
        self,
        workspace: Workspace,
        *,
        max_walk_entries: int = 5_000,
        max_output_chars: int = 16_000,
    ) -> None:
        if max_walk_entries < 1:
            raise ValueError("max_walk_entries must be at least 1")
        _validate_output_budget(max_output_chars)
        self._workspace = workspace
        self._max_walk_entries = max_walk_entries
        self._max_output_chars = max_output_chars
        self.output_budget_chars = max_output_chars

    def run(self, arguments: ListFilesArguments) -> ToolOutput:
        start = self._workspace.resolve(arguments.path, expected="directory")
        queue: deque[tuple[WorkspacePath, int]] = deque([(start, 1)])
        rendered_entries: list[str] = []
        file_count = 0
        directory_count = 0
        skipped = 0
        skipped_subtrees = 0
        examined = 0
        limit_truncated = False
        subtree_error = False

        while (
            queue and len(rendered_entries) < arguments.limit and examined < self._max_walk_entries
        ):
            directory, depth = queue.popleft()
            remaining = arguments.limit - len(rendered_entries)
            remaining_walk = self._max_walk_entries - examined
            try:
                scan = self._workspace.children(
                    directory,
                    max_entries=remaining,
                    max_examined=remaining_walk,
                )
            except WorkspaceError as exc:
                if directory == start or exc.code == "invalid_workspace":
                    raise
                skipped += 1
                skipped_subtrees += 1
                subtree_error = True
                continue
            examined += scan.examined
            skipped += scan.skipped
            limit_truncated = limit_truncated or scan.truncated

            for entry in scan.entries:
                if entry.is_directory:
                    directory_count += 1
                    rendered_entries.append(f"[D] {entry.relative}/")
                    if depth < arguments.max_depth:
                        queue.append((WorkspacePath(entry.path, entry.relative), depth + 1))
                else:
                    file_count += 1
                    rendered_entries.append(f"[F] {entry.relative}")

        if queue or examined >= self._max_walk_entries:
            limit_truncated = True

        header = (
            f"Workspace tree under {start.relative} "
            f"(depth <= {arguments.max_depth}, entries={len(rendered_entries)}):"
        )
        body = rendered_entries or ["[empty directory]"]
        scope_truncated = limit_truncated or subtree_error
        content, returned_body_lines, content_truncated, _ = _bounded_document(
            header,
            body,
            self._max_output_chars,
            incomplete=scope_truncated,
        )
        truncated = scope_truncated or content_truncated
        returned_entries = min(returned_body_lines, len(rendered_entries))

        return ToolOutput(
            content=content,
            summary=(
                f"Listed {returned_entries} of {len(rendered_entries)} discovered entries "
                f"under {summarize_path(start.relative)}"
            ),
            metadata={
                "path": start.relative,
                "entry_count": len(rendered_entries),
                "returned_entry_count": returned_entries,
                "file_count": file_count,
                "directory_count": directory_count,
                "skipped": skipped,
                "skipped_subtrees": skipped_subtrees,
                "entries_examined": examined,
                "max_depth": arguments.max_depth,
                "limit": arguments.limit,
                "truncation_reason": _truncation_reason(
                    walk_or_entry_limit=limit_truncated,
                    subtree_error=subtree_error,
                    output_limit=content_truncated,
                ),
            },
            truncated=truncated,
        )


class ReadFileTool(BaseTool[ReadFileArguments]):
    name = "read_file"
    description = (
        "Read a bounded UTF-8 text-file range with line numbers. Rejects ignored, sensitive, "
        "binary, and oversized files."
    )
    args_model = ReadFileArguments

    def __init__(
        self,
        workspace: Workspace,
        *,
        max_file_bytes: int = 1_000_000,
        max_line_chars: int = 2_000,
        max_output_chars: int = 16_000,
    ) -> None:
        if max_file_bytes < 1 or max_line_chars < 1:
            raise ValueError("file and line budgets must be at least 1")
        _validate_output_budget(max_output_chars)
        self._workspace = workspace
        self._max_file_bytes = max_file_bytes
        self._max_line_chars = max_line_chars
        self._max_output_chars = max_output_chars
        self.output_budget_chars = max_output_chars

    def run(self, arguments: ReadFileArguments) -> ToolOutput:
        target = self._workspace.resolve(arguments.path, expected="file")
        document = self._read_utf8(target)
        text = document.text
        lines = text.splitlines()

        if not lines and arguments.start_line != 1:
            raise ToolError(
                "line_out_of_range",
                f"start_line {arguments.start_line} exceeds 0 lines in {target.relative}",
            )
        if not lines:
            content, _, content_truncated, _ = _bounded_document(
                f"File: {target.relative}",
                ["[empty file]"],
                self._max_output_chars,
            )
            return ToolOutput(
                content=content,
                summary=f"Read empty file {summarize_path(target.relative)}",
                metadata={
                    "path": target.relative,
                    **self._document_metadata(document),
                    "start_line": 0,
                    "end_line": 0,
                    "requested_end_line": 0,
                    "returned_line_count": 0,
                    "total_lines": 0,
                    "has_more": False,
                    "next_start_line": None,
                    "line_truncated": False,
                    "truncation_reason": _truncation_reason(output_limit=content_truncated),
                },
                truncated=content_truncated,
            )

        if arguments.start_line > len(lines):
            raise ToolError(
                "line_out_of_range",
                f"start_line {arguments.start_line} exceeds "
                f"{len(lines)} lines in {target.relative}",
            )

        requested_end_line = min(len(lines), arguments.start_line + arguments.line_count - 1)
        width = len(str(requested_end_line))
        selected = lines[arguments.start_line - 1 : requested_end_line]
        rendered: list[str] = []
        clipped_line = False
        for line_number, line in enumerate(selected, start=arguments.start_line):
            visible = render_visible_text(line)
            if len(visible) > self._max_line_chars:
                visible = visible[: self._max_line_chars] + "...[line truncated]"
                clipped_line = True
            rendered.append(f"{line_number:>{width}} | {visible}")

        requested_has_more = requested_end_line < len(lines)
        header = (
            f"File: {target.relative} "
            f"(requested lines {arguments.start_line}-{requested_end_line} of {len(lines)})"
        )
        content, returned_line_count, content_truncated, forced_line_clip = _bounded_document(
            header,
            rendered,
            self._max_output_chars,
            incomplete=requested_has_more,
        )
        end_line = arguments.start_line + returned_line_count - 1
        has_more = end_line < len(lines)
        truncated = has_more or clipped_line or content_truncated
        summary_path = summarize_path(target.relative)

        summary = f"Read {summary_path} lines {arguments.start_line}-{end_line} of {len(lines)}"

        return ToolOutput(
            content=content,
            summary=summary,
            metadata={
                "path": target.relative,
                **self._document_metadata(document),
                "start_line": arguments.start_line,
                "end_line": end_line,
                "requested_end_line": requested_end_line,
                "returned_line_count": returned_line_count,
                "total_lines": len(lines),
                "has_more": has_more,
                "next_start_line": end_line + 1 if has_more else None,
                "line_truncated": clipped_line or forced_line_clip,
                "truncation_reason": _truncation_reason(
                    pagination=requested_has_more,
                    line_limit=clipped_line or forced_line_clip,
                    output_limit=content_truncated,
                ),
            },
            truncated=truncated,
        )

    def _read_utf8(self, target: WorkspacePath) -> TextDocument:
        data = self._workspace.read_bytes(target, max_bytes=self._max_file_bytes)
        try:
            return decode_utf8_document(data)
        except TextDocumentError as exc:
            if exc.reason == "binary":
                raise ToolError(
                    "binary_file", f"binary file cannot be read: {target.relative}"
                ) from exc
            raise ToolError(
                "unsupported_encoding",
                f"file is not valid UTF-8: {target.relative}",
            ) from exc

    @staticmethod
    def _document_metadata(document: TextDocument) -> dict[str, str | int | bool]:
        return {
            "sha256": document.sha256,
            "bytes": len(document.raw),
            "newline": document.newline,
            "utf8_bom": document.utf8_bom,
            "ends_with_newline": document.ends_with_newline,
        }


class SearchTextTool(BaseTool[SearchTextArguments]):
    name = "search_text"
    description = (
        "Find literal text in bounded UTF-8 workspace files. Respects .gitignore, skips "
        "sensitive or binary content, and returns path:line:column matches."
    )
    args_model = SearchTextArguments

    def __init__(
        self,
        workspace: Workspace,
        *,
        max_files: int = 2_000,
        max_walk_entries: int = 5_000,
        max_file_bytes: int = 1_000_000,
        max_total_bytes: int = 25_000_000,
        max_snippet_chars: int = 300,
        max_output_chars: int = 16_000,
    ) -> None:
        if (
            min(
                max_files,
                max_walk_entries,
                max_file_bytes,
                max_total_bytes,
                max_snippet_chars,
            )
            < 1
        ):
            raise ValueError("search budgets must be at least 1")
        _validate_output_budget(max_output_chars)
        self._workspace = workspace
        self._max_files = max_files
        self._max_walk_entries = max_walk_entries
        self._max_file_bytes = max_file_bytes
        self._max_total_bytes = max_total_bytes
        self._max_snippet_chars = max_snippet_chars
        self._max_output_chars = max_output_chars
        self.output_budget_chars = max_output_chars

    def run(self, arguments: SearchTextArguments) -> ToolOutput:
        start = self._workspace.resolve(arguments.path)
        collection = self._collect_files(start)

        matches: list[str] = []
        matches_seen = 0
        files_scanned = 0
        files_skipped = collection.skipped
        file_errors = 0
        files_limited = 0
        bytes_scanned = 0
        result_limit_reached = False
        byte_limit_reached = False
        file_error_incomplete = False

        for file_path in collection.files:
            remaining_total = self._max_total_bytes - bytes_scanned
            if remaining_total <= 0:
                byte_limit_reached = True
                break
            effective_file_limit = min(self._max_file_bytes, remaining_total)
            try:
                data = self._workspace.read_bytes(
                    file_path,
                    max_bytes=effective_file_limit,
                )
                bytes_scanned += len(data)
                text = _decode_searchable_text(data, file_path)
            except CodedError as exc:
                if collection.direct_file:
                    raise
                if exc.code == "file_too_large" and remaining_total < self._max_file_bytes:
                    byte_limit_reached = True
                    break
                if exc.code in {
                    "invalid_workspace",
                    "path_outside_workspace",
                    "sensitive_path",
                    "path_ignored",
                }:
                    raise
                files_skipped += 1
                if exc.code == "file_too_large":
                    files_limited += 1
                elif exc.code not in _EXPECTED_UNSEARCHABLE_CODES:
                    file_errors += 1
                    file_error_incomplete = True
                continue

            files_scanned += 1
            for line_number, line in enumerate(text.splitlines(), start=1):
                match_start = _literal_match_start(
                    line,
                    arguments.query,
                    case_sensitive=arguments.case_sensitive,
                )
                if match_start is None:
                    continue
                matches_seen += 1
                if len(matches) >= arguments.max_results:
                    result_limit_reached = True
                    break
                snippet = _search_snippet(line, match_start, self._max_snippet_chars)
                matches.append(f"{file_path.relative}:{line_number}:{match_start + 1} | {snippet}")
            if result_limit_reached:
                break

        count_label = f"at least {matches_seen}" if result_limit_reached else str(matches_seen)
        header = (
            f"Literal search under {start.relative}: {count_label} matching line(s) "
            f"in {files_scanned} scanned file(s)."
        )
        body = matches or ["[no matches]"]
        scope_truncated = (
            collection.walk_limited
            or collection.skipped_subtrees > 0
            or result_limit_reached
            or byte_limit_reached
            or files_limited > 0
            or file_error_incomplete
        )
        content, returned_body_lines, content_truncated, _ = _bounded_document(
            header,
            body,
            self._max_output_chars,
            incomplete=scope_truncated,
        )
        returned_matches = min(returned_body_lines, len(matches))
        truncated = scope_truncated or content_truncated

        return ToolOutput(
            content=content,
            summary=(
                f"Found {count_label} matches and showed {returned_matches} "
                f"from {files_scanned} files under {summarize_path(start.relative)}"
            ),
            metadata={
                "path": start.relative,
                "match_count": matches_seen,
                "match_count_is_lower_bound": scope_truncated,
                "returned_match_count": returned_matches,
                "files_scanned": files_scanned,
                "files_skipped": files_skipped,
                "file_errors": file_errors,
                "files_limited": files_limited,
                "skipped_subtrees": collection.skipped_subtrees,
                "aliases_skipped": collection.aliases_skipped,
                "entries_examined": collection.examined,
                "bytes_scanned": bytes_scanned,
                "max_results": arguments.max_results,
                "truncation_reason": _truncation_reason(
                    walk_limit=collection.walk_limited,
                    subtree_error=collection.skipped_subtrees > 0,
                    result_limit=result_limit_reached,
                    byte_limit=byte_limit_reached,
                    file_size_limit=files_limited > 0,
                    file_error=file_error_incomplete,
                    output_limit=content_truncated,
                ),
            },
            truncated=truncated,
        )

    def _collect_files(
        self,
        start: WorkspacePath,
    ) -> _CollectedFiles:
        direct_file = start.path.is_file()
        if direct_file:
            return _CollectedFiles((start,), True, 0, 0, 0, 0, False)

        files: list[WorkspacePath] = []
        queue: deque[WorkspacePath] = deque([start])
        seen_files: set[tuple[int, int] | Path] = set()
        skipped = 0
        skipped_subtrees = 0
        aliases_skipped = 0
        entries_seen = 0
        walk_limited = False

        while queue and len(files) < self._max_files and entries_seen < self._max_walk_entries:
            directory = queue.popleft()
            remaining_entries = self._max_walk_entries - entries_seen
            try:
                scan = self._workspace.children(
                    directory,
                    max_entries=remaining_entries,
                    max_examined=remaining_entries,
                )
            except WorkspaceError as exc:
                if directory == start or exc.code == "invalid_workspace":
                    raise
                skipped += 1
                skipped_subtrees += 1
                continue
            entries_seen += scan.examined
            skipped += scan.skipped
            walk_limited = walk_limited or scan.truncated
            for entry in scan.entries:
                candidate = WorkspacePath(entry.path, entry.relative)
                if entry.is_directory:
                    queue.append(candidate)
                else:
                    identity = _file_identity(entry.path)
                    if identity in seen_files:
                        aliases_skipped += 1
                        continue
                    seen_files.add(identity)
                    if len(files) < self._max_files:
                        files.append(candidate)
                    else:
                        walk_limited = True
                        break

        if queue:
            walk_limited = True
        return _CollectedFiles(
            tuple(files),
            False,
            skipped,
            skipped_subtrees,
            aliases_skipped,
            entries_seen,
            walk_limited,
        )


def _decode_searchable_text(data: bytes, target: WorkspacePath) -> str:
    if b"\x00" in data:
        raise ToolError("binary_file", f"binary file cannot be searched: {target.relative}")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ToolError(
            "unsupported_encoding",
            f"file is not valid searchable UTF-8: {target.relative}",
        ) from exc


def _search_snippet(line: str, match_start: int, max_chars: int) -> str:
    visible = render_visible_text(line)
    visible_match_start = len(render_visible_text(line[:match_start]))
    if len(visible) <= max_chars:
        return visible
    if max_chars <= 6:
        return clip_with_ellipsis(visible[visible_match_start:], max_chars)

    window_chars = max_chars - 6
    left = max(0, visible_match_start - window_chars // 3)
    left = min(left, max(0, len(visible) - window_chars))
    right = min(len(visible), left + window_chars)
    prefix = "..." if left else ""
    suffix = "..." if right < len(visible) else ""
    return prefix + visible[left:right] + suffix


def _literal_match_start(line: str, query: str, *, case_sensitive: bool) -> int | None:
    if case_sensitive:
        match_start = line.find(query)
        return match_start if match_start >= 0 else None

    folded_query = query.casefold()
    if not folded_query:
        return None
    folded_parts: list[str] = []
    boundary_to_original = {0: 0}
    folded_length = 0
    for index, character in enumerate(line):
        folded_character = character.casefold()
        folded_parts.append(folded_character)
        folded_length += len(folded_character)
        boundary_to_original[folded_length] = index + 1

    folded_line = "".join(folded_parts)
    search_from = 0
    while True:
        folded_start = folded_line.find(folded_query, search_from)
        if folded_start < 0:
            return None
        folded_end = folded_start + len(folded_query)
        if folded_start in boundary_to_original and folded_end in boundary_to_original:
            return boundary_to_original[folded_start]
        search_from = folded_start + 1


def _file_identity(path: Path) -> tuple[int, int] | Path:
    try:
        file_stat = path.stat()
    except OSError:
        return path
    if file_stat.st_ino:
        return file_stat.st_dev, file_stat.st_ino
    return path


def _truncation_reason(**reasons: bool) -> str | None:
    active = [name for name, enabled in reasons.items() if enabled]
    return ",".join(active) or None


def _bounded_document(
    header: str,
    body: list[str],
    max_chars: int,
    *,
    incomplete: bool = False,
) -> tuple[str, int, bool, bool]:
    complete = "\n".join([header, *body])
    if not incomplete and len(complete) <= max_chars:
        return complete, len(body), False, False

    suffix = "\n" + _OUTPUT_TRUNCATION_MARKER
    prefix_budget = max_chars - len(suffix)
    if len(header) <= prefix_budget:
        parts = [header]
        current_length = len(header)
        included = 0
        for line in body:
            added_length = 1 + len(line)
            if current_length + added_length > prefix_budget:
                break
            parts.append(line)
            current_length += added_length
            included += 1
        if included:
            content_clipped = included < len(body)
            return "\n".join(parts) + suffix, included, content_clipped, False

    clipped_line = clip_with_ellipsis(body[0], prefix_budget)
    return clipped_line + suffix, 1, True, clipped_line != body[0]


def _validate_output_budget(max_output_chars: int) -> None:
    minimum = len(_OUTPUT_TRUNCATION_MARKER) + 1 + 8
    if max_output_chars < minimum:
        raise ValueError("max_output_chars is too small for a marker and visible progress")
