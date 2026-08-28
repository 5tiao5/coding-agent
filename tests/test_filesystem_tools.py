"""Behaviour and safety tests for the built-in read-only repository tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from coding_agent.models import ToolCall, ToolExecution
from coding_agent.tools import ListFilesTool, ReadFileTool, SearchTextTool, ToolRegistry
from coding_agent.workspace import DirectoryScan, Workspace, WorkspaceError, WorkspacePath


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    _write(root / ".gitignore", "ignored.txt\nignored-dir/\n")
    _write(root / "README.md", "Project Needle\n")
    _write(root / "empty.txt", "")
    _write(root / "ignored.txt", "Needle must stay hidden\n")
    _write(root / "ignored-dir" / "hidden.py", "Needle must stay hidden\n")
    _write(root / ".env", "TOKEN=do-not-read\n")
    _write(root / ".git" / "config", "Needle must stay hidden\n")
    _write(
        root / "src" / "alpha.py",
        "Alpha\nneedle .*[] literal\n南京 module\ncontrol \x01 marker\n",
    )
    _write(root / "src" / "nested" / "beta.py", "NEEDLE again\nlast line\n")
    (root / "blob.bin").write_bytes(b"before\x00Needle\nafter")
    (root / "legacy.txt").write_bytes(b"\xffNeedle")
    _write(root / "large.txt", "x" * 2_000)
    return root


def _execute(
    tool: ListFilesTool | ReadFileTool | SearchTextTool,
    arguments: dict[str, object],
) -> ToolExecution:
    return ToolRegistry([tool]).execute(
        ToolCall(id=f"{tool.name}-1", name=tool.name, arguments=arguments)
    )


def test_filesystem_tool_schemas_are_strict_and_model_facing(repository: Path) -> None:
    registry = ToolRegistry(
        [
            ListFilesTool(Workspace(repository)),
            ReadFileTool(Workspace(repository)),
            SearchTextTool(Workspace(repository)),
        ]
    )

    specs = {spec.name: spec for spec in registry.specs()}

    assert list(specs) == ["list_files", "read_file", "search_text"]
    assert all(spec.input_schema["additionalProperties"] is False for spec in specs.values())
    assert (
        "regular expressions are not interpreted"
        in specs["search_text"].input_schema["properties"]["query"]["description"]
    )


def test_list_files_returns_a_stable_breadth_first_relative_tree(repository: Path) -> None:
    execution = _execute(
        ListFilesTool(Workspace(repository)),
        {"path": ".", "max_depth": 3, "limit": 100},
    )

    assert execution.ok is True
    assert execution.output is not None
    assert "[D] src/" in execution.output
    assert "[F] src/alpha.py" in execution.output
    assert "[F] src/nested/beta.py" in execution.output
    assert "ignored.txt" not in execution.output
    assert "ignored-dir" not in execution.output
    assert ".env" not in execution.output
    assert ".git/config" not in execution.output
    assert str(repository) not in execution.output
    assert execution.metadata["entry_count"] == 10
    assert execution.metadata["directory_count"] == 2
    assert execution.truncated is False


def test_list_files_reports_limit_and_output_truncation(repository: Path) -> None:
    execution = _execute(
        ListFilesTool(Workspace(repository), max_output_chars=100),
        {"path": ".", "max_depth": 4, "limit": 3},
    )

    assert execution.ok is True
    assert execution.truncated is True
    assert execution.metadata["entry_count"] == 3
    assert execution.output is not None
    assert "output truncated" in execution.output
    assert len(execution.output) <= 100


def test_list_files_handles_empty_directory_and_invalid_path(repository: Path) -> None:
    (repository / "vacant").mkdir()
    tool = ListFilesTool(Workspace(repository))

    empty = _execute(tool, {"path": "vacant"})
    escaped = _execute(tool, {"path": "../outside"})

    assert empty.ok is True
    assert "[empty directory]" in str(empty.output)
    assert escaped.ok is False
    assert escaped.error_code == "invalid_path"


def test_read_file_supports_bom_crlf_line_ranges_and_pagination(repository: Path) -> None:
    (repository / "paged.txt").write_bytes(
        b"\xef\xbb\xbffirst\r\n\xe5\x8d\x97\xe4\xba\xac\r\nthird\r\nfourth"
    )

    execution = _execute(
        ReadFileTool(Workspace(repository)),
        {"path": "paged.txt", "start_line": 2, "line_count": 2},
    )

    assert execution.ok is True
    assert execution.output is not None
    assert "2 | 南京" in execution.output
    assert "3 | third" in execution.output
    assert "\ufeff" not in execution.output
    assert execution.metadata == {
        "path": "paged.txt",
        "start_line": 2,
        "end_line": 3,
        "requested_end_line": 3,
        "returned_line_count": 2,
        "total_lines": 4,
        "has_more": True,
        "next_start_line": 4,
        "line_truncated": False,
        "truncation_reason": "pagination",
    }
    assert execution.truncated is True


def test_read_file_treats_an_empty_file_as_a_success(repository: Path) -> None:
    execution = _execute(ReadFileTool(Workspace(repository)), {"path": "empty.txt"})

    assert execution.ok is True
    assert execution.summary == "Read empty file empty.txt"
    assert "[empty file]" in str(execution.output)
    assert execution.metadata["total_lines"] == 0
    assert execution.truncated is False


@pytest.mark.parametrize(
    ("path", "expected_code", "tool_kwargs"),
    [
        ("blob.bin", "binary_file", {}),
        ("legacy.txt", "unsupported_encoding", {}),
        ("large.txt", "file_too_large", {"max_file_bytes": 100}),
        ("ignored.txt", "path_ignored", {}),
        (".env", "sensitive_path", {}),
    ],
)
def test_read_file_returns_stable_content_and_policy_errors(
    repository: Path,
    path: str,
    expected_code: str,
    tool_kwargs: dict[str, int],
) -> None:
    execution = _execute(ReadFileTool(Workspace(repository), **tool_kwargs), {"path": path})

    assert execution.ok is False
    assert execution.error_code == expected_code
    assert str(repository) not in str(execution.error_message)


def test_read_file_rejects_out_of_range_lines(repository: Path) -> None:
    execution = _execute(
        ReadFileTool(Workspace(repository)),
        {"path": "README.md", "start_line": 9},
    )

    assert execution.ok is False
    assert execution.error_code == "line_out_of_range"
    assert "1 lines" in str(execution.error_message)

    empty = _execute(
        ReadFileTool(Workspace(repository)),
        {"path": "empty.txt", "start_line": 2},
    )
    assert empty.ok is False
    assert empty.error_code == "line_out_of_range"
    assert "0 lines" in str(empty.error_message)


def test_read_file_marks_long_lines_and_content_as_truncated(repository: Path) -> None:
    _write(repository / "long-line.txt", "abcdefghijk\nsecond\n")
    execution = _execute(
        ReadFileTool(
            Workspace(repository),
            max_line_chars=5,
            max_output_chars=160,
        ),
        {"path": "long-line.txt"},
    )

    assert execution.ok is True
    assert execution.truncated is True
    assert "abcde...[line truncated]" in str(execution.output)
    assert len(str(execution.output)) <= 160


def test_read_file_character_budget_always_advances_pagination(repository: Path) -> None:
    _write(repository / "budgeted.txt", "\n".join("x" * 100 for _ in range(4)))

    execution = _execute(
        ReadFileTool(Workspace(repository), max_output_chars=140),
        {"path": "budgeted.txt", "start_line": 1, "line_count": 4},
    )

    assert execution.ok is True
    assert execution.metadata["returned_line_count"] >= 1
    assert execution.metadata["end_line"] == execution.metadata["returned_line_count"]
    assert execution.metadata["next_start_line"] == execution.metadata["end_line"] + 1
    assert execution.metadata["line_truncated"] is True
    assert execution.truncated is True
    assert len(str(execution.output)) <= 140


def test_read_file_bounds_empty_output_and_long_observable_summaries(tmp_path: Path) -> None:
    source = tmp_path / "empty.txt"
    source.write_bytes(b"")

    class LongDisplayWorkspace:
        def resolve(self, user_path: str, *, expected: str = "any") -> WorkspacePath:
            del user_path, expected
            return WorkspacePath(source, "a" * 490)

        def read_bytes(self, target: WorkspacePath, *, max_bytes: int) -> bytes:
            del target, max_bytes
            return b""

    execution = _execute(
        ReadFileTool(LongDisplayWorkspace(), max_output_chars=60),  # type: ignore[arg-type]
        {"path": "empty.txt"},
    )

    assert execution.ok is True
    assert execution.summary is not None
    assert len(execution.summary) <= 500
    assert len(str(execution.output)) <= 60
    assert "[empty" in str(execution.output)
    assert execution.truncated is True


def test_search_text_is_literal_case_insensitive_and_deterministic(repository: Path) -> None:
    literal = _execute(
        SearchTextTool(Workspace(repository)),
        {"query": ".*[]", "path": "src"},
    )
    insensitive = _execute(
        SearchTextTool(Workspace(repository)),
        {"query": "needle", "path": "src"},
    )

    assert literal.ok is True
    assert literal.metadata["match_count"] == 1
    assert "src/alpha.py:2:8" in str(literal.output)
    assert insensitive.ok is True
    assert insensitive.metadata["match_count"] == 2
    assert str(insensitive.output).index("src/alpha.py") < str(insensitive.output).index(
        "src/nested/beta.py"
    )
    assert "src/alpha.py:2:1" in str(insensitive.output)
    assert "src/nested/beta.py:1:1" in str(insensitive.output)


def test_search_text_case_sensitive_mode_and_no_match_are_successes(repository: Path) -> None:
    sensitive = _execute(
        SearchTextTool(Workspace(repository)),
        {"query": "NEEDLE", "path": "src", "case_sensitive": True},
    )
    absent = _execute(
        SearchTextTool(Workspace(repository)),
        {"query": "not-present", "path": "src"},
    )

    assert sensitive.ok is True
    assert sensitive.metadata["match_count"] == 1
    assert "beta.py" in str(sensitive.output)
    assert absent.ok is True
    assert absent.metadata["match_count"] == 0
    assert "[no matches]" in str(absent.output)


def test_search_text_exact_result_limit_is_not_a_false_truncation(repository: Path) -> None:
    execution = _execute(
        SearchTextTool(Workspace(repository)),
        {"query": "Project Needle", "path": "README.md", "max_results": 1},
    )

    assert execution.ok is True
    assert execution.metadata["match_count"] == 1
    assert execution.metadata["returned_match_count"] == 1
    assert execution.metadata["match_count_is_lower_bound"] is False
    assert execution.truncated is False


def test_search_text_uses_casefold_without_corrupting_original_columns(repository: Path) -> None:
    _write(repository / "unicode.txt", "Straße\nßxNeedle\nı\n")
    tool = SearchTextTool(Workspace(repository))

    expanded = _execute(tool, {"query": "STRASSE", "path": "unicode.txt"})
    shifted = _execute(tool, {"query": "needle", "path": "unicode.txt"})
    dotless = _execute(tool, {"query": "i", "path": "unicode.txt"})

    assert "unicode.txt:1:1" in str(expanded.output)
    assert "unicode.txt:2:3" in str(shifted.output)
    assert dotless.metadata["match_count"] == 0


def test_search_text_skips_unsearchable_files_in_a_directory(repository: Path) -> None:
    execution = _execute(
        SearchTextTool(Workspace(repository), max_file_bytes=100),
        {"query": "Needle", "path": "."},
    )

    assert execution.ok is True
    assert execution.metadata["files_skipped"] == 3
    assert execution.metadata["files_limited"] == 1
    assert execution.metadata["match_count"] == 3
    assert execution.metadata["match_count_is_lower_bound"] is True
    assert execution.truncated is True
    assert "file_size_limit" in str(execution.metadata["truncation_reason"])
    assert "ignored" not in str(execution.output)
    assert ".git" not in str(execution.output)


@pytest.mark.parametrize(
    ("path", "expected_code"),
    [
        ("blob.bin", "binary_file"),
        ("legacy.txt", "unsupported_encoding"),
        ("large.txt", "file_too_large"),
    ],
)
def test_searching_an_unsearchable_file_returns_its_precise_error(
    repository: Path,
    path: str,
    expected_code: str,
) -> None:
    execution = _execute(
        SearchTextTool(Workspace(repository), max_file_bytes=100),
        {"query": "Needle", "path": path},
    )

    assert execution.ok is False
    assert execution.error_code == expected_code


def test_search_text_reports_result_and_walk_limits(repository: Path) -> None:
    result_limited = _execute(
        SearchTextTool(Workspace(repository)),
        {"query": "Needle", "path": ".", "max_results": 1},
    )
    walk_limited = _execute(
        SearchTextTool(Workspace(repository), max_walk_entries=1),
        {"query": "Needle", "path": "."},
    )

    assert result_limited.ok is True
    assert result_limited.truncated is True
    assert result_limited.metadata["match_count"] == 2
    assert result_limited.metadata["returned_match_count"] == 1
    assert result_limited.metadata["match_count_is_lower_bound"] is True
    assert "output truncated" in str(result_limited.output)
    assert walk_limited.ok is True
    assert walk_limited.truncated is True


def test_search_text_bounds_snippets_and_escapes_control_characters(repository: Path) -> None:
    execution = _execute(
        SearchTextTool(Workspace(repository), max_snippet_chars=24),
        {"query": "marker", "path": "src/alpha.py"},
    )

    assert execution.ok is True
    assert "\\x01" in str(execution.output)
    snippet = str(execution.output).split(" | ", maxsplit=1)[1]
    assert len(snippet) <= 24


def test_search_text_bounds_snippets_after_control_character_escaping(repository: Path) -> None:
    _write(repository / "controls.txt", "Needle" + "\x01" * 100 + "\n")

    execution = _execute(
        SearchTextTool(Workspace(repository), max_snippet_chars=30),
        {"query": "Needle", "path": "controls.txt"},
    )

    assert execution.ok is True
    snippet = str(execution.output).split(" | ", maxsplit=1)[1]
    assert len(snippet) <= 30
    assert "\\x01" in snippet


def test_search_text_escapes_bidirectional_format_controls(repository: Path) -> None:
    _write(repository / "bidi.txt", "needle \u202eend\n")

    execution = _execute(
        SearchTextTool(Workspace(repository)),
        {"query": "needle", "path": "bidi.txt"},
    )

    assert execution.ok is True
    assert "\\u202e" in str(execution.output)
    assert "\u202e" not in str(execution.output)


@pytest.mark.parametrize(
    "query",
    ["   ", "two\nlines", "bad\x00query", "a\u2028b", "unsafe\u202equery"],
)
def test_search_text_rejects_non_printable_or_multiline_queries(
    repository: Path,
    query: str,
) -> None:
    execution = _execute(SearchTextTool(Workspace(repository)), {"query": query})

    assert execution.ok is False
    assert execution.error_code == "invalid_arguments"


def test_search_text_deduplicates_internal_file_aliases(tmp_path: Path) -> None:
    root = tmp_path / "aliases"
    root.mkdir()
    target = root / "target.txt"
    _write(target, "Needle\n")
    try:
        for index in range(3):
            (root / f"alias-{index}.txt").symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    execution = _execute(
        SearchTextTool(Workspace(root)),
        {"query": "Needle", "path": "."},
    )

    assert execution.ok is True
    assert execution.metadata["match_count"] == 1
    assert execution.metadata["files_scanned"] == 1
    assert execution.metadata["aliases_skipped"] == 3


def test_search_text_counts_raw_entries_against_one_global_walk_budget(tmp_path: Path) -> None:
    root = tmp_path / "ignored"
    root.mkdir()
    _write(root / ".gitignore", "*.tmp\n")
    for index in range(100):
        _write(root / f"{index:03}.tmp", "Needle\n")

    execution = _execute(
        SearchTextTool(Workspace(root), max_walk_entries=5),
        {"query": "Needle", "path": "."},
    )

    assert execution.ok is True
    assert execution.metadata["entries_examined"] == 5
    assert execution.metadata["match_count_is_lower_bound"] is True
    assert execution.truncated is True
    assert "walk_limit" in str(execution.metadata["truncation_reason"])


def test_search_text_enforces_aggregate_bytes_and_file_count_budgets(tmp_path: Path) -> None:
    root = tmp_path / "aggregate-budget"
    root.mkdir()
    _write(root / "a.txt", "Needle1234")
    _write(root / "b.txt", "Needle5678")

    byte_limited = _execute(
        SearchTextTool(Workspace(root), max_total_bytes=10),
        {"query": "Needle", "path": "."},
    )
    file_limited = _execute(
        SearchTextTool(Workspace(root), max_files=1),
        {"query": "Needle", "path": "."},
    )

    assert byte_limited.ok is True
    assert byte_limited.metadata["bytes_scanned"] == 10
    assert byte_limited.truncated is True
    assert "byte_limit" in str(byte_limited.metadata["truncation_reason"])
    assert file_limited.ok is True
    assert file_limited.metadata["files_scanned"] == 1
    assert file_limited.truncated is True
    assert "walk_limit" in str(file_limited.metadata["truncation_reason"])


def test_nested_workspace_policy_errors_fail_closed(repository: Path) -> None:
    (repository / "src" / ".gitignore").write_bytes(b"\xff")

    listed = _execute(ListFilesTool(Workspace(repository)), {"path": "."})
    searched = _execute(
        SearchTextTool(Workspace(repository)),
        {"query": "needle", "path": "."},
    )

    assert listed.ok is False
    assert listed.error_code == "invalid_workspace"
    assert searched.ok is False
    assert searched.error_code == "invalid_workspace"


def test_skipped_subtrees_and_disappearing_files_are_reported_as_incomplete(
    tmp_path: Path,
) -> None:
    root = tmp_path / "changing"
    root.mkdir()
    _write(root / "vanishing.txt", "Needle\n")

    class ChangingWorkspace(Workspace):
        def children(
            self,
            directory: WorkspacePath,
            *,
            max_entries: int = 10_000,
            max_examined: int = 20_000,
        ) -> DirectoryScan:
            scan = super().children(
                directory,
                max_entries=max_entries,
                max_examined=max_examined,
            )
            for entry in scan.entries:
                if not entry.is_directory and entry.path.exists():
                    entry.path.unlink()
                    break
            return scan

    execution = _execute(
        SearchTextTool(ChangingWorkspace(root)),
        {"query": "Needle", "path": "."},
    )

    assert execution.ok is True
    assert execution.metadata["file_errors"] == 1
    assert execution.metadata["files_skipped"] == 1
    assert execution.truncated is True
    assert "file_error" in str(execution.metadata["truncation_reason"])


def test_nested_io_errors_mark_list_and_search_as_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "subtree-error"
    _write(root / "nested" / "hit.txt", "Needle\n")

    class FailingSubtreeWorkspace(Workspace):
        def children(
            self,
            directory: WorkspacePath,
            *,
            max_entries: int = 10_000,
            max_examined: int = 20_000,
        ) -> DirectoryScan:
            if directory.relative == "nested":
                raise WorkspaceError("io_error", "nested directory became unavailable")
            return super().children(
                directory,
                max_entries=max_entries,
                max_examined=max_examined,
            )

    workspace = FailingSubtreeWorkspace(root)
    listed = _execute(ListFilesTool(workspace), {"path": ".", "max_depth": 3})
    searched = _execute(SearchTextTool(workspace), {"query": "Needle", "path": "."})

    assert listed.ok is True
    assert listed.metadata["skipped_subtrees"] == 1
    assert listed.truncated is True
    assert listed.metadata["truncation_reason"] == "subtree_error"
    assert searched.ok is True
    assert searched.metadata["skipped_subtrees"] == 1
    assert searched.truncated is True
    assert searched.metadata["match_count_is_lower_bound"] is True
    assert searched.metadata["truncation_reason"] == "subtree_error"


def test_tool_constructors_reject_nonsensical_resource_budgets(repository: Path) -> None:
    workspace = Workspace(repository)

    with pytest.raises(ValueError, match="file and line budgets"):
        ReadFileTool(workspace, max_file_bytes=0)
    with pytest.raises(ValueError, match="search budgets"):
        SearchTextTool(workspace, max_files=0)
    with pytest.raises(ValueError, match="max_output_chars"):
        ListFilesTool(workspace, max_output_chars=10)
