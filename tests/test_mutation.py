"""Semantic tests for compare-and-swap text changes and session undo."""

from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path

import pytest

from coding_agent.models import ToolCall, ToolExecution
from coding_agent.mutation import MutationError, MutationSession
from coding_agent.tools import ReplaceTextTool, ToolRegistry, UndoChangeTool, WriteFileTool
from coding_agent.workspace import Workspace


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    (root / "src").mkdir(parents=True)
    return root


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _replace_with_same_bytes(path: Path) -> None:
    original_identity = (path.stat().st_dev, path.stat().st_ino)
    replacement = path.with_name(f"{path.name}.external-replacement")
    replacement.write_bytes(path.read_bytes())
    replacement_identity = (replacement.stat().st_dev, replacement.stat().st_ino)
    assert replacement_identity != original_identity
    os.replace(replacement, path)
    assert (path.stat().st_dev, path.stat().st_ino) == replacement_identity


def _execute(
    tool: WriteFileTool | ReplaceTextTool | UndoChangeTool,
    arguments: dict[str, object],
) -> ToolExecution:
    return ToolRegistry([tool]).execute(
        ToolCall(id=f"{tool.name}-1", name=tool.name, arguments=arguments)
    )


def test_create_update_and_lifo_undo_restore_exact_bytes(repository: Path) -> None:
    session = MutationSession(Workspace(repository))

    created = session.write_text(
        path="src/value.py",
        content="value = 1\n",
        expected_sha256=None,
    )
    updated = session.replace_text(
        path="src/value.py",
        old_text="1",
        new_text="2",
        expected_sha256=created.after_sha256 or "",
    )

    assert created.change_kind == "create"
    assert updated.change_kind == "update"
    assert repository.joinpath("src/value.py").read_bytes() == b"value = 2\n"
    assert session.revision == 2
    assert [record.state for record in session.records] == ["applied", "applied"]

    with pytest.raises(MutationError) as out_of_order:
        session.undo_change(created.change_id or "")
    assert out_of_order.value.code == "undo_conflict"

    undone_update = session.undo_change(updated.change_id or "")
    assert undone_update.change_kind == "undo"
    assert repository.joinpath("src/value.py").read_bytes() == b"value = 1\n"

    undone_create = session.undo_change(created.change_id or "")
    assert undone_create.after_sha256 is None
    assert not repository.joinpath("src/value.py").exists()
    assert session.revision == 4

    repeated_undo = session.undo_change(created.change_id or "")
    assert repeated_undo.changed is False
    assert repeated_undo.idempotent_replay is True


@pytest.mark.skipif(os.name != "nt", reason="Windows paths are case-insensitive")
def test_windows_case_aliases_share_fingerprint_and_undo_chain(repository: Path) -> None:
    target = repository / "src" / "MixedCase.txt"
    initial = b"value = 0\n"
    target.write_bytes(initial)
    session = MutationSession(Workspace(repository))

    first = session.replace_text(
        path="src/MixedCase.txt",
        old_text="0",
        new_text="1",
        expected_sha256=_digest(initial),
    )
    replay = session.replace_text(
        path="SRC/mIXEDcASE.TXT",
        old_text="0",
        new_text="1",
        expected_sha256=_digest(initial),
    )
    second = session.replace_text(
        path="SRC/mIXEDcASE.TXT",
        old_text="1",
        new_text="2",
        expected_sha256=first.after_sha256 or "",
    )

    assert replay.idempotent_replay is True
    assert replay.change_id == first.change_id
    assert session.records[0].path == "src/MixedCase.txt"
    assert session.records[1].path == "SRC/mIXEDcASE.TXT"
    assert session.records[0].path_key == session.records[1].path_key
    assert target.read_bytes() == b"value = 2\n"

    session.undo_change(second.change_id or "")
    assert target.read_bytes() == b"value = 1\n"

    session.undo_change(first.change_id or "")
    assert target.read_bytes() == initial


def test_replace_preserves_bom_crlf_and_missing_final_newline(repository: Path) -> None:
    original = b"\xef\xbb\xbfalpha = 1\r\nbeta = 2"
    target = repository / "src" / "settings.py"
    target.write_bytes(original)
    session = MutationSession(Workspace(repository))

    result = session.replace_text(
        path="src/settings.py",
        old_text="beta = 2",
        new_text="beta = 3",
        expected_sha256=_digest(original),
    )

    assert target.read_bytes() == b"\xef\xbb\xbfalpha = 1\r\nbeta = 3"
    assert result.newline == "crlf"
    assert result.utf8_bom is True
    assert result.occurrences_replaced == 1
    assert "--- a/src/settings.py" in result.diff
    assert "+beta = 3" in result.diff

    replay = session.replace_text(
        path="src/settings.py",
        old_text="beta = 2",
        new_text="beta = 3",
        expected_sha256=_digest(original),
    )
    assert replay.idempotent_replay is True
    assert replay.change_id == result.change_id

    session.undo_change(result.change_id or "")
    assert target.read_bytes() == original


def test_same_successful_request_is_an_idempotent_replay(repository: Path) -> None:
    session = MutationSession(Workspace(repository))

    first = session.write_text(path="src/new.py", content="answer = 42\n", expected_sha256=None)
    replay = session.write_text(path="src/new.py", content="answer = 42\n", expected_sha256=None)

    assert replay.change_id == first.change_id
    assert replay.changed is False
    assert replay.change_kind == "replay"
    assert replay.idempotent_replay is True
    assert len(session.records) == 1
    assert session.revision == 1


def test_write_noop_does_not_consume_history_or_revision(repository: Path) -> None:
    target = repository / "src" / "same.py"
    target.write_bytes(b"same\n")
    session = MutationSession(Workspace(repository))

    result = session.write_text(
        path="src/same.py",
        content="same\n",
        expected_sha256=_digest(b"same\n"),
    )

    assert result.change_kind == "noop"
    assert result.change_id is None
    assert session.records == ()
    assert session.revision == 0


def test_stale_revision_and_external_edit_are_never_overwritten(repository: Path) -> None:
    target = repository / "src" / "counter.py"
    target.write_bytes(b"count = 1\n")
    session = MutationSession(Workspace(repository))

    with pytest.raises(MutationError) as stale:
        session.write_text(
            path="src/counter.py",
            content="count = 2\n",
            expected_sha256=None,
        )
    assert stale.value.code == "revision_conflict"
    assert stale.value.metadata["current_sha256"] == _digest(b"count = 1\n")
    assert stale.value.metadata["recovery"] == "read_file_then_retry"

    changed = session.replace_text(
        path="src/counter.py",
        old_text="1",
        new_text="2",
        expected_sha256=_digest(b"count = 1\n"),
    )
    target.write_bytes(b"count = 99\n")

    with pytest.raises(MutationError) as undo_conflict:
        session.undo_change(changed.change_id or "")
    assert undo_conflict.value.code == "undo_conflict"
    assert target.read_bytes() == b"count = 99\n"
    assert session.records[-1].state == "applied"


def test_commit_time_workspace_conflict_is_normalized_to_revision_conflict(
    repository: Path,
) -> None:
    target = repository / "src" / "racy.txt"
    original = b"value = 1\n"
    target.write_bytes(original)

    class RacyWorkspace(Workspace):
        def _before_mutation_commit(self, operation: str, path: Path) -> None:
            assert operation == "write"
            assert path == target.resolve()
            target.write_bytes(b"value = 9\n")

    session = MutationSession(RacyWorkspace(repository))

    with pytest.raises(MutationError) as caught:
        session.replace_text(
            path="src/racy.txt",
            old_text="value = 1",
            new_text="value = 2",
            expected_sha256=_digest(original),
        )

    assert caught.value.code == "revision_conflict"
    assert caught.value.metadata["recovery"] == "read_file_then_retry"
    assert caught.value.metadata["current_sha256"] == _digest(b"value = 9\n")
    assert target.read_bytes() == b"value = 9\n"
    assert session.records == ()


def test_undo_rejects_an_external_same_byte_replacement(repository: Path) -> None:
    target = repository / "src" / "owned.txt"
    session = MutationSession(Workspace(repository))
    created = session.write_text(path="src/owned.txt", content="same bytes\n", expected_sha256=None)
    _replace_with_same_bytes(target)

    with pytest.raises(MutationError) as caught:
        session.undo_change(created.change_id or "")

    assert caught.value.code == "undo_conflict"
    assert caught.value.metadata["identity_changed"] is True
    assert target.read_bytes() == b"same bytes\n"


@pytest.mark.parametrize(
    ("text", "old", "expected_occurrences", "expected_code"),
    [
        ("alpha", "missing", 1, "match_count_mismatch"),
        ("x x", "x", 1, "match_count_mismatch"),
        ("aaa", "aa", 1, "overlapping_matches"),
    ],
)
def test_replace_requires_an_exact_unambiguous_match_contract(
    repository: Path,
    text: str,
    old: str,
    expected_occurrences: int,
    expected_code: str,
) -> None:
    target = repository / "src" / "ambiguous.txt"
    raw = text.encode()
    target.write_bytes(raw)
    session = MutationSession(Workspace(repository))

    with pytest.raises(MutationError) as caught:
        session.replace_text(
            path="src/ambiguous.txt",
            old_text=old,
            new_text="changed",
            expected_sha256=_digest(raw),
            expected_occurrences=expected_occurrences,
        )

    assert caught.value.code == expected_code
    assert target.read_bytes() == raw
    assert session.records == ()


@pytest.mark.parametrize(
    ("old", "new"),
    [("", "x"), ("x", "x"), ("x\rbroken", "y"), ("x", "bad\x00text")],
)
def test_replace_rejects_invalid_text_arguments(repository: Path, old: str, new: str) -> None:
    target = repository / "src" / "valid.txt"
    target.write_bytes(b"x")
    session = MutationSession(Workspace(repository))

    with pytest.raises(MutationError) as caught:
        session.replace_text(
            path="src/valid.txt",
            old_text=old,
            new_text=new,
            expected_sha256=_digest(b"x"),
        )

    assert caught.value.code == "invalid_arguments"
    assert target.read_bytes() == b"x"


def test_mixed_newlines_require_an_explicit_full_rewrite(repository: Path) -> None:
    target = repository / "src" / "mixed.txt"
    original = b"a\r\nb\n"
    target.write_bytes(original)
    session = MutationSession(Workspace(repository))

    with pytest.raises(MutationError) as replacement:
        session.replace_text(
            path="src/mixed.txt",
            old_text="b",
            new_text="c",
            expected_sha256=_digest(original),
        )
    assert replacement.value.code == "mixed_newlines"

    with pytest.raises(MutationError) as preserve:
        session.write_text(
            path="src/mixed.txt",
            content="a\nc\n",
            expected_sha256=_digest(original),
        )
    assert preserve.value.code == "mixed_newlines"

    normalized = session.write_text(
        path="src/mixed.txt",
        content="a\nc\n",
        expected_sha256=_digest(original),
        newline="crlf",
    )
    assert normalized.changed is True
    assert target.read_bytes() == b"a\r\nc\r\n"


@pytest.mark.parametrize("raw", [b"binary\x00data", b"\xff"])
def test_mutation_rejects_non_utf8_text(repository: Path, raw: bytes) -> None:
    target = repository / "src" / "opaque.dat"
    target.write_bytes(raw)
    session = MutationSession(Workspace(repository))

    with pytest.raises(MutationError) as caught:
        session.write_text(
            path="src/opaque.dat",
            content="safe",
            expected_sha256=_digest(raw),
        )

    assert caught.value.code in {"binary_file", "unsupported_encoding"}
    assert target.read_bytes() == raw


def test_encoded_content_and_history_have_precommit_budgets(repository: Path) -> None:
    small_session = MutationSession(Workspace(repository), max_file_bytes=5)

    with pytest.raises(MutationError) as too_large:
        small_session.write_text(path="src/large.txt", content="南京", expected_sha256=None)
    assert too_large.value.code == "content_too_large"
    assert not repository.joinpath("src/large.txt").exists()

    history_session = MutationSession(Workspace(repository), max_history_records=1)
    history_session.write_text(path="src/first.txt", content="one", expected_sha256=None)
    with pytest.raises(MutationError) as full:
        history_session.write_text(path="src/second.txt", content="two", expected_sha256=None)
    assert full.value.code == "history_limit"
    assert not repository.joinpath("src/second.txt").exists()

    diff_session = MutationSession(Workspace(repository), max_diff_lines=2)
    bounded_diff = diff_session.write_text(
        path="src/many-lines.txt",
        content="one\ntwo\nthree\n",
        expected_sha256=None,
    )
    assert bounded_diff.diff_complete is False
    assert bounded_diff.diff_truncation_reason == "diff_line_budget"
    assert repository.joinpath("src/many-lines.txt").read_bytes() == b"one\ntwo\nthree\n"

    tool_diff_session = MutationSession(Workspace(repository), max_diff_lines=2)
    tool_diff = _execute(
        WriteFileTool(tool_diff_session),
        {
            "path": "src/tool-lines.txt",
            "content": "one\ntwo\nthree\n",
            "expected_sha256": None,
        },
    )
    assert tool_diff.ok is True
    assert tool_diff.truncated is True
    assert tool_diff.metadata["diff_complete"] is False
    assert tool_diff.metadata["truncation_reason"] == "diff_line_budget"

    with pytest.raises(ValueError, match="mutation budgets"):
        MutationSession(Workspace(repository), max_diff_lines=0)


def test_replace_rejects_expansion_before_constructing_the_full_result(repository: Path) -> None:
    target = repository / "src" / "expansion.txt"
    raw = ("x " * 1000).encode()
    target.write_bytes(raw)
    session = MutationSession(Workspace(repository), max_file_bytes=20_000)

    with pytest.raises(MutationError) as caught:
        session.replace_text(
            path="src/expansion.txt",
            old_text="x",
            new_text="y" * 10_000,
            expected_sha256=_digest(raw),
            expected_occurrences=1000,
        )

    assert caught.value.code == "content_too_large"
    assert target.read_bytes() == raw
    assert session.records == ()


def test_large_diff_uses_bounded_complete_or_sampled_changed_spans(repository: Path) -> None:
    narrow_target = repository / "src" / "narrow.py"
    narrow_before = "\n".join(f"line {index}" for index in range(20)) + "\n"
    narrow_target.write_bytes(narrow_before.encode())
    narrow_session = MutationSession(Workspace(repository), max_diff_lines=8)

    narrow = narrow_session.replace_text(
        path="src/narrow.py",
        old_text="line 10",
        new_text="changed 10",
        expected_sha256=_digest(narrow_before.encode()),
    )
    assert narrow.diff_complete is True
    assert "-line 10" in narrow.diff
    assert "+changed 10" in narrow.diff

    broad_target = repository / "src" / "broad.py"
    broad_before = "\n".join(f"old {index}" for index in range(8)) + "\n"
    broad_after = "\n".join(f"new {index}" for index in range(8)) + "\n"
    broad_target.write_bytes(broad_before.encode())
    broad_session = MutationSession(Workspace(repository), max_diff_lines=2)

    broad = broad_session.write_text(
        path="src/broad.py",
        content=broad_after,
        expected_sha256=_digest(broad_before.encode()),
    )
    assert broad.diff_complete is False
    assert "removed lines omitted" in broad.diff
    assert "added lines omitted" in broad.diff

    deletion = broad_session.write_text(
        path="src/broad.py",
        content="",
        expected_sha256=broad.after_sha256,
    )
    assert deletion.diff_complete is False
    assert deletion.removed_lines == 8
    assert broad_target.read_bytes() == b""


@pytest.mark.parametrize("content", ["bad\x00text", "bad\ud800text", "bad\rtext"])
def test_write_rejects_unrepresentable_or_ambiguous_text(repository: Path, content: str) -> None:
    session = MutationSession(Workspace(repository))

    with pytest.raises(MutationError) as caught:
        session.write_text(path="src/bad.txt", content=content, expected_sha256=None)

    assert caught.value.code == "invalid_arguments"
    assert not repository.joinpath("src/bad.txt").exists()


def test_model_facing_tools_expose_cas_and_bounded_safe_diffs(repository: Path) -> None:
    session = MutationSession(Workspace(repository))
    write_tool = WriteFileTool(session, max_output_chars=240)
    registry = ToolRegistry(
        [write_tool, ReplaceTextTool(session), UndoChangeTool(session)],
        max_output_chars=20_000,
    )

    specs = {spec.name: spec for spec in registry.specs()}
    assert list(specs) == ["write_file", "replace_text", "undo_change"]
    assert "expected_sha256" in specs["write_file"].input_schema["required"]
    assert "create_parents" not in specs["write_file"].input_schema["properties"]

    execution = registry.execute(
        ToolCall(
            id="write-1",
            name="write_file",
            arguments={
                "path": "src/colored.txt",
                "content": "\x1b[31m" + "x" * 400,
                "expected_sha256": None,
            },
        )
    )

    assert execution.ok is True
    assert execution.truncated is True
    assert execution.metadata["diff_complete"] is False
    assert execution.metadata["truncation_reason"] == "diff_output_limit"
    assert "\\x1b" in str(execution.output)
    assert "\x1b" not in str(execution.output)
    assert len(str(execution.output)) <= 240
    assert repository.joinpath("src/colored.txt").read_text(encoding="utf-8").endswith("x" * 400)


def test_tool_failure_returns_recovery_metadata_without_leaking_absolute_path(
    repository: Path,
) -> None:
    target = repository / "src" / "current.txt"
    target.write_bytes(b"current")
    session = MutationSession(Workspace(repository))

    execution = _execute(
        WriteFileTool(session),
        {
            "path": "src/current.txt",
            "content": "replacement",
            "expected_sha256": "0" * 64,
        },
    )

    assert execution.ok is False
    assert execution.error_code == "revision_conflict"
    assert execution.metadata["expected_sha256"] == "0" * 64
    assert execution.metadata["current_sha256"] == _digest(b"current")
    assert execution.metadata["recovery"] == "read_file_then_retry"
    assert str(repository) not in str(execution.error_message)


def test_write_tool_reports_noop_and_idempotent_replay_without_raw_bidi(
    repository: Path,
) -> None:
    target = repository / "src" / "stable.txt"
    target.write_bytes(b"stable\n")
    session = MutationSession(Workspace(repository))
    tool = WriteFileTool(session)

    noop = _execute(
        tool,
        {
            "path": "src/stable.txt",
            "content": "stable\n",
            "expected_sha256": _digest(b"stable\n"),
        },
    )
    assert noop.ok is True
    assert noop.metadata["change_kind"] == "noop"
    assert "No change was needed" in str(noop.output)

    dangerous_content = "\u202e\U000e0001safe\u2028hidden\x0bvertical\x85next"
    arguments: dict[str, object] = {
        "path": "src/bidi.txt",
        "content": dangerous_content,
        "expected_sha256": None,
    }
    created = _execute(tool, arguments)
    replay = _execute(tool, arguments)

    assert created.ok is True
    assert "\\u202e" in str(created.output)
    assert "\\U000e0001" in str(created.output)
    assert "\\u2028" in str(created.output)
    assert "\\x0b" in str(created.output)
    assert "\\x85" in str(created.output)
    assert "\u202e" not in str(created.output)
    assert "\u2028" not in str(created.output)
    assert replay.ok is True
    assert replay.metadata["idempotent_replay"] is True
    assert "already in the requested state" in str(replay.output)
    assert len(session.records) == 1


def test_session_rejects_invalid_contracts_and_history_byte_overflow(repository: Path) -> None:
    target = repository / "src" / "history.txt"
    target.write_bytes(b"old")
    session = MutationSession(Workspace(repository), max_history_bytes=2)

    with pytest.raises(MutationError) as bad_hash:
        session.write_text(path="src/new.txt", content="new", expected_sha256="NOT-A-DIGEST")
    assert bad_hash.value.code == "invalid_arguments"

    with pytest.raises(MutationError) as bad_newline:
        session.write_text(
            path="src/new.txt",
            content="new",
            expected_sha256=None,
            newline="native",  # type: ignore[arg-type]
        )
    assert bad_newline.value.code == "invalid_arguments"

    with pytest.raises(MutationError) as bad_occurrences:
        session.replace_text(
            path="src/history.txt",
            old_text="old",
            new_text="new",
            expected_sha256=_digest(b"old"),
            expected_occurrences=0,
        )
    assert bad_occurrences.value.code == "invalid_arguments"

    with pytest.raises(MutationError) as history_full:
        session.replace_text(
            path="src/history.txt",
            old_text="old",
            new_text="new",
            expected_sha256=_digest(b"old"),
        )
    assert history_full.value.code == "history_limit"
    assert target.read_bytes() == b"old"

    with pytest.raises(MutationError) as missing_change:
        session.undo_change("chg_missing")
    assert missing_change.value.code == "change_not_found"

    with pytest.raises(ValueError, match="max_output_chars"):
        WriteFileTool(session, max_output_chars=1)


@pytest.mark.parametrize(
    ("before", "after", "expected_marker", "unexpected_marker"),
    [
        (b"x\n", "x", "new file has no newline", "original file has no newline"),
        (b"x", "x\n", "original file has no newline", "new file has no newline"),
    ],
)
def test_diff_reports_final_newline_direction(
    repository: Path,
    before: bytes,
    after: str,
    expected_marker: str,
    unexpected_marker: str,
) -> None:
    target = repository / "src" / "eof.txt"
    target.write_bytes(before)
    session = MutationSession(Workspace(repository))

    result = session.write_text(
        path="src/eof.txt",
        content=after,
        expected_sha256=_digest(before),
    )

    assert expected_marker in result.diff
    assert unexpected_marker not in result.diff
    assert result.diff_complete is True


def test_result_reports_actual_newline_style_for_single_line_content(repository: Path) -> None:
    target = repository / "src" / "single.txt"
    target.write_bytes(b"before")
    session = MutationSession(Workspace(repository))

    result = session.write_text(
        path="src/single.txt",
        content="after",
        expected_sha256=_digest(b"before"),
    )

    assert result.before_newline == "none"
    assert result.newline == "none"
    assert "newline style" not in result.diff


def test_undo_tool_restores_a_created_file(repository: Path) -> None:
    session = MutationSession(Workspace(repository))
    written = _execute(
        WriteFileTool(session),
        {"path": "src/temporary.txt", "content": "temporary", "expected_sha256": None},
    )
    change_id = str(written.metadata["change_id"])

    undone = _execute(UndoChangeTool(session), {"change_id": change_id})

    assert undone.ok is True
    assert undone.metadata["change_kind"] == "undo"
    assert undone.metadata["after_sha256"] is None
    assert not repository.joinpath("src/temporary.txt").exists()
