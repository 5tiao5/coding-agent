"""Session-scoped, compare-and-swap text mutation orchestration."""

from __future__ import annotations

import difflib
import json
import os
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Literal

from coding_agent.errors import CodedError
from coding_agent.text import (
    TextDocument,
    TextDocumentError,
    WritableNewlineStyle,
    decode_utf8_document,
    encode_utf8_document,
    normalize_argument_text,
)
from coding_agent.workspace import FileIdentity, FileSnapshot, Workspace, WorkspaceError

RequestedNewline = Literal["preserve", "lf", "crlf"]
ChangeKind = Literal["create", "update", "undo", "noop", "replay"]
ChangeState = Literal["applied", "undone"]


class MutationError(CodedError):
    """Expected mutation failure with a stable recovery-oriented code."""


@dataclass(frozen=True, slots=True)
class ChangeRecord:
    """The exact byte preimage and postimage identity for one undoable change."""

    change_id: str
    sequence: int
    request_fingerprint: str
    tool_name: str
    path: str
    path_key: str
    before_data: bytes | None
    before_sha256: str | None
    after_sha256: str
    after_identity: FileIdentity
    state: ChangeState = "applied"


@dataclass(frozen=True, slots=True)
class MutationResult:
    """A complete mutation result with an independently bounded diff preview."""

    path: str
    change_id: str | None
    changed: bool
    change_kind: ChangeKind
    before_sha256: str | None
    after_sha256: str | None
    before_bytes: int
    after_bytes: int
    added_lines: int
    removed_lines: int
    occurrences_replaced: int
    newline: str
    utf8_bom: bool
    diff: str
    before_newline: str = "none"
    before_ends_with_newline: bool = False
    ends_with_newline: bool = False
    diff_complete: bool = True
    diff_truncation_reason: str | None = None
    idempotent_replay: bool = False
    durability_uncertain: bool = False


@dataclass(frozen=True, slots=True)
class _DiffPreview:
    content: str
    added_lines: int
    removed_lines: int
    complete: bool = True
    truncation_reason: str | None = None


class MutationSession:
    """One in-memory mutation ledger shared by all write tools in an agent run.

    The session owns semantic checks and undo state. ``Workspace`` remains the only layer
    allowed to commit bytes, so tools cannot bypass path policy or compare-and-swap checks.
    """

    def __init__(
        self,
        workspace: Workspace,
        *,
        max_file_bytes: int = 1_000_000,
        max_history_records: int = 100,
        max_history_bytes: int = 32_000_000,
        max_diff_lines: int = 5_000,
    ) -> None:
        if min(max_file_bytes, max_history_records, max_history_bytes, max_diff_lines) < 1:
            raise ValueError("mutation budgets must be at least 1")
        self._workspace = workspace
        self._max_file_bytes = max_file_bytes
        self._max_history_records = max_history_records
        self._max_history_bytes = max_history_bytes
        self._max_diff_lines = max_diff_lines
        self._records: list[ChangeRecord] = []
        self._record_by_id: dict[str, int] = {}
        self._active_fingerprints: dict[str, str] = {}
        self._history_bytes = 0
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def records(self) -> tuple[ChangeRecord, ...]:
        return tuple(self._records)

    def write_text(
        self,
        *,
        path: str,
        content: str,
        expected_sha256: str | None,
        newline: RequestedNewline = "preserve",
    ) -> MutationResult:
        """Create or replace one UTF-8 file if the caller's revision token is current."""
        self._validate_expected_hash(expected_sha256)
        if newline not in {"preserve", "lf", "crlf"}:
            raise MutationError("invalid_arguments", "newline must be preserve, lf, or crlf")
        try:
            logical_content = normalize_argument_text(content)
        except ValueError as exc:
            raise MutationError("invalid_arguments", str(exc)) from exc
        self._require_safe_argument_text(logical_content)

        snapshot = self._snapshot_for_change(path)
        fingerprint = self._fingerprint(
            "write_file",
            {
                "path": self._path_key(snapshot.relative),
                "content": logical_content,
                "expected_sha256": expected_sha256,
                "newline": newline,
            },
        )
        replay = self._idempotent_replay(fingerprint, snapshot)
        if replay is not None:
            return replay
        self._require_revision(snapshot, expected_sha256)

        before_document = self._decode_existing(snapshot)
        selected_newline = self._select_write_newline(before_document, newline)
        new_data = self._encode_document(
            logical_content,
            newline=selected_newline,
            utf8_bom=before_document.utf8_bom if before_document is not None else False,
        )
        self._require_output_budget(new_data)
        if snapshot.data == new_data:
            return self._no_change(snapshot, before_document)

        return self._commit(
            tool_name="write_file",
            fingerprint=fingerprint,
            snapshot=snapshot,
            new_data=new_data,
            logical_before=before_document.text if before_document is not None else "",
            logical_after=logical_content,
            occurrences_replaced=0,
        )

    def replace_text(
        self,
        *,
        path: str,
        old_text: str,
        new_text: str,
        expected_sha256: str,
        expected_occurrences: int = 1,
    ) -> MutationResult:
        """Replace an exact, counted text fragment while preserving bytes conventions."""
        self._validate_expected_hash(expected_sha256)
        if expected_occurrences < 1 or expected_occurrences > 1000:
            raise MutationError(
                "invalid_arguments", "expected_occurrences must be between 1 and 1000"
            )
        try:
            logical_old = normalize_argument_text(old_text)
            logical_new = normalize_argument_text(new_text)
        except ValueError as exc:
            raise MutationError("invalid_arguments", str(exc)) from exc
        if not logical_old:
            raise MutationError("invalid_arguments", "old_text must not be empty")
        if logical_old == logical_new:
            raise MutationError("invalid_arguments", "old_text and new_text must differ")
        self._require_safe_argument_text(logical_old)
        self._require_safe_argument_text(logical_new)

        snapshot = self._snapshot_for_change(path)
        fingerprint = self._fingerprint(
            "replace_text",
            {
                "path": self._path_key(snapshot.relative),
                "old_text": logical_old,
                "new_text": logical_new,
                "expected_sha256": expected_sha256,
                "expected_occurrences": expected_occurrences,
            },
        )
        replay = self._idempotent_replay(fingerprint, snapshot)
        if replay is not None:
            return replay
        self._require_revision(snapshot, expected_sha256)

        document = self._decode_existing(snapshot)
        assert document is not None
        if document.newline == "mixed":
            raise MutationError(
                "mixed_newlines",
                f"replace_text cannot safely edit mixed line endings: {snapshot.relative}",
                metadata={"recovery": "normalize_with_write_file_then_retry"},
            )

        actual_occurrences = document.text.count(logical_old)
        if actual_occurrences != expected_occurrences:
            raise MutationError(
                "match_count_mismatch",
                (
                    f"expected {expected_occurrences} occurrence(s) but found "
                    f"{actual_occurrences} in {snapshot.relative}"
                ),
                metadata={
                    "expected_occurrences": expected_occurrences,
                    "actual_occurrences": actual_occurrences,
                    "recovery": "read_file_and_use_more_context",
                },
            )
        if self._has_overlapping_match(document.text, logical_old):
            raise MutationError(
                "overlapping_matches",
                f"matching ranges overlap in {snapshot.relative}; use a larger unique fragment",
                metadata={
                    "actual_occurrences": actual_occurrences,
                    "recovery": "read_file_and_use_more_context",
                },
            )

        selected_newline: WritableNewlineStyle = "crlf" if document.newline == "crlf" else "lf"
        old_bytes = len(
            self._encode_document(logical_old, newline=selected_newline, utf8_bom=False)
        )
        new_bytes = len(
            self._encode_document(logical_new, newline=selected_newline, utf8_bom=False)
        )
        predicted_bytes = len(document.raw) + actual_occurrences * (new_bytes - old_bytes)
        self._require_output_size(predicted_bytes)
        logical_after = document.text.replace(logical_old, logical_new)
        new_data = self._encode_document(
            logical_after,
            newline=selected_newline,
            utf8_bom=document.utf8_bom,
        )
        self._require_output_budget(new_data)
        return self._commit(
            tool_name="replace_text",
            fingerprint=fingerprint,
            snapshot=snapshot,
            new_data=new_data,
            logical_before=document.text,
            logical_after=logical_after,
            occurrences_replaced=expected_occurrences,
        )

    def undo_change(self, change_id: str) -> MutationResult:
        """Undo the newest active change, refusing to overwrite later external edits."""
        record_index = self._record_by_id.get(change_id)
        if record_index is None:
            raise MutationError("change_not_found", f"unknown change id: {change_id}")
        record = self._records[record_index]
        if record.state == "undone":
            return self._idempotent_undo(record)

        latest = next((item for item in reversed(self._records) if item.state == "applied"), None)
        if latest is None or latest.change_id != change_id:
            raise MutationError(
                "undo_conflict",
                f"change {change_id} is not the latest active change",
                metadata={
                    "latest_change_id": latest.change_id if latest is not None else None,
                    "recovery": "undo_latest_change_first",
                },
            )

        current = self._snapshot_for_undo(record.path)
        if current.sha256 != record.after_sha256:
            raise MutationError(
                "undo_conflict",
                f"file changed after {change_id}; refusing to undo: {record.path}",
                metadata=self._conflict_metadata(record.after_sha256, current.sha256),
            )
        if current.identity != record.after_identity:
            raise MutationError(
                "undo_conflict",
                f"file identity changed after {change_id}; refusing to undo: {record.path}",
                metadata={
                    "identity_changed": True,
                    "recovery": "inspect_file_before_retry",
                },
            )

        current_document = self._decode_existing(current)
        assert current_document is not None
        logical_before = ""
        before_document: TextDocument | None = None
        if record.before_data is not None:
            try:
                before_document = decode_utf8_document(record.before_data)
            except TextDocumentError as exc:  # pragma: no cover - records originate here.
                raise RuntimeError("mutation journal contains invalid text bytes") from exc
            logical_before = before_document.text

        after_newline = before_document.newline if before_document is not None else "none"
        diff = self._unified_diff(
            record.path,
            current_document.text,
            logical_before,
            before_exists=True,
            after_exists=record.before_data is not None,
            before_newline=current_document.newline,
            after_newline=after_newline,
        )
        durability_uncertain = False
        restored_identity: FileIdentity | None = None
        try:
            if record.before_data is None:
                self._workspace.remove_if_unchanged(
                    record.path,
                    expected_sha256=record.after_sha256,
                    expected_identity=record.after_identity,
                )
                after_sha256 = None
                after_bytes = 0
                newline = "none"
                utf8_bom = False
            else:
                receipt = self._workspace.commit_bytes(current, record.before_data)
                after_sha256 = receipt.after_sha256
                after_bytes = receipt.bytes_written
                durability_uncertain = receipt.durability_uncertain
                restored_identity = receipt.after_identity
                assert before_document is not None
                newline = before_document.newline
                utf8_bom = before_document.utf8_bom
        except WorkspaceError as exc:
            if exc.code in {"write_conflict", "revision_conflict", "not_found"}:
                metadata = dict(exc.metadata)
                metadata["recovery"] = "inspect_file_before_retry"
                raise MutationError(
                    "undo_conflict",
                    f"file changed while undoing {change_id}: {record.path}",
                    metadata=metadata,
                ) from exc
            raise

        updated_record = replace(record, state="undone")
        self._records[record_index] = updated_record
        if restored_identity is not None:
            for previous_index in range(record_index - 1, -1, -1):
                previous = self._records[previous_index]
                if (
                    previous.state == "applied"
                    and previous.path_key == record.path_key
                    and previous.after_sha256 == after_sha256
                ):
                    self._records[previous_index] = replace(
                        previous, after_identity=restored_identity
                    )
                    break
        self._active_fingerprints.pop(record.request_fingerprint, None)
        self._revision += 1
        return MutationResult(
            path=record.path,
            change_id=change_id,
            changed=True,
            change_kind="undo",
            before_sha256=record.after_sha256,
            after_sha256=after_sha256,
            before_bytes=len(current.data or b""),
            after_bytes=after_bytes,
            added_lines=diff.added_lines,
            removed_lines=diff.removed_lines,
            occurrences_replaced=0,
            newline=newline,
            utf8_bom=utf8_bom,
            diff=diff.content,
            before_newline=current_document.newline,
            before_ends_with_newline=current_document.ends_with_newline,
            ends_with_newline=(
                before_document.ends_with_newline if before_document is not None else False
            ),
            diff_complete=diff.complete,
            diff_truncation_reason=diff.truncation_reason,
            durability_uncertain=durability_uncertain,
        )

    def _commit(
        self,
        *,
        tool_name: str,
        fingerprint: str,
        snapshot: FileSnapshot,
        new_data: bytes,
        logical_before: str,
        logical_after: str,
        occurrences_replaced: int,
    ) -> MutationResult:
        self._require_history_capacity(snapshot.data)
        before_document = self._decode_existing(snapshot)
        try:
            after_document = decode_utf8_document(new_data)
        except TextDocumentError as exc:  # pragma: no cover - new bytes originate here.
            raise RuntimeError("mutation produced invalid text bytes") from exc
        before_newline = before_document.newline if before_document is not None else "none"
        diff = self._unified_diff(
            snapshot.relative,
            logical_before,
            logical_after,
            before_exists=snapshot.data is not None,
            after_exists=True,
            before_newline=before_newline,
            after_newline=after_document.newline,
        )
        try:
            receipt = self._workspace.commit_bytes(snapshot, new_data)
        except WorkspaceError as exc:
            if exc.code == "write_conflict":
                metadata = dict(exc.metadata)
                metadata["recovery"] = "read_file_then_retry"
                raise MutationError(
                    "revision_conflict",
                    f"file changed while committing: {snapshot.relative}",
                    metadata=metadata,
                ) from exc
            raise
        sequence = len(self._records) + 1
        change_id = f"chg_{sequence:04d}_{receipt.after_sha256[:8]}"
        if receipt.after_identity is None:  # pragma: no cover - Workspace guarantees this.
            raise RuntimeError("workspace write receipt is missing file identity")
        record = ChangeRecord(
            change_id=change_id,
            sequence=sequence,
            request_fingerprint=fingerprint,
            tool_name=tool_name,
            path=receipt.relative,
            path_key=self._path_key(receipt.relative),
            before_data=snapshot.data,
            before_sha256=receipt.before_sha256,
            after_sha256=receipt.after_sha256,
            after_identity=receipt.after_identity,
        )
        self._record_by_id[change_id] = len(self._records)
        self._records.append(record)
        self._active_fingerprints[fingerprint] = change_id
        self._history_bytes += len(snapshot.data or b"")
        self._revision += 1
        return MutationResult(
            path=receipt.relative,
            change_id=change_id,
            changed=True,
            change_kind="create" if receipt.created else "update",
            before_sha256=receipt.before_sha256,
            after_sha256=receipt.after_sha256,
            before_bytes=len(snapshot.data or b""),
            after_bytes=receipt.bytes_written,
            added_lines=diff.added_lines,
            removed_lines=diff.removed_lines,
            occurrences_replaced=occurrences_replaced,
            newline=after_document.newline,
            utf8_bom=after_document.utf8_bom,
            diff=diff.content,
            before_newline=before_newline,
            before_ends_with_newline=(
                before_document.ends_with_newline if before_document is not None else False
            ),
            ends_with_newline=after_document.ends_with_newline,
            diff_complete=diff.complete,
            diff_truncation_reason=diff.truncation_reason,
            durability_uncertain=receipt.durability_uncertain,
        )

    def _idempotent_replay(self, fingerprint: str, snapshot: FileSnapshot) -> MutationResult | None:
        change_id = self._active_fingerprints.get(fingerprint)
        if change_id is None:
            return None
        record = self._records[self._record_by_id[change_id]]
        if snapshot.sha256 != record.after_sha256 or snapshot.identity != record.after_identity:
            metadata = self._conflict_metadata(record.after_sha256, snapshot.sha256)
            if snapshot.identity != record.after_identity:
                metadata["identity_changed"] = True
            raise MutationError(
                "revision_conflict",
                f"file changed after request {change_id}: {record.path}",
                metadata=metadata,
            )
        document = self._decode_existing(snapshot)
        return MutationResult(
            path=record.path,
            change_id=record.change_id,
            changed=False,
            change_kind="replay",
            before_sha256=record.after_sha256,
            after_sha256=record.after_sha256,
            before_bytes=len(snapshot.data or b""),
            after_bytes=len(snapshot.data or b""),
            added_lines=0,
            removed_lines=0,
            occurrences_replaced=0,
            newline=document.newline if document is not None else "none",
            utf8_bom=document.utf8_bom if document is not None else False,
            diff="",
            before_newline=document.newline if document is not None else "none",
            before_ends_with_newline=(
                document.ends_with_newline if document is not None else False
            ),
            ends_with_newline=document.ends_with_newline if document is not None else False,
            idempotent_replay=True,
        )

    def _idempotent_undo(self, record: ChangeRecord) -> MutationResult:
        current = self._snapshot_for_undo(record.path)
        expected = record.before_sha256
        if current.sha256 != expected:
            raise MutationError(
                "undo_conflict",
                f"file changed after undoing {record.change_id}: {record.path}",
                metadata=self._conflict_metadata(expected, current.sha256),
            )
        document = self._decode_existing(current)
        return MutationResult(
            path=record.path,
            change_id=record.change_id,
            changed=False,
            change_kind="replay",
            before_sha256=expected,
            after_sha256=expected,
            before_bytes=len(current.data or b""),
            after_bytes=len(current.data or b""),
            added_lines=0,
            removed_lines=0,
            occurrences_replaced=0,
            newline=document.newline if document is not None else "none",
            utf8_bom=document.utf8_bom if document is not None else False,
            diff="",
            before_newline=document.newline if document is not None else "none",
            before_ends_with_newline=(
                document.ends_with_newline if document is not None else False
            ),
            ends_with_newline=document.ends_with_newline if document is not None else False,
            idempotent_replay=True,
        )

    def _decode_existing(self, snapshot: FileSnapshot) -> TextDocument | None:
        if snapshot.data is None:
            return None
        try:
            return decode_utf8_document(snapshot.data)
        except TextDocumentError as exc:
            code = "binary_file" if exc.reason == "binary" else "unsupported_encoding"
            raise MutationError(code, f"file is not editable UTF-8: {snapshot.relative}") from exc

    def _snapshot_for_change(self, path: str) -> FileSnapshot:
        try:
            return self._workspace.snapshot_for_write(path, max_bytes=self._max_file_bytes)
        except WorkspaceError as exc:
            if exc.code == "write_conflict":
                metadata = dict(exc.metadata)
                metadata["recovery"] = "read_file_then_retry"
                raise MutationError(
                    "revision_conflict",
                    f"file changed while being inspected: {path}",
                    metadata=metadata,
                ) from exc
            raise

    def _snapshot_for_undo(self, path: str) -> FileSnapshot:
        try:
            return self._workspace.snapshot_for_write(path, max_bytes=self._max_file_bytes)
        except WorkspaceError as exc:
            metadata = dict(exc.metadata)
            metadata["recovery"] = "inspect_file_before_retry"
            raise MutationError(
                "undo_conflict",
                f"cannot verify current file state for {path}",
                metadata=metadata,
            ) from exc

    @staticmethod
    def _select_write_newline(
        document: TextDocument | None, requested: RequestedNewline
    ) -> WritableNewlineStyle:
        if requested == "lf" or requested == "crlf":
            return requested
        if document is not None and document.newline == "mixed":
            raise MutationError(
                "mixed_newlines",
                "newline='preserve' cannot rewrite a file with mixed line endings",
                metadata={"recovery": "retry_with_explicit_newline"},
            )
        return "crlf" if document is not None and document.newline == "crlf" else "lf"

    @staticmethod
    def _has_overlapping_match(text: str, fragment: str) -> bool:
        previous_end = -1
        start = 0
        while True:
            position = text.find(fragment, start)
            if position < 0:
                return False
            if position < previous_end:
                return True
            previous_end = position + len(fragment)
            start = position + 1

    @staticmethod
    def _path_key(path: str) -> str:
        """Return a platform-aware ledger key without changing the displayed path."""
        normalized = os.path.normcase(path)
        if os.name == "nt":
            normalized = normalized.casefold()
        return normalized.replace("\\", "/")

    @staticmethod
    def _fingerprint(tool_name: str, arguments: dict[str, object]) -> str:
        canonical = json.dumps(
            {"tool": tool_name, "arguments": arguments},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(canonical).hexdigest()

    @staticmethod
    def _conflict_metadata(
        expected: str | None, current: str | None
    ) -> dict[str, str | bool | None]:
        return {
            "expected_sha256": expected,
            "current_sha256": current,
            "recovery": "read_file_then_retry",
        }

    @staticmethod
    def _validate_expected_hash(expected: str | None) -> None:
        if expected is not None and (
            len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise MutationError(
                "invalid_arguments", "expected_sha256 must be null or 64 lowercase hex characters"
            )

    @staticmethod
    def _require_revision(snapshot: FileSnapshot, expected: str | None) -> None:
        if snapshot.sha256 == expected:
            return
        expected_label = expected or "missing"
        current_label = snapshot.sha256 or "missing"
        raise MutationError(
            "revision_conflict",
            (
                f"expected {snapshot.relative} at {expected_label}, "
                f"but current revision is {current_label}"
            ),
            metadata=MutationSession._conflict_metadata(expected, snapshot.sha256),
        )

    def _require_output_budget(self, data: bytes) -> None:
        self._require_output_size(len(data))

    def _require_output_size(self, size: int) -> None:
        if size > self._max_file_bytes:
            raise MutationError(
                "content_too_large",
                f"encoded content exceeds the {self._max_file_bytes}-byte limit",
            )

    @staticmethod
    def _require_safe_argument_text(text: str) -> None:
        if "\x00" in text:
            raise MutationError("invalid_arguments", "editable text cannot contain NUL bytes")
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise MutationError(
                "invalid_arguments", "text contains characters that cannot be encoded as UTF-8"
            ) from exc

    @staticmethod
    def _encode_document(text: str, *, newline: WritableNewlineStyle, utf8_bom: bool) -> bytes:
        try:
            return encode_utf8_document(text, newline=newline, utf8_bom=utf8_bom)
        except UnicodeEncodeError as exc:
            raise MutationError(
                "invalid_arguments", "text contains characters that cannot be encoded as UTF-8"
            ) from exc

    def _require_history_capacity(self, before_data: bytes | None) -> None:
        if len(self._records) >= self._max_history_records:
            raise MutationError(
                "history_limit",
                f"mutation history reached {self._max_history_records} records",
            )
        required = len(before_data or b"")
        if self._history_bytes + required > self._max_history_bytes:
            raise MutationError(
                "history_limit",
                f"mutation history exceeds the {self._max_history_bytes}-byte limit",
            )

    @staticmethod
    def _no_change(snapshot: FileSnapshot, document: TextDocument | None) -> MutationResult:
        return MutationResult(
            path=snapshot.relative,
            change_id=None,
            changed=False,
            change_kind="noop",
            before_sha256=snapshot.sha256,
            after_sha256=snapshot.sha256,
            before_bytes=len(snapshot.data or b""),
            after_bytes=len(snapshot.data or b""),
            added_lines=0,
            removed_lines=0,
            occurrences_replaced=0,
            newline=document.newline if document is not None else "lf",
            utf8_bom=document.utf8_bom if document is not None else False,
            diff="",
            before_newline=document.newline if document is not None else "none",
            before_ends_with_newline=(
                document.ends_with_newline if document is not None else False
            ),
            ends_with_newline=document.ends_with_newline if document is not None else False,
        )

    def _unified_diff(
        self,
        path: str,
        before: str,
        after: str,
        *,
        before_exists: bool,
        after_exists: bool,
        before_newline: str,
        after_newline: str,
    ) -> _DiffPreview:
        before_lines = self._logical_lines(before)
        after_lines = self._logical_lines(after)
        from_file = f"a/{path}" if before_exists else "/dev/null"
        to_file = f"b/{path}" if after_exists else "/dev/null"
        complete = True
        truncation_reason: str | None = None

        if len(before_lines) + len(after_lines) <= self._max_diff_lines:
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
            rendered, added, removed, complete = self._bounded_large_diff(
                before_lines,
                after_lines,
                from_file=from_file,
                to_file=to_file,
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

        return _DiffPreview(
            content="\n".join(rendered),
            added_lines=added,
            removed_lines=removed,
            complete=complete,
            truncation_reason=truncation_reason,
        )

    def _bounded_large_diff(
        self,
        before_lines: list[str],
        after_lines: list[str],
        *,
        from_file: str,
        to_file: str,
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

        changed_budget = max(
            2,
            self._max_diff_lines - len(context_before) - len(context_after),
        )
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
            rendered_before = self._sample_changed_lines(
                before_middle, prefix="-", budget=before_budget, label="removed"
            )
            rendered_after = self._sample_changed_lines(
                after_middle, prefix="+", budget=after_budget, label="added"
            )

        old_count = len(context_before) + len(before_middle) + len(context_after)
        new_count = len(context_before) + len(after_middle) + len(context_after)
        old_start = max(0, prefix - len(context_before)) + (1 if old_count else 0)
        new_start = max(0, prefix - len(context_before)) + (1 if new_count else 0)
        hunk = (
            f"@@ -{self._format_hunk_range(old_start, old_count)} "
            f"+{self._format_hunk_range(new_start, new_count)} @@"
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

    @staticmethod
    def _sample_changed_lines(
        lines: list[str], *, prefix: str, budget: int, label: str
    ) -> list[str]:
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

    @staticmethod
    def _format_hunk_range(start: int, count: int) -> str:
        return str(start) if count == 1 else f"{start},{count}"

    @staticmethod
    def _logical_lines(text: str) -> list[str]:
        """Split only logical LF so every other Unicode separator remains escapable data."""
        if not text:
            return []
        lines = text.split("\n")
        return lines[:-1] if text.endswith("\n") else lines
