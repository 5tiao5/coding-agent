"""Bounded host-owned facts that survive long runs without retaining raw observations."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from typing import Literal, Self

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from coding_agent.errors import CodedError
from coding_agent.models import FrozenModel, ToolCall, ToolExecution, VerificationKind
from coding_agent.plan import PlanSnapshot, PlanState

_MAX_FILE_CHANGES = 64
_MAX_FAILED_COMMANDS = 32
_MAX_VERIFICATION_FACTS = 32
_MAX_COMMAND_ARGUMENTS = 64
_MAX_COMMAND_ARGUMENT_CHARS = 16_000
_MIN_RUN_MEMORY_CHARS = 4_096
_MAX_RUN_MEMORY_CHARS = 64_000

ChangeKind = Literal["create", "update", "undo"]
CommandFailureKind = Literal["nonzero_exit", "timed_out", "control_failed", "tool_error"]


class RunMemoryError(CodedError):
    """Stable failure raised when host-owned memory cannot be restored safely."""


class FileChangeFact(FrozenModel):
    """Latest allowlisted mutation metadata for one workspace-relative path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1, max_length=1_000)
    change_count: int = Field(ge=1)
    last_change_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    last_change_kind: ChangeKind
    before_sha256: str | None = None
    after_sha256: str | None = None
    added_lines: int = Field(ge=0)
    removed_lines: int = Field(ge=0)
    mutation_revision: int = Field(ge=1)
    last_step: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        _require_printable(value, field="path")
        return value

    @field_validator("before_sha256", "after_sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        if value is not None and not _is_sha256(value):
            raise ValueError("file revisions must be lowercase SHA-256 values")
        return value


class FailedCommandFact(FrozenModel):
    """A failed command invocation without stdout, stderr, or arbitrary error detail."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    argv: tuple[str, ...] = Field(min_length=1, max_length=_MAX_COMMAND_ARGUMENTS)
    cwd: str = Field(min_length=1, max_length=1_000)
    failure_kind: CommandFailureKind
    exit_code: int | None = None
    error_code: str | None = Field(default=None, min_length=1, max_length=80)
    output_truncated: bool = False
    step: int = Field(ge=0)

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if sum(len(argument) for argument in value) > _MAX_COMMAND_ARGUMENT_CHARS:
            raise ValueError("remembered command arguments exceed the bounded character limit")
        for argument in value:
            if not argument:
                raise ValueError("remembered command arguments cannot be empty")
            _require_printable(argument, field="command argument")
        return value

    @field_validator("cwd")
    @classmethod
    def validate_cwd(cls, value: str) -> str:
        _require_printable(value, field="command cwd")
        return value

    @model_validator(mode="after")
    def validate_failure_shape(self) -> Self:
        if self.failure_kind == "nonzero_exit":
            if self.exit_code in {None, 0} or self.error_code is not None:
                raise ValueError("nonzero command failures require only a nonzero exit code")
        elif self.failure_kind == "tool_error":
            if self.error_code is None or self.exit_code is not None:
                raise ValueError("tool command failures require only a stable error code")
        elif self.exit_code is not None or self.error_code is not None:
            raise ValueError("timeout and control failures cannot contain exit or error codes")
        return self


class VerificationMemoryFact(FrozenModel):
    """Historical verifier outcome; ``stale`` keeps it outside completion authority."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(min_length=1, max_length=120)
    kind: VerificationKind
    passed: bool
    step: int = Field(ge=0)
    stale: bool = False

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        _require_printable(value, field="verification label")
        return value


class RunMemorySnapshot(FrozenModel):
    """Immutable, bounded checkpoint payload for host-owned run facts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    revision: int = Field(ge=0)
    plan: PlanSnapshot = Field(default_factory=lambda: PlanSnapshot(revision=0))
    file_changes: tuple[FileChangeFact, ...] = Field(default=(), max_length=_MAX_FILE_CHANGES)
    failed_commands: tuple[FailedCommandFact, ...] = Field(
        default=(), max_length=_MAX_FAILED_COMMANDS
    )
    verification_facts: tuple[VerificationMemoryFact, ...] = Field(
        default=(), max_length=_MAX_VERIFICATION_FACTS
    )

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        has_facts = bool(
            self.plan.revision
            or self.file_changes
            or self.failed_commands
            or self.verification_facts
        )
        if has_facts != (self.revision > 0):
            raise ValueError("run memory revision must distinguish empty and non-empty snapshots")
        path_keys = tuple(_path_key(fact.path) for fact in self.file_changes)
        if len(set(path_keys)) != len(path_keys):
            raise ValueError("run memory file paths must be unique")
        labels = tuple(fact.label for fact in self.verification_facts)
        if len(set(labels)) != len(labels):
            raise ValueError("run memory verification labels must be unique")
        return self

    def canonical_json(self) -> str:
        """Return deterministic JSON for context rendering and persistence tests."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def with_stale_verification(self) -> RunMemorySnapshot:
        """Return a passive-resume view whose historical verifier facts have no authority."""

        if all(fact.stale for fact in self.verification_facts):
            return self
        return RunMemorySnapshot(
            revision=self.revision + 1,
            plan=self.plan,
            file_changes=self.file_changes,
            failed_commands=self.failed_commands,
            verification_facts=tuple(
                fact if fact.stale else fact.model_copy(update={"stale": True})
                for fact in self.verification_facts
            ),
        )


class RunMemory:
    """Observe allowlisted tool facts while keeping raw tool data out of durable memory.

    The shared :class:`PlanState` is the only mutable domain object exposed to tools. File
    changes are coalesced by path, failed commands are a recent FIFO, and verification
    outcomes are historical hints only. Completion logic must continue to use its own fresh
    verification ledger.
    """

    def __init__(
        self,
        *,
        plan_state: PlanState | None = None,
        max_file_changes: int = 32,
        max_failed_commands: int = 8,
        max_verification_facts: int = 16,
        max_chars: int = 8_000,
    ) -> None:
        _validate_limit("max_file_changes", max_file_changes, _MAX_FILE_CHANGES)
        _validate_limit("max_failed_commands", max_failed_commands, _MAX_FAILED_COMMANDS)
        _validate_limit("max_verification_facts", max_verification_facts, _MAX_VERIFICATION_FACTS)
        _validate_limit("max_chars", max_chars, _MAX_RUN_MEMORY_CHARS, _MIN_RUN_MEMORY_CHARS)
        selected_plan = plan_state or PlanState()
        if selected_plan.revision or selected_plan.items:
            raise ValueError("plan_state must be empty; restore persisted state explicitly")
        self._plan_state = selected_plan
        self._observed_plan = selected_plan.snapshot()
        self._max_file_changes = max_file_changes
        self._max_failed_commands = max_failed_commands
        self._max_verification_facts = max_verification_facts
        self._max_chars = max_chars
        self._file_changes: dict[str, FileChangeFact] = {}
        self._failed_commands: list[FailedCommandFact] = []
        self._verification_facts: dict[str, VerificationMemoryFact] = {}
        self._revision = 0

    @property
    def plan_state(self) -> PlanState:
        return self._plan_state

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def max_chars(self) -> int:
        return self._max_chars

    def observe(self, call: ToolCall, execution: ToolExecution, *, step: int) -> None:
        """Consume one matched host execution using only explicit allowlisted fields."""

        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a non-negative integer")
        if call.id != execution.call_id or call.name != execution.tool_name:
            raise ValueError("run memory observations must match their tool call")

        before = self._content_key()
        changed = self._observe_plan(call, execution)
        changed = self._observe_file_change(execution, step=step) or changed
        changed = self._observe_failed_command(call, execution, step=step) or changed
        changed = self._observe_verification(execution, step=step) or changed
        if not changed:
            return
        candidate_revision = self._revision + 1
        self._prune_to_char_budget(candidate_revision)
        if self._content_key() != before:
            self._revision = candidate_revision

    def snapshot(self) -> RunMemorySnapshot:
        """Return a complete immutable fact snapshot without raw tool observations."""

        snapshot = RunMemorySnapshot(
            revision=self._revision,
            plan=self._plan_state.snapshot(),
            file_changes=tuple(self._file_changes.values()),
            failed_commands=tuple(self._failed_commands),
            verification_facts=tuple(self._verification_facts.values()),
        )
        if len(snapshot.canonical_json()) > self._max_chars:  # pragma: no cover - invariant.
            raise RuntimeError("run memory exceeded its configured character budget")
        return snapshot

    def restore(
        self,
        snapshot: RunMemorySnapshot,
        *,
        mark_verification_stale: bool = True,
    ) -> RunMemorySnapshot:
        """Restore into pristine memory; resumed verification is stale by construction."""

        if not self._is_pristine():
            raise RunMemoryError(
                "run_memory_restore_conflict",
                "run memory snapshots can be restored only into pristine memory",
            )
        resumed_snapshot = (
            snapshot.with_stale_verification() if mark_verification_stale else snapshot
        )
        snapshot_chars = len(resumed_snapshot.canonical_json())
        if snapshot_chars > self._max_chars:
            raise RunMemoryError(
                "run_memory_limit_mismatch",
                "snapshot exceeds this runtime's run-memory character budget",
                metadata={"snapshot_chars": snapshot_chars, "max_chars": self._max_chars},
            )
        if len(resumed_snapshot.file_changes) > self._max_file_changes:
            raise RunMemoryError(
                "run_memory_limit_mismatch",
                "snapshot contains more file facts than this runtime permits",
            )
        if len(resumed_snapshot.failed_commands) > self._max_failed_commands:
            raise RunMemoryError(
                "run_memory_limit_mismatch",
                "snapshot contains more command facts than this runtime permits",
            )
        if len(resumed_snapshot.verification_facts) > self._max_verification_facts:
            raise RunMemoryError(
                "run_memory_limit_mismatch",
                "snapshot contains more verification facts than this runtime permits",
            )

        self._plan_state.restore(resumed_snapshot.plan)
        self._observed_plan = resumed_snapshot.plan
        self._file_changes = {_path_key(fact.path): fact for fact in resumed_snapshot.file_changes}
        self._failed_commands = list(resumed_snapshot.failed_commands)
        self._verification_facts = {
            fact.label: fact for fact in resumed_snapshot.verification_facts
        }
        self._revision = resumed_snapshot.revision
        return self.snapshot()

    def mark_verification_stale(self) -> bool:
        """Demote every current verifier fact without restoring verification authority."""

        changed = self._mark_verification_stale()
        if changed:
            candidate_revision = self._revision + 1
            self._prune_to_char_budget(candidate_revision)
            self._revision = candidate_revision
        return changed

    def _observe_plan(self, call: ToolCall, execution: ToolExecution) -> bool:
        if call.name != "update_plan" or not execution.ok:
            return False
        current = self._plan_state.snapshot()
        if current == self._observed_plan:
            return False
        reported = _safe_int(execution.metadata.get("revision"), minimum=1)
        if reported != current.revision:
            return False
        self._observed_plan = current
        return True

    def _observe_file_change(self, execution: ToolExecution, *, step: int) -> bool:
        if execution.tool_name not in {"write_file", "replace_text", "undo_change"}:
            return False
        metadata = execution.metadata
        if not execution.ok or metadata.get("changed") is not True:
            return False
        path = metadata.get("path")
        change_id = metadata.get("change_id")
        change_kind = _safe_change_kind(metadata.get("change_kind"))
        if not isinstance(path, str) or not isinstance(change_id, str):
            return False
        if change_kind is None:
            return False
        before_sha256 = _safe_optional_sha256(metadata.get("before_sha256"))
        after_sha256 = _safe_optional_sha256(metadata.get("after_sha256"))
        if isinstance(before_sha256, _Invalid) or isinstance(after_sha256, _Invalid):
            return False
        added_lines = _safe_int(metadata.get("added_lines"), minimum=0)
        removed_lines = _safe_int(metadata.get("removed_lines"), minimum=0)
        mutation_revision = _safe_int(metadata.get("mutation_revision"), minimum=1)
        if added_lines is None or removed_lines is None or mutation_revision is None:
            return False

        key = _path_key(path)
        previous = self._file_changes.get(key)
        try:
            fact = FileChangeFact(
                path=path,
                change_count=1 if previous is None else previous.change_count + 1,
                last_change_id=change_id,
                last_change_kind=change_kind,
                before_sha256=before_sha256,
                after_sha256=after_sha256,
                added_lines=added_lines,
                removed_lines=removed_lines,
                mutation_revision=mutation_revision,
                last_step=step,
            )
        except ValidationError:
            return False
        self._file_changes.pop(key, None)
        self._file_changes[key] = fact
        while len(self._file_changes) > self._max_file_changes:
            del self._file_changes[next(iter(self._file_changes))]
        return True

    def _observe_failed_command(
        self,
        call: ToolCall,
        execution: ToolExecution,
        *,
        step: int,
    ) -> bool:
        if call.name != "run_command":
            return False
        argv = _safe_argv(call.arguments.get("argv"))
        if argv is None:
            return False
        raw_cwd = execution.metadata.get("cwd") if execution.ok else call.arguments.get("cwd", ".")
        if not isinstance(raw_cwd, str):
            return False

        failure_kind: CommandFailureKind
        exit_code: int | None = None
        error_code: str | None = None
        if not execution.ok:
            failure_kind = "tool_error"
            error_code = _safe_token(execution.error_code or "tool_error")
        else:
            status = execution.metadata.get("status")
            raw_exit_code = execution.metadata.get("exit_code")
            if status == "exited":
                exit_code = _safe_int(raw_exit_code)
                if exit_code in {None, 0}:
                    return False
                failure_kind = "nonzero_exit"
            elif status == "timed_out" and raw_exit_code is None:
                failure_kind = "timed_out"
            elif status == "control_failed" and raw_exit_code is None:
                failure_kind = "control_failed"
            else:
                return False
        try:
            fact = FailedCommandFact(
                argv=argv,
                cwd=raw_cwd,
                failure_kind=failure_kind,
                exit_code=exit_code,
                error_code=error_code,
                output_truncated=execution.truncated,
                step=step,
            )
        except ValidationError:
            return False
        self._failed_commands.append(fact)
        if len(self._failed_commands) > self._max_failed_commands:
            del self._failed_commands[: len(self._failed_commands) - self._max_failed_commands]
        return True

    def _observe_verification(self, execution: ToolExecution, *, step: int) -> bool:
        changed = False
        facts = execution.control
        if facts.invalidates_verification:
            changed = self._mark_verification_stale()
        if facts.verification is None:
            return changed
        assert facts.verification_kind is not None
        assert facts.verification_label is not None
        fact = VerificationMemoryFact(
            label=facts.verification_label,
            kind=facts.verification_kind,
            passed=facts.verification.value == "passed",
            step=step,
            stale=False,
        )
        self._verification_facts.pop(fact.label, None)
        self._verification_facts[fact.label] = fact
        while len(self._verification_facts) > self._max_verification_facts:
            del self._verification_facts[next(iter(self._verification_facts))]
        return True

    def _mark_verification_stale(self) -> bool:
        changed = False
        for label, fact in tuple(self._verification_facts.items()):
            if fact.stale:
                continue
            self._verification_facts[label] = fact.model_copy(update={"stale": True})
            changed = True
        return changed

    def _is_pristine(self) -> bool:
        return not (
            self._revision
            or self._plan_state.revision
            or self._plan_state.items
            or self._file_changes
            or self._failed_commands
            or self._verification_facts
        )

    def _content_key(self) -> tuple[object, ...]:
        return (
            self._observed_plan,
            tuple(self._file_changes.items()),
            tuple(self._failed_commands),
            tuple(self._verification_facts.items()),
        )

    def _prune_to_char_budget(self, revision: int) -> None:
        """Retain high-priority recent facts while keeping canonical JSON bounded."""

        plan = self._plan_state.snapshot()
        selected_verification: list[VerificationMemoryFact] = []
        selected_files: list[FileChangeFact] = []
        selected_commands: list[FailedCommandFact] = []

        if not self._facts_fit(
            revision,
            plan=plan,
            file_changes=(),
            failed_commands=(),
            verification_facts=(),
        ):
            raise RunMemoryError(
                "run_memory_budget_too_small",
                "the explicit plan cannot fit the configured run-memory budget",
                metadata={"max_chars": self._max_chars},
            )

        # The explicit plan is mandatory. Historical verifier facts are small and
        # explain freshness, followed by changed paths and potentially-large commands.
        for verification_fact in reversed(tuple(self._verification_facts.values())):
            verification_candidate = [verification_fact, *selected_verification]
            if self._facts_fit(
                revision,
                plan=plan,
                file_changes=tuple(selected_files),
                failed_commands=tuple(selected_commands),
                verification_facts=tuple(verification_candidate),
            ):
                selected_verification = verification_candidate
        for file_fact in reversed(tuple(self._file_changes.values())):
            file_candidate = [file_fact, *selected_files]
            if self._facts_fit(
                revision,
                plan=plan,
                file_changes=tuple(file_candidate),
                failed_commands=tuple(selected_commands),
                verification_facts=tuple(selected_verification),
            ):
                selected_files = file_candidate
        for command_fact in reversed(tuple(self._failed_commands)):
            command_candidate = [command_fact, *selected_commands]
            if self._facts_fit(
                revision,
                plan=plan,
                file_changes=tuple(selected_files),
                failed_commands=tuple(command_candidate),
                verification_facts=tuple(selected_verification),
            ):
                selected_commands = command_candidate

        self._verification_facts = {fact.label: fact for fact in selected_verification}
        self._file_changes = {_path_key(fact.path): fact for fact in selected_files}
        self._failed_commands = selected_commands

    def _facts_fit(
        self,
        revision: int,
        *,
        plan: PlanSnapshot,
        file_changes: tuple[FileChangeFact, ...],
        failed_commands: tuple[FailedCommandFact, ...],
        verification_facts: tuple[VerificationMemoryFact, ...],
    ) -> bool:
        has_facts = bool(plan.revision or file_changes or failed_commands or verification_facts)
        candidate = RunMemorySnapshot(
            revision=revision if has_facts else 0,
            plan=plan,
            file_changes=file_changes,
            failed_commands=failed_commands,
            verification_facts=verification_facts,
        )
        return len(candidate.canonical_json()) <= self._max_chars


class _Invalid:
    pass


_INVALID = _Invalid()


def _safe_optional_sha256(value: object) -> str | None | _Invalid:
    if value is None:
        return None
    if isinstance(value, str) and _is_sha256(value):
        return value
    return _INVALID


def _safe_change_kind(value: object) -> ChangeKind | None:
    if value == "create":
        return "create"
    if value == "update":
        return "update"
    if value == "undo":
        return "undo"
    return None


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _safe_int(value: object, *, minimum: int | None = None) -> int | None:
    if type(value) is not int:
        return None
    if minimum is not None and value < minimum:
        return None
    return value


def _safe_argv(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, list | tuple) or not 1 <= len(value) <= _MAX_COMMAND_ARGUMENTS:
        return None
    if not all(isinstance(argument, str) and argument for argument in value):
        return None
    argv = tuple(value)
    if sum(len(argument) for argument in argv) > _MAX_COMMAND_ARGUMENT_CHARS:
        return None
    if any(_has_control(argument) for argument in argv):
        return None
    return argv


def _safe_token(value: str) -> str:
    if 1 <= len(value) <= 80 and all(
        character.isascii() and (character.isalnum() or character in "._-") for character in value
    ):
        return value
    digest = sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()[:12]
    return f"sha256-{digest}"


def _require_printable(value: str, *, field: str) -> None:
    if _has_control(value):
        raise ValueError(f"{field} must contain printable single-line text")


def _has_control(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _path_key(path: str) -> str:
    normalized = os.path.normcase(path)
    if os.name == "nt":
        normalized = normalized.casefold()
    return normalized.replace("\\", "/")


def _validate_limit(name: str, value: int, maximum: int, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
