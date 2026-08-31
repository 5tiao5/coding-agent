"""Stable value objects shared by workspace implementation modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from coding_agent.errors import CodedError

ExpectedPathKind = Literal["file", "directory", "any"]
FileIdentity = tuple[int, int]
IgnoreSignature = tuple[int, int, int, int, int] | None


class WorkspaceError(CodedError):
    """Expected workspace-policy failure with a stable model-facing code."""


@dataclass(frozen=True, slots=True)
class WorkspacePath:
    """An allowed path with a physical target and a stable relative display name."""

    path: Path
    relative: str


@dataclass(frozen=True, slots=True)
class WorkspaceEntry:
    """One safe child returned by a non-recursive workspace scan."""

    path: Path
    relative: str
    is_directory: bool
    is_symlink: bool


@dataclass(frozen=True, slots=True)
class DirectoryScan:
    """A bounded directory scan with explicit incompleteness metadata."""

    entries: tuple[WorkspaceEntry, ...]
    examined: int
    skipped: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class DirectoryReceipt:
    """The stable outcome of one no-clobber directory creation request."""

    relative: str
    created: bool


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    """An authorized file version used as a compare-and-swap write precondition."""

    relative: str
    data: bytes | None
    sha256: str | None
    mode: int | None
    identity: FileIdentity | None = None


@dataclass(frozen=True, slots=True)
class WriteReceipt:
    """The stable outcome of one atomic workspace write."""

    relative: str
    before_sha256: str | None
    after_sha256: str
    bytes_written: int
    created: bool
    durability_uncertain: bool = False
    after_identity: FileIdentity | None = None
