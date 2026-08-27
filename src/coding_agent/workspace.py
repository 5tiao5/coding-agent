"""Rooted filesystem boundary shared by every local repository tool."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from pathspec import GitIgnoreSpec

from coding_agent.errors import CodedError

ExpectedPathKind = Literal["file", "directory", "any"]
IgnoreSignature = tuple[int, int, int, int, int] | None

_HARD_EXCLUDED_PARTS = frozenset({".git", ".coding-agent"})
_WINDOWS_RESERVED_NAMES = frozenset(
    {
        "aux",
        "clock$",
        "con",
        "conin$",
        "conout$",
        "nul",
        "prn",
        *(f"com{number}" for number in range(1, 10)),
        *(f"com{number}" for number in "¹²³"),
        *(f"lpt{number}" for number in range(1, 10)),
        *(f"lpt{number}" for number in "¹²³"),
    }
)
_SENSITIVE_FILENAMES = frozenset(
    {
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
    }
)
_SENSITIVE_SUFFIXES = frozenset({".key", ".p12", ".pfx", ".pem"})


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


class Workspace:
    """Resolve and enumerate paths without allowing access outside one root."""

    def __init__(self, root: str | Path) -> None:
        try:
            resolved_root = Path(root).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise WorkspaceError("invalid_workspace", "workspace root is not accessible") from exc
        if not resolved_root.is_dir():
            raise WorkspaceError("invalid_workspace", "workspace root must be a directory")
        if any(part.casefold() in _HARD_EXCLUDED_PARTS for part in resolved_root.parts):
            raise WorkspaceError("invalid_workspace", "workspace root is inside excluded metadata")

        self._root = resolved_root
        self._ignore_cache: dict[str, tuple[IgnoreSignature, GitIgnoreSpec]] = {}
        self._ignore_spec_for(PurePosixPath("."))

    @property
    def root(self) -> Path:
        return self._root

    def contains(self, path: Path) -> bool:
        """Resolve a candidate and report whether its physical path is inside the root."""
        try:
            resolved = path.resolve(strict=False)
            resolved.relative_to(self._root)
        except (OSError, RuntimeError, ValueError):
            return False
        return True

    def resolve(
        self,
        user_path: str,
        *,
        expected: ExpectedPathKind = "any",
        allow_ignored: bool = False,
        allow_sensitive: bool = False,
    ) -> WorkspacePath:
        """Resolve an existing relative path and enforce all workspace policies."""
        if expected not in {"file", "directory", "any"}:
            raise ValueError(f"unsupported expected path kind: {expected}")
        relative = self._normalize_user_path(user_path)
        lexical = self._root.joinpath(*relative.parts)

        try:
            resolved = lexical.resolve(strict=True)
        except FileNotFoundError as exc:
            raise WorkspaceError("not_found", f"path not found: {relative.as_posix()}") from exc
        except PermissionError as exc:
            raise WorkspaceError(
                "permission_denied", f"permission denied: {relative.as_posix()}"
            ) from exc
        except (OSError, RuntimeError) as exc:
            raise WorkspaceError("io_error", f"cannot resolve path: {relative.as_posix()}") from exc

        if not self.contains(resolved):
            raise WorkspaceError(
                "path_outside_workspace",
                f"path resolves outside the workspace: {relative.as_posix()}",
            )

        if self._contains_directory_link(relative):
            raise WorkspaceError(
                "invalid_path",
                f"directory links cannot be traversed: {relative.as_posix()}",
            )

        self._enforce_policy(
            relative,
            resolved,
            allow_ignored=allow_ignored,
            allow_sensitive=allow_sensitive,
        )

        if expected == "file" and not resolved.is_file():
            raise WorkspaceError("not_file", f"path is not a file: {relative.as_posix()}")
        if expected == "directory" and not resolved.is_dir():
            raise WorkspaceError("not_directory", f"path is not a directory: {relative.as_posix()}")

        return WorkspacePath(path=resolved, relative=relative.as_posix())

    def children(self, directory: WorkspacePath) -> tuple[WorkspaceEntry, ...]:
        """Return safe direct children in deterministic order without following directory links."""
        try:
            resolved_directory = directory.path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise WorkspaceError("io_error", "directory handle is not accessible") from exc
        if resolved_directory != directory.path or not self.contains(resolved_directory):
            raise WorkspaceError("path_outside_workspace", "directory handle is outside workspace")
        if not resolved_directory.is_dir():
            raise WorkspaceError("not_directory", "directory handle does not refer to a directory")
        try:
            rebound = self.resolve(directory.relative, expected="directory")
        except WorkspaceError as exc:
            raise WorkspaceError(
                "invalid_path", "directory handle does not match its relative path"
            ) from exc
        if rebound.path != resolved_directory:
            raise WorkspaceError(
                "invalid_path", "directory handle does not match its relative path"
            )

        try:
            with os.scandir(resolved_directory) as scan:
                raw_entries = sorted(scan, key=lambda entry: (entry.name.casefold(), entry.name))
        except PermissionError as exc:
            raise WorkspaceError(
                "permission_denied", f"permission denied: {directory.relative}"
            ) from exc
        except OSError as exc:
            raise WorkspaceError(
                "io_error", f"cannot list directory: {directory.relative}"
            ) from exc

        safe_entries: list[WorkspaceEntry] = []
        for raw_entry in raw_entries:
            relative = self._join_relative(directory.relative, raw_entry.name)
            try:
                normalized_relative = self._normalize_user_path(relative)
            except WorkspaceError:
                continue
            if normalized_relative.as_posix() != relative:
                continue
            lexical = Path(raw_entry.path)
            link_like = self._is_link_like(lexical)

            try:
                resolved = lexical.resolve(strict=True)
                is_directory = resolved.is_dir()
                is_file = resolved.is_file()
            except (OSError, RuntimeError):
                continue

            if not self.contains(resolved):
                continue
            if link_like and is_directory:
                continue
            if not (is_directory or is_file):
                continue

            relative_path = PurePosixPath(relative)
            if self._is_excluded(relative_path, resolved, is_directory=is_directory):
                continue
            resolved_relative = self._resolved_relative(resolved)
            if self._is_sensitive(relative_path) or self._is_sensitive(resolved_relative):
                continue

            safe_entries.append(
                WorkspaceEntry(
                    path=resolved,
                    relative=relative,
                    is_directory=is_directory,
                    is_symlink=link_like,
                )
            )

        return tuple(safe_entries)

    def _ignore_spec_for(self, directory: PurePosixPath) -> GitIgnoreSpec:
        cache_key = directory.as_posix()
        ignore_file = self._root.joinpath(*directory.parts) / ".gitignore"
        signature = self._ignore_signature(ignore_file)
        cached = self._ignore_cache.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[1]

        spec = self._load_gitignore(ignore_file)
        self._ignore_cache[cache_key] = (signature, spec)
        return spec

    def _load_gitignore(self, ignore_file: Path) -> GitIgnoreSpec:
        try:
            if self._is_link_like(ignore_file):
                raise WorkspaceError(
                    "invalid_workspace", ".gitignore files must not be filesystem links"
                )
            if not ignore_file.is_file():
                return GitIgnoreSpec.from_lines((), backend="simple")
            resolved_ignore_file = ignore_file.resolve(strict=True)
            if not self.contains(resolved_ignore_file):
                raise WorkspaceError("invalid_workspace", ".gitignore resolves outside workspace")
            lines = resolved_ignore_file.read_text(encoding="utf-8-sig").splitlines()
            return GitIgnoreSpec.from_lines(lines, backend="simple")
        except WorkspaceError:
            raise
        except (OSError, UnicodeError, ValueError) as exc:
            raise WorkspaceError(
                "invalid_workspace", ".gitignore could not be parsed as UTF-8"
            ) from exc

    @staticmethod
    def _ignore_signature(ignore_file: Path) -> IgnoreSignature:
        try:
            file_stat = ignore_file.lstat()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise WorkspaceError(
                "invalid_workspace", ".gitignore metadata is inaccessible"
            ) from exc
        return (
            file_stat.st_mode,
            file_stat.st_ino,
            file_stat.st_size,
            file_stat.st_mtime_ns,
            file_stat.st_ctime_ns,
        )

    def _normalize_user_path(self, user_path: str) -> PurePosixPath:
        if (
            not user_path
            or user_path.isspace()
            or any(ord(character) < 32 or ord(character) == 127 for character in user_path)
        ):
            raise WorkspaceError("invalid_path", "path must be a non-empty relative path")

        windows_path = PureWindowsPath(user_path)
        portable = user_path.replace("\\", "/")
        posix_path = PurePosixPath(portable)
        if windows_path.drive or windows_path.root or posix_path.is_absolute():
            raise WorkspaceError(
                "invalid_path", "absolute and drive-qualified paths are not allowed"
            )

        normalized_parts: list[str] = []
        for part in portable.split("/"):
            if not part or part == ".":
                continue
            if part == "..":
                raise WorkspaceError("invalid_path", "parent path segments are not allowed")
            if ":" in part:
                raise WorkspaceError("invalid_path", "colon characters are not allowed in paths")
            if any(character in '<>"|?*' for character in part):
                raise WorkspaceError(
                    "invalid_path", "portable paths cannot contain shell wildcards"
                )
            if part.endswith((" ", ".")):
                raise WorkspaceError(
                    "invalid_path", "path components cannot end with a space or dot"
                )
            reserved_candidate = part.split(".", maxsplit=1)[0].casefold()
            if reserved_candidate in _WINDOWS_RESERVED_NAMES:
                raise WorkspaceError("invalid_path", "reserved device names are not allowed")
            normalized_parts.append(part)

        if not normalized_parts:
            return PurePosixPath(".")
        return PurePosixPath(*normalized_parts)

    def _contains_directory_link(self, relative: PurePosixPath) -> bool:
        current = self._root
        for part in relative.parts:
            current /= part
            if self._is_link_like(current) and current.is_dir():
                return True
        return False

    def _enforce_policy(
        self,
        relative: PurePosixPath,
        resolved: Path,
        *,
        allow_ignored: bool,
        allow_sensitive: bool,
    ) -> None:
        resolved_relative = self._resolved_relative(resolved)
        is_directory = resolved.is_dir()
        if self._is_hard_excluded(relative) or self._is_hard_excluded(resolved_relative):
            raise WorkspaceError("path_ignored", f"path is ignored: {relative.as_posix()}")
        if not allow_ignored and (
            self._matches_gitignore(relative, is_directory=is_directory)
            or self._matches_gitignore(resolved_relative, is_directory=is_directory)
        ):
            raise WorkspaceError("path_ignored", f"path is ignored: {relative.as_posix()}")
        if not allow_sensitive and (
            self._is_sensitive(relative) or self._is_sensitive(resolved_relative)
        ):
            raise WorkspaceError(
                "sensitive_path", f"sensitive files cannot be read: {relative.as_posix()}"
            )

    def _is_excluded(
        self,
        relative: PurePosixPath,
        resolved: Path,
        *,
        is_directory: bool,
    ) -> bool:
        resolved_relative = self._resolved_relative(resolved)
        return (
            self._is_hard_excluded(relative)
            or self._is_hard_excluded(resolved_relative)
            or self._matches_gitignore(relative, is_directory=is_directory)
            or self._matches_gitignore(resolved_relative, is_directory=is_directory)
        )

    @staticmethod
    def _is_hard_excluded(relative: PurePosixPath) -> bool:
        return any(part.casefold() in _HARD_EXCLUDED_PARTS for part in relative.parts)

    @staticmethod
    def _is_sensitive(relative: PurePosixPath) -> bool:
        for index, part in enumerate(relative.parts):
            lowered = part.casefold()
            is_example_file = lowered == ".env.example" and index == len(relative.parts) - 1
            if lowered == ".env" or (lowered.startswith(".env.") and not is_example_file):
                return True
            if lowered in _SENSITIVE_FILENAMES or Path(lowered).suffix in _SENSITIVE_SUFFIXES:
                return True
        return False

    def _matches_gitignore(self, relative: PurePosixPath, *, is_directory: bool) -> bool:
        if relative == PurePosixPath("."):
            return False

        ignored = False
        for directory in self._ignore_directories(relative):
            spec = self._ignore_spec_for(directory)
            candidate = relative.relative_to(directory).as_posix()
            if is_directory:
                candidate += "/"
            result = spec.check_file(candidate)
            if result.include is not None:
                ignored = result.include
        return ignored

    @staticmethod
    def _ignore_directories(relative: PurePosixPath) -> tuple[PurePosixPath, ...]:
        directories = [PurePosixPath(".")]
        current = PurePosixPath(".")
        for part in relative.parent.parts:
            current /= part
            directories.append(current)
        return tuple(directories)

    def _resolved_relative(self, resolved: Path) -> PurePosixPath:
        return PurePosixPath(resolved.relative_to(self._root).as_posix())

    @staticmethod
    def _join_relative(parent: str, child: str) -> str:
        if parent == ".":
            return PurePosixPath(child).as_posix()
        return (PurePosixPath(parent) / child).as_posix()

    @staticmethod
    def _is_link_like(path: Path) -> bool:
        if path.is_symlink():
            return True
        if os.name != "nt":
            return False
        try:
            attributes = getattr(path.lstat(), "st_file_attributes", 0)
        except OSError:
            return False
        return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
