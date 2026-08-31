"""Portable workspace path, ignore, and sensitive-file policy."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol
from unicodedata import category

from pathspec import GitIgnoreSpec

from coding_agent._workspace.contracts import IgnoreSignature, WorkspaceError
from coding_agent._workspace.native import RESERVED_TEMP_PREFIX, is_link_like

_HARD_EXCLUDED_PARTS = frozenset({".git", ".coding-agent"})
_HARD_EXCLUDED_PREFIXES = (RESERVED_TEMP_PREFIX,)
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
_SENSITIVE_FILENAMES = frozenset({"id_dsa", "id_ecdsa", "id_ed25519", "id_rsa"})
_SENSITIVE_SUFFIXES = frozenset({".key", ".p12", ".pfx", ".pem"})
_MAX_GITIGNORE_BYTES = 256_000
_MAX_GITIGNORE_PATTERNS = 10_000
_MAX_GITIGNORE_LINE_CHARS = 8_000


class ApprovedFileReader(Protocol):
    """Secure reader callback used while loading repository policy files."""

    def __call__(
        self,
        path: Path,
        *,
        max_bytes: int,
        display_path: str,
        too_large_code: str,
    ) -> bytes: ...


class WorkspacePolicy:
    """Own path normalization and all repository visibility rules for one root."""

    def __init__(self, root: Path, read_approved_file: ApprovedFileReader) -> None:
        self._root = root
        self._read_approved_file = read_approved_file
        self._ignore_cache: dict[str, tuple[IgnoreSignature, GitIgnoreSpec]] = {}

    def initialize(self) -> None:
        """Validate and cache the root ignore policy after the owner is fully constructed."""
        self._ignore_spec_for(PurePosixPath("."))

    def contains(self, path: Path) -> bool:
        """Resolve a candidate and report whether its physical path is inside the root."""
        try:
            resolved = path.resolve(strict=False)
            resolved.relative_to(self._root)
        except (OSError, RuntimeError, ValueError):
            return False
        return True

    def normalize_user_path(self, user_path: str) -> PurePosixPath:
        if (
            not user_path
            or user_path.isspace()
            or any(_is_display_control(character) for character in user_path)
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

    def contains_directory_link(self, relative: PurePosixPath) -> bool:
        current = self._root
        for part in relative.parts:
            current /= part
            if is_link_like(current) and current.is_dir():
                return True
        return False

    def enforce(
        self,
        relative: PurePosixPath,
        resolved: Path,
        *,
        allow_ignored: bool,
        allow_sensitive: bool,
    ) -> None:
        resolved_relative = self.resolved_relative(resolved)
        is_directory = resolved.is_dir()
        if self._is_hard_excluded(relative) or self._is_hard_excluded(resolved_relative):
            raise WorkspaceError("path_ignored", f"path is ignored: {relative.as_posix()}")
        if not allow_ignored and (
            self._matches_gitignore(relative, is_directory=is_directory)
            or self._matches_gitignore(resolved_relative, is_directory=is_directory)
        ):
            raise WorkspaceError("path_ignored", f"path is ignored: {relative.as_posix()}")
        if not allow_sensitive and (
            self.is_sensitive(relative) or self.is_sensitive(resolved_relative)
        ):
            raise WorkspaceError(
                "sensitive_path", f"sensitive files cannot be read: {relative.as_posix()}"
            )

    def enforce_mutation(
        self,
        relative: PurePosixPath,
        *,
        is_directory: bool = False,
    ) -> None:
        """Apply mutation-only policy without resolving a potentially missing leaf."""
        if relative.name.casefold() == ".gitignore":
            raise WorkspaceError("path_ignored", "repository ignore-policy files cannot be changed")
        if self._is_hard_excluded(relative):
            raise WorkspaceError("path_ignored", f"path is ignored: {relative.as_posix()}")
        if self.is_sensitive(relative):
            raise WorkspaceError(
                "sensitive_path", f"sensitive files cannot be changed: {relative.as_posix()}"
            )
        if self._matches_gitignore(relative, is_directory=is_directory):
            raise WorkspaceError("path_ignored", f"path is ignored: {relative.as_posix()}")

    def is_excluded(
        self,
        relative: PurePosixPath,
        resolved: Path,
        *,
        is_directory: bool,
    ) -> bool:
        resolved_relative = self.resolved_relative(resolved)
        return (
            self._is_hard_excluded(relative)
            or self._is_hard_excluded(resolved_relative)
            or self._matches_gitignore(relative, is_directory=is_directory)
            or self._matches_gitignore(resolved_relative, is_directory=is_directory)
        )

    @staticmethod
    def is_sensitive(relative: PurePosixPath) -> bool:
        for index, part in enumerate(relative.parts):
            lowered = part.casefold()
            is_example_file = lowered == ".env.example" and index == len(relative.parts) - 1
            if lowered == ".env" or (lowered.startswith(".env.") and not is_example_file):
                return True
            if lowered in _SENSITIVE_FILENAMES or Path(lowered).suffix in _SENSITIVE_SUFFIXES:
                return True
        return False

    def resolved_relative(self, resolved: Path) -> PurePosixPath:
        return PurePosixPath(resolved.relative_to(self._root).as_posix())

    @staticmethod
    def join_relative(parent: str, child: str) -> str:
        if parent == ".":
            return PurePosixPath(child).as_posix()
        return (PurePosixPath(parent) / child).as_posix()

    @staticmethod
    def _is_hard_excluded(relative: PurePosixPath) -> bool:
        return any(
            part.casefold() in _HARD_EXCLUDED_PARTS
            or any(
                part.casefold().startswith(prefix.casefold()) for prefix in _HARD_EXCLUDED_PREFIXES
            )
            for part in relative.parts
        )

    def _matches_gitignore(self, relative: PurePosixPath, *, is_directory: bool) -> bool:
        if relative == PurePosixPath("."):
            return False

        inherited_specs: list[tuple[PurePosixPath, GitIgnoreSpec]] = []
        ignored = False
        for directory in self._ignore_directories(relative):
            if directory != PurePosixPath(".") and self._matches_ignore_specs(
                directory,
                is_directory=True,
                specs=inherited_specs,
            ):
                # Git never descends into an excluded parent, so rules inside that directory
                # cannot re-include one of its children.
                return True
            spec = self._ignore_spec_for(directory)
            inherited_specs.append((directory, spec))
            candidate = relative.relative_to(directory).as_posix()
            if is_directory:
                candidate += "/"
            if os.name == "nt":
                candidate = candidate.casefold()
            result = spec.check_file(candidate)
            if result.include is not None:
                ignored = result.include
        return ignored

    @staticmethod
    def _matches_ignore_specs(
        relative: PurePosixPath,
        *,
        is_directory: bool,
        specs: list[tuple[PurePosixPath, GitIgnoreSpec]],
    ) -> bool:
        ignored = False
        for base, spec in specs:
            candidate = relative.relative_to(base).as_posix()
            if is_directory:
                candidate += "/"
            if os.name == "nt":
                candidate = candidate.casefold()
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
            if is_link_like(ignore_file):
                raise WorkspaceError(
                    "invalid_workspace", ".gitignore files must not be filesystem links"
                )
            if not ignore_file.is_file():
                return GitIgnoreSpec.from_lines((), backend="simple")
            resolved_ignore_file = ignore_file.resolve(strict=True)
            if not self.contains(resolved_ignore_file):
                raise WorkspaceError("invalid_workspace", ".gitignore resolves outside workspace")
            data = self._read_approved_file(
                resolved_ignore_file,
                max_bytes=_MAX_GITIGNORE_BYTES,
                display_path=".gitignore",
                too_large_code="invalid_workspace",
            )
            if b"\x00" in data:
                raise WorkspaceError("invalid_workspace", ".gitignore contains binary data")
            lines = data.decode("utf-8-sig").splitlines()
            if len(lines) > _MAX_GITIGNORE_PATTERNS or any(
                len(line) > _MAX_GITIGNORE_LINE_CHARS for line in lines
            ):
                raise WorkspaceError(
                    "invalid_workspace", ".gitignore exceeds the pattern complexity limit"
                )
            policy_lines = [line.casefold() for line in lines] if os.name == "nt" else lines
            return GitIgnoreSpec.from_lines(policy_lines, backend="simple")
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


def _is_display_control(character: str) -> bool:
    return category(character) in {"Cc", "Cf", "Zl", "Zp"}
