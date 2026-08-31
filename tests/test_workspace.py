"""Security and portability tests for the rooted workspace boundary."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from coding_agent.workspace import Workspace, WorkspaceError, WorkspacePath


def _write(path: Path, text: str = "content") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    _write(
        root / ".gitignore",
        "*.log\n/build/\ngenerated/*\n!generated/keep.py\n",
    )
    _write(root / "src" / "南京.py")
    _write(root / "generated" / "drop.py")
    _write(root / "generated" / "keep.py")
    _write(root / "build" / "output.py")
    _write(root / "ignored.log")
    _write(root / ".git" / "config")
    _write(root / ".env", "TOKEN=secret")
    _write(root / ".env.example", "TOKEN=replace-me")
    return root


def test_protected_mutation_paths_remain_readable_but_reject_exact_and_descendant_writes(
    workspace_root: Path,
) -> None:
    _write(workspace_root / "tests" / "test_public.py", "def test_public(): pass\n")
    _write(workspace_root / "pyproject.toml", "[tool.pytest.ini_options]\n")
    workspace = Workspace(
        workspace_root,
        protected_mutation_paths=("tests/", "pyproject.toml"),
    )

    public_test = workspace.resolve("tests/test_public.py", expected="file")
    assert workspace.read_bytes(public_test, max_bytes=1_000).startswith(b"def test_public")
    with pytest.raises(WorkspaceError, match="trusted verification paths") as descendant:
        workspace.snapshot_for_write("tests/test_public.py", max_bytes=1_000)
    with pytest.raises(WorkspaceError, match="trusted verification paths") as exact:
        workspace.snapshot_for_write("pyproject.toml", max_bytes=1_000)

    assert descendant.value.code == "protected_path"
    assert exact.value.code == "protected_path"


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../secret",
        "a/../../secret",
        "/etc/passwd",
        r"C:\Windows\system.ini",
        r"C:relative",
        r"\Windows",
        r"\\server\share\secret",
        r"\\?\C:\secret",
        "bad\x00path",
        "line\nbreak",
        "bidi\u202efile",
        "separator\u2028file",
        "file.txt:stream",
        "CON",
        "CONIN$",
        "CONOUT$",
        "COM¹.txt",
        "LPT²",
        "folder/NUL.txt",
        "bad?.txt",
        "bad|name",
        "trailing.",
        "trailing ",
    ],
)
def test_portable_path_validation_rejects_ambiguous_or_escaping_paths(
    workspace_root: Path,
    unsafe_path: str,
) -> None:
    workspace = Workspace(workspace_root)

    with pytest.raises(WorkspaceError) as captured:
        workspace.resolve(unsafe_path)

    assert captured.value.code == "invalid_path"


def test_resolve_accepts_unicode_spaces_and_mixed_separators(workspace_root: Path) -> None:
    _write(workspace_root / "src" / "目录 名" / "功能.py")
    workspace = Workspace(workspace_root)

    resolved = workspace.resolve(r"src\目录 名/功能.py", expected="file")

    assert resolved.path == (workspace_root / "src" / "目录 名" / "功能.py").resolve()
    assert resolved.relative == "src/目录 名/功能.py"


def test_resolve_returns_stable_errors_without_absolute_path_leaks(workspace_root: Path) -> None:
    workspace = Workspace(workspace_root)

    with pytest.raises(WorkspaceError) as missing:
        workspace.resolve("missing.py")
    with pytest.raises(WorkspaceError) as not_file:
        workspace.resolve("src", expected="file")
    with pytest.raises(WorkspaceError) as not_directory:
        workspace.resolve("src/南京.py", expected="directory")

    assert missing.value.code == "not_found"
    assert not_file.value.code == "not_file"
    assert not_directory.value.code == "not_directory"
    for error in (missing.value, not_file.value, not_directory.value):
        assert str(workspace_root) not in error.message


def test_resolve_rejects_an_unknown_expected_kind(workspace_root: Path) -> None:
    workspace = Workspace(workspace_root)

    with pytest.raises(ValueError, match="unsupported expected path kind"):
        workspace.resolve(".", expected="bogus")  # type: ignore[arg-type]


def test_root_gitignore_negation_and_hard_exclusions_are_enforced(
    workspace_root: Path,
) -> None:
    workspace = Workspace(workspace_root)

    for ignored in ("ignored.log", "build", "generated/drop.py", ".git/config"):
        with pytest.raises(WorkspaceError) as captured:
            workspace.resolve(ignored)
        assert captured.value.code == "path_ignored"

    assert workspace.resolve("generated/keep.py", expected="file").relative == ("generated/keep.py")
    assert workspace.resolve(".gitignore", expected="file").relative == ".gitignore"


def test_anchored_gitignore_directory_does_not_match_nested_names(workspace_root: Path) -> None:
    _write(workspace_root / "src" / "build" / "keep.py")

    resolved = Workspace(workspace_root).resolve("src/build/keep.py", expected="file")

    assert resolved.relative == "src/build/keep.py"


def test_nested_gitignore_overrides_parent_rules(workspace_root: Path) -> None:
    _write(workspace_root / "nested" / ".gitignore", "*.tmp\n!keep.tmp\n")
    _write(workspace_root / "nested" / "drop.tmp")
    _write(workspace_root / "nested" / "keep.tmp")
    workspace = Workspace(workspace_root)

    entries = workspace.children(workspace.resolve("nested", expected="directory")).entries

    assert [entry.relative for entry in entries] == [
        "nested/.gitignore",
        "nested/keep.tmp",
    ]


def test_ignored_parent_cannot_be_reincluded_by_its_nested_gitignore(tmp_path: Path) -> None:
    root = tmp_path / "ignored-parent"
    root.mkdir()
    _write(root / ".gitignore", "hidden/\n")
    _write(root / "hidden" / ".gitignore", "!secret.txt\n")
    _write(root / "hidden" / "secret.txt", "must remain ignored")
    workspace = Workspace(root)

    with pytest.raises(WorkspaceError) as captured:
        workspace.resolve("hidden/secret.txt", expected="file")

    assert captured.value.code == "path_ignored"
    entries = workspace.children(workspace.resolve(".", expected="directory")).entries
    assert "hidden" not in [entry.relative for entry in entries]


def test_gitignore_cache_refreshes_when_rules_change(workspace_root: Path) -> None:
    _write(workspace_root / "later.ignore")
    workspace = Workspace(workspace_root)
    assert workspace.resolve("later.ignore", expected="file").relative == "later.ignore"

    with (workspace_root / ".gitignore").open("a", encoding="utf-8") as ignore_file:
        ignore_file.write("later.ignore\n")

    with pytest.raises(WorkspaceError) as captured:
        workspace.resolve("later.ignore", expected="file")

    assert captured.value.code == "path_ignored"


def test_allow_ignored_never_bypasses_hard_metadata_exclusions(workspace_root: Path) -> None:
    workspace = Workspace(workspace_root)

    with pytest.raises(WorkspaceError) as captured:
        workspace.resolve(".git/config", expected="file", allow_ignored=True)

    assert captured.value.code == "path_ignored"


def test_sensitive_files_are_separate_from_gitignore_policy(workspace_root: Path) -> None:
    workspace = Workspace(workspace_root)

    with pytest.raises(WorkspaceError) as captured:
        workspace.resolve(".env", expected="file", allow_ignored=True)

    assert captured.value.code == "sensitive_path"
    assert workspace.resolve(".env.example", expected="file").relative == ".env.example"


@pytest.mark.parametrize(
    "sensitive_path",
    [
        ".env.local",
        ".ENV.PRODUCTION",
        "nested/.env.example/secret.txt",
        "id_rsa",
        "private.PEM",
        "certificate.p12",
        "identity.pfx",
        "signing.key",
    ],
)
def test_sensitive_name_variants_are_blocked(
    workspace_root: Path,
    sensitive_path: str,
) -> None:
    _write(workspace_root / Path(sensitive_path))

    with pytest.raises(WorkspaceError) as captured:
        Workspace(workspace_root).resolve(
            sensitive_path,
            expected="file",
            allow_ignored=True,
        )

    assert captured.value.code == "sensitive_path"


def test_symlink_alias_cannot_hide_a_sensitive_target(workspace_root: Path) -> None:
    target = workspace_root / "private.pem"
    alias = workspace_root / "harmless.txt"
    _write(target, "private material")
    try:
        alias.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    with pytest.raises(WorkspaceError) as captured:
        Workspace(workspace_root).resolve("harmless.txt", expected="file")

    assert captured.value.code == "sensitive_path"


def test_children_are_sorted_filtered_and_never_expose_absolute_names(
    workspace_root: Path,
) -> None:
    workspace = Workspace(workspace_root)

    entries = workspace.children(workspace.resolve(".", expected="directory")).entries

    assert [entry.relative for entry in entries] == [
        ".env.example",
        ".gitignore",
        "generated",
        "src",
    ]
    assert all(not Path(entry.relative).is_absolute() for entry in entries)
    assert all(str(workspace_root) not in entry.relative for entry in entries)


def test_contains_rejects_an_already_resolved_outside_path(
    workspace_root: Path,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.txt"
    _write(outside)

    assert Workspace(workspace_root).contains(outside.resolve()) is False
    assert Workspace(workspace_root).contains(workspace_root / ".." / "outside.txt") is False


def test_children_revalidates_its_workspace_path_capability(
    workspace_root: Path,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    forged = WorkspacePath(path=outside.resolve(), relative=".")

    with pytest.raises(WorkspaceError) as captured:
        Workspace(workspace_root).children(forged)

    assert captured.value.code == "path_outside_workspace"


def test_children_rejects_a_forged_relative_display_name(workspace_root: Path) -> None:
    workspace = Workspace(workspace_root)
    forged = WorkspacePath(path=workspace.root, relative="../../spoof")

    with pytest.raises(WorkspaceError) as captured:
        workspace.children(forged)

    assert captured.value.code == "invalid_path"


@pytest.mark.skipif(os.name != "nt", reason="Windows filesystems use case-insensitive aliases")
def test_children_accepts_a_case_alias_that_resolves_to_the_same_directory(
    workspace_root: Path,
) -> None:
    workspace = Workspace(workspace_root)
    aliased = workspace.resolve("SRC", expected="directory")

    entries = workspace.children(aliased).entries

    assert [entry.relative for entry in entries] == ["SRC/南京.py"]


def test_workspace_root_cannot_be_git_metadata(workspace_root: Path) -> None:
    with pytest.raises(WorkspaceError) as captured:
        Workspace(workspace_root / ".git")

    assert captured.value.code == "invalid_workspace"


def test_root_gitignore_cannot_be_a_link_outside_workspace(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside_ignore = tmp_path / "outside.gitignore"
    _write(outside_ignore, "secret.txt\n")
    try:
        (root / ".gitignore").symlink_to(outside_ignore)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    with pytest.raises(WorkspaceError) as captured:
        Workspace(root)

    assert captured.value.code == "invalid_workspace"


def test_root_gitignore_cannot_link_to_sensitive_internal_content(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    _write(root / ".env", "TOKEN=secret")
    try:
        (root / ".gitignore").symlink_to(root / ".env")
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    with pytest.raises(WorkspaceError) as captured:
        Workspace(root)

    assert captured.value.code == "invalid_workspace"


def test_gitignore_supports_utf8_bom_and_rejects_invalid_utf8(tmp_path: Path) -> None:
    bom_root = tmp_path / "bom-workspace"
    bom_root.mkdir()
    _write(bom_root / "ignored.txt")
    (bom_root / ".gitignore").write_bytes(b"\xef\xbb\xbfignored.txt\n")

    with pytest.raises(WorkspaceError) as ignored:
        Workspace(bom_root).resolve("ignored.txt", expected="file")
    assert ignored.value.code == "path_ignored"

    invalid_root = tmp_path / "invalid-workspace"
    invalid_root.mkdir()
    (invalid_root / ".gitignore").write_bytes(b"\xff\xfe\x00")
    with pytest.raises(WorkspaceError) as invalid:
        Workspace(invalid_root)
    assert invalid.value.code == "invalid_workspace"


def test_workspace_without_gitignore_uses_an_empty_ignore_spec(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    _write(root / "visible.py")

    assert Workspace(root).resolve("visible.py", expected="file").relative == "visible.py"


@pytest.mark.skipif(os.name != "nt", reason="Windows-only reparse point metadata")
def test_windows_reparse_attribute_is_recognized_without_symlink_flag() -> None:
    class ReparsePath:
        @staticmethod
        def is_symlink() -> bool:
            return False

        @staticmethod
        def lstat() -> SimpleNamespace:
            return SimpleNamespace(st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)

    assert Workspace._is_link_like(cast(Path, ReparsePath())) is True


def test_file_symlink_cannot_bypass_workspace_or_ignore_policy(
    workspace_root: Path,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.txt"
    _write(outside, "secret")
    external_link = workspace_root / "external-link.txt"
    ignored_link = workspace_root / "ignored-link.txt"
    try:
        external_link.symlink_to(outside)
        ignored_link.symlink_to(workspace_root / "ignored.log")
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    workspace = Workspace(workspace_root)

    with pytest.raises(WorkspaceError) as outside_error:
        workspace.resolve("external-link.txt", expected="file")
    with pytest.raises(WorkspaceError) as ignored_error:
        workspace.resolve("ignored-link.txt", expected="file")

    assert outside_error.value.code == "path_outside_workspace"
    assert ignored_error.value.code == "path_ignored"


def test_directory_symlinks_are_not_traversed_even_when_target_is_internal(
    workspace_root: Path,
) -> None:
    directory_link = workspace_root / "src-link"
    try:
        directory_link.symlink_to(workspace_root / "src", target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    workspace = Workspace(workspace_root)
    root_entries = workspace.children(workspace.resolve(".", expected="directory")).entries

    assert "src-link" not in [entry.relative for entry in root_entries]
    with pytest.raises(WorkspaceError) as captured:
        workspace.resolve("src-link", expected="directory")
    assert captured.value.code == "invalid_path"


def test_directory_scan_has_a_deterministic_hard_entry_limit(workspace_root: Path) -> None:
    _write(workspace_root / "a.py")
    _write(workspace_root / "b.py")
    _write(workspace_root / "c.py")
    workspace = Workspace(workspace_root)

    scan = workspace.children(workspace.resolve(".", expected="directory"), max_entries=2)

    assert [entry.relative for entry in scan.entries] == [".env.example", ".gitignore"]
    assert scan.examined >= 2
    assert scan.truncated is True


def test_directory_scan_rejects_a_non_positive_limit(workspace_root: Path) -> None:
    workspace = Workspace(workspace_root)

    with pytest.raises(ValueError, match="max_entries must be at least 1"):
        workspace.children(workspace.resolve(".", expected="directory"), max_entries=0)

    with pytest.raises(ValueError, match="max_examined must be between"):
        workspace.children(workspace.resolve(".", expected="directory"), max_examined=0)


def test_directory_scan_counts_ignored_raw_entries_against_its_budget(tmp_path: Path) -> None:
    root = tmp_path / "raw-budget"
    root.mkdir()
    _write(root / ".gitignore", "*.tmp\n")
    for index in range(20):
        _write(root / f"{index:02}.tmp")
    workspace = Workspace(root)

    scan = workspace.children(
        workspace.resolve(".", expected="directory"),
        max_entries=20,
        max_examined=4,
    )

    assert scan.examined == 4
    assert scan.truncated is True
    assert {entry.relative for entry in scan.entries} <= {".gitignore"}


def test_workspace_reads_authorized_files_through_a_bounded_handle(workspace_root: Path) -> None:
    workspace = Workspace(workspace_root)
    target = workspace.resolve("src/南京.py", expected="file")

    assert workspace.read_bytes(target, max_bytes=100) == b"content"

    with pytest.raises(WorkspaceError) as captured:
        workspace.read_bytes(target, max_bytes=3)
    assert captured.value.code == "file_too_large"


def test_workspace_rejects_a_file_swapped_to_an_outside_symlink(
    workspace_root: Path,
    tmp_path: Path,
) -> None:
    victim = workspace_root / "victim.txt"
    outside = tmp_path / "outside-secret.txt"
    _write(victim, "safe")
    _write(outside, "OUTSIDE_SECRET")
    workspace = Workspace(workspace_root)
    authorized = workspace.resolve("victim.txt", expected="file")
    victim.unlink()
    try:
        victim.symlink_to(outside)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    with pytest.raises(WorkspaceError) as captured:
        workspace.read_bytes(authorized, max_bytes=100)

    assert captured.value.code in {"invalid_path", "path_outside_workspace"}
    assert "OUTSIDE_SECRET" not in captured.value.message


def test_workspace_rejects_hardlinked_sensitive_content(workspace_root: Path) -> None:
    public_alias = workspace_root / "public.txt"
    try:
        os.link(workspace_root / ".env", public_alias)
    except OSError:
        pytest.skip("hard links are unavailable on this platform")
    workspace = Workspace(workspace_root)
    target = workspace.resolve("public.txt", expected="file")

    with pytest.raises(WorkspaceError) as captured:
        workspace.read_bytes(target, max_bytes=100)

    assert captured.value.code == "unsafe_file_link"
    assert "TOKEN=secret" not in captured.value.message


def test_oversized_gitignore_is_rejected_before_parsing(tmp_path: Path) -> None:
    root = tmp_path / "oversized-ignore"
    root.mkdir()
    (root / ".gitignore").write_bytes(b"a" * 256_001)

    with pytest.raises(WorkspaceError) as captured:
        Workspace(root)

    assert captured.value.code == "invalid_workspace"
