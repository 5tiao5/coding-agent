"""Atomic mutation tests for the rooted workspace boundary."""

from __future__ import annotations

import os
import stat
import subprocess
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

import pytest

import coding_agent.workspace as workspace_module
from coding_agent.workspace import FileSnapshot, Workspace, WorkspaceError


class HookedWorkspace(Workspace):
    def __init__(self, root: Path, hook: Callable[[str, Path], None]) -> None:
        super().__init__(root)
        self._hook = hook

    def _before_mutation_commit(self, operation: str, path: Path) -> None:
        self._hook(operation, path)


@pytest.fixture
def mutation_root(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "src").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / ".coding-agent").mkdir()
    (root / ".gitignore").write_text("*.log\nignored/\n", encoding="utf-8")
    (root / "src" / "existing.py").write_bytes(b"before\n")
    (root / ".env").write_text("TOKEN=secret", encoding="utf-8")
    return root


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _temporary_files(root: Path) -> list[Path]:
    return list(root.rglob(".coding-agent-tmp-*"))


def _windows_file_attributes(path: Path) -> int:
    import ctypes
    from ctypes import wintypes

    get_attributes = ctypes.WinDLL("kernel32", use_last_error=True).GetFileAttributesW
    get_attributes.argtypes = [wintypes.LPCWSTR]
    get_attributes.restype = wintypes.DWORD
    attributes = get_attributes(str(path))
    if attributes == 0xFFFFFFFF:
        raise ctypes.WinError(ctypes.get_last_error())
    return int(attributes)


def _set_windows_file_attributes(path: Path, attributes: int) -> None:
    import ctypes
    from ctypes import wintypes

    set_attributes = ctypes.WinDLL("kernel32", use_last_error=True).SetFileAttributesW
    set_attributes.argtypes = [wintypes.LPCWSTR, wintypes.DWORD]
    set_attributes.restype = wintypes.BOOL
    if not set_attributes(str(path), attributes):
        raise ctypes.WinError(ctypes.get_last_error())


def _windows_dacl(path: Path) -> bytes:
    import ctypes
    from ctypes import wintypes

    get_security = ctypes.WinDLL("advapi32", use_last_error=True).GetFileSecurityW
    get_security.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.LPDWORD,
    ]
    get_security.restype = wintypes.BOOL
    required = wintypes.DWORD()
    dacl_security_information = 0x00000004
    get_security(
        str(path),
        dacl_security_information,
        None,
        0,
        ctypes.byref(required),
    )
    if required.value == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    descriptor = ctypes.create_string_buffer(required.value)
    if not get_security(
        str(path),
        dacl_security_information,
        descriptor,
        len(descriptor),
        ctypes.byref(required),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    return descriptor.raw[: required.value]


def test_snapshot_and_atomic_create_return_a_stable_receipt(mutation_root: Path) -> None:
    workspace = Workspace(mutation_root)
    snapshot = workspace.snapshot_for_write("src/new.py", max_bytes=1_000)

    assert snapshot == FileSnapshot(
        relative="src/new.py",
        data=None,
        sha256=None,
        mode=None,
    )

    receipt = workspace.commit_bytes(snapshot, b"print('new')\n")

    assert (mutation_root / "src" / "new.py").read_bytes() == b"print('new')\n"
    assert receipt.relative == "src/new.py"
    assert receipt.before_sha256 is None
    assert receipt.after_sha256 == _digest(b"print('new')\n")
    assert receipt.bytes_written == len(b"print('new')\n")
    assert receipt.created is True
    assert receipt.after_identity is not None
    assert _temporary_files(mutation_root) == []


def test_existing_file_is_replaced_only_from_its_snapshot(mutation_root: Path) -> None:
    target = mutation_root / "src" / "existing.py"
    if os.name != "nt":
        target.chmod(0o754)
    workspace = Workspace(mutation_root)
    snapshot = workspace.snapshot_for_write("src/existing.py", max_bytes=1_000)
    assert snapshot.identity is not None

    receipt = workspace.commit_bytes(snapshot, b"after\n")

    assert target.read_bytes() == b"after\n"
    assert receipt.before_sha256 == _digest(b"before\n")
    assert receipt.after_sha256 == _digest(b"after\n")
    assert receipt.created is False
    if os.name != "nt":
        assert stat.S_IMODE(target.stat().st_mode) == 0o754


def test_windows_replace_preserves_existing_file_attributes(mutation_root: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows file attributes are platform-specific")
    target = mutation_root / "src" / "existing.py"
    hidden_attribute = 0x00000002
    original_attributes = _windows_file_attributes(target)
    _set_windows_file_attributes(target, original_attributes | hidden_attribute)

    snapshot = Workspace(mutation_root).snapshot_for_write("src/existing.py", max_bytes=1_000)
    Workspace(mutation_root).commit_bytes(snapshot, b"after\n")

    assert _windows_file_attributes(target) & hidden_attribute
    assert _temporary_files(mutation_root) == []


def test_windows_replace_preserves_existing_dacl(mutation_root: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows DACLs are platform-specific")
    target = mutation_root / "src" / "existing.py"
    granted = subprocess.run(
        ["icacls", str(target), "/grant", "*S-1-1-0:R"],
        check=False,
        capture_output=True,
        text=True,
    )
    if granted.returncode != 0:
        pytest.skip(f"icacls could not prepare the DACL fixture: {granted.stderr.strip()}")
    expected_dacl = _windows_dacl(target)

    snapshot = Workspace(mutation_root).snapshot_for_write("src/existing.py", max_bytes=1_000)
    Workspace(mutation_root).commit_bytes(snapshot, b"after\n")

    assert _windows_dacl(target) == expected_dacl
    assert _temporary_files(mutation_root) == []


def test_windows_partial_replace_failure_restores_original_file(
    mutation_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "nt":
        pytest.skip("ReplaceFileW failure recovery is Windows-specific")
    target = mutation_root / "src" / "existing.py"

    def partial_move(replaced: Path, replacement: Path, backup: Path) -> int:
        del replacement
        os.rename(replaced, backup)
        return 1177

    monkeypatch.setattr(workspace_module, "_invoke_replace_file_windows", partial_move)
    workspace = Workspace(mutation_root)
    snapshot = workspace.snapshot_for_write("src/existing.py", max_bytes=1_000)

    with pytest.raises(WorkspaceError) as captured:
        workspace.commit_bytes(snapshot, b"after\n")

    assert captured.value.code == "io_error"
    assert target.read_bytes() == b"before\n"
    assert _temporary_files(mutation_root) == []


def test_windows_failed_replace_cleans_only_the_uncommitted_replacement(
    mutation_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "nt":
        pytest.skip("ReplaceFileW failure recovery is Windows-specific")
    target = mutation_root / "src" / "existing.py"

    def failed_move(replaced: Path, replacement: Path, backup: Path) -> int:
        del replaced, replacement, backup
        return 1176

    monkeypatch.setattr(workspace_module, "_invoke_replace_file_windows", failed_move)
    workspace = Workspace(mutation_root)
    snapshot = workspace.snapshot_for_write("src/existing.py", max_bytes=1_000)

    with pytest.raises(WorkspaceError) as captured:
        workspace.commit_bytes(snapshot, b"after\n")

    assert captured.value.code == "io_error"
    assert target.read_bytes() == b"before\n"
    assert _temporary_files(mutation_root) == []


def test_windows_partial_replace_preserves_backup_when_restore_is_blocked(
    mutation_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "nt":
        pytest.skip("ReplaceFileW failure recovery is Windows-specific")
    target = mutation_root / "src" / "existing.py"

    def raced_partial_move(replaced: Path, replacement: Path, backup: Path) -> int:
        del replacement
        os.rename(replaced, backup)
        replaced.write_bytes(b"external version\n")
        return 1177

    monkeypatch.setattr(
        workspace_module,
        "_invoke_replace_file_windows",
        raced_partial_move,
    )
    workspace = Workspace(mutation_root)
    snapshot = workspace.snapshot_for_write("src/existing.py", max_bytes=1_000)

    with pytest.raises(WorkspaceError) as captured:
        workspace.commit_bytes(snapshot, b"after\n")

    assert captured.value.code == "write_recovery_required"
    assert target.read_bytes() == b"external version\n"
    reserved_files = _temporary_files(mutation_root)
    assert len(reserved_files) == 1
    assert reserved_files[0].read_bytes() == b"before\n"
    assert captured.value.metadata["backup_name"] == reserved_files[0].name


def test_empty_file_content_is_a_valid_atomic_write(mutation_root: Path) -> None:
    workspace = Workspace(mutation_root)
    snapshot = workspace.snapshot_for_write("src/empty.py", max_bytes=1)

    receipt = workspace.commit_bytes(snapshot, b"")

    assert (mutation_root / "src" / "empty.py").read_bytes() == b""
    assert receipt.bytes_written == 0
    assert receipt.after_sha256 == _digest(b"")


def test_existing_file_content_change_causes_a_cas_conflict(mutation_root: Path) -> None:
    target = mutation_root / "src" / "existing.py"
    workspace = Workspace(mutation_root)
    snapshot = workspace.snapshot_for_write("src/existing.py", max_bytes=1_000)
    target.write_bytes(b"raced!\n")

    with pytest.raises(WorkspaceError) as captured:
        workspace.commit_bytes(snapshot, b"agent change\n")

    assert captured.value.code == "write_conflict"
    assert captured.value.metadata == {
        "path": "src/existing.py",
        "expected_sha256": _digest(b"before\n"),
        "current_sha256": _digest(b"raced!\n"),
        "recovery": "read_file_then_retry",
    }
    assert target.read_bytes() == b"raced!\n"
    assert _temporary_files(mutation_root) == []


def test_missing_target_that_appears_is_never_clobbered(mutation_root: Path) -> None:
    target = mutation_root / "src" / "new.py"
    workspace = Workspace(mutation_root)
    snapshot = workspace.snapshot_for_write("src/new.py", max_bytes=1_000)
    target.write_bytes(b"created elsewhere\n")

    with pytest.raises(WorkspaceError) as captured:
        workspace.commit_bytes(snapshot, b"agent content\n")

    assert captured.value.code == "write_conflict"
    assert target.read_bytes() == b"created elsewhere\n"


def test_final_revalidation_detects_a_hooked_race_and_cleans_temporary_file(
    mutation_root: Path,
) -> None:
    target = mutation_root / "src" / "existing.py"
    seen: list[tuple[str, Path]] = []

    def race(operation: str, path: Path) -> None:
        seen.append((operation, path))
        target.write_bytes(b"raced\n")

    workspace = HookedWorkspace(mutation_root, race)
    snapshot = workspace.snapshot_for_write("src/existing.py", max_bytes=1_000)

    with pytest.raises(WorkspaceError) as captured:
        workspace.commit_bytes(snapshot, b"agent content\n")

    assert captured.value.code == "write_conflict"
    assert target.read_bytes() == b"raced\n"
    assert seen == [("write", target.resolve())]
    assert _temporary_files(mutation_root) == []


def test_final_revalidation_detects_same_byte_file_replacement(
    mutation_root: Path,
) -> None:
    target = mutation_root / "src" / "existing.py"

    def race(operation: str, path: Path) -> None:
        assert operation == "write"
        assert path == target.resolve()
        target.unlink()
        target.write_bytes(b"before\n")

    workspace = HookedWorkspace(mutation_root, race)
    snapshot = workspace.snapshot_for_write("src/existing.py", max_bytes=1_000)

    with pytest.raises(WorkspaceError) as captured:
        workspace.commit_bytes(snapshot, b"agent change\n")

    assert captured.value.code == "write_conflict"
    assert target.read_bytes() == b"before\n"
    assert _temporary_files(mutation_root) == []


def test_precommit_io_failure_leaves_original_and_cleans_temporary_file(
    mutation_root: Path,
) -> None:
    target = mutation_root / "src" / "existing.py"

    def fail(_operation: str, _path: Path) -> None:
        raise OSError("injected failure")

    workspace = HookedWorkspace(mutation_root, fail)
    snapshot = workspace.snapshot_for_write("src/existing.py", max_bytes=1_000)

    with pytest.raises(WorkspaceError) as captured:
        workspace.commit_bytes(snapshot, b"agent content\n")

    assert captured.value.code == "io_error"
    assert target.read_bytes() == b"before\n"
    assert _temporary_files(mutation_root) == []


def test_precommit_create_race_preserves_the_other_writer(mutation_root: Path) -> None:
    target = mutation_root / "src" / "new.py"

    def race(_operation: str, _path: Path) -> None:
        target.write_bytes(b"won by another writer\n")

    workspace = HookedWorkspace(mutation_root, race)
    snapshot = workspace.snapshot_for_write("src/new.py", max_bytes=1_000)

    with pytest.raises(WorkspaceError) as captured:
        workspace.commit_bytes(snapshot, b"agent content\n")

    assert captured.value.code == "write_conflict"
    assert target.read_bytes() == b"won by another writer\n"
    assert _temporary_files(mutation_root) == []


def test_precommit_symlink_swap_never_writes_through_to_outside(
    mutation_root: Path,
    tmp_path: Path,
) -> None:
    target = mutation_root / "src" / "existing.py"
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"outside\n")
    probe = mutation_root / "src" / "symlink-probe"
    try:
        probe.symlink_to(outside)
        probe.unlink()
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    def race(_operation: str, _path: Path) -> None:
        target.unlink()
        target.symlink_to(outside)

    workspace = HookedWorkspace(mutation_root, race)
    snapshot = workspace.snapshot_for_write("src/existing.py", max_bytes=1_000)

    with pytest.raises(WorkspaceError):
        workspace.commit_bytes(snapshot, b"agent content\n")

    assert outside.read_bytes() == b"outside\n"
    assert target.is_symlink()
    assert _temporary_files(mutation_root) == []


@pytest.mark.parametrize(
    ("path", "code"),
    [
        ("../outside.py", "invalid_path"),
        (".git/config", "path_ignored"),
        (".coding-agent/state", "path_ignored"),
        ("src/.coding-agent-tmp-user", "path_ignored"),
        (".env", "sensitive_path"),
        ("ignored/new.py", "path_ignored"),
        ("debug.log", "path_ignored"),
        ("missing/new.py", "not_found"),
    ],
)
def test_snapshot_reuses_workspace_policy_for_new_destinations(
    mutation_root: Path,
    path: str,
    code: str,
) -> None:
    workspace = Workspace(mutation_root)

    with pytest.raises(WorkspaceError) as captured:
        workspace.snapshot_for_write(path, max_bytes=1_000)

    assert captured.value.code == code


def test_file_links_are_not_valid_mutation_targets(mutation_root: Path) -> None:
    target = mutation_root / "src" / "existing.py"
    alias = mutation_root / "src" / "alias.py"
    try:
        alias.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    with pytest.raises(WorkspaceError) as captured:
        Workspace(mutation_root).snapshot_for_write("src/alias.py", max_bytes=1_000)

    assert captured.value.code == "unsafe_file_link"


def test_hardlinks_are_not_valid_mutation_targets(mutation_root: Path) -> None:
    target = mutation_root / "src" / "existing.py"
    alias = mutation_root / "src" / "hardlink.py"
    try:
        os.link(target, alias)
    except OSError:
        pytest.skip("hard links are unavailable on this platform")

    with pytest.raises(WorkspaceError) as captured:
        Workspace(mutation_root).snapshot_for_write("src/hardlink.py", max_bytes=1_000)

    assert captured.value.code == "unsafe_file_link"


def test_windows_named_streams_are_not_silently_discarded(mutation_root: Path) -> None:
    if os.name != "nt":
        pytest.skip("NTFS named streams are Windows-specific")
    target = mutation_root / "src" / "existing.py"
    stream = Path(f"{target}:coding-agent-test")
    try:
        stream.write_bytes(b"hidden metadata")
    except OSError:
        pytest.skip("the temporary filesystem does not support named streams")

    with pytest.raises(WorkspaceError) as captured:
        Workspace(mutation_root).snapshot_for_write("src/existing.py", max_bytes=1_000)

    assert captured.value.code == "unsafe_file_stream"
    assert target.read_bytes() == b"before\n"
    assert stream.read_bytes() == b"hidden metadata"


def test_forged_snapshot_is_rejected_before_any_write(mutation_root: Path) -> None:
    forged = FileSnapshot(
        relative="src/existing.py",
        data=b"forged",
        sha256=_digest(b"different"),
        mode=0o644,
    )

    with pytest.raises(WorkspaceError) as captured:
        Workspace(mutation_root).commit_bytes(forged, b"new")

    assert captured.value.code == "invalid_snapshot"
    assert (mutation_root / "src" / "existing.py").read_bytes() == b"before\n"


@pytest.mark.parametrize(
    "forged",
    [
        FileSnapshot(
            relative="src/new.py",
            data=None,
            sha256="a" * 64,
            mode=None,
        ),
        FileSnapshot(
            relative="src/existing.py",
            data=b"before\n",
            sha256=None,
            mode=None,
        ),
    ],
)
def test_internally_inconsistent_snapshots_are_rejected(
    mutation_root: Path,
    forged: FileSnapshot,
) -> None:
    with pytest.raises(WorkspaceError) as captured:
        Workspace(mutation_root).commit_bytes(forged, b"new")

    assert captured.value.code == "invalid_snapshot"


def test_mutation_primitives_validate_basic_types_and_budgets(mutation_root: Path) -> None:
    workspace = Workspace(mutation_root)

    with pytest.raises(ValueError, match="max_bytes"):
        workspace.snapshot_for_write("src/existing.py", max_bytes=0)
    snapshot = workspace.snapshot_for_write("src/existing.py", max_bytes=1_000)
    with pytest.raises(TypeError, match="new_data"):
        workspace.commit_bytes(snapshot, bytearray(b"not immutable"))  # type: ignore[arg-type]


def test_snapshot_rejects_a_directory_as_a_file_target(mutation_root: Path) -> None:
    with pytest.raises(WorkspaceError) as captured:
        Workspace(mutation_root).snapshot_for_write("src", max_bytes=1_000)

    assert captured.value.code == "not_file"


def test_mode_change_is_a_write_conflict(mutation_root: Path) -> None:
    target = mutation_root / "src" / "existing.py"
    workspace = Workspace(mutation_root)
    snapshot = workspace.snapshot_for_write("src/existing.py", max_bytes=1_000)
    changed_mode = 0o444 if snapshot.mode != 0o444 else 0o666
    target.chmod(changed_mode)

    with pytest.raises(WorkspaceError) as captured:
        workspace.commit_bytes(snapshot, b"new\n")

    assert captured.value.code == "write_conflict"
    assert captured.value.metadata["expected_mode"] == snapshot.mode
    assert captured.value.metadata["current_mode"] != snapshot.mode


def test_remove_if_unchanged_deletes_only_the_expected_version(mutation_root: Path) -> None:
    target = mutation_root / "src" / "created.py"
    target.write_bytes(b"created by agent\n")
    workspace = Workspace(mutation_root)

    with pytest.raises(WorkspaceError) as captured:
        workspace.remove_if_unchanged("src/created.py", _digest(b"another version"))
    assert captured.value.code == "write_conflict"
    assert target.exists()

    workspace.remove_if_unchanged("src/created.py", _digest(b"created by agent\n"))

    assert not target.exists()


def test_remove_missing_file_is_a_conflict(mutation_root: Path) -> None:
    with pytest.raises(WorkspaceError) as captured:
        Workspace(mutation_root).remove_if_unchanged("src/missing.py", _digest(b"missing"))

    assert captured.value.code == "write_conflict"


def test_remove_revalidates_after_hook_and_keeps_the_raced_version(
    mutation_root: Path,
) -> None:
    target = mutation_root / "src" / "created.py"
    target.write_bytes(b"created by agent\n")

    def race(operation: str, path: Path) -> None:
        assert operation == "remove"
        assert path == target.resolve()
        target.write_bytes(b"external change\n")

    workspace = HookedWorkspace(mutation_root, race)

    with pytest.raises(WorkspaceError) as captured:
        workspace.remove_if_unchanged("src/created.py", _digest(b"created by agent\n"))

    assert captured.value.code == "write_conflict"
    assert target.read_bytes() == b"external change\n"


def test_remove_rejects_a_same_byte_replacement_by_identity(mutation_root: Path) -> None:
    target = mutation_root / "src" / "created.py"
    target.write_bytes(b"created by agent\n")

    def race(operation: str, path: Path) -> None:
        assert operation == "remove"
        assert path == target.resolve()
        target.unlink()
        target.write_bytes(b"created by agent\n")

    workspace = HookedWorkspace(mutation_root, race)
    snapshot = workspace.snapshot_for_write("src/created.py", max_bytes=1_000)

    with pytest.raises(WorkspaceError) as captured:
        workspace.remove_if_unchanged(
            "src/created.py",
            _digest(b"created by agent\n"),
            expected_identity=snapshot.identity,
        )

    assert captured.value.code == "write_conflict"
    assert target.read_bytes() == b"created by agent\n"


def test_remove_rejects_malformed_digest(mutation_root: Path) -> None:
    with pytest.raises(ValueError, match="lowercase 64-character"):
        Workspace(mutation_root).remove_if_unchanged("src/existing.py", "not-a-digest")


def test_windows_ignore_matching_is_fail_closed_for_path_case(mutation_root: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows case-insensitive ignore policy")
    (mutation_root / ".gitignore").write_text("secret.txt\n", encoding="utf-8")
    workspace = Workspace(mutation_root)

    with pytest.raises(WorkspaceError) as caught:
        workspace.snapshot_for_write("SECRET.TXT", max_bytes=100)

    assert caught.value.code == "path_ignored"
    assert not mutation_root.joinpath("SECRET.TXT").exists()


@pytest.mark.parametrize("relative", [".gitignore", "nested/.gitignore"])
def test_repository_ignore_policy_files_are_not_mutation_targets(
    mutation_root: Path, relative: str
) -> None:
    workspace = Workspace(mutation_root)

    with pytest.raises(WorkspaceError) as caught:
        workspace.snapshot_for_write(relative, max_bytes=100)

    assert caught.value.code == "path_ignored"
