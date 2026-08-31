"""Repeatable project verification integrity-check tests."""

from __future__ import annotations

from pathlib import Path

from coding_agent.integrity import check_integrity
from coding_agent.project_config import load_project_policy


def _configured_project(root: Path) -> tuple[Path, Path, Path]:
    interpreter = root / "runtime" / "python"
    interpreter.parent.mkdir()
    interpreter.write_bytes(b"runtime-v1")
    interpreter.chmod(0o755)
    tests = root / "tests"
    tests.mkdir()
    protected_test = tests / "test_public.py"
    protected_test.write_text("def test_public(): assert True\n", "utf-8")
    exact = root / "pyproject.toml"
    exact.write_text("[project]\nname='fixture'\n", "utf-8")
    config = root / ".coding-agent" / "project.toml"
    config.parent.mkdir()
    config.write_text(
        """schema_version = 1
protected_paths = ["tests/", "pyproject.toml"]

[python]
executable = "runtime/python"

[[verifiers]]
label = "pytest"
type = "pytest"
cwd = "."
scopes = ["tests"]
required = true
""",
        "utf-8",
    )
    return interpreter, protected_test, config


def test_check_integrity_is_repeatable_without_mutation(tmp_path: Path) -> None:
    _configured_project(tmp_path)
    policy = load_project_policy(tmp_path)

    first = check_integrity(policy)
    second = check_integrity(policy)

    assert first == second
    assert first.intact is True
    assert first.violations == ()
    assert first.current_policy_fingerprint == policy.policy_fingerprint


def test_configuration_creation_invalidates_an_unconfigured_policy(tmp_path: Path) -> None:
    policy = load_project_policy(tmp_path)
    config = tmp_path / ".coding-agent" / "project.toml"
    config.parent.mkdir()
    config.write_text("schema_version = 1\n", "utf-8")

    report = check_integrity(policy)

    assert report.intact is False
    assert report.violations == ("configuration_changed",)


def test_configuration_byte_change_invalidates_policy(tmp_path: Path) -> None:
    _, _, config = _configured_project(tmp_path)
    policy = load_project_policy(tmp_path)
    config.write_text(config.read_text("utf-8") + "\n# owner edit\n", "utf-8")

    report = check_integrity(policy)

    assert report.intact is False
    assert "configuration_changed" in report.violations
    assert report.current_policy_fingerprint != policy.policy_fingerprint


def test_interpreter_byte_change_invalidates_policy(tmp_path: Path) -> None:
    interpreter, _, _ = _configured_project(tmp_path)
    policy = load_project_policy(tmp_path)
    interpreter.write_bytes(b"runtime-v2")
    interpreter.chmod(0o755)

    report = check_integrity(policy)

    assert report.intact is False
    assert "interpreter_changed" in report.violations


def test_protected_file_change_and_restoration_are_observed(tmp_path: Path) -> None:
    _, protected_test, _ = _configured_project(tmp_path)
    original = protected_test.read_bytes()
    policy = load_project_policy(tmp_path)

    protected_test.write_text("def test_public(): assert False\n", "utf-8")
    changed = check_integrity(policy)
    protected_test.write_bytes(original)
    restored = check_integrity(policy)

    assert changed.intact is False
    assert "protected_manifest_changed" in changed.violations
    assert restored.intact is True


def test_protected_tree_addition_and_deletion_are_observed(tmp_path: Path) -> None:
    _, protected_test, _ = _configured_project(tmp_path)
    policy = load_project_policy(tmp_path)

    added = tmp_path / "tests" / "test_added.py"
    added.write_text("def test_added(): assert True\n", "utf-8")
    addition = check_integrity(policy)
    added.unlink()
    protected_test.unlink()
    deletion = check_integrity(policy)

    assert addition.intact is False
    assert "protected_manifest_changed" in addition.violations
    assert deletion.intact is False
    assert "protected_manifest_changed" in deletion.violations


def test_protected_exact_file_type_change_fails_closed(tmp_path: Path) -> None:
    _configured_project(tmp_path)
    policy = load_project_policy(tmp_path)
    exact = tmp_path / "pyproject.toml"
    exact.unlink()
    exact.mkdir()

    report = check_integrity(policy)

    assert report.intact is False
    assert "protected_manifest_unavailable" in report.violations
    assert report.current_policy_fingerprint is None


def test_missing_interpreter_fails_closed_without_raw_path_or_content(tmp_path: Path) -> None:
    interpreter, _, _ = _configured_project(tmp_path)
    policy = load_project_policy(tmp_path)
    interpreter.unlink()

    report = check_integrity(policy)

    assert report.intact is False
    assert report.violations == ("interpreter_changed",)
    rendered = repr(report)
    assert "runtime-v1" not in rendered
    assert str(interpreter) not in rendered
