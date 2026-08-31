"""Strict project verification configuration tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from coding_agent.integrity import check_integrity
from coding_agent.models import VerificationKind
from coding_agent.project_config import (
    ProjectConfigError,
    VerifierType,
    load_project_policy,
)


def _executable(root: Path, relative: str = ".venv/bin/python") -> Path:
    path = root.joinpath(*relative.replace("\\", "/").split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-python-runtime")
    path.chmod(0o755)
    return path


def _write_config(
    root: Path,
    *,
    executable: str | None,
    protected: str = '["tests/", "pyproject.toml"]',
    extra: str = "",
) -> Path:
    config = root / ".coding-agent" / "project.toml"
    config.parent.mkdir(parents=True, exist_ok=True)
    python = "" if executable is None else f"[python]\nexecutable = {json.dumps(executable)}\n\n"
    config.write_text(
        (
            "schema_version = 1\n"
            f"protected_paths = {protected}\n\n"
            f"{python}"
            "[[verifiers]]\n"
            'label = "pytest"\n'
            'type = "pytest"\n'
            'cwd = "."\n'
            'scopes = ["tests"]\n'
            "required = true\n\n"
            "[[verifiers]]\n"
            'label = "module-smoke"\n'
            'type = "python-module"\n'
            'module = "telemetry_app"\n'
            'cwd = "."\n'
            'scopes = ["runtime:entrypoint"]\n'
            "required = true\n\n"
            "[completion]\n"
            'required_scopes = ["tests", "runtime:entrypoint"]\n'
            f"{extra}"
        ),
        encoding="utf-8",
    )
    return config


def _fixture(root: Path, executable: str = ".venv/bin/python") -> Path:
    runtime = _executable(root, executable)
    (root / "tests").mkdir()
    (root / "tests" / "test_public.py").write_text("def test_ok(): assert True\n", "utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", "utf-8")
    _write_config(root, executable=executable)
    return runtime


def test_missing_config_returns_explicit_unconfigured_policy(tmp_path: Path) -> None:
    policy = load_project_policy(tmp_path)

    assert policy.configured is False
    assert policy.interpreter is None
    assert policy.verifiers == ()
    assert policy.required_labels == ()
    assert policy.required_scopes == ()
    assert policy.target_runtime_eligible is False
    assert policy.target_runtime_id == "unconfigured-python"
    assert check_integrity(policy).intact is True


def test_config_compiles_typed_verifiers_and_completion_inputs(tmp_path: Path) -> None:
    runtime = _fixture(tmp_path)

    policy = load_project_policy(tmp_path)

    assert policy.configured is True
    assert policy.config_sha256 is not None
    assert policy.interpreter is not None
    assert policy.interpreter.invocation_path == runtime.absolute()
    assert policy.interpreter.explicitly_configured is True
    assert policy.target_runtime_eligible is True
    assert policy.target_runtime_id.startswith("project-python:")
    assert policy.required_labels == ("pytest", "module-smoke")
    assert policy.required_scopes == (
        "tests",
        "runtime:entrypoint",
        "integrity:protected",
    )
    assert [verifier.verifier_type for verifier in policy.verifiers] == [
        VerifierType.PYTEST,
        VerifierType.PYTHON_MODULE,
    ]
    assert [verifier.kind for verifier in policy.verifiers] == [
        VerificationKind.TEST,
        VerificationKind.CHECK,
    ]
    assert policy.verifiers[0].argv[1:] == (
        "-I",
        "-B",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
    )
    assert policy.verifiers[1].argv[1:] == ("-B", "-m", "telemetry_app")
    assert check_integrity(policy).intact is True


def test_omitted_interpreter_binds_host_fallback_without_validation_eligibility(
    tmp_path: Path,
) -> None:
    host = _executable(tmp_path, "host/python")
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text("fixture", "utf-8")
    _write_config(tmp_path, executable=None)

    policy = load_project_policy(tmp_path, host_python=host)

    assert policy.interpreter is not None
    assert policy.interpreter.invocation_path == host.absolute()
    assert policy.interpreter.explicitly_configured is False
    assert policy.target_runtime_eligible is False
    assert policy.target_runtime_id.startswith("host-fallback-python:")


@pytest.mark.parametrize(
    "relative",
    [".venv/bin/python", ".venv\\Scripts\\python.exe"],
    ids=["posix-relative", "windows-relative"],
)
def test_portable_relative_interpreter_paths_are_canonicalized(
    tmp_path: Path,
    relative: str,
) -> None:
    runtime = _executable(tmp_path, relative)
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text("fixture", "utf-8")
    _write_config(tmp_path, executable=relative)

    policy = load_project_policy(tmp_path)

    assert policy.interpreter is not None
    assert policy.interpreter.invocation_path == runtime.absolute()


def test_native_absolute_interpreter_path_is_supported(tmp_path: Path) -> None:
    runtime = _executable(tmp_path, "runtime/python")
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text("fixture", "utf-8")
    _write_config(tmp_path, executable=str(runtime.absolute()))

    policy = load_project_policy(tmp_path)

    assert policy.interpreter is not None
    assert policy.interpreter.invocation_path == runtime.absolute()


@pytest.mark.parametrize(
    "extra",
    [
        "\nunknown = true\n",
        "\n[completion.extra]\nvalue = true\n",
    ],
    ids=["unknown-completion-field", "unknown-nested-table"],
)
def test_schema_v1_rejects_unknown_fields(tmp_path: Path, extra: str) -> None:
    _fixture(tmp_path)
    config = tmp_path / ".coding-agent" / "project.toml"
    content = config.read_text("utf-8")
    config.write_text(content + extra, "utf-8")

    with pytest.raises(ProjectConfigError) as raised:
        load_project_policy(tmp_path)

    assert raised.value.code == "project_config_unknown_field"


def test_schema_v1_rejects_unknown_top_level_and_verifier_fields(tmp_path: Path) -> None:
    _fixture(tmp_path)
    config = tmp_path / ".coding-agent" / "project.toml"
    original = config.read_text("utf-8")
    config.write_text(
        original.replace("schema_version = 1", "schema_version = 1\nsurprise = 2"), "utf-8"
    )

    with pytest.raises(ProjectConfigError) as top_level:
        load_project_policy(tmp_path)
    assert top_level.value.code == "project_config_unknown_field"

    config.write_text(
        original.replace('label = "pytest"', 'label = "pytest"\nshell = "pwsh"'), "utf-8"
    )
    with pytest.raises(ProjectConfigError) as verifier:
        load_project_policy(tmp_path)
    assert verifier.value.code == "project_config_unknown_field"


@pytest.mark.parametrize(
    "field, replacement",
    [
        ('module = "telemetry_app"', 'module = "telemetry_app;rm"'),
        ('cwd = "."', 'cwd = "../outside"'),
        ('scopes = ["tests"]', 'scopes = ["Tests"]'),
        ('label = "pytest"', 'label = " pytest"'),
        ('type = "pytest"', 'type = "shell"'),
    ],
    ids=["module", "cwd", "scope", "label", "untyped-command"],
)
def test_unsafe_verifier_fields_are_rejected(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    _fixture(tmp_path)
    config = tmp_path / ".coding-agent" / "project.toml"
    config.write_text(config.read_text("utf-8").replace(field, replacement, 1), "utf-8")

    with pytest.raises(ProjectConfigError) as raised:
        load_project_policy(tmp_path)

    assert raised.value.code == "project_config_invalid"


@pytest.mark.parametrize(
    "protected",
    [
        '["../tests/"]',
        '["tests/**"]',
        '["tests/*"]',
        '["/tests/"]',
        '["tests/", "tests/test_public.py"]',
        '["CON"]',
        '["COM¹.txt"]',
    ],
    ids=[
        "parent",
        "globstar",
        "glob",
        "absolute",
        "overlap",
        "windows-device",
        "windows-superscript-device",
    ],
)
def test_protected_paths_are_portable_exact_or_directory_prefixes(
    tmp_path: Path,
    protected: str,
) -> None:
    _executable(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text("fixture", "utf-8")
    _write_config(tmp_path, executable=".venv/bin/python", protected=protected)

    with pytest.raises(ProjectConfigError) as raised:
        load_project_policy(tmp_path)

    assert raised.value.code in {"project_config_invalid", "protected_path_invalid"}


def test_missing_configured_interpreter_is_rejected_without_path_leak(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text("fixture", "utf-8")
    secret_path = "private-secret-runtime/python"
    _write_config(tmp_path, executable=secret_path)

    with pytest.raises(ProjectConfigError) as raised:
        load_project_policy(tmp_path)

    assert raised.value.code == "project_interpreter_unavailable"
    assert "private-secret" not in raised.value.message


def test_oversized_configuration_is_rejected_before_toml_parse(tmp_path: Path) -> None:
    config = tmp_path / ".coding-agent" / "project.toml"
    config.parent.mkdir()
    config.write_bytes(b"x" * 65_537)

    with pytest.raises(ProjectConfigError) as raised:
        load_project_policy(tmp_path)

    assert raised.value.code == "project_config_too_large"


def test_verifier_and_protected_path_count_limits_are_enforced(tmp_path: Path) -> None:
    runtime = _executable(tmp_path)
    del runtime
    config = tmp_path / ".coding-agent" / "project.toml"
    config.parent.mkdir()
    verifiers = "\n".join(
        (f'[[verifiers]]\nlabel = "check-{index}"\ntype = "pytest"\nscopes = ["tests"]\n')
        for index in range(9)
    )
    config.write_text(
        'schema_version = 1\n[python]\nexecutable = ".venv/bin/python"\n' + verifiers,
        "utf-8",
    )
    with pytest.raises(ProjectConfigError) as verifier_limit:
        load_project_policy(tmp_path)
    assert verifier_limit.value.code == "project_config_too_large"

    protected = ", ".join(json.dumps(f"protected-{index}") for index in range(129))
    config.write_text(
        (
            f"schema_version = 1\nprotected_paths = [{protected}]\n"
            '[python]\nexecutable = ".venv/bin/python"\n'
            '[[verifiers]]\nlabel = "pytest"\ntype = "pytest"\nscopes = ["tests"]\n'
        ),
        "utf-8",
    )
    with pytest.raises(ProjectConfigError) as protected_limit:
        load_project_policy(tmp_path)
    assert protected_limit.value.code == "project_config_too_large"


def test_invalid_toml_error_does_not_echo_configuration_content(tmp_path: Path) -> None:
    config = tmp_path / ".coding-agent" / "project.toml"
    config.parent.mkdir()
    config.write_text('schema_version = "SECRET-CONTENT"\n[', "utf-8")

    with pytest.raises(ProjectConfigError) as raised:
        load_project_policy(tmp_path)

    assert raised.value.code == "project_config_invalid"
    assert "SECRET-CONTENT" not in raised.value.message


def test_config_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "outside.toml"
    target.write_text("schema_version = 1", "utf-8")
    config = tmp_path / ".coding-agent" / "project.toml"
    config.parent.mkdir()
    try:
        config.symlink_to(target)
    except OSError:
        pytest.skip("configuration symlinks are unavailable on this platform")

    with pytest.raises(ProjectConfigError) as raised:
        load_project_policy(tmp_path)

    assert raised.value.code == "project_config_unsafe"


def test_protected_symlink_is_rejected(tmp_path: Path) -> None:
    _executable(tmp_path)
    outside = tmp_path / "outside-tests"
    outside.mkdir()
    tests = tmp_path / "tests"
    try:
        tests.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this platform")
    (tmp_path / "pyproject.toml").write_text("fixture", "utf-8")
    _write_config(tmp_path, executable=".venv/bin/python")

    with pytest.raises(ProjectConfigError) as raised:
        load_project_policy(tmp_path)

    assert raised.value.code == "protected_path_unsafe"


@pytest.mark.skipif(os.name == "nt", reason="POSIX virtualenv launchers are symlinks")
def test_interpreter_symlink_is_bound_and_retargeting_is_detected(tmp_path: Path) -> None:
    first = _executable(tmp_path, "runtime/python-one")
    second = _executable(tmp_path, "runtime/python-two")
    second.write_bytes(b"different-runtime")
    launcher = tmp_path / ".venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(first)
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text("fixture", "utf-8")
    _write_config(tmp_path, executable=".venv/bin/python")
    policy = load_project_policy(tmp_path)

    assert policy.interpreter is not None
    assert policy.interpreter.link_target is not None

    launcher.unlink()
    launcher.symlink_to(second)
    report = check_integrity(policy)

    assert report.intact is False
    assert "interpreter_changed" in report.violations


def test_policy_fingerprint_is_stable_for_unchanged_inputs(tmp_path: Path) -> None:
    _fixture(tmp_path)

    first = load_project_policy(tmp_path)
    second = load_project_policy(tmp_path)

    assert first.definition_sha256 == second.definition_sha256
    assert first.protected_manifest == second.protected_manifest
    assert first.policy_fingerprint == second.policy_fingerprint
