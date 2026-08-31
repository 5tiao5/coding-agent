"""Tests for application-level repository runtime wiring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from coding_agent.command import (
    CommandPermissionMode,
    VerificationCommandSpec,
)
from coding_agent.completion import CompletionContract
from coding_agent.models import ToolCall, VerificationKind
from coding_agent.runtime import (
    build_runtime,
    policy_fingerprint_from_prompt,
    system_prompt_for,
)
from coding_agent.workspace import WorkspaceError


def test_runtime_exposes_the_complete_tool_surface(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path, permission_mode=CommandPermissionMode.AUTO)

    assert [spec.name for spec in runtime.tools.specs()] == [
        "list_files",
        "read_file",
        "search_text",
        "run_command",
        "create_directory",
        "write_file",
        "replace_text",
        "undo_change",
        "update_plan",
    ]
    assert runtime.workspace.root == tmp_path.resolve()
    assert len(runtime.verification_commands) == 1
    assert runtime.verification_profile is not None
    assert runtime.completion_contract is not None
    assert runtime.verification_profile.required_labels == ("pytest",)
    assert runtime.verification_profile.checks[0].scopes == ("tests",)
    assert runtime.verification_profile.target_runtime.runtime_id == "unconfigured-python"
    assert runtime.verification_profile.target_runtime.eligible_for_task_validation is False
    assert runtime.completion_contract.required_scopes == ("tests",)
    assert runtime.project_policy.configured is False
    assert len(runtime.policy_fingerprint) == 64
    assert runtime.run_memory.plan_state is runtime.plan_state


def test_runtime_plan_tool_and_host_memory_share_one_plan_state(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path, permission_mode=CommandPermissionMode.AUTO)
    call = ToolCall(
        id="plan",
        name="update_plan",
        arguments={
            "items": [
                {"id": "inspect", "step": "Inspect files", "status": "completed"},
                {"id": "verify", "step": "Run tests", "status": "in_progress"},
            ]
        },
    )

    execution = runtime.tools.execute(call)
    runtime.run_memory.observe(call, execution, step=1)

    assert execution.ok is True
    assert runtime.plan_state.revision == 1
    assert runtime.run_memory.snapshot().plan == runtime.plan_state.snapshot()


def test_system_prompt_names_only_exact_host_registered_verifier() -> None:
    runtime = build_runtime(Path.cwd(), permission_mode=CommandPermissionMode.AUTO)

    prompt = system_prompt_for(runtime.verification_commands)

    verifier = runtime.verification_commands[0]
    assert verifier.label in prompt
    assert verifier.cwd in prompt
    assert json.dumps(list(verifier.argv), ensure_ascii=False) in prompt
    assert "-B" in verifier.argv
    assert verifier.argv[-2:] == ("-p", "no:cacheprovider")
    assert "nearby commands" in prompt


def test_unconfigured_prompt_discloses_checks_only_limit(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path)

    prompt = system_prompt_for(
        runtime.verification_commands,
        target_runtime_eligible=False,
    )

    assert "cannot validate task completion" in prompt
    assert "explicit project interpreter policy" in prompt


def test_policy_prompt_anchor_is_unique_canonical_and_suffix_tolerant() -> None:
    fingerprint = "a" * 64
    prompt = system_prompt_for((), policy_fingerprint=fingerprint)

    assert policy_fingerprint_from_prompt(prompt + "\nMaximum model turns: 20.") == fingerprint
    with pytest.raises(ValueError, match="multiple"):
        policy_fingerprint_from_prompt(prompt + "\n" + prompt)
    with pytest.raises(ValueError, match="malformed"):
        policy_fingerprint_from_prompt(" Verification policy fingerprint: " + fingerprint)
    with pytest.raises(ValueError, match="invalid"):
        policy_fingerprint_from_prompt("Verification policy fingerprint: short")


def test_system_prompt_explains_when_no_verifier_is_configured() -> None:
    assert "remain unverified" in system_prompt_for(())


def test_runtime_profiles_every_registered_verifier_and_target_runtime(tmp_path: Path) -> None:
    executable = str(Path(sys.executable).absolute())
    verifiers = (
        VerificationCommandSpec(
            argv=(executable, "-m", "pytest"),
            cwd=".",
            kind=VerificationKind.TEST,
            label="pytest",
        ),
        VerificationCommandSpec(
            argv=(executable, "-m", "build"),
            cwd=".",
            kind=VerificationKind.BUILD,
            label="package-build",
        ),
        VerificationCommandSpec(
            argv=(executable, "-m", "ruff", "check", "."),
            cwd=".",
            kind=VerificationKind.CHECK,
            label="ruff-check",
        ),
    )

    runtime = build_runtime(
        tmp_path,
        verification_commands=verifiers,
        target_runtime_id="project-python",
        target_runtime_eligible=False,
    )

    assert runtime.verification_profile is not None
    assert runtime.verification_profile.required_labels == (
        "pytest",
        "package-build",
        "ruff-check",
    )
    assert [check.scopes for check in runtime.verification_profile.checks] == [
        ("tests",),
        ("build",),
        ("checks",),
    ]
    assert runtime.verification_profile.target_runtime.runtime_id == "project-python"
    assert runtime.verification_profile.target_runtime.eligible_for_task_validation is False


def test_runtime_without_verifiers_has_no_strict_completion_pair(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path, verification_commands=())

    assert runtime.verification_profile is None
    assert runtime.completion_contract is None


def test_configured_policy_wires_typed_scopes_integrity_and_mutation_denial(
    tmp_path: Path,
) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    tests.joinpath("test_public.py").write_text(
        "def test_public():\n    assert True\n",
        encoding="utf-8",
    )
    package = tmp_path / "sample_app"
    package.mkdir()
    package.joinpath("__main__.py").write_text("print('ok')\n", encoding="utf-8")
    metadata = tmp_path / ".coding-agent"
    metadata.mkdir()
    metadata.joinpath("project.toml").write_text(
        "\n".join(
            (
                "schema_version = 1",
                'protected_paths = ["tests/"]',
                "[python]",
                "executable = " + json.dumps(str(Path(sys.executable).absolute())),
                "[[verifiers]]",
                'label = "pytest"',
                'type = "pytest"',
                'scopes = ["tests"]',
                "required = true",
                "[[verifiers]]",
                'label = "module-smoke"',
                'type = "python-module"',
                'module = "sample_app"',
                'scopes = ["runtime:entrypoint"]',
                "required = true",
                "[completion]",
                'required_scopes = ["tests", "runtime:entrypoint"]',
                "",
            )
        ),
        encoding="utf-8",
    )

    runtime = build_runtime(tmp_path)

    assert runtime.project_policy.configured is True
    assert runtime.verification_profile is not None
    assert runtime.verification_profile.required_labels == ("pytest", "module-smoke")
    assert [check.scopes for check in runtime.verification_profile.checks] == [
        ("tests", "integrity:protected"),
        ("runtime:entrypoint", "integrity:protected"),
    ]
    assert runtime.verification_profile.target_runtime.eligible_for_task_validation is True
    assert runtime.completion_contract is not None
    assert runtime.completion_contract.required_scopes == (
        "tests",
        "runtime:entrypoint",
        "integrity:protected",
    )
    assert runtime.verification_commands[1].python_module == "sample_app"
    with pytest.raises(WorkspaceError, match="trusted verification paths") as raised:
        runtime.workspace.snapshot_for_write("tests/test_public.py", max_bytes=1_000)
    assert raised.value.code == "protected_path"


def test_configured_policy_rejects_a_second_completion_contract(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    metadata = tmp_path / ".coding-agent"
    metadata.mkdir()
    metadata.joinpath("project.toml").write_text(
        "\n".join(
            (
                "schema_version = 1",
                "[python]",
                "executable = " + json.dumps(str(Path(sys.executable).absolute())),
                "[[verifiers]]",
                'label = "pytest"',
                'type = "pytest"',
                'scopes = ["tests"]',
                "required = true",
                "",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot override configured"):
        build_runtime(
            tmp_path,
            completion_contract=CompletionContract(required_scopes=("tests",)),
        )
