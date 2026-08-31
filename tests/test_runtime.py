"""Tests for application-level repository runtime wiring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from coding_agent.command import (
    CommandPermissionMode,
    VerificationCommandSpec,
)
from coding_agent.models import ToolCall, VerificationKind
from coding_agent.runtime import build_runtime, system_prompt_for


def test_runtime_exposes_the_complete_m4_tool_surface(tmp_path: Path) -> None:
    runtime = build_runtime(tmp_path, permission_mode=CommandPermissionMode.AUTO)

    assert [spec.name for spec in runtime.tools.specs()] == [
        "list_files",
        "read_file",
        "search_text",
        "run_command",
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
    assert runtime.verification_profile.target_runtime.runtime_id == "configured-python"
    assert runtime.verification_profile.target_runtime.eligible_for_task_validation is True
    assert runtime.completion_contract.required_scopes == ("tests",)
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
