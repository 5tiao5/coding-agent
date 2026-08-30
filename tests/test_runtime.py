"""Tests for application-level repository runtime wiring."""

from __future__ import annotations

import json
from pathlib import Path

from coding_agent.command import CommandPermissionMode
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
