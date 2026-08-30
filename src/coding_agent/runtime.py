"""Application wiring for one local repository Agent run."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from coding_agent.agent import DEFAULT_SYSTEM_PROMPT
from coding_agent.approval import CommandApprover
from coding_agent.command import (
    CommandPermissionMode,
    CommandPolicy,
    VerificationCommandSpec,
    executable_sha256,
)
from coding_agent.models import VerificationKind
from coding_agent.mutation import MutationSession
from coding_agent.plan import PlanState
from coding_agent.tools import (
    ListFilesTool,
    ReadFileTool,
    ReplaceTextTool,
    RunCommandTool,
    SearchTextTool,
    ToolRegistry,
    UndoChangeTool,
    UpdatePlanTool,
    WriteFileTool,
)
from coding_agent.workspace import Workspace


@dataclass(frozen=True, slots=True)
class RuntimeComponents:
    workspace: Workspace
    mutation_session: MutationSession
    plan_state: PlanState
    tools: ToolRegistry
    verification_commands: tuple[VerificationCommandSpec, ...]


def default_pytest_verifier(workspace_root: Path | None = None) -> VerificationCommandSpec:
    # Resolving a POSIX virtualenv launcher dereferences it to the base interpreter,
    # which loses the virtualenv and its installed pytest package when executed.
    executable = Path(os.path.abspath(sys.executable))
    if not executable.is_file():
        raise ValueError("verification executable is unavailable")
    workspace_digest: str | None = None
    if workspace_root is not None:
        resolved_root = workspace_root.resolve(strict=True)
        if executable == resolved_root or resolved_root in executable.parents:
            workspace_digest = executable_sha256(executable)
    return VerificationCommandSpec(
        argv=(
            str(executable),
            "-I",
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
        ),
        cwd=".",
        kind=VerificationKind.TEST,
        label="pytest",
        workspace_executable_sha256=workspace_digest,
    )


def build_runtime(
    root: Path,
    *,
    permission_mode: CommandPermissionMode = CommandPermissionMode.SAFE,
    approver: CommandApprover | None = None,
    verification_commands: Sequence[VerificationCommandSpec] | None = None,
) -> RuntimeComponents:
    workspace = Workspace(root)
    mutation_session = MutationSession(workspace)
    plan_state = PlanState()
    verifiers = (
        (default_pytest_verifier(root),)
        if verification_commands is None
        else tuple(verification_commands)
    )
    tools = ToolRegistry(
        [
            ListFilesTool(workspace),
            ReadFileTool(workspace),
            SearchTextTool(workspace),
            RunCommandTool(
                workspace,
                policy=CommandPolicy(
                    permission_mode,
                    verification_commands=verifiers,
                ),
                approver=approver,
            ),
            WriteFileTool(mutation_session),
            ReplaceTextTool(mutation_session),
            UndoChangeTool(mutation_session),
            UpdatePlanTool(plan_state),
        ]
    )
    return RuntimeComponents(
        workspace=workspace,
        mutation_session=mutation_session,
        plan_state=plan_state,
        tools=tools,
        verification_commands=verifiers,
    )


def system_prompt_for(verifiers: Sequence[VerificationCommandSpec]) -> str:
    lines = [DEFAULT_SYSTEM_PROMPT]
    if not verifiers:
        lines.append(
            "No trusted verification capability is configured; be explicit that completion "
            "will remain unverified."
        )
        return "\n\n".join(lines)

    lines.append(
        "Trusted verification capabilities are listed below. Use the exact argv and cwd; "
        "nearby commands may run but cannot issue verification evidence."
    )
    lines.extend(
        f"- {spec.label}: cwd={json.dumps(spec.cwd)}, "
        f"argv={json.dumps(list(spec.argv), ensure_ascii=False)}"
        for spec in verifiers
    )
    return "\n".join(lines)
