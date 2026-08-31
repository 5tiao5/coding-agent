"""Application wiring for one local repository Agent run."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable, Sequence
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
from coding_agent.completion import (
    CompletionContract,
    TargetRuntime,
    VerificationCheck,
    VerificationProfile,
)
from coding_agent.integrity import check_integrity
from coding_agent.models import VerificationKind
from coding_agent.mutation import MutationSession
from coding_agent.plan import PlanState
from coding_agent.project_config import (
    ResolvedProjectPolicy,
    ResolvedVerifier,
    VerifierType,
    load_project_policy,
)
from coding_agent.run_memory import RunMemory
from coding_agent.tools import (
    CreateDirectoryTool,
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

_POLICY_ANCHOR_PREFIX = "Verification policy fingerprint: "
_POLICY_ANCHOR_MARKER = "Verification policy fingerprint:"


@dataclass(frozen=True, slots=True)
class RuntimeComponents:
    workspace: Workspace
    mutation_session: MutationSession
    run_memory: RunMemory
    plan_state: PlanState
    tools: ToolRegistry
    verification_commands: tuple[VerificationCommandSpec, ...]
    verification_profile: VerificationProfile | None
    completion_contract: CompletionContract | None
    project_policy: ResolvedProjectPolicy
    policy_fingerprint: str


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
    completion_contract: CompletionContract | None = None,
    target_runtime_id: str = "configured-python",
    target_runtime_eligible: bool = True,
) -> RuntimeComponents:
    project_policy = load_project_policy(root)
    protected_mutations = (
        tuple(item.path for item in project_policy.protected_paths)
        if verification_commands is None and project_policy.configured
        else ()
    )
    workspace = Workspace(root, protected_mutation_paths=protected_mutations)
    mutation_session = MutationSession(workspace)
    run_memory = RunMemory()
    plan_state = run_memory.plan_state
    configured_policy_active = verification_commands is None and project_policy.configured
    if configured_policy_active and completion_contract is not None:
        raise ValueError(
            "completion_contract cannot override configured project verification policy"
        )
    integrity_guard: Callable[[], bool] | None
    if configured_policy_active:
        verifiers = tuple(
            _configured_verification_command(item, project_policy)
            for item in project_policy.verifiers
        )
        profile = _configured_verification_profile(project_policy)
        contract = CompletionContract(required_scopes=project_policy.required_scopes)

        def policy_integrity_is_intact() -> bool:
            return check_integrity(project_policy).intact

        integrity_guard = policy_integrity_is_intact
        policy_fingerprint = project_policy.policy_fingerprint
    else:
        verifiers = (
            (default_pytest_verifier(root),)
            if verification_commands is None
            else tuple(verification_commands)
        )
        integrity_guard = None
        profile = None
        contract = None
        policy_fingerprint = ""

    if verifiers and profile is None:
        runtime_id = "unconfigured-python" if verification_commands is None else target_runtime_id
        runtime_eligible = False if verification_commands is None else target_runtime_eligible
        profile = VerificationProfile(
            checks=tuple(
                VerificationCheck(
                    label=spec.label,
                    kind=spec.kind,
                    scopes=(_verification_scope(spec.kind),),
                )
                for spec in verifiers
            ),
            required_labels=tuple(spec.label for spec in verifiers),
            target_runtime=TargetRuntime(
                runtime_id=runtime_id,
                eligible_for_task_validation=runtime_eligible,
            ),
        )
        contract = completion_contract or CompletionContract(required_scopes=("tests",))
    elif not verifiers:
        if completion_contract is not None:
            raise ValueError("completion_contract requires at least one verifier")
        profile = None
        contract = None
    if not policy_fingerprint:
        policy_fingerprint = _override_policy_fingerprint(
            project_policy=project_policy,
            verifiers=verifiers,
            profile=profile,
            contract=contract,
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
                verification_integrity_guard=integrity_guard,
            ),
            CreateDirectoryTool(workspace),
            WriteFileTool(mutation_session),
            ReplaceTextTool(mutation_session),
            UndoChangeTool(mutation_session),
            UpdatePlanTool(plan_state),
        ]
    )
    return RuntimeComponents(
        workspace=workspace,
        mutation_session=mutation_session,
        run_memory=run_memory,
        plan_state=plan_state,
        tools=tools,
        verification_commands=verifiers,
        verification_profile=profile,
        completion_contract=contract,
        project_policy=project_policy,
        policy_fingerprint=policy_fingerprint,
    )


def _configured_verification_command(
    verifier: ResolvedVerifier,
    policy: ResolvedProjectPolicy,
) -> VerificationCommandSpec:
    interpreter = policy.interpreter
    if interpreter is None:  # pragma: no cover - configured policy validation guarantees this.
        raise ValueError("configured verification policy requires an interpreter")
    return VerificationCommandSpec(
        argv=verifier.argv,
        cwd=verifier.cwd,
        kind=verifier.kind,
        label=verifier.label,
        workspace_executable_sha256=interpreter.sha256,
        python_module=(
            verifier.module if verifier.verifier_type is VerifierType.PYTHON_MODULE else None
        ),
    )


def _configured_verification_profile(
    policy: ResolvedProjectPolicy,
) -> VerificationProfile:
    integrity_scope = ("integrity:protected",) if policy.protected_paths else ()
    return VerificationProfile(
        checks=tuple(
            VerificationCheck(
                label=verifier.label,
                kind=verifier.kind,
                scopes=tuple(dict.fromkeys((*verifier.scopes, *integrity_scope))),
            )
            for verifier in policy.verifiers
        ),
        required_labels=policy.required_labels,
        target_runtime=TargetRuntime(
            runtime_id=policy.target_runtime_id,
            eligible_for_task_validation=policy.target_runtime_eligible,
        ),
    )


def _override_policy_fingerprint(
    *,
    project_policy: ResolvedProjectPolicy,
    verifiers: tuple[VerificationCommandSpec, ...],
    profile: VerificationProfile | None,
    contract: CompletionContract | None,
) -> str:
    payload = {
        "project_policy": project_policy.policy_fingerprint,
        "verifiers": [
            {
                "argv": list(spec.argv),
                "cwd": spec.cwd,
                "kind": spec.kind.value,
                "label": spec.label,
                "executable_sha256": spec.workspace_executable_sha256,
                "python_module": spec.python_module,
            }
            for spec in verifiers
        ],
        "target_runtime": (
            None
            if profile is None
            else {
                "id": profile.target_runtime.runtime_id,
                "eligible": profile.target_runtime.eligible_for_task_validation,
            }
        ),
        "required_scopes": [] if contract is None else list(contract.required_scopes),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verification_scope(kind: VerificationKind) -> str:
    if kind is VerificationKind.TEST:
        return "tests"
    if kind is VerificationKind.BUILD:
        return "build"
    return "checks"


def system_prompt_for(
    verifiers: Sequence[VerificationCommandSpec],
    *,
    policy_fingerprint: str | None = None,
    target_runtime_eligible: bool | None = None,
) -> str:
    lines = [DEFAULT_SYSTEM_PROMPT]
    if not verifiers:
        lines.append(
            "No trusted verification capability is configured; be explicit that completion "
            "will remain unverified."
        )
        prompt = "\n\n".join(lines)
        return _append_policy_anchor(prompt, policy_fingerprint)

    lines.append(
        "Trusted verification capabilities are listed below. Use the exact argv and cwd; "
        "nearby commands may run but cannot issue verification evidence."
    )
    if target_runtime_eligible is False:
        lines.append(
            "These checks may pass, but they cannot validate task completion until an "
            "explicit project interpreter policy is configured. Report that limit plainly."
        )
    lines.extend(
        f"- {spec.label}: cwd={json.dumps(spec.cwd)}, "
        f"argv={json.dumps(list(spec.argv), ensure_ascii=False)}"
        for spec in verifiers
    )
    return _append_policy_anchor("\n".join(lines), policy_fingerprint)


def _append_policy_anchor(prompt: str, policy_fingerprint: str | None) -> str:
    if policy_fingerprint is None:
        return prompt
    if len(policy_fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in policy_fingerprint
    ):
        raise ValueError("policy fingerprint must be lowercase SHA-256")
    return prompt + "\n" + _POLICY_ANCHOR_PREFIX + policy_fingerprint


def policy_fingerprint_from_prompt(prompt: str) -> str | None:
    """Extract one canonical host anchor from a checkpoint system prompt.

    Runtime-limit text may be appended after the anchor.  Duplicate or malformed anchors are
    rejected so a checkpoint cannot smuggle an ambiguous policy identity across resume.
    """

    candidates = [line for line in prompt.splitlines() if _POLICY_ANCHOR_MARKER in line]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError("system prompt contains multiple policy fingerprints")
    if not candidates[0].startswith(_POLICY_ANCHOR_PREFIX):
        raise ValueError("system prompt contains a malformed policy fingerprint")
    fingerprint = candidates[0].removeprefix(_POLICY_ANCHOR_PREFIX)
    if len(fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint
    ):
        raise ValueError("system prompt contains an invalid policy fingerprint")
    return fingerprint
