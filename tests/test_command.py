"""Safety, observation, and verification tests for local command execution."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from pydantic import TypeAdapter

import coding_agent._command_process as process_module
import coding_agent.command as command_module
from coding_agent.approval import CommandApprovalRequest, CommandApprover
from coding_agent.command import (
    CommandClass,
    CommandClassification,
    CommandEnvironmentProfile,
    CommandPermissionMode,
    CommandPolicy,
    CommandRequest,
    CommandResult,
    CommandStatus,
    LocalCommandRunner,
    VerificationCommandSpec,
    decode_command_output,
    executable_sha256,
)
from coding_agent.models import ToolCall, ToolExecution, VerificationKind, VerificationSignal
from coding_agent.tools.base import ToolRegistry
from coding_agent.tools.command import RunCommandTool
from coding_agent.workspace import ExpectedPathKind, Workspace, WorkspacePath

_PAYLOAD_ADAPTER = TypeAdapter(dict[str, object])


class RecordingRunner:
    def __init__(self, result: CommandResult) -> None:
        self.result = result
        self.requests: list[CommandRequest] = []

    def run(self, request: CommandRequest) -> CommandResult:
        self.requests.append(request)
        return self.result


class RecordingApprover:
    def __init__(self, decision: bool) -> None:
        self.decision = decision
        self.requests: list[CommandApprovalRequest] = []

    def approve(self, request: CommandApprovalRequest) -> bool:
        self.requests.append(request)
        return self.decision


class InterruptingProcess:
    def __init__(self) -> None:
        self.stdout = io.BytesIO(b"partial output\n")

    def wait(self, *, timeout: float) -> int:
        del timeout
        raise KeyboardInterrupt


class InterruptThenFailContainment:
    def __init__(self) -> None:
        self.cleanup_calls = 0
        self.closed = False

    def has_active_descendants(self, process: subprocess.Popen[bytes]) -> bool | None:
        del process
        return False

    def terminate_and_confirm(
        self,
        process: subprocess.Popen[bytes],
        grace_seconds: float,
    ) -> str | None:
        del process, grace_seconds
        self.cleanup_calls += 1
        if self.cleanup_calls == 1:
            raise KeyboardInterrupt
        return "simulated cleanup confirmation failure"

    def close(self) -> None:
        self.closed = True


class LongDisplayWorkspace(Workspace):
    def __init__(self, root: Path) -> None:
        self._display_root = root

    @property
    def root(self) -> Path:
        return self._display_root

    def resolve(
        self,
        user_path: str,
        *,
        expected: ExpectedPathKind = "any",
        allow_ignored: bool = False,
        allow_sensitive: bool = False,
    ) -> WorkspacePath:
        del user_path, expected, allow_ignored, allow_sensitive
        return WorkspacePath(
            path=self._display_root,
            relative=("very-long-directory/" * 60)[:1_000],
        )


def _result(
    *,
    exit_code: int = 0,
    output: bytes = b"ok\n",
) -> CommandResult:
    return CommandResult(
        status=CommandStatus.EXITED,
        exit_code=exit_code,
        output=output,
        total_output_bytes=len(output),
        captured_output_bytes=len(output),
        output_truncated=False,
    )


def _execute(
    root: Path,
    arguments: dict[str, object],
    *,
    runner: RecordingRunner | LocalCommandRunner,
    permission_mode: CommandPermissionMode = CommandPermissionMode.SAFE,
    verification_commands: tuple[VerificationCommandSpec, ...] = (),
    integrity_guard: Callable[[], bool] | None = None,
    approver: CommandApprover | None = None,
    max_output_chars: int = 16_000,
) -> ToolExecution:
    tool = RunCommandTool(
        Workspace(root),
        runner=runner,
        policy=CommandPolicy(
            permission_mode,
            verification_commands=verification_commands,
        ),
        approver=approver,
        verification_integrity_guard=integrity_guard,
        max_output_chars=max_output_chars,
    )
    return ToolRegistry([tool]).execute(
        ToolCall(id="command-1", name="run_command", arguments=arguments)
    )


def _trusted_spec(
    argv: list[str] | tuple[str, ...],
    kind: VerificationKind,
    label: str,
    *,
    cwd: str = ".",
) -> VerificationCommandSpec:
    absolute_argv = (str(Path(sys.executable).with_name(Path(argv[0]).name)), *argv[1:])
    return VerificationCommandSpec(
        argv=absolute_argv,
        cwd=cwd,
        kind=kind,
        label=label,
    )


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    root.joinpath("src").mkdir()
    return root


@pytest.mark.parametrize(
    ("argv", "kind", "label"),
    [
        (["uv", "run", "pytest"], VerificationKind.TEST, "pytest"),
        (
            ["python", "-m", "unittest"],
            VerificationKind.TEST,
            "unittest",
        ),
        (
            ["ruff", "format", "--check", "."],
            VerificationKind.CHECK,
            "ruff format --check",
        ),
        (
            ["cargo", "build"],
            VerificationKind.BUILD,
            "cargo build",
        ),
    ],
)
def test_host_registered_exact_argv_can_produce_verification(
    argv: list[str],
    kind: VerificationKind,
    label: str,
) -> None:
    spec = _trusted_spec(argv, kind, label)
    classification = CommandPolicy(
        verification_commands=(spec,),
    ).classify(spec.argv)

    assert classification.command_class is CommandClass.VERIFIER
    assert classification.verification_kind is kind
    assert classification.verification_label == label


@pytest.mark.parametrize(
    ("argv", "kind", "label"),
    [
        (["mypy", "src"], VerificationKind.CHECK, "mypy"),
        (["pyright"], VerificationKind.CHECK, "pyright"),
        (["ruff", "check", "."], VerificationKind.CHECK, "ruff check"),
        (["go", "test", "./..."], VerificationKind.TEST, "go test"),
        (["npm", "test"], VerificationKind.TEST, "npm test"),
        (["pnpm", "run", "lint"], VerificationKind.CHECK, "pnpm lint"),
        (["yarn", "run", "build"], VerificationKind.BUILD, "yarn build"),
        (["make", "typecheck"], VerificationKind.CHECK, "make typecheck"),
        (["ninja", "compile"], VerificationKind.BUILD, "ninja compile"),
    ],
)
def test_policy_recognizes_supported_verification_families(
    argv: list[str],
    kind: VerificationKind,
    label: str,
) -> None:
    spec = _trusted_spec(argv, kind, label)
    classification = CommandPolicy(verification_commands=(spec,)).classify(spec.argv)

    assert classification.command_class is CommandClass.VERIFIER
    assert classification.verification_kind is kind
    assert classification.verification_label == label


@pytest.mark.parametrize(
    "argv",
    [
        ["pytest", "-q"],
        ["ruff", "check", "."],
        ["git", "status"],
        ["rg", "needle"],
        ["python", "--version"],
    ],
)
def test_unregistered_commands_never_receive_trusted_or_read_only_status(
    argv: list[str],
) -> None:
    with pytest.raises(command_module.CommandError) as raised:
        CommandPolicy().classify(argv)
    assert raised.value.code == "command_approval_required"

    classification = CommandPolicy(CommandPermissionMode.AUTO).classify(argv)
    assert classification.command_class is CommandClass.GENERAL
    assert classification.verification_kind is None


def test_safe_policy_accepts_only_a_host_owned_one_call_approval() -> None:
    policy = CommandPolicy()

    approved = policy.classify(["git", "status"], approved=True)

    assert approved.command_class is CommandClass.GENERAL
    with pytest.raises(command_module.CommandError) as raised:
        policy.classify(["git", "status"])
    assert raised.value.code == "command_approval_required"


def test_interactive_approval_runs_the_exact_general_command(repository: Path) -> None:
    runner = RecordingRunner(_result())
    approver = RecordingApprover(True)

    execution = _execute(
        repository,
        {"argv": ["git", "status"], "cwd": "src"},
        runner=runner,
        approver=approver,
    )

    assert execution.ok is True
    assert execution.control.invalidates_verification is True
    assert approver.requests == [CommandApprovalRequest(argv=("git", "status"), cwd="src")]
    assert runner.requests[0].argv == ("git", "status")


def test_denied_general_command_never_reaches_the_runner(repository: Path) -> None:
    runner = RecordingRunner(_result())
    approver = RecordingApprover(False)

    execution = _execute(
        repository,
        {"argv": ["git", "status"]},
        runner=runner,
        approver=approver,
    )

    assert execution.ok is False
    assert execution.error_code == "command_denied"
    assert approver.requests == [CommandApprovalRequest(argv=("git", "status"), cwd=".")]
    assert runner.requests == []


def test_verification_capability_binds_working_directory_and_exact_arguments() -> None:
    spec = _trusted_spec(["pytest", "-q"], VerificationKind.TEST, "pytest")
    policy = CommandPolicy(
        CommandPermissionMode.AUTO,
        verification_commands=(spec,),
    )

    assert policy.classify(spec.argv).command_class is CommandClass.VERIFIER
    assert policy.classify((*spec.argv, "tests/unit")).command_class is CommandClass.GENERAL
    assert policy.classify(spec.argv, cwd="src").command_class is CommandClass.GENERAL


def test_verification_preflight_fails_closed_for_near_or_unrecognized_commands() -> None:
    spec = _trusted_spec(["python", "-m", "pytest", "-q"], VerificationKind.TEST, "pytest")
    safe_policy = CommandPolicy(verification_commands=(spec,))

    assert safe_policy.is_verification_call(spec.argv, cwd=".") is True
    assert safe_policy.is_verification_call((*spec.argv, "tests/unit"), cwd=".") is False
    assert safe_policy.is_verification_call(spec.argv, cwd="src") is False
    assert safe_policy.is_verification_call((str(Path(sys.executable)), "-c", "pass")) is False
    assert safe_policy.is_verification_call(("cmd.exe", "/c", "pytest")) is False


def test_registry_verification_preflight_never_executes_or_requests_approval(
    repository: Path,
) -> None:
    spec = _trusted_spec(["python", "-m", "pytest", "-q"], VerificationKind.TEST, "pytest")
    runner = RecordingRunner(_result())
    approver = RecordingApprover(True)
    registry = ToolRegistry(
        [
            RunCommandTool(
                Workspace(repository),
                runner=runner,
                policy=CommandPolicy(verification_commands=(spec,)),
                approver=approver,
            )
        ]
    )

    exact = ToolCall(
        id="verify-exact",
        name="run_command",
        arguments={"argv": list(spec.argv)},
    )
    near = exact.model_copy(
        update={
            "id": "verify-near",
            "arguments": {"argv": [*spec.argv, "tests/unit"], "cwd": "."},
        }
    )
    extra_field = exact.model_copy(
        update={
            "id": "verify-extra",
            "arguments": {"argv": list(spec.argv), "cwd": ".", "unrecognized": True},
        }
    )

    assert registry.is_verification_call(exact) is True
    assert registry.is_verification_call(near) is False
    assert registry.is_verification_call(extra_field) is False
    assert (
        registry.is_verification_call(
            ToolCall(id="empty-argv", name="run_command", arguments={"argv": []})
        )
        is False
    )
    assert (
        registry.is_verification_call(
            ToolCall(
                id="missing-cwd",
                name="run_command",
                arguments={"argv": list(spec.argv), "cwd": "missing"},
            )
        )
        is False
    )
    assert (
        registry.is_verification_call(
            ToolCall(id="unknown", name="unknown_tool", arguments={"argv": list(spec.argv)})
        )
        is False
    )
    assert runner.requests == []
    assert approver.requests == []


@pytest.mark.parametrize(
    "argv",
    [
        ["python", "-c", "pass"],
        ["python", "scripts/check.py"],
        ["echo", "all checks passed"],
    ],
)
def test_generic_or_interpreted_commands_cannot_be_registered_as_verifiers(
    argv: list[str],
) -> None:
    spec = _trusted_spec(argv, VerificationKind.TEST, "misconfigured verifier")
    classification = CommandPolicy(
        CommandPermissionMode.AUTO,
        verification_commands=(spec,),
    ).classify(spec.argv)

    assert classification.command_class is CommandClass.GENERAL
    assert classification.verification_kind is None


def test_typed_python_module_smoke_is_exact_and_cannot_generalize() -> None:
    executable = str(Path(sys.executable).absolute())
    typed = VerificationCommandSpec(
        argv=(executable, "-B", "-m", "sample_app"),
        cwd=".",
        kind=VerificationKind.CHECK,
        label="module-smoke",
        python_module="sample_app",
    )
    generic = VerificationCommandSpec(
        argv=typed.argv,
        cwd=".",
        kind=VerificationKind.CHECK,
        label="generic-module",
    )

    typed_classification = CommandPolicy(
        CommandPermissionMode.AUTO,
        verification_commands=(typed,),
    ).classify(typed.argv)
    generic_classification = CommandPolicy(
        CommandPermissionMode.AUTO,
        verification_commands=(generic,),
    ).classify(generic.argv)

    assert typed_classification.command_class is CommandClass.VERIFIER
    assert typed_classification.verification_kind is VerificationKind.CHECK
    assert generic_classification.command_class is CommandClass.GENERAL
    with pytest.raises(ValueError, match="exact typed smoke-test"):
        VerificationCommandSpec(
            argv=(*typed.argv, "--extra"),
            cwd=".",
            kind=VerificationKind.CHECK,
            label="broadened-module",
            python_module="sample_app",
        )


@pytest.mark.parametrize("executable_name", ["python3.11", "python3.12", "pypy3.10"])
def test_versioned_python_executables_keep_trusted_semantics(
    executable_name: str,
) -> None:
    host_executable = Path(sys.executable)
    executable = str(host_executable.with_name(executable_name + host_executable.suffix).absolute())
    pytest_spec = VerificationCommandSpec(
        argv=(executable, "-m", "pytest", "-q"),
        cwd=".",
        kind=VerificationKind.TEST,
        label="pytest",
    )
    module_spec = VerificationCommandSpec(
        argv=(executable, "-B", "-m", "sample_app"),
        cwd=".",
        kind=VerificationKind.CHECK,
        label="module-smoke",
        python_module="sample_app",
    )
    policy = CommandPolicy(
        CommandPermissionMode.AUTO,
        verification_commands=(pytest_spec, module_spec),
    )

    pytest_classification = policy.classify(pytest_spec.argv)
    module_classification = policy.classify(module_spec.argv)

    assert pytest_classification.command_class is CommandClass.VERIFIER
    assert pytest_classification.verification_kind is VerificationKind.TEST
    assert module_classification.command_class is CommandClass.VERIFIER
    assert module_classification.verification_kind is VerificationKind.CHECK


def test_verification_spec_rejects_malformed_trust_boundaries() -> None:
    executable = str(Path(sys.executable).absolute())

    with pytest.raises(ValueError, match="argv cannot be empty"):
        VerificationCommandSpec(
            argv=(),
            cwd=".",
            kind=VerificationKind.TEST,
            label="pytest",
        )
    with pytest.raises(ValueError, match="workspace-relative"):
        VerificationCommandSpec(
            argv=(executable, "-m", "pytest"),
            cwd="../outside",
            kind=VerificationKind.TEST,
            label="pytest",
        )
    with pytest.raises(ValueError, match="1-120"):
        VerificationCommandSpec(
            argv=(executable, "-m", "pytest"),
            cwd=".",
            kind=VerificationKind.TEST,
            label=" ",
        )
    with pytest.raises(ValueError, match="SHA-256"):
        VerificationCommandSpec(
            argv=(executable, "-m", "pytest"),
            cwd=".",
            kind=VerificationKind.TEST,
            label="pytest",
            workspace_executable_sha256="0" * 63,
        )
    with pytest.raises(ValueError, match="importable module"):
        VerificationCommandSpec(
            argv=(executable, "-B", "-m", "bad-module"),
            cwd=".",
            kind=VerificationKind.CHECK,
            label="module-smoke",
            python_module="bad-module",
        )
    with pytest.raises(ValueError, match="printable"):
        VerificationCommandSpec(
            argv=(executable, "-m", "pytest"),
            cwd=".",
            kind=VerificationKind.TEST,
            label="pytest\n",
        )


def test_integrity_guard_rejects_verifier_before_process_start(repository: Path) -> None:
    spec = _trusted_spec(["pytest", "-q"], VerificationKind.TEST, "pytest")
    runner = RecordingRunner(_result())

    execution = _execute(
        repository,
        {"argv": list(spec.argv), "cwd": "."},
        runner=runner,
        permission_mode=CommandPermissionMode.AUTO,
        verification_commands=(spec,),
        integrity_guard=lambda: False,
    )

    assert runner.requests == []
    assert execution.ok is True
    assert execution.control.verification is VerificationSignal.FAILED
    assert execution.control.invalidates_verification is True
    assert execution.metadata["integrity_phase"] == "before"
    assert execution.output is not None
    assert execution.output.startswith("Status: trusted verification rejected")


def test_integrity_guard_rejects_large_success_after_process_without_hiding_reason(
    repository: Path,
) -> None:
    spec = _trusted_spec(["pytest", "-q"], VerificationKind.TEST, "pytest")
    runner = RecordingRunner(_result(output=b"x" * 4_000))
    checks = iter((True, False))

    execution = _execute(
        repository,
        {"argv": list(spec.argv), "cwd": "."},
        runner=runner,
        permission_mode=CommandPermissionMode.AUTO,
        verification_commands=(spec,),
        integrity_guard=lambda: next(checks),
        max_output_chars=512,
    )

    assert len(runner.requests) == 1
    assert execution.control.verification is VerificationSignal.FAILED
    assert execution.metadata["integrity_phase"] == "after"
    assert execution.output is not None
    assert execution.output.startswith("Status: trusted verification rejected")
    assert "project verification policy integrity changed" in execution.output
    assert len(execution.output) <= 512


def test_workspace_owned_executable_cannot_issue_verification(repository: Path) -> None:
    executable = repository / ("pytest.exe" if os.name == "nt" else "pytest")
    executable.write_bytes(b"mutable launcher")
    spec = VerificationCommandSpec(
        argv=(str(executable), "-q"),
        cwd=".",
        kind=VerificationKind.TEST,
        label="workspace pytest",
    )

    policy = CommandPolicy(
        CommandPermissionMode.AUTO,
        verification_commands=(spec,),
    )
    classification = policy.classify(spec.argv, workspace_root=repository)

    assert classification.command_class is CommandClass.GENERAL
    assert classification.verification_kind is None
    assert policy.is_verification_call(spec.argv, workspace_root=repository) is False


def test_hash_bound_workspace_verifier_is_valid_only_while_unchanged(repository: Path) -> None:
    executable = repository / ("pytest.exe" if os.name == "nt" else "pytest")
    executable.write_bytes(b"trusted launcher")
    spec = VerificationCommandSpec(
        argv=(str(executable), "-q"),
        cwd=".",
        kind=VerificationKind.TEST,
        label="workspace pytest",
        workspace_executable_sha256=executable_sha256(executable),
    )
    policy = CommandPolicy(
        CommandPermissionMode.AUTO,
        verification_commands=(spec,),
    )

    assert (
        policy.classify(spec.argv, workspace_root=repository).command_class is CommandClass.VERIFIER
    )
    assert policy.is_verification_call(spec.argv, workspace_root=repository) is True

    executable.write_bytes(b"rewritten launcher")
    classification = policy.classify(spec.argv, workspace_root=repository)
    assert classification.command_class is CommandClass.GENERAL
    assert policy.is_verification_call(spec.argv, workspace_root=repository) is False


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable symlink policy")
def test_workspace_symlink_verifier_requires_an_unchanged_hash(repository: Path) -> None:
    target = repository.parent / "external-pytest"
    target.write_bytes(b"trusted launcher")
    launcher = repository / "pytest"
    launcher.symlink_to(target)
    unbound = VerificationCommandSpec(
        argv=(str(launcher), "-q"),
        cwd=".",
        kind=VerificationKind.TEST,
        label="workspace pytest symlink",
    )
    bound = VerificationCommandSpec(
        argv=unbound.argv,
        cwd=unbound.cwd,
        kind=unbound.kind,
        label=unbound.label,
        workspace_executable_sha256=executable_sha256(launcher),
    )

    assert (
        CommandPolicy(CommandPermissionMode.AUTO, verification_commands=(unbound,))
        .classify(unbound.argv, workspace_root=repository)
        .command_class
        is CommandClass.GENERAL
    )

    policy = CommandPolicy(
        CommandPermissionMode.AUTO,
        verification_commands=(bound,),
    )
    assert (
        policy.classify(bound.argv, workspace_root=repository).command_class
        is CommandClass.VERIFIER
    )

    target.write_bytes(b"rewritten launcher")
    classification = policy.classify(bound.argv, workspace_root=repository)
    assert classification.command_class is CommandClass.GENERAL


def test_verification_capability_requires_an_absolute_executable() -> None:
    with pytest.raises(ValueError, match="absolute"):
        VerificationCommandSpec(
            argv=("pytest", "-q"),
            cwd=".",
            kind=VerificationKind.TEST,
            label="pytest",
        )


@pytest.mark.parametrize(
    "argv",
    [
        ["git", "clean", "-fd"],
        ["git", "-C", "repository", "clean", "-fd"],
        ["git", "reset", "--hard"],
        ["mkfs.ext4", "/dev/example"],
        ["cmd.exe", "/c", "del important.txt"],
        ["bash", "-c", "rm important.txt"],
        ["script.cmd", "argument"],
    ],
)
def test_policy_blocks_destructive_variants(argv: list[str]) -> None:
    with pytest.raises(command_module.CommandError, match="blocked|not allowed"):
        CommandPolicy().classify(argv)


def test_policy_handles_empty_version_and_non_verifying_commands() -> None:
    with pytest.raises(command_module.CommandError, match="cannot be empty"):
        CommandPolicy().classify([])

    auto = CommandPolicy(CommandPermissionMode.AUTO)
    assert auto.classify(["python", "--version"]).command_class is CommandClass.GENERAL
    assert auto.classify(["tool.exe", "--version"]).command_class is CommandClass.GENERAL
    assert auto.classify(["ruff", "format", "."]).command_class is CommandClass.GENERAL
    assert auto.classify(["npm", "install"]).command_class is CommandClass.GENERAL


@pytest.mark.parametrize(
    "argv",
    [
        ["pytest", "--version"],
        ["pytest", "--help"],
        ["pytest", "-h"],
        ["python", "-m", "pytest", "--help"],
        ["python", "-V"],
        ["uv", "run", "python", "-V"],
        ["ruff", "check", "--help"],
        ["cargo", "test", "--help"],
        ["cargo", "test", "--no-run"],
        ["pytest", "--fixtures"],
        ["pytest", "--markers"],
        ["pytest", "--setup-plan"],
        ["pytest", "--setup-only"],
        ["pytest", "--cache-show"],
        ["python", "-m", "pytest", "--fixtures-per-test"],
        ["python", "-I", "-m", "pytest", "--fixtures"],
    ],
)
def test_information_queries_never_become_verification_evidence(argv: list[str]) -> None:
    spec = _trusted_spec(argv, VerificationKind.TEST, "misconfigured verifier")
    classification = CommandPolicy(
        CommandPermissionMode.AUTO,
        verification_commands=(spec,),
    ).classify(spec.argv)

    assert classification.command_class is CommandClass.GENERAL
    assert classification.verification_kind is None
    assert classification.verification_label is None


@pytest.mark.parametrize(
    "argv",
    [
        ["pytest", "-v"],
        ["uv", "run", "pytest", "-v"],
    ],
)
def test_pytest_v_flags_are_not_mistaken_for_version_queries(argv: list[str]) -> None:
    spec = _trusted_spec(argv, VerificationKind.TEST, "pytest")
    classification = CommandPolicy(verification_commands=(spec,)).classify(spec.argv)

    assert classification.command_class is CommandClass.VERIFIER
    assert classification.verification_kind is VerificationKind.TEST


@pytest.mark.parametrize(
    "argv",
    [
        ["python", "script.py", "--version"],
        ["tool.exe", "--help", "extra"],
        ["pytest", "--collect-only", "--help"],
    ],
)
def test_information_flags_do_not_hide_executable_work(argv: list[str]) -> None:
    with pytest.raises(command_module.CommandError) as raised:
        CommandPolicy().classify(argv)

    assert raised.value.code == "command_approval_required"


@pytest.mark.parametrize(
    "argv",
    [
        ["git", "diff", "--output=report.patch"],
        ["git", "log", "--output", "history.txt"],
    ],
)
def test_read_only_git_commands_cannot_write_an_output_file(argv: list[str]) -> None:
    with pytest.raises(command_module.CommandError) as raised:
        CommandPolicy().classify(argv)

    assert raised.value.code == "command_approval_required"
    assert (
        CommandPolicy(CommandPermissionMode.AUTO).classify(argv).command_class
        is CommandClass.GENERAL
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["pytest", "--collect-only"],
        ["pytest", "--co"],
        ["python", "-m", "pytest", "--collect-only"],
        ["uv", "run", "pytest", "--collect-only"],
    ],
)
def test_pytest_collection_is_general_not_verification(argv: list[str]) -> None:
    spec = _trusted_spec(argv, VerificationKind.TEST, "misconfigured pytest")
    policy = CommandPolicy(verification_commands=(spec,))
    with pytest.raises(command_module.CommandError) as raised:
        policy.classify(spec.argv)
    assert raised.value.code == "command_approval_required"

    classification = CommandPolicy(
        CommandPermissionMode.AUTO,
        verification_commands=(spec,),
    ).classify(spec.argv)
    assert classification.command_class is CommandClass.GENERAL
    assert classification.verification_kind is None


def test_destructive_policy_denial_never_starts_a_process(repository: Path) -> None:
    runner = RecordingRunner(_result())

    execution = _execute(repository, {"argv": ["rm", "-rf", "."]}, runner=runner)

    assert execution.ok is False
    assert execution.error_code == "command_denied"
    assert execution.metadata == {"reason": "destructive_executable"}
    assert runner.requests == []


def test_safe_mode_requires_approval_before_a_general_command_starts(repository: Path) -> None:
    runner = RecordingRunner(_result())

    execution = _execute(
        repository,
        {"argv": ["python", "script.py"]},
        runner=runner,
    )

    assert execution.ok is False
    assert execution.error_code == "command_approval_required"
    assert execution.metadata == {
        "permission_mode": "safe",
        "reason": "general_command",
    }
    assert runner.requests == []


def test_auto_mode_allows_general_commands_but_marks_their_effect_unknown(
    repository: Path,
) -> None:
    runner = RecordingRunner(_result())

    execution = _execute(
        repository,
        {"argv": ["python", "script.py"]},
        runner=runner,
        permission_mode=CommandPermissionMode.AUTO,
    )

    assert execution.ok is True
    assert execution.control.invalidates_verification is True
    assert len(runner.requests) == 1


def test_invalid_cwd_never_reaches_the_process_runner(repository: Path) -> None:
    runner = RecordingRunner(_result())

    execution = _execute(
        repository,
        {"argv": [sys.executable, "--version"], "cwd": "../outside"},
        runner=runner,
    )

    assert execution.ok is False
    assert execution.error_code == "invalid_path"
    assert runner.requests == []


def test_arguments_reject_control_characters_before_execution(repository: Path) -> None:
    runner = RecordingRunner(_result())

    execution = _execute(
        repository,
        {"argv": [sys.executable, "line-one\nline-two"]},
        runner=runner,
    )

    assert execution.ok is False
    assert execution.error_code == "invalid_arguments"
    assert runner.requests == []


@pytest.mark.parametrize(
    ("argv", "exit_code", "verification", "invalidates"),
    [
        (["pytest"], 0, VerificationSignal.PASSED, False),
        (["pytest"], 1, VerificationSignal.FAILED, False),
        (["git", "status"], 1, None, True),
        (["python", "script.py"], 0, None, True),
    ],
)
def test_command_control_facts_come_from_policy_and_exit_status(
    repository: Path,
    argv: list[str],
    exit_code: int,
    verification: VerificationSignal | None,
    invalidates: bool,
) -> None:
    runner = RecordingRunner(_result(exit_code=exit_code))
    actual_argv: list[str] | tuple[str, ...] = argv
    verification_commands: tuple[VerificationCommandSpec, ...] = ()
    if verification is not None:
        spec = _trusted_spec(argv, VerificationKind.TEST, "pytest")
        actual_argv = spec.argv
        verification_commands = (spec,)

    execution = _execute(
        repository,
        {"argv": actual_argv},
        runner=runner,
        permission_mode=CommandPermissionMode.AUTO,
        verification_commands=verification_commands,
    )

    assert execution.ok is True
    expected_profile = (
        CommandEnvironmentProfile.VERIFIER
        if verification is not None
        else CommandEnvironmentProfile.SANITIZED
    )
    assert runner.requests[0].environment_profile is expected_profile
    assert execution.control.verification is verification
    assert execution.control.invalidates_verification is invalidates
    if verification is None:
        assert execution.control.verification_kind is None
        assert execution.control.verification_label is None
    else:
        assert execution.control.verification_kind is VerificationKind.TEST
        assert execution.control.verification_label == "pytest"
    payload = _PAYLOAD_ADAPTER.validate_json(execution.as_message_content())
    assert "control" not in payload


def test_timeout_keeps_partial_output_and_marks_failed_verification(repository: Path) -> None:
    partial = b"collection started\n"
    runner = RecordingRunner(
        CommandResult(
            status=CommandStatus.TIMED_OUT,
            exit_code=None,
            output=partial,
            total_output_bytes=len(partial),
            captured_output_bytes=len(partial),
            output_truncated=False,
        )
    )

    spec = _trusted_spec(["pytest"], VerificationKind.TEST, "pytest")
    execution = _execute(
        repository,
        {"argv": spec.argv, "timeout_seconds": 5},
        runner=runner,
        verification_commands=(spec,),
    )

    assert execution.ok is True
    assert execution.metadata["status"] == "timed_out"
    assert execution.metadata["exit_code"] is None
    assert "collection started" in str(execution.output)
    assert execution.control.verification is VerificationSignal.FAILED
    assert execution.control.invalidates_verification is False


def test_process_control_failure_forces_a_terminal_stop(repository: Path) -> None:
    runner = RecordingRunner(
        CommandResult(
            status=CommandStatus.CONTROL_FAILED,
            exit_code=None,
            output=b"started\n",
            total_output_bytes=8,
            captured_output_bytes=8,
            output_truncated=False,
            terminal_reason="descendant could not be terminated",
        )
    )

    execution = _execute(
        repository,
        {"argv": ["python", "worker.py"]},
        runner=runner,
        permission_mode=CommandPermissionMode.AUTO,
    )

    assert execution.ok is True
    assert execution.control.invalidates_verification is True
    assert execution.control.terminal_stop is True
    assert execution.control.terminal_reason == "descendant could not be terminated"
    assert "Safety stop: descendant could not be terminated" in str(execution.output)


def test_tool_escapes_terminal_controls_and_keeps_head_and_tail(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "coding_agent.command.locale.getpreferredencoding",
        lambda _setlocale=False: "missing-codec",
    )
    output = b"HEAD\x00\x1b[31m" + b"x" * 1_000 + b"TAIL\xff"
    runner = RecordingRunner(_result(output=output))

    execution = _execute(
        repository,
        {"argv": [sys.executable, "script.py"]},
        runner=runner,
        permission_mode=CommandPermissionMode.AUTO,
        max_output_chars=512,
    )

    assert execution.ok is True
    assert execution.truncated is True
    assert execution.output is not None
    assert "HEAD\\x00\\x1b[31m" in execution.output
    assert "...[command output truncated]..." in execution.output
    assert "TAIL" in execution.output
    assert "\x1b" not in execution.output
    assert execution.metadata["output_encoding"] == "utf-8-replacement"


def test_local_runner_uses_direct_argv_and_combines_stdout_and_stderr(repository: Path) -> None:
    script = (
        "import sys; "
        "print('ARG='+repr(sys.argv[1]), flush=True); "
        "print('ERR', file=sys.stderr, flush=True); "
        "raise SystemExit(7)"
    )
    request = CommandRequest(
        argv=(sys.executable, "-c", script, "x & echo not-executed"),
        cwd=repository,
        timeout_seconds=5,
    )

    result = LocalCommandRunner().run(request)

    assert result.status is CommandStatus.EXITED
    assert result.exit_code == 7
    decoded = result.output.decode("utf-8")
    assert "ARG='x & echo not-executed'" in decoded
    assert decoded.index("ARG=") < decoded.index("ERR")


def test_local_runner_strips_secret_environment_variables(repository: Path) -> None:
    environment = dict(os.environ)
    environment["OPENAI_API_KEY"] = "must-not-reach-child"
    environment["PGPASSWORD"] = "must-not-reach-child"
    environment["PGPASSFILE"] = "must-not-reach-child"
    environment["ORDINARY_SETTING"] = "visible"
    script = (
        "import os; "
        "print(os.getenv('OPENAI_API_KEY', 'missing')); "
        "print(os.getenv('PGPASSWORD', 'missing')); "
        "print(os.getenv('PGPASSFILE', 'missing')); "
        "print(os.getenv('ORDINARY_SETTING', 'missing'))"
    )

    result = LocalCommandRunner(environment=environment).run(
        CommandRequest(
            argv=(sys.executable, "-c", script),
            cwd=repository,
            timeout_seconds=5,
        )
    )

    assert result.status is CommandStatus.EXITED
    assert result.output.decode("utf-8").splitlines() == [
        "missing",
        "missing",
        "missing",
        "visible",
    ]


def test_verifier_environment_cannot_turn_pytest_into_collection_only(
    repository: Path,
) -> None:
    repository.joinpath("test_failure.py").write_text(
        "def test_failure():\n    assert False\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTEST_ADDOPTS"] = "--collect-only"

    result = LocalCommandRunner(environment=environment).run(
        CommandRequest(
            argv=(sys.executable, "-I", "-m", "pytest", "-q"),
            cwd=repository,
            timeout_seconds=10,
            environment_profile=CommandEnvironmentProfile.VERIFIER,
        )
    )

    assert result.status is CommandStatus.EXITED
    assert result.exit_code == 1
    assert "FAILED" in result.output.decode("utf-8")


def test_local_runner_bounds_raw_output_with_a_head_tail_capture(repository: Path) -> None:
    script = "import sys; sys.stdout.write('HEAD' + 'x' * 1000 + 'TAIL')"

    result = LocalCommandRunner(max_output_bytes=40).run(
        CommandRequest(
            argv=(sys.executable, "-c", script),
            cwd=repository,
            timeout_seconds=5,
        )
    )

    assert result.status is CommandStatus.EXITED
    assert result.total_output_bytes == 1_008
    assert result.captured_output_bytes == 40
    assert result.output_truncated is True
    assert result.output.startswith(b"HEAD")
    assert result.output.endswith(b"TAIL")
    assert b"968 output bytes omitted" in result.output


def test_local_runner_timeout_returns_partial_output_promptly(repository: Path) -> None:
    script = "import time; print('started', flush=True); time.sleep(30)"
    started = time.monotonic()

    result = LocalCommandRunner(termination_grace_seconds=1).run(
        CommandRequest(
            argv=(sys.executable, "-c", script),
            cwd=repository,
            timeout_seconds=0.2,
        )
    )

    assert time.monotonic() - started < 5
    assert result.status is CommandStatus.TIMED_OUT
    assert b"started" in result.output


def test_cleanup_failure_outranks_keyboard_interrupt_and_is_retried(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = LocalCommandRunner(termination_grace_seconds=0.01)
    process = cast(subprocess.Popen[bytes], InterruptingProcess())
    containment = InterruptThenFailContainment()

    def start_interrupted_command(
        request: CommandRequest,
    ) -> tuple[subprocess.Popen[bytes], process_module.ProcessContainment]:
        del request
        return process, containment

    monkeypatch.setattr(runner, "_start", start_interrupted_command)

    result = runner.run(
        CommandRequest(
            argv=(sys.executable, "--version"),
            cwd=repository,
            timeout_seconds=1,
        )
    )

    assert result.status is CommandStatus.CONTROL_FAILED
    assert result.terminal_reason == "simulated cleanup confirmation failure"
    assert b"partial output" in result.output
    assert containment.cleanup_calls == 2
    assert containment.closed is True


def test_timeout_terminates_a_spawned_descendant(repository: Path) -> None:
    sentinel = repository / "descendant-finished.txt"
    child_script = (
        "import pathlib,time; time.sleep(1); "
        f"pathlib.Path({str(sentinel)!r}).write_text('alive', encoding='utf-8')"
    )
    parent_script = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child_script!r}]); "
        "print('spawned', flush=True); time.sleep(30)"
    )

    result = LocalCommandRunner(termination_grace_seconds=1).run(
        CommandRequest(
            argv=(sys.executable, "-c", parent_script),
            cwd=repository,
            timeout_seconds=0.2,
        )
    )
    time.sleep(1.2)

    assert result.status is CommandStatus.TIMED_OUT
    assert b"spawned" in result.output
    assert sentinel.exists() is False


def test_runner_cleans_pipe_holding_child_after_parent_exits(
    repository: Path,
) -> None:
    child_script = "import time; time.sleep(0.5)"
    parent_script = (
        "import subprocess,sys; "
        f"subprocess.Popen([sys.executable, '-c', {child_script!r}]); "
        "print('parent-exited', flush=True)"
    )

    result = LocalCommandRunner(termination_grace_seconds=0.05).run(
        CommandRequest(
            argv=(sys.executable, "-c", parent_script),
            cwd=repository,
            timeout_seconds=5,
        )
    )

    assert result.status is CommandStatus.CONTROL_FAILED
    assert result.exit_code is None
    assert result.terminal_reason is not None
    assert "descendant processes were still active" in result.terminal_reason
    assert b"parent-exited" in result.output


def test_normal_parent_exit_cleans_child_that_closed_standard_streams(
    repository: Path,
) -> None:
    sentinel = repository / "detached-child-finished.txt"
    child_script = (
        "import pathlib,time; time.sleep(0.7); "
        f"pathlib.Path({str(sentinel)!r}).write_text('escaped', encoding='utf-8')"
    )
    parent_script = (
        "import subprocess,sys; "
        "subprocess.Popen("
        f"[sys.executable, '-c', {child_script!r}], "
        "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
        "stderr=subprocess.DEVNULL, close_fds=True); "
        "print('parent-exited', flush=True)"
    )

    result = LocalCommandRunner(termination_grace_seconds=1).run(
        CommandRequest(
            argv=(sys.executable, "-c", parent_script),
            cwd=repository,
            timeout_seconds=5,
        )
    )
    time.sleep(1)

    assert result.status is CommandStatus.CONTROL_FAILED
    assert result.exit_code is None
    assert result.terminal_reason is not None
    assert "descendant processes were still active" in result.terminal_reason
    assert b"parent-exited" in result.output
    assert sentinel.exists() is False


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object failure boundary")
def test_windows_job_assignment_failure_never_resumes_the_command(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = repository / "uncontained-command-ran.txt"
    script = f"import pathlib; pathlib.Path({str(sentinel)!r}).write_text('ran')"

    def reject_assignment(
        _job: process_module._WindowsJobObject,
        process: subprocess.Popen[bytes],
    ) -> str:
        assert process_module._terminate_suspended_process(process)
        return "Windows Job Object assignment failed; simulated host restriction"

    monkeypatch.setattr(
        process_module._WindowsJobObject,
        "attach_and_resume",
        reject_assignment,
    )

    result = LocalCommandRunner().run(
        CommandRequest(
            argv=(sys.executable, "-c", script),
            cwd=repository,
            timeout_seconds=5,
        )
    )

    assert result.status is CommandStatus.CONTROL_FAILED
    assert result.terminal_reason is not None
    assert "simulated host restriction" in result.terminal_reason
    assert sentinel.exists() is False


def test_missing_executable_is_a_stable_structured_failure(repository: Path) -> None:
    execution = _execute(
        repository,
        {"argv": ["coding-agent-command-that-does-not-exist-4dd554"]},
        runner=LocalCommandRunner(),
        permission_mode=CommandPermissionMode.AUTO,
    )

    assert execution.ok is False
    assert execution.error_code == "command_not_found"
    assert execution.error_message == "command executable was not found"
    assert "repository" not in execution.error_message


@pytest.mark.parametrize(
    ("raised", "error_code"),
    [
        (PermissionError("private absolute path"), "command_permission_denied"),
        (OSError("private absolute path"), "command_launch_failed"),
    ],
)
def test_launch_os_errors_are_sanitized_before_reaching_the_model(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    raised: OSError,
    error_code: str,
) -> None:
    def fail_launch(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise raised

    monkeypatch.setattr("coding_agent.command.subprocess.Popen", fail_launch)

    execution = _execute(
        repository,
        {"argv": ["example-tool"]},
        runner=LocalCommandRunner(),
        permission_mode=CommandPermissionMode.AUTO,
    )

    assert execution.ok is False
    assert execution.error_code == error_code
    assert "private absolute path" not in str(execution.error_message)


def test_output_decoder_uses_locale_then_replacement_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "coding_agent.command.locale.getpreferredencoding",
        lambda _setlocale=False: "cp1252",
    )
    decoded, encoding = decode_command_output(b"caf\xe9")
    assert decoded == "café"
    assert encoding == "cp1252"

    monkeypatch.setattr(
        "coding_agent.command.locale.getpreferredencoding",
        lambda _setlocale=False: "missing-codec",
    )
    decoded, encoding = decode_command_output(b"\xff")
    assert decoded == "\ufffd"
    assert encoding == "utf-8-replacement"


def test_capture_primitive_handles_small_chunks_and_tail_overflow() -> None:
    capture = command_module._BoundedByteCapture(6)
    capture.append(b"")
    capture.append(b"ab")
    assert capture.render() == (b"ab", 2)
    assert capture.truncated is False

    capture.append(b"cdef")
    capture.append(b"gh")
    rendered, captured = capture.render()
    assert captured == 6
    assert capture.total_bytes == 8
    assert capture.truncated is True
    assert rendered.startswith(b"abc")
    assert rendered.endswith(b"fgh")


def test_environment_sanitizer_removes_prompts_and_secret_markers() -> None:
    sanitized = command_module._sanitized_environment(
        {
            "PATH": "visible",
            "SSH_ASKPASS": "hidden",
            "DATABASE_PASSWORD": "hidden",
            "SERVICE_TOKEN": "hidden",
            "PYTHONPATH": "untrusted-module-path",
            "PGPASSWORD": "hidden",
            "PGPASSFILE": "hidden",
            "TOKENIZERS_PARALLELISM": "visible",
        }
    )

    assert sanitized["PATH"] == "visible"
    assert "SSH_ASKPASS" not in sanitized
    assert "DATABASE_PASSWORD" not in sanitized
    assert "SERVICE_TOKEN" not in sanitized
    assert "PYTHONPATH" not in sanitized
    assert "PGPASSWORD" not in sanitized
    assert "PGPASSFILE" not in sanitized
    assert sanitized["TOKENIZERS_PARALLELISM"] == "visible"
    assert sanitized["GIT_TERMINAL_PROMPT"] == "0"


def test_verifier_environment_uses_a_small_allowlist() -> None:
    environment = command_module._verification_environment(
        {
            "PATH": "visible",
            "TEMP": "visible",
            "ORDINARY_SETTING": "hidden",
            "PYTEST_ADDOPTS": "--collect-only",
            "PYTEST_PLUGINS": "untrusted_plugin",
            "PGPASSWORD": "hidden",
        }
    )

    assert environment["PATH"] == "visible"
    assert environment["TEMP"] == "visible"
    assert "ORDINARY_SETTING" not in environment
    assert "PYTEST_ADDOPTS" not in environment
    assert "PYTEST_PLUGINS" not in environment
    assert "PGPASSWORD" not in environment


def test_command_classification_and_result_reject_incoherent_shapes() -> None:
    with pytest.raises(ValueError, match="set together"):
        CommandClassification(
            command_class=CommandClass.VERIFIER,
            verification_kind=VerificationKind.TEST,
        )
    with pytest.raises(ValueError, match="require verification"):
        CommandClassification(command_class=CommandClass.VERIFIER)
    with pytest.raises(ValueError, match="only verifier"):
        CommandClassification(
            command_class=CommandClass.READ_ONLY,
            verification_kind=VerificationKind.TEST,
            verification_label="pytest",
        )

    with pytest.raises(ValueError, match="cannot be negative"):
        CommandResult(
            status=CommandStatus.EXITED,
            exit_code=0,
            output=b"",
            total_output_bytes=-1,
            captured_output_bytes=0,
            output_truncated=False,
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        CommandResult(
            status=CommandStatus.EXITED,
            exit_code=0,
            output=b"",
            total_output_bytes=0,
            captured_output_bytes=1,
            output_truncated=False,
        )
    with pytest.raises(ValueError, match="non-exited"):
        CommandResult(
            status=CommandStatus.TIMED_OUT,
            exit_code=1,
            output=b"",
            total_output_bytes=0,
            captured_output_bytes=0,
            output_truncated=False,
        )
    with pytest.raises(ValueError, match="control failures"):
        CommandResult(
            status=CommandStatus.CONTROL_FAILED,
            exit_code=None,
            output=b"",
            total_output_bytes=0,
            captured_output_bytes=0,
            output_truncated=False,
        )


def test_command_result_rejects_incoherent_terminal_shapes() -> None:
    with pytest.raises(ValueError, match="exited commands require"):
        CommandResult(
            status=CommandStatus.EXITED,
            exit_code=None,
            output=b"",
            total_output_bytes=0,
            captured_output_bytes=0,
            output_truncated=False,
        )


def test_runner_configuration_rejects_unbounded_or_invalid_limits() -> None:
    with pytest.raises(ValueError, match="max_output_bytes"):
        LocalCommandRunner(max_output_bytes=1)
    with pytest.raises(ValueError, match="termination_grace_seconds"):
        LocalCommandRunner(termination_grace_seconds=0)
    with pytest.raises(ValueError, match="host-owned"):
        CommandRequest(
            argv=("tool",),
            cwd=Path.cwd(),
            timeout_seconds=1,
            environment_profile=cast(CommandEnvironmentProfile, "verifier"),
        )


def test_command_tool_rejects_an_output_budget_too_small_for_safe_headers(
    repository: Path,
) -> None:
    with pytest.raises(ValueError, match="max_output_chars"):
        RunCommandTool(Workspace(repository), max_output_chars=511)


def test_longest_control_failure_header_never_exceeds_the_tool_budget(
    repository: Path,
) -> None:
    reason = "control-failure-" + "x" * (500 - len("control-failure-"))
    result = CommandResult(
        status=CommandStatus.CONTROL_FAILED,
        exit_code=None,
        output=b"tail",
        total_output_bytes=4,
        captured_output_bytes=4,
        output_truncated=False,
        terminal_reason=reason,
    )
    tool = RunCommandTool(
        LongDisplayWorkspace(repository),
        runner=RecordingRunner(result),
        policy=CommandPolicy(CommandPermissionMode.AUTO),
        max_output_chars=512,
    )

    execution = ToolRegistry([tool]).execute(
        ToolCall(id="long-header", name="run_command", arguments={"argv": ["python"]})
    )

    assert execution.ok is True
    assert execution.output is not None
    assert len(execution.output) <= 512
    assert execution.truncated is True
    assert execution.control.terminal_stop is True
