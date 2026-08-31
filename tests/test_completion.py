"""Pure completion-contract tests independent of presentation and the Agent loop."""

from __future__ import annotations

import pytest

from coding_agent.completion import (
    CompletionContract,
    CompletionStatus,
    TargetRuntime,
    VerificationCheck,
    VerificationProfile,
    evaluate_completion,
)
from coding_agent.models import (
    ToolControlFacts,
    ToolExecution,
    VerificationKind,
    VerificationSignal,
    VerificationStatus,
)
from coding_agent.verification import VerificationLedger, VerificationReport


def _execution(
    label: str,
    signal: VerificationSignal,
    *,
    kind: VerificationKind = VerificationKind.TEST,
    invalidates: bool = False,
) -> ToolExecution:
    return ToolExecution(
        call_id=f"call-{label}",
        tool_name="host-verifier",
        ok=True,
        output="bounded verifier output",
        summary="Host verifier completed",
        control=ToolControlFacts(
            invalidates_verification=invalidates,
            verification=signal,
            verification_kind=kind,
            verification_label=label,
        ),
    )


def _profile(
    *,
    runtime_eligible: bool = True,
    required_labels: tuple[str, ...] = ("pytest",),
) -> VerificationProfile:
    return VerificationProfile(
        checks=(
            VerificationCheck(
                label="pytest",
                kind=VerificationKind.TEST,
                scopes=("tests",),
            ),
            VerificationCheck(
                label="mypy",
                kind=VerificationKind.CHECK,
                scopes=("types",),
            ),
            VerificationCheck(
                label="gui-smoke",
                kind=VerificationKind.CHECK,
                scopes=("startup:gui",),
            ),
        ),
        required_labels=required_labels,
        target_runtime=TargetRuntime(
            runtime_id="python-project-venv",
            eligible_for_task_validation=runtime_eligible,
        ),
    )


def _report(*executions: ToolExecution) -> VerificationReport:
    ledger = VerificationLedger()
    for step, execution in enumerate(executions, start=1):
        ledger.observe(execution, step=step)
    return ledger.report()


def test_profile_rejects_empty_duplicate_and_unknown_required_labels() -> None:
    checks = (
        VerificationCheck("pytest", VerificationKind.TEST, ("tests",)),
        VerificationCheck("mypy", VerificationKind.CHECK, ("types",)),
    )
    runtime = TargetRuntime("python-project-venv", eligible_for_task_validation=True)

    with pytest.raises(ValueError, match="required_labels cannot be empty"):
        VerificationProfile(checks=checks, required_labels=(), target_runtime=runtime)
    with pytest.raises(ValueError, match="required_labels must be unique"):
        VerificationProfile(
            checks=checks,
            required_labels=("pytest", "pytest"),
            target_runtime=runtime,
        )
    with pytest.raises(ValueError, match="required_labels must reference known checks"):
        VerificationProfile(
            checks=checks,
            required_labels=("pytest", "ruff"),
            target_runtime=runtime,
        )


def test_profile_rejects_duplicate_check_labels() -> None:
    runtime = TargetRuntime("python-project-venv", eligible_for_task_validation=True)

    with pytest.raises(ValueError, match="check labels must be unique"):
        VerificationProfile(
            checks=(
                VerificationCheck("pytest", VerificationKind.TEST, ("tests",)),
                VerificationCheck("pytest", VerificationKind.CHECK, ("types",)),
            ),
            required_labels=("pytest",),
            target_runtime=runtime,
        )


def test_required_checks_can_pass_without_validating_missing_task_scope() -> None:
    completion = evaluate_completion(
        _profile(),
        CompletionContract(required_scopes=("tests", "types")),
        _report(_execution("pytest", VerificationSignal.PASSED)),
    )

    assert completion.check_status is VerificationStatus.VERIFIED
    assert completion.checks_passed is True
    assert completion.task_validated is False
    assert completion.verified is False
    assert completion.completion_status is CompletionStatus.CHECKS_ONLY
    assert completion.missing_labels == ()
    assert completion.passed_scopes == ("tests",)
    assert completion.missing_scopes == ("types",)


def test_all_required_labels_scopes_and_runtime_validate_the_task() -> None:
    completion = evaluate_completion(
        _profile(required_labels=("pytest", "mypy")),
        CompletionContract(required_scopes=("tests", "types")),
        _report(
            _execution("pytest", VerificationSignal.PASSED),
            _execution("mypy", VerificationSignal.PASSED, kind=VerificationKind.CHECK),
        ),
    )

    assert completion.checks_passed is True
    assert completion.task_validated is True
    assert completion.verified is completion.task_validated
    assert completion.completion_status is CompletionStatus.VALIDATED
    assert completion.missing_labels == ()
    assert completion.missing_scopes == ()


def test_ineligible_target_runtime_keeps_passing_checks_at_checks_only() -> None:
    completion = evaluate_completion(
        _profile(runtime_eligible=False),
        CompletionContract(required_scopes=("tests",)),
        _report(_execution("pytest", VerificationSignal.PASSED)),
    )

    assert completion.checks_passed is True
    assert completion.target_runtime_eligible is False
    assert completion.task_validated is False
    assert completion.completion_status is CompletionStatus.CHECKS_ONLY


def test_contract_can_explicitly_accept_a_fallback_runtime() -> None:
    completion = evaluate_completion(
        _profile(runtime_eligible=False),
        CompletionContract(required_scopes=("tests",), require_target_runtime=False),
        _report(_execution("pytest", VerificationSignal.PASSED)),
    )

    assert completion.checks_passed is True
    assert completion.task_validated is True
    assert completion.completion_status is CompletionStatus.VALIDATED


def test_current_failed_optional_evidence_blocks_validation() -> None:
    completion = evaluate_completion(
        _profile(),
        CompletionContract(required_scopes=("tests",)),
        _report(
            _execution("pytest", VerificationSignal.PASSED),
            _execution("mypy", VerificationSignal.FAILED, kind=VerificationKind.CHECK),
        ),
    )

    assert completion.check_status is VerificationStatus.FAILED
    assert completion.checks_passed is False
    assert completion.task_validated is False
    assert completion.completion_status is CompletionStatus.FAILED


def test_missing_required_label_is_not_hidden_by_another_passing_check() -> None:
    completion = evaluate_completion(
        _profile(required_labels=("pytest", "mypy")),
        CompletionContract(required_scopes=("tests",)),
        _report(_execution("pytest", VerificationSignal.PASSED)),
    )

    assert completion.check_status is VerificationStatus.MISSING
    assert completion.checks_passed is False
    assert completion.task_validated is False
    assert completion.completion_status is CompletionStatus.MISSING
    assert completion.missing_labels == ("mypy",)


def test_mutation_after_passing_evidence_makes_completion_stale() -> None:
    ledger = VerificationLedger()
    ledger.observe(_execution("pytest", VerificationSignal.PASSED), step=1)
    ledger.observe(
        ToolExecution(
            call_id="call-mutation",
            tool_name="write_file",
            ok=True,
            output="workspace changed",
            summary="Changed workspace",
            control=ToolControlFacts(invalidates_verification=True),
        ),
        step=2,
    )

    completion = evaluate_completion(
        _profile(),
        CompletionContract(required_scopes=("tests",)),
        ledger.report(),
    )

    assert completion.check_status is VerificationStatus.STALE
    assert completion.checks_passed is False
    assert completion.task_validated is False
    assert completion.completion_status is CompletionStatus.STALE
    assert completion.missing_labels == ("pytest",)


def test_kind_mismatch_and_unknown_evidence_fail_closed() -> None:
    kind_mismatch = evaluate_completion(
        _profile(),
        CompletionContract(required_scopes=("tests",)),
        _report(_execution("pytest", VerificationSignal.PASSED, kind=VerificationKind.CHECK)),
    )
    unknown = evaluate_completion(
        _profile(),
        CompletionContract(required_scopes=("tests",)),
        _report(_execution("unregistered", VerificationSignal.PASSED)),
    )

    assert kind_mismatch.check_status is VerificationStatus.FAILED
    assert kind_mismatch.mismatched_labels == ("pytest",)
    assert unknown.check_status is VerificationStatus.FAILED
    assert unknown.unexpected_labels == ("unregistered",)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: VerificationCheck("pytest", VerificationKind.TEST, ("tests", "tests")),
            "check scopes must be unique",
        ),
        (
            lambda: CompletionContract(required_scopes=("tests", "tests")),
            "required_scopes must be unique",
        ),
        (
            lambda: CompletionContract(required_scopes=("UPPERCASE",)),
            "invalid verification scope",
        ),
    ],
)
def test_scope_contracts_are_canonical(factory: object, message: str) -> None:
    callable_factory = factory
    assert callable(callable_factory)
    with pytest.raises(ValueError, match=message):
        callable_factory()
