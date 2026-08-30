"""Tests for trusted verification evidence and invalidation epochs."""

from __future__ import annotations

from coding_agent.models import (
    ToolControlFacts,
    ToolExecution,
    VerificationKind,
    VerificationSignal,
    VerificationStatus,
)
from coding_agent.verification import VerificationLedger


def _execution(
    *,
    invalidates: bool = False,
    verification: VerificationSignal | None = None,
    kind: VerificationKind | None = None,
    label: str | None = None,
) -> ToolExecution:
    return ToolExecution(
        call_id="call-1",
        tool_name="test-tool",
        ok=True,
        output="observable output",
        summary="Observed test tool",
        control=ToolControlFacts(
            invalidates_verification=invalidates,
            verification=verification,
            verification_kind=kind,
            verification_label=label,
        ),
    )


def test_missing_evidence_is_not_verified() -> None:
    report = VerificationLedger().report()

    assert report.status is VerificationStatus.MISSING
    assert report.verified is False
    assert report.epoch == 0


def test_passed_evidence_is_bound_to_the_current_epoch() -> None:
    ledger = VerificationLedger()

    ledger.observe(
        _execution(
            invalidates=True,
            verification=VerificationSignal.PASSED,
            kind=VerificationKind.TEST,
            label="pytest",
        ),
        step=3,
    )

    report = ledger.report()
    assert report.status is VerificationStatus.VERIFIED
    assert report.verified is True
    assert report.epoch == 1
    assert len(report.evidence) == 1
    assert report.evidence[0].label == "pytest"
    assert report.evidence[0].step == 3
    assert report.event_data()["evidence"] == [
        {
            "label": "pytest",
            "kind": "test",
            "passed": True,
            "step": 3,
            "epoch": 1,
        }
    ]


def test_later_possible_write_makes_passed_evidence_stale() -> None:
    ledger = VerificationLedger()
    ledger.observe(
        _execution(
            verification=VerificationSignal.PASSED,
            kind=VerificationKind.TEST,
            label="pytest",
        ),
        step=2,
    )

    ledger.observe(_execution(invalidates=True), step=3)

    report = ledger.report()
    assert report.status is VerificationStatus.STALE
    assert report.verified is False
    assert report.epoch == 1


def test_current_failed_check_is_reported_as_failed() -> None:
    ledger = VerificationLedger()

    ledger.observe(
        _execution(
            invalidates=True,
            verification=VerificationSignal.FAILED,
            kind=VerificationKind.TEST,
            label="pytest",
        ),
        step=4,
    )

    report = ledger.report()
    assert report.status is VerificationStatus.FAILED
    assert report.verified is False


def test_control_facts_are_not_exposed_in_model_context() -> None:
    execution = _execution(
        invalidates=True,
        verification=VerificationSignal.PASSED,
        kind=VerificationKind.TEST,
        label="pytest",
    )

    content = execution.as_message_content()

    assert "control" not in content
    assert "invalidates_verification" not in content
    assert "verification_label" not in content


def test_all_current_verification_keys_must_pass() -> None:
    ledger = VerificationLedger()
    ledger.observe(
        _execution(
            verification=VerificationSignal.PASSED,
            kind=VerificationKind.TEST,
            label="pytest",
        ),
        step=2,
    )
    ledger.observe(
        _execution(
            verification=VerificationSignal.FAILED,
            kind=VerificationKind.CHECK,
            label="ruff-check",
        ),
        step=3,
    )

    assert ledger.report().status is VerificationStatus.FAILED

    ledger.observe(
        _execution(
            verification=VerificationSignal.PASSED,
            kind=VerificationKind.CHECK,
            label="ruff-check",
        ),
        step=4,
    )

    assert ledger.report().status is VerificationStatus.VERIFIED
