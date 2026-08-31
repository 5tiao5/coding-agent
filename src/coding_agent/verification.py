"""Trusted post-mutation verification evidence for terminal run decisions."""

from __future__ import annotations

from dataclasses import dataclass

from coding_agent.models import (
    ToolExecution,
    VerificationKind,
    VerificationSignal,
    VerificationStatus,
)


@dataclass(frozen=True, slots=True)
class VerificationEvidence:
    epoch: int
    signal: VerificationSignal
    kind: VerificationKind
    label: str
    step: int


@dataclass(frozen=True, slots=True)
class VerificationReport:
    status: VerificationStatus
    epoch: int
    invalidation_count: int
    evidence: tuple[VerificationEvidence, ...] = ()

    @property
    def checks_passed(self) -> bool:
        """Whether all evidence currently known to this evidence-only ledger passed."""

        return self.status is VerificationStatus.VERIFIED

    @property
    def verified(self) -> bool:
        """Legacy evidence-only alias retained until completion policy is integrated."""

        return self.checks_passed

    def event_data(self) -> dict[str, object]:
        return {
            "verified": self.verified,
            "status": self.status.value,
            "epoch": self.epoch,
            "invalidation_count": self.invalidation_count,
            "evidence_count": len(self.evidence),
            "evidence_labels": [evidence.label for evidence in self.evidence],
            "evidence": [
                {
                    "label": evidence.label,
                    "kind": evidence.kind.value,
                    "passed": evidence.signal is VerificationSignal.PASSED,
                    "step": evidence.step,
                    "epoch": evidence.epoch,
                }
                for evidence in self.evidence
            ],
        }


class VerificationLedger:
    """Bind trusted verification evidence to the latest possible workspace epoch."""

    def __init__(self) -> None:
        self._epoch = 0
        self._invalidation_count = 0
        self._latest_evidence: dict[str, VerificationEvidence] = {}

    @property
    def epoch(self) -> int:
        return self._epoch

    def observe(self, execution: ToolExecution, *, step: int) -> None:
        facts = execution.control
        if facts.invalidates_verification:
            self._epoch += 1
            self._invalidation_count += 1
        if facts.verification is not None:
            assert facts.verification_kind is not None
            assert facts.verification_label is not None
            self._latest_evidence[facts.verification_label] = VerificationEvidence(
                epoch=self._epoch,
                signal=facts.verification,
                kind=facts.verification_kind,
                label=facts.verification_label,
                step=step,
            )

    def report(self) -> VerificationReport:
        all_evidence = tuple(self._latest_evidence.values())
        current_evidence = tuple(
            evidence for evidence in all_evidence if evidence.epoch == self._epoch
        )
        if current_evidence and all(
            evidence.signal is VerificationSignal.PASSED for evidence in current_evidence
        ):
            status = VerificationStatus.VERIFIED
        elif current_evidence:
            status = VerificationStatus.FAILED
        elif all_evidence:
            status = VerificationStatus.STALE
        else:
            status = VerificationStatus.MISSING
        return VerificationReport(
            status=status,
            epoch=self._epoch,
            invalidation_count=self._invalidation_count,
            evidence=current_evidence,
        )
