"""Host-owned completion requirements evaluated over trusted verifier evidence.

This module deliberately contains no Agent-loop or presentation logic.  A
``VerificationProfile`` freezes which labels the host recognizes and what semantic
scopes each check covers.  A ``CompletionContract`` states which of those scopes a
task must cover before the host may call it validated.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from coding_agent.models import VerificationKind, VerificationSignal, VerificationStatus
from coding_agent.verification import VerificationEvidence, VerificationReport

_TOKEN_PATTERN = re.compile(r"^[a-z][a-z0-9_.:-]{0,63}$")
_MAX_LABEL_CHARS = 120


class CompletionStatus(StrEnum):
    """Why trusted checks do or do not validate the whole task."""

    VALIDATED = "validated"
    CHECKS_ONLY = "checks_only"
    MISSING = "missing"
    FAILED = "failed"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class TargetRuntime:
    """Safe identity and validation eligibility of the host-selected runtime."""

    runtime_id: str
    eligible_for_task_validation: bool

    def __post_init__(self) -> None:
        _require_token(self.runtime_id, field="runtime_id")
        if type(self.eligible_for_task_validation) is not bool:
            raise ValueError("eligible_for_task_validation must be a boolean")


@dataclass(frozen=True, slots=True)
class VerificationCheck:
    """One host-recognized evidence label and the task scopes it can establish."""

    label: str
    kind: VerificationKind
    scopes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_label(self.label)
        if not isinstance(self.kind, VerificationKind):
            raise ValueError("check kind must be a VerificationKind")
        if not self.scopes:
            raise ValueError("check scopes cannot be empty")
        _require_unique(self.scopes, field="check scopes")
        for scope in self.scopes:
            _require_scope(scope)


@dataclass(frozen=True, slots=True)
class VerificationProfile:
    """Immutable host policy for recognized and required verification labels."""

    checks: tuple[VerificationCheck, ...]
    required_labels: tuple[str, ...]
    target_runtime: TargetRuntime

    def __post_init__(self) -> None:
        if not self.checks:
            raise ValueError("checks cannot be empty")
        if not self.required_labels:
            raise ValueError("required_labels cannot be empty")
        if not isinstance(self.target_runtime, TargetRuntime):
            raise ValueError("target_runtime must be a TargetRuntime")

        labels = tuple(check.label for check in self.checks)
        _require_unique(labels, field="check labels")
        _require_unique(self.required_labels, field="required_labels")
        for label in self.required_labels:
            _require_label(label)
        unknown = tuple(label for label in self.required_labels if label not in labels)
        if unknown:
            raise ValueError("required_labels must reference known checks")


@dataclass(frozen=True, slots=True)
class CompletionContract:
    """Scopes and runtime guarantees required to validate one task."""

    required_scopes: tuple[str, ...]
    require_target_runtime: bool = True

    def __post_init__(self) -> None:
        _require_unique(self.required_scopes, field="required_scopes")
        for scope in self.required_scopes:
            _require_scope(scope)
        if type(self.require_target_runtime) is not bool:
            raise ValueError("require_target_runtime must be a boolean")


@dataclass(frozen=True, slots=True)
class CompletionReport:
    """Pure result of applying a completion contract to current trusted evidence."""

    check_status: VerificationStatus
    completion_status: CompletionStatus
    checks_passed: bool
    task_validated: bool
    required_labels: tuple[str, ...]
    missing_labels: tuple[str, ...]
    required_scopes: tuple[str, ...]
    passed_scopes: tuple[str, ...]
    missing_scopes: tuple[str, ...]
    target_runtime_id: str
    target_runtime_eligible: bool
    unexpected_labels: tuple[str, ...] = ()
    mismatched_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.task_validated and not self.checks_passed:
            raise ValueError("task validation requires passing checks")
        if self.task_validated != (self.completion_status is CompletionStatus.VALIDATED):
            raise ValueError("validated completion status must match task_validated")

    @property
    def verified(self) -> bool:
        """Backward-compatible public alias for full task validation."""

        return self.task_validated

    def event_data(self, verification: VerificationReport) -> dict[str, object]:
        """Return additive, presentation-safe facts with legacy aliases preserved."""

        data = verification.event_data()
        data.update(
            {
                "verified": self.task_validated,
                "status": (
                    "verified"
                    if self.completion_status is CompletionStatus.VALIDATED
                    else self.completion_status.value
                ),
                "checks_passed": self.checks_passed,
                "task_validated": self.task_validated,
                "completion_status": self.completion_status.value,
                "required_labels": list(self.required_labels),
                "missing_labels": list(self.missing_labels),
                "required_scopes": list(self.required_scopes),
                "passed_scopes": list(self.passed_scopes),
                "missing_scopes": list(self.missing_scopes),
                "target_runtime_id": self.target_runtime_id,
                "target_runtime_eligible": self.target_runtime_eligible,
                "unexpected_labels": list(self.unexpected_labels),
                "mismatched_labels": list(self.mismatched_labels),
            }
        )
        return data


def evaluate_completion(
    profile: VerificationProfile,
    contract: CompletionContract,
    verification: VerificationReport,
) -> CompletionReport:
    """Evaluate current evidence without trusting model-authored completion claims."""

    if not isinstance(profile, VerificationProfile):
        raise TypeError("profile must be a VerificationProfile")
    if not isinstance(contract, CompletionContract):
        raise TypeError("contract must be a CompletionContract")
    if not isinstance(verification, VerificationReport):
        raise TypeError("verification must be a VerificationReport")

    checks_by_label = {check.label: check for check in profile.checks}
    evidence_by_label = {evidence.label: evidence for evidence in verification.evidence}
    unexpected_labels = tuple(label for label in evidence_by_label if label not in checks_by_label)
    mismatched_labels = tuple(
        label
        for label, evidence in evidence_by_label.items()
        if label in checks_by_label and evidence.kind is not checks_by_label[label].kind
    )
    missing_labels = tuple(
        label for label in profile.required_labels if label not in evidence_by_label
    )
    current_failure = any(
        evidence.signal is VerificationSignal.FAILED for evidence in verification.evidence
    )

    if verification.status is VerificationStatus.STALE:
        check_status = VerificationStatus.STALE
    elif (
        verification.status is VerificationStatus.FAILED
        or current_failure
        or unexpected_labels
        or mismatched_labels
    ):
        check_status = VerificationStatus.FAILED
    elif verification.status is VerificationStatus.MISSING or missing_labels:
        check_status = VerificationStatus.MISSING
    else:
        check_status = VerificationStatus.VERIFIED

    passed_scopes = _passed_scopes(profile, evidence_by_label, mismatched_labels)
    missing_scopes = tuple(
        scope for scope in contract.required_scopes if scope not in passed_scopes
    )
    checks_passed = check_status is VerificationStatus.VERIFIED
    target_runtime_eligible = profile.target_runtime.eligible_for_task_validation
    runtime_satisfied = target_runtime_eligible or not contract.require_target_runtime
    task_validated = checks_passed and not missing_scopes and runtime_satisfied

    if task_validated:
        completion_status = CompletionStatus.VALIDATED
    elif check_status is VerificationStatus.FAILED:
        completion_status = CompletionStatus.FAILED
    elif check_status is VerificationStatus.STALE:
        completion_status = CompletionStatus.STALE
    elif check_status is VerificationStatus.MISSING:
        completion_status = CompletionStatus.MISSING
    else:
        completion_status = CompletionStatus.CHECKS_ONLY

    return CompletionReport(
        check_status=check_status,
        completion_status=completion_status,
        checks_passed=checks_passed,
        task_validated=task_validated,
        required_labels=profile.required_labels,
        missing_labels=missing_labels,
        required_scopes=contract.required_scopes,
        passed_scopes=passed_scopes,
        missing_scopes=missing_scopes,
        target_runtime_id=profile.target_runtime.runtime_id,
        target_runtime_eligible=target_runtime_eligible,
        unexpected_labels=unexpected_labels,
        mismatched_labels=mismatched_labels,
    )


def _passed_scopes(
    profile: VerificationProfile,
    evidence_by_label: Mapping[str, VerificationEvidence],
    mismatched_labels: tuple[str, ...],
) -> tuple[str, ...]:
    # Keep profile order stable so reports and future serialized events are deterministic.
    passed: list[str] = []
    mismatched = set(mismatched_labels)
    for check in profile.checks:
        evidence = evidence_by_label.get(check.label)
        if (
            evidence is None
            or check.label in mismatched
            or evidence.signal is not VerificationSignal.PASSED
        ):
            continue
        for scope in check.scopes:
            if scope not in passed:
                passed.append(scope)
    return tuple(passed)


def _require_label(value: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_LABEL_CHARS:
        raise ValueError("check labels must contain 1-120 characters")
    contains_control = any(ord(character) < 32 or ord(character) == 127 for character in value)
    if value != value.strip() or contains_control:
        raise ValueError("check labels must be canonical printable text")


def _require_scope(value: str) -> None:
    if not isinstance(value, str) or _TOKEN_PATTERN.fullmatch(value) is None:
        raise ValueError("invalid verification scope")


def _require_token(value: str, *, field: str) -> None:
    if not isinstance(value, str) or _TOKEN_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a canonical safe token")


def _require_unique(values: tuple[str, ...], *, field: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field} must be unique")
