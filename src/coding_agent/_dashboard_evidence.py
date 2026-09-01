"""Bounded evidence values shared by terminal and Web dashboard projections."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from coding_agent._dashboard_activity import public_verification_kind
from coding_agent._presentation_safety import sanitize_public_label


@dataclass(frozen=True, slots=True)
class ChangedFile:
    """One sanitized mutation fact safe to show in presentation surfaces."""

    path: str
    added_lines: int
    removed_lines: int
    revision: int
    change_kind: str


@dataclass(frozen=True, slots=True)
class VerificationEvidenceItem:
    """One current trusted verifier fact bound to a mutation revision."""

    label: str
    kind: str
    passed: bool
    step: int
    epoch: int


def changed_file_from_event(data: Mapping[str, object]) -> ChangedFile | None:
    """Read only the mutation metadata explicitly approved for presentation."""

    metadata = data.get("metadata")
    if not isinstance(metadata, Mapping) or _mapping_bool(metadata, "changed") is not True:
        return None
    path = _mapping_string(metadata, "path", limit=180)
    if not path:
        return None
    return ChangedFile(
        path=path,
        added_lines=_mapping_nonnegative_int(metadata, "added_lines"),
        removed_lines=_mapping_nonnegative_int(metadata, "removed_lines"),
        revision=_mapping_nonnegative_int(metadata, "mutation_revision"),
        change_kind=_mapping_string(metadata, "change_kind", limit=24) or "update",
    )


def verification_evidence_items(
    value: object,
    *,
    max_items: int,
) -> tuple[VerificationEvidenceItem, ...]:
    """Parse the report's bounded structured evidence without stringifying unknown data."""

    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    items: list[VerificationEvidenceItem] = []
    labels: set[str] = set()
    for candidate in value[:max_items]:
        if not isinstance(candidate, Mapping):
            continue
        label, _ = sanitize_public_label(candidate.get("label"), limit=100)
        kind = public_verification_kind(candidate.get("kind"))
        passed = _mapping_bool(candidate, "passed")
        step = _mapping_nonnegative_int(candidate, "step")
        epoch = _mapping_nonnegative_int(candidate, "epoch")
        if not label or not kind or passed is None or label in labels:
            continue
        labels.add(label)
        items.append(
            VerificationEvidenceItem(
                label=label,
                kind=kind,
                passed=passed,
                step=step,
                epoch=epoch,
            )
        )
    return tuple(items)


def changed_file_label(change: ChangedFile) -> str:
    return f"{change.path} (+{change.added_lines}/-{change.removed_lines})"


def verification_evidence_label(evidence: VerificationEvidenceItem) -> str:
    result = "PASS" if evidence.passed else "FAIL"
    return (
        f"{evidence.label} {result} "
        f"({evidence.kind}, step {evidence.step}, workspace revision {evidence.epoch})"
    )


def _mapping_string(data: Mapping[object, object], key: str, *, limit: int) -> str | None:
    value = data.get(key)
    if not isinstance(value, str):
        return None
    cleaned = _clean_text(value, limit=limit)
    return cleaned or None


def _mapping_bool(data: Mapping[object, object], key: str) -> bool | None:
    value = data.get(key)
    return value if isinstance(value, bool) else None


def _mapping_nonnegative_int(data: Mapping[object, object], key: str) -> int:
    value = data.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _clean_text(value: str, *, limit: int) -> str:
    printable = "".join(" " if not char.isprintable() else char for char in value)
    collapsed = " ".join(printable.split())
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[: max(0, limit - 3)]}..."


__all__ = [
    "ChangedFile",
    "VerificationEvidenceItem",
    "changed_file_from_event",
    "changed_file_label",
    "verification_evidence_items",
    "verification_evidence_label",
]
