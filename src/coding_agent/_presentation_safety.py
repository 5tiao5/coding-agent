"""Shared credential-safe text rules for public presentation projections."""

from __future__ import annotations

import re

_CREDENTIAL_LIKE_PATTERNS = (
    re.compile(r"(?i)(?:^|[^A-Za-z0-9])sk-[A-Za-z0-9_-]{4,}"),
    re.compile(r"(?i)(?:^|[^A-Za-z0-9])gh[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)(?:^|[^A-Za-z0-9])github_pat_[A-Za-z0-9_]{6,}"),
    re.compile(r"(?:^|[^A-Z0-9])(?:AKIA|ASIA)[0-9A-Z_/-]{4,}"),
    re.compile(
        r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\."
        r"[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
    ),
    re.compile(
        r"(?i)(?:^|[^A-Za-z0-9_.-])[A-Za-z][A-Za-z0-9_.-]{0,63}"
        r"\s*=\s*\S+"
    ),
    re.compile(
        r"(?i)(?:^|[^a-z0-9])(?:api[ _-]?key|access[ _-]?key|private[ _-]?key|"
        r"secret[ _-]?key|authorization|credential|password|passwd|token|key)"
        r"\s*[:=]\s*[^\s,;]+"
    ),
    re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"(?i)(?<![a-z0-9])(?:glpat-|xox[baprs]-)[a-z0-9_-]{8,}"),
    re.compile(r"(?<![A-Za-z0-9_-])AIza[A-Za-z0-9_-]{20,}"),
    re.compile(r"://[^/\s:@]+:[^/\s@]+@"),
)


def contains_credential_like_value(value: str) -> bool:
    """Return whether display text contains a known credential value shape."""

    return any(pattern.search(value) is not None for pattern in _CREDENTIAL_LIKE_PATTERNS)


def sanitize_public_label(
    value: object,
    *,
    limit: int = 120,
) -> tuple[str | None, bool]:
    """Return a credential-free label and whether it remained byte-for-byte visible."""

    if not isinstance(value, str):
        return None, False
    printable = "".join(" " if not character.isprintable() else character for character in value)
    collapsed = " ".join(printable.split())
    if not collapsed or len(collapsed) > limit or contains_credential_like_value(collapsed):
        return None, False
    return collapsed, collapsed == value


__all__ = ["contains_credential_like_value", "sanitize_public_label"]
