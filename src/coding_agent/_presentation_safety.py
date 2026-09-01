"""Shared credential-safe text rules for local presentation projections.

The local UI is an audit surface for the workspace owner, so ordinary command
arguments and output must remain visible.  These helpers redact credential
*values* without treating every ``name=value`` token as secret.
"""

from __future__ import annotations

import re

_REDACTED = "[REDACTED]"
_SENSITIVE_NAME_PATTERN = (
    r"(?:[A-Za-z][A-Za-z0-9.-]*[_-])?"
    r"(?:api[ _.-]?key|access[ _.-]?key|private[ _.-]?key|secret(?:[ _.-]?key)?|"
    r"authorization|credential|password|passwd|token|client[ _.-]?secret)"
)

_SENSITIVE_QUOTED_VALUE_PATTERN = re.compile(
    rf"(?i)(?P<prefix>[\"']?{_SENSITIVE_NAME_PATTERN}[\"']?\s*[:=]\s*)"
    r"(?P<quote>[\"'])(?P<value>.*?)(?P=quote)"
)
_SENSITIVE_BARE_VALUE_PATTERN = re.compile(
    rf"(?i)(?P<prefix>(?<![A-Za-z0-9_.-]){_SENSITIVE_NAME_PATTERN}\s*[:=]\s*)"
    r"(?P<value>(?!\[REDACTED\])[^\s,;\]}]+)"
)
_AUTH_SCHEME_PATTERN = re.compile(
    r"(?i)(?P<prefix>\b(?:bearer|basic)\s+)(?P<value>[A-Za-z0-9._~+/=-]{4,})"
)
_AUTHORIZATION_VALUE_PATTERN = re.compile(
    r"(?i)(?P<prefix>\bauthorization\s*[:=]\s*)"
    r"(?P<value>(?:(?:bearer|basic)\s+)?[^\s,;]+)"
)
_URL_CREDENTIAL_PATTERN = re.compile(r"(?P<prefix>://[^/\s:@]+:)(?P<value>[^/\s@]+)(?P<suffix>@)")

_KNOWN_TOKEN_VALUE_PATTERNS = (
    re.compile(r"(?i)(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{4,}"),
    re.compile(r"(?i)(?<![A-Za-z0-9_-])gh(?:p|o|u|s|r)_[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)(?<![A-Za-z0-9_-])github_pat_[A-Za-z0-9_]{6,}"),
    re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[0-9A-Z]{12,}(?![A-Z0-9])"),
    re.compile(
        r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\."
        r"[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
    ),
    re.compile(r"(?i)(?<![a-z0-9])(?:glpat-|xox[baprs]-)[a-z0-9_-]{8,}"),
    re.compile(r"(?<![A-Za-z0-9_-])AIza[A-Za-z0-9_-]{20,}"),
)

_SENSITIVE_ARGUMENT_NAMES = frozenset(
    {
        "api_key",
        "apikey",
        "access_key",
        "accesskey",
        "private_key",
        "privatekey",
        "secret",
        "secret_key",
        "secretkey",
        "authorization",
        "credential",
        "credentials",
        "password",
        "passwd",
        "token",
        "access_token",
        "auth_token",
        "client_secret",
    }
)

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


def redact_credential_values(value: str) -> tuple[str, bool]:
    """Redact recognizable credential values while preserving surrounding text."""

    redacted = value
    for pattern in _KNOWN_TOKEN_VALUE_PATTERNS:
        redacted = pattern.sub(_REDACTED, redacted)
    # Handle the authorization key and its optional scheme as one unit before
    # the generic scheme and assignment matchers.
    redacted = _AUTHORIZATION_VALUE_PATTERN.sub(
        lambda match: f"{match.group('prefix')}{_REDACTED}",
        redacted,
    )
    redacted = _AUTH_SCHEME_PATTERN.sub(
        lambda match: f"{match.group('prefix')}{_REDACTED}",
        redacted,
    )
    redacted = _SENSITIVE_QUOTED_VALUE_PATTERN.sub(
        lambda match: (
            f"{match.group('prefix')}{match.group('quote')}{_REDACTED}{match.group('quote')}"
        ),
        redacted,
    )
    redacted = _SENSITIVE_BARE_VALUE_PATTERN.sub(
        lambda match: f"{match.group('prefix')}{_REDACTED}",
        redacted,
    )
    redacted = _URL_CREDENTIAL_PATTERN.sub(
        lambda match: f"{match.group('prefix')}{_REDACTED}{match.group('suffix')}",
        redacted,
    )
    return redacted, redacted != value


def redact_command_argv(argv: tuple[str, ...]) -> tuple[tuple[str, ...], bool]:
    """Redact sensitive option values without hiding benign positional arguments."""

    visible: list[str] = []
    redact_next = False
    changed = False
    for token in argv:
        if redact_next:
            visible.append(_REDACTED)
            redact_next = False
            changed = True
            continue

        name, separator, assigned = token.partition("=")
        if separator and _sensitive_argument_name(name):
            visible.append(f"{name}={_REDACTED}")
            changed |= assigned != _REDACTED
            continue

        sanitized, token_changed = redact_credential_values(token)
        visible.append(sanitized)
        changed |= token_changed
        if _sensitive_argument_name(token):
            redact_next = True
    return tuple(visible), changed


def redact_command_output(value: str, argv: tuple[str, ...]) -> tuple[str, bool]:
    """Redact credentials printed without a label when argv identified their role."""

    redacted, changed = redact_credential_values(value)
    for credential in _credential_argument_values(argv):
        if not credential or credential == _REDACTED:
            continue
        pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(credential)}(?![A-Za-z0-9])")
        redacted, count = pattern.subn(_REDACTED, redacted)
        changed |= count > 0
    return redacted, changed


def _credential_argument_values(argv: tuple[str, ...]) -> tuple[str, ...]:
    values: list[str] = []
    take_next = False
    for token in argv:
        if take_next:
            values.append(token)
            take_next = False
            continue
        name, separator, assigned = token.partition("=")
        if separator and _sensitive_argument_name(name):
            values.append(assigned)
            continue
        if _sensitive_argument_name(token):
            take_next = True
    return tuple(values)


def _sensitive_argument_name(value: str) -> bool:
    candidate = value.lstrip("-/").strip().casefold().replace("-", "_").replace(".", "_")
    return candidate in _SENSITIVE_ARGUMENT_NAMES or any(
        candidate.endswith(f"_{name}") for name in _SENSITIVE_ARGUMENT_NAMES
    )


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


__all__ = [
    "contains_credential_like_value",
    "redact_command_argv",
    "redact_command_output",
    "redact_credential_values",
    "sanitize_public_label",
]
