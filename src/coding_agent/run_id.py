"""One portable filename-safe identifier contract shared by durable stores."""

from __future__ import annotations

import re

_RUN_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


def require_run_id(value: str) -> str:
    """Return a canonical run ID or raise a provider-neutral ``ValueError``."""
    if (
        not isinstance(value, str)
        or not _RUN_ID_PATTERN.fullmatch(value)
        or value.upper() in _WINDOWS_RESERVED_NAMES
    ):
        raise ValueError(
            "run_id must be 1-64 lowercase ASCII characters using letters, digits, _ or -"
        )
    return value
