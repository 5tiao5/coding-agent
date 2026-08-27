"""Stable operational errors shared across project-owned boundaries."""

from __future__ import annotations


class CodedError(Exception):
    """An expected failure that can be reported without leaking raw OS details."""

    def __init__(self, code: str, message: str) -> None:
        normalized_message = message.strip() or "operation failed"
        super().__init__(normalized_message)
        self.code = code.strip() or "operation_error"
        self.message = normalized_message
