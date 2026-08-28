"""Stable operational errors shared across project-owned boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from types import MappingProxyType

ErrorMetadataValue = str | int | float | bool | None


class CodedError(Exception):
    """An expected failure that can be reported without leaking raw OS details."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        metadata: Mapping[str, ErrorMetadataValue] | None = None,
    ) -> None:
        normalized_message = message.strip() or "operation failed"
        super().__init__(normalized_message)
        self.code = code.strip() or "operation_error"
        self.message = normalized_message
        self.metadata: Mapping[str, ErrorMetadataValue] = MappingProxyType(
            _validated_metadata(metadata)
        )


def _validated_metadata(
    metadata: Mapping[str, ErrorMetadataValue] | None,
) -> dict[str, ErrorMetadataValue]:
    """Copy model-facing metadata while rejecting non-JSON scalar values."""
    if metadata is None:
        return {}

    validated: dict[str, ErrorMetadataValue] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise TypeError("error metadata keys must be non-empty strings")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError("error metadata values must be safe scalar values")
        if isinstance(value, float) and not isfinite(value):
            raise TypeError("error metadata floats must be finite")
        validated[key] = value
    return validated
