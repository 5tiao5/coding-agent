"""Small cooperative-cancellation primitives for host/runner boundaries.

Cancellation is deliberately advisory: requesting it wakes code that is already
observing the token, but it never attempts to terminate a thread or interrupt a
blocking model/tool call.
"""

from __future__ import annotations

from collections.abc import Callable
from threading import Event, Lock


class CancellationToken:
    """Read-only view of one cooperative cancellation request."""

    def __init__(self, requested: Event) -> None:
        self._requested = requested

    @property
    def is_cancellation_requested(self) -> bool:
        """Whether the owning source has requested cancellation."""
        return self._requested.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait until cancellation is requested, returning ``False`` on timeout."""
        return self._requested.wait(timeout)


class CancellationSource:
    """Own the write side of one idempotent cooperative cancellation signal."""

    def __init__(self) -> None:
        self._requested = Event()
        self._request_lock = Lock()
        self._token = CancellationToken(self._requested)

    @property
    def token(self) -> CancellationToken:
        """Return the stable read-only token shared with a cancellable runner."""
        return self._token

    def request_cancellation(self) -> bool:
        """Request cancellation once; return whether this call set the signal."""
        with self._request_lock:
            if self._requested.is_set():
                return False
            self._requested.set()
            return True


def cancellation_requested(token: CancellationToken | None) -> bool:
    """Return whether an optional host token has been requested."""

    return token is not None and token.is_cancellation_requested


def wait_for_retry_or_cancellation(
    token: CancellationToken | None,
    delay_seconds: float,
    sleeper: Callable[[float], None],
) -> bool:
    """Run ordinary backoff, or let a cooperative token wake it early."""

    if token is not None:
        return token.wait(delay_seconds)
    sleeper(delay_seconds)
    return False
