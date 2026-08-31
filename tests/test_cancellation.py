from __future__ import annotations

from coding_agent.cancellation import (
    CancellationSource,
    cancellation_requested,
    wait_for_retry_or_cancellation,
)


def test_cancellation_source_is_stable_idempotent_and_observable() -> None:
    source = CancellationSource()

    assert source.token is source.token
    assert cancellation_requested(None) is False
    assert cancellation_requested(source.token) is False
    assert source.token.wait(timeout=0) is False

    assert source.request_cancellation() is True
    assert source.request_cancellation() is False
    assert cancellation_requested(source.token) is True
    assert source.token.wait(timeout=0) is True


def test_retry_wait_uses_the_supplied_sleep_strategy_until_cancelled() -> None:
    delays: list[float] = []

    assert wait_for_retry_or_cancellation(None, 0.25, delays.append) is False
    assert delays == [0.25]

    source = CancellationSource()
    source.request_cancellation()
    assert wait_for_retry_or_cancellation(source.token, 60, delays.append) is True
    assert delays == [0.25]
