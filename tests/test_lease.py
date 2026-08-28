"""Cross-process run lease tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from coding_agent.lease import RunLease, RunLeaseError


def test_only_one_lease_can_hold_a_run(tmp_path: Path) -> None:
    first = RunLease(tmp_path / "leases", "run-1")
    second = RunLease(tmp_path / "leases", "run-1")

    with first:
        with pytest.raises(RunLeaseError) as raised:
            second.acquire()
        assert raised.value.code == "run_already_active"

    with second:
        assert second.path.is_file()


def test_lease_rejects_reuse_and_unsafe_locations(tmp_path: Path) -> None:
    lease = RunLease(tmp_path / "leases", "run-1")
    lease.acquire()
    try:
        with pytest.raises(RunLeaseError) as reused:
            lease.acquire()
        assert reused.value.code == "run_lease_reused"
    finally:
        lease.release()
    lease.release()

    with pytest.raises(ValueError, match="absolute"):
        RunLease(Path("relative"), "run-1")
    with pytest.raises(RunLeaseError) as invalid:
        RunLease(tmp_path / "leases", "UPPERCASE")
    assert invalid.value.code == "invalid_run_id"


def test_lease_rejects_a_hardlinked_lock_file(tmp_path: Path) -> None:
    state = tmp_path / "leases"
    state.mkdir()
    target = state / "run-1.lock"
    target.write_bytes(b"\0")
    os.link(target, tmp_path / "lease-alias.lock")

    with pytest.raises(RunLeaseError) as raised:
        RunLease(state, "run-1").acquire()

    assert raised.value.code == "unsafe_run_lease"
