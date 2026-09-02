"""Deterministic tests for the passive Rich dashboard projection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from io import BytesIO, StringIO, TextIOWrapper
from typing import ClassVar

import pytest
from rich.console import Console

import coding_agent.dashboard as dashboard_module
from coding_agent.dashboard import DashboardEventSink, DashboardProjection
from coding_agent.events import EventKind, RunEvent

_START = datetime(2026, 8, 28, 4, 0, tzinfo=UTC)


def _event(
    kind: EventKind,
    *,
    run_id: str = "run-1",
    step: int = 0,
    seconds: float = 0,
    message: str = "event message must not be rendered",
    data: dict[str, object] | None = None,
) -> RunEvent:
    return RunEvent(
        run_id=run_id,
        kind=kind,
        message=message,
        step=step,
        timestamp=_START + timedelta(seconds=seconds),
        data=data or {},
    )


def _plain_console(stream: StringIO, *, width: int = 100) -> Console:
    return Console(file=stream, force_terminal=False, color_system=None, width=width)


def test_projection_folds_a_verified_run_without_agent_internals() -> None:
    projection = DashboardProjection(task_label="Repair discount bug", max_timeline=20)
    events = [
        _event(
            EventKind.RUN_STARTED,
            data={
                "task_chars": 500,
                "limits": {
                    "max_model_turns": 20,
                    "max_calls_per_turn": 8,
                    "max_total_tool_calls": 40,
                    "private": "must remain hidden",
                },
            },
        ),
        _event(
            EventKind.STATE_CHANGED,
            step=1,
            seconds=0.1,
            data={"previous": "planning", "current": "acting"},
        ),
        _event(EventKind.MODEL_REQUESTED, step=1, seconds=0.2),
        _event(
            EventKind.MODEL_RESPONDED,
            step=1,
            seconds=0.3,
            data={"tool_count": 1, "has_content": False},
        ),
        _event(
            EventKind.TOOL_STARTED,
            step=1,
            seconds=0.4,
            data={"call_id": "call-1", "tool_name": "update_plan"},
        ),
        _event(
            EventKind.TOOL_FINISHED,
            step=1,
            seconds=0.7,
            data={
                "call_id": "call-1",
                "tool_name": "update_plan",
                "ok": True,
                "summary": "Plan now has three steps",
                "duration_ms": 300.0,
                "preview": {
                    "plan": [
                        {"status": "completed", "step": "Inspect failure"},
                        {"status": "in_progress", "step": "Apply repair"},
                    ]
                },
                "raw_output": "must remain hidden",
            },
        ),
        _event(EventKind.VERIFICATION_INVALIDATED, step=1, seconds=0.8),
        _event(
            EventKind.VERIFICATION_RECORDED,
            step=2,
            seconds=1.2,
            data={"passed": True, "kind": "test", "label": "pytest -q"},
        ),
        _event(
            EventKind.VERIFICATION_RECORDED,
            step=2,
            seconds=1.3,
            data={"passed": True, "kind": "test", "label": "pytest -q"},
        ),
        _event(
            EventKind.VERIFICATION_EVALUATED,
            step=3,
            seconds=1.5,
            data={
                "verified": True,
                "status": "verified",
                "evidence_labels": ["pytest -q"],
            },
        ),
        _event(
            EventKind.SESSION_CHECKPOINTED,
            step=3,
            seconds=1.7,
            data={"boundary": "terminal"},
        ),
        _event(
            EventKind.RUN_FINISHED,
            step=3,
            seconds=2,
            data={
                "verified": True,
                "status": "verified",
                "evidence_labels": ["pytest -q"],
            },
        ),
    ]

    for event in events:
        projection.apply(event)

    snapshot = projection.snapshot
    assert snapshot.run_id == "run-1"
    assert snapshot.task_label == "Repair discount bug"
    assert snapshot.phase == "COMPLETED"
    assert snapshot.current_step == 3
    assert snapshot.limits is not None
    assert snapshot.limits.max_model_turns == 20
    assert snapshot.limits.max_calls_per_turn == 8
    assert snapshot.limits.max_total_tool_calls == 40
    assert snapshot.tools_started == 1
    assert snapshot.tools_finished == 1
    assert snapshot.tools_failed == 0
    assert snapshot.active_tools == ()
    assert snapshot.verification_status == "verified"
    assert snapshot.verification_labels == ("pytest -q",)
    assert snapshot.elapsed_seconds == 2
    assert snapshot.terminal is True
    assert snapshot.run_failed is False
    assert snapshot.stop_reason is None
    assert snapshot.outcome == "VERIFIED"
    assert snapshot.plan_lines == (
        "[COMPLETED] Inspect failure",
        "[IN PROGRESS] Apply repair",
    )
    assert snapshot.workspace_changes == ()
    assert snapshot.workspace_changes_complete is True
    assert snapshot.omitted_change_count == 0
    assert snapshot.latest_change is None
    tool_result = next(
        item
        for item in snapshot.timeline
        if item.category == "TOOL" and item.headline.endswith("completed")
    )
    assert tool_result.preview == (
        "[COMPLETED] Inspect failure",
        "[IN PROGRESS] Apply repair",
    )
    assert tool_result.duration_ms == 300
    assert all("must remain hidden" not in str(item) for item in snapshot.timeline)
    assert "must remain hidden" not in str(snapshot)


def test_projection_correlates_tool_cards_and_discloses_only_bounded_activity_facts() -> None:
    projection = DashboardProjection(max_timeline=20)
    invocation = {
        "executable": "python",
        "argument_count": 3,
        "argv": ["python", "-m", "pytest", "-q"],
        "credentials_redacted": False,
        "cwd": ".",
        "timeout_seconds": 120.0,
        "verification_label": "pytest",
        "verification_kind": "test",
        "private": "PRIVATE INVOCATION",
    }
    projection.apply(
        _event(
            EventKind.TOOL_STARTED,
            step=2,
            data={
                "call_id": "provider-call-secret",
                "tool_name": "run_command",
                "public_invocation": invocation,
            },
        )
    )
    projection.apply(
        _event(
            EventKind.TOOL_FINISHED,
            step=2,
            seconds=0.5,
            data={
                "call_id": "provider-call-secret",
                "tool_name": "run_command",
                "ok": True,
                "duration_ms": 500,
                "output_chars": 42,
                "truncated": False,
                "summary": "Command exited 0 in .",
                "public_invocation": invocation,
                "metadata": {
                    "command_class": "verifier",
                    "cwd": ".",
                    "status": "exited",
                    "exit_code": 0,
                    "captured_output_bytes": 21,
                    "total_output_bytes": 21,
                    "private": "PRIVATE COMMAND METADATA",
                },
                "public_output": {
                    "captured_text": "21 passed",
                    "captured_projection_truncated": False,
                    "observation_truncated": False,
                    "credentials_redacted": False,
                },
                "raw_output": "PRIVATE COMMAND OUTPUT",
            },
        )
    )
    projection.apply(
        _event(
            EventKind.VERIFICATION_RECORDED,
            step=2,
            seconds=0.6,
            data={
                "call_id": "provider-call-secret",
                "label": "pytest",
                "kind": "test",
                "passed": True,
                "epoch": 3,
                "scopes": ["unit-tests", "route-cost"],
                "private": "PRIVATE VERIFICATION",
            },
        )
    )
    projection.apply(
        _event(
            EventKind.VERIFICATION_EVALUATED,
            step=3,
            seconds=0.7,
            data={
                "status": "verified",
                "verified": True,
                "required_scopes": ["unit-tests", "route-cost"],
                "passed_scopes": ["unit-tests", "route-cost"],
                "missing_scopes": [],
                "private": "PRIVATE GATE",
            },
        )
    )

    started, finished, recorded, gate = projection.snapshot.timeline
    assert started.activity_state == "started"
    assert finished.activity_state == "finished"
    assert started.activity_id == finished.activity_id == recorded.activity_id
    assert started.activity_id is not None
    assert started.activity_id.startswith("act_")
    assert "provider-call-secret" not in str(projection.snapshot)
    assert [(fact.label, fact.value, fact.format) for fact in finished.facts] == [
        ("Command", "python -m pytest -q", "pre"),
        ("Working directory", ".", "code"),
        ("Timeout", "120 second(s)", "text"),
        ("Verification", "test / pytest", "text"),
        ("Category", "verifier", "text"),
        ("Result", "Exited with code 0", "status"),
        ("Output status", "Captured 21 of 21 byte(s)", "text"),
        ("Captured output", "21 passed", "pre"),
    ]
    assert started.facts_complete is True
    assert finished.facts_complete is True
    assert [(fact.label, fact.value) for fact in recorded.facts] == [
        ("Verification", "pytest"),
        ("Kind", "test"),
        ("Result", "Passed"),
        ("Step", "2"),
        ("Workspace revision", "3"),
        ("Scopes", "unit-tests, route-cost"),
    ]
    assert [(fact.label, fact.value) for fact in gate.facts] == [
        ("Gate", "verified"),
        ("Required scopes", "unit-tests, route-cost"),
        ("Passed scopes", "unit-tests, route-cost"),
        ("Missing scopes", "None"),
    ]
    assert recorded.facts_complete is True
    assert gate.facts_complete is True
    assert "PRIVATE" not in str(projection.snapshot)


def test_activity_projection_fails_closed_and_enforces_fact_bounds() -> None:
    projection = DashboardProjection(max_timeline=5)
    projection.apply(
        _event(
            EventKind.TOOL_FINISHED,
            data={
                "call_id": "call-1",
                "tool_name": "run_command",
                "ok": True,
                "public_invocation": {
                    "executable": "python",
                    "argument_count": 3,
                    "display_command": "python " + ("x" * 400),
                },
                "metadata": {
                    "command_class": "verifier",
                    "cwd": "C:/Users/private/workspace",
                    "status": "exited",
                    "exit_code": 0,
                    "captured_output_bytes": 10,
                    "total_output_bytes": 20,
                    "private": "NEVER PROJECT THIS",
                },
                "truncated": True,
            },
        )
    )
    projection.apply(
        _event(
            EventKind.VERIFICATION_EVALUATED,
            step=1,
            data={
                "status": "verified",
                "required_scopes": [f"scope-{index}" for index in range(12)],
                "passed_scopes": ["scope-0", {"private": "SECRET"}],
                "missing_scopes": [],
            },
        )
    )

    command, gate = projection.snapshot.timeline
    assert command.facts_complete is False
    assert gate.facts_complete is False
    assert len(command.facts) <= 12
    assert len(gate.facts) <= 12
    assert all(len(fact.label) <= 32 and len(fact.value) <= 16_000 for fact in command.facts)
    assert command.facts[0].value == "python (3 argument(s) unavailable in old trace)"
    assert command.facts[0].format == "pre"
    assert all(fact.label != "Working directory" for fact in command.facts)
    serialized = str(projection.snapshot)
    assert "C:/Users/private" not in serialized
    assert "NEVER PROJECT THIS" not in serialized
    assert "SECRET" not in serialized


def test_command_activity_handles_missing_invocation_and_empty_output_honestly() -> None:
    projection = DashboardProjection(max_timeline=2)
    projection.apply(
        _event(
            EventKind.TOOL_FINISHED,
            data={
                "call_id": "missing-invocation",
                "tool_name": "run_command",
                "ok": True,
                "metadata": {"cwd": ".", "status": "exited", "exit_code": 0},
                "public_output": {
                    "captured_text": "",
                    "captured_projection_truncated": False,
                    "observation_truncated": False,
                    "credentials_redacted": False,
                },
            },
        )
    )
    projection.apply(
        _event(
            EventKind.TOOL_FINISHED,
            data={
                "call_id": "empty-output",
                "tool_name": "run_command",
                "ok": True,
                "public_invocation": {
                    "executable": "python",
                    "argument_count": 3,
                    "argv": ["python", "tests/unit case.py", "--name", "测试"],
                    "credentials_redacted": False,
                    "cwd": ".",
                    "timeout_seconds": 120.0,
                },
                "metadata": {
                    "cwd": ".",
                    "status": "exited",
                    "exit_code": 0,
                    "captured_output_bytes": 0,
                    "total_output_bytes": 0,
                },
                "public_output": {
                    "captured_text": "",
                    "captured_projection_truncated": False,
                    "observation_truncated": False,
                    "credentials_redacted": False,
                },
            },
        )
    )

    missing, empty = projection.snapshot.timeline
    assert missing.facts_complete is False
    assert all(fact.label != "Command" for fact in missing.facts)
    assert empty.facts_complete is True
    empty_facts = {fact.label: fact.value for fact in empty.facts}
    assert empty_facts["Command"] == 'python "tests/unit case.py" --name 测试'
    assert all(fact.label != "Captured output" for fact in empty.facts)


def test_activity_projection_never_hides_source_truncation_as_complete() -> None:
    projection = DashboardProjection(max_timeline=4)
    visible_cwd = "nested/" + ("a" * 193)
    invocation = {
        "executable": "python",
        "argument_count": 0,
        "argv": ["python"],
        "credentials_redacted": False,
        "cwd": ".",
        "timeout_seconds": 120.0,
    }
    public_output = {
        "captured_text": "ok",
        "captured_projection_truncated": False,
        "observation_truncated": False,
        "credentials_redacted": False,
    }
    projection.apply(
        _event(
            EventKind.TOOL_FINISHED,
            data={
                "call_id": "long-cwd",
                "tool_name": "run_command",
                "ok": True,
                "public_invocation": {**invocation, "cwd": visible_cwd},
                "public_output": public_output,
                "metadata": {"cwd": visible_cwd, "status": "exited", "exit_code": 0},
            },
        )
    )
    projection.apply(
        _event(
            EventKind.TOOL_FINISHED,
            data={
                "call_id": "oversized-cwd",
                "tool_name": "run_command",
                "ok": True,
                "public_invocation": {**invocation, "cwd": "nested/" + ("b" * 250)},
                "public_output": public_output,
                "metadata": {"cwd": "nested/" + ("b" * 250)},
            },
        )
    )
    projection.apply(
        _event(
            EventKind.TOOL_FINISHED,
            data={
                "call_id": "oversized-error",
                "tool_name": "run_command",
                "ok": False,
                "error_code": "failure_" + ("c" * 90),
                "public_invocation": invocation,
                "metadata": {"cwd": "."},
            },
        )
    )

    visible, oversized_cwd, oversized_error = projection.snapshot.timeline
    visible_facts = {fact.label: fact.value for fact in visible.facts}
    assert visible_facts["Working directory"] == visible_cwd
    assert "..." not in visible_facts["Working directory"]
    assert visible.facts_complete is True

    oversized_cwd_facts = {fact.label: fact.value for fact in oversized_cwd.facts}
    assert len(oversized_cwd_facts["Working directory"]) == 240
    assert oversized_cwd.facts_complete is False
    assert all("c" * 40 not in fact.value for fact in oversized_error.facts)
    assert {fact.label: fact.value for fact in oversized_error.facts}["Result"] == "Failed"
    assert oversized_error.facts_complete is False


def test_legacy_verification_activity_without_scopes_is_explicitly_incomplete() -> None:
    projection = DashboardProjection(max_timeline=2)
    projection.apply(
        _event(
            EventKind.VERIFICATION_RECORDED,
            step=4,
            data={
                "call_id": "legacy-check",
                "label": "pytest",
                "kind": "test",
                "passed": True,
                "epoch": 2,
            },
        )
    )

    entry = projection.snapshot.timeline[-1]
    assert entry.facts_complete is False
    assert all(fact.label != "Scopes" for fact in entry.facts)
    assert [(fact.label, fact.value) for fact in entry.facts] == [
        ("Verification", "pytest"),
        ("Kind", "test"),
        ("Result", "Passed"),
        ("Step", "4"),
        ("Workspace revision", "2"),
    ]


@pytest.mark.parametrize(
    ("raw_label", "visible_label"),
    [
        ("pytest   suite", "pytest suite"),
        ("v" * 300, None),
    ],
    ids=("collapsed-whitespace", "truncated-source"),
)
def test_verification_fact_sanitization_never_claims_complete_source_text(
    raw_label: str,
    visible_label: str | None,
) -> None:
    projection = DashboardProjection(max_timeline=2)
    projection.apply(
        _event(
            EventKind.VERIFICATION_RECORDED,
            data={
                "label": raw_label,
                "kind": "test",
                "passed": True,
                "epoch": 0,
                "scopes": [],
            },
        )
    )

    entry = projection.snapshot.timeline[-1]
    labels = {fact.label: fact.value for fact in entry.facts}
    if visible_label is None:
        assert "Verification" not in labels
        assert raw_label not in str(entry)
    else:
        assert labels["Verification"] == visible_label
    assert entry.facts_complete is False


def test_tampered_history_omits_commands_credentials_and_invalid_verifier_facts() -> None:
    projection = DashboardProjection(max_timeline=10)
    sentinels = (
        "sk-PROJECTOR_SENTINEL",
        "ghp_PROJECTOR_SENTINEL",
        "AKIA_PROJECTOR_SENTINEL",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJQUk9KRUNUT1IifQ.c2lnbmF0dXJlX3NlbnRpbmVs",
        "password=PROJECTOR_SENTINEL",
    )
    projection.apply(
        _event(
            EventKind.TOOL_FINISHED,
            data={
                "call_id": "tampered-command",
                "tool_name": "run_command",
                "ok": True,
                "public_invocation": {
                    "display_command": f"python --token {sentinels[0]}",
                    "executable": "../python",
                    "argument_count": 2,
                    "verification_label": sentinels[1],
                    "verification_kind": "credential_dump",
                },
                "metadata": {
                    "command_class": "verifier",
                    "cwd": ".",
                    "status": "exited",
                    "exit_code": 0,
                    "captured_output_bytes": 0,
                    "total_output_bytes": 0,
                },
            },
        )
    )
    projection.apply(
        _event(
            EventKind.VERIFICATION_RECORDED,
            step=1,
            data={
                "call_id": "tampered-command",
                "label": sentinels[3],
                "kind": "credential_dump",
                "passed": True,
                "epoch": 0,
                "scopes": [],
            },
        )
    )
    projection.apply(
        _event(
            EventKind.VERIFICATION_EVALUATED,
            step=2,
            data={
                "verified": True,
                "status": "verified",
                "evidence_labels": [*sentinels, "API_KEY verifier"],
                "evidence": [
                    {
                        "label": sentinels[2],
                        "kind": "test",
                        "passed": True,
                        "step": 1,
                        "epoch": 0,
                    },
                    {
                        "label": sentinels[4],
                        "kind": "test",
                        "passed": True,
                        "step": 1,
                        "epoch": 0,
                    },
                    {
                        "label": "API_KEY verifier",
                        "kind": "test",
                        "passed": True,
                        "step": 1,
                        "epoch": 0,
                    },
                ],
                "required_scopes": [],
                "passed_scopes": [],
                "missing_scopes": [],
            },
        )
    )

    command, recorded, _ = projection.snapshot.timeline
    assert command.facts_complete is False
    assert recorded.facts_complete is False
    assert all(fact.label != "Command" for fact in command.facts)
    assert all(fact.label not in {"Verification", "Kind"} for fact in recorded.facts)
    assert projection.snapshot.verification_labels == ("API_KEY verifier",)
    assert [item.label for item in projection.snapshot.verification_evidence] == [
        "API_KEY verifier"
    ]
    serialized = repr(projection.snapshot)
    assert "PROJECTOR_SENTINEL" not in serialized
    assert all(sentinel not in serialized for sentinel in sentinels)


def test_projection_covers_resume_failure_and_safe_fallbacks() -> None:
    projection = DashboardProjection(max_timeline=20)
    events = [
        _event(
            EventKind.RUN_RESUMED,
            data={"completed_steps": 2, "completed_tool_calls": 4},
        ),
        _event(
            EventKind.MODEL_RESPONDED,
            step=3,
            seconds=0.1,
            data={"tool_count": 0, "has_content": True},
        ),
        _event(
            EventKind.CONTEXT_COMPACTED,
            step=3,
            seconds=0.2,
            data={"compacted_blocks": 2},
        ),
        _event(EventKind.TOOL_STARTED, step=3, seconds=0.3),
        _event(
            EventKind.TOOL_FINISHED,
            step=3,
            seconds=0.4,
            data={
                "ok": False,
                "error_code": "permission_denied",
                "duration_ms": float("nan"),
            },
        ),
        _event(
            EventKind.VERIFICATION_RECORDED,
            step=3,
            seconds=0.5,
            data={"passed": False},
        ),
        _event(
            EventKind.VERIFICATION_EVALUATED,
            step=3,
            seconds=0.6,
            data={"verified": False, "status": "failed"},
        ),
        _event(
            EventKind.SESSION_CHECKPOINT_FAILED,
            step=3,
            seconds=0.7,
            data={"error_code": "checkpoint_too_large"},
        ),
        _event(
            EventKind.RUN_FAILED,
            step=3,
            seconds=0.8,
            message='{"private_reasoning":"never render me"}',
            data={"stop_reason": "model_error"},
        ),
    ]

    for event in events:
        projection.apply(event)

    snapshot = projection.snapshot
    assert snapshot.phase == "FAILED"
    assert snapshot.outcome == "UNVERIFIED"
    assert snapshot.run_failed is True
    assert snapshot.tools_started == snapshot.tools_finished == 1
    assert snapshot.tools_failed == 1
    assert snapshot.verification_status == "failed"
    assert snapshot.verification_labels == ()
    assert snapshot.stop_reason == "model_error"
    assert snapshot.elapsed_seconds == pytest.approx(0.8)
    assert snapshot.timeline[-1].headline == "Checkpoint not saved"
    assert snapshot.timeline[-1].detail == "CHECKPOINT TOO LARGE"
    assert all("private_reasoning" not in str(item) for item in snapshot.timeline)


def test_projection_exposes_only_bounded_model_retry_facts() -> None:
    projection = DashboardProjection(max_timeline=4)
    projection.apply(_event(EventKind.RUN_STARTED))

    entry = projection.apply(
        _event(
            EventKind.MODEL_RETRYING,
            step=2,
            seconds=1,
            message="provider response and credentials must stay private",
            data={
                "attempt": 1,
                "next_attempt": 2,
                "max_attempts": 3,
                "delay_seconds": 0.5,
                "error_code": "model_request_transient",
                "provider_body": "must not render",
            },
        )
    )

    assert entry is not None
    assert projection.snapshot.phase == "RETRYING"
    assert entry.level == "warning"
    assert entry.headline == "Transient model failure; retry scheduled"
    assert entry.detail == "Attempt 2 of 3 · after 0.5s · MODEL REQUEST TRANSIENT"
    assert "provider" not in entry.detail.casefold()


def test_projection_distinguishes_protocol_correction_without_exposing_response() -> None:
    projection = DashboardProjection(max_timeline=4)
    projection.apply(_event(EventKind.RUN_STARTED))

    entry = projection.apply(
        _event(
            EventKind.MODEL_RETRYING,
            step=2,
            seconds=1,
            message="malformed provider arguments must stay private",
            data={
                "attempt": 1,
                "next_attempt": 2,
                "max_attempts": 3,
                "delay_seconds": 0.0,
                "error_code": "model_response_invalid",
                "retry_kind": "protocol_correction",
                "raw_arguments": "TEST_PRIVATE_MALFORMED_JSON",
            },
        )
    )

    assert entry is not None
    assert projection.snapshot.phase == "RETRYING"
    assert entry.level == "warning"
    assert entry.headline == "Invalid model response; protocol correction scheduled"
    assert entry.detail == "Attempt 2 of 3 · after 0s · MODEL RESPONSE INVALID"
    assert "TEST_PRIVATE" not in str(entry)


def test_projection_treats_verification_closeout_as_verification_not_model_failure() -> None:
    projection = DashboardProjection(max_timeline=4)
    projection.apply(_event(EventKind.RUN_STARTED))

    entry = projection.apply(
        _event(
            EventKind.MODEL_RETRYING,
            step=2,
            seconds=1,
            message="host instruction must stay private",
            data={
                "retry_kind": "verification_closeout",
                "remaining_model_turns": 2,
                "instruction_chars": 417,
                "provider_private": "TEST_PRIVATE_CLOSEOUT_SENTINEL",
            },
        )
    )

    assert entry is not None
    assert projection.snapshot.phase == "VERIFYING"
    assert entry.category == "VERIFY"
    assert entry.level == "warning"
    assert entry.headline == "Final response deferred; verification scheduled"
    assert entry.detail == "Fresh verification is required"
    assert "failure" not in str(entry).casefold()
    assert "TEST_PRIVATE" not in str(entry)


def test_projection_whitelists_run_limits_and_treats_batch_rejection_as_retry() -> None:
    projection = DashboardProjection(max_timeline=5)
    projection.apply(
        _event(
            EventKind.RUN_STARTED,
            data={
                "limits": {
                    "max_model_turns": 20,
                    "max_calls_per_turn": 8,
                    "max_total_tool_calls": 40,
                    "provider_private": "SECRET",
                }
            },
        )
    )

    entry = projection.apply(
        _event(
            EventKind.TOOL_BATCH_REJECTED,
            step=2,
            data={
                "requested_calls": 9,
                "max_calls_per_turn": 8,
                "rejection_count": 1,
                "max_rejections": 3,
                "raw_tool_arguments": "SECRET",
            },
        )
    )

    snapshot = projection.snapshot
    assert snapshot.limits is not None
    assert snapshot.limits.max_model_turns == 20
    assert snapshot.limits.max_calls_per_turn == 8
    assert snapshot.limits.max_total_tool_calls == 40
    assert snapshot.phase == "REPLANNING"
    assert snapshot.tools_started == snapshot.tools_finished == snapshot.tools_failed == 0
    assert entry is not None
    assert entry.category == "MODEL"
    assert entry.level == "warning"
    assert entry.headline == "Tool batch too large; split retry requested"
    assert entry.detail == (
        "Requested 9 tool calls; per-turn limit 8; split retry requested; rejection 1 of 3"
    )
    assert "SECRET" not in str(snapshot)

    legacy = DashboardProjection()
    legacy.apply(_event(EventKind.RUN_STARTED, data={"limits": {"max_model_turns": 20}}))
    assert legacy.snapshot.limits is None

    resumed = DashboardProjection()
    resumed.apply(
        _event(
            EventKind.RUN_RESUMED,
            data={
                "limits": {
                    "max_model_turns": 30,
                    "max_calls_per_turn": 6,
                    "max_total_tool_calls": 35,
                }
            },
        )
    )
    assert resumed.snapshot.limits is not None
    assert resumed.snapshot.limits.max_model_turns == 30


def test_projection_bounds_timeline_and_rejects_mixed_runs() -> None:
    projection = DashboardProjection(task_label="\n", max_timeline=2)
    projection.apply(_event(EventKind.RUN_STARTED))
    projection.apply(_event(EventKind.MODEL_REQUESTED, step=1, seconds=1))
    projection.apply(_event(EventKind.MODEL_RESPONDED, step=1, seconds=2))

    assert projection.snapshot.task_label == "Coding task"
    assert len(projection.snapshot.timeline) == 2
    assert projection.snapshot.timeline[0].headline == "Selecting the next action"
    with pytest.raises(ValueError, match="only contain one run"):
        projection.apply(_event(EventKind.RUN_STARTED, run_id="run-2"))
    with pytest.raises(ValueError, match="at least 1"):
        DashboardProjection(max_timeline=0)
    for invalid_preview_lines in (-1, 81):
        with pytest.raises(ValueError, match="expanded_mutation_preview_lines"):
            DashboardProjection(expanded_mutation_preview_lines=invalid_preview_lines)


def test_projection_handles_missing_and_unknown_verification_status() -> None:
    missing = DashboardProjection()
    missing.apply(_event(EventKind.RUN_STARTED))
    missing.apply(
        _event(
            EventKind.VERIFICATION_EVALUATED,
            step=1,
            seconds=1,
            data={"verified": False, "status": "missing"},
        )
    )
    missing.apply(
        _event(
            EventKind.RUN_FINISHED,
            step=1,
            seconds=2,
            data={"verified": False, "status": "missing"},
        )
    )
    assert missing.snapshot.outcome == "UNVERIFIED"
    assert missing.snapshot.verification_status == "missing"

    unknown = DashboardProjection()
    unknown.apply(_event(EventKind.RUN_STARTED))
    unknown.apply(
        _event(
            EventKind.RUN_FINISHED,
            step=1,
            data={"verified": False, "status": "invented"},
        )
    )
    assert unknown.snapshot.verification_status == "unverified"

    checks_only = DashboardProjection()
    checks_only.apply(_event(EventKind.RUN_STARTED))
    checks_only.apply(
        _event(
            EventKind.RUN_FINISHED,
            step=1,
            data={
                "verified": False,
                "status": "checks_only",
                "evidence_labels": ["pytest"],
            },
        )
    )
    assert checks_only.snapshot.outcome == "UNVERIFIED"
    assert checks_only.snapshot.verification_status == "checks_only"
    assert checks_only.snapshot.verification_labels == ("pytest",)


def test_verification_reports_replace_labels_and_stale_state_hides_old_evidence() -> None:
    projection = DashboardProjection(max_timeline=20)
    projection.apply(_event(EventKind.RUN_STARTED))
    projection.apply(
        _event(
            EventKind.VERIFICATION_RECORDED,
            step=1,
            data={"passed": True, "kind": "test", "label": "old pytest", "epoch": 0},
        )
    )
    assert projection.snapshot.verification_labels == ("old pytest",)

    projection.apply(_event(EventKind.VERIFICATION_INVALIDATED, step=2, data={"epoch": 1}))
    assert projection.snapshot.verification_status == "stale"
    assert not projection.snapshot.verification_labels
    assert not projection.snapshot.verification_evidence
    assert projection.snapshot.verification_epoch == 1
    assert projection.snapshot.invalidation_count == 1

    projection.apply(
        _event(
            EventKind.VERIFICATION_EVALUATED,
            step=2,
            data={
                "verified": False,
                "status": "stale",
                "evidence_labels": ["old pytest", {"private": "SECRET EVIDENCE"}],
                "raw_report": "SECRET RAW REPORT",
            },
        )
    )
    assert not projection.snapshot.verification_labels
    assert "SECRET" not in str(projection.snapshot)

    projection.apply(
        _event(
            EventKind.VERIFICATION_EVALUATED,
            step=3,
            data={
                "verified": True,
                "status": "verified",
                "evidence_labels": ["fresh pytest", "fresh pytest", 3],
                "epoch": 1,
                "invalidation_count": 1,
                "evidence": [
                    {
                        "label": "fresh pytest",
                        "kind": "test",
                        "passed": True,
                        "step": 3,
                        "epoch": 1,
                    },
                    {"label": "unsafe", "kind": {"private": "SECRET"}},
                ],
            },
        )
    )
    assert projection.snapshot.verification_labels == ("fresh pytest",)
    assert projection.snapshot.verification_evidence[0].label == "fresh pytest"

    projection.apply(
        _event(
            EventKind.RUN_FINISHED,
            step=3,
            data={
                "verified": True,
                "status": "verified",
                "evidence_labels": ["final pytest"],
                "private_labels": ["SECRET FINAL EVIDENCE"],
            },
        )
    )
    assert projection.snapshot.verification_labels == ("final pytest",)
    assert "SECRET" not in str(projection.snapshot)


def test_non_tty_sink_prints_stable_timeline_and_verified_card() -> None:
    stream = StringIO()
    sink = DashboardEventSink(
        _plain_console(stream),
        live=False,
        task_label="Repair [bold] safely",
        max_timeline=20,
    )
    sink.emit(_event(EventKind.RUN_STARTED, message="SECRET START MESSAGE"))
    sink.emit(_event(EventKind.MODEL_REQUESTED, step=1, message="SECRET MODEL REQUEST"))
    sink.emit(
        _event(
            EventKind.MODEL_RESPONDED,
            step=1,
            message="SECRET MODEL RESPONSE",
            data={"tool_count": 1, "has_content": True},
        )
    )
    sink.emit(
        _event(
            EventKind.TOOL_STARTED,
            step=1,
            seconds=0.1,
            data={"call_id": "write-1", "tool_name": "replace_text"},
        )
    )
    sink.emit(
        _event(
            EventKind.TOOL_FINISHED,
            step=1,
            seconds=0.2,
            message="SECRET TOOL OUTPUT",
            data={
                "call_id": "write-1",
                "tool_name": "replace_text",
                "ok": True,
                "summary": "Changed one file",
                "duration_ms": 12.5,
                "preview": ["- old line", "+ new line"],
                "metadata": {
                    "path": "src/pricing.py",
                    "changed": True,
                    "added_lines": 1,
                    "removed_lines": 1,
                    "mutation_revision": 1,
                    "change_kind": "update",
                    "after_sha256": "SECRET HASH",
                },
                "output": "SECRET RAW OUTPUT",
            },
        )
    )
    sink.emit(
        _event(
            EventKind.VERIFICATION_RECORDED,
            step=2,
            seconds=0.5,
            data={"passed": True, "kind": "test", "label": "pytest", "epoch": 1},
        )
    )
    sink.emit(
        _event(
            EventKind.RUN_FINISHED,
            step=2,
            seconds=1,
            message='{"private":"SECRET FINAL"}',
            data={
                "verified": True,
                "status": "verified",
                "epoch": 1,
                "invalidation_count": 1,
                "evidence_labels": ["pytest"],
                "evidence": [
                    {
                        "label": "pytest",
                        "kind": "test",
                        "passed": True,
                        "step": 2,
                        "epoch": 1,
                        "private": "SECRET EVIDENCE",
                    }
                ],
            },
        )
    )

    output = stream.getvalue()
    assert "[INFO]" in output
    assert "[PASS]" in output
    assert "replace_text completed" in output
    assert "- old line" in output
    assert "+ new line" in output
    assert "FINAL RESULT" in output
    assert "VERIFIED" in output
    assert "Changes: src/pricing.py (+1/-1)" in output
    assert "Evidence: pytest PASS (test, step 2, workspace revision 1)" in output
    assert "Freshness: evidence matches workspace revision 1" in output
    assert "Selecting the next action" not in output
    assert "Action selected" not in output
    assert "SECRET" not in output
    assert "\x1b" not in output
    assert sink.snapshot.outcome == "VERIFIED"
    assert sink.snapshot.changed_files[0].path == "src/pricing.py"
    assert sink.snapshot.verification_evidence[0].step == 2
    assert sink.snapshot.verification_epoch == 1
    assert sink.snapshot.invalidation_count == 1


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("checks_only", "completion contract is incomplete"),
        ("missing", "No current trusted verification evidence"),
        ("failed", "latest trusted verification evidence failed"),
        ("stale", "workspace changed after the latest passing evidence"),
    ],
)
def test_final_card_explains_each_unverified_status(status: str, expected: str) -> None:
    stream = StringIO()
    sink = DashboardEventSink(_plain_console(stream), live=False)
    sink.emit(_event(EventKind.RUN_STARTED))
    sink.emit(
        _event(
            EventKind.RUN_FINISHED,
            step=1,
            seconds=1,
            data={"verified": False, "status": status},
        )
    )

    assert expected in stream.getvalue()
    assert sink.snapshot.outcome == "UNVERIFIED"


def test_failed_run_card_is_unverified_and_does_not_repeat() -> None:
    stream = StringIO()
    sink = DashboardEventSink(_plain_console(stream), live=False)
    sink.emit(_event(EventKind.RUN_STARTED))
    failure = _event(
        EventKind.RUN_FAILED,
        step=1,
        seconds=1,
        message="SECRET TERMINAL MESSAGE",
        data={
            "stop_reason": "command_control_failed",
            "raw_error": "SECRET RAW ERROR",
            "private": {"reason": "SECRET PRIVATE REASON"},
        },
    )
    sink.emit(failure)
    sink.emit(failure)

    output = stream.getvalue()
    assert output.count("FINAL RESULT") == 1
    assert "UNVERIFIED" in output
    assert "Run state: FAILED" in output
    assert "Stop reason: COMMAND CONTROL FAILED" in output
    assert "stopped before trustworthy completion" in output
    assert "SECRET" not in output
    assert sink.snapshot.stop_reason == "command_control_failed"


def test_failed_run_clears_previously_passing_structured_evidence() -> None:
    stream = StringIO()
    sink = DashboardEventSink(_plain_console(stream), live=False)
    sink.emit(_event(EventKind.RUN_STARTED))
    sink.emit(
        _event(
            EventKind.VERIFICATION_RECORDED,
            step=2,
            data={
                "passed": True,
                "kind": "test",
                "label": "pytest",
                "epoch": 1,
            },
        )
    )
    sink.emit(
        _event(
            EventKind.VERIFICATION_EVALUATED,
            step=2,
            data={
                "verified": True,
                "status": "verified",
                "evidence_labels": ["pytest"],
            },
        )
    )
    sink.emit(
        _event(
            EventKind.RUN_FAILED,
            step=3,
            data={"stop_reason": "model_error"},
        )
    )

    output = stream.getvalue()
    assert sink.snapshot.verification_status == "unverified"
    assert sink.snapshot.verification_labels == ()
    assert sink.snapshot.verification_evidence == ()
    assert "pytest PASS" not in output
    assert "Freshness:" not in output


def test_full_dashboard_render_has_header_timeline_and_gate() -> None:
    stream = StringIO()
    console = _plain_console(stream, width=120)
    sink = DashboardEventSink(console, live=False, task_label="Demo repair")
    console.print(sink.render())
    sink.emit(_event(EventKind.RUN_STARTED))
    sink.emit(
        _event(
            EventKind.STATE_CHANGED,
            step=1,
            data={"current": "acting"},
        )
    )
    console.print(sink.render())

    output = stream.getvalue()
    assert "CODING AGENT" in output
    assert "ACTIVITY TIMELINE" in output
    assert "CURRENT PLAN" in output
    assert "No structured plan recorded yet" in output
    assert "LATEST CHANGE" in output
    assert "No workspace mutation recorded yet" in output
    assert "VERIFICATION GATE" in output
    assert "Waiting for the first runtime event" in output
    assert "Demo repair" in output
    assert "Step 1 active" in output
    assert "TASK" in output


def test_latest_change_survives_timeline_eviction_and_renders_modified_lines() -> None:
    stream = StringIO()
    console = _plain_console(stream, width=120)
    sink = DashboardEventSink(console, live=False, max_timeline=2)
    sink.emit(_event(EventKind.RUN_STARTED))
    sink.emit(
        _event(
            EventKind.TOOL_FINISHED,
            step=1,
            data={
                "call_id": "change-1",
                "tool_name": "replace_text",
                "ok": True,
                "summary": "Changed pricing.py",
                "preview": (
                    "Change chg_1 was applied.\nDiff preview:\n"
                    "--- a/pricing.py\n+++ b/pricing.py\n@@ -1 +1 @@\n"
                    "-return discounted - discount\n+return discounted\n"
                ),
            },
        )
    )
    for step in range(2, 6):
        sink.emit(_event(EventKind.MODEL_REQUESTED, step=step, seconds=step))

    snapshot = sink.snapshot
    assert len(snapshot.timeline) == 2
    assert all(entry.category == "MODEL" for entry in snapshot.timeline)
    assert snapshot.latest_change is not None
    assert snapshot.latest_change.preview[-2:] == (
        "-return discounted - discount",
        "+return discounted",
    )
    console.print(sink.render())
    output = stream.getvalue()
    assert "LATEST CHANGE" in output
    assert "-return discounted - discount" in output
    assert "+return discounted" in output


def test_current_plan_survives_timeline_eviction_and_only_success_replaces_it() -> None:
    stream = StringIO()
    console = _plain_console(stream, width=120)
    sink = DashboardEventSink(console, live=False, max_timeline=2)
    sink.emit(_event(EventKind.RUN_STARTED))
    plan = [
        {"status": "completed" if index == 1 else "pending", "step": f"Plan step {index}"}
        for index in range(1, 10)
    ]
    sink.emit(
        _event(
            EventKind.TOOL_FINISHED,
            step=1,
            data={
                "tool_name": "update_plan",
                "ok": True,
                "summary": "Plan updated",
                "preview": {"plan": plan},
                "raw_output": "SECRET PLAN OUTPUT",
                "private": {"analysis": "SECRET PLAN REASONING"},
            },
        )
    )
    expected = (
        "[COMPLETED] Plan step 1",
        "[PENDING] Plan step 2",
        "[PENDING] Plan step 3",
        "[PENDING] Plan step 4",
        "[PENDING] Plan step 5",
        "[PENDING] Plan step 6",
        "[PENDING] Plan step 7",
        "[PENDING] Plan step 8",
    )
    assert sink.snapshot.plan_lines == expected

    sink.emit(
        _event(
            EventKind.TOOL_FINISHED,
            step=2,
            data={
                "tool_name": "update_plan",
                "ok": False,
                "error_code": "invalid_plan",
                "raw_output": "SECRET FAILED PLAN",
                "private": {"analysis": "SECRET FAILED REASONING"},
            },
        )
    )
    for step in range(3, 7):
        sink.emit(_event(EventKind.MODEL_REQUESTED, step=step, seconds=step))

    snapshot = sink.snapshot
    assert len(snapshot.timeline) == 2
    assert all(entry.category == "MODEL" for entry in snapshot.timeline)
    assert snapshot.plan_lines == expected
    console.print(sink.render())
    output = stream.getvalue()
    assert "CURRENT PLAN" in output
    assert "Plan step 1" in output
    assert "Plan step 8" in output
    assert "Plan step 9" not in output
    assert "SECRET" not in output


def test_host_can_delay_final_card_until_all_other_output_is_complete() -> None:
    stream = StringIO()
    console = _plain_console(stream)
    sink = DashboardEventSink(console, live=False, auto_final_card=False)

    assert sink.print_final_card() is False
    sink.emit(_event(EventKind.RUN_STARTED))
    sink.emit(
        _event(
            EventKind.RUN_FINISHED,
            step=1,
            seconds=1,
            data={"verified": True, "status": "verified"},
        )
    )
    assert "FINAL RESULT" not in stream.getvalue()

    console.print("HOST RESPONSE")
    assert sink.print_final_card() is True
    assert sink.print_final_card() is False
    output = stream.getvalue()
    assert output.count("FINAL RESULT") == 1
    assert output.index("HOST RESPONSE") < output.index("FINAL RESULT")


def test_dashboard_output_is_safe_for_an_ascii_console() -> None:
    raw_stream = BytesIO()
    ascii_stream = TextIOWrapper(raw_stream, encoding="ascii")
    console = Console(file=ascii_stream, force_terminal=False, color_system=None, width=100)
    sink = DashboardEventSink(console, live=False, task_label="Fix Ω")

    sink.emit(_event(EventKind.RUN_STARTED))
    sink.emit(
        _event(
            EventKind.TOOL_FINISHED,
            step=1,
            data={
                "tool_name": "read_file",
                "ok": True,
                "summary": "Read emoji 🚀",
                "preview": "Ω preview",
            },
        )
    )
    sink.emit(
        _event(
            EventKind.RUN_FINISHED,
            step=1,
            data={"verified": False, "status": "missing"},
        )
    )
    ascii_stream.flush()

    output = raw_stream.getvalue().decode("ascii")
    assert "?" in output
    assert "FINAL RESULT" in output


def test_preview_is_bounded_and_mappings_are_never_dumped_as_json() -> None:
    projection = DashboardProjection(max_timeline=20)
    projection.apply(_event(EventKind.RUN_STARTED))
    previews: list[object] = [
        {"lines": ["line 1", "line 2"]},
        {"status": "pending", "path": "src/app.py"},
        {"secret": "must-not-render"},
        ["one", 2, {"status": "done", "summary": "three"}, "four\nfive\nsix\nseven"],
        "single\x00 control",
    ]
    for index, preview in enumerate(previews, start=1):
        projection.apply(
            _event(
                EventKind.TOOL_FINISHED,
                step=index,
                seconds=index,
                data={
                    "tool_name": "tool",
                    "ok": True,
                    "summary": "safe",
                    "preview": preview,
                },
            )
        )

    result_previews = [entry.preview for entry in projection.snapshot.timeline if entry.preview]
    assert result_previews[0] == ("line 1", "line 2")
    assert result_previews[1] == ("[PENDING] src/app.py",)
    assert result_previews[2] == ("one", "[DONE] three", "four", "five", "six", "seven")
    assert result_previews[3] == ("single control",)
    assert "must-not-render" not in str(result_previews)


def test_mutation_preview_prioritizes_the_actual_changed_lines() -> None:
    projection = DashboardProjection(max_timeline=5)
    projection.apply(_event(EventKind.RUN_STARTED))
    projection.apply(
        _event(
            EventKind.TOOL_FINISHED,
            step=1,
            data={
                "tool_name": "replace_text",
                "ok": True,
                "summary": "Changed pricing.py",
                "preview": (
                    "Change chg_1 was applied.\nDiff preview:\n"
                    "--- a/pricing.py\n+++ b/pricing.py\n@@ -1 +1 @@\n"
                    " context\n-old total\n+new total\n"
                ),
            },
        )
    )

    preview = projection.snapshot.timeline[-1].preview
    assert preview == (
        "--- a/pricing.py",
        "+++ b/pricing.py",
        "@@ -1 +1 @@",
        "-old total",
        "+new total",
    )
    assert projection.snapshot.latest_change is not None
    assert projection.snapshot.latest_change.expanded_preview == ()


def test_workspace_change_ledger_retains_order_after_timeline_eviction() -> None:
    projection = DashboardProjection(
        max_timeline=1,
        expanded_mutation_preview_lines=80,
    )
    projection.apply(_event(EventKind.RUN_STARTED))
    for step, path in ((1, "pricing.py"), (2, "checkout.py")):
        projection.apply(
            _event(
                EventKind.TOOL_FINISHED,
                step=step,
                seconds=step,
                data={
                    "call_id": f"change-{step}",
                    "tool_name": "replace_text",
                    "ok": True,
                    "summary": f"Changed {path}",
                    "preview": (
                        f"Diff preview:\n--- a/{path}\n+++ b/{path}\n"
                        f"@@ -1 +1 @@\n-old {step}\n+new {step}"
                    ),
                    "truncated": False,
                    "metadata": {"diff_complete": True},
                },
            )
        )
    projection.apply(_event(EventKind.MODEL_REQUESTED, step=3, seconds=3))

    snapshot = projection.snapshot
    assert len(snapshot.timeline) == 1
    assert snapshot.timeline[0].category == "MODEL"
    assert [change.detail for change in snapshot.workspace_changes] == [
        "Changed pricing.py",
        "Changed checkout.py",
    ]
    assert [change.expanded_preview[-1] for change in snapshot.workspace_changes] == [
        "+new 1",
        "+new 2",
    ]
    assert snapshot.workspace_changes_complete is True
    assert snapshot.omitted_change_count == 0
    assert snapshot.latest_change == snapshot.workspace_changes[-1]


def test_workspace_change_ledger_ignores_failed_or_unsafe_mutation_previews() -> None:
    projection = DashboardProjection(expanded_mutation_preview_lines=80)
    projection.apply(_event(EventKind.RUN_STARTED))
    projection.apply(
        _event(
            EventKind.TOOL_FINISHED,
            step=1,
            data={
                "tool_name": "replace_text",
                "ok": False,
                "summary": "Edit failed",
                "preview": "Diff preview:\n--- a/file.py\n+++ b/file.py\n+unsafe",
            },
        )
    )
    projection.apply(
        _event(
            EventKind.TOOL_FINISHED,
            step=2,
            data={
                "tool_name": "write_file",
                "ok": True,
                "summary": "No public preview",
                "preview": {"private": "must-not-render"},
            },
        )
    )

    snapshot = projection.snapshot
    assert snapshot.workspace_changes == ()
    assert snapshot.workspace_changes_complete is True
    assert snapshot.omitted_change_count == 0
    assert snapshot.latest_change is None


def test_workspace_change_ledger_is_bounded_and_reports_omissions() -> None:
    projection = DashboardProjection(
        max_timeline=1,
        expanded_mutation_preview_lines=1,
    )
    projection.apply(_event(EventKind.RUN_STARTED))
    total_changes = dashboard_module.MAX_WORKSPACE_CHANGES + 2
    for step in range(1, total_changes + 1):
        projection.apply(
            _event(
                EventKind.TOOL_FINISHED,
                step=step,
                data={
                    "tool_name": "undo_change",
                    "ok": True,
                    "summary": f"Undid change {step}",
                    "preview": f"Diff preview:\n--- a/file.py\n+++ b/file.py\n+revision {step}",
                },
            )
        )

    snapshot = projection.snapshot
    assert len(snapshot.workspace_changes) == dashboard_module.MAX_WORKSPACE_CHANGES
    assert snapshot.workspace_changes[0].step == 3
    assert snapshot.workspace_changes[-1].step == total_changes
    assert snapshot.workspace_changes_complete is False
    assert snapshot.omitted_change_count == 2
    assert snapshot.latest_change == snapshot.workspace_changes[-1]


def test_web_mutation_preview_expands_safely_without_growing_the_timeline() -> None:
    projection = DashboardProjection(
        max_timeline=5,
        expanded_mutation_preview_lines=80,
    )
    projection.apply(_event(EventKind.RUN_STARTED))
    diff_lines = [
        "--- a/pricing.py",
        "+++ b/pricing.py",
        "@@ -1,100 +1,100 @@",
        *(f"+new line {index}" for index in range(100)),
    ]
    projection.apply(
        _event(
            EventKind.TOOL_FINISHED,
            step=1,
            data={
                "tool_name": "write_file",
                "ok": True,
                "summary": "Changed pricing.py",
                "preview": "Change applied.\nDiff preview:\n" + "\n".join(diff_lines),
                "truncated": False,
                "metadata": {"diff_complete": True},
            },
        )
    )

    timeline_entry = projection.snapshot.timeline[-1]
    latest_change = projection.snapshot.latest_change
    assert timeline_entry.expanded_preview == ()
    assert len(timeline_entry.preview) == 6
    assert latest_change is not None
    assert projection.snapshot.workspace_changes == (latest_change,)
    assert projection.snapshot.workspace_changes_complete is True
    assert projection.snapshot.omitted_change_count == 0
    assert len(latest_change.expanded_preview) == 80
    assert latest_change.expanded_preview[0] == "--- a/pricing.py"
    assert latest_change.expanded_preview[-1] == "+new line 99"
    assert any(
        "omitted from expanded Diff preview" in line for line in latest_change.expanded_preview
    )
    assert latest_change.expanded_preview_complete is False


def test_web_mutation_preview_honors_a_single_line_projection_limit() -> None:
    projection = DashboardProjection(expanded_mutation_preview_lines=1)
    projection.apply(_event(EventKind.RUN_STARTED))
    projection.apply(
        _event(
            EventKind.TOOL_FINISHED,
            step=1,
            data={
                "tool_name": "write_file",
                "ok": True,
                "summary": "Changed pricing.py",
                "preview": "Diff preview:\n--- a/pricing.py\n+++ b/pricing.py\n+new line",
            },
        )
    )

    latest_change = projection.snapshot.latest_change
    assert latest_change is not None
    assert latest_change.expanded_preview == ("...[3 lines omitted from expanded Diff preview]...",)
    assert latest_change.expanded_preview_complete is False


def test_web_mutation_preview_marks_a_truncated_line_as_incomplete() -> None:
    projection = DashboardProjection(expanded_mutation_preview_lines=80)
    projection.apply(_event(EventKind.RUN_STARTED))
    projection.apply(
        _event(
            EventKind.TOOL_FINISHED,
            step=1,
            data={
                "tool_name": "replace_text",
                "ok": True,
                "summary": "Changed pricing.py",
                "preview": "Diff preview:\n--- a/pricing.py\n+++ b/pricing.py\n+" + ("x" * 300),
                "truncated": False,
                "metadata": {"diff_complete": True},
            },
        )
    )

    latest_change = projection.snapshot.latest_change
    assert latest_change is not None
    assert latest_change.expanded_preview[-1].endswith("...")
    assert len(latest_change.expanded_preview[-1]) == 240
    assert latest_change.expanded_preview_complete is False


@pytest.mark.parametrize(
    "source_signal",
    [
        {"truncated": True},
        {"metadata": {"diff_complete": False}},
        {"preview": "Diff preview:\n...[2 lines omitted from Diff preview]..."},
        {"preview": {"diff": "--- a/pricing.py\n+++ b/pricing.py\n+new line"}},
    ],
    ids=("tool-output", "diff-budget", "preview-marker", "structured-fallback"),
)
def test_web_mutation_preview_propagates_source_truncation(
    source_signal: dict[str, object],
) -> None:
    projection = DashboardProjection(expanded_mutation_preview_lines=80)
    projection.apply(_event(EventKind.RUN_STARTED))
    data: dict[str, object] = {
        "tool_name": "replace_text",
        "ok": True,
        "summary": "Changed pricing.py",
        "preview": "Diff preview:\n--- a/pricing.py\n+++ b/pricing.py\n+new line",
        "truncated": False,
        "metadata": {"diff_complete": True},
    }
    data.update(source_signal)
    projection.apply(_event(EventKind.TOOL_FINISHED, step=1, data=data))

    latest_change = projection.snapshot.latest_change
    assert latest_change is not None
    assert latest_change.expanded_preview_complete is False


class FakeLive:
    instances: ClassVar[list[FakeLive]] = []

    def __init__(self, renderable: object, **kwargs: object) -> None:
        self.renderable = renderable
        self.kwargs = kwargs
        self.started = 0
        self.updated = 0
        self.stopped = 0
        self.__class__.instances.append(self)

    def start(self, *, refresh: bool = False) -> None:
        assert refresh is True
        self.started += 1

    def update(self, renderable: object, *, refresh: bool = False) -> None:
        assert refresh is True
        self.renderable = renderable
        self.updated += 1

    def stop(self) -> None:
        self.stopped += 1


def test_live_mode_starts_updates_stops_and_context_manager_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeLive.instances.clear()
    monkeypatch.setattr(dashboard_module, "Live", FakeLive)
    stream = StringIO()
    sink = DashboardEventSink(_plain_console(stream), live=True, refresh_per_second=5)

    with sink as opened:
        assert opened is sink
        sink.emit(_event(EventKind.RUN_STARTED))
        sink.emit(_event(EventKind.MODEL_REQUESTED, step=1))
        sink.emit(
            _event(
                EventKind.RUN_FINISHED,
                step=1,
                data={"verified": False, "status": "missing"},
            )
        )

    live = FakeLive.instances[0]
    assert live.started == 1
    assert live.updated == 2
    assert live.stopped == 1
    assert live.kwargs["auto_refresh"] is False
    assert live.kwargs["refresh_per_second"] == 5
    assert "FINAL RESULT" in stream.getvalue()


def test_auto_live_selection_and_constructor_validation() -> None:
    terminal_console = Console(file=StringIO(), force_terminal=True)
    recording_console = Console(file=StringIO(), force_terminal=False)

    assert DashboardEventSink(terminal_console).uses_live_rendering is True
    assert DashboardEventSink(recording_console).uses_live_rendering is False
    with pytest.raises(ValueError, match="positive"):
        DashboardEventSink(recording_console, refresh_per_second=0)


def test_sink_resets_cleanly_for_a_second_run() -> None:
    stream = StringIO()
    sink = DashboardEventSink(_plain_console(stream), live=False)
    sink.emit(_event(EventKind.RUN_STARTED, run_id="first"))
    sink.emit(
        _event(
            EventKind.RUN_FINISHED,
            run_id="first",
            data={"verified": False, "status": "missing"},
        )
    )
    sink.emit(_event(EventKind.RUN_STARTED, run_id="second"))
    sink.emit(
        _event(
            EventKind.RUN_FINISHED,
            run_id="second",
            data={"verified": True, "status": "verified"},
        )
    )

    assert sink.snapshot.run_id == "second"
    assert sink.snapshot.outcome == "VERIFIED"
    assert stream.getvalue().count("FINAL RESULT") == 2
