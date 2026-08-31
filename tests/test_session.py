"""Tests for passive, versioned, and bounded session checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from coding_agent.models import (
    ChatMessage,
    MessageRole,
    StopReason,
    ToolCall,
    VerificationKind,
)
from coding_agent.run_memory import RunMemorySnapshot, VerificationMemoryFact
from coding_agent.session import (
    SessionBoundary,
    SessionCheckpoint,
    SessionError,
    SessionStore,
)


def _tool_transcript(
    *, secret_arguments: dict[str, object] | None = None
) -> tuple[ChatMessage, ...]:
    call = ToolCall(
        id="read-1",
        name="read_file",
        arguments=secret_arguments or {"path": "src/example.py"},
    )
    return (
        ChatMessage(role=MessageRole.SYSTEM, content="system"),
        ChatMessage(role=MessageRole.USER, content="inspect the repository"),
        ChatMessage(role=MessageRole.ASSISTANT, content="I will inspect it.", tool_calls=(call,)),
        ChatMessage(
            role=MessageRole.TOOL,
            content='{"call_id":"read-1","tool_name":"read_file","ok":true,"output":"ok"}',
            tool_call_id=call.id,
            tool_name=call.name,
        ),
    )


def _checkpoint(
    *,
    run_id: str = "run-1",
    messages: tuple[ChatMessage, ...] | None = None,
    task: str = "inspect the repository",
    completed_steps: int = 1,
    completed_tool_calls: int = 1,
    secret_arguments: dict[str, object] | None = None,
    run_memory: RunMemorySnapshot | None = None,
) -> SessionCheckpoint:
    transcript = messages or _tool_transcript(secret_arguments=secret_arguments)
    return SessionCheckpoint(
        run_id=run_id,
        task=task,
        system_prompt="system",
        messages=transcript,
        completed_steps=completed_steps,
        completed_tool_calls=completed_tool_calls,
        run_memory=run_memory or RunMemorySnapshot(revision=0),
        stop_boundary=SessionBoundary.READY_FOR_MODEL,
    )


def test_checkpoint_round_trip_is_passive_and_requires_fresh_verification(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "state")
    checkpoint = _checkpoint()

    saved = store.save(checkpoint)
    loaded = store.load(checkpoint.run_id)

    assert saved == (tmp_path / "state" / "run-1.json").resolve()
    assert loaded.checkpoint == checkpoint
    assert loaded.verification_evidence_restored is False
    assert loaded.requires_reverification is True
    assert loaded.auto_replay_tool_calls is False
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 3
    assert loaded.source_schema_version == 3
    assert loaded.schema_migrated is False
    assert loaded.checkpoint.completed_work_tool_calls == 1
    assert loaded.checkpoint.completed_verification_tool_calls == 0
    assert "verification_evidence" not in saved.read_text(encoding="utf-8")
    assert "verification_report" not in saved.read_text(encoding="utf-8")
    assert "environment" not in payload


def test_checkpoint_counts_only_tool_calls_that_were_accepted_for_execution(
    tmp_path: Path,
) -> None:
    calls = tuple(
        ToolCall(id=f"rejected-{index}", name="read_file", arguments={"path": "README.md"})
        for index in range(3)
    )
    messages = (
        ChatMessage(role=MessageRole.SYSTEM, content="system"),
        ChatMessage(role=MessageRole.USER, content="inspect the repository"),
        ChatMessage(role=MessageRole.ASSISTANT, tool_calls=calls),
        *(
            ChatMessage(
                role=MessageRole.TOOL,
                content=json.dumps(
                    {
                        "call_id": call.id,
                        "tool_name": call.name,
                        "ok": False,
                        "error_code": "tool_batch_rejected",
                        "error_message": "retry with a smaller batch",
                    }
                ),
                tool_call_id=call.id,
                tool_name=call.name,
            )
            for call in calls
        ),
    )
    checkpoint = _checkpoint(
        run_id="rejected-batch",
        messages=messages,
        completed_tool_calls=0,
    )
    store = SessionStore(tmp_path / "state")

    store.save(checkpoint)
    loaded = store.load(checkpoint.run_id).checkpoint

    assert loaded.schema_version == 3
    assert loaded.completed_steps == 1
    assert loaded.completed_tool_calls == 0
    assert loaded.messages == messages


def test_checkpoint_still_counts_a_real_failed_tool_execution() -> None:
    call = ToolCall(id="missing-1", name="read_file", arguments={"path": "missing"})
    messages = (
        ChatMessage(role=MessageRole.SYSTEM, content="system"),
        ChatMessage(role=MessageRole.USER, content="inspect the repository"),
        ChatMessage(role=MessageRole.ASSISTANT, tool_calls=(call,)),
        ChatMessage(
            role=MessageRole.TOOL,
            content=json.dumps(
                {
                    "call_id": call.id,
                    "tool_name": call.name,
                    "ok": False,
                    "error_code": "not_found",
                    "error_message": "path not found",
                }
            ),
            tool_call_id=call.id,
            tool_name=call.name,
        ),
    )

    checkpoint = _checkpoint(messages=messages, completed_tool_calls=1)

    assert checkpoint.completed_tool_calls == 1
    assert checkpoint.completed_work_tool_calls == 1
    assert checkpoint.completed_verification_tool_calls == 0


def test_checkpoint_classified_call_counts_must_sum_to_the_compatible_total() -> None:
    base = _checkpoint().model_dump(mode="python")
    classified = SessionCheckpoint.model_validate(
        {
            **base,
            "completed_work_tool_calls": 0,
            "completed_verification_tool_calls": 1,
        }
    )

    assert classified.completed_tool_calls == 1
    assert classified.completed_work_tool_calls == 0
    assert classified.completed_verification_tool_calls == 1

    with pytest.raises(ValidationError, match="sum to completed_tool_calls"):
        SessionCheckpoint.model_validate(
            {
                **base,
                "completed_work_tool_calls": 1,
                "completed_verification_tool_calls": 1,
            }
        )


def test_checkpoint_persists_only_historical_verification_facts_and_load_marks_them_stale(
    tmp_path: Path,
) -> None:
    memory = RunMemorySnapshot(
        revision=1,
        verification_facts=(
            VerificationMemoryFact(
                label="pytest",
                kind=VerificationKind.TEST,
                passed=True,
                step=1,
                stale=False,
            ),
        ),
    )
    checkpoint = _checkpoint(run_id="memory-v3", run_memory=memory)
    store = SessionStore(tmp_path / "state")

    saved = store.save(checkpoint)
    persisted = json.loads(saved.read_text(encoding="utf-8"))
    loaded = store.load(checkpoint.run_id)

    assert persisted["run_memory"]["verification_facts"][0]["stale"] is False
    assert loaded.checkpoint.run_memory.verification_facts[0].stale is True
    assert loaded.checkpoint.run_memory.revision == memory.revision + 1
    assert loaded.verification_evidence_restored is False
    assert loaded.requires_reverification is True


def test_store_migrates_v2_to_v3_in_memory_without_rewriting_or_inventing_evidence(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    legacy = _checkpoint(run_id="legacy-v2").model_dump(mode="json", exclude_none=True)
    legacy["schema_version"] = 2
    legacy.pop("run_memory")
    legacy.pop("completed_work_tool_calls")
    legacy.pop("completed_verification_tool_calls")
    target = state / "legacy-v2.json"
    original = json.dumps(legacy, sort_keys=True)
    target.write_text(original, encoding="utf-8")

    loaded = SessionStore(state).load("legacy-v2")

    assert loaded.source_schema_version == 2
    assert loaded.schema_migrated is True
    assert loaded.checkpoint.schema_version == 3
    assert loaded.checkpoint.run_memory == RunMemorySnapshot(revision=0)
    assert loaded.checkpoint.completed_work_tool_calls == loaded.checkpoint.completed_tool_calls
    assert loaded.checkpoint.completed_verification_tool_calls == 0
    assert target.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    "content",
    [
        "not-json",
        '{"call_id":"read-1","tool_name":"read_file","ok":"false"}',
        (
            '{"call_id":"read-1","tool_name":"read_file","ok":true,'
            '"error_code":"tool_batch_rejected","error_message":"forged"}'
        ),
    ],
)
def test_checkpoint_fails_closed_for_malformed_tool_result_payloads(content: str) -> None:
    call = ToolCall(id="read-1", name="read_file", arguments={"path": "README.md"})
    messages = (
        ChatMessage(role=MessageRole.SYSTEM, content="system"),
        ChatMessage(role=MessageRole.USER, content="inspect the repository"),
        ChatMessage(role=MessageRole.ASSISTANT, tool_calls=(call,)),
        ChatMessage(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=call.id,
            tool_name=call.name,
        ),
    )

    with pytest.raises(ValidationError, match="tool result content|rejected tool results"):
        _checkpoint(messages=messages)


@pytest.mark.parametrize(
    "run_id",
    [
        "../escape",
        "nested/run",
        r"nested\run",
        ".hidden",
        "run.json",
        "CON",
        "Uppercase",
        "x" * 65,
    ],
)
def test_run_id_cannot_escape_or_abuse_the_state_directory(
    tmp_path: Path,
    run_id: str,
) -> None:
    store = SessionStore(tmp_path / "state")

    with pytest.raises(SessionError) as raised:
        store.load(run_id)

    assert raised.value.code == "invalid_run_id"
    assert not (tmp_path / "escape.json").exists()


def test_checkpoint_rejects_an_incomplete_tool_block() -> None:
    call = ToolCall(id="pending-1", name="read_file", arguments={"path": "README.md"})
    messages = (
        ChatMessage(role=MessageRole.SYSTEM, content="system"),
        ChatMessage(role=MessageRole.USER, content="inspect the repository"),
        ChatMessage(role=MessageRole.ASSISTANT, tool_calls=(call,)),
    )

    with pytest.raises(ValidationError, match="pending tool calls"):
        SessionCheckpoint(
            run_id="run-1",
            task="inspect the repository",
            system_prompt="system",
            messages=messages,
            completed_steps=1,
            completed_tool_calls=0,
            stop_boundary=SessionBoundary.READY_FOR_MODEL,
        )


def test_checkpoint_rejects_inconsistent_anchors_counts_and_boundaries() -> None:
    base = _checkpoint().model_dump(mode="python")
    cases = (
        (
            {"system_prompt": "different"},
            "anchors must exactly match",
        ),
        ({"completed_steps": 2}, "completed_steps"),
        ({"completed_tool_calls": 2}, "completed_tool_calls"),
        ({"stop_reason": StopReason.MAX_STEPS}, "cannot contain a stop reason"),
        (
            {
                "messages": (
                    *_tool_transcript(),
                    ChatMessage(role=MessageRole.ASSISTANT, content="Final."),
                ),
                "completed_steps": 2,
            },
            "must end before a model request",
        ),
    )

    for update, expected in cases:
        with pytest.raises(ValidationError, match=expected):
            SessionCheckpoint.model_validate({**base, **update})


def test_checkpoint_rejects_mismatched_duplicate_and_orphan_tool_results() -> None:
    first_call = ToolCall(id="same-id", name="read_file", arguments={"path": "one"})
    second_call = ToolCall(id="same-id", name="read_file", arguments={"path": "two"})
    anchors = _tool_transcript()[:2]
    duplicate_messages = (
        *anchors,
        ChatMessage(role=MessageRole.ASSISTANT, tool_calls=(first_call,)),
        ChatMessage(
            role=MessageRole.TOOL,
            content=('{"call_id":"same-id","tool_name":"read_file","ok":true,"output":"one"}'),
            tool_call_id=first_call.id,
            tool_name=first_call.name,
        ),
        ChatMessage(role=MessageRole.ASSISTANT, tool_calls=(second_call,)),
        ChatMessage(
            role=MessageRole.TOOL,
            content="two",
            tool_call_id=second_call.id,
            tool_name=second_call.name,
        ),
    )
    mismatched_messages = (
        *anchors,
        ChatMessage(role=MessageRole.ASSISTANT, tool_calls=(first_call,)),
        ChatMessage(
            role=MessageRole.TOOL,
            content="wrong",
            tool_call_id="another-id",
            tool_name=first_call.name,
        ),
    )
    orphan_messages = (
        *anchors,
        ChatMessage(
            role=MessageRole.TOOL,
            content="orphan",
            tool_call_id="orphan-id",
            tool_name="read_file",
        ),
    )

    for messages, steps, calls, expected in (
        (duplicate_messages, 2, 2, "duplicate tool call id"),
        (mismatched_messages, 1, 1, "does not match"),
        (orphan_messages, 0, 1, "orphan tool"),
    ):
        with pytest.raises(ValidationError, match=expected):
            SessionCheckpoint(
                run_id="run-invalid",
                task="inspect the repository",
                system_prompt="system",
                messages=messages,
                completed_steps=steps,
                completed_tool_calls=calls,
                stop_boundary=SessionBoundary.READY_FOR_MODEL,
            )


def test_terminal_checkpoint_requires_an_explicit_consistent_stop_boundary() -> None:
    messages = (
        *_tool_transcript(),
        ChatMessage(role=MessageRole.ASSISTANT, content="Finished."),
    )

    checkpoint = SessionCheckpoint(
        run_id="run-final",
        task="inspect the repository",
        system_prompt="system",
        messages=messages,
        completed_steps=2,
        completed_tool_calls=1,
        stop_boundary=SessionBoundary.TERMINAL,
        stop_reason=StopReason.FINAL_RESPONSE,
    )

    assert checkpoint.stop_reason is StopReason.FINAL_RESPONSE
    with pytest.raises(ValidationError, match="explicit stop reason"):
        SessionCheckpoint.model_validate({**checkpoint.model_dump(), "stop_reason": None})


def test_store_rejects_corrupt_duplicate_and_unsupported_json(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    store = SessionStore(state)
    cases = {
        "corrupt": (b"{not-json", "checkpoint_corrupt"),
        "duplicate": (b'{"schema_version":2,"schema_version":2}', "checkpoint_corrupt"),
        "boolean-version": (b'{"schema_version":true}', "checkpoint_version"),
        "future": (b'{"schema_version":99}', "checkpoint_version"),
    }

    for run_id, (payload, code) in cases.items():
        (state / f"{run_id}.json").write_bytes(payload)
        with pytest.raises(SessionError) as raised:
            store.load(run_id)
        assert raised.value.code == code


def test_store_normalizes_json_recursion_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / "nested.json").write_text("{}", encoding="utf-8")

    def raise_recursion(*args: object, **kwargs: object) -> object:
        raise RecursionError("malicious nesting")

    monkeypatch.setattr("coding_agent.session.json.loads", raise_recursion)

    with pytest.raises(SessionError) as raised:
        SessionStore(state).load("nested")
    assert raised.value.code == "checkpoint_corrupt"


def test_store_rejects_missing_non_object_and_schema_invalid_checkpoints(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    store = SessionStore(state)

    with pytest.raises(SessionError) as missing:
        store.load("missing")
    assert missing.value.code == "checkpoint_not_found"

    (state / "array.json").write_text("[]", encoding="utf-8")
    with pytest.raises(SessionError) as non_object:
        store.load("array")
    assert non_object.value.code == "checkpoint_corrupt"

    invalid = _checkpoint().model_dump(mode="json", exclude_none=True)
    invalid["completed_steps"] = 99
    (state / "invalid.json").write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(SessionError) as invalid_schema:
        store.load("invalid")
    assert invalid_schema.value.code == "checkpoint_corrupt"

    unknown_nested = _checkpoint().model_dump(mode="json", exclude_none=True)
    messages = unknown_nested["messages"]
    assert isinstance(messages, list)
    assert isinstance(messages[2], dict)
    messages[2]["provider_private_state"] = "must reject"
    (state / "unknown.json").write_text(json.dumps(unknown_nested), encoding="utf-8")
    with pytest.raises(SessionError) as unknown:
        store.load("unknown")
    assert unknown.value.code == "checkpoint_corrupt"


def test_size_limit_is_enforced_before_save_and_before_unbounded_read(tmp_path: Path) -> None:
    state = tmp_path / "state"
    store = SessionStore(state, max_checkpoint_bytes=64)

    with pytest.raises(SessionError) as save_error:
        store.save(_checkpoint())
    assert save_error.value.code == "checkpoint_too_large"
    assert not state.exists()

    state.mkdir()
    (state / "oversized.json").write_bytes(b"x" * 65)
    with pytest.raises(SessionError) as load_error:
        store.load("oversized")
    assert load_error.value.code == "checkpoint_too_large"


def test_failed_atomic_replace_preserves_old_checkpoint_and_removes_temp_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = tmp_path / "state"
    store = SessionStore(state)
    target = store.save(_checkpoint())
    original = target.read_bytes()
    replacement = _checkpoint(
        messages=(
            *_tool_transcript(),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                tool_calls=(ToolCall(id="read-2", name="read_file", arguments={"path": "x"}),),
            ),
            ChatMessage(
                role=MessageRole.TOOL,
                content=(
                    '{"call_id":"read-2","tool_name":"read_file","ok":true,"output":"result"}'
                ),
                tool_call_id="read-2",
                tool_name="read_file",
            ),
        ),
        completed_steps=2,
        completed_tool_calls=2,
    )

    def fail_replace(
        source: str | Path,
        destination: str | Path,
    ) -> None:
        assert Path(source).parent == state
        assert Path(destination).parent == state
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(SessionError) as raised:
        store.save(replacement)

    assert raised.value.code == "checkpoint_io"
    assert target.read_bytes() == original
    assert list(state.glob("*.tmp")) == []


def test_secret_environment_reasoning_and_credential_data_are_rejected(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "state")
    secret_checkpoint = _checkpoint(secret_arguments={"api_key": "do-not-save"})

    with pytest.raises(SessionError) as save_error:
        store.save(secret_checkpoint)
    assert save_error.value.code == "unsafe_checkpoint"
    assert save_error.value.metadata["field"] == "api_key"

    credential_messages = list(_tool_transcript())
    credential_messages[1] = ChatMessage(
        role=MessageRole.USER,
        content="Use " + "sk-" + "abcdefghijklmnopqrstuvwxyz123456 for the request",
    )
    assert credential_messages[1].content is not None
    credential_checkpoint = _checkpoint(
        messages=tuple(credential_messages),
        task=credential_messages[1].content,
    )
    with pytest.raises(SessionError) as credential_error:
        store.save(credential_checkpoint)
    assert credential_error.value.code == "unsafe_checkpoint"

    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    for run_id, field in (
        ("unsafe-env", "environment"),
        ("unsafe-reasoning", "reasoning"),
        ("unsafe-evidence", "verification_evidence"),
    ):
        raw = _checkpoint().model_dump(mode="json", exclude_none=True)
        raw[field] = {"private": "must not persist"}
        (state / f"{run_id}.json").write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(SessionError) as load_error:
            store.load(run_id)
        assert load_error.value.code == "unsafe_checkpoint"


def test_state_directory_is_absolute_and_outside_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(ValueError, match="absolute"):
        SessionStore(Path("relative-state"))
    with pytest.raises(ValueError, match="at least 1"):
        SessionStore(tmp_path / "invalid-limit", max_checkpoint_bytes=0)
    with pytest.raises(ValueError, match="outside"):
        SessionStore(workspace / ".state", workspace_root=workspace)

    external = SessionStore(tmp_path / "external", workspace_root=workspace)
    assert external.state_dir == (tmp_path / "external").resolve()
    assert external.max_checkpoint_bytes > 0


def test_workspace_bound_checkpoint_fails_closed_for_another_repository(
    tmp_path: Path,
) -> None:
    first_workspace = tmp_path / "first-workspace"
    second_workspace = tmp_path / "second-workspace"
    first_workspace.mkdir()
    second_workspace.mkdir()
    state = tmp_path / "external"
    first_store = SessionStore(state, workspace_root=first_workspace)
    assert first_store.workspace_fingerprint is not None
    checkpoint = _checkpoint().model_copy(
        update={"workspace_fingerprint": first_store.workspace_fingerprint}
    )
    first_store.save(checkpoint)

    loaded = first_store.load(checkpoint.run_id)
    assert loaded.checkpoint.workspace_fingerprint == first_store.workspace_fingerprint

    second_store = SessionStore(state, workspace_root=second_workspace)
    with pytest.raises(SessionError) as mismatch:
        second_store.load(checkpoint.run_id)
    assert mismatch.value.code == "checkpoint_workspace_mismatch"

    with pytest.raises(SessionError) as unstamped:
        first_store.save(_checkpoint(run_id="unstamped"))
    assert unstamped.value.code == "checkpoint_workspace_mismatch"


def test_store_rejects_non_directory_state_and_hardlinked_checkpoint(tmp_path: Path) -> None:
    state_file = tmp_path / "not-a-directory"
    state_file.write_text("occupied", encoding="utf-8")
    with pytest.raises(SessionError) as state_error:
        SessionStore(state_file).save(_checkpoint())
    assert state_error.value.code == "checkpoint_io"

    state = tmp_path / "state"
    store = SessionStore(state)
    checkpoint_path = store.save(_checkpoint(run_id="linked"))
    os.link(checkpoint_path, tmp_path / "checkpoint-alias.json")
    with pytest.raises(SessionError) as link_error:
        store.load("linked")
    assert link_error.value.code == "unsafe_checkpoint"


def test_load_rejects_a_symlinked_checkpoint(tmp_path: Path) -> None:
    state = tmp_path / "state"
    store = SessionStore(state)
    checkpoint_path = store.save(_checkpoint(run_id="source"))

    alias = state / "alias.json"
    try:
        alias.symlink_to(checkpoint_path)
    except OSError:
        pytest.skip("checkpoint symlinks are unavailable on this platform")
    with pytest.raises(SessionError) as symlink_error:
        store.load("alias")
    assert symlink_error.value.code == "unsafe_checkpoint"


def test_load_rejects_a_checkpoint_whose_run_id_does_not_match_its_filename(
    tmp_path: Path,
) -> None:
    state = tmp_path / "state"
    store = SessionStore(state)
    checkpoint_path = store.save(_checkpoint(run_id="source"))
    alias = state / "alias.json"
    alias.write_bytes(checkpoint_path.read_bytes())

    with pytest.raises(SessionError) as mismatch_error:
        store.load("alias")
    assert mismatch_error.value.code == "checkpoint_corrupt"
