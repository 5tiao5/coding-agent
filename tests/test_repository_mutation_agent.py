"""End-to-end mutation flow through the project-owned agent loop."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from pydantic import TypeAdapter

from coding_agent.agent import AgentRunner
from coding_agent.events import EventKind, MemoryEventSink
from coding_agent.model import ScriptedModel
from coding_agent.models import AgentState, MessageRole, ModelResponse, ToolCall
from coding_agent.mutation import MutationSession
from coding_agent.tools import ReadFileTool, ReplaceTextTool, ToolRegistry, UndoChangeTool
from coding_agent.workspace import Workspace

_PAYLOAD_ADAPTER = TypeAdapter(dict[str, object])


def _observation(model: ScriptedModel, request_index: int) -> dict[str, object]:
    message = model.requests[request_index].messages[-1]
    assert message.role is MessageRole.TOOL
    assert message.content is not None
    return _PAYLOAD_ADAPTER.validate_json(message.content)


def test_agent_reads_replaces_verifies_and_reports_a_real_change(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    source = root / "src" / "pricing.py"
    source.parent.mkdir(parents=True)
    original = (
        b"def calculate_total(subtotal: int, discount: int) -> int:\n"
        b"    return subtotal - discount - discount\n"
    )
    corrected = (
        b"def calculate_total(subtotal: int, discount: int) -> int:\n"
        b"    return subtotal - discount\n"
    )
    source.write_bytes(original)
    workspace = Workspace(root)
    session = MutationSession(workspace)
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="read-before",
                        name="read_file",
                        arguments={"path": "src/pricing.py", "line_count": 20},
                    ),
                )
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="replace",
                        name="replace_text",
                        arguments={
                            "path": "src/pricing.py",
                            "old_text": "return subtotal - discount - discount",
                            "new_text": "return subtotal - discount",
                            "expected_sha256": sha256(original).hexdigest(),
                        },
                    ),
                )
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="read-after",
                        name="read_file",
                        arguments={"path": "src/pricing.py", "line_count": 20},
                    ),
                )
            ),
            ModelResponse(content="Fixed the duplicate discount and verified the new bytes."),
        ]
    )
    events = MemoryEventSink()
    registry = ToolRegistry(
        [ReadFileTool(workspace), ReplaceTextTool(session), UndoChangeTool(session)]
    )

    result = AgentRunner(model, registry, event_sink=events, max_steps=5).run(
        "Fix the duplicate-discount defect and verify the edited file."
    )

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert source.read_bytes() == corrected
    assert session.revision == 1
    before = _observation(model, 1)
    changed = _observation(model, 2)
    after = _observation(model, 3)
    before_metadata = before["metadata"]
    changed_metadata = changed["metadata"]
    after_metadata = after["metadata"]
    assert isinstance(before_metadata, dict)
    assert isinstance(changed_metadata, dict)
    assert isinstance(after_metadata, dict)
    assert before_metadata["sha256"] == sha256(original).hexdigest()
    assert changed_metadata["change_kind"] == "update"
    assert changed_metadata["after_sha256"] == sha256(corrected).hexdigest()
    assert "-    return subtotal - discount - discount" in str(changed["output"])
    assert "+    return subtotal - discount" in str(changed["output"])
    assert after_metadata["sha256"] == sha256(corrected).hexdigest()

    finished = [event for event in events.events if event.kind is EventKind.TOOL_FINISHED]
    assert [event.data["tool_name"] for event in finished] == [
        "read_file",
        "replace_text",
        "read_file",
    ]
    assert "subtotal - discount" not in str(finished[0].data)
    assert "subtotal - discount" not in str(finished[2].data)
    assert "-    return subtotal - discount - discount" in str(finished[1].data["preview"])
    assert "+    return subtotal - discount" in str(finished[1].data["preview"])
