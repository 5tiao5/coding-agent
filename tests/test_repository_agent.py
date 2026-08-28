"""End-to-end tests for repository inspection through the real agent loop."""

from __future__ import annotations

from pathlib import Path

from pydantic import TypeAdapter

from coding_agent.agent import AgentRunner
from coding_agent.events import EventKind, MemoryEventSink
from coding_agent.model import ScriptedModel
from coding_agent.models import AgentState, MessageRole, ModelResponse, ToolCall
from coding_agent.tools import ListFilesTool, ReadFileTool, SearchTextTool, ToolRegistry
from coding_agent.workspace import Workspace

_PAYLOAD_ADAPTER = TypeAdapter(dict[str, object])


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _observation(model: ScriptedModel, request_index: int) -> dict[str, object]:
    message = model.requests[request_index].messages[-1]
    assert message.role is MessageRole.TOOL
    assert message.content is not None
    return _PAYLOAD_ADAPTER.validate_json(message.content)


def test_agent_lists_searches_reads_and_finishes_against_a_real_workspace(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    _write(
        root / "src" / "pricing.py",
        "def calculate_total(subtotal: int, discount: int) -> int:\n"
        "    return subtotal - discount - discount\n",
    )
    _write(root / "tests" / "test_pricing.py", "assert calculate_total(100, 15) == 85\n")
    workspace = Workspace(root)
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="list-1",
                        name="list_files",
                        arguments={"path": ".", "max_depth": 3, "limit": 50},
                    ),
                )
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="search-1",
                        name="search_text",
                        arguments={"query": "def calculate_total", "path": "src"},
                    ),
                )
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="read-1",
                        name="read_file",
                        arguments={"path": "src/pricing.py", "line_count": 20},
                    ),
                )
            ),
            ModelResponse(content="The discount is subtracted twice in src/pricing.py."),
        ]
    )
    events = MemoryEventSink()
    registry = ToolRegistry(
        [
            ListFilesTool(workspace),
            ReadFileTool(workspace),
            SearchTextTool(workspace),
        ]
    )

    result = AgentRunner(model, registry, event_sink=events, max_steps=5).run(
        "Locate the calculate_total defect without editing the repository."
    )

    assert result.state is AgentState.COMPLETED_UNVERIFIED
    assert result.steps == 4
    assert result.final_text == "The discount is subtracted twice in src/pricing.py."
    assert [spec.name for spec in model.requests[0].tools] == [
        "list_files",
        "read_file",
        "search_text",
    ]

    listed = _observation(model, 1)
    searched = _observation(model, 2)
    read = _observation(model, 3)
    assert listed["ok"] is True
    assert "[F] src/pricing.py" in str(listed["output"])
    searched_metadata = searched["metadata"]
    read_metadata = read["metadata"]
    assert isinstance(searched_metadata, dict)
    assert isinstance(read_metadata, dict)
    assert searched_metadata["match_count"] == 1
    assert "src/pricing.py:1:1" in str(searched["output"])
    assert read_metadata["returned_line_count"] == 2
    assert "return subtotal - discount - discount" in str(read["output"])

    finished = [event for event in events.events if event.kind is EventKind.TOOL_FINISHED]
    assert [event.data["tool_name"] for event in finished] == [
        "list_files",
        "search_text",
        "read_file",
    ]
    assert all("subtotal - discount" not in str(event.data) for event in finished)
