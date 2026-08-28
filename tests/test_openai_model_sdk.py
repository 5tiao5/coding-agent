"""Offline wire-format check using the real OpenAI SDK and an HTTP mock transport."""

from __future__ import annotations

import json
from typing import cast

import httpx
import pytest
from openai import OpenAI

from coding_agent.models import ChatMessage, MessageRole, ModelResponse, ToolSpec
from coding_agent.openai_model import (
    OpenAIClientProtocol,
    OpenAIModelError,
    OpenAIResponsesModel,
    ReasoningEffort,
)


def test_real_sdk_serializes_a_stateless_two_turn_function_loop() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        payload = json.loads(request.content)
        assert isinstance(payload, dict)
        requests.append(payload)
        output: list[dict[str, object]]
        if len(requests) == 1:
            output = [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": '{"path":"README.md"}',
                    "status": "completed",
                }
            ]
        else:
            output = [
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "README inspected.",
                            "annotations": [],
                        }
                    ],
                }
            ]
        return httpx.Response(
            200,
            request=request,
            json={
                "id": f"resp_{len(requests)}",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-test",
                "output": output,
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    client = OpenAI(
        api_key="test-key",
        base_url="https://provider.invalid/v1",
        http_client=http_client,
        max_retries=0,
    )
    adapter = OpenAIResponsesModel(
        cast(OpenAIClientProtocol, client),
        model="gpt-test",
        reasoning_effort=ReasoningEffort.NONE,
    )
    tools = (
        ToolSpec(
            name="read_file",
            description="Read one file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "line_count": {"type": "integer", "default": 200},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
    )
    history: list[ChatMessage] = [
        ChatMessage(role=MessageRole.SYSTEM, content="Work locally."),
        ChatMessage(role=MessageRole.USER, content="Inspect README."),
    ]
    try:
        first = adapter.complete(history, tools)
        assert first.tool_calls[0].id == "call_1"
        history.append(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=first.content,
                tool_calls=first.tool_calls,
            )
        )
        history.append(
            ChatMessage(
                role=MessageRole.TOOL,
                content='{"ok":true,"output":"hello"}',
                tool_call_id="call_1",
                tool_name="read_file",
            )
        )
        second = adapter.complete(history, tools)
    finally:
        client.close()

    assert second.content == "README inspected."
    assert requests[0]["store"] is False
    assert requests[1]["store"] is False
    assert requests[0]["reasoning"] == {"effort": "none"}
    assert requests[1]["reasoning"] == {"effort": "none"}
    assert cast(list[dict[str, object]], requests[0]["tools"])[0]["strict"] is False
    second_input = cast(list[dict[str, object]], requests[1]["input"])
    assert second_input[-2] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "read_file",
        "arguments": '{"path":"README.md"}',
    }
    assert second_input[-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"ok":true,"output":"hello"}',
    }


def _complete_sdk_payload(payload: dict[str, object]) -> ModelResponse:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, request=request, json=payload)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OpenAI(
        api_key="test-key",
        base_url="https://provider.invalid/v1",
        http_client=http_client,
        max_retries=0,
    )
    adapter = OpenAIResponsesModel(cast(OpenAIClientProtocol, client), model="gpt-test")
    try:
        return adapter.complete(
            (ChatMessage(role=MessageRole.USER, content="Respond."),),
            (),
        )
    finally:
        client.close()


def _sdk_response_payload(output: list[dict[str, object]]) -> dict[str, object]:
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "model": "gpt-test",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _sdk_text_output() -> list[dict[str, object]]:
    return [
        {
            "type": "message",
            "id": "msg_1",
            "status": "completed",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "Done.",
                    "annotations": [],
                }
            ],
        }
    ]


def test_real_sdk_rejects_response_with_missing_status() -> None:
    payload = _sdk_response_payload(_sdk_text_output())

    with pytest.raises(OpenAIModelError) as error:
        _complete_sdk_payload(payload)

    assert error.value.code == "openai_invalid_response"


def test_real_sdk_rejects_incomplete_response_status() -> None:
    payload = _sdk_response_payload(_sdk_text_output())
    payload["status"] = "incomplete"

    with pytest.raises(OpenAIModelError) as error:
        _complete_sdk_payload(payload)

    assert error.value.code == "openai_incomplete_response"


def test_real_sdk_rejects_provider_error_without_exposing_payload() -> None:
    secret = "sk-" + "wire-error-secret"
    payload = _sdk_response_payload(_sdk_text_output())
    payload["status"] = "completed"
    payload["error"] = {
        "code": "server_error",
        "message": f"Provider failure for {secret}",
    }

    with pytest.raises(OpenAIModelError) as error:
        _complete_sdk_payload(payload)

    assert error.value.code == "openai_provider_error"
    assert secret not in str(error.value)


@pytest.mark.parametrize(
    ("item_status", "expected_code"),
    [
        (None, "openai_invalid_response"),
        ("incomplete", "openai_incomplete_response"),
        ("in_progress", "openai_incomplete_response"),
    ],
    ids=["missing", "incomplete", "in-progress"],
)
def test_real_sdk_rejects_noncompleted_function_call_item_status(
    item_status: str | None,
    expected_code: str,
) -> None:
    function_call: dict[str, object] = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "read_file",
        "arguments": '{"path":"README.md"}',
    }
    if item_status is not None:
        function_call["status"] = item_status
    payload = _sdk_response_payload([function_call])
    payload["status"] = "completed"

    with pytest.raises(OpenAIModelError) as error:
        _complete_sdk_payload(payload)

    assert error.value.code == expected_code
