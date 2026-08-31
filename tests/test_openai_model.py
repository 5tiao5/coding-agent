"""Offline contract tests for the OpenAI Responses API adapter."""

from __future__ import annotations

from dataclasses import dataclass

import httpx
import pytest
from openai import APIStatusError, APITimeoutError

from coding_agent.errors import CodedError
from coding_agent.model import RecoverableModelResponseError, RetryableModelError
from coding_agent.models import (
    ChatMessage,
    MessageRole,
    ToolCall,
    ToolSpec,
)
from coding_agent.openai_model import (
    OpenAIModelError,
    OpenAIResponsesModel,
    ReasoningEffort,
    create_openai_responses_model,
)


@dataclass(frozen=True, slots=True)
class FakeOutputItem:
    type: str
    call_id: str = ""
    name: str = ""
    arguments: str = ""
    status: str | None = "completed"


@dataclass(frozen=True, slots=True)
class FakeResponse:
    output_text: str | None
    output: tuple[object, ...]
    status: str | None = "completed"
    error: object | None = None


class FakeResponsesResource:
    def __init__(self, result: object) -> None:
        self.result = result
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeClient:
    def __init__(self, result: object) -> None:
        self._responses = FakeResponsesResource(result)

    @property
    def responses(self) -> FakeResponsesResource:
        return self._responses


def _adapter(
    response: object,
    *,
    model: str = "gpt-test",
) -> tuple[OpenAIResponsesModel, FakeClient]:
    client = FakeClient(response)
    return OpenAIResponsesModel(client, model=model), client


def test_maps_neutral_history_and_function_tools_to_responses_input() -> None:
    adapter, client = _adapter(FakeResponse(output_text="Repair complete.", output=()))
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }
    messages = (
        ChatMessage(role=MessageRole.SYSTEM, content="Work inside the repository."),
        ChatMessage(role=MessageRole.USER, content="Inspect the README."),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="I will inspect it.",
            tool_calls=(
                ToolCall(
                    id="call-read",
                    name="read_file",
                    arguments={"path": "文档/README.md", "line_end": 20},
                ),
            ),
        ),
        ChatMessage(
            role=MessageRole.TOOL,
            content='{"ok":true,"output":"hello"}',
            tool_call_id="call-read",
            tool_name="read_file",
        ),
    )
    tools = (
        ToolSpec(
            name="read_file",
            description="Read a UTF-8 text file.",
            input_schema=schema,
        ),
    )

    response = adapter.complete(messages, tools)

    assert response.content == "Repair complete."
    assert response.tool_calls == ()
    assert client.responses.requests == [
        {
            "model": "gpt-test",
            "input": [
                {"role": "system", "content": "Work inside the repository."},
                {"role": "user", "content": "Inspect the README."},
                {"role": "assistant", "content": "I will inspect it."},
                {
                    "type": "function_call",
                    "call_id": "call-read",
                    "name": "read_file",
                    "arguments": '{"line_end":20,"path":"文档/README.md"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-read",
                    "output": '{"ok":true,"output":"hello"}',
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "description": "Read a UTF-8 text file.",
                    "parameters": schema,
                    "strict": False,
                }
            ],
            "store": False,
        }
    ]


def test_maps_text_and_parallel_function_calls_back_to_neutral_response() -> None:
    adapter, _ = _adapter(
        FakeResponse(
            output_text="I need two observations.",
            output=(
                FakeOutputItem(
                    type="function_call",
                    call_id="call-1",
                    name="read_file",
                    arguments='{"path":"README.md"}',
                ),
                FakeOutputItem(type="message"),
                FakeOutputItem(
                    type="function_call",
                    call_id="call-2",
                    name="search_text",
                    arguments='{"query":"TODO","paths":["src"]}',
                ),
            ),
        )
    )

    response = adapter.complete(
        (ChatMessage(role=MessageRole.USER, content="Investigate the project."),),
        (),
    )

    assert response.content == "I need two observations."
    assert response.tool_calls == (
        ToolCall(id="call-1", name="read_file", arguments={"path": "README.md"}),
        ToolCall(
            id="call-2",
            name="search_text",
            arguments={"query": "TODO", "paths": ["src"]},
        ),
    )


def test_blank_text_is_omitted_when_function_calls_are_present() -> None:
    adapter, _ = _adapter(
        FakeResponse(
            output_text="  \n",
            output=(
                FakeOutputItem(
                    type="function_call",
                    call_id="call-1",
                    name="list_files",
                    arguments="{}",
                ),
            ),
        )
    )

    response = adapter.complete(
        (ChatMessage(role=MessageRole.USER, content="List files."),),
        (),
    )

    assert response.content is None
    assert response.tool_calls == (ToolCall(id="call-1", name="list_files"),)


def test_rejects_reasoning_state_that_cannot_be_replayed_with_tool_calls() -> None:
    adapter, _ = _adapter(
        FakeResponse(
            output_text="I need an observation.",
            output=(
                FakeOutputItem(type="reasoning"),
                FakeOutputItem(
                    type="function_call",
                    call_id="call-1",
                    name="read_file",
                    arguments='{"path":"README.md"}',
                ),
            ),
        )
    )

    with pytest.raises(OpenAIModelError) as error:
        adapter.complete(
            (ChatMessage(role=MessageRole.USER, content="Investigate."),),
            (),
        )

    assert error.value.code == "openai_reasoning_continuation_unsupported"


@pytest.mark.parametrize(
    ("items", "message", "error_type", "error_code"),
    [
        (
            (
                FakeOutputItem(
                    type="function_call",
                    call_id="call-1",
                    name="read_file",
                    arguments="{not-json",
                ),
            ),
            "arguments are not valid JSON",
            RecoverableModelResponseError,
            "openai_invalid_function_arguments",
        ),
        (
            (
                FakeOutputItem(
                    type="function_call",
                    call_id="call-1",
                    name="read_file",
                    arguments='["README.md"]',
                ),
            ),
            "arguments are not a JSON object",
            RecoverableModelResponseError,
            "openai_invalid_function_arguments",
        ),
        (
            (
                FakeOutputItem(
                    type="function_call",
                    call_id="call-1",
                    name="read_file",
                    arguments="{}",
                ),
                FakeOutputItem(
                    type="function_call",
                    call_id="call-1",
                    name="search_text",
                    arguments="{}",
                ),
            ),
            "IDs are not unique",
            OpenAIModelError,
            "openai_invalid_response",
        ),
        (
            (
                FakeOutputItem(
                    type="function_call",
                    call_id="call-1",
                    name="read_file",
                    arguments='{"value":NaN}',
                ),
            ),
            "arguments are not valid JSON",
            RecoverableModelResponseError,
            "openai_invalid_function_arguments",
        ),
    ],
    ids=["invalid-json", "non-object-json", "duplicate-call-id", "non-standard-json"],
)
def test_rejects_untrusted_function_call_shapes(
    items: tuple[object, ...],
    message: str,
    error_type: type[CodedError],
    error_code: str,
) -> None:
    adapter, _ = _adapter(FakeResponse(output_text=None, output=items))

    with pytest.raises(error_type, match=message) as error:
        adapter.complete(
            (ChatMessage(role=MessageRole.USER, content="Use a tool."),),
            (),
        )

    assert error.value.code == error_code


def test_malformed_function_arguments_never_escape_in_recoverable_error() -> None:
    secret = "TEST_PRIVATE_MALFORMED_ARGUMENT_SENTINEL"
    adapter, _ = _adapter(
        FakeResponse(
            output_text=None,
            output=(
                FakeOutputItem(
                    type="function_call",
                    call_id="call-1",
                    name="read_file",
                    arguments=f'{{"path":"{secret}"',
                ),
            ),
        )
    )

    with pytest.raises(RecoverableModelResponseError) as error:
        adapter.complete(
            (ChatMessage(role=MessageRole.USER, content="Use a tool."),),
            (),
        )

    assert error.value.code == "openai_invalid_function_arguments"
    assert str(error.value) == (
        "OpenAI returned invalid function-call arguments: arguments are not valid JSON"
    )
    assert secret not in str(error.value)
    assert error.value.__cause__ is None


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(output_text=None, output=()),
        FakeResponse(output_text=" \n", output=(FakeOutputItem(type="reasoning"),)),
    ],
    ids=["empty-output", "non-user-visible-output"],
)
def test_rejects_empty_provider_responses(response: FakeResponse) -> None:
    adapter, _ = _adapter(response)

    with pytest.raises(OpenAIModelError, match="no text or function calls") as error:
        adapter.complete(
            (ChatMessage(role=MessageRole.USER, content="Respond."),),
            (),
        )

    assert error.value.code == "openai_invalid_response"


def test_sanitizes_sdk_exceptions_without_chaining_provider_details() -> None:
    secret = "sk-" + "super-secret-token"
    adapter, _ = _adapter(RuntimeError(f"Authorization failed for {secret}"))

    with pytest.raises(OpenAIModelError) as error:
        adapter.complete(
            (ChatMessage(role=MessageRole.USER, content="Respond."),),
            (),
        )

    assert error.value.code == "openai_request_failed"
    assert str(error.value) == "OpenAI model request failed"
    assert secret not in str(error.value)
    assert error.value.__cause__ is None


@pytest.mark.parametrize("kind", ["timeout", "rate-limit", "server-error"])
def test_maps_only_transient_sdk_request_failures_to_retryable_errors(kind: str) -> None:
    request = httpx.Request("POST", "https://provider.invalid/v1/responses")
    if kind == "timeout":
        provider_error: Exception = APITimeoutError(request)
    else:
        status_code = 429 if kind == "rate-limit" else 503
        response = httpx.Response(status_code, request=request)
        provider_error = APIStatusError(
            "provider body must not escape",
            response=response,
            body={"secret": "TEST_PROVIDER_SECRET_SENTINEL"},
        )
    adapter, _ = _adapter(provider_error)

    with pytest.raises(RetryableModelError) as error:
        adapter.complete(
            (ChatMessage(role=MessageRole.USER, content="Respond."),),
            (),
        )

    assert error.value.code == "openai_request_transient"
    assert str(error.value) == "OpenAI model request failed transiently"
    assert "provider body" not in str(error.value)
    assert error.value.__cause__ is None


def test_nontransient_sdk_status_failure_remains_nonretryable() -> None:
    request = httpx.Request("POST", "https://provider.invalid/v1/responses")
    response = httpx.Response(400, request=request)
    adapter, _ = _adapter(APIStatusError("invalid request detail", response=response, body=None))

    with pytest.raises(OpenAIModelError) as error:
        adapter.complete(
            (ChatMessage(role=MessageRole.USER, content="Respond."),),
            (),
        )

    assert not isinstance(error.value, RetryableModelError)
    assert error.value.code == "openai_request_failed"
    assert str(error.value) == "OpenAI model request failed"


def test_rejects_non_json_history_arguments_before_contacting_provider() -> None:
    adapter, client = _adapter(FakeResponse(output_text="unused", output=()))
    message = ChatMessage(
        role=MessageRole.ASSISTANT,
        tool_calls=(
            ToolCall(
                id="call-1",
                name="bad_tool",
                arguments={"value": float("nan")},
            ),
        ),
    )

    tool_output = ChatMessage(
        role=MessageRole.TOOL,
        content="unused",
        tool_call_id="call-1",
        tool_name="bad_tool",
    )

    with pytest.raises(OpenAIModelError, match="non-JSON tool arguments") as error:
        adapter.complete((message, tool_output), ())

    assert error.value.code == "openai_invalid_request"
    assert client.responses.requests == []


@pytest.mark.parametrize(
    "messages",
    [
        (
            ChatMessage(role=MessageRole.USER, content="Inspect."),
            ChatMessage(
                role=MessageRole.TOOL,
                content="orphan",
                tool_call_id="call-1",
                tool_name="read_file",
            ),
        ),
        (
            ChatMessage(role=MessageRole.USER, content="Inspect."),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                tool_calls=(ToolCall(id="call-1", name="read_file"),),
            ),
            ChatMessage(
                role=MessageRole.TOOL,
                content="mismatch",
                tool_call_id="call-1",
                tool_name="search_text",
            ),
        ),
        (
            ChatMessage(role=MessageRole.USER, content="Inspect."),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                tool_calls=(ToolCall(id="call-1", name="read_file"),),
            ),
        ),
        (
            ChatMessage(role=MessageRole.USER, content="Inspect."),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                tool_calls=(ToolCall(id="call-1", name="read_file"),),
            ),
            ChatMessage(
                role=MessageRole.TOOL,
                content="first",
                tool_call_id="call-1",
                tool_name="read_file",
            ),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                tool_calls=(ToolCall(id="call-1", name="read_file"),),
            ),
            ChatMessage(
                role=MessageRole.TOOL,
                content="second",
                tool_call_id="call-1",
                tool_name="read_file",
            ),
        ),
    ],
    ids=["orphan-output", "name-mismatch", "missing-output", "duplicate-call-id"],
)
def test_rejects_noncanonical_function_history_before_request(
    messages: tuple[ChatMessage, ...],
) -> None:
    adapter, client = _adapter(FakeResponse(output_text="unused", output=()))

    with pytest.raises(OpenAIModelError) as error:
        adapter.complete(messages, ())

    assert error.value.code == "openai_invalid_request"
    assert client.responses.requests == []


def test_requires_a_non_empty_model_name() -> None:
    client = FakeClient(FakeResponse(output_text="unused", output=()))

    with pytest.raises(ValueError, match="model must be a non-empty string"):
        OpenAIResponsesModel(client, model="  ")


def test_sdk_factory_passes_only_explicit_transport_configuration() -> None:
    created: list[dict[str, object]] = []
    client = FakeClient(FakeResponse(output_text="ready", output=()))

    def factory(**kwargs: object) -> object:
        created.append(kwargs)
        return client

    adapter = create_openai_responses_model(
        model="gpt-test",
        api_key="test-key",
        base_url="https://example.invalid/v1",
        timeout_seconds=30,
        max_retries=1,
        reasoning_effort=ReasoningEffort.NONE,
        client_factory=factory,
    )

    assert isinstance(adapter, OpenAIResponsesModel)
    assert created == [
        {
            "api_key": "test-key",
            "base_url": "https://example.invalid/v1",
            "timeout": 30,
            "max_retries": 1,
        }
    ]
    adapter.complete(
        (
            ChatMessage(role=MessageRole.SYSTEM, content="Work locally."),
            ChatMessage(role=MessageRole.USER, content="Inspect the repository."),
        ),
        (),
    )
    assert client.responses.requests[0]["reasoning"] == {"effort": "none"}


def test_sdk_factory_disables_hidden_transport_retries_by_default() -> None:
    created: list[dict[str, object]] = []

    def factory(**kwargs: object) -> object:
        created.append(kwargs)
        return FakeClient(FakeResponse(output_text="ready", output=()))

    create_openai_responses_model(model="gpt-test", client_factory=factory)

    assert created[0]["max_retries"] == 0


@pytest.mark.parametrize(
    "base_url",
    [
        "http://remote.example/v1",
        "https://user:secret@example.com/v1",
        "https://example.com/v1?token=secret",
        "file:///tmp/provider",
    ],
)
def test_sdk_factory_rejects_unsafe_base_urls(base_url: str) -> None:
    with pytest.raises(ValueError, match="base_url"):
        create_openai_responses_model(
            model="gpt-test",
            base_url=base_url,
            client_factory=lambda **_: object(),
        )


def test_sdk_factory_allows_plain_http_only_for_loopback() -> None:
    created: list[dict[str, object]] = []

    def factory(**kwargs: object) -> object:
        created.append(kwargs)
        return FakeClient(FakeResponse(output_text="ready", output=()))

    create_openai_responses_model(
        model="gpt-test",
        base_url="http://127.0.0.1:8000/v1/",
        client_factory=factory,
    )

    assert created[0]["base_url"] == "http://127.0.0.1:8000/v1"


def test_rejects_incomplete_provider_response_without_exposing_details() -> None:
    adapter, _ = _adapter(FakeResponse(output_text="partial", output=(), status="incomplete"))

    with pytest.raises(OpenAIModelError) as error:
        adapter.complete(
            (ChatMessage(role=MessageRole.USER, content="Respond."),),
            (),
        )

    assert error.value.code == "openai_incomplete_response"
    assert "incomplete" in str(error.value)


def test_rejects_provider_response_with_missing_status() -> None:
    adapter, _ = _adapter(FakeResponse(output_text="complete?", output=(), status=None))

    with pytest.raises(OpenAIModelError, match="status is missing") as error:
        adapter.complete(
            (ChatMessage(role=MessageRole.USER, content="Respond."),),
            (),
        )

    assert error.value.code == "openai_invalid_response"


def test_rejects_provider_error_without_exposing_details() -> None:
    secret = "sk-" + "provider-error-secret"
    adapter, _ = _adapter(
        FakeResponse(
            output_text="unused",
            output=(),
            error={"message": f"Request failed for {secret}"},
        )
    )

    with pytest.raises(OpenAIModelError) as error:
        adapter.complete(
            (ChatMessage(role=MessageRole.USER, content="Respond."),),
            (),
        )

    assert error.value.code == "openai_provider_error"
    assert str(error.value) == "OpenAI response reported a provider error"
    assert secret not in str(error.value)


@pytest.mark.parametrize(
    ("status", "expected_code"),
    [
        (None, "openai_invalid_response"),
        ("incomplete", "openai_incomplete_response"),
        ("in_progress", "openai_incomplete_response"),
    ],
    ids=["missing", "incomplete", "in-progress"],
)
def test_rejects_noncompleted_function_call_items(
    status: str | None,
    expected_code: str,
) -> None:
    adapter, _ = _adapter(
        FakeResponse(
            output_text=None,
            output=(
                FakeOutputItem(
                    type="function_call",
                    call_id="call-1",
                    name="read_file",
                    arguments='{"path":"README.md"}',
                    status=status,
                ),
            ),
        )
    )

    with pytest.raises(OpenAIModelError) as error:
        adapter.complete(
            (ChatMessage(role=MessageRole.USER, content="Read it."),),
            (),
        )

    assert error.value.code == expected_code


def test_sdk_factory_validates_ambient_base_url_before_sdk_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, object]] = []
    monkeypatch.setenv("OPENAI_BASE_URL", "http://remote.example/v1")

    with pytest.raises(ValueError, match="remote base_url endpoints must use HTTPS"):
        create_openai_responses_model(
            model="gpt-test",
            client_factory=lambda **kwargs: created.append(kwargs),
        )

    assert created == []


def test_explicit_base_url_overrides_ambient_sdk_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, object]] = []
    monkeypatch.setenv("OPENAI_BASE_URL", "http://remote.example/v1")

    def factory(**kwargs: object) -> object:
        created.append(kwargs)
        return FakeClient(FakeResponse(output_text="ready", output=()))

    create_openai_responses_model(
        model="gpt-test",
        base_url="https://explicit.example/v1/",
        client_factory=factory,
    )

    assert created[0]["base_url"] == "https://explicit.example/v1"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timeout_seconds": 0}, "timeout_seconds"),
        ({"max_retries": 11}, "max_retries"),
    ],
)
def test_sdk_factory_rejects_unbounded_transport_configuration(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        create_openai_responses_model(model="gpt-test", **kwargs)  # type: ignore[arg-type]


def test_sdk_factory_sanitizes_configuration_failures() -> None:
    def fail(**kwargs: object) -> object:
        del kwargs
        raise RuntimeError("secret setup details")

    with pytest.raises(OpenAIModelError) as error:
        create_openai_responses_model(model="gpt-test", client_factory=fail)

    assert error.value.code == "openai_client_configuration"
    assert error.value.__cause__ is None
