"""OpenAI Responses API adapter for the provider-neutral model boundary."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from copy import deepcopy
from enum import StrEnum
from ipaddress import ip_address
from typing import Any, NoReturn, Protocol, cast
from urllib.parse import urlsplit

from coding_agent.errors import CodedError
from coding_agent.models import (
    ChatMessage,
    MessageRole,
    ModelResponse,
    ToolCall,
    ToolSpec,
)

ResponseInputItem = dict[str, object]
FunctionTool = dict[str, object]


class ResponsesResourceProtocol(Protocol):
    """Small injectable surface used instead of coupling the core to an SDK."""

    @property
    def create(self) -> Callable[..., object]:
        """Create one non-streaming Responses API response."""


class OpenAIClientProtocol(Protocol):
    """Structural client boundary implemented by an OpenAI SDK client or a fake."""

    @property
    def responses(self) -> ResponsesResourceProtocol:
        """Expose the Responses API resource."""


class OpenAIModelError(CodedError):
    """A sanitized, stable failure at the OpenAI provider boundary."""


class ReasoningEffort(StrEnum):
    """Explicit Responses API reasoning effort requested from a provider."""

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class OpenAIResponsesModel:
    """Translate project-owned messages and tools to and from Responses API values.

    The adapter deliberately performs no tool execution. It only transports custom
    function definitions and turns returned function calls into provider-neutral
    ``ToolCall`` values for ``AgentRunner`` to dispatch.
    """

    def __init__(
        self,
        client: OpenAIClientProtocol,
        *,
        model: str,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        normalized_model = model.strip()
        if not normalized_model:
            raise ValueError("model must be a non-empty string")
        self._client = client
        self._model = normalized_model
        self._reasoning_effort = reasoning_effort

    def complete(
        self,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
    ) -> ModelResponse:
        """Request and validate one provider-neutral model turn."""

        response_input = _encode_messages(messages)
        function_tools = _encode_tools(tools)
        request: dict[str, object] = {
            "model": self._model,
            "input": response_input,
            "tools": function_tools,
            "store": False,
        }
        if self._reasoning_effort is not None:
            request["reasoning"] = {"effort": self._reasoning_effort.value}
        try:
            response = self._client.responses.create(**request)
        except Exception:
            # Provider exceptions can contain request headers, URLs, or credentials.
            # Keep the public error deliberately generic and suppress exception chaining.
            raise OpenAIModelError(
                "openai_request_failed",
                "OpenAI model request failed",
            ) from None
        return _decode_response(response)


def create_openai_responses_model(
    *,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_seconds: float = 120.0,
    max_retries: int = 0,
    reasoning_effort: ReasoningEffort | None = None,
    client_factory: Callable[..., object] | None = None,
) -> OpenAIResponsesModel:
    """Create the official SDK transport while keeping it outside the Agent core."""
    if not 1 <= timeout_seconds <= 600:
        raise ValueError("timeout_seconds must be between 1 and 600")
    if not 0 <= max_retries <= 10:
        raise ValueError("max_retries must be between 0 and 10")
    if base_url is None:
        # The OpenAI SDK also reads this variable internally. Resolve it here so
        # every endpoint, including ambient configuration, crosses our boundary.
        base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url is not None:
        base_url = _validated_base_url(base_url)
    if client_factory is None:
        from openai import OpenAI

        client_factory = OpenAI
    try:
        client = client_factory(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
    except Exception:
        raise OpenAIModelError(
            "openai_client_configuration",
            "OpenAI client could not be configured",
        ) from None
    return OpenAIResponsesModel(
        cast(OpenAIClientProtocol, client),
        model=model,
        reasoning_effort=reasoning_effort,
    )


def _encode_messages(messages: Sequence[ChatMessage]) -> list[ResponseInputItem]:
    _validate_history(messages)
    encoded: list[ResponseInputItem] = []
    for message in messages:
        if message.role is MessageRole.TOOL:
            encoded.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": message.content,
                }
            )
            continue

        if message.content is not None:
            encoded.append(
                {
                    "role": message.role.value,
                    "content": message.content,
                }
            )
        if message.role is MessageRole.ASSISTANT:
            encoded.extend(_encode_tool_call(call) for call in message.tool_calls)
    return encoded


def _validate_history(messages: Sequence[ChatMessage]) -> None:
    pending: dict[str, str] = {}
    seen_call_ids: set[str] = set()
    for message in messages:
        if message.role is MessageRole.TOOL:
            call_id = message.tool_call_id
            assert call_id is not None
            expected_name = pending.get(call_id)
            if expected_name is None:
                _raise_invalid_request("tool output has no pending function call")
            if message.tool_name != expected_name:
                _raise_invalid_request("tool output name does not match its function call")
            del pending[call_id]
            continue
        if pending:
            _raise_invalid_request("function calls are missing tool outputs")
        if message.role is not MessageRole.ASSISTANT:
            continue
        for call in message.tool_calls:
            if call.id in seen_call_ids:
                _raise_invalid_request("function call IDs are not globally unique")
            seen_call_ids.add(call.id)
            pending[call.id] = call.name
    if pending:
        _raise_invalid_request("function calls are missing tool outputs")


def _encode_tool_call(call: ToolCall) -> ResponseInputItem:
    try:
        arguments = json.dumps(
            call.arguments,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        raise OpenAIModelError(
            "openai_invalid_request",
            "OpenAI request contains non-JSON tool arguments",
        ) from None
    return {
        "type": "function_call",
        "call_id": call.id,
        "name": call.name,
        "arguments": arguments,
    }


def _encode_tools(tools: Sequence[ToolSpec]) -> list[FunctionTool]:
    return [
        {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": deepcopy(tool.input_schema),
            # Pydantic remains the authoritative argument validator. Several local
            # tool schemas intentionally contain optional fields with defaults, so
            # advertising OpenAI strict mode would make those schemas invalid.
            "strict": False,
        }
        for tool in tools
    ]


def _decode_response(response: object) -> ModelResponse:
    if getattr(response, "error", None) is not None:
        # Provider error objects can contain request details. Keep the public
        # exception fixed instead of interpolating or chaining those values.
        raise OpenAIModelError(
            "openai_provider_error",
            "OpenAI response reported a provider error",
        )

    status = getattr(response, "status", None)
    if not isinstance(status, str):
        _raise_invalid_response("status is missing or not a string")
    if status != "completed":
        raise OpenAIModelError(
            "openai_incomplete_response",
            f"OpenAI response did not complete (status: {_safe_status(status)})",
        )

    output_text = getattr(response, "output_text", None)
    if output_text is not None and not isinstance(output_text, str):
        _raise_invalid_response("output_text is not a string")
    content = output_text if isinstance(output_text, str) and output_text.strip() else None

    output = getattr(response, "output", None)
    if not isinstance(output, Sequence) or isinstance(output, (str, bytes, bytearray)):
        _raise_invalid_response("output is not an item sequence")

    calls: list[ToolCall] = []
    seen_call_ids: set[str] = set()
    contains_reasoning = False
    for item in output:
        item_type = getattr(item, "type", None)
        if not isinstance(item_type, str):
            _raise_invalid_response("an output item has no valid type")
        if item_type == "reasoning":
            contains_reasoning = True
            continue
        if item_type != "function_call":
            continue

        item_status = getattr(item, "status", None)
        if not isinstance(item_status, str):
            _raise_invalid_response("function call status is missing or not a string")
        if item_status != "completed":
            raise OpenAIModelError(
                "openai_incomplete_response",
                f"OpenAI function call did not complete (status: {_safe_status(item_status)})",
            )

        call_id = _required_item_string(item, "call_id")
        name = _required_item_string(item, "name")
        raw_arguments = _required_item_string(item, "arguments", allow_blank=True)
        if call_id in seen_call_ids:
            _raise_invalid_response("function call IDs are not unique")
        seen_call_ids.add(call_id)
        arguments = _decode_arguments(raw_arguments)
        calls.append(ToolCall(id=call_id, name=name, arguments=arguments))

    if contains_reasoning and calls:
        raise OpenAIModelError(
            "openai_reasoning_continuation_unsupported",
            "OpenAI returned reasoning state with tool calls; this adapter cannot "
            "safely continue that model without provider-managed context",
        )

    if content is None and not calls:
        _raise_invalid_response("no text or function calls were returned")
    return ModelResponse(content=content, tool_calls=tuple(calls))


def _required_item_string(item: object, field: str, *, allow_blank: bool = False) -> str:
    value = getattr(item, field, None)
    if not isinstance(value, str) or (not allow_blank and not value.strip()):
        _raise_invalid_response(f"function call {field} is not a valid string")
    return value


def _decode_arguments(raw_arguments: str) -> dict[str, Any]:
    try:
        parsed: object = json.loads(
            raw_arguments,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (json.JSONDecodeError, ValueError):
        _raise_invalid_response("function call arguments are not valid JSON")
    if not isinstance(parsed, dict):
        _raise_invalid_response("function call arguments are not a JSON object")
    if not all(isinstance(key, str) for key in parsed):
        _raise_invalid_response("function call argument keys are not strings")
    return cast(dict[str, Any], parsed)


def _reject_nonstandard_json_constant(_: str) -> NoReturn:
    raise ValueError("non-standard JSON constant")


def _raise_invalid_response(reason: str) -> NoReturn:
    raise OpenAIModelError(
        "openai_invalid_response",
        f"OpenAI returned an invalid response: {reason}",
    )


def _raise_invalid_request(reason: str) -> NoReturn:
    raise OpenAIModelError(
        "openai_invalid_request",
        f"OpenAI request history is invalid: {reason}",
    )


def _validated_base_url(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("base_url must not be blank")
    parsed = urlsplit(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("base_url must be a plain HTTP(S) endpoint without credentials or query")
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise ValueError("remote base_url endpoints must use HTTPS")
    return normalized.rstrip("/")


def _is_loopback_host(host: str) -> bool:
    if host.casefold() == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def _safe_status(status: str) -> str:
    allowed = {"cancelled", "failed", "incomplete", "in_progress", "queued"}
    return status if status in allowed else "unknown"
