"""Safe, user-facing explanations for terminal and local-Web run failures."""

from __future__ import annotations

import re

from coding_agent.errors import CodedError
from coding_agent.models import AgentResult, StopReason

_PER_TURN_TOOL_LIMIT_ERROR = re.compile(
    r"Model exceeded the per-step tool call limit "
    r"([1-9][0-9]{0,6}) consecutive times: "
    r"([1-9][0-9]{0,6}) > ([1-9][0-9]{0,6})"
)
_TOTAL_TOOL_LIMIT_ERROR = re.compile(
    r"Model exceeded the total tool call limit: ([1-9][0-9]{0,6}) > ([1-9][0-9]{0,6})"
)

_MODEL_ERROR_MESSAGES = {
    "Model returned invalid tool-call arguments after protocol recovery attempts": (
        "模型返回的工具参数格式无效，自动纠正重试后仍未恢复。"
    ),
    "Model request failed after transient retries": (
        "模型服务暂时不可用，自动重试后仍未恢复。请稍后重试。"
    ),
    "Model request failed: OpenAI model request failed": (
        "模型请求被服务端拒绝。请检查 API Key、模型名称、接口地址和账户权限。"
    ),
    "Model request failed: OpenAI response reported a provider error": (
        "模型服务返回了错误状态。请检查模型配置后重试。"
    ),
}

_MODEL_ERROR_PREFIXES = (
    (
        "Model request failed: OpenAI returned an invalid response:",
        "模型返回了无法识别的响应格式，请重试或更换兼容模型。",
    ),
    (
        "Model request failed: OpenAI response did not complete",
        "模型响应未完整生成，请重试。",
    ),
    (
        "Model request failed: OpenAI function call did not complete",
        "模型的工具调用未完整生成，请重试。",
    ),
    (
        "Model request failed: OpenAI returned reasoning state with tool calls",
        "当前模型的推理状态与无状态工具调用模式不兼容。",
    ),
    (
        "Model request failed: OpenAI request history is invalid:",
        "模型请求上下文不符合工具调用协议，建议从最近检查点重新运行。",
    ),
    (
        "Model request failed: OpenAI request contains non-JSON tool arguments",
        "模型请求中的工具参数无法安全编码。",
    ),
)

_CODED_ERROR_MESSAGES = {
    "openai_client_configuration": ("模型客户端配置失败。请检查 API Key、接口地址和相关环境变量。"),
    "openai_request_failed": (
        "模型请求被服务端拒绝。请检查 API Key、模型名称、接口地址和账户权限。"
    ),
    "openai_request_transient": "模型服务暂时不可用，请稍后重试。",
    "openai_invalid_function_arguments": ("模型返回的工具参数格式无效，自动纠正后仍未恢复。"),
    "openai_invalid_response": "模型返回了无法识别的响应格式，请重试或更换兼容模型。",
    "openai_provider_error": "模型服务返回了错误状态。请检查模型配置后重试。",
    "openai_incomplete_response": "模型响应未完整生成，请重试。",
    "openai_invalid_request": ("模型请求上下文不符合工具调用协议，建议从最近检查点重新运行。"),
    "openai_reasoning_continuation_unsupported": ("当前模型的推理状态与无状态工具调用模式不兼容。"),
}


def public_coded_error(exc: CodedError) -> str:
    """Translate known model-boundary codes without exposing provider details."""

    known = _CODED_ERROR_MESSAGES.get(exc.code)
    if known is not None:
        return known
    message = " ".join(exc.message.split())
    bounded = message if len(message) <= 400 else f"{message[:397]}..."
    return f"运行失败：{bounded} [{exc.code}]"


def public_result_error(result: AgentResult) -> str:
    """Return one bounded explanation for an Agent-owned terminal result."""

    if result.stop_reason is StopReason.TOOL_LIMIT:
        return _public_tool_limit_error(result.error)
    if result.stop_reason is StopReason.MODEL_ERROR:
        return _public_model_error(result.error)
    messages = {
        StopReason.MAX_STEPS: "任务已达到设定的最大步骤数。",
        StopReason.USER_INTERRUPTED: "任务已由用户中断。",
        StopReason.COMMAND_CONTROL_FAILED: "命令进程控制失败。",
        StopReason.CONTEXT_LIMIT: "模型上下文超出了设定预算。",
        StopReason.REPEATED_TOOL_CALL: "Agent 因重复执行相同工具调用而停止。",
    }
    return messages.get(result.stop_reason, "本地 Agent 运行失败。")


def _public_model_error(error: str | None) -> str:
    if error is None:
        return "模型请求失败。"
    exact = _MODEL_ERROR_MESSAGES.get(error)
    if exact is not None:
        return exact
    for prefix, message in _MODEL_ERROR_PREFIXES:
        if error.startswith(prefix):
            return message
    return "模型请求失败。请检查模型配置或稍后重试。"


def _public_tool_limit_error(error: str | None) -> str:
    if error is None:
        return "任务已达到设定的工具调用上限。"
    per_turn = _PER_TURN_TOOL_LIMIT_ERROR.fullmatch(error)
    if per_turn is not None:
        _rejections, requested, limit = (int(value) for value in per_turn.groups())
        if requested > limit:
            return f"模型连续提交过大的工具批次（本轮 {requested}，上限 {limit}）。"
    total = _TOTAL_TOOL_LIMIT_ERROR.fullmatch(error)
    if total is not None:
        requested, limit = (int(value) for value in total.groups())
        if requested > limit:
            return f"模型达到工具调用总上限（累计请求 {requested}，上限 {limit}）。"
    return "任务已达到设定的工具调用上限。"
