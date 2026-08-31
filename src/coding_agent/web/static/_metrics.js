"use strict";

function asObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function nonNegativeInteger(value) {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : 0;
}

function positiveInteger(value) {
  return typeof value === "number" && Number.isInteger(value) && value > 0 ? value : null;
}

export function runtimeMetricView(rawSnapshot) {
  const snapshot = asObject(rawSnapshot);
  const limits = asObject(snapshot.limits);
  const currentTurn = nonNegativeInteger(snapshot.current_step);
  const toolsStarted = nonNegativeInteger(snapshot.tools_started);
  const toolsFinished = Math.min(toolsStarted, nonNegativeInteger(snapshot.tools_finished));
  const activeTools = Array.isArray(snapshot.active_tools) ? snapshot.active_tools.length : 0;
  const maxModelTurns = positiveInteger(limits.max_model_turns);
  const maxCallsPerTurn = positiveInteger(limits.max_calls_per_turn);
  const maxTotalToolCalls = positiveInteger(limits.max_total_tool_calls);
  const limitsRecorded =
    maxModelTurns !== null && maxCallsPerTurn !== null && maxTotalToolCalls !== null;

  return {
    activeTools,
    modelTurnsAriaLabel: maxModelTurns === null
      ? `当前是第 ${currentTurn} 个模型决策轮次，旧记录未保存轮次上限`
      : `当前是第 ${currentTurn} 个模型决策轮次，上限 ${maxModelTurns}`,
    modelTurnsText: maxModelTurns === null
      ? String(currentTurn)
      : `${currentTurn} / ${maxModelTurns}`,
    toolBudgetText: limitsRecorded
      ? `执行中 ${activeTools} · 工具预算 ${toolsStarted} / ${maxTotalToolCalls} · 单轮上限 ${maxCallsPerTurn}`
      : `执行中 ${activeTools} · 旧记录未保存运行上限`,
    toolCallsAriaLabel:
      `累计启动 ${toolsStarted} 次工具调用，已结束 ${toolsFinished} 次，执行中 ${activeTools} 次`,
    toolCallsText: String(toolsStarted),
  };
}
