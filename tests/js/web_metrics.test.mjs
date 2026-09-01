import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");

async function importBrowserModule(relativePath) {
  const source = readFileSync(path.join(repositoryRoot, relativePath), "utf8");
  const encoded = Buffer.from(source).toString("base64");
  return import(`data:text/javascript;base64,${encoded}`);
}

test("runtime metrics distinguish cumulative calls, active work, and real budgets", async () => {
  const { runtimeMetricView } = await importBrowserModule(
    "src/coding_agent/web/static/_metrics.js",
  );

  const metrics = runtimeMetricView({
    active_tools: ["read_file"],
    current_step: 2,
    limits: {
      max_calls_per_turn: 8,
      max_model_turns: 20,
      max_total_tool_calls: 40,
    },
    tools_finished: 1,
    tools_started: 2,
  });

  assert.equal(metrics.modelTurnsText, "2 / 20");
  assert.equal(metrics.toolCallsText, "2");
  assert.equal(metrics.toolBudgetText, "执行中 1 · 工具预算 2 / 40 · 单轮上限 8");
  assert.match(metrics.toolCallsAriaLabel, /累计启动 2 次.*已结束 1 次.*执行中 1 次/);
});

test("runtime metrics remain honest for legacy or malformed trace snapshots", async () => {
  const { runtimeMetricView } = await importBrowserModule(
    "src/coding_agent/web/static/_metrics.js",
  );

  const legacy = runtimeMetricView({
    active_tools: [],
    current_step: 3,
    limits: { max_model_turns: "20" },
    tools_finished: 1,
    tools_started: 1,
  });

  assert.equal(legacy.modelTurnsText, "3");
  assert.equal(legacy.toolCallsText, "1");
  assert.equal(legacy.toolBudgetText, "执行中 0 · 旧记录未保存运行上限");
  assert.match(legacy.modelTurnsAriaLabel, /旧记录未保存轮次上限/);
});

test("tool batch rejection copy describes a retry rather than a failed tool", async () => {
  const { translateTimelineDetail, translateTimelineHeadline } = await importBrowserModule(
    "src/coding_agent/web/static/locale-zh.js",
  );

  assert.equal(
    translateTimelineHeadline("Tool batch too large; split retry requested"),
    "工具批次过大，正在拆分重试",
  );
  assert.equal(
    translateTimelineDetail(
      "Requested 9 tool calls; per-turn limit 8; split retry requested; rejection 1 of 3",
    ),
    "本轮请求 9 个工具调用，上限 8；已要求模型拆分重试（第 1 / 3 次调整）",
  );
});

test("protocol correction copy is distinct from transport backoff", async () => {
  const { translateTimelineDetail, translateTimelineHeadline } = await importBrowserModule(
    "src/coding_agent/web/static/locale-zh.js",
  );

  assert.equal(
    translateTimelineHeadline("Invalid model response; protocol correction scheduled"),
    "模型工具参数格式异常，正在自动纠正",
  );
  assert.equal(
    translateTimelineDetail("Attempt 2 of 3 · after 0s · MODEL RESPONSE INVALID"),
    "第 2 / 3 次请求 · 立即重试 · 模型工具参数格式无效",
  );
  assert.equal(
    translateTimelineDetail("Attempt 2 of 3 · after 0.5s · MODEL REQUEST TRANSIENT"),
    "第 2 / 3 次请求 · 0.5 秒后重试 · 瞬态请求错误",
  );
});

test("verification closeout copy does not imply a model request failure", async () => {
  const { translatePhase, translateTimelineDetail, translateTimelineHeadline } =
    await importBrowserModule("src/coding_agent/web/static/locale-zh.js");

  assert.equal(translatePhase("VERIFYING", []), "正在验证");
  assert.equal(
    translateTimelineHeadline("Final response deferred; verification scheduled"),
    "暂缓结束，正在安排验证",
  );
  assert.equal(translateTimelineDetail("Fresh verification is required"), "需要重新验证");
});

test("history final fallback distinguishes missing, failed, and interrupted results", async () => {
  const { historyFinalFallback } = await importBrowserModule(
    "src/coding_agent/web/static/locale-zh.js",
  );

  const missing = historyFinalFallback("completed");
  assert.equal(missing, "未能从终态检查点恢复最终回复；代码修改与验证记录仍已保留。");
  assert.doesNotMatch(missing, /未持久化/);
  assert.equal(
    historyFinalFallback("failed"),
    "此次运行以失败结束；这里只回放经过白名单过滤的事件，错误详情未纳入历史回放。",
  );
  assert.equal(
    historyFinalFallback("interrupted"),
    "此次运行的轨迹未正常终止；这里只回放中断前经过白名单过滤的事件。",
  );
});

test("diff view shows small previews completely and folds larger previews", async () => {
  const { COLLAPSED_DIFF_LINES, diffPreviewView } = await importBrowserModule(
    "src/coding_agent/web/static/_diff_view.js",
  );
  const smallLines = ["--- a/app.py", "+++ b/app.py", "@@ -1 +1 @@", "-old", "+new"];
  const small = diffPreviewView({ expanded_preview: smallLines });
  assert.equal(small.canToggle, false);
  assert.deepEqual(small.visibleLines, smallLines);

  const longLines = Array.from({ length: COLLAPSED_DIFF_LINES + 4 }, (_, index) => `+line ${index}`);
  const folded = diffPreviewView({ expanded_preview: longLines });
  assert.equal(folded.canToggle, true);
  assert.equal(folded.isExpanded, false);
  assert.equal(folded.visibleLines.length, COLLAPSED_DIFF_LINES);
  assert.equal(folded.toggleLabel, `展开 Diff（${longLines.length} 行）`);

  const expanded = diffPreviewView({ expanded_preview: longLines }, true);
  assert.equal(expanded.isExpanded, true);
  assert.deepEqual(expanded.visibleLines, longLines);
  assert.equal(expanded.toggleLabel, "收起 Diff");
});

test("diff view labels bounded previews honestly and ignores malformed lines", async () => {
  const { diffPreviewView } = await importBrowserModule(
    "src/coding_agent/web/static/_diff_view.js",
  );
  const view = diffPreviewView({
    expanded_preview: ["+safe", { private: "must-not-render" }, 3, ...Array(9).fill("+line")],
    expanded_preview_complete: false,
    preview: "not-an-array",
  });

  assert.equal(view.previewTruncated, true);
  assert.equal(view.totalLines, 10);
  assert.match(view.toggleLabel, /^展开可用预览/);
  assert.match(view.note, /安全上限/);
  assert.doesNotMatch(view.toggleLabel, /完整/);
  assert.equal(view.visibleLines.includes("must-not-render"), false);
});
