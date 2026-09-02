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

test("activity view replaces paired starts with their finished activity", async () => {
  const { visibleActivityEntries } = await importBrowserModule(
    "src/coding_agent/web/static/_activity_view.js",
  );
  const unpaired = { activity_id: "read-2", activity_state: "started", headline: "reading" };
  const finished = { activity_id: "read-1", activity_state: "finished", headline: "read" };
  const standalone = { activity_id: null, activity_state: null, headline: "checkpoint" };
  const visible = visibleActivityEntries([
    { activity_id: "read-1", activity_state: "started", headline: "reading" },
    unpaired,
    finished,
    standalone,
  ]);

  assert.deepEqual(visible, [unpaired, finished, standalone]);
});

test("activity details are available only for bounded public facts and default folded", async () => {
  const { activityCardView } = await importBrowserModule(
    "src/coding_agent/web/static/_activity_view.js",
  );
  const entry = {
    facts: [
      { format: "code", label: "命令", value: "python -m pytest" },
      { format: "status", label: "结果", value: "通过" },
    ],
    facts_complete: true,
  };
  const folded = activityCardView(entry);
  assert.equal(folded.canToggle, true);
  assert.equal(folded.isExpanded, false);
  assert.equal(folded.factsComplete, true);
  assert.equal(folded.toggleLabel, "查看操作详情");

  const expanded = activityCardView(entry, true);
  assert.equal(expanded.isExpanded, true);
  assert.equal(expanded.toggleLabel, "收起操作详情");

  const empty = activityCardView({ facts: [], facts_complete: true }, true);
  assert.equal(empty.canToggle, false);
  assert.equal(empty.isExpanded, false);
});

test("activity facts cap at twelve, preserve preformatted output, and degrade unknown formats", async () => {
  const { MAX_ACTIVITY_FACTS, activityCardView } = await importBrowserModule(
    "src/coding_agent/web/static/_activity_view.js",
  );
  const facts = Array.from({ length: MAX_ACTIVITY_FACTS + 2 }, (_, index) => ({
    format: index === 0 ? "private-html" : index === 1 ? "pre" : "text",
    label: `字段 ${index}`,
    value: `值 ${index}`,
  }));
  const view = activityCardView({ facts, facts_complete: true });

  assert.equal(view.facts.length, MAX_ACTIVITY_FACTS);
  assert.equal(view.facts[0].format, "text");
  assert.equal(view.facts[1].format, "pre");
  assert.equal(view.factsComplete, false);
  assert.equal(view.facts.some((fact) => fact.label === `字段 ${MAX_ACTIVITY_FACTS}`), false);
});

test("activity expansion keys are stable, run-scoped, and prefer activity ids", async () => {
  const { activityEntryKey } = await importBrowserModule(
    "src/coding_agent/web/static/_activity_view.js",
  );
  const entry = {
    activity_id: "command-1",
    headline: "run command",
    offset_seconds: 1,
    step: 2,
  };

  assert.equal(activityEntryKey(entry, "run-1", 0), activityEntryKey(entry, "run-1", 9));
  assert.notEqual(activityEntryKey(entry, "run-1", 0), activityEntryKey(entry, "run-2", 0));

  const started = {
    ...entry,
    activity_state: "started",
    category: "TOOL",
    headline: "Running run_command",
  };
  const finished = {
    ...entry,
    activity_state: "finished",
    category: "TOOL",
    headline: "run_command completed",
  };
  const verification = {
    ...entry,
    activity_state: null,
    category: "VERIFY",
    headline: "Passing evidence recorded",
  };
  assert.equal(
    activityEntryKey(started, "run-1", 0),
    activityEntryKey(finished, "run-1", 1),
  );
  assert.notEqual(
    activityEntryKey(finished, "run-1", 1),
    activityEntryKey(verification, "run-1", 2),
  );
  assert.equal(
    activityEntryKey(verification, "run-1", 2),
    activityEntryKey(verification, "run-1", 7),
  );
});

test("activity fact labels and transparent values are localized without changing commands", async () => {
  const { translateActivityFactLabel, translateActivityFactValue } =
    await importBrowserModule("src/coding_agent/web/static/locale-zh.js");

  assert.equal(translateActivityFactLabel("Command"), "命令");
  assert.equal(translateActivityFactLabel("Verification"), "验证项");
  assert.equal(translateActivityFactLabel("Working directory"), "工作目录");
  assert.equal(translateActivityFactLabel("Captured output"), "捕获输出");
  assert.equal(translateActivityFactLabel("Agent observation"), "Agent 实际观察");
  assert.equal(translateActivityFactLabel("Output status"), "输出状态");
  assert.equal(translateActivityFactLabel("Redaction"), "脱敏说明");
  assert.equal(translateActivityFactLabel("Timeout"), "超时上限");
  assert.equal(translateActivityFactLabel("Workspace revision"), "工作区修订");
  assert.equal(translateActivityFactLabel("Required scopes"), "必需范围");
  assert.equal(
    translateActivityFactValue('python -I -B -m pytest "tests/a b.py"', "pre"),
    'python -I -B -m pytest "tests/a b.py"',
  );
  assert.equal(translateActivityFactValue("  line 1\nline 2\n", "pre"), "  line 1\nline 2\n");
  assert.equal(translateActivityFactValue("verifier"), "可信验证命令");
  assert.equal(translateActivityFactValue("test / pytest"), "测试 / pytest");
  assert.equal(translateActivityFactValue("Exited with code 0", "status"), "已退出，退出码 0");
  assert.equal(translateActivityFactValue("30 second(s)"), "30 秒");
  assert.equal(
    translateActivityFactValue("Credential-like values redacted"),
    "检测到凭据样式内容，相关值已脱敏",
  );
  assert.equal(
    translateActivityFactValue("Agent observation compacted"),
    "Agent 实际观察因上下文预算被压缩",
  );
  assert.equal(
    translateActivityFactValue("tests, runtime:entrypoint"),
    "测试，运行时入口",
  );
  assert.equal(
    translateActivityFactValue(
      "Captured 12 of 20 byte(s); runtime capture truncated; Agent observation compacted",
    ),
    "已捕获 12 / 20 字节；运行时捕获已截断；Agent 实际观察因上下文预算被压缩",
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

test("workspace change ledger preserves every change and falls back for older responses", async () => {
  const { workspaceChangeKey, workspaceChangeLedgerView } = await importBrowserModule(
    "src/coding_agent/web/static/_diff_view.js",
  );
  const first = { activity_id: "act_1111111111111111", headline: "domain first", step: 7 };
  const second = { activity_id: "act_2222222222222222", headline: "domain second", step: 8 };
  const current = workspaceChangeLedgerView({
    latest_change: second,
    omitted_change_count: 3,
    workspace_changes: [first, second],
    workspace_changes_complete: false,
  });

  assert.deepEqual(current.changes, [first, second]);
  assert.equal(current.changes.length, 2, "latest_change must not be rendered twice");
  assert.equal(current.omittedCount, 3);
  assert.match(current.note, /较早的 3 次/);
  assert.match(current.note, /最近 2 次/);

  assert.deepEqual(workspaceChangeLedgerView({ latest_change: first }).changes, [first]);
  assert.deepEqual(workspaceChangeLedgerView({}).changes, []);
  assert.notEqual(workspaceChangeKey(first, "run-1"), workspaceChangeKey(second, "run-1"));
});
