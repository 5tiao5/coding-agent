import {
  historyFinalFallback,
  translateMetadataLabel,
  translatePhase,
  translateServerDetail,
  translateTimelineDetail,
  translateTimelineHeadline,
} from "./locale-zh.js";
import { createWorkbench } from "./_workbench.js";
import { runtimeMetricView } from "./_metrics.js";

"use strict";

const POLL_INTERVAL_MS = 250;
const MAX_VISIBLE_TIMELINE = 12;

const elements = {
  activeToolList: document.querySelector("#active-tool-list"),
  activeTools: document.querySelector("#active-tools"),
  connectionDot: document.querySelector("#connection-dot"),
  connectionLabel: document.querySelector("#connection-label"),
  changeScopeList: document.querySelector("#change-scope-list"),
  conversation: document.querySelector("#conversation"),
  emptyState: document.querySelector("#empty-state"),
  evidenceList: document.querySelector("#evidence-list"),
  footerLabel: document.querySelector("#footer-label"),
  footerPulse: document.querySelector("#footer-pulse"),
  form: document.querySelector("#run-form"),
  formMessage: document.querySelector("#form-message"),
  historyBadge: document.querySelector("#history-badge"),
  historyReturn: document.querySelector("#history-return"),
  historyTitle: document.querySelector("#history-title"),
  messageList: document.querySelector("#message-list"),
  metricFailed: document.querySelector("#metric-failed"),
  metricBudget: document.querySelector("#metric-budget"),
  metricStep: document.querySelector("#metric-step"),
  metricTools: document.querySelector("#metric-tools"),
  newProjectButton: document.querySelector("#new-project-button"),
  openProjectButton: document.querySelector("#open-project-button"),
  phaseLabel: document.querySelector("#phase-label"),
  planCount: document.querySelector("#plan-count"),
  planList: document.querySelector("#plan-list"),
  projectDialog: document.querySelector("#project-dialog"),
  projectDialogCancel: document.querySelector("#project-dialog-cancel"),
  projectDialogClose: document.querySelector("#project-dialog-close"),
  projectDialogCopy: document.querySelector("#project-dialog-copy"),
  projectDialogForm: document.querySelector("#project-dialog-form"),
  projectDialogMessage: document.querySelector("#project-dialog-message"),
  projectDialogSubmit: document.querySelector("#project-dialog-submit"),
  projectDialogTitle: document.querySelector("#project-dialog-title"),
  projectDisplayNameField: document.querySelector("#project-display-name-field"),
  projectDisplayNameInput: document.querySelector("#project-display-name-input"),
  projectBrowser: document.querySelector("#project-browser"),
  projectList: document.querySelector("#project-list"),
  projectNameField: document.querySelector("#project-name-field"),
  projectNameInput: document.querySelector("#project-name-input"),
  projectRootBrowse: document.querySelector("#project-root-browse"),
  projectRootInput: document.querySelector("#project-root-input"),
  projectRootLabel: document.querySelector("#project-root-label"),
  runButton: document.querySelector("#run-button"),
  runButtonLabel: document.querySelector("#run-button-label"),
  runId: document.querySelector("#run-id"),
  runHistoryList: document.querySelector("#run-history-list"),
  runStatus: document.querySelector("#run-status"),
  runStatusLabel: document.querySelector("#run-status-label"),
  runtimeLabel: document.querySelector("#runtime-label"),
  sessionHeading: document.querySelector("#session-heading"),
  taskInput: document.querySelector("#task-input"),
  toast: document.querySelector("#toast"),
  toastText: document.querySelector("#toast-text"),
  verificationOrb: document.querySelector("#verification-orb"),
  verificationOutcome: document.querySelector("#verification-outcome"),
  verificationStatus: document.querySelector("#verification-status"),
  workspaceName: document.querySelector("#workspace-name"),
};

let appMetadata = {
  controlToken: "",
  defaultTask: "",
  nativeFolderPickerAvailable: false,
  taskLocked: false,
};
let currentState = null;
let liveState = null;
let workbench = null;
let lastRenderSignature = "";
let lastServerTask = "";
let pollInFlight = false;
let consecutivePollFailures = 0;
let toastTimer = null;
let taskInputDirty = false;

const RUN_STATUS = {
  completed: { label: "已完成", phase: "运行已完成", tone: "completed" },
  completed_unverified: {
    label: "未验证",
    phase: "运行已完成，但缺少最新验证证据",
    tone: "unverified",
  },
  failed: { label: "失败", phase: "运行因错误停止", tone: "failed" },
  idle: { label: "就绪", phase: "等待任务", tone: "idle" },
  interrupted: { label: "已中断", phase: "运行轨迹未正常终止", tone: "unverified" },
  running: { label: "运行中", phase: "智能体正在工作", tone: "running" },
};

const CATEGORY_META = {
  CONTEXT: { label: "上下文", symbol: "C" },
  MODEL: { label: "决策", symbol: "M" },
  PLAN: { label: "计划", symbol: "P" },
  RUN: { label: "运行", symbol: "R" },
  SAVE: { label: "存档", symbol: "S" },
  SESSION: { label: "会话", symbol: "S" },
  TOOL: { label: "工具", symbol: "T" },
  VERIFY: { label: "验证", symbol: "V" },
};

const LEVELS = new Set(["info", "success", "warning", "error"]);
const PLAN_STATES = new Set(["pending", "in_progress", "completed"]);
const TERMINAL_STATUSES = new Set([
  "completed",
  "completed_unverified",
  "failed",
  "interrupted",
]);

function asObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function asText(value, fallback = "") {
  return typeof value === "string" ? value : fallback;
}

function asNumber(value, fallback = 0) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function createElement(tagName, className, text) {
  const element = document.createElement(tagName);
  if (className) {
    element.className = className;
  }
  if (text !== undefined) {
    element.textContent = text;
  }
  return element;
}

function normalizeRunStatus(value) {
  return Object.hasOwn(RUN_STATUS, value) ? value : "idle";
}

function normalizedLevel(value) {
  return LEVELS.has(value) ? value : "info";
}

function formatPhase(snapshot, status) {
  const phase = asText(snapshot.phase).trim();
  if (status === "running" && phase) {
    return translatePhase(phase, asArray(snapshot.active_tools));
  }
  return RUN_STATUS[status].phase;
}

function formatDuration(durationMs) {
  const value = asNumber(durationMs, -1);
  if (value < 0) {
    return "";
  }
  if (value < 1000) {
    return `${Math.round(value)} ms`;
  }
  return `${(value / 1000).toFixed(value < 10000 ? 1 : 0)} s`;
}

function formatOffset(offsetSeconds) {
  const value = asNumber(offsetSeconds, -1);
  if (value < 0) {
    return "";
  }
  return `+${value.toFixed(value < 10 ? 1 : 0)}s`;
}

function shortRunId(runId) {
  const text = asText(runId).trim();
  return text.length > 12 ? text.slice(0, 8) : text;
}

function mutationHeaders() {
  const headers = {
    Accept: "application/json",
    "Content-Type": "application/json",
  };
  if (appMetadata.controlToken) {
    headers["X-Coding-Agent-Token"] = appMetadata.controlToken;
  }
  return headers;
}

function setConnection(connected) {
  elements.connectionDot.classList.toggle("is-offline", !connected);
  elements.connectionLabel.textContent = connected ? "本地服务已连接" : "服务不可用";
}

function setFormState(status) {
  const running = status === "running";
  const hasProject = appMetadata.controlToken
    ? workbench?.hasSelectedProject() === true
    : true;
  elements.taskInput.disabled = running || appMetadata.taskLocked || !hasProject;
  elements.runButton.disabled = running || !hasProject;
  elements.runButton.classList.toggle("is-running", running);
  elements.runButtonLabel.textContent = running
    ? "智能体运行中"
    : hasProject
      ? "启动智能体"
      : "请先选择项目";
  workbench?.updateControls(status);
}

function renderMetadata(rawMetadata) {
  const metadata = asObject(rawMetadata);
  appMetadata = {
    controlToken: asText(metadata.control_token).trim(),
    defaultTask: asText(metadata.default_task).trim(),
    nativeFolderPickerAvailable: metadata.native_folder_picker_available === true,
    taskLocked: metadata.task_locked === true,
  };
  elements.projectBrowser.hidden = !appMetadata.controlToken;
  const project = workbench?.projectSummary();
  elements.workspaceName.textContent = project
    ? project.name
    : translateMetadataLabel(metadata.workspace, "尚未选择项目");
  elements.runtimeLabel.textContent = project
    ? project.root
    : translateMetadataLabel(metadata.runtime, "请选择本地项目");
  if (appMetadata.defaultTask && (!taskInputDirty || appMetadata.taskLocked)) {
    elements.taskInput.value = appMetadata.defaultTask;
  }
  setFormState(normalizeRunStatus(asText(currentState?.status)));
}

function preserveServerTask(state) {
  if (workbench?.isReplaying()) {
    return;
  }
  const serverTask = asText(state.task).trim();
  if (!serverTask) {
    return;
  }
  if (!lastServerTask && !taskInputDirty && !elements.taskInput.value.trim()) {
    // The offline demo starts before the browser opens. Hydrating its fixed task
    // keeps the completed screen replayable instead of presenting a misleading
    // blank composer whose arbitrary input the deterministic runner cannot accept.
    elements.taskInput.value = serverTask;
  }
  lastServerTask = serverTask;
}

function renderHeader(state, snapshot, status) {
  const meta = RUN_STATUS[status];
  elements.runStatus.className = `status-pill status-${meta.tone}`;
  elements.runStatusLabel.textContent = meta.label;
  elements.phaseLabel.textContent = formatPhase(snapshot, status);
  elements.footerLabel.textContent = status === "running" ? formatPhase(snapshot, status) : meta.label;
  elements.footerPulse.classList.toggle("is-running", status === "running");

  const runId = asText(state.run_id);
  elements.runId.hidden = !runId;
  elements.runId.textContent = runId ? `#${shortRunId(runId)}` : "";
}

function makeAvatar(kind) {
  const avatar = createElement("div", `message-avatar avatar-${kind}`);
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = kind === "user" ? "你" : "R";
  return avatar;
}

function makeMessageShell(kind, author) {
  const row = createElement("article", `message-row message-${kind}`);
  const body = createElement("div", "message-body");
  const heading = createElement("header", "message-heading");
  heading.append(createElement("strong", "", author));
  body.append(heading);
  row.append(makeAvatar(kind), body);
  return { body, heading, row };
}

function appendInlineText(container, text) {
  const source = asText(text);
  const fragments = source.split(/(`[^`\n]+`)/g);
  for (const fragment of fragments) {
    if (fragment.startsWith("`") && fragment.endsWith("`") && fragment.length > 2) {
      container.append(createElement("code", "inline-code", fragment.slice(1, -1)));
    } else if (fragment) {
      container.append(document.createTextNode(fragment));
    }
  }
}

function appendRichText(container, text) {
  const lines = asText(text).replaceAll("\r\n", "\n").split("\n");
  let codeLines = null;
  let list = null;

  function closeList() {
    if (list) {
      container.append(list);
      list = null;
    }
  }

  function closeCode() {
    if (codeLines) {
      const pre = createElement("pre", "answer-code");
      const code = createElement("code");
      code.textContent = codeLines.join("\n");
      pre.append(code);
      container.append(pre);
      codeLines = null;
    }
  }

  for (const line of lines) {
    if (line.trimStart().startsWith("```")) {
      closeList();
      if (codeLines) {
        closeCode();
      } else {
        codeLines = [];
      }
      continue;
    }
    if (codeLines) {
      codeLines.push(line);
      continue;
    }

    const headingMatch = line.match(/^\s{0,3}(#{1,3})\s+(.+)$/);
    if (headingMatch) {
      closeList();
      const heading = createElement("h3", `answer-heading answer-heading-${headingMatch[1].length}`);
      appendInlineText(heading, headingMatch[2]);
      container.append(heading);
      continue;
    }

    const bulletMatch = line.match(/^\s*[-*]\s+(.+)$/);
    if (bulletMatch) {
      if (!list) {
        list = createElement("ul", "answer-list");
      }
      const item = createElement("li");
      appendInlineText(item, bulletMatch[1]);
      list.append(item);
      continue;
    }

    closeList();
    if (!line.trim()) {
      continue;
    }
    const paragraph = createElement("p", "answer-paragraph");
    appendInlineText(paragraph, line);
    container.append(paragraph);
  }

  closeList();
  closeCode();
}

function renderUserMessage(task) {
  const message = makeMessageShell("user", "你");
  message.heading.append(createElement("span", "message-role", "任务"));
  message.body.append(createElement("p", "user-task", task));
  return message.row;
}

function categoryMeta(entry) {
  const key = asText(entry.category).toUpperCase();
  return CATEGORY_META[key] || { label: key || "活动", symbol: "·" };
}

function renderPreview(preview) {
  const block = createElement("div", "tool-preview");
  for (const line of asArray(preview).slice(0, 12)) {
    block.append(createElement("code", "preview-line", asText(line, String(line))));
  }
  return block;
}

function renderTimelineEntry(rawEntry) {
  const entry = asObject(rawEntry);
  const level = normalizedLevel(asText(entry.level));
  const meta = categoryMeta(entry);
  const card = createElement("article", `activity-card level-${level}`);

  const rail = createElement("div", "activity-rail");
  rail.append(createElement("span", "activity-symbol", meta.symbol));

  const content = createElement("div", "activity-content");
  const heading = createElement("div", "activity-heading");
  const titleGroup = createElement("div", "activity-title-group");
  titleGroup.append(
    createElement("span", "activity-category", meta.label),
    createElement(
      "strong",
      "activity-title",
      translateTimelineHeadline(asText(entry.headline, "智能体活动")),
    ),
  );

  const timing = createElement("div", "activity-timing");
  const duration = formatDuration(entry.duration_ms);
  const offset = formatOffset(entry.offset_seconds);
  if (duration) {
    timing.append(createElement("span", "", duration));
  }
  if (offset) {
    timing.append(createElement("span", "", offset));
  }
  heading.append(titleGroup, timing);
  content.append(heading);

  const detail = asText(entry.detail).trim();
  if (detail) {
    content.append(createElement("p", "activity-detail", translateTimelineDetail(detail)));
  }
  const preview = asArray(entry.preview);
  if (preview.length && asText(entry.category).toUpperCase() !== "TOOL") {
    content.append(renderPreview(preview));
  }

  card.append(rail, content);
  return card;
}

function renderThinkingCard(snapshot) {
  const card = createElement("div", "thinking-card");
  const dots = createElement("span", "thinking-dots");
  dots.append(createElement("i"), createElement("i"), createElement("i"));
  card.append(dots, createElement("span", "", `${formatPhase(snapshot, "running")}…`));
  return card;
}

function renderDiffLine(line) {
  const text = asText(line, String(line));
  let tone = "context";
  if (text.startsWith("+") && !text.startsWith("+++")) {
    tone = "addition";
  } else if (text.startsWith("-") && !text.startsWith("---")) {
    tone = "deletion";
  } else if (text.startsWith("@@")) {
    tone = "hunk";
  } else if (text.startsWith("+++") || text.startsWith("---")) {
    tone = "header";
  }
  return createElement("code", `diff-line diff-${tone}`, text || " ");
}

function renderLatestChange(rawChange) {
  const change = asObject(rawChange);
  if (!Object.keys(change).length) {
    return null;
  }

  const card = createElement("section", "change-card");
  const heading = createElement("header", "change-heading");
  const title = createElement("div");
  title.append(
    createElement("span", "change-mark", "±"),
    createElement(
      "strong",
      "",
      translateTimelineHeadline(asText(change.headline, "最近变更")),
    ),
  );
  heading.append(title, createElement("span", "change-badge", "工作区变更"));
  card.append(heading);

  const detail = asText(change.detail).trim();
  if (detail) {
    card.append(createElement("p", "change-detail", translateTimelineDetail(detail)));
  }

  const preview = asArray(change.preview);
  if (preview.length) {
    const diff = createElement("div", "diff-block");
    for (const line of preview.slice(0, 18)) {
      diff.append(renderDiffLine(line));
    }
    card.append(diff);
  }
  return card;
}

function renderFinalMessage(finalText, status, error) {
  const failed = status === "failed" || status === "interrupted";
  const message = makeMessageShell("agent", "Relay");
  message.heading.append(
    createElement(
      "span",
      `message-role ${failed ? "role-error" : "role-complete"}`,
      failed ? "已停止" : "最终回复",
    ),
  );
  const answer = createElement("div", `final-answer ${failed ? "final-error" : ""}`);
  const content = asText(finalText).trim() || asText(error).trim();
  const historyFallback = workbench?.isReplaying() ? historyFinalFallback(status) : "";
  appendRichText(answer, content || (failed ? "运行失败，但没有错误信息。" : "运行已完成。"));
  if (!content && historyFallback) {
    answer.replaceChildren();
    appendRichText(answer, historyFallback);
  }
  message.body.append(answer);
  return message.row;
}

function renderConversation(state, snapshot, status) {
  const task = asText(state.task).trim() || asText(snapshot.task_label).trim();
  const hasRun = Boolean(asText(state.run_id) || task || status !== "idle");
  elements.emptyState.hidden = hasRun;
  if (!hasRun) {
    elements.messageList.replaceChildren();
    return;
  }

  const wasNearBottom =
    elements.conversation.scrollHeight - elements.conversation.scrollTop - elements.conversation.clientHeight < 100;
  const fragment = document.createDocumentFragment();

  if (task) {
    fragment.append(renderUserMessage(task));
  }

  const agentMessage = makeMessageShell("agent", "Relay");
  agentMessage.heading.append(
    createElement("span", "message-role", status === "running" ? "进行中" : "活动"),
  );
  const activityStack = createElement("div", "activity-stack");
  const timeline = asArray(snapshot.timeline).slice(-MAX_VISIBLE_TIMELINE);
  for (const entry of timeline) {
    activityStack.append(renderTimelineEntry(entry));
  }
  if (!timeline.length && status === "running") {
    activityStack.append(renderThinkingCard(snapshot));
  }

  const latestChange = renderLatestChange(snapshot.latest_change);
  if (latestChange) {
    activityStack.append(latestChange);
  }
  if (activityStack.childElementCount) {
    agentMessage.body.append(activityStack);
    fragment.append(agentMessage.row);
  }

  if (TERMINAL_STATUSES.has(status)) {
    fragment.append(renderFinalMessage(state.final_text, status, state.error));
  }

  elements.messageList.replaceChildren(fragment);
  if (wasNearBottom || status === "running") {
    window.requestAnimationFrame(() => {
      elements.conversation.scrollTop = elements.conversation.scrollHeight;
    });
  }
}

function parsePlanLine(rawLine, index) {
  const source = asText(rawLine, String(rawLine)).trim();
  const match = source.match(/^\s*(?:[-*]\s*)?\[(pending|in_progress|completed)\]\s*(.*)$/i);
  if (!match) {
    return { index, state: "pending", text: source.replace(/^[-*]\s*/, "") || `步骤 ${index + 1}` };
  }
  const state = match[1].toLowerCase();
  const content = match[2].trim();
  const identifier = content.match(/^[a-z0-9_-]{1,40}:\s+(.+)$/i);
  return {
    index,
    state: PLAN_STATES.has(state) ? state : "pending",
    text: (identifier ? identifier[1] : content) || `步骤 ${index + 1}`,
  };
}

function renderPlan(snapshot) {
  const plan = asArray(snapshot.plan_lines).map(parsePlanLine);
  const completed = plan.filter((item) => item.state === "completed").length;
  elements.planCount.textContent = `${completed} / ${plan.length}`;

  if (!plan.length) {
    elements.planList.replaceChildren(
      createElement("p", "panel-placeholder", "运行开始后，这里会显示结构化计划。"),
    );
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const item of plan) {
    const row = createElement("div", `plan-item plan-${item.state}`);
    const marker = createElement("span", "plan-marker");
    marker.setAttribute("aria-hidden", "true");
    if (item.state === "completed") {
      marker.textContent = "✓";
    } else if (item.state === "in_progress") {
      marker.textContent = "";
      marker.append(createElement("i"));
    } else {
      marker.textContent = String(item.index + 1);
    }
    const copy = createElement("div", "plan-copy");
    copy.append(
      createElement("p", "", item.text),
      createElement(
        "span",
        "",
        item.state === "completed" ? "已完成" : item.state === "in_progress" ? "进行中" : "待处理",
      ),
    );
    row.append(marker, copy);
    fragment.append(row);
  }
  elements.planList.replaceChildren(fragment);
}

function verificationCopy(status, outcome) {
  if (outcome === "VERIFIED" || status === "verified") {
    return { detail: "最新变更后的检查已通过", label: "已验证", tone: "verified" };
  }
  if (status === "failed") {
    return { detail: "至少一项验证检查失败", label: "失败", tone: "failed" };
  }
  if (status === "stale") {
    return { detail: "上次检查通过后代码又发生了变更", label: "已过期", tone: "stale" };
  }
  if (status === "checks_only") {
    return { detail: "已配置检查通过，但任务验收范围仍不完整", label: "部分验证", tone: "stale" };
  }
  if (status === "missing" || outcome === "UNVERIFIED") {
    return { detail: "没有最新的通过证据", label: "未验证", tone: "missing" };
  }
  return { detail: "等待外部检查证据", label: "等待验证", tone: "pending" };
}

function renderVerification(snapshot, runStatus) {
  let status = asText(snapshot.verification_status, "pending").toLowerCase();
  if (runStatus === "failed" && status === "pending") {
    status = "failed";
  }
  const outcome = asText(snapshot.outcome, "RUNNING").toUpperCase();
  const copy = verificationCopy(status, outcome);
  elements.verificationOutcome.textContent = copy.label;
  elements.verificationStatus.textContent = copy.detail;
  elements.verificationOrb.className = `verification-orb verification-${copy.tone}`;

  const changes = asArray(snapshot.changed_files).map(asObject).filter((item) => asText(item.path).trim());
  if (!changes.length) {
    elements.changeScopeList.replaceChildren(createElement("span", "evidence-empty", "尚无文件变更"));
  } else {
    const changeFragment = document.createDocumentFragment();
    for (const change of changes.slice(0, 12)) {
      const path = asText(change.path).trim();
      const additions = Math.max(0, asNumber(change.added_lines));
      const deletions = Math.max(0, asNumber(change.removed_lines));
      const item = createElement("span", "evidence-item change-scope-item");
      item.append(
        createElement("i", "", "±"),
        document.createTextNode(`${path}  +${additions}/-${deletions}`),
      );
      changeFragment.append(item);
    }
    elements.changeScopeList.replaceChildren(changeFragment);
  }

  const evidence = asArray(snapshot.verification_evidence).map(asObject);
  if (!evidence.length) {
    const labels = asArray(snapshot.verification_labels)
      .map((item) => asText(item, String(item)))
      .filter(Boolean);
    if (labels.length) {
      const fallbackFragment = document.createDocumentFragment();
      for (const label of labels.slice(0, 8)) {
        const item = createElement("span", "evidence-item");
        item.append(createElement("i", "", "✓"), document.createTextNode(label));
        fallbackFragment.append(item);
      }
      elements.evidenceList.replaceChildren(fallbackFragment);
      return;
    }
    elements.evidenceList.replaceChildren(createElement("span", "evidence-empty", "尚无检查记录"));
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const evidenceItem of evidence.slice(0, 8)) {
    const label = asText(evidenceItem.label, "检查");
    const kind = asText(evidenceItem.kind, "check").toUpperCase();
    const step = Math.max(0, asNumber(evidenceItem.step));
    const epoch = Math.max(0, asNumber(evidenceItem.epoch));
    const passed = evidenceItem.passed === true;
    const item = createElement("span", "evidence-item");
    item.classList.toggle("evidence-failed", !passed);
    item.append(
      createElement("i", "", passed ? "✓" : "×"),
      document.createTextNode(`${label} · ${kind} · 步骤 ${step} · 修订 ${epoch}`),
    );
    fragment.append(item);
  }
  elements.evidenceList.replaceChildren(fragment);
}

function renderMetrics(snapshot) {
  const metrics = runtimeMetricView(snapshot);
  const failed = Math.max(0, asNumber(snapshot.tools_failed));
  elements.metricStep.textContent = metrics.modelTurnsText;
  elements.metricStep.setAttribute("aria-label", metrics.modelTurnsAriaLabel);
  elements.metricStep.title = metrics.modelTurnsAriaLabel;
  elements.metricTools.textContent = metrics.toolCallsText;
  elements.metricTools.setAttribute("aria-label", metrics.toolCallsAriaLabel);
  elements.metricTools.title = metrics.toolCallsAriaLabel;
  elements.metricBudget.textContent = metrics.toolBudgetText;
  elements.metricFailed.textContent = String(failed);
  elements.metricFailed.classList.toggle("has-failures", failed > 0);

  const active = asArray(snapshot.active_tools).map((item) => asText(item, String(item))).filter(Boolean);
  elements.activeTools.hidden = !active.length;
  const fragment = document.createDocumentFragment();
  for (const tool of active.slice(0, 5)) {
    const chip = createElement("span", "active-tool-chip");
    chip.append(createElement("i"), document.createTextNode(tool));
    fragment.append(chip);
  }
  elements.activeToolList.replaceChildren(fragment);
}

function renderState(rawState) {
  const state = asObject(rawState);
  const snapshot = asObject(state.snapshot);
  const status = normalizeRunStatus(asText(state.status));
  const signature = JSON.stringify(state);
  if (signature === lastRenderSignature) {
    setConnection(true);
    return;
  }
  lastRenderSignature = signature;
  currentState = state;

  setConnection(true);
  preserveServerTask(state);
  setFormState(status);
  renderHeader(state, snapshot, status);
  renderConversation(state, snapshot, status);
  renderPlan(snapshot);
  renderVerification(snapshot, status);
  renderMetrics(snapshot);
}

function showToast(message, tone = "error") {
  if (toastTimer !== null) {
    window.clearTimeout(toastTimer);
  }
  elements.toast.className = `toast toast-${tone}`;
  elements.toastText.textContent = message;
  elements.toast.hidden = false;
  toastTimer = window.setTimeout(() => {
    elements.toast.hidden = true;
    toastTimer = null;
  }, 4200);
}

workbench = createWorkbench({
  elements,
  formatServerDetail: translateServerDetail,
  getControlToken: () => appMetadata.controlToken,
  getNativeFolderPickerAvailable: () => appMetadata.nativeFolderPickerAvailable,
  onBusyStateChanged() {
    setFormState(normalizeRunStatus(asText(currentState?.status)));
  },
  onBeforeProjectChanged() {
    currentState = null;
    liveState = null;
    lastRenderSignature = "";
    lastServerTask = "";
    taskInputDirty = false;
    if (!appMetadata.taskLocked) {
      elements.taskInput.value = "";
    }
  },
  onHistoryState(state) {
    lastRenderSignature = "";
    renderState(state);
  },
  async onProjectChanged() {
    await fetchMetadata();
    await fetchState();
  },
  onReturnLive(state) {
    lastRenderSignature = "";
    renderState(state);
  },
  showToast,
});

async function fetchState() {
  if (pollInFlight) {
    return;
  }
  pollInFlight = true;
  try {
    const response = await fetch("/api/state", {
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error(`状态请求失败（HTTP ${response.status}）`);
    }
    const state = asObject(await response.json());
    const nextStatus = normalizeRunStatus(asText(state.status));
    liveState = state;
    workbench.syncLiveState(state);
    consecutivePollFailures = 0;
    if (workbench.isReplaying()) {
      setConnection(true);
      setFormState(nextStatus);
    } else {
      renderState(state);
    }
  } catch (error) {
    consecutivePollFailures += 1;
    if (consecutivePollFailures >= 2) {
      setConnection(false);
    }
    if (consecutivePollFailures === 2) {
      showToast(error instanceof Error ? error.message : "无法连接本地智能体服务。");
    }
  } finally {
    pollInFlight = false;
  }
}

async function fetchMetadata() {
  try {
    const response = await fetch("/api/meta", {
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new Error(`运行信息请求失败（HTTP ${response.status}）`);
    }
    renderMetadata(await response.json());
  } catch (error) {
    setConnection(false);
    showToast(error instanceof Error ? error.message : "无法加载本地运行信息。");
  }
}

async function submitRun(event) {
  event.preventDefault();
  if (appMetadata.controlToken && !workbench.hasSelectedProject()) {
    elements.formMessage.textContent = "请先打开或新建一个项目。";
    return;
  }
  const task = elements.taskInput.value.trim();
  if (!task) {
    elements.formMessage.textContent = "请先描述任务，再启动智能体。";
    elements.taskInput.focus();
    return;
  }

  elements.formMessage.textContent = "";
  elements.runButton.disabled = true;
  elements.runButton.classList.add("is-running");
  elements.runButtonLabel.textContent = "正在启动…";
  if (workbench.isReplaying()) {
    workbench.returnToCurrent();
  }

  try {
    const response = await fetch("/api/runs", {
      body: JSON.stringify({ task }),
      headers: mutationHeaders(),
      method: "POST",
    });

    if (response.status !== 202) {
      let detail = `任务请求失败（HTTP ${response.status}）`;
      try {
        const body = asObject(await response.json());
        detail = translateServerDetail(body.detail) || detail;
      } catch {
        // Keep the status-based message when the server did not return JSON.
      }
      if (response.status === 409) {
        showToast(detail, "warning");
        await fetchState();
        return;
      }
      throw new Error(detail);
    }

    const acceptedState = await response.json();
    liveState = acceptedState;
    workbench.beginRun(acceptedState);
    lastRenderSignature = "";
    renderState(acceptedState);
    showToast("任务已接收，Relay 正在规划第一步。", "success");
    await fetchState();
  } catch (error) {
    showToast(error instanceof Error ? error.message : "无法启动任务。");
    const status = normalizeRunStatus(asText(currentState?.status));
    setFormState(status);
  }
}

elements.form.addEventListener("submit", submitRun);
elements.taskInput.addEventListener("input", () => {
  taskInputDirty = true;
  elements.formMessage.textContent = "";
});
elements.taskInput.addEventListener("keydown", (event) => {
  const modifier = event.metaKey || event.ctrlKey;
  if (modifier && event.key === "Enter") {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});

async function initializeApp() {
  await fetchMetadata();
  if (appMetadata.controlToken) {
    await workbench.initialize();
  }
  await fetchState();
}

void initializeApp();
window.setInterval(fetchState, POLL_INTERVAL_MS);
