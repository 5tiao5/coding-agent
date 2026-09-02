"use strict";

const TERMINAL_STATUSES = new Set(["completed", "completed_unverified", "failed"]);
const RUN_LABELS = {
  completed: "已完成",
  completed_unverified: "未验证",
  failed: "失败",
  idle: "就绪",
  interrupted: "已中断",
  pending: "等待轨迹",
  running: "运行中",
  unavailable: "轨迹不可用",
};

function asObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function asArray(value) {
  return Array.isArray(value) ? value : [];
}

function asText(value, fallback = "") {
  return typeof value === "string" ? value : fallback;
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

function boundedText(value, limit) {
  return asText(value).trim().slice(0, limit);
}

export function resumeControlView(rawRun) {
  const run = asObject(rawRun);
  const known = typeof run.resume_available === "boolean";
  return {
    available: known && run.resume_available === true,
    known,
    reason: boundedText(run.resume_reason, 240),
  };
}

export function continuationControlView(rawRun) {
  const run = asObject(rawRun);
  const continuation = asObject(run.continuation);
  const kind = boundedText(continuation.kind, 20);
  if (["resume", "follow_up", "none"].includes(kind)) {
    return {
      available: continuation.available === true,
      kind,
      known: typeof continuation.available === "boolean",
      reason: boundedText(continuation.reason, 240),
    };
  }
  const legacy = resumeControlView(run);
  return {
    available: legacy.available,
    kind: legacy.available ? "resume" : "none",
    known: legacy.known,
    reason: legacy.reason,
  };
}

export function normalizeProjectMemoryContext(rawContext) {
  const context = asObject(rawContext);
  const known = Object.keys(context).length > 0;
  const sourceById = new Map();
  const rawSources = asArray(context.sources);
  for (const rawSource of rawSources.slice(0, 8)) {
    const source = asObject(rawSource);
    const runId = boundedText(source.run_id || source.source_run_id, 64);
    if (!runId || sourceById.has(runId)) {
      continue;
    }
    sourceById.set(runId, {
      completedAt: boundedText(source.completed_at || source.created_at, 80),
      runId,
      task: boundedText(source.task || source.task_goal || source.title, 180),
    });
  }
  for (const rawRunId of asArray(context.source_run_ids).slice(0, 8)) {
    const runId = boundedText(rawRunId, 64);
    if (runId && !sourceById.has(runId)) {
      sourceById.set(runId, { completedAt: "", runId, task: "" });
    }
  }
  const sources = [...sourceById.values()].slice(0, 8);
  const requested =
    context.requested === true ||
    context.enabled === true ||
    (known && context.requested !== false && context.enabled !== false);
  const applied = context.applied === true || context.used === true || sources.length > 0;
  const contextChars = Number.isInteger(context.context_chars)
    ? Math.max(0, context.context_chars)
    : null;
  const error = boundedText(context.error, 240);
  return { applied, contextChars, error, known, requested, sources };
}

export function projectMemoryContextView(rawContext) {
  const context = normalizeProjectMemoryContext(rawContext);
  const parentOnly =
    context.known &&
    context.requested === false &&
    context.applied === true &&
    context.sources.length > 0;
  return {
    ...context,
    parentOnly,
    visible:
      context.known &&
      (context.requested || context.applied || context.sources.length > 0 || Boolean(context.error)),
  };
}

function normalizeStatus(value) {
  return Object.hasOwn(RUN_LABELS, value) ? value : "idle";
}

function itemProjectId(rawProject) {
  const project = asObject(rawProject);
  return asText(project.id).trim() || asText(project.project_id).trim();
}

function itemProjectRoot(rawProject) {
  const project = asObject(rawProject);
  return asText(project.root).trim() || asText(project.path).trim();
}

function basename(path) {
  const trimmed = asText(path).trim().replace(/[\\/]+$/, "");
  return trimmed.split(/[\\/]/).at(-1) || trimmed;
}

function itemProjectName(rawProject) {
  const project = asObject(rawProject);
  return (
    asText(project.display_name).trim() ||
    asText(project.name).trim() ||
    basename(itemProjectRoot(project)) ||
    "未命名项目"
  );
}

function itemRunId(rawRun) {
  const run = asObject(rawRun);
  return asText(run.run_id).trim() || asText(run.id).trim();
}

function shortRunId(runId) {
  const text = asText(runId).trim();
  return text.length > 12 ? text.slice(0, 8) : text;
}

function formatHistoryTime(value) {
  const source = asText(value).trim();
  if (!source) {
    return "时间未知";
  }
  const date = new Date(source);
  if (Number.isNaN(date.getTime())) {
    return source;
  }
  return new Intl.DateTimeFormat("zh-CN", {
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    month: "2-digit",
  }).format(date);
}

function isAbsolutePath(path) {
  return /^(?:[A-Za-z]:[\\/]|\\\\[^\\]|\/)/.test(path);
}

function joinProjectRoot(parent, name) {
  const base = parent.replace(/[\\/]+$/, "");
  const separator = /[\\]|^[A-Za-z]:/.test(base) ? "\\" : "/";
  return `${base}${separator}${name}`;
}

export function createWorkbench({
  elements,
  formatResumeReason,
  formatServerDetail,
  getControlToken,
  getNativeFolderPickerAvailable,
  onBeforeProjectChanged,
  onBusyStateChanged,
  onFollowUpCleared,
  onFollowUpPrepared,
  onHistoryState,
  onProjectChanged,
  onReturnLive,
  onRunAccepted,
  showToast,
}) {
  let projects = [];
  let projectRuns = [];
  let selectedProjectId = "";
  let projectRunActive = false;
  let viewingHistoryRunId = "";
  let pendingFollowUp = null;
  let dialogMode = "open";
  let dialogReturnFocus = null;
  let dialogSubmissionActive = false;
  let liveState = null;
  let nativeFolderPickerDisabled = false;
  let nativePickerActive = false;
  let pendingProjectRemoval = null;
  let projectRemovalReturnFocus = null;
  let restoreProjectRemovalTriggerFocus = true;
  let projectRemovalSubmitting = false;

  function selectedProject() {
    return projects.find((project) => itemProjectId(project) === selectedProjectId) || null;
  }

  function projectSummary() {
    const project = selectedProject();
    return project
      ? { id: selectedProjectId, name: itemProjectName(project), root: itemProjectRoot(project) }
      : null;
  }

  function mutationHeaders() {
    const headers = {
      Accept: "application/json",
      "Content-Type": "application/json",
    };
    const token = asText(getControlToken()).trim();
    if (token) {
      headers["X-Coding-Agent-Token"] = token;
    }
    return headers;
  }

  function nativeFolderPickerAvailable() {
    return (
      !nativeFolderPickerDisabled &&
      Boolean(asText(getControlToken()).trim()) &&
      getNativeFolderPickerAvailable?.() === true
    );
  }

  function projectChangeBlocked() {
    return projectRunActive || normalizeStatus(asText(liveState?.status)) === "running";
  }

  function pickerError(message, status = null) {
    const error = new Error(message);
    error.httpStatus = status;
    return error;
  }

  function pickerErrorStatus(error) {
    return error instanceof Error && Number.isInteger(error.httpStatus)
      ? error.httpStatus
      : null;
  }

  async function responseDetail(response, fallback) {
    try {
      const body = asObject(await response.json());
      return formatServerDetail(body.detail) || asText(body.message).trim() || fallback;
    } catch {
      return fallback;
    }
  }

  async function postJson(url, payload, expectedStatuses = [200, 201, 204]) {
    const response = await fetch(url, {
      body: JSON.stringify(payload),
      headers: mutationHeaders(),
      method: "POST",
    });
    if (!expectedStatuses.includes(response.status)) {
      throw new Error(await responseDetail(response, `请求失败（HTTP ${response.status}）`));
    }
    return response.status === 204 ? {} : asObject(await response.json());
  }

  async function deleteJson(url) {
    const response = await fetch(url, {
      headers: mutationHeaders(),
      method: "DELETE",
    });
    if (!response.ok) {
      throw new Error(await responseDetail(response, `移除请求失败（HTTP ${response.status}）`));
    }
    return response.status === 204 ? {} : asObject(await response.json());
  }

  async function projectListingState(projectId) {
    try {
      const response = await fetch("/api/projects", {
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        return null;
      }
      const body = asObject(await response.json());
      return asArray(body.projects).some(
        (project) => itemProjectId(asObject(project)) === projectId,
      );
    } catch {
      return null;
    }
  }

  async function requestNativeFolder() {
    if (!nativeFolderPickerAvailable()) {
      throw pickerError("系统文件夹选择器当前不可用。");
    }
    const response = await fetch("/api/folders/pick", {
      body: JSON.stringify({}),
      headers: mutationHeaders(),
      method: "POST",
    });
    if (response.status === 204) {
      return null;
    }
    if (!response.ok) {
      throw pickerError(
        await responseDetail(response, `无法打开文件夹选择器（HTTP ${response.status}）`),
        response.status,
      );
    }
    const body = asObject(await response.json());
    if (asText(body.status).trim() === "cancelled") {
      return null;
    }
    const path = asText(body.path).trim();
    if (asText(body.status).trim() !== "selected" || !isAbsolutePath(path)) {
      throw pickerError("文件夹选择器返回了无效路径。");
    }
    return path;
  }

  function renderHistoryMode() {
    const replaying = Boolean(viewingHistoryRunId);
    elements.historyBadge.hidden = !replaying;
    elements.historyReturn.hidden = !replaying;
    elements.sessionHeading.textContent = replaying ? "历史回放" : "智能体会话";
  }

  function updateControls(status = normalizeStatus(asText(liveState?.status))) {
    const running = normalizeStatus(status) === "running" || projectRunActive;
    const dialogBusy = nativePickerActive || dialogSubmissionActive;
    const removalDialogOpen = elements.projectRemoveDialog.open;
    const navigationLocked =
      running || dialogBusy || projectRemovalSubmitting || removalDialogOpen;
    elements.openProjectButton.disabled = navigationLocked;
    elements.newProjectButton.disabled = navigationLocked;
    elements.projectRootBrowse.disabled = navigationLocked;
    elements.historyReturn.disabled = navigationLocked;
    elements.projectDialogClose.disabled = dialogBusy;
    elements.projectDialogCancel.disabled = dialogBusy;
    elements.projectDialogSubmit.disabled = running || dialogBusy;
    elements.projectDialog.setAttribute("aria-busy", String(dialogBusy));
    elements.projectDialogForm.setAttribute("aria-busy", String(dialogBusy));
    elements.projectRemoveCancel.disabled = projectRemovalSubmitting;
    elements.projectRemoveClose.disabled = projectRemovalSubmitting;
    elements.projectRemoveSubmit.disabled = projectRemovalSubmitting;
    elements.projectRemoveDialog.setAttribute(
      "aria-busy",
      String(projectRemovalSubmitting),
    );
    elements.projectRemoveForm.setAttribute(
      "aria-busy",
      String(projectRemovalSubmitting),
    );
    elements.projectBrowser.setAttribute(
      "aria-busy",
      String(dialogBusy || projectRemovalSubmitting),
    );
    elements.openProjectButton.setAttribute("aria-busy", String(dialogBusy));
    if (dialogBusy || projectRemovalSubmitting || removalDialogOpen) {
      elements.taskInput.disabled = true;
      elements.runButton.disabled = true;
      if (elements.useProjectMemory) {
        elements.useProjectMemory.disabled = true;
      }
    }
    for (const button of document.querySelectorAll(
      ".project-item, .project-remove, .history-item-main, .history-resume",
    )) {
      const permanentlyDisabled = button.dataset.continuationAvailable === "false";
      button.disabled = navigationLocked || permanentlyDisabled;
    }
  }

  function refreshBusyState() {
    updateControls();
    onBusyStateChanged?.(
      nativePickerActive ||
        dialogSubmissionActive ||
        projectRemovalSubmitting ||
        elements.projectRemoveDialog.open,
    );
  }

  function setNativePickerActive(active) {
    nativePickerActive = active;
    refreshBusyState();
  }

  function setDialogSubmissionActive(active) {
    dialogSubmissionActive = active;
    refreshBusyState();
  }

  function setProjectRemovalSubmitting(active) {
    projectRemovalSubmitting = active;
    refreshBusyState();
  }

  function focusAfterDialogClose() {
    const target = dialogReturnFocus;
    dialogReturnFocus = null;
    if (!(target instanceof HTMLElement)) {
      return;
    }
    window.requestAnimationFrame(() => {
      if (target.isConnected && !target.disabled && !target.hidden) {
        target.focus();
      }
    });
  }

  function focusAfterProjectRemovalClose() {
    const target = restoreProjectRemovalTriggerFocus ? projectRemovalReturnFocus : null;
    projectRemovalReturnFocus = null;
    restoreProjectRemovalTriggerFocus = true;
    pendingProjectRemoval = null;
    elements.projectRemoveMessage.textContent = "";
    refreshBusyState();
    if (!(target instanceof HTMLElement)) {
      return;
    }
    window.requestAnimationFrame(() => {
      if (target.isConnected && !target.disabled && !target.hidden) {
        target.focus();
      }
    });
  }

  function focusProjectNavigation() {
    window.requestAnimationFrame(() => {
      const selectedButton = Array.from(document.querySelectorAll(".project-item")).find(
        (button) => button.dataset.projectId === selectedProjectId,
      );
      const target = selectedButton || elements.openProjectButton;
      if (target instanceof HTMLElement && target.isConnected && !target.disabled) {
        target.focus();
      }
    });
  }

  function renderProjects() {
    if (!projects.length) {
      elements.projectList.replaceChildren(
        createElement("p", "navigation-placeholder", "还没有项目。打开已有目录，或新建一个空项目。"),
      );
    } else {
      const fragment = document.createDocumentFragment();
      for (const project of projects) {
        const id = itemProjectId(project);
        if (!id) {
          continue;
        }
        const row = createElement("div", "project-item-row");
        const button = createElement("button", "project-item");
        button.type = "button";
        button.dataset.projectId = id;
        button.classList.toggle("is-selected", id === selectedProjectId);
        if (id === selectedProjectId) {
          button.setAttribute("aria-current", "true");
        }
        button.disabled = projectRunActive;
        const glyph = createElement("span", "project-item-glyph");
        glyph.setAttribute("aria-hidden", "true");
        const copy = createElement("span", "project-item-copy");
        copy.append(
          createElement("strong", "", itemProjectName(project)),
          createElement("span", "", itemProjectRoot(project) || "本地目录"),
        );
        button.append(glyph, copy);
        button.addEventListener("click", () => selectProjectById(id));
        const removeButton = createElement("button", "project-remove", "移除");
        removeButton.type = "button";
        removeButton.disabled = projectRunActive;
        removeButton.title = `从 Relay 列表移除 ${itemProjectName(project)}`;
        removeButton.setAttribute("aria-label", removeButton.title);
        removeButton.addEventListener("click", () =>
          openProjectRemovalDialog(project, removeButton),
        );
        row.append(button, removeButton);
        fragment.append(row);
      }
      elements.projectList.replaceChildren(
        fragment.childNodes.length
          ? fragment
          : createElement("p", "navigation-placeholder", "没有可显示的项目。"),
      );
    }

    const summary = projectSummary();
    elements.workspaceName.textContent = summary?.name || "尚未选择项目";
    elements.runtimeLabel.textContent = summary?.root || "请选择或新建一个本地项目";
    elements.historyTitle.textContent = summary ? `${summary.name} · 历史` : "运行历史";
    updateControls();
  }

  function renderRunHistory() {
    if (!selectedProjectId) {
      elements.runHistoryList.replaceChildren(
        createElement("p", "navigation-placeholder", "选择项目后查看历史。"),
      );
      return;
    }
    if (!projectRuns.length) {
      elements.runHistoryList.replaceChildren(
        createElement("p", "navigation-placeholder", "这个项目还没有运行记录。"),
      );
      return;
    }

    const fragment = document.createDocumentFragment();
    for (const run of projectRuns) {
      const id = itemRunId(run);
      if (!id) {
        continue;
      }
      const status = normalizeStatus(asText(run.status));
      const item = createElement("div", "history-item");
      item.classList.toggle("is-selected", id === viewingHistoryRunId);
      const button = createElement("button", "history-item-main");
      button.type = "button";
      button.disabled = projectRunActive;
      button.setAttribute("aria-label", `回放任务：${asText(run.task).trim() || shortRunId(id)}`);
      const dot = createElement("span", `history-status-dot status-${status}`);
      dot.setAttribute("aria-hidden", "true");
      const copy = createElement("span", "history-item-copy");
      copy.append(
        createElement(
          "strong",
          "",
          asText(run.task).trim() || asText(run.title).trim() || `运行 #${shortRunId(id)}`,
        ),
        createElement(
          "span",
          "",
          `${RUN_LABELS[status]} · ${formatHistoryTime(run.created_at || run.started_at)}`,
        ),
      );
      button.append(dot, copy);
      button.addEventListener("click", () => showHistory(id));
      item.append(button);

      const continuation = continuationControlView(run);
      if (continuation.known) {
        const actionLabel = continuation.kind === "resume" ? "恢复" : "继续";
        const resumeButton = createElement("button", "history-resume", actionLabel);
        resumeButton.type = "button";
        resumeButton.dataset.continuationAvailable = String(continuation.available);
        resumeButton.disabled = projectRunActive || !continuation.available;
        resumeButton.setAttribute(
          "aria-label",
          continuation.available
            ? `${actionLabel}任务：${asText(run.task).trim() || shortRunId(id)}`
            : "该任务不可继续",
        );
        resumeButton.title = continuation.available
          ? continuation.kind === "resume"
            ? "从已保存的检查点恢复；已完成的工具不会重放"
            : "输入新的要求后创建一轮任务；父任务摘要会作为历史上下文"
          : formatResumeReason?.(continuation.reason) ||
            continuation.reason ||
            "该任务当前不能继续";
        resumeButton.addEventListener("click", () => {
          if (continuation.kind === "resume") {
            void resumeRun(id);
          } else if (continuation.kind === "follow_up") {
            prepareFollowUp(id);
          }
        });
        item.append(resumeButton);
      }
      fragment.append(item);
    }
    elements.runHistoryList.replaceChildren(
      fragment.childNodes.length
        ? fragment
        : createElement("p", "navigation-placeholder", "没有可回放的运行记录。"),
    );
  }

  async function fetchProjectRuns() {
    if (!selectedProjectId) {
      projectRuns = [];
      renderRunHistory();
      return;
    }
    try {
      const response = await fetch(`/api/projects/${encodeURIComponent(selectedProjectId)}/runs`, {
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        throw new Error(`历史请求失败（HTTP ${response.status}）`);
      }
      projectRuns = asArray(asObject(await response.json()).runs).map(asObject);
      renderRunHistory();
    } catch (error) {
      projectRuns = [];
      renderRunHistory();
      showToast(error instanceof Error ? error.message : "无法加载项目历史。", "warning");
    }
  }

  async function fetchProjects({ loadRuns = true } = {}) {
    try {
      const response = await fetch("/api/projects", {
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        throw new Error(`项目请求失败（HTTP ${response.status}）`);
      }
      const body = asObject(await response.json());
      projects = asArray(body.projects).map(asObject);
      const serverSelection =
        asText(body.active_project_id).trim() || asText(body.selected_project_id).trim();
      selectedProjectId =
        serverSelection ||
        (projects.some((project) => itemProjectId(project) === selectedProjectId)
          ? selectedProjectId
          : "");
      projectRunActive =
        body.run_active === true || normalizeStatus(asText(liveState?.status)) === "running";
      renderProjects();
      if (loadRuns) {
        await fetchProjectRuns();
      }
      return true;
    } catch (error) {
      projects = [];
      selectedProjectId = "";
      projectRunActive = false;
      renderProjects();
      renderRunHistory();
      showToast(error instanceof Error ? error.message : "无法加载本地项目。");
      return false;
    }
  }

  async function selectProjectById(id) {
    if (!id) {
      return;
    }
    if (projectRunActive || normalizeStatus(asText(liveState?.status)) === "running") {
      showToast("任务运行期间不能切换项目。", "warning");
      return;
    }
    if (id === selectedProjectId) {
      clearFollowUp();
      returnToCurrent();
      return;
    }

    try {
      await postJson(`/api/projects/${encodeURIComponent(id)}/select`, {});
      selectedProjectId = id;
      viewingHistoryRunId = "";
      clearFollowUp();
      liveState = null;
      onBeforeProjectChanged();
      await fetchProjects();
      await onProjectChanged();
      showToast(`已切换到 ${projectSummary()?.name || "所选项目"}。`, "success");
    } catch (error) {
      showToast(error instanceof Error ? error.message : "无法切换项目。");
    }
  }

  function openProjectRemovalDialog(project, returnFocus) {
    if (projectChangeBlocked()) {
      showToast("任务运行期间不能移除项目。", "warning");
      return;
    }
    const id = itemProjectId(project);
    if (!id || elements.projectRemoveDialog.open) {
      return;
    }
    pendingProjectRemoval = {
      id,
      name: itemProjectName(project),
      root: itemProjectRoot(project),
    };
    projectRemovalReturnFocus = returnFocus;
    restoreProjectRemovalTriggerFocus = true;
    elements.projectRemoveName.textContent = pendingProjectRemoval.name;
    elements.projectRemoveRoot.textContent = pendingProjectRemoval.root || "本地目录";
    elements.projectRemoveMessage.textContent = "";
    elements.projectRemoveDialog.showModal();
    refreshBusyState();
    window.requestAnimationFrame(() => elements.projectRemoveCancel.focus());
  }

  async function submitProjectRemoval(event) {
    event.preventDefault();
    if (projectRemovalSubmitting || pendingProjectRemoval === null) {
      return;
    }
    const target = { ...pendingProjectRemoval };
    const removingSelectedProject = target.id === selectedProjectId;
    let removalCommitted = false;
    elements.projectRemoveMessage.textContent = "";
    setProjectRemovalSubmitting(true);
    try {
      let result;
      try {
        result = await deleteJson(`/api/projects/${encodeURIComponent(target.id)}`);
      } catch (error) {
        const stillListed = await projectListingState(target.id);
        if (stillListed !== false) {
          throw error;
        }
        result = { workspace_deleted: false };
      }
      if (result.workspace_deleted !== false) {
        throw new Error("服务端没有确认本地目录保持不变，界面拒绝静默完成。");
      }
      removalCommitted = true;
      restoreProjectRemovalTriggerFocus = false;
      elements.projectRemoveDialog.close();
      if (removingSelectedProject) {
        selectedProjectId = "";
        projectRuns = [];
        viewingHistoryRunId = "";
        clearFollowUp();
        liveState = null;
        renderHistoryMode();
        onBeforeProjectChanged();
      }
      const refreshed = await fetchProjects();
      if (removingSelectedProject) {
        await onProjectChanged();
      }
      showToast(
        refreshed
          ? `已从列表移除 ${target.name}；本地目录未删除。`
          : `已移除 ${target.name}，但项目列表刷新失败；本地目录未删除。`,
        refreshed ? "success" : "warning",
      );
    } catch (error) {
      if (removalCommitted) {
        showToast(
          `已移除 ${target.name}，但部分界面未刷新；请重新加载页面。本地目录未删除。`,
          "warning",
        );
      } else {
        const message = error instanceof Error ? error.message : "无法移除这个项目。";
        elements.projectRemoveMessage.textContent = message;
        showToast(message, "warning");
      }
    } finally {
      setProjectRemovalSubmitting(false);
      if (removalCommitted) {
        focusProjectNavigation();
      }
    }
  }

  async function showHistory(id) {
    if (!id || projectRunActive) {
      return;
    }
    try {
      clearFollowUp();
      const response = await fetch(`/api/history/${encodeURIComponent(id)}`, {
        cache: "no-store",
        headers: { Accept: "application/json" },
      });
      if (!response.ok) {
        throw new Error(`回放请求失败（HTTP ${response.status}）`);
      }
      const body = asObject(await response.json());
      const run = asObject(body.run);
      const historyState = Object.keys(run).length
        ? {
            ...run,
            memory_context: body.memory_context || run.memory_context,
            project_memory: body.project_memory || run.project_memory,
            run_id: itemRunId(run) || id,
            snapshot: asObject(body.snapshot),
          }
        : body;
      viewingHistoryRunId = id;
      renderHistoryMode();
      renderRunHistory();
      onHistoryState(historyState);
    } catch (error) {
      showToast(error instanceof Error ? error.message : "无法回放这次运行。");
    }
  }

  async function resumeRun(id) {
    if (!id || projectRunActive || normalizeStatus(asText(liveState?.status)) === "running") {
      showToast("已有任务正在运行，暂时不能继续历史任务。", "warning");
      return;
    }
    try {
      const acceptedState = await postJson(
        `/api/runs/${encodeURIComponent(id)}/resume`,
        {},
        [202],
      );
      viewingHistoryRunId = "";
      liveState = acceptedState;
      projectRunActive = true;
      renderHistoryMode();
      renderRunHistory();
      updateControls("running");
      onRunAccepted?.(acceptedState);
      showToast("已从检查点继续；Relay 会重新验证当前工作区。", "success");
    } catch (error) {
      showToast(error instanceof Error ? error.message : "无法继续这次任务。");
      await fetchProjectRuns();
    }
  }

  function prepareFollowUp(id) {
    if (!id || projectRunActive || normalizeStatus(asText(liveState?.status)) === "running") {
      showToast("已有任务正在运行，暂时不能继续历史任务。", "warning");
      return;
    }
    const run = projectRuns.find((candidate) => itemRunId(candidate) === id);
    const continuation = continuationControlView(run);
    if (!run || continuation.kind !== "follow_up" || !continuation.available) {
      showToast(continuation.reason || "该历史任务当前不能继续。", "warning");
      return;
    }
    pendingFollowUp = {
      runId: id,
      task: boundedText(run.task || run.title, 240),
    };
    viewingHistoryRunId = "";
    renderHistoryMode();
    renderRunHistory();
    onReturnLive(liveState || {});
    onFollowUpPrepared?.({ ...pendingFollowUp });
    updateControls();
  }

  function clearFollowUp() {
    if (pendingFollowUp === null) {
      return;
    }
    pendingFollowUp = null;
    onFollowUpCleared?.();
  }

  function returnToCurrent() {
    if (!viewingHistoryRunId) {
      return;
    }
    viewingHistoryRunId = "";
    renderHistoryMode();
    renderRunHistory();
    onReturnLive(liveState || {});
  }

  async function registerProject(targetRoot, { creating = false, displayName = "" } = {}) {
    const payload = { create: creating, root: targetRoot };
    if (displayName) {
      payload.display_name = displayName;
    }
    const previousProjectId = selectedProjectId;
    const result = await postJson("/api/projects", payload);
    await fetchProjects();
    const returnedProject = asObject(result.project);
    const returnedId = itemProjectId(returnedProject) || itemProjectId(result);
    const registered =
      projects.find((project) => itemProjectId(project) === returnedId) ||
      projects.find(
        (project) => itemProjectRoot(project).toLowerCase() === targetRoot.toLowerCase(),
      );
    if (registered && itemProjectId(registered) !== selectedProjectId) {
      await selectProjectById(itemProjectId(registered));
    } else {
      if (selectedProjectId !== previousProjectId) {
        clearFollowUp();
        liveState = null;
        onBeforeProjectChanged();
      }
      await onProjectChanged();
    }
    showToast(creating ? "新项目已创建并打开。" : "项目已登记并打开。", "success");
  }

  function openProjectDialog(
    mode,
    { message = "", returnFocus = document.activeElement, root = "" } = {},
  ) {
    if (projectChangeBlocked()) {
      showToast("任务运行期间不能更改项目。", "warning");
      return;
    }
    if (elements.projectDialog.open) {
      return;
    }
    dialogMode = mode;
    dialogReturnFocus = returnFocus;
    const creating = mode === "create";
    elements.projectDialogTitle.textContent = creating ? "新建项目" : "打开项目";
    elements.projectDialogCopy.textContent = creating
      ? "浏览或输入父目录，再填写项目名。Relay 会创建一个空目录并登记为本地项目。"
      : "手动输入已有项目的绝对路径。Relay 只会在选中的项目中工作。";
    elements.projectRootLabel.textContent = creating ? "父目录绝对路径" : "项目绝对路径";
    elements.projectRootInput.placeholder = creating ? "D:\\code" : "D:\\code\\my-project";
    elements.projectRootBrowse.hidden = !creating || !nativeFolderPickerAvailable();
    elements.projectNameField.hidden = !creating;
    elements.projectNameInput.required = creating;
    elements.projectDisplayNameField.hidden = creating;
    elements.projectDialogSubmit.textContent = creating ? "创建并打开" : "打开项目";
    elements.projectDialogForm.reset();
    elements.projectRootInput.value = root;
    elements.projectDialogMessage.textContent = message;
    elements.projectDialog.showModal();
    window.requestAnimationFrame(() => elements.projectRootInput.focus());
  }

  async function openExistingProject() {
    if (projectChangeBlocked()) {
      showToast("任务运行期间不能更改项目。", "warning");
      return;
    }
    if (!nativeFolderPickerAvailable()) {
      openProjectDialog("open", { returnFocus: elements.openProjectButton });
      return;
    }

    let selectedRoot = null;
    setNativePickerActive(true);
    elements.openProjectButton.textContent = "正在选择…";
    try {
      selectedRoot = await requestNativeFolder();
    } catch (error) {
      if (pickerErrorStatus(error) !== 409) {
        nativeFolderPickerDisabled = true;
      }
      const detail = error instanceof Error ? error.message : "无法打开系统文件夹选择器。";
      setNativePickerActive(false);
      elements.openProjectButton.textContent = "打开项目";
      showToast(`${detail} 请改用手动路径。`, "warning");
      openProjectDialog("open", {
        message: "系统文件夹选择器未完成，请检查或手动输入绝对路径。",
        returnFocus: elements.openProjectButton,
      });
      return;
    }

    if (selectedRoot === null) {
      setNativePickerActive(false);
      elements.openProjectButton.textContent = "打开项目";
      window.requestAnimationFrame(() => elements.openProjectButton.focus());
      return;
    }
    if (projectChangeBlocked()) {
      setNativePickerActive(false);
      elements.openProjectButton.textContent = "打开项目";
      showToast("任务已经开始，暂不能打开另一个项目。", "warning");
      return;
    }

    try {
      await registerProject(selectedRoot);
    } catch (error) {
      showToast(error instanceof Error ? error.message : "无法登记这个项目。");
    } finally {
      setNativePickerActive(false);
      elements.openProjectButton.textContent = "打开项目";
    }
  }

  async function browseProjectParent() {
    if (!elements.projectDialog.open) {
      return;
    }
    if (projectChangeBlocked()) {
      showToast("任务运行期间不能更改项目。", "warning");
      return;
    }
    let selectedRoot = null;
    setNativePickerActive(true);
    elements.projectRootBrowse.textContent = "选择中…";
    elements.projectDialogMessage.textContent = "";
    try {
      selectedRoot = await requestNativeFolder();
      if (selectedRoot !== null && elements.projectDialog.open) {
        elements.projectRootInput.value = selectedRoot;
      }
    } catch (error) {
      if (pickerErrorStatus(error) !== 409) {
        nativeFolderPickerDisabled = true;
      }
      if (elements.projectDialog.open) {
        elements.projectRootBrowse.hidden = !nativeFolderPickerAvailable();
        elements.projectDialogMessage.textContent =
          error instanceof Error
            ? `${error.message} 你仍可手动输入父目录。`
            : "请手动输入父目录。";
      }
    } finally {
      setNativePickerActive(false);
      elements.projectRootBrowse.textContent = "浏览…";
      if (elements.projectDialog.open) {
        const focusTarget = selectedRoot === null
          ? elements.projectRootBrowse.hidden
            ? elements.projectRootInput
            : elements.projectRootBrowse
          : elements.projectNameInput;
        window.requestAnimationFrame(() => focusTarget.focus());
      }
    }
  }

  async function submitProjectDialog(event) {
    event.preventDefault();
    if (nativePickerActive || dialogSubmissionActive || projectChangeBlocked()) {
      return;
    }
    const root = elements.projectRootInput.value.trim();
    const creating = dialogMode === "create";
    const name = elements.projectNameInput.value.trim();
    const displayName = elements.projectDisplayNameInput.value.trim();

    if (!isAbsolutePath(root)) {
      elements.projectDialogMessage.textContent = "请输入绝对路径，例如 D:\\code\\project。";
      elements.projectRootInput.focus();
      return;
    }
    if (creating && (!name || /[<>:"/\\|?*]/.test(name) || name === "." || name === "..")) {
      elements.projectDialogMessage.textContent = "请输入不含路径分隔符或系统保留字符的项目名。";
      elements.projectNameInput.focus();
      return;
    }

    const targetRoot = creating ? joinProjectRoot(root, name) : root;
    elements.projectDialogMessage.textContent = "";
    setDialogSubmissionActive(true);
    try {
      const desiredName = creating ? name : displayName;
      await registerProject(targetRoot, { creating, displayName: desiredName });
      elements.projectDialog.close();
    } catch (error) {
      if (elements.projectDialog.open) {
        elements.projectDialogMessage.textContent =
          error instanceof Error ? error.message : "无法登记这个项目。";
      }
    } finally {
      setDialogSubmissionActive(false);
    }
  }

  elements.openProjectButton.addEventListener("click", openExistingProject);
  elements.newProjectButton.addEventListener("click", () =>
    openProjectDialog("create", { returnFocus: elements.newProjectButton }),
  );
  elements.projectRootBrowse.addEventListener("click", browseProjectParent);
  elements.historyReturn.addEventListener("click", returnToCurrent);
  elements.projectDialogForm.addEventListener("submit", submitProjectDialog);
  elements.projectRemoveForm.addEventListener("submit", submitProjectRemoval);
  elements.projectDialogCancel.addEventListener("click", () => {
    if (!nativePickerActive && !dialogSubmissionActive) {
      elements.projectDialog.close();
    }
  });
  elements.projectDialogClose.addEventListener("click", () => {
    if (!nativePickerActive && !dialogSubmissionActive) {
      elements.projectDialog.close();
    }
  });
  elements.projectDialog.addEventListener("cancel", (event) => {
    if (nativePickerActive || dialogSubmissionActive) {
      event.preventDefault();
    }
  });
  elements.projectDialog.addEventListener("close", focusAfterDialogClose);
  elements.projectRemoveCancel.addEventListener("click", () => {
    if (!projectRemovalSubmitting) {
      elements.projectRemoveDialog.close();
    }
  });
  elements.projectRemoveClose.addEventListener("click", () => {
    if (!projectRemovalSubmitting) {
      elements.projectRemoveDialog.close();
    }
  });
  elements.projectRemoveDialog.addEventListener("cancel", (event) => {
    if (projectRemovalSubmitting) {
      event.preventDefault();
    }
  });
  elements.projectRemoveDialog.addEventListener("close", focusAfterProjectRemovalClose);

  renderHistoryMode();

  return {
    beginRun(state) {
      liveState = asObject(state);
      projectRunActive = true;
      viewingHistoryRunId = "";
      clearFollowUp();
      renderHistoryMode();
      updateControls("running");
    },
    hasSelectedProject() {
      return Boolean(selectedProjectId);
    },
    initialize: fetchProjects,
    isReplaying() {
      return Boolean(viewingHistoryRunId);
    },
    cancelFollowUp: clearFollowUp,
    followUpContext() {
      return pendingFollowUp === null ? null : { ...pendingFollowUp };
    },
    projectSummary,
    showHistory,
    returnToCurrent,
    syncLiveState(state) {
      const previousStatus = normalizeStatus(asText(liveState?.status));
      liveState = asObject(state);
      const nextStatus = normalizeStatus(asText(liveState.status));
      projectRunActive = nextStatus === "running";
      updateControls(nextStatus);
      if (previousStatus === "running" && TERMINAL_STATUSES.has(nextStatus)) {
        void fetchProjects();
      }
    },
    updateControls,
  };
}
