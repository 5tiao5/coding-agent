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
  formatServerDetail,
  getControlToken,
  getNativeFolderPickerAvailable,
  onBeforeProjectChanged,
  onBusyStateChanged,
  onHistoryState,
  onProjectChanged,
  onReturnLive,
  showToast,
}) {
  let projects = [];
  let projectRuns = [];
  let selectedProjectId = "";
  let projectRunActive = false;
  let viewingHistoryRunId = "";
  let dialogMode = "open";
  let dialogReturnFocus = null;
  let dialogSubmissionActive = false;
  let liveState = null;
  let nativeFolderPickerDisabled = false;
  let nativePickerActive = false;

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
    const navigationLocked = running || dialogBusy;
    elements.openProjectButton.disabled = navigationLocked;
    elements.newProjectButton.disabled = navigationLocked;
    elements.projectRootBrowse.disabled = navigationLocked;
    elements.historyReturn.disabled = navigationLocked;
    elements.projectDialogClose.disabled = dialogBusy;
    elements.projectDialogCancel.disabled = dialogBusy;
    elements.projectDialogSubmit.disabled = running || dialogBusy;
    elements.projectDialog.setAttribute("aria-busy", String(dialogBusy));
    elements.projectDialogForm.setAttribute("aria-busy", String(dialogBusy));
    elements.projectBrowser.setAttribute("aria-busy", String(dialogBusy));
    elements.openProjectButton.setAttribute("aria-busy", String(dialogBusy));
    if (dialogBusy) {
      elements.taskInput.disabled = true;
      elements.runButton.disabled = true;
    }
    for (const button of document.querySelectorAll(".project-item, .history-item")) {
      button.disabled = navigationLocked;
    }
  }

  function refreshBusyState() {
    updateControls();
    onBusyStateChanged?.(nativePickerActive || dialogSubmissionActive);
  }

  function setNativePickerActive(active) {
    nativePickerActive = active;
    refreshBusyState();
  }

  function setDialogSubmissionActive(active) {
    dialogSubmissionActive = active;
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
        const button = createElement("button", "project-item");
        button.type = "button";
        button.classList.toggle("is-selected", id === selectedProjectId);
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
        fragment.append(button);
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
      const button = createElement("button", "history-item");
      button.type = "button";
      button.classList.toggle("is-selected", id === viewingHistoryRunId);
      button.disabled = projectRunActive;
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
      fragment.append(button);
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
    } catch (error) {
      projects = [];
      selectedProjectId = "";
      projectRunActive = false;
      renderProjects();
      renderRunHistory();
      showToast(error instanceof Error ? error.message : "无法加载本地项目。");
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
      returnToCurrent();
      return;
    }

    try {
      await postJson(`/api/projects/${encodeURIComponent(id)}/select`, {});
      selectedProjectId = id;
      viewingHistoryRunId = "";
      liveState = null;
      onBeforeProjectChanged();
      await fetchProjects();
      await onProjectChanged();
      showToast(`已切换到 ${projectSummary()?.name || "所选项目"}。`, "success");
    } catch (error) {
      showToast(error instanceof Error ? error.message : "无法切换项目。");
    }
  }

  async function showHistory(id) {
    if (!id || projectRunActive) {
      return;
    }
    try {
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
        ? { ...run, run_id: itemRunId(run) || id, snapshot: asObject(body.snapshot) }
        : body;
      viewingHistoryRunId = id;
      renderHistoryMode();
      renderRunHistory();
      onHistoryState(historyState);
    } catch (error) {
      showToast(error instanceof Error ? error.message : "无法回放这次运行。");
    }
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

  renderHistoryMode();

  return {
    beginRun(state) {
      liveState = asObject(state);
      projectRunActive = true;
      viewingHistoryRunId = "";
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
    projectSummary,
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
