import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");

async function importWorkbenchModule() {
  const source = readFileSync(
    path.join(repositoryRoot, "src/coding_agent/web/static/_workbench.js"),
    "utf8",
  );
  return import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
}

class FakeHTMLElement {}

class FakeClassList {
  constructor(element) {
    this.element = element;
  }

  contains(name) {
    return this.element._classes.has(name);
  }

  toggle(name, force) {
    const enabled = force === undefined ? !this.contains(name) : Boolean(force);
    if (enabled) {
      this.element._classes.add(name);
    } else {
      this.element._classes.delete(name);
    }
    return enabled;
  }
}

class FakeElement extends FakeHTMLElement {
  constructor(document, tagName = "div") {
    super();
    this.ownerDocument = document;
    this.tagName = tagName.toUpperCase();
    this.children = [];
    this.listeners = new Map();
    this.attributes = new Map();
    this.dataset = {};
    this._classes = new Set();
    this._text = "";
    this.checked = false;
    this.disabled = false;
    this.hidden = false;
    this.isConnected = true;
    this.open = false;
    this.parentNode = null;
    this.title = "";
    this.type = "";
    this.value = "";
    this.classList = new FakeClassList(this);
    document.elements.add(this);
  }

  get childElementCount() {
    return this.children.length;
  }

  get childNodes() {
    return this.children;
  }

  get className() {
    return [...this._classes].join(" ");
  }

  set className(value) {
    this._classes = new Set(String(value).split(/\s+/).filter(Boolean));
  }

  get textContent() {
    if (this.children.length) {
      return `${this._text}${this.children.map((child) => child.textContent).join("")}`;
    }
    return this._text;
  }

  set textContent(value) {
    this._disconnectChildren();
    this.children = [];
    this._text = String(value);
  }

  addEventListener(type, listener) {
    const listeners = this.listeners.get(type) || [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  append(...nodes) {
    for (const node of nodes) {
      this._appendNode(node);
    }
  }

  close() {
    this.open = false;
    this._dispatchSync("close");
  }

  async dispatch(type, values = {}) {
    const event = {
      defaultPrevented: false,
      preventDefault() {
        this.defaultPrevented = true;
      },
      ...values,
    };
    for (const listener of this.listeners.get(type) || []) {
      await listener(event);
    }
    return event;
  }

  focus() {
    this.ownerDocument.activeElement = this;
  }

  replaceChildren(...nodes) {
    this._disconnectChildren();
    this.children = [];
    this._text = "";
    this.append(...nodes);
  }

  setAttribute(name, value) {
    this.attributes.set(name, String(value));
  }

  showModal() {
    this.open = true;
  }

  _appendNode(node) {
    if (!(node instanceof FakeElement)) {
      return;
    }
    if (node.tagName === "#FRAGMENT") {
      for (const child of [...node.children]) {
        this._appendNode(child);
      }
      node.children = [];
      return;
    }
    node.parentNode = this;
    markConnected(node, this.isConnected);
    this.children.push(node);
  }

  _disconnectChildren() {
    for (const child of this.children) {
      child.parentNode = null;
      markConnected(child, false);
    }
  }

  _dispatchSync(type) {
    const event = {
      defaultPrevented: false,
      preventDefault() {
        this.defaultPrevented = true;
      },
    };
    for (const listener of this.listeners.get(type) || []) {
      listener(event);
    }
  }
}

class FakeDocument {
  constructor() {
    this.activeElement = null;
    this.elements = new Set();
  }

  createDocumentFragment() {
    const fragment = new FakeElement(this, "#fragment");
    fragment.isConnected = false;
    return fragment;
  }

  createElement(tagName) {
    return new FakeElement(this, tagName);
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }

  querySelectorAll(selectors) {
    const options = selectors.split(",").map((selector) => selector.trim());
    return [...this.elements].filter(
      (element) => element.isConnected && options.some((selector) => matches(element, selector)),
    );
  }
}

function markConnected(element, connected) {
  element.isConnected = connected;
  for (const child of element.children) {
    markConnected(child, connected);
  }
}

function matches(element, selector) {
  if (!selector.startsWith(".")) {
    return false;
  }
  const classes = selector.slice(1).split(".");
  return classes.every((name) => element.classList.contains(name));
}

function jsonResponse(status, body) {
  return {
    ok: status >= 200 && status < 300,
    status,
    async json() {
      return structuredClone(body);
    },
  };
}

function createElements(document) {
  const elements = new Proxy(
    {},
    {
      get(target, name) {
        if (typeof name !== "string") {
          return target[name];
        }
        if (!Object.hasOwn(target, name)) {
          target[name] = document.createElement(name.toLowerCase().includes("dialog") ? "dialog" : "div");
        }
        return target[name];
      },
    },
  );
  elements.projectRemoveDialog.open = false;
  elements.projectDialog.open = false;
  elements.useProjectMemory.checked = true;
  return elements;
}

function project(id, name) {
  return {
    created_at: "2026-09-02T08:00:00Z",
    display_name: name,
    last_opened_at: "2026-09-02T08:00:00Z",
    project_id: id,
    root: `D:\\projects\\${name}`,
  };
}

function findProjectControl(document, className, name) {
  const found = document
    .querySelectorAll(`.${className}`)
    .find((element) => element.title.includes(name) || element.textContent.includes(name));
  assert.ok(found, `expected .${className} control for ${name}`);
  return found;
}

async function withHarness(callback, { deleteResponses = [] } = {}) {
  const originalDocument = globalThis.document;
  const originalFetch = globalThis.fetch;
  const originalHTMLElement = globalThis.HTMLElement;
  const originalWindow = globalThis.window;
  const document = new FakeDocument();
  const elements = createElements(document);
  const requests = [];
  const toasts = [];
  const callbacks = { beforeProjectChanged: 0, projectChanged: 0 };
  let projects = [project("p-alpha", "Alpha"), project("p-beta", "Beta")];
  let activeProjectId = "p-alpha";

  globalThis.document = document;
  globalThis.HTMLElement = FakeHTMLElement;
  globalThis.window = {
    requestAnimationFrame(callback) {
      callback();
      return 1;
    },
  };
  globalThis.fetch = async (url, options = {}) => {
    const method = options.method || "GET";
    requests.push({ headers: options.headers || {}, method, url });
    if (url === "/api/projects" && method === "GET") {
      return jsonResponse(200, { active_project_id: activeProjectId, projects });
    }
    const runsMatch = /^\/api\/projects\/([^/]+)\/runs$/.exec(url);
    if (runsMatch && method === "GET") {
      return jsonResponse(200, { project_id: decodeURIComponent(runsMatch[1]), runs: [] });
    }
    const deleteMatch = /^\/api\/projects\/([^/]+)$/.exec(url);
    if (deleteMatch && method === "DELETE") {
      const configured = deleteResponses.shift();
      if (configured && configured.status >= 400) {
        return jsonResponse(configured.status, configured.body);
      }
      const projectId = decodeURIComponent(deleteMatch[1]);
      projects = projects.filter((candidate) => candidate.project_id !== projectId);
      if (activeProjectId === projectId) {
        activeProjectId = null;
      }
      return jsonResponse(200, {
        history_preserved: true,
        project_id: projectId,
        removed_from_sidebar: true,
        workspace_deleted: false,
      });
    }
    throw new Error(`unexpected request: ${method} ${url}`);
  };

  try {
    const { createWorkbench } = await importWorkbenchModule();
    let workbench;
    const restoreComposerState = () => {
      const hasProject = workbench.hasSelectedProject();
      elements.taskInput.disabled = !hasProject;
      elements.runButton.disabled = !hasProject;
      elements.useProjectMemory.disabled = !hasProject;
      workbench.updateControls("idle");
    };
    workbench = createWorkbench({
      elements,
      formatResumeReason: (value) => String(value || ""),
      formatServerDetail: (value) => (typeof value === "string" ? value : ""),
      getControlToken: () => "control-token",
      getNativeFolderPickerAvailable: () => false,
      onBeforeProjectChanged() {
        callbacks.beforeProjectChanged += 1;
      },
      onBusyStateChanged: restoreComposerState,
      onFollowUpCleared() {},
      onFollowUpPrepared() {},
      onHistoryState() {},
      async onProjectChanged() {
        callbacks.projectChanged += 1;
        restoreComposerState();
      },
      onReturnLive() {},
      onRunAccepted() {},
      showToast(message, tone = "error") {
        toasts.push({ message, tone });
      },
    });
    await workbench.initialize();
    restoreComposerState();
    await callback({ callbacks, document, elements, requests, toasts, workbench });
  } finally {
    globalThis.document = originalDocument;
    globalThis.fetch = originalFetch;
    globalThis.HTMLElement = originalHTMLElement;
    globalThis.window = originalWindow;
  }
}

test("project removal controls preserve form state, focus, and retry semantics", async (t) => {
  await t.test("cancelling restores the composer and returns focus", async () => {
    await withHarness(async ({ document, elements }) => {
      const removeBeta = findProjectControl(document, "project-remove", "Beta");
      await removeBeta.dispatch("click");

      assert.equal(elements.projectRemoveDialog.open, true);
      assert.equal(elements.taskInput.disabled, true);
      assert.equal(document.activeElement, elements.projectRemoveCancel);

      await elements.projectRemoveCancel.dispatch("click");

      assert.equal(elements.projectRemoveDialog.open, false);
      assert.equal(elements.taskInput.disabled, false);
      assert.equal(elements.runButton.disabled, false);
      assert.equal(elements.useProjectMemory.disabled, false);
      assert.equal(document.activeElement, removeBeta);
    });
  });

  await t.test("removing an inactive project restores the composer and focuses a live control", async () => {
    await withHarness(async ({ document, elements, requests, workbench }) => {
      await findProjectControl(document, "project-remove", "Beta").dispatch("click");
      await elements.projectRemoveForm.dispatch("submit");

      assert.equal(elements.projectRemoveDialog.open, false);
      assert.deepEqual(workbench.projectSummary(), {
        id: "p-alpha",
        name: "Alpha",
        root: "D:\\projects\\Alpha",
      });
      assert.equal(document.querySelectorAll(".project-remove").length, 1);
      assert.equal(elements.taskInput.disabled, false);
      assert.equal(elements.runButton.disabled, false);
      assert.equal(elements.useProjectMemory.disabled, false);
      assert.equal(document.activeElement?.isConnected, true);
      assert.equal(document.activeElement?.classList.contains("project-item"), true);
      assert.equal(document.activeElement?.classList.contains("is-selected"), true);

      const deletion = requests.find((request) => request.method === "DELETE");
      assert.deepEqual(deletion, {
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
          "X-Coding-Agent-Token": "control-token",
        },
        method: "DELETE",
        url: "/api/projects/p-beta",
      });
    });
  });

  await t.test("removing the active project clears selection and focuses Open Project", async () => {
    await withHarness(async ({ callbacks, document, elements, workbench }) => {
      await findProjectControl(document, "project-remove", "Alpha").dispatch("click");
      await elements.projectRemoveForm.dispatch("submit");

      assert.equal(workbench.hasSelectedProject(), false);
      assert.equal(workbench.projectSummary(), null);
      assert.equal(callbacks.beforeProjectChanged, 1);
      assert.equal(callbacks.projectChanged, 1);
      assert.equal(elements.taskInput.disabled, true);
      assert.equal(elements.runButton.disabled, true);
      assert.match(elements.runHistoryList.textContent, /选择项目后查看历史/);
      assert.equal(document.activeElement, elements.openProjectButton);
      assert.equal(document.activeElement.isConnected, true);
      assert.equal(document.activeElement.disabled, false);
    });
  });

  await t.test("a failed request keeps the dialog retryable and a retry converges", async () => {
    await withHarness(
      async ({ document, elements, requests }) => {
        await findProjectControl(document, "project-remove", "Beta").dispatch("click");
        await elements.projectRemoveForm.dispatch("submit");

        assert.equal(elements.projectRemoveDialog.open, true);
        assert.match(elements.projectRemoveMessage.textContent, /暂时不能移除/);
        assert.equal(elements.projectRemoveSubmit.disabled, false);
        assert.equal(elements.projectRemoveCancel.disabled, false);
        assert.equal(elements.taskInput.disabled, true);

        await elements.projectRemoveForm.dispatch("submit");

        assert.equal(
          requests.filter((request) => request.method === "DELETE").length,
          2,
        );
        assert.equal(elements.projectRemoveDialog.open, false);
        assert.equal(document.querySelectorAll(".project-remove").length, 1);
        assert.equal(elements.taskInput.disabled, false);
        assert.equal(document.activeElement?.isConnected, true);
      },
      {
        deleteResponses: [
          { body: { detail: "任务运行期间暂时不能移除项目" }, status: 409 },
        ],
      },
    );
  });
});
