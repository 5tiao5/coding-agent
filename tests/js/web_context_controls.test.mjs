import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");

async function importStaticModule(relativePath) {
  const source = readFileSync(path.join(repositoryRoot, relativePath), "utf8");
  const encoded = Buffer.from(source).toString("base64");
  return import(`data:text/javascript;base64,${encoded}`);
}

test("resume controls fail closed when an older backend omits capability fields", async () => {
  const { continuationControlView, resumeControlView } = await importStaticModule(
    "src/coding_agent/web/static/_workbench.js",
  );

  assert.deepEqual(resumeControlView({ status: "interrupted" }), {
    available: false,
    known: false,
    reason: "",
  });
  assert.deepEqual(
    resumeControlView({
      resume_available: true,
      resume_reason: null,
      run_id: "run-1",
    }),
    { available: true, known: true, reason: "" },
  );
  assert.deepEqual(
    continuationControlView({
      continuation: { available: true, kind: "follow_up", reason: null },
      status: "completed",
    }),
    { available: true, kind: "follow_up", known: true, reason: "" },
  );
  assert.deepEqual(
    continuationControlView({ resume_available: true, resume_reason: null }),
    { available: true, kind: "resume", known: true, reason: "" },
  );
  assert.deepEqual(
    resumeControlView({ resume_available: false, resume_reason: "checkpoint_missing" }),
    { available: false, known: true, reason: "checkpoint_missing" },
  );
});

test("project memory presentation accepts rich sources and run-id-only provenance", async () => {
  const { normalizeProjectMemoryContext, projectMemoryContextView } = await importStaticModule(
    "src/coding_agent/web/static/_workbench.js",
  );

  assert.deepEqual(normalizeProjectMemoryContext(undefined), {
    applied: false,
    contextChars: null,
    error: "",
    known: false,
    requested: false,
    sources: [],
  });

  const normalized = normalizeProjectMemoryContext({
    applied: true,
    context_chars: 321,
    requested: true,
    source_run_ids: ["run-1", "run-2"],
    sources: [
      {
        completed_at: "2026-09-01T12:00:00Z",
        run_id: "run-1",
        task_goal: "修复路径规划",
      },
    ],
  });

  assert.equal(normalized.known, true);
  assert.equal(normalized.requested, true);
  assert.equal(normalized.applied, true);
  assert.equal(normalized.contextChars, 321);
  assert.equal(normalized.error, "");
  assert.deepEqual(normalized.sources, [
    {
      completedAt: "2026-09-01T12:00:00Z",
      runId: "run-1",
      task: "修复路径规划",
    },
    { completedAt: "", runId: "run-2", task: "" },
  ]);

  assert.equal(
    normalizeProjectMemoryContext({ requested: true, error: "项目记忆暂时不可用" }).error,
    "项目记忆暂时不可用",
  );

  const explicitParent = projectMemoryContextView({
    applied: true,
    requested: false,
    source_run_ids: ["parent-run"],
  });
  assert.equal(explicitParent.visible, true);
  assert.equal(explicitParent.parentOnly, true);
  assert.deepEqual(explicitParent.sources, [
    { completedAt: "", runId: "parent-run", task: "" },
  ]);
  assert.equal(
    projectMemoryContextView({ applied: false, requested: false }).visible,
    false,
  );
});

test("resume reasons have stable Chinese fallbacks", async () => {
  const { translateResumeReason } = await importStaticModule(
    "src/coding_agent/web/static/locale-zh.js",
  );

  assert.equal(translateResumeReason("checkpoint_missing"), "没有可恢复的检查点");
  assert.equal(translateResumeReason("future_reason"), "future_reason");
  assert.equal(translateResumeReason(null), "该任务当前不能继续");
});

test("the task composer exposes an enabled-by-default project memory choice", () => {
  const html = readFileSync(
    path.join(repositoryRoot, "src/coding_agent/web/static/index.html"),
    "utf8",
  );

  assert.match(html, /id="use-project-memory"[^>]*type="checkbox"[^>]*checked/);
  assert.match(html, />使用项目记忆</);
});

test("the workbench separates checkpoint resume from completed-run follow-up", () => {
  const source = readFileSync(
    path.join(repositoryRoot, "src/coding_agent/web/static/_workbench.js"),
    "utf8",
  );

  assert.match(source, /`\/api\/runs\/\$\{encodeURIComponent\(id\)\}\/resume`/);
  assert.match(source, /createElement\("button", "history-item-main"\)/);
  assert.match(source, /continuation\.kind === "resume" \? "恢复" : "继续"/);
  assert.match(source, /function prepareFollowUp\(id\)/);

  const appSource = readFileSync(
    path.join(repositoryRoot, "src/coding_agent/web/static/app.js"),
    "utf8",
  );
  assert.match(appSource, /payload\.parent_run_id = followUp\.runId/);
});

test("project removal is explicit, reversible, and never claims to delete the workspace", () => {
  const html = readFileSync(
    path.join(repositoryRoot, "src/coding_agent/web/static/index.html"),
    "utf8",
  );
  const source = readFileSync(
    path.join(repositoryRoot, "src/coding_agent/web/static/_workbench.js"),
    "utf8",
  );

  assert.match(html, /id="project-remove-dialog"/);
  assert.match(html, /不会删除本地目录/);
  assert.match(html, /重新打开同一路径，可恢复历史与项目记忆/);
  assert.match(source, /createElement\("button", "project-remove", "移除"\)/);
  assert.match(source, /method: "DELETE"/);
  assert.match(source, /`\/api\/projects\/\$\{encodeURIComponent\(target\.id\)\}`/);
  assert.match(source, /result\.workspace_deleted !== false/);
  assert.match(source, /本地目录未删除/);
});
