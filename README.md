# Coding Agent

[![CI](https://github.com/5tiao5/coding-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/5tiao5/coding-agent/actions/workflows/ci.yml)

A small, observable coding agent built from first principles for the NJU software engineering recommendation assessment.

The M4 core and M5 reliability/evaluation slice are complete. **M5.5** adds a bounded local project workbench around the same Agent runtime: choose or create a project, run one task, and replay that project's trusted trace history from a Codex-style sidebar. Both demos tell one complete repair story:

`plan → failing pytest → search/read → revision-checked diff → passing pytest → VERIFIED`

`VERIFIED` is not model prose. The runtime grants it only when a host-registered exact test/build/check command passes after the latest known workspace mutation.

## Quick start

```powershell
uv sync --all-groups
uv run coding-agent demo
uv run coding-agent web --demo
uv run coding-agent evaluate --help
uv run ruff check src tests
uv run mypy src tests
uv run pytest
```

Neither demo needs an API key or network. `coding-agent demo` uses the Rich terminal dashboard;
`coding-agent web --demo` binds only to `127.0.0.1:8765`, opens the browser, and starts the same
fixed repair scenario in an ephemeral fixture. `ScriptedModel` supplies deterministic decisions while
the real Agent loop, tools, subprocess, diff, verification gate, and events still execute. Use
`--no-open-browser` or `--port <port>` when needed.

For a real repository run, provide the API key through the process environment or the fixed,
untracked `.env.local` file and select the model explicitly:

```powershell
$env:OPENAI_API_KEY = "..."
$env:CODING_AGENT_MODEL = "<responses-compatible-model>"
uv run coding-agent run "Fix the failing tests and verify the repair" --root . --mode safe
```

For repeat local runs, copy the committed placeholder file once, edit the copy, and keep it private:

```powershell
Copy-Item .env.example .env.local
notepad .env.local
uv run coding-agent evaluate --live --allow-paid-api --case single-file --format json
```

Only `.env.local` in the launch directory is read; parent directories are never searched. The
loader accepts only the four model settings shown in `.env.example`, disables value interpolation,
and never replaces a value already present in the process environment. The file is Git-ignored but
is still plaintext on disk, so do not display it in a recording or reuse a key that was committed.

For DeepSeek's Responses-compatible endpoint, use its currently supported Responses model and
explicitly disable reasoning for the stateless tool loop:

```powershell
$env:OPENAI_API_KEY = "<DeepSeek API key>"
$env:OPENAI_BASE_URL = "https://api.deepseek.com"
$env:CODING_AGENT_MODEL = "deepseek-v4-flash"
$env:CODING_AGENT_REASONING_EFFORT = "none"
uv run coding-agent run "Inspect this repository and report one verified improvement" --root . --mode safe
```

The optional live Web workbench uses the same environment and repository-owned application service:

```powershell
# Start at the project chooser. The model may also come from .env.local.
uv run coding-agent web --model <model> --mode safe

# Optional shortcut: register and select this directory immediately.
uv run coding-agent web --root . --model <model> --mode safe
```

On Windows, “打开项目” first opens the native folder chooser; manual absolute-path input remains as
a fallback. “新建项目” can browse for its parent and then creates exactly one previously absent empty
leaf directory; it does not initialize Git, add a template, or overwrite a directory. Project and
new-run history metadata live in the private state directory, not inside the repository. The sidebar
replays only bounded trace projections; older runs that predate M5.5 metadata are not retroactively
catalogued.

### Run budgets

`--max-steps` limits model-decision turns, not files or tool calls. Increasing it gives the model
more opportunities to observe results and decide what to do next; it does **not** enlarge the fixed
tool budgets.

| Host | Model-turn budget | Tool-call budget |
|---|---|---|
| `run` / live `web` | 20 by default; `--max-steps` accepts 1–100 | 8 per turn, 40 per run (fixed) |
| `resume` | 20 cumulative turns by default; `--max-steps` accepts 1–100 | 8 per turn, 40 cumulative calls (fixed) |
| live `evaluate` | 12 per case by default; `--max-steps` accepts 1–20 | 8 per turn, 40 per case (fixed) |
| deterministic CLI/Web demo | 12 fixed turns | 8 per turn, 40 per run (fixed) |

One model turn may declare several independent tools, which the Agent executes sequentially. If a
model proposes more than eight at once, the Agent rejects the whole oversized batch before any of
its calls execute, returns bounded feedback, and lets the model automatically resubmit the work in
smaller batches. A multi-file task therefore does not need to be split into separate user prompts;
it still has to fit within the model-turn and 40-call run budgets. For Web, `--max-steps` is selected
when the local server starts and applies to its live runs; the browser cannot override it per task.

### Opt-in live evaluation

The evaluation command is network-disabled by default. A real-model run requires both explicit
consent flags; there is no API-key argument:

```powershell
# One isolated case (recommended first smoke)
uv run coding-agent evaluate --live --allow-paid-api --case single-file

# Four categories, stable JSON, and a new report file
uv run coding-agent evaluate --live --allow-paid-api --case all `
  --format json --output evaluation.json
```

The command prints the endpoint host and the maximum request-attempt budget before starting. Each
case begins with a known-red public test in a fresh temporary repository. After the Agent stops, the
host copies only allowlisted source files into a sibling directory and runs separate regression tests
with credentials removed. Public tests and pytest control files must remain byte-identical, while a
whole-repository before/after manifest rejects every change outside the scenario's
explicit path allowlist. A zero exit without a completed, non-empty all-passing oracle is rejected.
Reports omit raw test output,
commands, provider details, credentials, run IDs, and base URLs, and existing report files are never
overwritten. This is an independent scoring path, not a secrecy boundary: the built-in scenario and
oracle definitions ship with the project and code executing in the same environment could inspect
them. Use an external container or service for a genuinely private benchmark.

Evaluation exits `0` when every attempted case passes, `3` for a genuine task/oracle failure, `1`
for a runner/provider/harness error, and `2` for invalid configuration. A suite stops after the first
`error` to limit paid requests. Unit tests exercise the same harness with injected deterministic
models; they are not presented as evidence of real-model intelligence.

The browser can select a locally registered project, submit one task, and observe its bounded plan, timeline, latest Diff, structured
file-change-to-verifier evidence chain, and final response. In this first slice, Web `safe` mode has no browser approval broker:
registered verification commands may run, while ordinary commands fail closed. Select `--mode auto`
only when that broader local command authority is intentional; it remains subject to the command
deny rules and is not an OS sandbox.

Omit the task string for a `prompt-toolkit` input prompt. No `--api-key` option exists, so the key
never enters the Agent argv/process list. Prefer a secret manager or session-scoped environment for
stronger protection; `.env.local` is the convenient untracked development option, and a shell may
still record a literal environment-assignment command in its own history.

Useful commands:

```powershell
uv run coding-agent runs
uv run coding-agent inspect <run-id>
uv run coding-agent resume <run-id> --root . --model <model>
```

`safe` is the default. The CLI offers an exact-argv confirmation for ordinary commands and denies
non-interactive input; the Web view currently denies those commands because it has no approval
broker. `auto` skips confirmation in either host but keeps the destructive-command deny rules; it is
not an OS sandbox.

For `run` and `resume`, exit codes distinguish control outcomes:

| Code | Meaning |
|---:|---|
| `0` | `COMPLETED` with current trusted verification |
| `1` | runtime/setup failure or refused resume |
| `2` | CLI usage or argument validation error |
| `3` | final response exists, but verification is missing, failed, or stale |

## What M4 adds

- A raw OpenAI Responses adapter that maps the project-owned transcript and custom function tools without an Agent SDK, hosted tools, automatic tool execution, or provider-managed conversation state.
- A real `run` command, interactive task entry, `safe`/`auto` selection, exact-argv approval panel, and environment-only credential flow.
- A passive Rich dashboard with plan/diff previews, tool durations, verification evidence, and an explicit `VERIFIED`/`UNVERIFIED` final card.
- Bounded per-run JSONL traces plus read-only `runs` and `inspect` commands.
- Schema-v2 checkpoints bound to an opaque workspace identity, completed-trace rejection, and a cross-process lease held by both the original run and same-ID resume.
- A deployment-safe pytest capability: if Python itself lives inside the repository, its exact executable hash is bound before it may issue verification evidence.

The default live runtime registers only the exact current-Python
`-I -B -m pytest -q -p no:cacheprovider` capability. The bytecode and cache-provider flags keep
trusted verification observational: running it does not create Python or pytest cache files in the
workspace. Other
commands may run under the selected permission mode, but their output cannot manufacture
verification evidence.

## M5.5 local project workbench

- CLI and Web both call the same repository application service and therefore share the Agent loop,
  tools, permission policy, trace-first event ordering, checkpoints, and Verification Gate.
- The token-protected local host may open a native Windows folder chooser and return its selection to
  the page; manual absolute-path registration remains available. Both paths enter the same server
  validation. After selection, the server resolves and fingerprints that project; a run captures an
  immutable project context and never accepts a root from the task request.
- A private project registry and immutable per-run catalog power the left sidebar. History is rebuilt
  from the latest validated trace segment through the same dashboard whitelist; it is read-only and
  does not expose canonical messages or pretend to persist the final answer. Each run is bound to the
  directory fingerprint captured at start, so replacing a directory cannot inherit its old history;
  an unterminated trace from an earlier process is labeled `interrupted`.
- At most one background run is active across all projects, and project changes are locked while it
  runs. Graceful server shutdown stops accepting tasks and attempts
  a bounded drain before process exit. Browser state is a whitelist projection, not raw events,
  canonical messages, tool output, raw provider response objects, or hidden reasoning.
- This is a loopback presentation slice, not a remote or multi-user Web product, packaged desktop
  shell, Git clone/template service, browser terminal, editor, or alternate Agent implementation.
- Transient connection, timeout, throttling, and selected server failures are retried by the
  project-owned loop, not invisibly by the SDK. Retry scheduling is visible in both event projections
  without exposing provider exception text.

## M5 reliability and evaluation slice

- SDK retries remain disabled; the Agent owns every bounded retry. Connection/timeouts and
  `408/409/429/5xx` use `0.5s → 1s` backoff, while malformed function-argument JSON is discarded
  before tool execution and retried immediately with a fixed, sanitized protocol-correction
  instruction. Both paths share the configured attempt ceiling and are distinguished in events.
- Four stable categories cover a single-file repair, cross-file change, new feature, and indirect
  fault. Success requires `AgentState.COMPLETED`, unchanged public-test controls, required source
  changes, no workspace change outside the scenario allowlist, and a separate sibling regression
  oracle. JSON reports expose the bounded scope proof as `coding-agent.eval.v2`.
- Case results distinguish `passed`, `failed`, and `error`; aggregate metrics include success rate,
  steps, duration, and tool failures.
- On 2026-08-29, `deepseek-v4-flash` passed all four categories individually and again in one
  continuous suite: 8/8 case runs, five steps per case, zero tool errors; the full suite took 43.05s.

## Important limits

- The Responses adapter uses `store=False` and explicitly sends a bounded context projection derived
  from the canonical transcript. It currently rejects a response that combines a reasoning item with
  function calls because safely continuing that turn requires encrypted provider state that this
  provider-neutral checkpoint deliberately does not persist. Use a Responses-compatible model that
  does not emit reasoning items during tool turns.
- Resume restores the canonical transcript, not `PlanState`, the in-memory undo journal, approval decisions, or old verification evidence. Fresh verification is mandatory.
- Resume is not crash-safe exactly-once execution: if a process dies after a tool changes disk but before the next checkpoint, the checkpoint can lag behind the workspace.
- A workspace-relative command `cwd` is containment for starting location, not malicious-code isolation. Run untrusted repositories in a container, VM, or low-privilege account.
- The local Web view is not authenticated for hostile same-machine users. It binds to loopback,
  requires a per-process control token for mutations, and applies Host/Origin/header restrictions,
  but it is not intended for port forwarding or remote exposure.
- Graceful Web shutdown gives an active run five seconds to drain. A longer run, forced process kill,
  or machine failure can interrupt its final trace/checkpoint write; this is not crash-safe execution.
- Final delivery still needs a clean-environment rehearsal, three runs of the frozen demo task, and final video freeze.

## Design boundary

Generic libraries handle HTTP transport, validation, terminal rendering, input editing, process enumeration, ignore matching, and tests. Agent orchestration, context selection, tool definitions and dispatch, command launch/capability policy, stopping rules, checkpoint semantics, verification, and error semantics remain project-owned code.

```text
CLI ───────────────────────────────────────────────┐
loopback Web ─ WebWorkbench                        │
                 ├─ ProjectRegistry + RunCatalog   │
                 └─ one WebRunService ─────────────┤
                                                   ▼
                                  Repository application service
                                    ├─ OpenAIResponsesModel / ScriptedModel
                                    ├─ AgentRunner ─ Context + Stop + Verification
                                    │   └─ ToolRegistry ─ Workspace + CommandPolicy + MutationSession
                                    ├─ SessionStore + RunLease
                                    └─ CompositeEventSink
                                        ├─ JsonlEventSink       durable audit facts
                                        └─ DashboardProjection  bounded live/history presentation

evaluate CLI
 └─ isolated public fixture → same application service
     └─ allowlisted source copy → sibling regression oracle
```

The runtime result, resumable checkpoint, audit trace, and dashboard are deliberately separate kinds of truth; none may authorize another. See [PLAN.md](PLAN.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/SECURITY.md](docs/SECURITY.md), and [docs/REFERENCES.md](docs/REFERENCES.md).
