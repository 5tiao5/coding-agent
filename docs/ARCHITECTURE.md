# Architecture

## One control loop, separated projections of truth

```text
CLI host ────────────────────────────────────────┐
loopback Web host ── WebWorkbench                │
                         ├── ProjectRegistry     │
                         ├── RunCatalog          │
                         └── one WebRunService ──┤
                                                ▼
                               Repository application service
                                                │
                                                ▼
                         AgentRunner state machine ◄── task + system policy
                                  │
                                  ├── project-owned retry policy
                                  │        └── ModelAdapter.complete(messages, tool_specs)
                                  │             ├── ScriptedModel        offline tests/demo
                                  │             └── OpenAIResponsesModel transport only
                                  │
                                  ├── ToolRegistry.execute(tool_call)
                                  │        ├── read/search/list
                                  │        ├── compare-and-swap write/replace/undo
                                  │        ├── direct-argv command + approval policy
                                  │        └── explicit plan updates
                                  │
                                  ├── VerificationLedger + stop/context policies
                                  │
                                  └── RunEvent fan-out
                                           ├── JsonlEventSink           durable facts
                                           ├── DashboardEventSink       passive Rich UI
                                           └── DashboardProjection      bounded Web state
```

Live evaluation is another host, not another Agent implementation:

```text
evaluate CLI
   └── fresh public red fixture ──► Repository application service ──► AgentRunner
                                          │
                                          ▼ after the run
                               allowlisted source-only copy
                                          │
                                          ▼
                           sibling regression pytest oracle
```

The important separation is semantic, not cosmetic:

| Component | Owns | Must never do |
|---|---|---|
| `AgentResult` | outcome of this execution | trust model prose as verification |
| `SessionStore` | passive canonical transcript checkpoint | replay a tool or restore verification |
| `TraceStore` | observable audit history | authorize a command or resume evidence |
| `DashboardEventSink` | human-readable projection | influence runtime control flow |
| `WebRunService` | one worker and whitelisted browser snapshots | expose raw events/messages or execute tools itself |
| `WebWorkbench` | active project, global run admission, catalog navigation | become a second Agent loop or change roots mid-run |
| `ProjectRegistry` / `RunCatalog` | bounded navigation metadata | authorize tools, verification, or transcript resume |

This prevents a pretty UI, stale trace, or loaded JSON file from becoming an accidental authority.

## Stable interfaces

- `ModelAdapter.complete()` isolates provider protocols from the loop.
- `ToolDispatcher.specs()/execute()` isolates validation and tool implementation.
- `CommandPolicy.classify()` separates permission/evidence decisions from process launch.
- `EventSink.emit()` separates runtime facts from storage and presentation.
- `execute_repository_run()` owns shared repository-run composition for both CLI and Web hosts.
- `evaluate_case()` owns fixture isolation and independent regression judgement while reusing
  `execute_repository_run()` for the candidate repair.

The core runs one tool call at a time. Filesystem mutations are order-dependent; concurrency is not added until a tool is proven read-only and the gain justifies the extra failure states.

## Run budgets and oversized batches

The loop has three independent bounds: 20 model-decision turns by default, at most eight declared
tool calls in one model response, and at most 40 tool calls across the run. `--max-steps` changes only
the first bound. The live `run` and `web` hosts accept 1–100 turns, `resume` interprets the value as a
cumulative turn ceiling, and live evaluation accepts 1–20 turns per case. Neither CLI nor browser
configuration raises the per-turn or total tool budgets.

A response may declare several independent tool calls, but the registry still executes them in
order. An oversized batch is rejected atomically before any prefix executes. The loop appends a
bounded error result for every declared call and gives the model a bounded opportunity to resubmit
smaller batches on later turns. This preserves provider transcript invariants, avoids partially
applying an arbitrary prefix of mutations, and lets a multi-file task progress without requiring the
user to split it into separate prompts. The resubmitted calls remain subject to the 40-call total and
the remaining model-turn budget.

## CLI and loopback Web hosts

The CLI and live Web host both build a `RepositoryRunSpec` and call the same application service.
That service constructs the model, runtime tools, session store, trace-first event fan-out, and
`AgentRunner`; neither presentation owns another loop, transcript, Verification Gate, or tool path.
The deterministic Web demo uses the same repository demo core and public events against an ephemeral
fixture.

`WebWorkbench` adds a local project/navigation host around a replaceable `WebRunService`. It stores
bounded project records and immutable per-run metadata outside repositories, permits only one global
run, and captures an immutable project ID/root/fingerprint before entering the application service.
Registering a project is explicit; creating one means creating exactly one absent empty leaf. A
directory whose identity changed must be explicitly registered again. Project switching is rejected
while a run is active. Every run record also freezes that fingerprint: replacing a directory hides
its previous physical identity's runs and rejects stale direct-history URLs rather than reattributing
them. A nonterminal trace from a previous process is reported as interrupted, not still running.

`WebRunService` adds only a one-worker lifecycle and folds live events through `DashboardProjection`.
The FastAPI lifespan first closes admission and gives the current run a finite best-effort drain
window. The worker is daemonized as a final process-exit bound; normally completed runs still flush
their trace and checkpoint, while a shutdown that outlasts the window may interrupt them.
The runtime-to-browser document is an explicit read-only whitelist: task label, phase, bounded
counters and active-tool names, sanitized file-mutation summaries, current structured verifier
evidence and revision, outcome, plan lines, recent timeline, and latest Diff. The terminal response
and public error are separate result fields. Raw
`RunEvent` objects, event messages, canonical chat history, raw provider responses, read/search/command
output, provider state, credentials, and server configuration never enter that projection. Historical
views read a validated trace and rebuild the same whitelist from its latest run/resume segment; they
do not read the canonical checkpoint or claim to preserve final assistant text. In the live Windows
host, a token-protected adapter may open the native folder chooser; cancellation has no side effect,
and a returned path still enters the ordinary project-registration boundary. A short navigation
reservation prevents runs and project mutations from racing the open dialog. Manual absolute-path
input is the fallback. Normal run requests contain only a bounded task and use the server-held active
project; the browser cannot set a run root, model, base URL, state root, permission mode, verifier, or
limits.

## Runtime sequence

```text
CREATED → PLANNING → ACTING ↔ OBSERVING → VERIFYING
                                      ├→ COMPLETED
                                      ├→ COMPLETED_UNVERIFIED
                                      └→ FAILED
```

For each turn, the runner prepares a bounded model view and requests one response. Explicitly
classified transport failures receive bounded exponential backoff inside the runner. A malformed
function-argument JSON response is instead discarded before it enters canonical history; the runner
re-budgets an ephemeral, sanitized protocol-correction view and retries immediately. Both paths
share the host-owned attempt ceiling, the SDK retry count remains zero, and every retry is an event.
A successful response is appended, each declared tool executes through the registry, structured
observations are appended, a ready checkpoint is saved, and the loop continues. A response without
tool calls enters the Verification Gate. The terminal checkpoint event is emitted before
`RUN_FINISHED`, so the final event and UI card are genuinely terminal.

## Module map

| Area | Main modules |
|---|---|
| orchestration | `agent.py`, `models.py`, `context.py`, `stopping.py`, `verification.py` |
| provider boundary | `model.py`, `openai_model.py` |
| tools and policy | `tools/`, `workspace.py`, `mutation.py`, `command.py`, `approval.py` |
| application wiring | `application.py`, `runtime.py`, `cli.py`, `local_config.py`, `demo.py`, `web/runtime.py`, `web/workbench.py` |
| evaluation | `evaluation.py`, `evaluation_scenarios.py`, `evaluation_cli.py` |
| persistence | `session.py`, `trace.py`, `state.py`, `projects.py`, `run_catalog.py`, `lease.py`, `run_id.py` |
| presentation | `events.py`, `dashboard.py`, `presentation.py`, `ui.py`, `web/app.py`, `web/service.py`, `web/static/` |

`cli.py` performs argument handling and selects a host; its root callback loads only the
allowlisted keys from an exact launch-directory `.env.local` before Typer parses environment-backed
options. `application.py` owns shared repository composition. `demo.py` owns only the deterministic scenario, keeping presentation changes from
inflating either entry point. `evaluation.py` owns the host-side red/green protocol and result
metrics; scenario fixtures and regression-oracle tests remain declarative in
`evaluation_scenarios.py`. Those definitions are project code, not a secret benchmark boundary.
`evaluation_cli.py` owns only the stable JSON/table projection and report-file policy, keeping the
Typer host focused on argument handling and control flow.

## Dependency boundary

`openai` transports raw Responses requests, `python-dotenv` parses the fixed untracked local
configuration, `pydantic` validates data, `rich` renders,
`prompt-toolkit` reads task input, `fastapi`/`uvicorn` host the optional loopback view, `psutil` helps
control descendants, `pathspec` interprets ignore rules, and `pytest` supplies the default executable
check used by the demo/runtime. None of them selects tools, executes tools automatically, manages the
Agent state machine, grants verification, or decides when the run is complete.
