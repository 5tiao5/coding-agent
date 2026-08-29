# Architecture

## One control loop, separated projections of truth

```text
CLI host ───────────┐
                    ├── Repository application service
loopback Web host ──┘             │
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

## CLI and loopback Web hosts

The CLI and live Web host both build a `RepositoryRunSpec` and call the same application service.
That service constructs the model, runtime tools, session store, trace-first event fan-out, and
`AgentRunner`; neither presentation owns another loop, transcript, Verification Gate, or tool path.
The deterministic Web demo uses the same repository demo core and public events against an ephemeral
fixture.

`WebRunService` adds only a one-worker lifecycle and folds events through `DashboardProjection`.
The FastAPI lifespan first closes admission and gives the current run a finite best-effort drain
window. The worker is daemonized as a final process-exit bound; normally completed runs still flush
their trace and checkpoint, while a shutdown that outlasts the window may interrupt them.
The runtime-to-browser document is an explicit read-only whitelist: task label, phase, bounded
counters and active-tool names, current verification labels, outcome, plan lines, recent timeline,
and latest Diff. The terminal response and public error are separate result fields. Raw
`RunEvent` objects, event messages, canonical chat history, raw provider responses, read/search/command
output, provider state, credentials, and server configuration never enter that projection. The
browser may submit a bounded task, but it cannot select the repository, model, base URL, state root,
permission mode, verifier, or run limits.

## Runtime sequence

```text
CREATED → PLANNING → ACTING ↔ OBSERVING → VERIFYING
                                      ├→ COMPLETED
                                      ├→ COMPLETED_UNVERIFIED
                                      └→ FAILED
```

For each turn, the runner prepares a bounded model view and requests one response. Explicitly
classified transient failures receive bounded exponential backoff inside the runner; the SDK retry
count is zero, and every attempt/retry is an event. A successful response is appended, each declared
tool executes through the registry, structured observations are appended, a ready checkpoint is
saved, and the loop continues. A response without tool calls enters the Verification Gate. The
terminal checkpoint event is emitted before `RUN_FINISHED`, so the final event and UI card are
genuinely terminal.

## Module map

| Area | Main modules |
|---|---|
| orchestration | `agent.py`, `models.py`, `context.py`, `stopping.py`, `verification.py` |
| provider boundary | `model.py`, `openai_model.py` |
| tools and policy | `tools/`, `workspace.py`, `mutation.py`, `command.py`, `approval.py` |
| application wiring | `application.py`, `runtime.py`, `cli.py`, `local_config.py`, `demo.py`, `web/runtime.py` |
| evaluation | `evaluation.py`, `evaluation_scenarios.py`, `evaluation_cli.py` |
| persistence | `session.py`, `trace.py`, `state.py`, `lease.py`, `run_id.py` |
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
