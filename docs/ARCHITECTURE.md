# Architecture

## One control loop, four projections of truth

```text
task + system policy
        │
        ▼
  AgentRunner state machine
        │
        ├── ModelAdapter.complete(messages, tool_specs)
        │        ├── ScriptedModel            offline tests/demo
        │        └── OpenAIResponsesModel     transport only
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
                 └── DashboardEventSink       passive UI
```

The important separation is semantic, not cosmetic:

| Component | Owns | Must never do |
|---|---|---|
| `AgentResult` | outcome of this execution | trust model prose as verification |
| `SessionStore` | passive canonical transcript checkpoint | replay a tool or restore verification |
| `TraceStore` | observable audit history | authorize a command or resume evidence |
| `DashboardEventSink` | human-readable projection | influence runtime control flow |

This prevents a pretty UI, stale trace, or loaded JSON file from becoming an accidental authority.

## Stable interfaces

- `ModelAdapter.complete()` isolates provider protocols from the loop.
- `ToolDispatcher.specs()/execute()` isolates validation and tool implementation.
- `CommandPolicy.classify()` separates permission/evidence decisions from process launch.
- `EventSink.emit()` separates runtime facts from storage and presentation.

The core runs one tool call at a time. Filesystem mutations are order-dependent; concurrency is not added until a tool is proven read-only and the gain justifies the extra failure states.

## Runtime sequence

```text
CREATED → PLANNING → ACTING ↔ OBSERVING → VERIFYING
                                      ├→ COMPLETED
                                      ├→ COMPLETED_UNVERIFIED
                                      └→ FAILED
```

For each turn, the runner prepares a bounded model view, requests one response, appends the assistant message, executes every declared tool through the registry, appends structured observations, saves a ready checkpoint, and continues. A response without tool calls enters the Verification Gate. The terminal checkpoint event is emitted before `RUN_FINISHED`, so the final event and UI card are genuinely terminal.

## Module map

| Area | Main modules |
|---|---|
| orchestration | `agent.py`, `models.py`, `context.py`, `stopping.py`, `verification.py` |
| provider boundary | `model.py`, `openai_model.py` |
| tools and policy | `tools/`, `workspace.py`, `mutation.py`, `command.py`, `approval.py` |
| application wiring | `runtime.py`, `cli.py`, `demo.py` |
| persistence | `session.py`, `trace.py`, `state.py`, `lease.py`, `run_id.py` |
| presentation | `events.py`, `dashboard.py`, `presentation.py`, `ui.py` |

`cli.py` performs composition and argument handling; it does not implement Agent behavior. `demo.py` owns only the deterministic scenario, keeping presentation changes from inflating the CLI entry point.

## Dependency boundary

`openai` transports raw Responses requests, `pydantic` validates data, `rich` renders,
`prompt-toolkit` reads task input, `psutil` helps control descendants, `pathspec` interprets
ignore rules, and `pytest` supplies the default executable check used by the demo/runtime. None of
them selects tools, executes tools automatically, manages the Agent state machine, grants
verification, or decides when the run is complete.
