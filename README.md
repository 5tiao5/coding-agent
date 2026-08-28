# Coding Agent

[![CI](https://github.com/5tiao5/coding-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/5tiao5/coding-agent/actions/workflows/ci.yml)

A small, observable coding agent built from first principles for the NJU software engineering recommendation assessment.

The project is at **M4 (real model CLI, safe approval, durable traces, and demo UX)**. Its offline demo tells one complete repair story:

`plan → failing pytest → search/read → revision-checked diff → passing pytest → VERIFIED`

`VERIFIED` is not model prose. The runtime grants it only when a host-registered exact test/build/check command passes after the latest known workspace mutation.

## Quick start

```powershell
uv sync --all-groups
uv run coding-agent demo
uv run ruff check src tests
uv run mypy src tests
uv run pytest
```

The demo needs no API key or network. `ScriptedModel` supplies deterministic decisions, while the real Agent loop, tools, subprocess, diff, verification gate, events, and Rich dashboard still execute.

For a real repository run, provide the API key only through the environment and select the model explicitly:

```powershell
$env:OPENAI_API_KEY = "..."
$env:CODING_AGENT_MODEL = "<responses-compatible-model>"
uv run coding-agent run "Fix the failing tests and verify the repair" --root . --mode safe
```

For DeepSeek's Responses-compatible endpoint, use its currently supported Responses model and
explicitly disable reasoning for the stateless tool loop:

```powershell
$env:OPENAI_API_KEY = "<DeepSeek API key>"
$env:OPENAI_BASE_URL = "https://api.deepseek.com"
$env:CODING_AGENT_MODEL = "deepseek-v4-flash"
$env:CODING_AGENT_REASONING_EFFORT = "none"
uv run coding-agent run "Inspect this repository and report one verified improvement" --root . --mode safe
```

Omit the task string for a `prompt-toolkit` input prompt. No `--api-key` option exists, so the key
never enters the Agent argv/process list. Prefer a secret manager or session-scoped environment;
your shell may still record a literal environment-assignment command in its own history.

Useful commands:

```powershell
uv run coding-agent runs
uv run coding-agent inspect <run-id>
uv run coding-agent resume <run-id> --root . --model <model>
```

`safe` is the default: registered verification commands run directly, ordinary commands require a human confirmation, and non-interactive input denies them. `auto` skips that confirmation but keeps the destructive-command deny rules; it is not an OS sandbox.

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

The default live runtime registers only the exact current-Python `-I -m pytest -q` capability. Other
commands may run under the selected permission mode, but their output cannot manufacture
verification evidence.

## Important limits

- The Responses adapter uses `store=False` and explicitly sends a bounded context projection derived
  from the canonical transcript. It currently rejects a response that combines a reasoning item with
  function calls because safely continuing that turn requires encrypted provider state that this
  provider-neutral checkpoint deliberately does not persist. Use a Responses-compatible model that
  does not emit reasoning items during tool turns.
- Resume restores the canonical transcript, not `PlanState`, the in-memory undo journal, approval decisions, or old verification evidence. Fresh verification is mandatory.
- Resume is not crash-safe exactly-once execution: if a process dies after a tool changes disk but before the next checkpoint, the checkpoint can lag behind the workspace.
- A workspace-relative command `cwd` is containment for starting location, not malicious-code isolation. Run untrusted repositories in a container, VM, or low-privilege account.
- M5 still needs the four-category automated evaluation suite, clean-environment rehearsal, and final demo freeze.

## Design boundary

Generic libraries handle HTTP transport, validation, terminal rendering, input editing, process enumeration, ignore matching, and tests. Agent orchestration, context selection, tool definitions and dispatch, command launch/capability policy, stopping rules, checkpoint semantics, verification, and error semantics remain project-owned code.

```text
CLI
 ├─ OpenAIResponsesModel / ScriptedModel
 ├─ AgentRunner ─ Context + Stop + Verification
 │   └─ ToolRegistry ─ Workspace + CommandPolicy + MutationSession
 ├─ SessionStore + RunLease
 └─ CompositeEventSink
     ├─ JsonlEventSink       durable audit facts
     └─ DashboardEventSink   best-effort presentation
```

The runtime result, resumable checkpoint, audit trace, and dashboard are deliberately separate kinds of truth; none may authorize another. See [PLAN.md](PLAN.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and [docs/SECURITY.md](docs/SECURITY.md).
