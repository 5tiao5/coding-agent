# Coding Agent

A small, observable coding agent built from first principles for the NJU software engineering recommendation assessment.

The project is currently at **M3 (runtime control core)**. Its deterministic offline demo follows a real repair story:

`plan → failing pytest → search/read → revision-checked edit → passing pytest → VERIFIED`

`VERIFIED` is not model prose. It is a terminal state granted only by fresh runtime evidence from a host-registered exact command capability whose entry point is recognized as a test, build, or check.

## Development setup

```powershell
uv sync
uv run coding-agent demo
uv run ruff check .
uv run mypy
uv run pytest
```

The demo creates a temporary repository, reproduces a failing test, locates a duplicate-discount defect, applies one compare-and-swap edit, and reruns the same test capability. It requires no API key or network access. `ScriptedModel` is deliberately deterministic: the demo validates the real runtime/tool protocol, not autonomous model reasoning.

M3 adds direct-argv command execution, bounded output, timeouts and descendant cleanup, `safe`/`auto` policy primitives, context compaction, repeated-call stopping, explicit plan state, passive checkpoints, and a trusted Verification Gate. The M2 workspace layer still provides rooted paths, ignore/sensitive filtering, SHA-256 stale-write checks, link rejection, atomic replacement, bounded escaped diffs, and conditional undo.

Current limits are intentional: the real model adapter, interactive task/approval CLI, JSONL `inspect`, richer live UI, and automated evaluation suite remain later milestones. Checkpoint resume restores the canonical transcript, not `PlanState` or the in-memory undo journal. A workspace-relative command `cwd` is not an OS sandbox; see [docs/SECURITY.md](docs/SECURITY.md).

See [PLAN.md](PLAN.md) for scope, compliance boundaries, milestones, and acceptance criteria.

## Design boundary

Generic libraries handle validation, terminal rendering, process enumeration, ignore matching, and testing. Agent orchestration, command launch and capability policy, context selection, tool dispatch, verification, stopping rules, and error semantics remain project-owned code.
