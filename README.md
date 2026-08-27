# Coding Agent

A small, observable coding agent built from first principles for the NJU software engineering recommendation assessment.

The project is currently at **M0**: a deterministic offline model can drive the hand-written agent loop, invoke a locally registered tool, consume its result, and finish with a structured event trace.

## Development setup

```powershell
uv sync
uv run coding-agent demo
uv run pytest
```

The offline demo requires no API key or network access. See [PLAN.md](PLAN.md) for scope, compliance boundaries, milestones, and acceptance criteria.

## Design boundary

Generic libraries handle transport, validation, terminal rendering, and testing. Agent orchestration, context policy, tool dispatch, stopping rules, and error semantics remain project-owned code.
