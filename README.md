# Coding Agent

A small, observable coding agent built from first principles for the NJU software engineering recommendation assessment.

The project is currently at **M1 (read-only reconnaissance)**: a deterministic offline model can drive the hand-written agent loop through `list_files → search_text → read_file → final`, using the same rooted workspace and structured observations that the real model adapter will use.

## Development setup

```powershell
uv sync
uv run coding-agent demo
uv run pytest
```

The offline demo creates a tiny temporary repository, maps it, locates a symbol, reads the relevant code, and reports a defect. It requires no API key or network access, and it never edits the fixture. Its `ScriptedModel` is deliberately deterministic: the demo validates the real runtime/tool protocol, not autonomous model reasoning.

Current safety properties include workspace-relative paths only, `.gitignore` and sensitive-file filtering, no linked-directory traversal, verified file handles before reads, bounded directory work, per-file and aggregate search budgets, bounded model output, and explicit truncation metadata. Read-only tools intentionally fail closed when repository policy becomes ambiguous.

Next comes the write/diff/command layer and verification gate. See [PLAN.md](PLAN.md) for scope, compliance boundaries, milestones, and acceptance criteria.

## Design boundary

Generic libraries handle transport, validation, terminal rendering, and testing. Agent orchestration, context policy, tool dispatch, stopping rules, and error semantics remain project-owned code.
