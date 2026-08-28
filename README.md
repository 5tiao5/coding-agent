# Coding Agent

A small, observable coding agent built from first principles for the NJU software engineering recommendation assessment.

The project is currently at **M2 (safe workspace mutation)**: a deterministic offline model drives the hand-written agent loop through `list_files → search_text → read_file → replace_text → read_file → final`, using the same rooted workspace and structured observations that the real model adapter will use.

## Development setup

```powershell
uv sync
uv run coding-agent demo
uv run pytest
```

The offline demo creates a tiny temporary repository, maps it, locates a defect, applies one compare-and-swap edit, and reads the changed file as postcondition evidence. It requires no API key or network access. Its `ScriptedModel` is deliberately deterministic: the demo validates the real runtime/tool protocol, not autonomous model reasoning.

Current safety properties include workspace-relative paths only, `.gitignore` and sensitive-file filtering, verified file handles before reads, SHA-256 stale-write checks, link/hardlink rejection, same-directory atomic replacement, no-clobber creation, bounded escaped diffs, and conditional session undo. Tools intentionally fail closed when repository policy becomes ambiguous. The precise threat model and the deliberately documented Windows race boundary are in [docs/SECURITY.md](docs/SECURITY.md).

Next comes the command tool and verification gate. See [PLAN.md](PLAN.md) for scope, compliance boundaries, milestones, and acceptance criteria.

## Design boundary

Generic libraries handle transport, validation, terminal rendering, and testing. Agent orchestration, context policy, tool dispatch, stopping rules, and error semantics remain project-owned code.
