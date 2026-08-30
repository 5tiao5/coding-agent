# References and design provenance

This project implements its Agent loop, context policy, tool dispatch, stopping rules,
permissions, persistence semantics, verification, and evaluation path independently in this
repository. External material is used to understand requirements, transport protocols, and design
trade-offs; referenced projects are not runtime dependencies, and their source code is not copied
into this project.

| Source | What was studied | Decision and local boundary |
|---|---|---|
| NJU software engineering recommendation assessment (`推免考核题目学生版.pdf`, supplied outside this repository) | Required capabilities, deliverables, and prohibited use of ready-made Agent frameworks or hosted coding-agent implementations | Treated as the governing constraint. The resulting self-built/reused/prohibited boundary is recorded in [PLAN.md](../PLAN.md#2-合规边界). |
| [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) | Responses request/response objects and native function-call transport | Adopted only as a provider wire protocol through the low-level SDK. The Agent loop, tool execution, retries, transcript projection, and completion decision remain project-owned; provider storage and SDK retries are disabled. |
| [my-pi-agent](https://github.com/zxj-2023/my-pi-agent) | A comparative study of runtime steering/follow-up, layered context compaction, hooks, sessions, plugins, MCP, subagents, and coding-tool ergonomics | No implementation was copied and the repository is not a dependency. Runtime cancellation/steering and richer compaction remain possible future design inputs, not current capabilities. Plugins, MCP, subagents, and a general framework surface are rejected for this assessment because they expand scope and authority without strengthening the required verified coding loop. |
| Third-party libraries pinned in `pyproject.toml` / `uv.lock` | HTTP transport, validation, CLI/Web rendering, input, process enumeration, ignore matching, and tests | Reused as generic infrastructure only. The per-library responsibility and explicit non-responsibility are listed in [PLAN.md](../PLAN.md#6-依赖策略); architectural enforcement is summarized in [ARCHITECTURE.md](ARCHITECTURE.md#dependency-boundary). |

Similar concepts do not imply identical trust semantics. In particular, this project keeps runtime
events passive, launches commands through its own capability policy, and grants `VERIFIED` only from
fresh host-registered evidence after the latest known mutation. These choices are described in
[ARCHITECTURE.md](ARCHITECTURE.md) and [SECURITY.md](SECURITY.md).

When a future change is materially informed by an external design, this file should be updated in
the same commit to distinguish **adopted**, **adapted**, **deferred**, and **rejected** ideas.
