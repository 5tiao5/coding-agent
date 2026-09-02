# Relay Coding Agent

[![CI](https://github.com/5tiao5/coding-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/5tiao5/coding-agent/actions/workflows/ci.yml)

> 一个从核心循环开始自行实现、可观察、可恢复、可验证的本地编程智能体。

本项目面向南京大学软件工程推免考核。Relay 能在指定项目目录中阅读与搜索代码、制定计划、修改文件、运行命令，并根据结果继续迭代。它同时提供终端界面和中文 Web 工作台，但两种界面共用同一套 Agent 内核、工具系统、权限策略与可信验证逻辑。

一次完整任务的主线是：

~~~text
任务 → 计划 → 读取/搜索 → 修改与 Diff → 执行检查 → 可信验证 → 最终回复
~~~

其中 **VERIFIED（已验证）不是模型自己说了算**。只有宿主在最新文件变更之后运行登记过的精确检查，并确认项目策略与受保护内容没有漂移，Verification Gate 才会授予这一状态。

第一次了解项目，推荐先阅读 [工作原理与源码导览](docs/HOW_IT_WORKS.md)。

## 核心特色

- **核心机制自行实现**：自行实现 Model–Tool–Observation 循环、上下文选择、工具调度、停止条件、重试、检查点和验证判定，不依赖 LangChain、OpenAI Agents SDK 等 Agent 框架。
- **CLI 与 Web 双界面**：Rich 终端适合开发与调试；中文 Web 工作台适合项目管理、过程观察与视频演示。
- **全过程可观察**：展示结构化计划、工具调用、命令参数、工作目录、输出、耗时、跨恢复阶段的有界变更账本、可展开 Diff 和验证证据；超限会明确标注。
- **可信验证**：模型文本、命令退出码和“任务已验证”三者分离，旧证据在文件变化后立即失效。
- **任务连续性**：支持恢复中断运行、继续已完成任务，以及项目级轻量记忆 ProjectMemory。
- **独立评测**：在临时仓库运行真实模型，再由工作区外 sibling oracle 独立检查结果，统计假阳性。
- **有界且诚实**：模型轮次、工具调用、上下文、输出和历史展示都有明确上限；发生裁剪、过期或缺失时会直接标注。

## 快速开始

需要 Python 3.11 或更高版本，以及 [uv](https://docs.astral.sh/uv/)。

~~~powershell
git clone https://github.com/5tiao5/coding-agent.git
cd coding-agent
uv sync --locked --all-groups
~~~

### 离线终端演示

~~~powershell
uv run coding-agent demo
~~~

### 离线 Web 演示

~~~powershell
uv run coding-agent web --demo
~~~

浏览器将打开 http://127.0.0.1:8765。两个离线演示都不需要 API Key 或网络；ScriptedModel 只提供确定性的模型决策，真实的 Agent 循环、工具、子进程、Diff、事件与 Verification Gate 仍会完整执行。

如不希望自动打开浏览器：

~~~powershell
uv run coding-agent web --demo --no-open-browser
~~~

## 使用真实模型

### 配置 API

Relay 只从进程环境或启动目录中的未跟踪文件 .env.local 读取模型配置。推荐先复制模板：

~~~powershell
Copy-Item .env.example .env.local
notepad .env.local
~~~

配置项如下：

~~~dotenv
OPENAI_API_KEY=你的密钥
OPENAI_BASE_URL=Responses API 兼容端点
CODING_AGENT_MODEL=模型名称
CODING_AGENT_REASONING_EFFORT=none
~~~

**OPENAI_API_KEY 只是统一变量名，并不限制密钥厂商或前缀。** 当前适配器调用 OpenAI Responses API，并使用 function_call / function_call_output 工具语义，因此服务端必须真正兼容 Responses API。仅支持 Chat Completions，或只提供 Claude、Gemini 原生协议的服务不能直接接入。

- 使用 OpenAI 官方服务时，删除 OPENAI_BASE_URL，并选择支持 Responses 与函数工具调用的模型。
- DeepSeek 已通过本项目真实评测；无状态工具循环应将推理强度设为 none。
- 其他网关或本地模型服务只有完整兼容 Responses API 和工具调用时才可能使用，不能只凭“OpenAI-compatible”字样判断。
- 远程端点必须使用 HTTPS；只有 localhost 或回环地址允许 HTTP。URL 不能包含凭据、查询参数或 fragment。

DeepSeek 配置示例：

~~~dotenv
OPENAI_API_KEY=<DeepSeek API Key>
OPENAI_BASE_URL=https://api.deepseek.com
CODING_AGENT_MODEL=deepseek-v4-flash
CODING_AGENT_REASONING_EFFORT=none
~~~

.env.local 只会从命令启动目录读取，不向父目录搜索；进程环境中已有的值优先。该文件已被 Git 忽略，但仍是磁盘明文，请勿在录屏中展示，也不要提交真实密钥。

### 启动 Web 工作台

~~~powershell
# 进入项目选择页
uv run coding-agent web --mode auto --max-steps 50

# 启动时直接选择当前项目
uv run coding-agent web --root . --mode auto --max-steps 50
~~~

Web 工作台支持：

- 使用 Windows 原生文件夹选择器打开项目，也可手动填写绝对路径；
- 创建一个此前不存在的空目录作为新项目；
- 在左侧边栏切换项目、查看历史、继续任务或移除项目；
- 实时查看计划、工具、命令、验证、文件变更、可展开 Diff 和最终回复；
- 回放历史运行，并汇总该 run-id 跨恢复阶段最近 100 次文件变更；若更早记录被省略，界面会明确显示数量；
- 一次只运行一个后台任务，避免两个 Agent 同时修改工作区。

“移除项目”只会从 Relay 边栏隐藏登记项，不会删除磁盘目录和私有历史；重新打开相同目录即可恢复其身份、历史与项目记忆。

### 使用 CLI

~~~powershell
# safe 是默认模式，普通命令会请求确认
uv run coding-agent run "修复失败测试并验证修改" --root . --mode safe

# 不传任务字符串时进入交互输入
uv run coding-agent run --root . --mode safe

# auto 跳过普通命令确认，但仍受拒绝规则约束
uv run coding-agent run "完成项目并运行测试" --root . --mode auto --max-steps 50
~~~

CLI 的任务结果使用稳定退出码：

| 退出码 | 含义 |
|---:|---|
| 0 | 任务完成，且当前修订获得可信验证 |
| 1 | 运行或初始化失败，或恢复被拒绝 |
| 2 | CLI 用法或参数校验错误 |
| 3 | 已生成最终回复，但验证缺失、失败或已经过期 |

## Agent 如何工作

Relay 不把大模型当作“拥有电脑的程序”，而把它当作可替换的决策器。真正的执行权始终属于宿主。

1. 宿主固定项目根目录、权限模式、预算、工具清单和项目验证策略。
2. ContextManager 从规范 transcript 与 RunMemory 中选择有界上下文。
3. ModelAdapter 请求模型，让模型返回自然语言回复或结构化工具调用。
4. ToolRegistry 校验参数、路径、版本和命令能力后执行工具。
5. 工具结果以 Observation 返回模型；文件写入同时产生变更记录和 Diff。
6. 文件变化会让旧验证证据失效；Agent 必须在当前修订上重新运行登记检查。
7. Stop Policy 与 Verification Gate 根据宿主事实决定任务终态，而不是相信模型的自我声明。
8. JsonlEventSink、终端和 Web 分别保存或展示同一批事件的不同投影。

~~~text
CREATED → PLANNING → ACTING ↔ OBSERVING → VERIFYING
                                      ├→ COMPLETED
                                      ├→ COMPLETED_UNVERIFIED
                                      └→ FAILED
~~~

文件工具默认采用顺序执行，因为修改通常具有先后依赖。单轮可以声明多个独立工具，但最多八个；超大批次会在任何调用执行前整体拒绝，并让模型拆分后重试。

## 可信验证

### 为什么 pytest 返回 0 还不一定算 VERIFIED

未配置项目可以运行默认 pytest，但结果只记为 checks_only。原因是 Relay 无法凭空知道宿主 Python 是否就是项目真实运行环境，也不知道哪些测试、构建或入口检查足以证明任务完成。

要启用可信项目验证，请在仓库根目录创建 .coding-agent/project.toml：

~~~toml
schema_version = 1
protected_paths = ["tests/", "conftest.py", "pyproject.toml"]

[python]
# Windows；使用正斜杠可避免 TOML 反斜杠转义
executable = ".venv/Scripts/python.exe"
# macOS/Linux 可使用：executable = ".venv/bin/python"

[[verifiers]]
label = "pytest"
type = "pytest"
cwd = "."
scopes = ["tests"]
required = true

[completion]
required_scopes = ["tests"]
~~~

可信验证同时绑定：

- 明确声明的解释器及其文件哈希；
- verifier 的类型、工作目录和作用域；
- 项目策略指纹；
- protected_paths 的前后内容；
- 最近一次工作区修改之后的新鲜检查证据。

Agent 的工作区文件工具不能读取或修改 .coding-agent/project.toml。每次 verifier 执行前后，宿主都会重新检查策略与受保护内容；任何漂移都会使证据无效。

可运行包还可以增加 type = "python-module" 的入口冒烟检查。该类型固定执行无参数命令 python -B -m your_package，不接受任意 shell 文本或额外参数。

## 恢复、继续与 ProjectMemory

这三个概念解决不同层次的问题：

- **恢复中断任务**：只接受停在 READY_FOR_MODEL 边界、工作区和策略指纹仍匹配的检查点；沿用原 run-id、transcript 和累计预算，但不会恢复旧验证证据。
- **继续已完成任务**：创建新的子运行，通过 parent_run_id 指向只读父运行；子任务获得父摘要，但会重新读取当前代码并重新验证。
- **ProjectMemory**：为普通新任务选择少量同项目历史摘要，包括目标、结果、变更统计、验证结果和未解决项。它不保存源码、完整 Diff、命令输出、provider payload 或隐式推理。

历史内容始终是不可信参考，不能授权工具，也不能替代当前验证。Web 中可以关闭普通项目记忆；显式“继续”仍必须使用父任务摘要，否则操作会失败而不是悄悄退化成无关的新任务。

CLI 也提供只读历史与同运行恢复：

~~~powershell
uv run coding-agent runs
uv run coding-agent inspect <run-id>
uv run coding-agent resume <run-id> --root . --model <model>
~~~

## 权限模式与运行预算

| 宿主 | 模型轮次 | 工具调用预算 |
|---|---|---|
| run / live web | 默认 20；max-steps 可设 1–100 | 单轮最多 8；累计 max(8, 2 × 轮次) |
| CLI / Web resume | 默认累计 20；max-steps 可设 1–100 | 包含恢复前已消费调用 |
| live evaluate | 每案例默认 12；可设 1–20 | 单轮最多 8；默认累计 24 |
| 离线 CLI/Web demo | 13 | 单轮最多 8；累计 26 |

- safe：CLI 会对普通命令展示精确 argv 并请求确认；当前 Web 没有审批 broker，因此会拒绝普通命令，但登记 verifier 仍可执行。
- auto：跳过普通命令确认，仍保留破坏性命令拒绝、路径边界、超时与资源限制。

safe 和 auto 都不是操作系统沙箱。命令从项目目录启动，只能说明起始 cwd 受控，不能隔离恶意代码；不可信仓库必须放入容器、虚拟机或低权限账户。

## 真实模型评测

评测默认断网。真实模型评测必须同时显式提供 live 和 allow-paid-api，不存在接收密钥的 CLI 参数：

~~~powershell
# 先运行一个案例
uv run coding-agent evaluate --live --allow-paid-api --case single-file

# 运行全部案例并输出稳定 JSON
uv run coding-agent evaluate --live --allow-paid-api --case all --format json --output evaluation.json
~~~

五类案例覆盖：

- single-file：单文件故障修复；
- cross-file：跨文件修改；
- new-feature：新增功能；
- indirect-debug：间接故障定位；
- runtime-integration：公开浅层测试难以发现的运行时集成问题。

每个案例都会在新的临时仓库中放置已知失败的公开测试。Agent 停止后，宿主只把白名单源文件复制到相邻目录，再运行独立 regression oracle；同时检查公开测试与 pytest 控制文件字节未变、必需文件和修改存在、全仓没有越界变更。仅退出码为 0 不足以通过，没有完整、非空、全绿的 oracle 也会失败。oracle 是与 Agent 运行分离的评分路径，但其定义随仓库发布，并非隐藏测试或保密边界。

评测退出码为：0 全部通过，3 真实任务或 oracle 失败，1 runner/provider/harness 错误，2 配置错误。套件遇到首个 error 后会停止，避免继续消耗付费请求。

2026-08-29 的历史实测中，deepseek-v4-flash 对原四类案例逐项运行及整套连续运行均通过，共 8/8 次、每案例 5 个模型步骤、0 个工具错误；四案例连续运行耗时 43.05 秒。随后新增的 runtime-integration 案例也完成一次 live 评测：1/1 通过、6 个模型步骤、0 个工具错误，耗时 38.90 秒。

## 架构

~~~text
CLI ───────────────────────────────────────────────┐
loopback Web ─ WebWorkbench                        │
                 ├─ ProjectRegistry + RunCatalog   │
                 ├─ ProjectMemoryCoordinator       │
                 │    └─ ProjectMemoryStore        │
                 └─ one WebRunService ─────────────┤
                                                   ▼
                                  Repository application service
                                    ├─ OpenAIResponsesModel / ScriptedModel
                                    ├─ AgentRunner ─ Context + Stop + Verification
                                    │   └─ ToolRegistry
                                    │       ├─ Workspace file/search tools
                                    │       ├─ CommandPolicy + process runner
                                    │       └─ MutationSession
                                    ├─ SessionStore + RunLease
                                    └─ CompositeEventSink
                                        ├─ JsonlEventSink
                                        └─ DashboardProjection

evaluate CLI
 └─ isolated public fixture → same application service
     └─ allowlisted source copy → sibling regression oracle
~~~

几个容易混淆的数据层被刻意分开：

| 数据 | 作用 | 不能做什么 |
|---|---|---|
| AgentResult | 描述本次执行结果 | 不能用模型文字伪造验证 |
| SessionStore | 保存可恢复 transcript 检查点 | 不能重放工具或恢复验证证据 |
| JSONL trace | 保存可审计事实 | 不能授权命令或验证 |
| DashboardProjection | 生成终端/Web 的有界视图 | 不能影响运行控制流 |
| RunMemory | 压缩同一运行的计划、变更与失败事实 | 不能跨项目授权 |
| ProjectMemory | 连接同项目的不同任务 | 不是检查点，也不能恢复可信状态 |

这种设计的核心原则是：**事实记录、模型上下文、恢复状态、项目记忆和界面展示互不冒充权威。**

## 可观察性与安全边界

Web 和终端只呈现模型的显式计划、外部动作和验证证据，不保存或展示隐式思维链。命令卡片可展开查看经过凭据值脱敏的 argv、工作目录、超时、退出状态、耗时和有界 stdout/stderr；运行历史会分别标注工具捕获截断、Web 投影截断和模型观察压缩。

需要注意：

- 密钥、完整环境变量、原始 provider payload 和隐式推理不会进入浏览器投影；
- 通用模式脱敏无法保证识别任意秘密，不要把密钥写入参数、仓库文件或程序输出；
- Web 仅监听 loopback，并使用进程级控制 token、Host/Origin/header 检查，但它不是远程或多用户产品；
- 强制结束进程或机器断电可能发生在工具修改之后、下一检查点写入之前，因此恢复不是 crash-safe exactly-once；
- Responses 适配器使用 store=False；当前不支持在工具调用轮同时依赖 provider 私有 reasoning state 的模型。

## 空仓库支持

Relay 可以在空目录中工作。模型需要逐层调用 create_directory，再通过 write_file 创建代码、测试与说明文件。目录创建不会覆盖已有路径，也不能越过工作区或策略保护范围。

空仓库没有天然的可信标准：模型可以自行补充测试并运行，但“自己写的测试通过”只能说明获得了检查证据。若普通任务需要 VERIFIED，用户必须预先提供不可被 Agent 修改的项目策略与登记 verifier；外部 oracle 只能在 evaluate 流程中独立评分，不能为普通运行授予 VERIFIED。

## 开发与质量检查

~~~powershell
uv run ruff check src tests
uv run mypy src tests
node --test tests/js/*.test.mjs
uv run pytest
~~~

pytest 配置要求分支覆盖率不低于 90%。CI 徽章反映 GitHub 上当前提交的自动检查结果。

## 文档索引

- [工作原理与源码导览](docs/HOW_IT_WORKS.md)：用完整任务解释模型、工具、上下文、验证、恢复与 Web 的因果链。
- [架构说明](docs/ARCHITECTURE.md)：模块职责、稳定接口与数据权威边界。
- [安全说明](docs/SECURITY.md)：威胁模型、命令权限、工作区边界与密钥处理。
- [项目计划](PLAN.md)：阶段目标、合规边界和演示策略。
- [参考资料](docs/REFERENCES.md)：协议与第三方库依据。
- [千字项目简介](README.txt)：提交材料所需的精简版说明。

## 实现边界

项目复用 openai、FastAPI、Pydantic、Rich、Typer、prompt-toolkit、python-dotenv、pathspec、psutil 与 pytest，分别承担 HTTP 传输、Web、数据校验、界面、输入、配置解析、忽略规则、进程枚举和测试等通用能力。

Agent 编排、上下文选择、工具定义与调度、命令策略、停止规则、检查点、运行记忆、项目记忆、可信验证、事件语义及独立评测流程均为项目自行实现。这样既避免重复制造通用基础设施，也确保考核要求中的核心 Agent 逻辑清晰、可读、可解释。
