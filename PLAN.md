# Coding Agent 项目计划

> 版本：v0.5.5 · 2026-08-31 · 截止：2026-09-02 24:00（北京时间）

> 当前进度：M5、M5.5 已完成；M5.6 已完成合同、耦合预算、聚合观察预算、可恢复运行记忆、可信项目策略、协作取消、结构拆分和第五个运行时集成评测，本地全量复验已通过。DeepSeek 旧四场景两轮共 8/8 通过；新增难例仍需 live 复测，之后进入演示三连测和视频彩排。

## 1. 目标

实现一个本地编程智能体：模型自主读写本地项目、执行命令、根据反馈继续工作，直到完成任务或触发明确的停止条件；CLI 与可选的 loopback Web 视图共享同一核心。

项目追求“小内核、强工程性”：核心 Agent 机制自行实现，通用基础能力复用成熟库；最终成果必须可运行、可测试、可追踪、可解释，并能在两分钟内清楚展示一个真实编程闭环。

## 2. 合规边界

- **自行实现**：Agent 循环、对话历史与上下文策略、工具定义与调度、模型输出映射、终止条件、权限决策、重试与错误语义。
- **允许复用**：模型 HTTP 客户端、数据校验、CLI/终端渲染、进程管理、`.gitignore` 匹配和测试框架。
- **不使用**：LangChain、LlamaIndex、OpenAI Agents SDK、Claude Agent SDK、AutoGen、CrewAI 等 Agent 框架；服务端代码执行、文件或上下文托管工具；现成 Coding Agent 的封装。
- 对处于灰区、会隐藏核心逻辑的高级抽象库（如通用 Agent/Tool 编排层）默认不用。
- API key 只从环境变量或未入库配置读取；运行轨迹默认不入库且采用最小化字段，但计划和 Diff 仍可能含源码，须保存在私有状态目录。若密钥误提交，立即作废更换。

## 3. 交付范围

### P0：核心闭环

- 单次任务与交互式 CLI。
- OpenAI-compatible 模型适配器，以及用于测试的 `ScriptedModel`。
- 工具：`list_files`、`read_file`、`search_text`、`write_file`、`replace_text`、`run_command`。
- 手写模型—工具—观察循环；工具失败作为结构化观察返回模型。
- 定义稳定的 `RunEvent` 事件模型和最小文本渲染器；状态变化、工具调用和验证结果从第一天即可观察。
- 工作区路径隔离、命令超时、输出截断、最大步数、重复调用检测和用户中断。
- 上下文预算：保留任务与近期动作，压缩旧工具输出，避免上下文无限增长。

### P1：承诺完成的增强能力

- 事件驱动的 Rich 终端界面：事件级实时反馈、计划状态、工具时间线、折叠输出、Diff 和“修改范围 → 当前修订验证”证据卡片。
- 结构化任务计划：维护待办、进行中和已完成状态，但不记录模型隐式思维过程。
- Verification Gate：最后一次写操作之后没有成功测试、构建或检查证据时，只能报告“完成但未验证”。
- 可信项目策略：仅解析根目录 `.coding-agent/project.toml` 的 typed verifier、显式解释器和保护清单；未配置项目即使 pytest 通过也只记为 `checks_only`，验证前后完整性漂移一律拒绝证据。
- `safe` / `auto` 两档策略、精确 argv 交互确认和 CLI 模式选择。
- 被动 transcript checkpoint/load/resume、工作区绑定、完成轨迹拒绝和并发恢复 lease；工具局部状态不跨进程恢复。
- 写文件前生成 Diff，并支持本次会话内撤销。
- JSONL 事件轨迹与 `inspect <run-id>`：记录工具调用、结果、耗时、错误和停止原因；不记录模型隐式思维过程。
- M5.5 本地项目工作台：绑定 loopback、Windows 原生文件夹选择与手动路径兜底、全局单后台任务、项目登记/安全空目录创建、左侧项目与新运行历史、可信轨迹回放；复用同一 application service，只向浏览器提供白名单状态投影。
- 自动评测：单文件修复、跨文件修改、新增功能、间接故障调试和运行时集成五类任务；每类使用临时公开红测、受保护测试清单、全仓前后 manifest、显式变更路径白名单、白名单源码复制和工作区外独立回归 oracle，统计成功率、步数、耗时、工具错误和“Agent 声称已验证但 oracle 失败”的假阳性。oracle 定义随项目发布，这是评分路径隔离而非秘密 benchmark 边界。

### P2：时间允许再做

- 第二种模型协议适配器。
- 只读工具调用的安全并发。
- Python 符号/AST 摘要工具。

### 非目标

完整的远程/多用户 Web 产品、打包桌面壳、Git clone/初始化与模板、浏览器终端或编辑器、Web 审批 broker、多 Agent、向量数据库、MCP、插件市场、远程沙箱和 IDE 集成不进入本次范围。M5.5 仍是 `127.0.0.1` 上的本地单-worker 工作台。

## 4. 架构

```text
CLI / loopback Web
 ├─ WebWorkbench（项目注册表 + 运行目录 + 单 worker）
 └─ Repository application service
     └─ AgentRunner（状态机与循环）
         ├─ ModelAdapter（供应商协议隔离）
         ├─ BudgetPolicy + ContextManager（耦合预算、聚合裁剪与压缩）
         ├─ RunMemory（计划、变更、失败命令与过期验证事实）
         ├─ Stop / CompletionContract / Verification Policy
         ├─ ToolRegistry（校验与执行）
         │   ├─ File/Search Tools
         │   └─ Command Tool
         ├─ Policy（路径、命令能力、资源限制）
         ├─ SessionStore（保存与恢复）
         └─ EventSink
             ├─ JsonlEventSink（持久轨迹与 inspect）
             ├─ DashboardEventSink（Rich 时间线与验证卡）
             └─ WebRunService（实时白名单投影）

evaluate CLI
 └─ 临时公开 fixture → 同一 Repository application service
     └─ 白名单源码复制 → sibling regression oracle
```

内部边界采用少量稳定接口：`ModelAdapter.complete()`、`ToolDispatcher.execute()`、`CommandPolicy.classify()`、`EventSink.emit()`。事件轨迹是事实记录，上下文是从事实中选择出的模型视图，两者分离。

```text
CREATED → PLANNING → ACTING ↔ OBSERVING → VERIFYING
                                      ├→ COMPLETED
                                      ├→ COMPLETED_UNVERIFIED
                                      └→ FAILED
```

工具调用默认顺序执行，因为文件修改具有顺序依赖；只有证明无副作用的只读工具才考虑并发。

## 5. 演示策略

演示效果采用“前置设计、后置打磨”：事件语义、任务叙事和验收标准从核心阶段确定；颜色、动画、录屏和剪辑在功能稳定后完成。

演示任务必须满足：真实失败测试；涉及一至两个文件；评委十秒内能理解目标；完整运行控制在约 90 秒；固定模型与配置连续三次成功。视频展示真实运行，可按题目要求剪辑或加速，不用预录回放冒充执行。

两分钟叙事固定为：说明任务 → Agent 制定计划 → 搜索与阅读 → 修改并展示 Diff → 执行验证 → 展示 `VERIFIED` 与统计。界面不倾倒原始 JSON，不展示隐式思维链，只呈现意图摘要、外部动作和验证证据。

## 6. 依赖策略

| 依赖 | 用途 | 不由它负责的内容 |
|---|---|---|
| `openai` | 原始 Responses 请求与原生 function calling 传输 | 自动工具执行、托管工具、重试决策、上下文 |
| `pydantic` | 配置、工具参数和事件校验 | 工具调度与业务规则 |
| `typer`、`rich` | 当前 CLI 与事件渲染 | Agent 状态机与对话历史 |
| `prompt-toolkit` | 交互式任务输入 | Agent 循环与权限决策 |
| `python-dotenv` | 解析固定且未入库的 `.env.local` | 配置搜索、密钥存储与 Agent 编排 |
| `fastapi`、`uvicorn` | 本地 loopback 路由、静态资源与进程托管 | Agent 循环、工具执行、远程服务或多用户控制 |
| `pathspec` | 尊重 `.gitignore` | 工作区安全边界 |
| `psutil` | 枚举和终止进程树 | 命令启动、授权、超时与错误语义 |
| `pytest` | 单元/集成测试、离线演示和默认验证能力 | Agent 决策与验证判定 |

依赖统一由 `uv` 管理并锁定版本；API key 仅通过环境变量或未入库配置提供。

## 7. 里程碑与提交

| 日期 | 里程碑 | 状态 | 主要验收 |
|---|---|---|---|
| 8/27 | M0 需求与设计 | 完成 | 合规审查、架构、范围和验收标准落档 |
| 8/28 | M1 骨架与核心循环 | 完成 | `ScriptedModel`、`RunEvent` 驱动离线多轮工具调用 |
| 8/29 | M2 工具与安全 | 完成 | Agent 在 fixture 中定位 Bug、原子修改并支持撤销 |
| 8/30 | M3 上下文与验证 | 完成 | 命令、计划、压缩、停止、被动恢复和 Verification Gate 测试通过 |
| 8/31 | M4 Demo UX 与轨迹 | 完成 | 真实模型 CLI、审批、时间线、Diff、恢复、`inspect` 和验证卡片可完整演示 |
| 8/31 | M5 评测与冻结 | 完成 | M5-UX、可观察重试和四类外置 oracle 评测已实现并推送；DeepSeek 两轮 8/8 通过 |
| 9/1 | M5.5 项目工作台 | 完成 | 项目选择/安全新建、左侧历史、可信回放和运行期根目录冻结已实现并完成多轮手测 |
| 9/1–9/2 | M5.6 可靠性加固 | 进行中 | 本地实现、结构拆分与全量复验已完成；待新难例 live 复测与最终冻结 |
| 9/2 | 最终交付 | 待开始 | 安全检查、录制视频、整理材料并在截止前完成最终推送 |

按可解释的功能单元小步提交并保留历史，不 squash、不改写已推送提交；截止后不再 push，包括文档修正和 tag。

## 8. 完成与提交标准

- Agent 能在隔离的示例仓库中发现失败测试、定位代码、修改并验证全部测试通过。
- 最终状态区分“已验证完成”和“完成但未验证”，模型的口头声明不能替代外部验证证据。
- 无 API 的单元测试覆盖循环、工具、权限、停止和上下文；真实 API 测试通过显式开关运行。
- 路径穿越、重复工具调用、命令超时、超长输出和模型暂时失败均有确定行为。
- 演示任务连续三次稳定完成；观众无需阅读原始日志即可看懂计划、动作、修改和验证结果。
- 公开仓库确认在题目发布后创建；仓库 `README.md` 讲清运行、设计与依赖边界。
- 提交用 `README.txt` 不超过 1000 汉字，包含仓库地址、运行方法和特色功能。
- 视频展示真实编程任务并简述实现，MP4 格式、两分钟以内、不超过 200 MB。
- 最终仅提交 `<姓名>.zip`，其中只含 `README.txt` 和视频；提交前完成密钥与历史扫描。
- 将 ZIP 提交至题目指定表单；允许重复提交，以最后一次为准。
