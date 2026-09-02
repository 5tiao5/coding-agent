Relay Coding Agent 项目说明

Git 仓库：https://github.com/5tiao5/coding-agent.git

一、运行方法
环境要求：Python 3.11+、uv。在源码目录执行：

uv sync --locked --all-groups
uv run coding-agent demo
uv run coding-agent web --demo

以上为无需 API Key 和网络的离线演示。使用真实模型前，将 .env.example 复制为同目录的 .env.local，填写 OPENAI_API_KEY、OPENAI_BASE_URL、CODING_AGENT_MODEL 和 CODING_AGENT_REASONING_EFFORT；DeepSeek 无状态工具循环的推理强度设为 none。启动中文 Web 工作台：

uv run coding-agent web --root "<项目目录>" --mode auto --max-steps 50

也可在界面选项目，或用 CLI：

uv run coding-agent run "任务描述" --root "<项目目录>" --mode safe

二、特色功能
系统自行实现 Model–Tool–Observation 循环：模型提出决策，宿主负责工具校验、工作区边界、修改、命令策略和停止判断。CLI 与中文 Web 共用核心，支持项目管理、过程展示、完整变更与 Diff、历史回放、中断恢复、继续任务和有界 ProjectMemory。文件变化后旧证据立即失效；只有当前版本通过登记验证器，且受保护内容保持完整，宿主 Verification Gate 才授予“已验证”。隔离评测还可用独立 oracle 避免虚假通过。

三、设计思想与边界
大模型是可替换的决策器，不直接拥有执行权；可信结论来自宿主事实和最新验证。项目复用 FastAPI、Pydantic、Rich 等通用库，但 Agent 编排、工具、上下文、检查点和验证机制均自行实现。ProjectMemory 只提供不可信的有界摘要，继续任务仍会重读代码并验证。auto 不是操作系统沙箱，不可信仓库应放入容器或虚拟机。
