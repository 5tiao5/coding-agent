Relay Coding Agent 项目说明

Git 仓库：https://github.com/5tiao5/coding-agent.git

一、如何运行
需 Python 3.11+、uv，在目录执行：

uv sync --locked --all-groups
uv run coding-agent demo
uv run coding-agent web --demo

以上为无需 API Key 和网络的离线演示。使用真实模型时，将 .env.example 复制为 .env.local 并填写配置。OPENAI_API_KEY 是统一变量名，不限制密钥厂商；服务端须兼容 OpenAI Responses API 和函数工具调用，仅支持 Chat Completions 或 Claude、Gemini 原生协议的服务不能直连。DeepSeek 推理强度设为 none；使用 OpenAI 官方服务时删除 OPENAI_BASE_URL。启动 Web 工作台：

uv run coding-agent web --root "<项目目录>" --mode auto --max-steps 50

也可在界面选择项目，或使用 CLI：

uv run coding-agent run "任务描述" --root "<项目目录>" --mode safe

二、特色功能
系统自行实现 Model–Tool–Observation 循环：模型决策，宿主负责工具校验、目录边界、文件修改、命令策略和停止判断。CLI 与中文 Web 共用核心，支持项目管理、命令详情、完整变更和 Diff、历史回放、中断恢复、继续任务及有界 ProjectMemory。文件变化后旧证据立即失效；只有当前版本通过登记验证器且受保护内容完整，Verification Gate 才授予“已验证”。隔离评测使用独立 oracle 防止虚假通过。

三、设计思想与边界
大模型是可替换的决策器，不直接执行；可信结论来自宿主事实与最新验证。项目复用 FastAPI、Pydantic、Rich 等通用库，Agent 编排、工具、上下文、检查点和验证机制自行实现。ProjectMemory 仅提供有界摘要，继续任务仍会重读代码并验证。auto 不是操作系统沙箱，不可信仓库应放入容器或虚拟机。
