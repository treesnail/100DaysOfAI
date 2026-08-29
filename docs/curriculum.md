# 100 天 AI 学习计划课程索引

> 主线项目：智研 AI 助手（SmartResearch Agent）
>
> 总周期：2026-07-27（周一）~ 2026-11-03（周二），共 100 个日历日  
> 学习日：86 天；休息日/缓冲日：14 天（每周日休息）  
> 每日学习时长：约 2 小时

## 阶段总览

| 阶段 | 学习日编号 | 学习天数 | 核心目标 |
|------|-----------|---------|---------|
| P 预备阶段 | P1 ~ P2 | 2 | Python 工程化速通 + 环境搭建 + 项目初始化 |
| M1 智能体应用开发 | M1-D1 ~ M1-D11 | 11 | Agent 架构、ReAct、工具调用、记忆、规划、多 Agent、LangGraph、安全、评估 |
| M2 MCP 应用开发 | M2-D1 ~ M2-D7 | 7 | MCP 协议、Resources/Tools/Prompts、FastMCP 开发、Docker 部署 |
| R1 复习缓冲日 | R1 | 1 | M1 ~ M2 复习、补做练习、整理笔记 |
| M3 Harness 评估框架 | M3-D1 ~ M3-D7 | 7 | AI 评估框架、Prompt/RAG/Agent 评估、安全红队、CI/CD、可观测性 |
| M4 大模型应用开发 | M4-D1 ~ M4-D12 | 12 | LLM 原理、Tokenization、Prompt 工程、Function Calling、FastAPI 后端、多模态、Embedding、成本优化、安全、开源部署 |
| R2 复习缓冲日 | R2 | 1 | M3 ~ M4 复习、补做练习、整理笔记 |
| M5 大模型微调 | M5-D1 ~ M5-D11 | 11 | SFT、LoRA/QLoRA、RLHF/DPO、数据工程、评估、MLOps、持续微调 |
| M6 RAG 应用开发 | M6-D1 ~ M6-D10 | 10 | 文档解析、分块策略、向量数据库、混合检索、重排序、高级 RAG、评估、生产化部署 |
| 数学基础过渡 | Math-D1 ~ Math-D2 | 2 | 线性代数 + 概率论、微积分 + 优化基础（为 Transformer/深度学习预热） |
| M7 Transformer 架构 | M7-D1 ~ M7-D12 | 12 | Attention 机制、多头注意力、位置编码、Encoder/Decoder、从零实现、训练优化、变体架构、可解释性、源码精读 |
| M8 深度学习（简化） | M8-D1 ~ M8-D7 | 7 | 神经网络基础 + 反向传播、优化器、CNN、RNN/LSTM、训练技巧、PyTorch 高级、综合实践 |
| R3 复习缓冲日 | R3 | 1 | M7 ~ M8 复习、全栈知识串联 |
| 结业项目 | G1 ~ G2 | 2 | 整合所学完成综合项目、项目文档、学习总结、能力自评、后续规划 |

## 每日课程索引

| 日历日 | 日期 | 星期 | 阶段标签 | 主题标题 | 学习目标 | 前置依赖 |
|--------|------|------|----------|----------|----------|----------|
| day001 | 07/27 | 周一 | P1 | Python 工程化基础（一） | - 搭建 smart-research-agent 目录结构并初始化 Python 包<br>- 掌握虚拟环境、requirements.txt 与 pyproject.toml 的使用<br>- 配置 ruff / black / mypy / pre-commit 代码规范流水线<br>- 使用 Pydantic Settings + `.env` 管理配置，并编写 logging 基础模块 | 无 |
| day002 | 07/28 | 周二 | P2 | Python 工程化基础（二） | - 建立 Git 工作流与分支管理规范<br>- 编写 pytest 单元测试并配置覆盖率检查<br>- 编写一键环境初始化脚本 `scripts/setup_env.sh` 与环境验证脚本<br>- 完成项目 README、docs 目录与 Docker 基础镜像准备 | day001 |
| day003 | 07/29 | 周三 | M1-D1 | Agent 概述与 ReAct 原理 | - 理解 Agent 的核心组件（感知、推理、行动、记忆）<br>- 掌握 ReAct 推理-行动协同循环的设计思想<br>- 在 SmartResearch Agent 中设计首个 `ReactAgent` 类骨架 | day002 |
| day004 | 07/30 | 周四 | M1-D2 | 工具抽象基类与计算器工具 | - 设计可扩展的 `BaseTool` 抽象基类（名称、描述、参数、执行接口）<br>- 实现 `CalculatorTool` 并接入 Agent<br>- 理解工具调用对 Agent 能力边界的扩展作用 | day003 |
| day005 | 07/31 | 周五 | M1-D3 | 工具注册与 LLM 调用层 | - 实现工具注册表 `ToolRegistry`，支持动态加载与校验<br>- 搭建 LLM 调用层基类与 OpenAI 兼容实现<br>- 完成 Agent 对工具列表的自动发现与描述生成 | day004 |
| day006 | 08/01 | 周六 | M1-D4 | ReAct Agent 循环实现 | - 实现完整的 ReAct 推理循环：Thought → Action → Observation<br>- 解析 LLM 输出并调用对应工具<br>- 对 SmartResearch Agent 执行端到端单步推理测试 | day005 |
| day007 | 08/02 | 周日 | 休息日/缓冲日 | 休息日 / 缓冲复习 | - 复习 P 阶段与 M1-D1~D4 的核心概念<br>- 补做本周练习并整理笔记 | 前一天学习内容 |
| day008 | 08/03 | 周一 | M1-D5 | 短期记忆与会话管理 | - 设计短期记忆存储结构（消息列表、最大长度、截断策略）<br>- 为 Agent 添加多轮对话上下文管理能力<br>- 实现会话级别的记忆隔离 | day006 |
| day009 | 08/04 | 周二 | M1-D6 | 长期记忆与向量检索 | - 使用 ChromaDB/FAISS 构建长期记忆向量库<br>- 实现历史对话与关键知识的向量化存储与检索<br>- 将检索结果注入 Agent 上下文 | day008 |
| day010 | 08/05 | 周三 | M1-D7 | 规划模块 Planner | - 理解任务分解与规划在 Agent 中的作用<br>- 实现基于 LLM 的 `Planner` 模块，将用户目标拆分为可执行子任务<br>- 在 SmartResearch Agent 中接入规划前置步骤 | day009 |
| day011 | 08/06 | 周四 | M1-D8 | 反思模块 Reflexion | - 理解自我反思在提升 Agent 可靠性中的作用<br>- 实现 Reflexion 机制：失败检测、原因总结、改进建议<br>- 将反思结果反馈到后续推理循环 | day010 |
| day012 | 08/07 | 周五 | M1-D9 | 多 Agent 协作 | - 设计研究员、分析师、撰稿人三类 Agent 角色<br>- 实现 Agent 间的消息传递与结果交接<br>- 在 SmartResearch Agent 中完成多 Agent 协作初版 | day011 |
| day013 | 08/08 | 周六 | M1-D10 | LangGraph 状态图编排 | - 理解 LangGraph 的状态、节点与边模型<br>- 将 ReAct / 多 Agent 流程重构为 LangGraph 状态图<br>- 实现状态持久化与断点调试能力 | day012 |
| day014 | 08/09 | 周日 | 休息日/缓冲日 | 休息日 / 缓冲复习 | - 复习 M1 Agent 架构、记忆、规划与 LangGraph 编排<br>- 整理多 Agent 协作示例与笔记 | 前一天学习内容 |
| day015 | 08/10 | 周一 | M1-D11 | Agent 安全防护与评估追踪 | - 实现输入侧 Prompt 注入检测与输出侧内容审核<br>- 为 Agent 添加工具权限控制与执行审计日志<br>- 设计 Agent 评估追踪指标（成功率、步数、延迟） | day013 |
| day016 | 08/11 | 周二 | M2-D1 | MCP 协议与架构 | - 理解 MCP（Model Context Protocol）协议的核心概念<br>- 区分 Resources、Tools、Prompts 三类能力<br>- 规划 SmartResearch Agent 的 MCP Server 边界 | day015 |
| day017 | 08/12 | 周三 | M2-D2 | MCP Resources | - 使用 FastMCP 实现 Resources 接口（知识库文档、配置文件）<br>- 掌握 Resource URI、订阅与读取机制<br>- 将 SmartResearch Agent 的静态资源接入 MCP Server | day016 |
| day018 | 08/13 | 周四 | M2-D3 | MCP Tools 与 Prompts | - 将现有工具重构为 MCP Tool 并注册到 Server<br>- 实现 MCP Prompt 模板，支持动态参数填充<br>- 在 Client 侧调用 MCP Server 的工具与提示词 | day017 |
| day019 | 08/14 | 周五 | M2-D4 | FastMCP Server 开发 | - 使用 FastMCP 框架完整搭建 MCP Server<br>- 配置 Server 元数据、能力声明与错误处理<br>- 实现本地 stdio 与 SSE 两种传输方式 | day018 |
| day020 | 08/15 | 周六 | M2-D5 | MCP Client 集成 | - 实现 MCP Client 的连接、初始化与能力发现<br>- 将 MCP 工具动态接入 SmartResearch Agent 的工具调用层<br>- 处理工具调用结果与异常回退 | day019 |
| day021 | 08/16 | 周日 | 休息日/缓冲日 | 休息日 / 缓冲复习 | - 复习 MCP 协议、Resources/Tools/Prompts 与 Client 集成<br>- 补做 M2 阶段练习 | 前一天学习内容 |
| day022 | 08/17 | 周一 | M2-D6 | Docker 部署 MCP Server | - 编写 MCP Server 的 Dockerfile 与 docker-compose 配置<br>- 实现容器内环境变量注入与健康检查<br>- 完成本地 Docker 构建与运行验证 | day020 |
| day023 | 08/18 | 周二 | M2-D7 | MCP 协议测试与文档 | - 编写 MCP Server/Client 的单元测试与集成测试<br>- 生成 MCP 能力文档（resources、tools、prompts 清单）<br>- 完成 M2 阶段项目代码整理与提交 | day022 |
| day024 | 08/19 | 周三 | R1 | M1 ~ M2 复习缓冲日 | - 回顾 Agent 与 MCP 的核心实现，修复遗留问题<br>- 补做未完成的练习与测试用例<br>- 整理阶段学习笔记与项目文档 | day023 |
| day025 | 08/20 | 周四 | M3-D1 | 评估指标与 Harness 框架 | - 理解 AI 系统评估的指标体系（正确性、相关性、安全性、成本）<br>- 设计 SmartResearch Agent 的 `EvaluationHarness` 基类<br>- 实现评估数据集加载与结果汇总 | day024 |
| day026 | 08/21 | 周五 | M3-D2 | Prompt 评估 | - 实现 Prompt 质量评估（清晰度、一致性、潜在歧义）<br>- 使用 LLM-as-a-judge 对系统提示词打分<br>- 将评估结果反馈到 Prompt 模板优化 | day025 |
| day027 | 08/22 | 周六 | M3-D3 | 输出质量评估 | - 设计输出评估指标（有用性、准确性、格式合规）<br>- 实现基于规则与模型打分的双重评估<br>- 为 SmartResearch Agent 添加输出质量日志 | day026 |
| day028 | 08/23 | 周日 | 休息日/缓冲日 | 休息日 / 缓冲复习 | - 复习 M3 评估框架与 Prompt/输出评估方法<br>- 整理评估用例与数据集 | 前一天学习内容 |
| day029 | 08/24 | 周一 | M3-D4 | RAG 评估 | - 实现 RAG 检索评估（召回率、精确率、MRR、NDCG）<br>- 实现生成答案与参考答案的对比评估<br>- 将 RAG 评估接入 SmartResearch Agent 知识库模块 | day027 |
| day030 | 08/25 | 周二 | M3-D5 | Agent 评估 | - 设计 Agent 任务完成率、步数效率与工具使用正确性指标<br>- 实现端到端 Agent 评估用例<br>- 将评估结果可视化并定位薄弱环节 | day029 |
| day031 | 08/26 | 周三 | M3-D6 | 安全红队评估 | - 构建 Prompt 注入、越狱、PII 泄露等攻击用例集合<br>- 实现安全评估器并执行红队测试<br>- 根据测试结果加固 SmartResearch Agent 安全模块 | day030 |
| day032 | 08/27 | 周四 | M3-D7 | CI/CD 与可观测性 | - 配置 GitHub Actions 自动化测试与评估流水线<br>- 接入日志、指标与追踪（cost_tracker、metrics）<br>- 实现评估报告自动生成与失败告警 | day031 |
| day033 | 08/28 | 周五 | M4-D1 | 大模型原理与 API 调用 | - 理解 GPT 类模型的自回归生成原理<br>- 掌握 OpenAI 兼容 API 的参数（temperature、top_p、max_tokens）<br>- 为 SmartResearch Agent 添加多模型路由能力 | day032 |
| day034 | 08/29 | 周六 | M4-D2 | Tokenization 基础 | - 理解 BPE、WordPiece、SentencePiece 等分词算法<br>- 使用 tiktoken 计算 token 数量并预估成本<br>- 在 SmartResearch Agent 中实现 token 计数工具 | day033 |
| day035 | 08/30 | 周日 | 休息日/缓冲日 | 休息日 / 缓冲复习 | - 复习 M3 评估体系与 M4 大模型原理<br>- 整理 Tokenization 实验笔记 | 前一天学习内容 |
| day036 | 08/31 | 周一 | M4-D3 | Prompt 工程 | - 掌握 Zero-shot / Few-shot / CoT / ToT 等 Prompt 技巧<br>- 为 SmartResearch Agent 编写高质量系统提示词与任务模板<br>- 使用 Prompt 版本管理记录变更 | day034 |
| day037 | 09/01 | 周二 | M4-D4 | Function Calling | - 理解 Function Calling 的机制与参数 schema 设计<br>- 将 Agent 工具调用迁移到 LLM 原生 function calling 接口<br>- 处理函数调用失败与重试逻辑 | day036 |
| day038 | 09/02 | 周三 | M4-D5 | FastAPI 服务搭建 | - 使用 FastAPI 搭建 SmartResearch Agent 后端服务<br>- 设计 `/chat`、`/agent/run`、`/health` 等核心接口<br>- 接入 Pydantic 模型进行请求/响应校验 | day037 |
| day039 | 09/03 | 周四 | M4-D6 | 流式输出与多模型路由 | - 实现 SSE 流式输出，提升用户体验<br>- 设计多模型路由策略（按能力、成本、可用性）<br>- 为 FastAPI 接口添加模型切换能力 | day038 |
| day040 | 09/04 | 周五 | M4-D7 | 多模态输入 | - 理解视觉-语言模型输入格式<br>- 实现图像分析工具 `ImageAnalysisTool` 并接入 Agent<br>- 在 FastAPI 中支持图片上传与描述接口 | day039 |
| day041 | 09/05 | 周六 | M4-D8 | Embedding 与向量表示 | - 理解 Embedding 的原理与应用场景<br>- 使用 SentenceTransformers 生成文本/查询向量<br>- 为 SmartResearch Agent 替换/升级 Embedding 模块 | day040 |
| day042 | 09/06 | 周日 | 休息日/缓冲日 | 休息日 / 缓冲复习 | - 复习 FastAPI、Function Calling、多模态与 Embedding<br>- 补做 M4 前半段练习 | 前一天学习内容 |
| day043 | 09/07 | 周一 | M4-D9 | 成本追踪与优化 | - 实现 `CostTracker`：按模型、按接口统计 token 与费用<br>- 设计缓存策略降低重复调用成本<br>- 输出成本报告并识别高成本调用路径 | day041 |
| day044 | 09/08 | 周二 | M4-D10 | 安全合规与内容审核 | - 实现输出内容审核与敏感词过滤<br>- 添加 PII 检测与脱敏处理<br>- 为 FastAPI 接口配置访问日志与审计 | day043 |
| day045 | 09/09 | 周三 | M4-D11 | 开源模型本地部署 | - 使用 Ollama / vLLM 本地部署开源模型<br>- 实现 `LocalModel` 调用层并与云端模型统一接口<br>- 评估本地模型在 SmartResearch Agent 上的效果 | day044 |
| day046 | 09/10 | 周四 | M4-D12 | 大模型应用集成与优化 | - 将 M4 所学整合到 SmartResearch Agent 主线<br>- 完成 Prompt、模型路由、成本、安全的一体化调优<br>- 编写 M4 阶段集成测试与性能基线 | day045 |
| day047 | 09/11 | 周五 | R2 | M3 ~ M4 复习缓冲日 | - 回顾评估框架与大模型应用开发全流程<br>- 修复集成测试中的问题并补充文档 | day046 |
| day048 | 09/12 | 周六 | M5-D1 | 微调概览与数据工程 | - 理解 SFT、LoRA、RLHF/DPO 的适用场景<br>- 准备领域问答数据集并清洗格式（JSONL / alpaca）<br>- 设计 SmartResearch Agent 专属数据的收集流程 | day047 |
| day049 | 09/13 | 周日 | 休息日/缓冲日 | 休息日 / 缓冲复习 | - 复习 M4 集成内容与 M5 微调基础<br>- 整理数据集准备经验 | 前一天学习内容 |
| day050 | 09/14 | 周一 | M5-D2 | SFT 监督微调基础 | - 使用 Transformers + Trainer 编写 SFT 训练脚本<br>- 配置训练参数（学习率、batch size、epoch）<br>- 在样本数据上完成一次 SFT 训练流程 | day048 |
| day051 | 09/15 | 周二 | M5-D3 | LoRA / QLoRA 微调 | - 理解 LoRA 低秩适配原理与参数高效微调优势<br>- 使用 PEFT 配置 LoRA/QLoRA 并挂载到基座模型<br>- 对比全参数微调与 LoRA 的资源占用 | day050 |
| day052 | 09/16 | 周三 | M5-D4 | 训练脚本与 PEFT 实践 | - 编写完整的 LoRA SFT 训练脚本（数据加载、训练、保存）<br>- 使用 Accelerate 支持多卡/混合精度训练<br>- 保存适配器权重并合并为可部署模型 | day051 |
| day053 | 09/17 | 周四 | M5-D5 | 微调模型评估 | - 设计领域评估指标与测试集<br>- 使用 Harness 对微调后模型进行自动评估<br>- 对比微调前后在 SmartResearch Agent 任务上的表现 | day052 |
| day054 | 09/18 | 周五 | M5-D6 | RLHF / DPO 原理 | - 理解 RLHF（PPO）与 DPO 的对齐目标<br>- 掌握偏好数据格式（chosen / rejected）<br>- 规划 SmartResearch Agent 输出风格的偏好对齐策略 | day053 |
| day055 | 09/19 | 周六 | M5-D7 | DPO 对齐实践 | - 使用 TRL 的 DPOTrainer 完成直接偏好优化<br>- 构造领域偏好数据集并训练<br>- 评估对齐后模型的安全性与有用性 | day054 |
| day056 | 09/20 | 周日 | 休息日/缓冲日 | 休息日 / 缓冲复习 | - 复习 SFT、LoRA、DPO 训练流程<br>- 整理训练日志与评估结果 | 前一天学习内容 |
| day057 | 09/21 | 周一 | M5-D8 | 领域数据准备与增强 | - 设计领域数据 pipeline：采集、清洗、去重、质量过滤<br>- 使用数据增强技术扩充训练样本<br>- 为 SmartResearch Agent 构建可持续更新的领域数据集 | day055 |
| day058 | 09/22 | 周二 | M5-D9 | 持续微调与版本管理 | - 设计增量微调与版本回滚策略<br>- 使用模型版本管理记录基座、适配器与合并模型<br>- 实现微调触发条件与自动化重训练流程 | day057 |
| day059 | 09/23 | 周三 | M5-D10 | MLOps 微调流水线 | - 搭建微调训练、评估、部署的端到端流水线<br>- 配置实验追踪（如 MLflow / Weights & Biases 基础用法）<br>- 将微调流水线集成到 GitHub Actions | day058 |
| day060 | 09/24 | 周四 | M5-D11 | 专属模型部署替换 | - 将微调后的专属模型部署到 SmartResearch Agent<br>- 修改 LLM 调用层支持本地/专属模型切换<br>- 完成端到端效果验证与成本对比 | day059 |
| day061 | 09/25 | 周五 | M6-D1 | 文档解析与加载 | - 实现 PDF、Markdown、TXT、Word 等文档解析<br>- 设计 `DocumentLoader` 统一接口<br>- 将解析结果接入 SmartResearch Agent 知识库 | day060 |
| day062 | 09/26 | 周六 | M6-D2 | 分块策略 | - 理解固定长度、递归、语义、结构等分块方法<br>- 实现 `Chunker` 模块并支持多种策略切换<br>- 评估不同分块对检索效果的影响 | day061 |
| day063 | 09/27 | 周日 | 休息日/缓冲日 | 休息日 / 缓冲复习 | - 复习 M5 微调与 M6 文档解析/分块<br>- 补做相关练习 | 前一天学习内容 |
| day064 | 09/28 | 周一 | M6-D3 | 向量数据库 ChromaDB / FAISS | - 掌握 ChromaDB 与 FAISS 的索引与查询 API<br>- 实现向量存储的增删改查与持久化<br>- 为 SmartResearch Agent 选择默认向量后端 | day062 |
| day065 | 09/29 | 周二 | M6-D4 | Embedding 索引构建 | - 将文档分块批量编码为 Embedding 向量<br>- 构建可增量更新的索引<br>- 实现索引版本控制与备份 | day064 |
| day066 | 09/30 | 周三 | M6-D5 | 检索器实现 | - 实现基础向量检索器（Top-K 相似度搜索）<br>- 支持过滤条件（metadata、时间范围）<br>- 将检索器接入 RAG 生成链路 | day065 |
| day067 | 10/01 | 周四 | M6-D6 | 混合检索 | - 结合向量检索与关键词检索（BM25）<br>- 实现检索结果融合与去重策略<br>- 在 SmartResearch Agent 中启用混合检索 | day066 |
| day068 | 10/02 | 周五 | M6-D7 | 重排序 Reranker | - 理解重排序在提升检索质量中的作用<br>- 实现基于交叉编码器的 `Reranker` 模块<br>- 对混合检索结果进行重排序并评估提升 | day067 |
| day069 | 10/03 | 周六 | M6-D8 | RAG 生成器与 Prompt | - 设计 RAG 专用 Prompt（context、question、约束）<br>- 实现 `RAGGenerator` 并接入 LLM<br>- 处理检索不到结果时的回退策略 | day068 |
| day070 | 10/04 | 周日 | 休息日/缓冲日 | 休息日 / 缓冲复习 | - 复习 RAG 检索、重排序与生成链路<br>- 整理 RAG 实验数据 | 前一天学习内容 |
| day071 | 10/05 | 周一 | M6-D9 | RAG 评估与调试 | - 使用 M3 评估器对 RAG 进行端到端评估<br>- 分析 bad case 并优化分块/检索/生成环节<br>- 记录 RAG 性能基线 | day069 |
| day072 | 10/06 | 周二 | M6-D10 | RAG 生产化部署 | - 实现 RAG 索引的定时更新与增量同步<br>- 配置 Docker Compose 部署向量库与服务<br>- 完成 RAG 模块文档与监控接入 | day071 |
| day073 | 10/07 | 周三 | Math-D1 | 线性代数 + 概率论基础 | - 掌握向量、矩阵、点积、注意力分数的数学含义<br>- 理解概率分布、期望、条件概率在 LLM 输出中的应用<br>- 为 Transformer 与深度学习公式推导打基础 | day072 |
| day074 | 10/08 | 周四 | Math-D2 | 微积分 + 优化基础 | - 理解导数、梯度、链式法则与反向传播的关系<br>- 掌握梯度下降、学习率、损失函数的基本概念<br>- 将数学直觉关联到模型训练过程 | day073 |
| day075 | 10/09 | 周五 | M7-D1 | Attention 机制原理 | - 理解 Query / Key / Value 与注意力权重的计算<br>- 实现 Scaled Dot-Product Attention<br>- 将 Attention 与 SmartResearch Agent 的检索相关性做类比 | day074 |
| day076 | 10/10 | 周六 | M7-D2 | 多头注意力 | - 理解多头注意力如何并行捕捉不同子空间信息<br>- 使用 PyTorch 实现 Multi-Head Attention<br>- 对比单头与多头在表达能力上的差异 | day075 |
| day077 | 10/11 | 周日 | 休息日/缓冲日 | 休息日 / 缓冲复习 | - 复习数学基础与 Attention 机制<br>- 补做推导练习 | 前一天学习内容 |
| day078 | 10/12 | 周一 | M7-D3 | 位置编码 | - 理解为什么 Transformer 需要位置信息<br>- 实现正弦/余弦位置编码与可学习位置编码<br>- 分析位置编码对序列建模的影响 | day076 |
| day079 | 10/13 | 周二 | M7-D4 | Encoder / Decoder 结构 | - 理解 Transformer 的 Encoder-Decoder 架构与各自作用<br>- 掌握残差连接、LayerNorm、Feed-Forward 子层<br>- 绘制 SmartResearch Agent 底层模型结构简图 | day078 |
| day080 | 10/14 | 周三 | M7-D5 | 从零实现 Transformer 块 | - 使用 PyTorch 组装 Multi-Head Attention、FFN、LayerNorm<br>- 实现可堆叠的 Transformer Block<br>- 验证前向传播输出形状 | day079 |
| day081 | 10/15 | 周四 | M7-D6 | 训练优化与正则化 | - 理解 Dropout、学习率调度、权重初始化<br>- 实现训练循环并监控损失下降<br>- 应用梯度裁剪与早停策略 | day080 |
| day082 | 10/16 | 周五 | M7-D7 | 变体架构（BERT / GPT / T5） | - 对比 Encoder-only、Decoder-only、Encoder-Decoder 变体<br>- 理解 BERT、GPT、T5 的设计差异与适用任务<br>- 将变体知识关联到 SmartResearch Agent 使用的模型 | day081 |
| day083 | 10/17 | 周六 | M7-D8 | 可解释性与注意力可视化 | - 实现注意力权重的抽取与热力图可视化<br>- 分析模型关注区域与输出质量的关系<br>- 为 SmartResearch Agent 的模型选择提供可解释依据 | day082 |
| day084 | 10/18 | 周日 | 休息日/缓冲日 | 休息日 / 缓冲复习 | - 复习 Transformer 结构、训练与变体<br>- 整理从零实现代码 | 前一天学习内容 |
| day085 | 10/19 | 周一 | M7-D9 | Transformer 源码精读 | - 阅读 Hugging Face Transformers 中 GPT/BERT 的核心源码<br>- 理解分词、Embedding、模型前向、生成策略的实现细节<br>- 记录源码阅读笔记 | day083 |
| day086 | 10/20 | 周二 | M7-D10 | 与 Hugging Face 生态集成 | - 使用 Transformers 加载预训练模型与 Tokenizer<br>- 实现文本生成、特征抽取等基础任务<br>- 将 Hugging Face 模型接入 SmartResearch Agent 的 local_model 模块 | day085 |
| day087 | 10/21 | 周三 | M7-D11 | 高效推理与量化 | - 理解 KV Cache、量化（INT8/INT4）、批处理等加速手段<br>- 使用 accelerate / bitsandbytes 进行模型量化加载<br>- 评估量化模型在 SmartResearch Agent 上的速度与效果 | day086 |
| day088 | 10/22 | 周四 | M7-D12 | 项目底层原理串联 | - 将 Transformer、Embedding、Attention 与 Agent/RAG 能力串联<br>- 撰写 SmartResearch Agent 底层模型原理解析文档<br>- 设计一场“原理 → 应用”的分享提纲 | day087 |
| day089 | 10/23 | 周五 | M8-D1 | 神经网络基础 | - 理解神经元、激活函数、前向传播与损失函数<br>- 使用 PyTorch 搭建简单全连接网络<br>- 将神经网络基础与 Transformer 中的 FFN 关联 | day088 |
| day090 | 10/24 | 周六 | M8-D2 | 反向传播 | - 理解计算图、链式法则与自动求导<br>- 使用 PyTorch 验证梯度计算<br>- 分析反向传播在训练深层网络中的关键作用 | day089 |
| day091 | 10/25 | 周日 | 休息日/缓冲日 | 休息日 / 缓冲复习 | - 复习神经网络基础与反向传播推导<br>- 补做微分与梯度练习 | 前一天学习内容 |
| day092 | 10/26 | 周一 | M8-D3 | 优化器 | - 理解 SGD、Momentum、Adam 等优化器的更新规则<br>- 对比不同优化器在简单任务上的收敛表现<br>- 为 SmartResearch Agent 的微调脚本选择合适优化器 | day090 |
| day093 | 10/27 | 周二 | M8-D4 | 卷积神经网络 CNN | - 理解卷积、池化、感受野与特征图<br>- 使用 PyTorch 实现简单 CNN 并训练图像分类<br>- 认识 CNN 在视觉预处理中的位置 | day092 |
| day094 | 10/28 | 周三 | M8-D5 | 循环神经网络 RNN / LSTM | - 理解 RNN/LSTM 的序列建模能力<br>- 实现 LSTM 并分析门控机制<br>- 对比 Transformer 与 RNN 在长序列上的优劣 | day093 |
| day095 | 10/29 | 周四 | M8-D6 | 训练技巧与正则化 | - 掌握批量归一化、Dropout、学习率衰减、早停<br>- 实现训练日志可视化<br>- 将训练技巧应用到 SmartResearch Agent 相关实验 | day094 |
| day096 | 10/30 | 周五 | M8-D7 | PyTorch 高级与综合实践 | - 使用 DataLoader、Dataset、GPU 训练流程<br>- 实现模型保存/加载与推理封装<br>- 完成一个端到端小项目并整合到学习笔记 | day095 |
| day097 | 10/31 | 周六 | R3 | M7 ~ M8 复习缓冲日 | - 复习 Transformer 与深度学习核心知识<br>- 串联底层原理与 SmartResearch Agent 上层应用<br>- 整理问题清单与后续学习重点 | day096 |
| day098 | 11/01 | 周日 | 休息日/缓冲日 | 休息日 / 缓冲复习 | - 回顾全栈知识体系<br>- 准备结业项目所需素材与数据 | 前一天学习内容 |
| day099 | 11/02 | 周一 | G1 | 结业项目整合（一） | - 整合 Agent、MCP、RAG、微调模型、评估与 API 服务<br>- 完成智研 AI 助手的最终功能联调<br>- 编写最终版 README 与架构文档 | day097 |
| day100 | 11/03 | 周二 | G2 | 结业项目整合（二） | - 完成项目演示脚本、学习总结与能力自评<br>- 规划后续持续学习方向<br>- 提交最终代码并庆祝 100 天学习里程碑 | day099 |
