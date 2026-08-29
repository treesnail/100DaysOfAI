

> 100 天 AI 技术系统学习实战项目 · 从应用层到模型层的全栈进阶

---

## 📖 项目概述

本项目是一个为期 **100 天** 的 AI 技术系统学习计划，以「智研 AI 助手」（SmartResearch Agent）为主线项目，从 Python 工程化基础出发，逐步深入 AI Agent、MCP 协议、评估框架、大模型应用、模型微调、RAG、Transformer 架构、深度学习等核心领域，最终完成一个功能完整、可评估、可部署的智能研究助手系统。

**核心理念**：以项目驱动学习，每天的学习内容都直接落地到主线项目中，避免"学了就忘"。所有代码、笔记、练习都在同一个仓库中持续迭代，形成可追溯的学习轨迹。

---

## 🎯 学习目标

完成 100 天学习后，达到 **AI 应用开发中级水平**：

- **原理理解**：对 Agent、MCP、RAG、Transformer、微调等核心技术有较为深刻的原理理解，能说清"为什么这样设计"
- **工程能力**：掌握主流开发工具和流程（LangChain/LangGraph、FastMCP、FastAPI、PyTorch、Docker），能独立完成从设计到部署的全流程
- **项目实战**：拥有一个功能完整的智能研究助手项目，涵盖 Agent 推理、工具调用、MCP 协议、RAG 检索、模型微调、评估监控等能力
- **持续学习**：建立 AI 技术知识图谱，具备自主学习新技术、阅读论文和源码的能力

---

## 📅 时间规划

**总周期**：2026-07-27 ~ 2026-11-03，共 100 个日历日
**学习日**：86 天（每周日休息，共 14 个休息日）
**每日时长**：2 小时
**总学习时长**：约 172 小时

### 阶段总览

| 阶段 | 学习日 | 日历日范围 | 天数 | 核心目标 |
|------|--------|-----------|------|----------|
| **P 预备阶段** | P1-P2 | 7/27 - 7/28 | 2天 | Python 工程化速通 + 环境搭建 + 项目初始化 |
| **M1 智能体应用开发** | D1-D11 | 7/29 - 8/11 | 11天 | Agent 架构、ReAct、工具调用、记忆、规划、多 Agent、LangGraph、安全、评估 |
| **M2 MCP 应用开发** | D12-D18 | 8/12 - 8/20 | 7天 | MCP 协议、Resources/Tools/Prompts、FastMCP 开发、Docker 部署 |
| **R1 复习缓冲日** | R1 | 8/22 | 1天 | M1-M2 复习、补做练习、整理笔记 |
| **M3 Harness 评估框架** | D19-D25 | 8/23 - 8/30 | 7天 | AI 评估框架、Prompt/RAG/Agent 评估、安全红队、CI/CD、可观测性 |
| **M4 大模型应用开发** | D26-D37 | 9/1 - 9/14 | 12天 | LLM 原理、Tokenization、Prompt 工程、Function Calling、FastAPI 后端、多模态、Embedding、成本优化、安全、开源部署 |
| **R2 复习缓冲日** | R2 | 9/19 | 1天 | M3-M4 复习、补做练习、整理笔记 |
| **M5 大模型微调** | D38-D48 | 9/15 - 10/1 | 11天 | SFT、LoRA/QLoRA、RLHF/DPO、数据工程、评估、MLOps、持续微调 |
| **M6 RAG 应用开发** | D49-D58 | 10/3 - 10/14 | 10天 | 文档解析、分块策略、向量数据库、混合检索、重排序、高级 RAG、评估、生产化部署 |
| **数学基础过渡** | M1-M2 | 10/16 - 10/17 | 2天 | 线性代数+概率论、微积分+优化基础（为 Transformer/深度学习预热） |
| **M7 Transformer 架构** | D59-D70 | 10/18 - 11/1 | 12天 | Attention 机制、多头注意力、位置编码、Encoder/Decoder、从零实现、训练优化、变体架构、可解释性、源码精读 |
| **M8 深度学习（简化）** | D71-D77 | 11/3 - 11/12 | 7天 | 神经网络基础+反向传播、优化器、CNN、RNN/LSTM、训练技巧、PyTorch 高级、综合实践 |
| **R3 复习缓冲日** | R3 | 11/14 | 1天 | M7-M8 复习、全栈知识串联 |
| **结业项目** | G1-G2 | 11/15 - 11/16 | 2天 | 整合所学完成综合项目、项目文档、学习总结、能力自评、后续规划 |

> **注**：以上日历日范围为近似值，实际以"每周日休息"规则顺延为准。

### 学习路径设计逻辑

```
应用层（能做什么）          原理层（为什么这样）          基础层（底层是什么）
─────────────────          ─────────────────          ─────────────────
M1 Agent 应用      ──→     M3 评估 Harness              M7 Transformer
M2 MCP 协议         ──→     M4 大模型应用       ──→      M8 深度学习
                   M5 模型微调
                   M6 RAG 检索
```

**设计思路**：
1. **先应用后原理**：从 Agent、MCP 等能快速看到效果的应用层入手，建立兴趣和全局认知
2. **边用边评估**：学完应用后引入评估框架，建立"量化思维"，知道怎么衡量好坏
3. **深入模型层**：在有了应用和评估的基础上，深入大模型应用、微调、RAG 等模型相关技术
4. **最后补基础**：有了上层经验后，再学习 Transformer 和深度学习底层原理，理解更深刻
5. **每周日休息**：保证学习可持续性，留出复习和消化的时间

---

## 🤖 主线项目：智研 AI 助手

贯穿全程的综合项目，从预备阶段初始化，每个模块都对其进行对应升级：

| 阶段 | 项目状态 | 新增能力 |
|------|----------|----------|
| 预备阶段 | 项目初始化 | 目录结构、配置管理、Git 仓库、虚拟环境 |
| M1 Agent | 单 Agent → 多 Agent | ReAct 推理、工具调用、记忆系统、规划、多 Agent 协作、LangGraph 编排、安全防护、评估追踪 |
| M2 MCP | 工具协议化 | 工具重构为 MCP Server、Resources/Prompts 能力、Docker 部署 |
| M3 Harness | 可评估 | 全维度评估 Harness、安全红队、CI/CD 集成、可观测性监控 |
| M4 大模型应用 | 应用层深化 | Prompt 优化、Function Calling、FastAPI 后端服务、多模态、Embedding、成本优化、安全合规、本地模型部署 |
| M5 微调 | 模型专属化 | 领域数据准备、SFT 微调、DPO 对齐、专属模型部署替换、MLOps 流水线 |
| M6 RAG | 知识增强 | 文档解析、向量索引、混合检索+重排序、知识库集成、生产化部署 |
| M7 Transformer | 底层理解 | 理解系统底层模型原理、从零实现注意力机制 |
| M8 深度学习 | 基础夯实 | 理解神经网络底层原理、PyTorch 实战 |
| 结业 | 完整系统 | 整合所有能力的最终版智能研究助手 |

### 最终项目功能

完成 100 天后，「智研 AI 助手」将具备以下能力：

- 🧠 **智能推理**：基于 ReAct/LangGraph 的多步骤推理与规划能力
- 🔧 **工具调用**：计算器、搜索、文件操作、代码执行、时间查询等多工具组合
- 🔌 **MCP 协议**：标准化的工具/资源/提示词服务，支持 Host 应用集成
- 📚 **知识检索**：基于 RAG 的文档检索与问答，支持混合检索与重排序
- 🎯 **专属模型**：经过领域微调的专属模型，替换通用模型
- 📊 **评估监控**：全维度评估 Harness、成本追踪、性能监控、安全审计
- 🌐 **API 服务**：FastAPI 后端服务，支持流式输出、JWT 认证、多模型路由
- 🛡️ **安全防护**：Prompt 注入防御、PII 检测、内容审核、工具权限控制

---

## 🛠️ 技术栈

### 核心框架
- **LangChain / LangGraph** — Agent 编排与状态管理
- **MCP / FastMCP** — 模型上下文协议，工具标准化
- **FastAPI / Uvicorn** — 后端 API 服务
- **Pydantic / Pydantic Settings** — 数据验证与配置管理

### 模型与数据
- **Transformers / PEFT / TRL / Accelerate** — 模型微调
- **PyTorch** — 深度学习框架
- **ChromaDB / FAISS** — 向量数据库
- **tiktoken / SentencePiece** — Tokenization

### 工程化
- **pytest / pytest-cov** — 单元测试
- **logging / RotatingFileHandler** — 日志系统
- **ruff / black / mypy** — 代码规范与类型检查
- **pre-commit** — 提交前检查
- **Docker / Docker Compose** — 容器化部署

### 基础设施
- **Python 3.10+** — 开发语言
- **Git / GitHub** — 版本控制与协作
- **GitHub Actions** — CI/CD
- **Redis** — 缓存（语义缓存、会话存储）

---

## 📁 项目结构

```
smart-research-agent/
├── smart_research_agent/       # 主包
│   ├── __init__.py
│   ├── config.py                # 配置管理（Pydantic Settings）
│   ├── main.py                  # 入口文件
│   ├── llm/                     # LLM 调用层
│   │   ├── base.py              # 抽象基类
│   │   ├── openai_compatible.py # OpenAI 兼容实现
│   │   ├── local_model.py       # 本地开源模型（Ollama/vLLM）
│   │   └── embedding.py         # Embedding 模块
│   ├── tools/                   # 工具层
│   │   ├── base.py              # 工具抽象基类
│   │   ├── calculator.py        # 计算器工具
│   │   ├── web_search.py        # 搜索工具
│   │   ├── file_operations.py   # 文件操作工具
│   │   ├── datetime_tool.py     # 时间查询工具
│   │   ├── code_executor.py     # 代码执行工具
│   │   ├── image_analysis.py    # 图像分析工具
│   │   └── retry_decorator.py   # 自动重试装饰器
│   ├── agent/                   # Agent 核心
│   │   ├── react_agent.py       # ReAct Agent
│   │   ├── planner.py           # 规划模块
│   │   ├── reflexion.py         # 反思模块
│   │   └── graph.py             # LangGraph 状态图
│   ├── memory/                  # 记忆系统
│   │   ├── short_term.py        # 短期记忆
│   │   ├── long_term.py         # 长期记忆（向量检索）
│   │   └── hybrid.py            # 混合记忆
│   ├── mcp/                     # MCP 协议
│   │   ├── server.py            # MCP Server
│   │   ├── client.py            # MCP Client
│   │   └── Dockerfile           # Docker 部署
│   ├── rag/                     # RAG 检索
│   │   ├── document_loader.py   # 文档加载
│   │   ├── chunker.py           # 分块策略
│   │   ├── retriever.py         # 检索器
│   │   ├── reranker.py          # 重排序
│   │   └── generator.py         # 生成器
│   ├── evaluation/              # 评估框架
│   │   ├── base.py              # 评估基类
│   │   ├── prompt_evaluator.py  # Prompt 评估
│   │   ├── output_evaluator.py  # 输出质量评估
│   │   ├── rag_evaluator.py     # RAG 评估
│   │   ├── agent_evaluator.py   # Agent 评估
│   │   └── security_evaluator.py # 安全评估
│   ├── security/                # 安全防护
│   │   ├── guard.py             # 安全防护层
│   │   ├── prompt_injection.py  # 注入检测
│   │   └── pii_detector.py      # PII 检测
│   ├── monitoring/              # 监控体系
│   │   ├── cost_tracker.py      # 成本追踪
│   │   └── metrics.py           # 指标采集
│   ├── api/                     # API 服务
│   │   ├── main.py              # FastAPI 入口
│   │   └── routes/              # 路由
│   ├── finetune/                # 模型微调
│   │   ├── data_prep.py         # 数据准备
│   │   ├── sft.py               # SFT 微调
│   │   ├── dpo.py               # DPO 对齐
│   │   └── mlops.py             # MLOps 流水线
│   ├── multi_agent/             # 多 Agent 协作
│   │   ├── researcher.py        # 研究员
│   │   ├── analyst.py           # 分析师
│   │   └── writer.py            # 撰稿人
│   ├── prompts/                 # Prompt 模板库
│   └── utils/                   # 工具函数
│       ├── logger.py            # 日志工具
│       └── token_counter.py     # Token 计数
├── tests/                       # 测试代码
├── data/                        # 数据（文档、数据集）
├── notebooks/                   # Jupyter 实验笔记
├── logs/                        # 日志文件
├── outputs/                     # 生成结果
├── scripts/                     # 辅助脚本
│   ├── setup_env.sh             # 环境初始化脚本
│   ├── verify_env.py            # 环境验证脚本
│   └── push.sh                  # Git 推送脚本
├── .github/workflows/           # GitHub Actions CI/CD
├── docs/                        # 文档
├── .env.example                 # 环境变量模板
├── .gitignore
├── pyproject.toml               # 项目配置
├── requirements.txt             # 依赖列表
├── docker-compose.yml           # Docker Compose
└── README.md                    # 项目说明（本文件）
```

---

## 🚀 快速开始

### 环境要求

- **操作系统**：Ubuntu 22.04+（推荐 VirtualBox 虚拟机）
- **Python**：3.10 或更高版本
- **内存**：最低 4GB，推荐 8GB+
- **存储**：至少 20GB 可用空间
- **GPU**：可选（微调模块建议使用 Google Colab 或 GPU 云平台）

### 一键环境初始化

```bash
# 1. 克隆仓库
git clone git@github.com:your-username/smart-research-agent.git
cd smart-research-agent

# 2. 运行一键初始化脚本（安装依赖、配置环境、创建虚拟环境）
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 API Key

# 4. 验证环境
python scripts/verify_env.py

# 5. 运行第一个 LLM 调用
source venv/bin/activate
python -m smart_research_agent.main
```

### 每日学习流程

```bash
# 1. 激活虚拟环境
cd ~/ai_learning_project/02_project
source venv/bin/activate

# 2. 验证环境（确保一切正常）
python scripts/verify_env.py

# 3. 查看当天学习材料
# （学习材料在 docs/learning_materials/ 目录下）

# 4. 动手实践，在主线项目中实现当天内容

# 5. 运行测试，验证实现
pytest tests/ -v

# 6. 提交代码
git add .
git commit -m "feat: Day X - 简要描述今天的内容"
git push
```

---

## 📚 学习方法

### 每日 2 小时分配建议

| 时间段 | 内容 | 时长 |
|--------|------|------|
| 0:00 - 0:30 | 阅读学习材料，理解原理 | 30 分钟 |
| 0:30 - 1:30 | 动手实践，在主线项目中实现 | 60 分钟 |
| 1:30 - 1:50 | 完成练习题，巩固知识 | 20 分钟 |
| 1:50 - 2:00 | 整理笔记，提交代码 | 10 分钟 |

### 学习原则

1. **项目驱动**：每天的学习都落地到主线项目，不做孤立的练习
2. **原理先行**：先理解"为什么"，再动手"怎么做"
3. **动手验证**：每个概念都用代码验证，不满足于"看懂了"
4. **持续迭代**：代码可以不完美，但必须能运行，后续不断优化
5. **每周复盘**：周日休息时回顾本周内容，补做未完成的练习
6. **输出倒逼输入**：通过写代码、写笔记、提交 GitHub 来巩固学习

---

## 📊 学习进度追踪

| 阶段 | 状态 | 完成日期 | 备注 |
|------|------|----------|------|
| P1 Python 工程化 | ⬜ 待开始 | - | - |
| P2 环境搭建 | ⬜ 待开始 | - | - |
| M1 Agent 应用 | ⬜ 待开始 | - | - |
| M2 MCP 协议 | ⬜ 待开始 | - | - |
| M3 Harness 评估 | ⬜ 待开始 | - | - |
| M4 大模型应用 | ⬜ 待开始 | - | - |
| M5 模型微调 | ⬜ 待开始 | - | - |
| M6 RAG 检索 | ⬜ 待开始 | - | - |
| M7 Transformer | ⬜ 待开始 | - | - |
| M8 深度学习 | ⬜ 待开始 | - | - |
| 结业项目 | ⬜ 待开始 | - | - |

> 进度将随学习推进持续更新。

---

## 📖 参考资源

### 官方文档
- [LangChain 官方文档](https://python.langchain.com/)
- [MCP 官方规范](https://modelcontextprotocol.io/)
- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [PyTorch 官方文档](https://pytorch.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

### 经典论文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原始论文
- [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) — ReAct Agent
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) — 低秩适配微调
- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) — RAG 原始论文

### 优质教程
- [Real Python](https://realpython.com/) — Python 进阶教程
- [Andrej Karpathy - Neural Networks](https://www.youtube.com/@AndrejKarpathy) — 深度学习精品课程
- [Hugging Face Course](https://huggingface.co/learn) — NLP 与模型实战课程

---

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。

学习材料和代码仅供学习交流使用，引用第三方资源请遵守相应许可证。

---

## 🤝 致谢

感谢所有开源社区和开发者，是你们的工作让 AI 技术的学习变得更加 accessible。

特别感谢：
- LangChain 团队提供的优秀 Agent 开发框架
- Anthropic 团队提出的 MCP 协议
- Hugging Face 团队的开源模型生态
- PyTorch 团队的深度学习框架
- 所有为 AI 开源社区做出贡献的开发者

---

> **开始你的 100 天 AI 学习之旅吧！** 🚀
>
> 千里之行，始于足下。每天 2 小时，100 天后你将拥有一个完整的 AI 项目和扎实的技术功底。
