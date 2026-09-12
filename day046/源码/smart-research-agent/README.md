> 智研 AI 助手（SmartResearch Agent）

一个为期 100 天的 AI 技术系统学习主线项目，从 Python 工程化基础出发，逐步构建具备 Agent 推理、MCP 协议、RAG 检索、模型微调、评估监控等能力的智能研究助手。

## 当前阶段

P2 - Python 工程化基础（二）：Git 工作流与分支规范、pytest 测试与覆盖率、环境脚本完善、Docker 基础镜像。

## 环境要求

- Python 3.10+
- Git
- （可选）Docker Desktop（容器化验证用）
- （可选）Linux/macOS；Windows 建议使用 WSL 或 Git Bash

## 快速开始

```bash
# 1. 进入项目目录
cd smart-research-agent

# 2. 一键初始化（创建虚拟环境、安装依赖、安装 pre-commit）
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 OPENAI_API_KEY

# 4. 验证环境
python scripts/verify_env.py

# 5. 运行入口
python -m smart_research_agent.main
```

## 项目结构

```text
smart-research-agent/
├── smart_research_agent/   # 主包
├── scripts/                # 辅助脚本
├── tests/                  # 单元测试
├── docs/                   # 文档
├── logs/                   # 日志输出
├── pyproject.toml          # 项目配置与依赖
├── requirements.txt        # 依赖列表
├── .env.example            # 环境变量模板
└── README.md               # 项目说明
```

## 一体化流水线（day046）

`/pipeline/run` 把 M4 的六层能力串成一条调用链：

```text
输入侧护栏 → 语义缓存 → 模型路由与生成 → 成本归因 → 输出侧审核 → 缓存回写
```

```bash
# 跑一次完整流水线（响应携带逐阶段耗时、单次费用归因与护栏/审核报告）
curl -s http://127.0.0.1:8000/pipeline/run \
  -H 'Content-Type: application/json' \
  -d '{"task": "请分析并对比 RAG 与微调的优劣"}' | python -m json.tool

# 查看已存档的性能基线（无基线时返回 404，而不是一份全零的假基线）
curl -s http://127.0.0.1:8000/pipeline/baseline | python -m json.tool
```

离线跑通六层能力、采集基线并做回归比对：

```bash
python scripts/integration_demo.py
# 产出/更新 data/eval/perf_baseline.json（随仓库提交的参考基线）
```

开关（见 `.env.example`）：`PIPELINE_CACHE_ENABLED` 控制是否启用缓存阶段，
`PIPELINE_BLOCK_ON_INJECTION` 控制注入命中时"拒答 / 只记录"，
`PERF_*_TOLERANCE` 控制回归判定阈值。

## 开发命令

```bash
# 代码格式化
black smart_research_agent tests

# 代码检查
ruff check smart_research_agent tests

# 类型检查
mypy smart_research_agent

# 运行测试
pytest tests/ -v

# pre-commit 手动触发
pre-commit run --all-files
```

## 许可证

MIT License
