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

## 微调数据工程（day048）

`smart_research_agent/finetune/` 是一条离线、确定、可测试的数据流水线：

```text
采集（种子 / 评估轨迹 / 红队安全样本）→ 清洗 → 质量规则 → 去重 → 统计 → 切分 → 落盘
```

- 格式规范：`alpaca` / `chat` / `prompt-completion` 三选一，统一收敛到
  `TrainingExample`（`finetune/schema.py`）；
- 质量规则：7 条默认规则逐条记账，拒绝原因计入 `FilterReport.drop_reasons`
  （`finetune/cleaner.py`）；
- 方法选型：`METHODS` 五个方法画像 + 确定性 `recommend_method`
  （`finetune/overview.py`）；
- 数据集画像与落盘：`compute_stats` / `split_dataset` / `dump_bundle`
  （`finetune/dataset.py`）。

```bash
# 一键演示：采集 → 清洗 → 统计 → 切分，产出 data/finetune/out/{train,eval}.jsonl
python scripts/finetune_data_demo.py
```

新增的三个只读端点：

```bash
# 五种微调方法的画像
curl -s http://127.0.0.1:8000/finetune/methods | python -m json.tool

# 当前数据集画像（种子 + 评估轨迹 + 红队样本）
curl -s http://127.0.0.1:8000/finetune/dataset/stats | python -m json.tool

# 校验一批原始样本能否用于训练（未知 format → 400，examples 为空 → 422）
curl -s http://127.0.0.1:8000/finetune/dataset/validate \
  -H 'Content-Type: application/json' \
  -d '{"format": "alpaca", "examples": [{"instruction": "什么是 RAG？", "output": "检索增强生成。"}]}'
```

数据流水线的细节（五个数据源、格式字段表、质量规则清单、评估集轨迹为何要打
`unverified` 标签）见 [`docs/finetune_data_pipeline.md`](docs/finetune_data_pipeline.md)。

## SFT 监督微调（day050）

`smart_research_agent/sft/` 把 day048 落盘的 `train.jsonl` / `eval.jsonl`
真正喂进模型：

```text
渲染（ChatML / Llama3 / Plain）→ 编码（prompt 段 label 置 -100）→ 长度分位定 max_length
  → 步数/有效批（纯算术）→ 训练循环（梯度累积 + 调度 + 评估）→ 检查点落盘
```

- 模板与监督切分：`render_supervised` 产出 `prompt_chars` / `supervised_chars`
  两个字符偏移（`sft/template.py`）；
- 编码与 mask：**前缀与答案分别 tokenize 再拼接**，`labels` 里前缀段为
  `IGNORE_INDEX = -100`（`sft/encoding.py`，与 PyTorch / HF / TRL 同一约定）；
- 长度分位：`length_summary` / `suggest_max_length`——`max_length` 必须按
  **渲染并分词之后**的长度分位数定（本数据集 `p95 = 302`、`max = 320`，
  而 day048 画像里的估计只有 84.24 token/条）；
- 超参与派生量：`SFTTrainingArgs` 字段名与 `transformers.TrainingArguments`
  逐字对齐，`plan_training` 在训练前算出步数并生成 8 类风险告警；
- 训练循环：`SFTTrainer` + 参考模型（557×557 bigram + 纯 SGD，真实梯度下降，
  离线可跑）；真实框架路径由 `render_hf_sft_script()` 生成。

```bash
# 一键演示：数据 → 渲染 → 编码 → 长度分位 → 计划 → 训练 → 落盘 → 生成脚本
python scripts/sft_demo.py
```

新增的三个只读端点：

```bash
# SFT 默认超参（与 TrainingArguments 同名）+ SFT 专属项 + 依赖清单
curl -s http://127.0.0.1:8000/finetune/sft/defaults | python -m json.tool

# 算一次训练的派生量（步数 / warmup / 有效批）并给出风险告警与 max_length 建议
curl -s http://127.0.0.1:8000/finetune/sft/plan \
  -H 'Content-Type: application/json' -d '{}' | python -m json.tool

# 预览一条样本的监督切分与 label mask（prompt_fully_masked 必须为 true）
curl -s http://127.0.0.1:8000/finetune/sft/preview \
  -H 'Content-Type: application/json' \
  -d '{"example": {"instruction": "什么是 RAG？", "output": "检索增强生成。"}}'
```

训练细节（三种模板、label mask 机制、`max_length` 与学习率怎么标定、
参考实现与真实框架的接缝、常见错误对照表）见
[`docs/sft_training.md`](docs/sft_training.md)。

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
