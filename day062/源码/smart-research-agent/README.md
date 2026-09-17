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

## LoRA / QLoRA 参数高效微调（day051）

day050 回答了"会跑多少步"，`smart_research_agent/peft/` 回答它的姊妹问题：
**"要训多少参数、占多少显存"**。两者都不需要 GPU。

```text
LoRA 配置（r / alpha / target_modules）→ 参数量与占比（模型级 + 单层级）
  → 4-bit 量化（NF4 码本 / 分块 / 二级量化 → 每参数字节数）
  → 显存预算（权重 + 梯度 + 优化器状态六项）→ 生成 peft / bitsandbytes 脚本
```

- 矩阵算术：`lora_delta` / `merge_lora_weight` / `matrix_rank`——`ΔW = (alpha/r)·B@A`
  的秩上界是 `min(r, in, out)`，这是 LoRA 少数能被断言钉死的结构性质
  （`peft/layers.py`）；
- 参考模型：`LoRAReferenceModel` 与 day050 的 `ReferenceSFTModel`
  **接口完全一致**，因此 `SFTTrainer` 一行不改就能训 LoRA——**LoRA 只改变
  "哪些参数有梯度"，训练循环与它无关**（`peft/models.py`）；
- 参数量算术：`plan_lora` 在公开配置（`llama-2-7b` / `qwen3-0.6b`）上算出
  模型级与单层级的可训练参数。参数量对 `r` 严格线性，而容量收益不是；
- 量化：本课自实现 NF4 码本（正态分位数归一化到 `[-1,1]`）与分块量化，
  并给出**真实的存储字节数**（`nf4` + 二级量化 + `block=64` 是 0.515869 B/参数，
  16-bit 是 2 B/参数——**4-bit 并没有压到 0.5 字节**）；
- 显存：`compare_strategies` 给出全参 / LoRA / QLoRA 的六项明细。7B 上分别是
  **75.31 GiB / 12.65 GiB / 3.33 GiB**（不含激活显存）；
- 脚本生成：`render_lora_script()` / `render_qlora_script()` 产出可直接运行的
  `peft` / `bitsandbytes` 训练脚本，参数只有 `LoRAConfig` / `SFTTrainingArgs`
  一个来源。

```bash
# 一键演示：参数量 → 量化误差 → LoRA 训练 → 合并 → QLoRA → 显存 → 生成脚本
python scripts/lora_demo.py
```

新增的三个只读端点：

```bash
# LoRA / QLoRA 默认配置 + 每参数存储表 + 参考模型的参数量账
curl -s http://127.0.0.1:8000/finetune/lora/defaults | python -m json.tool

# 算一次 LoRA 训多少参数（含目标预设与秩的对照表）并给出风险告警
curl -s http://127.0.0.1:8000/finetune/lora/plan \
  -H 'Content-Type: application/json' -d '{}' | python -m json.tool

# 全参 / LoRA / QLoRA 三种策略的显存预算（不含激活）
curl -s http://127.0.0.1:8000/finetune/lora/memory \
  -H 'Content-Type: application/json' -d '{}' | python -m json.tool
```

一个必须记住的实测结论：**LoRA 的学习率区间比全参窄得多**（同一预算下，
全参在 `lr ∈ [8, 100]` 稳定，LoRA 在 `lr=22` 就发散），而且把两边都调到最优时
**全参优于 r=8 的 LoRA**——LoRA 省的是显存与落盘体积，不是效果。详见
[`docs/lora_training.md`](docs/lora_training.md)。

## LoRA 训练与部署流水线（day052）

day051 回答了"要训多少参数、占多少显存"，day052 把结论做成**可交付的产物**：
适配器检查点、多卡与混合精度计划、合并与部署门禁。

```text
适配器三文件 + 内容哈希清单 → 多卡/混合精度计划（每设备显存与通信量）
  → accelerate / DeepSpeed 配置（代码生成、可被 review）→ 合并门禁 → 部署形态对照
```

- 适配器生命周期：`save_adapter` 写**三个文件**（配置 / 权重 / 训练状态），
  `adapter_content_hash` 只覆盖配置与权重（"权重是不是同一份"），
  `prune_adapters` 清理中间副本而 `adapter-final` 永不删除（`peft/trainer.py`）；
- 训练循环只有一份：给 day050 的 `SFTTrainer` 加了 `on_step` 回调，
  `LoRATrainer` 用它把落盘插进循环，**循环逻辑一行没有复制**；
- 多卡算术：`plan_distributed` 给出全局批、步数、学习率缩放、每设备显存与
  每步通信量。**同一份配置在 4 卡上是"ddp 75.31 / zero2 28.24 / zero3 18.83 GiB"**
  （全参微调 7B），而 **LoRA 场景下 `zero2` 几乎不省**（12.65 → 12.59 GiB）——
  因为它的优化器状态只有适配器那一百来 MB，占大头的是冻结的基座权重；
- 配置是数据：`accelerate_config` / `deepspeed_zero_config` / `render_config_yaml`
  生成的配置字段名与官方一致，可被 `accelerate launch --config_file` 直接消费；
- 部署门禁：`verify_merge` 在全部上下文上逐位比较合并前后的 logits（实测差
  `0.00e+00`），`tolerance` 缺省为 0；
- 部署形态：**一个基座 + 10 个适配器**比 10 个合并模型省 **10.00×**
  （适配器 274 678 B vs 基座 13 476 831 232 B）。

```bash
# 一键演示：适配器生命周期 → 多卡计划 → 合并验证 → 生成脚本
python scripts/lora_pipeline_demo.py
```

新增的两个只读端点：

```bash
# 多卡/混合精度计划 + accelerate 与 DeepSpeed 配置 + 启动命令 + 告警
curl -s http://127.0.0.1:8000/finetune/lora/distributed \
  -H 'Content-Type: application/json' \
  -d '{"model": "llama-2-7b", "devices": 4, "strategy": "zero3"}' | python -m json.tool

# 保存适配器 → 生成清单 → 验证合并 → 部署体积对照（train_steps>0 时真跑几步）
curl -s http://127.0.0.1:8000/finetune/lora/deploy \
  -H 'Content-Type: application/json' -d '{"train_steps": 4}' | python -m json.tool
```

细节（适配器三个文件、分片策略的收益来源、学习率缩放法则、YAML/DeepSpeed 的
两个陷阱、常见故障对照表）见 [`docs/lora_deployment.md`](docs/lora_deployment.md)。

## 微调模型评估（day053）

day052 交出了适配器与合并门禁，day053 回答下一个问题：**这份适配器值不值得上线**。
微调的特殊之处在于适配器只改了 2.787% 的参数，输出往往"看起来差不多"，
所以"我觉得好了点"是最没有价值的判断——本课把它换成一组可复现的数字。

```text
领域评估集（6 桶 × 3 条 + 难度体检 + 泄漏体检 + 参考答案自洽体检）
  → 六个文本分量（事实点 / 禁项 / 词面 / 顺序 / 片段 / 格式 / 拒答）
  → 白盒探针（困惑度与命中率，真的训练一次 LoRA）
  → 配对比较（McNemar 精确检验 + 配对自助法 + Wilson 区间）
  → 报告（绑定适配器哈希与评估集指纹 + 三项门禁 + 失败清单）
```

- 评估集：18 条手写用例覆盖 6 类能力（引用 / 工具语义 / 格式 / 事实 / 安全边界 /
  简洁度），**难度由结构负载推导**（`audit_suite` 报出"声明 ≠ 推导"的条目），
  **切分按桶分层且用轮转位次**（否则每个桶都贡献自己最简单的一条，
  `by_difficulty` 那张表会失去信息量）；
- 报告绑定：`verify_manifest_binding` 核对适配器内容哈希 / 基座 / 步数，
  报告同时带评估集指纹（本课套件是 `93daad77e78b9721`）——**没有绑定信息的分数不是证据**；
- 六个文本分量（`finetune_eval/metrics.py`）：`fact_recall` 0.40 / `token_f1` 0.20 /
  `rouge_l` 0.15 / `chrf` 0.15 / `format` 0.05 / `refusal` 0.05。
  **合格是合取**（事实全覆盖 + 无禁项 + 格式合规 + 拒答符合预期），
  加权总分只回答"好多少"；
- 三种统计口径（`finetune_eval/compare.py`）：Wilson 区间（小样本上不用正态近似）、
  McNemar **精确**检验（只看翻转的两格）、配对自助法（对逐条差值重采样）。
  本课实测：合格率 `33.33% → 83.33%`，自助法区间不含 0（`+0.2957`，
  `[+0.1994, +0.4059]`），而 **McNemar 的 `p = 0.25` 并不显著**（只有 3 个不一致对）——
  "看起来涨了 50%"与"统计上证明涨了"是两件事；
- 白盒探针（`finetune_eval/model_probe.py`）：在参考模型上真的训练 LoRA，
  量"模型对领域答案有多熟"。实测三组配置：`lr=2.0` 评估集困惑度
  `421.76 → 290.49`（泛化间隙 149.76）、`lr=8.0` 训练集 `421.82 → 14.46`
  而评估集 `587.64`（**过拟合**，间隙 573.24）、`lr=20.0` **发散**（困惑度 `inf`）；
- 门禁三项：`min_pass_rate` / `no_regression` / `execution`，且**每条门禁都有
  "失败时会怎样"的用例**（day052 那条纪律沿用）。

```bash
# 一键演示：评估集体检 → 切分与泄漏 → 两臂比较 → 探针 → 报告与绑定 → 预算外推
python scripts/finetune_eval_demo.py
```

新增的两个端点：

```bash
# 评估集画像：桶/难度分布、难度体检、切分、泄漏体检、参考答案自洽体检（不跑模型）
curl -s http://127.0.0.1:8000/finetune/eval/suite | python -m json.tool

# 跑一次完整评估（probe=true 时会真的训练一次 LoRA，约 2 秒）
curl -s http://127.0.0.1:8000/finetune/eval/run \
  -H 'Content-Type: application/json' \
  -d '{"probe": true, "min_pass_rate": 0.5}' | python -m json.tool
```

注意 `FINETUNE_EVAL_SUITE_RATIO`（切**评估套件**）与 `FINETUNE_EVAL_RATIO`（切
**微调数据**，day048）**不是一回事**：两者名字都含 eval，混用会让"评估集"
这个说法同时指两件事。

细节（六个分量的边界定义、chrF 的短文本陷阱、`is_refusal` 的假阳性代价、
探针与训练 loss 的分母差、三组标定配置的完整数据）见
[`docs/finetune_eval.md`](docs/finetune_eval.md)。

## 领域数据准备与增强（day057）

day048 的 `finetune/` 回答了"这一批数据能不能用"，`smart_research_agent/domain_data/`
回答它的后半句：**下一批来了怎么办**。六个阶段，顺序即策略：

```text
clean → quality → near_dedup → mixing → augment → freeze
清洗     五维打分   两级去重       削峰     填谷      打指纹
```

- 近重复指纹（`domain_data/fingerprint.py`）：shingle → MinHash 签名 →
  Jaccard 估计。**阈值 0.7 是标定出来的**：短中文提问上取常见的 0.8
  会让近重复一条都检不出（实测 69 条探针语料上 `near=0`），而 0.7
  在不误伤 37 条真实样本的前提下抓出 9/16 条前缀复制；
- 两级去重器（`domain_data/dedup.py`）：精确指纹先挡、MinHash 近重复补漏，
  `NearDuplicateIndex` 支持**增量**（历史进索引 → 新批次 `filter`），这是
  跨批次更新的基础；
- 五维质量打分（`domain_data/quality.py`）：`length` / `repetition` /
  `structure` / `traceability` / `placeholder` 加权成 0~1 连续分。门槛 0.6
  精确等价于"**加权损失超过 0.4 才拒**"，因此**任何单个维度归零都不会被拒**
  ——单点致命问题交给 day048 的七条硬规则，打分器只负责"都合格但有好劣"的排序；
- 配比控制（`domain_data/mixing.py`）：上限约束会**连锁收紧**（`16:5:16`
  收到 0.4 要跑 6 轮不动点迭代，最终 `10:5:10`，代价是丢掉 12/37 条）；
  且**未必可达**（`safety` 口径 21:16 收 0.4 会崩到 2 条，`feasible=false`
  并给出告警）。缺省上限 0.5 是"安全网"档，本课程数据集一条不丢；
- 数据增强（`domain_data/augment.py`）：四个算子，三条纪律——**只改
  prompt 侧（`output` 逐字不动）**、**每条约束自证"答案本来就满足"**、
  **看不见风险的算子默认关闭**（`noise`）。增强阶段的去重范围刻意
  收窄为"增强样本之间"：否则实测会浪费 38% 的 `prefix` 产出
  （"请帮我看看：X"与"X"的 Jaccard 是 0.8667，会被亲本判重）；
- 六阶段编排与清单（`domain_data/pipeline.py`）：`STAGE_ORDER` 固定，
  逐阶段记账（`dropped` / `added` 分开，增强阶段不会印出负数），
  `manifest.json` 带**内容指纹 + 父指纹 + 全部参数快照**，构成可复现的
  版本链；`merge_with_history` 做跨批次增量合并。

实测量级：41 条原始样本 → 清洗 37（丢掉 1 条过短 + 1 条占位 + 2 条未核对）
→ 质量 37 → 去重 37 → 配比 37 → 增强 **+29** → **66 条**，指纹
`ffa3f96a774809ee`，质量分 `min 0.675 / mean 0.8939`。

```bash
# 一键演示：维度表 → 六阶段 → 清单 → 配比连锁削减 → 阈值标定 → 增量合并 → 落盘
python scripts/domain_data_demo.py
```

新增的三个只读端点：

```bash
# 五维质量表（代码生成）+ 默认权重 + 门槛 + 增强算子表 + 六阶段顺序
curl -s http://127.0.0.1:8000/data/domain/dimensions | python -m json.tool

# 增强预览（返回的每条增强样本 output 与输入逐字相同）
curl -s http://127.0.0.1:8000/data/domain/augment \
  -H 'Content-Type: application/json' \
  -d '{"examples": [{"instruction": "什么是 RAG？", "output": "检索增强生成。"}], "ops": ["prefix"]}' \
  | python -m json.tool

# 对仓库内置数据源跑一次六阶段流水线（只返回清单与阶段报告，不含样本正文）
curl -s http://127.0.0.1:8000/data/domain/run \
  -H 'Content-Type: application/json' -d '{}' | python -m json.tool
```

细节（门槛 0.6 的精确算法、近重复 ≠ 语义重复、阈值敏感性表、配比不动点
迭代与不可达案例、增强的三条纪律与"与亲本撞车"的真实冲突、12 条常见错误
对照表）见 [`docs/domain_data_augmentation.md`](docs/domain_data_augmentation.md)。

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
