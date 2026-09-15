# 微调数据流水线（M5-D1）

本文件说明 `smart_research_agent/finetune/` 的数据工程链路：**采集 → 清洗 →
去重 → 统计 → 切分 → 落盘**。所有步骤离线、确定、可测试，不依赖任何模型
调用或网络。

## 1. 全景

```text
data/finetune/seed_examples.jsonl ─┐
data/eval/agent_tasks.jsonl ───────┼─→ JSONLSource / EvalTaskSource / RedTeamSource
data/eval/redteam_cases.jsonl ─────┘              │
                                                  ▼  TrainingExample[]
                              DatasetCleaner.run()  （清洗 → 质量规则 → 去重）
                                                  │
                                                  ▼  (list[TrainingExample], FilterReport)
                     compute_stats(bundle.examples) → DatasetStats
                     split_dataset(bundle.examples) → (train, eval)
                     dump_bundle(bundle, "data/finetune/out") → train.jsonl / eval.jsonl
```

| 阶段 | 函数 / 类 | 文件 |
| --- | --- | --- |
| 格式定义与解析 | `TrainingExample`、`parse_example`、`load_jsonl`、`dump_jsonl` | `smart_research_agent/finetune/schema.py` |
| 方法选型 | `METHODS`、`recommend_method`、`lora_trainable_ratio`、`render_methods_table` | `smart_research_agent/finetune/overview.py` |
| 清洗与过滤 | `clean_text`、`dedupe_key`、`default_rules`、`DatasetCleaner`、`FilterReport` | `smart_research_agent/finetune/cleaner.py` |
| 采集 | `DataSource`、`JSONLSource`、`EvalTaskSource`、`RedTeamSource`、`DataCollector`、`default_collector` | `smart_research_agent/finetune/collector.py` |
| 统计 / 切分 / 落盘 | `compute_stats`、`split_dataset`、`dump_bundle`、`build_dataset` | `smart_research_agent/finetune/dataset.py` |
| 一键演示 | `main()` | `scripts/finetune_data_demo.py` |

清洗**发生在质量过滤之前**：规则要判断的是"最终会喂给模型的那些字符"。
若拿原始文本判长度，一条内容为 `"\n\n\n"` 的输出会以 3 个字符通过
`min_output_chars=8`，却在训练时变成空样本——顺序错了，规则就形同虚设。

## 2. 五个数据源

| # | 数据源 | 文件路径 | 解析类 | 默认是否收集 |
| --- | --- | --- | --- | --- |
| 1 | 手写领域种子样本 | `data/finetune/seed_examples.jsonl` | `JSONLSource` | 是 |
| 2 | Agent 评估轨迹 | `data/eval/agent_tasks.jsonl` | `EvalTaskSource` | 是 |
| 3 | 红队安全用例 | `data/eval/redteam_cases.jsonl` | `RedTeamSource` | 是 |
| 4 | 任意第三方 JSONL 语料 | 任意路径 | `JSONLSource`（自行构造 `SourceSpec`） | 按需接入 |
| 5 | 课程知识库文档 | `data/knowledge/rag.md` | 需先转成指令对（本阶段未自动收集） | 否（规划中） |

`default_collector(data_dir=None)` 装配 1~3 三个源：种子目录取
`settings.finetune_data_dir`（`data/finetune`），评估集固定在其同级的
`data/eval/` 下；清洗阈值全部取自 `settings.finetune_*` 配置项
（`FINETUNE_MIN_OUTPUT_CHARS` 等，见 `.env.example`）。

新增来源只需实现 `DataSource` 的两个成员（`name` 属性与 `load()` 方法），
采集、清洗、统计、落盘全部零改动。

## 3. 格式规范

内部统一表示是 `TrainingExample`（`instruction` / `input` / `output` /
`system` / `source` / `tags` / `license`），落盘那一刻才通过
`to_dict(fmt)` 投影为目标格式；`parse_example(raw, fmt)` 反向解析。
`SUPPORTED_FORMATS = ("alpaca", "chat", "prompt-completion")`，
`DEFAULT_FORMAT = "alpaca"`。

### alpaca（默认）

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `instruction` | 是 | 任务描述，不能为空 |
| `input` | 否 | 待处理材料，可为空字符串；投影时**保留空键** |
| `output` | 是 | 标准答案，不能为空 |
| `system` / `source` / `tags` / `license` | 否 | 治理元信息，解析时一并带出 |

### chat

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `messages` | 是 | 消息列表，至少各含一条**非空**的 `user` 与 `assistant` |
| `messages[].role` | 是 | `system` / `user` / `assistant`（`system` 可选） |
| `messages[].content` | 是 | 消息文本 |

### prompt-completion

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `prompt` | 是 | 映射到内部 `instruction`，不能为空 |
| `completion` | 是 | 映射到内部 `output`，不能为空 |

任一格式不满足上表即抛 `DatasetFormatError`（继承 `ValueError`）。

## 4. 质量规则清单

`default_rules(...)` 生成 7 条规则，**顺序即优先级，第一条不通过即拒**；
`check` 返回 `True` 表示通过。所有规则都在清洗后的文本上判定。

| 顺序 | 规则名 | 判定（返回 False 即拒绝） | 默认阈值 |
| --- | --- | --- | --- |
| 1 | `output_too_short` | `len(output) >= min_output_chars` | `finetune_min_output_chars = 8` |
| 2 | `output_too_long` | `len(output) <= max_output_chars` | `finetune_max_output_chars = 4000` |
| 3 | `instruction_too_long` | `len(instruction) <= max_instruction_chars` | `finetune_max_instruction_chars = 1000` |
| 4 | `empty_instruction` | `instruction.strip()` 非空 | — |
| 5 | `placeholder_output` | 输出不含 `TODO` / `待补充` / `lorem` / `tbd` / `fixme` | `PLACEHOLDER_MARKERS` |
| 6 | `banned_pattern` | 指令与输出均不含禁用词 | `banned_patterns`（默认空） |
| 7 | `unverified_source` | `tags` 中不含 `unverified` | — |

去重在规则之后执行（`dedupe_key` = `sha1(clean_text(prompt_text).lower())[:16]`，
保留首次出现），`data` 重复计入 `FilterReport.drop_reasons["duplicate"]`。
三类结果都逐条记录在 `FilterReport.decisions`（`rule` 为 `kept`、规则名或
`duplicate`），可通过 `FilterReport.to_dict()` 直接序列化。

## 5. 为什么评估集轨迹要打 `unverified` 标签

`data/eval/agent_tasks.jsonl` 是 **评估集**，不是训练集：它只标注了
`expected_tools`（期望调用哪个工具）和可选的 `answer_contains`（答案必须
包含的字符串），轨迹本身由脚本化的 `mock_responses` 拼出。`EvalTaskSource`
的转换规则是：

1. 轨迹文本 = `mock_responses` 用 `\n` 连接，最后一条含 `Final Answer:`
   的文本即最终答案；
2. **没有** `answer_contains` → 无法核对答案，标记
   `("from:eval", "unverifiable")` 并跳过（不进数据集）；
3. 有 `answer_contains` 且最终答案**包含**它 → 保留，
   `tags=("from:eval", "react-trace")`；
4. 有 `answer_contains` 但最终答案**不包含**它 → 仍然解析出来，但打上
   `unverified`，由第 4 节的 `unverified_source` 规则丢弃。

第 4 条是刻意的：**评估集里的轨迹只保证"期望调用哪个工具"，并不保证
答案正确**——`task-04` 把 7×8 答成 99（期望包含 `56`），`task-05` 把
15×15 答成 150（期望包含 `225`）。如果把这些轨迹无脑当训练数据，我们会
亲手把错误答案教给模型。保留到清洗阶段再由规则丢弃，好处是
`FilterReport.drop_reasons["unverified_source"]` 会出现一个**非零计数**：
"评估集不能直接当训练集"这条结论因此在流水线里可见、可断言，而不是一句
口号。

实测（`python scripts/finetune_data_demo.py`，8 条任务）：

| 结论 | 条数 | 对应样本 |
| --- | --- | --- |
| 保留（答案包含 `answer_contains`） | 5 | `task-01`、`task-02`、`task-03`、`task-07`、`task-08` |
| 打 `unverified`（答案对不上） | 2 | `task-04`（`56` vs `99`）、`task-05`（`225` vs `150`） |
| 跳过（无 `answer_contains`） | 1 | `task-06` |

## 6. 实测数字

在 `day048/源码/smart-research-agent` 下运行 `python scripts/finetune_data_demo.py`：

- 载入 41 条，保留 **37** 条，丢弃 4 条（保留率 90.24%）；
- 拒绝原因：`output_too_short` 1 条、`placeholder_output` 1 条
  （两条故意写入 `data/finetune/seed_examples.jsonl` 的脏数据）、
  `unverified_source` 2 条（上表的 `task-04` / `task-05`）；
- 来源分布：`seed` 16、`eval/agent_tasks` 5、`eval/redteam_cases` 16；
- 长度分布：平均指令 26.81 字、平均输出 88.11 字，输出长度 67~165 字，
  平均约 84.24 token/条（字符级估算，离线确定）；
- 落盘：`data/finetune/out/train.jsonl`（30 条）与
  `data/finetune/out/eval.jsonl`（7 条），切分种子
  `settings.finetune_split_seed = 42`。

## 7. 怎么跑

```bash
# 一键演示：采集 → 清洗 → 统计 → 切分落盘（不联网、不调用模型）
python scripts/finetune_data_demo.py

# 单元测试（含 finetune 包与缓存包装器的成本归因）
python -m pytest tests/test_finetune_schema.py tests/test_finetune_cleaner.py \
  tests/test_finetune_collector.py tests/test_finetune_dataset.py \
  tests/test_finetune_overview.py tests/test_finetune_api.py \
  tests/test_wrapper_accounting.py -q
```

三个只读 HTTP 端点（`smart_research_agent/api/routes.py`）：

```bash
curl -s http://127.0.0.1:8000/finetune/methods | python -m json.tool
curl -s http://127.0.0.1:8000/finetune/dataset/stats | python -m json.tool
curl -s http://127.0.0.1:8000/finetune/dataset/validate \
  -H 'Content-Type: application/json' \
  -d '{"format": "alpaca", "examples": [{"instruction": "什么是 RAG？", "output": "检索增强生成。"}]}' \
  | python -m json.tool
```

契约与错误码：请求不合法（`examples` 为空）→ `422`；未知 `format` → `400`；
问题样本的 `index` 与 `reason` 通过 `issues` 返回（最多前
`routes.MAX_DATASET_ISSUES = 20` 条）。
