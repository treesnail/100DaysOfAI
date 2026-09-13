# day049 源码说明

复习日无新增代码，也**不产生新的代码快照**（与 day007 / day028 / day035 / day042 / day047 同例）。

当前最新代码快照见 [../../day048/源码/smart-research-agent/](../../day048/源码/smart-research-agent/)（M5-D1 **微调概览与数据工程**完成后的完整累积快照，**1124 个测试全绿，覆盖率 96.44%**，覆盖率硬护栏 `fail_under = 90` 已生效）。

全部快照均为**累积式**：day048 的快照已包含 M5-D1 的 `finetune/` 包（6 个文件、477 语句、110 分支，100% 覆盖），以及更早的 M1~M4 全部代码与测试。走读时以 day048 快照为主即可。

> 本日与 day047 的区别：day047 是 **R2 跨阶段复习日**，把 M3 与 M4 合并收口，并做了一次真实的包装器用量核对；**day049 是 M5 起步期的缓冲日**，范围为 day044~day048 五天，重点是把"M4 收尾三课 + M5 第一课"串成一条交接链，并**为 day050 的 SFT 把超参与步数先算准**。

本日复习涉及的三段代码（均在 day048 快照内）：

- **M4 收尾 · 防线与部署（day044~day045）**：
  - [../../day048/源码/smart-research-agent/smart_research_agent/security/content_moderator.py](../../day048/源码/smart-research-agent/smart_research_agent/security/content_moderator.py)（Luhn 校验 + IPv4 值域约束；`content_moderator.py` 100% 覆盖）
  - [../../day048/源码/smart-research-agent/smart_research_agent/api/middleware.py](../../day048/源码/smart-research-agent/smart_research_agent/api/middleware.py)（`AccessLogStore` 内存 + JSONL 双写、`AccessLogMiddleware` 横切）
  - [../../day048/源码/smart-research-agent/smart_research_agent/llm/local_runtime.py](../../day048/源码/smart-research-agent/smart_research_agent/llm/local_runtime.py)（`estimate_vram_gb` / `plan_deployment` / `fitting_quantizations` / `max_context_tokens` / `OllamaRuntime`；99% 覆盖）
  - [../../day048/源码/smart-research-agent/smart_research_agent/llm/local_model.py](../../day048/源码/smart-research-agent/smart_research_agent/llm/local_model.py)（`normalize_base_url` / `LocalModelSpec` / `LocalModelLLM`；100% 覆盖）
  - [../../day048/源码/smart-research-agent/smart_research_agent/evaluation/local_eval.py](../../day048/源码/smart-research-agent/smart_research_agent/evaluation/local_eval.py)（一致性 / 延迟 / 成本三指标；100% 覆盖）
- **M4 收口 · 集成（day046）**：
  - [../../day048/源码/smart-research-agent/smart_research_agent/integration/pipeline.py](../../day048/源码/smart-research-agent/smart_research_agent/integration/pipeline.py)（`STAGE_ORDER` 六阶段、`StageRecord`、`PipelineResult`、`_account` 的差值归因与叶子归因链）
  - [../../day048/源码/smart-research-agent/smart_research_agent/evaluation/perf_baseline.py](../../day048/源码/smart-research-agent/smart_research_agent/evaluation/perf_baseline.py)（`percentile` 最近秩法 + `PerformanceGuard` 判定下限）
  - [../../day048/源码/smart-research-agent/smart_research_agent/llm/cache.py](../../day048/源码/smart-research-agent/smart_research_agent/llm/cache.py)（`CachedLLM.last_used_llm`：跨 day046→047→048 的工单在本快照收口）
- **M5 起步 · 数据工程（day048）**：
  - [../../day048/源码/smart-research-agent/smart_research_agent/finetune/schema.py](../../day048/源码/smart-research-agent/smart_research_agent/finetune/schema.py)（`TrainingExample` / `prompt_text` / `parse_example` / `dump_jsonl`，108 语句 44 分支 100%）
  - [../../day048/源码/smart-research-agent/smart_research_agent/finetune/cleaner.py](../../day048/源码/smart-research-agent/smart_research_agent/finetune/cleaner.py)（`clean_text` 三段管道 / 七条 `QualityRule` / `dedupe_key` / `FilterReport`，93 语句 100%）
  - [../../day048/源码/smart-research-agent/smart_research_agent/finetune/collector.py](../../day048/源码/smart-research-agent/smart_research_agent/finetune/collector.py)（`JSONLSource` / `EvalTaskSource` / `RedTeamSource` / `DataCollector`，160 语句 100%）
  - [../../day048/源码/smart-research-agent/smart_research_agent/finetune/dataset.py](../../day048/源码/smart-research-agent/smart_research_agent/finetune/dataset.py)（`compute_stats` / `split_dataset` / `dump_bundle`，58 语句 100%）
  - [../../day048/源码/smart-research-agent/smart_research_agent/finetune/overview.py](../../day048/源码/smart-research-agent/smart_research_agent/finetune/overview.py)（`METHODS` / `recommend_method` / `lora_trainable_ratio` / `render_methods_table`，51 语句 100%）

## 本日的真实核对（可复现）

复习日不是"只读一遍"。本日写了两段**只读脚本**，在 day048 快照上重跑了四组核对（完整输出见 [../教程/教程.md](../教程/教程.md) 第六章与 [../习题答案/习题答案.md](../习题答案/习题答案.md)）。以下为结论摘要：

| 核对项 | 结果 |
|--------|------|
| 全量测试 | **1124 passed / 96.44%**（`fail_under = 90` 生效） |
| 数据产线 | 载入 41（18 + 7 + 16）→ 保留 37，`drop_reasons = {output_too_short: 1, placeholder_output: 1, unverified_source: 2}` |
| 数据集画像 | 37 条 / 平均指令 26.8 字 / 平均输出 88.1 字 / 平均 84.24 token / `tag_distribution` 26 键 |
| 切分复现性 | `seed=42` 两次切分完全一致；换 `seed=7` 评估集发生变化；37 条 → train 30 / eval 7 |
| `EvalTaskSource` 交叉核对 | `kept=5 / unverified=2 / skipped=1`，且与 `drop_reasons["unverified_source"] == 2` 严格对应 |
| 归因链（继承 day046/047 结论） | `MockLLM` 可直接归因；`ModelRouter` / `CachedLLM` 经 `last_used_llm` 落叶子；`_TimedLLMProxy` 经 `__getattr__` 全量透传 |
| LoRA 比例公式 | `rank*(in+out)/(in*out)`：4096/r8 = 0.3906%、4096/r16 = 0.7812%、2048/r8 = 0.7812%、11008/r16 = 0.2907% |
| `METHODS` 键顺序 | `['sft', 'lora', 'qlora', 'dpo', 'rlhf-ppo']` 的稳定顺序（同时决定 API 与文档表格） |

### 为 day050 预先算准的六个数字（本轮新增产出）

这六个数字**不依赖任何深度学习框架**，因此可以在今天全部确定；day050 写训练脚本时只需"把数字填进去"：

```text
训练集 30 条 / 评估集 7 条
per_device_train_batch_size=2  gradient_accumulation_steps=4  num_train_epochs=3.0
micro-batches/epoch = 30 // 2 = 15
optimizer steps/epoch = 15 // 4 = 3
total_steps = 9
warmup_steps = int(9 * 0.03) = 0      ← 退化：3% < 1/9 ≈ 11.11%，取整为 0
effective batch = 2 * 4 = 8 条/次更新
max_length 建议 = 128~256（平均 84.24 token/条；取 512 有 83.55% 是 padding，取 2048 有 95.89%）
```

## 推荐走读路线

按"防线 → 部署 → 焊接 → 矿藏"的顺序读，每天一条，正好对应 day044 / day045 / day046 / day048：

1. **防线（day044）**：`security/content_moderator.py` 的 `luhn_check` 与 IPv4 分支 → `api/middleware.py` 的 `AccessLogStore` 双写与 `AccessLogMiddleware` → `api/routes.py` 里审核被调用的位置（对照"审核落点在路由里，而非中间件里"）。
2. **部署（day045）**：`llm/local_runtime.py` 的 `estimate_vram_gb`（逐项对照公式，注意 `× 2` 是 K 与 V 两份张量）→ `plan_deployment` → `fitting_quantizations` → `max_context_tokens` → `llm/local_model.py` 的 `normalize_base_url` → `OllamaRuntime` 的四个原生端点 → `evaluation/local_eval.py` 的三个指标。
3. **焊接（day046）**：`integration/pipeline.py` 的 `run()` 逐行对照 `STAGE_ORDER` 与 `_account` 的**差值**逻辑 → `evaluation/perf_baseline.py` 的 `percentile`（最近秩法）与 `PerformanceGuard` → `api/routes.py` 的 `/pipeline/run` 与 `/pipeline/baseline`（缺基线返回 404）。
4. **矿藏（day048）**：`finetune/schema.py`（`TrainingExample` 与三格式往返）→ `cleaner.py`（三段管道 + 七条规则 + prompt 指纹）→ `collector.py`（三个源与两种相反的容错策略）→ `dataset.py`（画像、切分、落盘）→ `overview.py`（选型规则与 `METHODS`）。

每条路线都请对照 `tests/` 下的同名测试阅读——`MockLLM` / `TestClient` / 假时钟 / `tmp_path` 驱动的**离线测试是这些模块"行为契约"最精确的描述**，比教程更不容易过期。三处特别值得读：

- `tests/test_finetune_collector.py::test_distribution_on_real_data` 直接读 `data/eval/agent_tasks.jsonl` 断言 `kept=5 / unverified=2 / skipped=1`——**这条测试就是"评估集不能直接当训练集"的可执行版本**；
- `tests/test_finetune_overview.py` 里 LoRA 比例用**公式对照**而不是照抄实现；
- `tests/test_wrapper_accounting.py`（8 例）守住 `last_used_llm` 透传这条跨三天工单。

上一阶段的复习日文档见 [../../day047/源码/README.md](../../day047/源码/README.md)（R2 跨阶段，M3 ~ M4 全量）与 [../../day042/源码/README.md](../../day042/源码/README.md)（M4 前半段，day036~041），可与之对照衔接。

下一份快照（day050）将在本快照之上叠加 **SFT 监督微调基础**：把 day048 产出的 `train.jsonl` / `eval.jsonl` 渲染成监督文本、编码为 `input_ids` / `labels`（prompt 段置 `IGNORE_INDEX = -100`）、跑通一次真实训练循环，并用今天的六个数字作为超参。
