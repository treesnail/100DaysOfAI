# day056 源码说明

复习日无新增代码，也**不产生新的代码快照**（与 day007 / day014 / day021 / day024 / day028 / day035 / day042 / day047 / day049 同例）。

当前最新代码快照见 [../../day055/源码/smart-research-agent/](../../day055/源码/smart-research-agent/)（M5-D7 **DPO 对齐实践**完成后的完整累积快照，**4495 个测试全绿，覆盖率 97.75%**，覆盖率硬护栏 `fail_under = 90` 已生效）。

全部快照均为**累积式**：day055 的快照已包含 M5 前七课的全部代码与测试——`sft/`（8 文件）、`peft/`（10 文件）、`finetune_eval/`（8 文件）、`alignment/`（6 文件）、`dpo/`（4 文件），以及更早的 M1~M4 全部模块。走读时以 day055 快照为主即可。

> 本日与 day049 的区别：day049 是 **M5 起步期的缓冲日**，范围为 day044~day048，重点是把 M4 收尾三课与 M5 第一课串成交接链，并**为 day050 的 SFT 把超参与步数先算准**；**day056 是 M5 中场缓冲日**，范围为 day050~day055 六课，重点是把"从能训练到能交付"的六天收成**三本账**（步数账 / 资源账 / 证据账），并**逐项回访 day049 留下的那份预算**——五行被采纳、一行被 day050 的口径升级修正。

本日复习涉及的五段代码（均在 day055 快照内）：

- **步数账（day050）**：
  - [../../day055/源码/smart-research-agent/smart_research_agent/sft/template.py](../../day055/源码/smart-research-agent/smart_research_agent/sft/template.py)（`render_supervised` 产出 `prompt_chars` / `supervised_chars`；`length_summary` / `suggest_max_length`——`max_length` 由**渲染分词后**的长度分位定，本数据集 `p95 = 302`、`max = 320`）
  - [../../day055/源码/smart-research-agent/smart_research_agent/sft/encoding.py](../../day055/源码/smart-research-agent/smart_research_agent/sft/encoding.py)（前缀与答案**分别 tokenize 再拼接**，前缀段 label 置 `IGNORE_INDEX = -100`）
  - [../../day055/源码/smart-research-agent/smart_research_agent/sft/args.py](../../day055/源码/smart-research-agent/smart_research_agent/sft/args.py)（`SFTTrainingArgs` 字段名与 `TrainingArguments` 逐字对齐；`plan_training` 八类风险告警）
  - [../../day055/源码/smart-research-agent/smart_research_agent/sft/reference_model.py](../../day055/源码/smart-research-agent/smart_research_agent/sft/reference_model.py)（557×557 bigram + 纯 SGD；M5 全线"离线可跑"的地基）
- **资源账（day051）**：
  - [../../day055/源码/smart-research-agent/smart_research_agent/peft/layers.py](../../day055/源码/smart-research-agent/smart_research_agent/peft/layers.py)（`lora_delta` / `merge_lora_weight` / `matrix_rank`，秩上界 `min(r, in, out)`）
  - [../../day055/源码/smart-research-agent/smart_research_agent/peft/targets.py](../../day055/源码/smart-research-agent/smart_research_agent/peft/targets.py)（`MODEL_SPECS` / `parameter_check` / `theoretical_adapter_parameter_cap`）
  - [../../day055/源码/smart-research-agent/smart_research_agent/peft/qlora.py](../../day055/源码/smart-research-agent/smart_research_agent/peft/qlora.py)（NF4 码本 + `block=64` + 二级量化 → **4.126953 bit/参数**，不是 4）
  - [../../day055/源码/smart-research-agent/smart_research_agent/peft/memory.py](../../day055/源码/smart-research-agent/smart_research_agent/peft/memory.py)（`plan_memory` / `compare_strategies` 六项明细；全参策略里 AdamW 优化器状态占 53 907 324 928 字节）
- **交付（day052）**：
  - [../../day055/源码/smart-research-agent/smart_research_agent/peft/trainer.py](../../day055/源码/smart-research-agent/smart_research_agent/peft/trainer.py)（`save_adapter` 三文件 / `adapter_content_hash` 只覆盖配置与权重 / `prune_adapters` 保留 `adapter-final`）
  - [../../day055/源码/smart-research-agent/smart_research_agent/peft/accelerate.py](../../day055/源码/smart-research-agent/smart_research_agent/peft/accelerate.py)（`plan_distributed`：每设备显存与每步通信量）
  - [../../day055/源码/smart-research-agent/smart_research_agent/peft/deploy.py](../../day055/源码/smart-research-agent/smart_research_agent/peft/deploy.py)（`verify_merge` 逐位比较 logits，容差缺省 **0**）
- **证据（day053）**：
  - [../../day055/源码/smart-research-agent/smart_research_agent/finetune_eval/metrics.py](../../day055/源码/smart-research-agent/smart_research_agent/finetune_eval/metrics.py)（六个文本分量；**合格是合取**，加权总分只回答"好多少"）
  - [../../day055/源码/smart-research-agent/smart_research_agent/finetune_eval/compare.py](../../day055/源码/smart-research-agent/smart_research_agent/finetune_eval/compare.py)（Wilson 区间 / McNemar **精确**检验 / 配对自助法）
  - [../../day055/源码/smart-research-agent/smart_research_agent/finetune_eval/model_probe.py](../../day055/源码/smart-research-agent/smart_research_agent/finetune_eval/model_probe.py)（白盒探针：`lr=8` 训练集 14.46 / 评估集 587.64）
  - [../../day055/源码/smart-research-agent/smart_research_agent/finetune_eval/report.py](../../day055/源码/smart-research-agent/smart_research_agent/finetune_eval/report.py)（`verify_manifest_binding` + 三项门禁；**没有绑定信息的分数不是证据**）
- **对齐（day054 + day055）**：
  - [../../day055/源码/smart-research-agent/smart_research_agent/alignment/reward.py](../../day055/源码/smart-research-agent/smart_research_agent/alignment/reward.py)（隐式奖励：奖励 = 策略与参考模型的似然比）
  - [../../day055/源码/smart-research-agent/smart_research_agent/alignment/objectives.py](../../day055/源码/smart-research-agent/smart_research_agent/alignment/objectives.py)（β 的双重身份）
  - [../../day055/源码/smart-research-agent/smart_research_agent/dpo/losses.py](../../day055/源码/smart-research-agent/smart_research_agent/dpo/losses.py)（`LOSS_TYPES` / `loss_table` / `zero_margin_loss` / `numeric_gradient`）
  - [../../day055/源码/smart-research-agent/smart_research_agent/dpo/config.py](../../day055/源码/smart-research-agent/smart_research_agent/dpo/config.py)（`DPOTrainingConfig.step_plan` / `warmup_floor`）
  - [../../day055/源码/smart-research-agent/smart_research_agent/dpo/dataset.py](../../day055/源码/smart-research-agent/smart_research_agent/dpo/dataset.py)（`DEGRADATION_OPS` / `score_candidate` / `build_preferences` 的两条体检 / `TRL_COLUMNS`）

## 本日的真实核对（可复现）

复习日不是"只读一遍"。本日写了两段**只读脚本**，在 day055 快照上重跑了五组核对（完整输出见 [../教程/教程.md](../教程/教程.md) 第七章与 [../习题答案/习题答案.md](../习题答案/习题答案.md)）。以下为结论摘要：

| 核对项 | 结果 |
|--------|------|
| 全量测试 | **4495 passed / 97.75%**，耗时 1279.78s（`fail_under = 90` 生效） |
| 各日快照用例数（`pytest --collect-only` 实测） | day048 1124 → day050 1469 → day051 1952 → day052 2182 → day053 3333 → day054 4363 → day055 4495 |
| 数据产线 | 载入 41（18 + 7 + 16）→ 保留 37，`drop_reasons = {output_too_short: 1, placeholder_output: 1, unverified_source: 2}` |
| 数据集画像 | 37 条 / 平均指令 26.8 字 / 平均输出 88.1 字 / 输出长度 67~165 字 / 平均约 84.2 token/条 |
| 切分复现性 | `seed=42` 两次切分完全一致；37 条 → train 30 / eval 7 |
| LoRA 比例算术 | 四组 `d=4096 r=8` / `d=4096 r=16` / `d=2048 r=8` / `d=11008 r=16`，实现值与公式手算**逐位一致** |
| 显存三策略（字节级） | `full` 80 860 987 392 B（≈75.3 GiB）/ `lora` 13 577 371 136 B（≈12.6 GiB）/ `qlora` 3 577 432 321 B（≈3.3 GiB） |
| 偏好数据构造 | 18 → 清洗后 16（`output_too_short: 2`）→ 成对 16；`by_op = {drop_tail: 4, vague_specifics: 1, pad_verbose: 7, add_overclaim: 4}`；偏好对层面丢弃 0 |
| DPO 步数算术 | 7 对 → total 3 / warmup 0 / 退化 True；16 对 → total 12 / warmup 0 / 退化 True；32 对 → total 24 / warmup 0 / 退化 True |
| `warmup_floor` | 3→0.333333、12→0.083333、30→0.033333、100→0.010000；比例略小一档时，四组全部退化为 0 |

三条可从表里直接读出的结论：

1. **入口账六天后仍然逐项一致**（41 = 18 + 7 + 16，四类丢弃原因也一致）——M5 的入口没有被任何一课悄悄改坏。
2. **`4363 + 132 = 4495` 是闭合的**：day055 教程公布的"新增 132 个用例"与实测收集数严格对应。
3. **`warmup_degenerate` 在 3 / 12 / 24 步三组上全为 `True`**——"小数据集上 warmup 不生效"不是个位数步数的特例，两位数量级也一样（缺省 3% < 下限 4.17%）。
