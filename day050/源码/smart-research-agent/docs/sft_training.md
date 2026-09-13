# SFT 监督微调训练指南（M5-D2）

本文档是 day050 的配套说明，覆盖三件事：**数据怎么变成监督样本**、
**超参怎么算出来**、**怎么从本课的离线参考实现切到真实框架**。

所有数字都来自本仓库的实测（`python scripts/sft_demo.py`），不是经验估计。

---

## 1. 全景

```text
day048 落盘产物                      day050 SFT 链路
─────────────────                    ─────────────────────────────────────
data/finetune/out/train.jsonl  ──┐
data/finetune/out/eval.jsonl   ──┤
                                 ├─→ render_supervised()      模板渲染（ChatML / Llama3 / Plain）
TrainingExample（内存对象）      │        ↓  产出 prompt_chars / supervised_chars 两个字符偏移
                                 │   encode_supervised()      分别编码再拼接，prompt 段 label 置 -100
                                 │        ↓  产出 input_ids / labels / attention_mask
                                 │   length_summary()         渲染后的长度分位数 → max_length
                                 │   plan_training()          步数 / warmup / 有效批（纯算术）
                                 │   SFTTrainer.fit()         梯度累积 + 调度 + 评估 + 落盘
                                 └─→ hf_script.render_*()     生成 transformers / TRL 训练脚本
```

三个模块边界（刻意的）：

| 层 | 模块 | 性质 |
|----|------|------|
| 纯函数 | `template.py` / `encoding.py` / `loss.py` | 无状态，每个数字都能手算验证 |
| 纯数据 + 纯算术 | `args.py` | 训练前就能算准，不需要框架 |
| 触碰外部世界 | `trainer.py` / `checkpoint.py` | 随机数、时间、磁盘 |

---

## 2. 数据怎么变成监督样本

### 2.1 三种模板

| 模板 | 前缀（不参与 loss） | 监督区间（参与 loss） |
|------|--------------------|---------------------|
| `chatml`（缺省） | `<\|im_start\|>system\n…<\|im_end\|>\n<\|im_start\|>user\n…<\|im_end\|>\n<\|im_start\|>assistant\n` | `答案<\|im_end\|>\n` |
| `llama3` | `<\|begin_of_text\|><\|start_header_id\|>…<\|end_header_id\|>\n\n…` | `答案<\|eot_id\|>` |
| `plain` | `系统提示\n\n### 指令\n…\n\n### 回答\n` | `答案\n` |

**轮次终止符计入监督区间**：模型必须学会"在哪里停下"，而 EOS/终止符正是
这个信号。不计入会让模型学会"永远继续写下去"。

`RenderedSample` 的不变式：

```python
text[prompt_chars : prompt_chars + supervised_chars] == 监督区间
prompt_chars + supervised_chars == len(text)
```

### 2.2 label mask：本课的核心机制

```text
input_ids = prompt_ids + answer_ids
labels    = [-100] * len(prompt_ids) + answer_ids
              ↑ 前缀段被屏蔽，只有答案段参与交叉熵
```

`-100` 是 PyTorch `nn.CrossEntropyLoss` 的 `ignore_index` 默认值，也是
HuggingFace / TRL 生态的通行约定。**用同一个约定值，意味着本课的数据可以
直接喂给 `transformers.Trainer` 而不需要任何转换。**

两个必须记住的实现细节：

1. **前缀与监督区间必须分别 tokenize 再拼接**，不能编码整段再按字符偏移
   切片——BPE 的合并可能跨越边界，切给谁都错。
2. **答案放不下的样本直接丢弃，绝不截断答案**。截断答案等于把"说了一半
   就停"教给模型。

### 2.3 LLaMA-Factory 与 TRL 的等价开关

| 本课实现 | TRL `SFTConfig` |
|----------|----------------|
| prompt 段 label 置 `-100` | `assistant_only_loss=True` |
| `max_length` 截断（保留答案） | `max_length`（旧名 `max_seq_length`） |
| 不做序列拼接 | `packing=False` |
| 只用 (prompt, completion) | `completion_only_loss=True` |

本课**刻意关闭 `packing`**：把多条短样本拼进一条序列会改变 loss 的分母
口径与样本边界，让"每步看到什么"不再可复现。在 37 条样本这个规模上，
吞吐收益远小于可解释性损失。

---

## 3. `max_length` 怎么定：本课最重要的实操教训

**不要按基座模型的上下文窗口定，要按"渲染并分词之后"的长度分位数定。**

实测（本课程数据集、ChatML、字符级分词器）：

```text
min=219  p50=248  p90=276  p95=302  max=320  mean=250.9
  其中 prompt 均值 151.8 / 监督区间均值 99.1
```

而 day048 `DatasetStats` 给出的估计是 **84.24 token/条**——相差 **2.98 倍**。
差距来自三处：

1. 系统提示词与角色标记（ChatML 约 100 字符）；
2. 轮次终止符；
3. 字符级分词把中文按约 1 字 1 token 计（BPE 会低得多）。

取 320 时 **0/30 条被截断**；若取 day048 的估计值 84，全部样本都会失败。
若取 2048，则有 **87.75%** 的位置是 padding（`1 - 250.9 / 2048`）——而
padding 同样参与注意力计算（对序列长度是平方复杂度）。

```python
from smart_research_agent.sft import render_supervised_list, length_summary, suggest_max_length

rendered = render_supervised_list(examples)
suggest_max_length(rendered, tokenizer, quantile=0.95)   # → 320
```

`suggest_max_length` 用 **最近秩法**（与 day046 `perf_baseline.percentile`
同一种纪律）：分位数必须对应一个真实存在的样本长度，插值出来的数字不是
任何一条样本的长度。

---

## 4. 超参：训练之前就能算准的部分

```python
from smart_research_agent.sft import SFTTrainingArgs, plan_training

args = SFTTrainingArgs(num_train_epochs=3.0)   # 其余用缺省
plan = plan_training(args, train_size=30, eval_size=7)
```

实测（30 条训练样本 / batch 2 / 累积 4 / 3 epochs）：

```text
micro_batches_per_epoch = 15
steps_per_epoch         = 3      （drop_last 口径）
total_steps             = 9
steps_per_epoch_flushing= 4      （HF Trainer 会 flush 尾部梯度）
total_steps_flushing    = 12
warmup_steps            = 0      ← 3% 的 warmup 在 9 步下取整为 0
effective_batch_size    = 8
```

### 4.1 两种"步数"口径必须说清

| 口径 | 计算 | 本课实测 |
|------|------|---------|
| `drop_last`（本课训练循环） | `(micro // accum) × epochs` | 9 |
| `flush` 尾部（HF `Trainer` 默认） | `ceil(micro / accum) × epochs` | 12 |

两者在数据量不是 `batch × accum` 整数倍时必然不同。`TrainingPlan` 会同时
给出两个数字，并在它们不等时告警——**做对照实验时不说口径，步数就不可比。**

### 4.2 学习率：没有绝对尺度

| 场景 | 合理量级 |
|------|---------|
| 7B 全参 + AdamW | `1e-5 ~ 5e-4`（`LR_SOFT_RANGE`） |
| LoRA / PEFT | `1e-4 ~ 5e-4`（`LR_SOFT_RANGE_PEFT`） |
| 本课参考模型（557×557 bigram + 纯 SGD） | `1 ~ 50`（`LR_SOFT_RANGE_REFERENCE`） |

实测（30 条样本、9 次更新，起点 loss = `ln(557) = 6.3226`）：

```text
lr = 2e-4  ->  6.3212 -> 6.3212   降幅  0.00%   ← 完全学不动
lr = 2.0   ->  6.3212 -> 6.2128   降幅  1.72%
lr = 4.0   ->  6.3212 -> 6.1070   降幅  3.39%
lr = 8.0   ->  6.3212 -> 5.9048   降幅  6.59%
lr = 20.0  ->  6.3212 -> 5.3933   降幅 14.68%
```

**照抄别人的学习率是本课最容易犯的错。** 告警区间必须与模型规模匹配，
否则它只会变成噪声（与 day046 性能基线的判定下限是同一条纪律）。

### 4.3 学习率调度

`lr_at(step, total_steps)` 按 HF `get_*_schedule_with_warmup` 的分段定义：
warmup 段线性升到峰值，其后按调度器衰减（cosine 半周期 / linear / constant）。

注意 `step = 0` 且 `warmup > 0` 时返回 **0.0**——`torch.optim.lr_scheduler.
LambdaLR` 在**构造时就会求值一次** `lr_lambda(0)`，因此 warmup 段的第 0 步
是一次空更新。训练循环必须照常执行它并清零梯度，否则这段梯度会漏进下一个
窗口。

### 4.4 其余七条告警

`plan_training` 生成的每一条 `warnings` 都对应一个真实失败模式：

1. drop_last 口径下每个 epoch 有 0 次更新（累积窗口比数据还大）；
2. 总步数过少（`< 20`）；
3. warmup 比例退化（`1/total_steps` 才换得到一步）；
4. `eval_steps` / `save_steps` 大于总步数（训练中永不评估/保存）；
5. 未提供评估集（无法判断过拟合）；
6. 学习率超出按模型规模标定的区间；
7. `max_length` 与真实长度分布不匹配（太小 → 截断；太大 → 纯浪费）；
8. 训练样本不足 500 条（day048 的选型规则）。

---

## 5. 参考实现与真实框架的接缝

### 5.1 参考模型（`reference_model.py`）

上下文长度 1 的 next-token 模型 + 真实梯度下降：

```text
前向：  logits = W[ctx] + b ；  probs = softmax(logits)
损失：  L = -(1/N) Σ_{t∈S} log probs[y_t]        S = 监督位置
梯度：  ∂L/∂logits_t = (probs_t - onehot(y_t)) / N
```

**它存在的唯一理由是让训练循环在离线、确定、零依赖的条件下真的发生一次。**
实测它确实在学：`loss 6.3212 → 5.4760`（降幅 13.37%，30 步），
评估 `5.3596`、困惑度 `212.64`（均匀分布是 557），最高概率达到均匀分布的
**13.46 倍**。

它**不是**用来训出可用模型的（bigram 没有能力做有意义的生成）。

### 5.2 "屏蔽 prompt 确实更好"的实证

两种训练法都用**同一把尺子**（评估集一律按屏蔽口径编码）测量答案 token
上的 loss：

```text
屏蔽 prompt（标准 SFT）  统一尺子下 5.3596
不屏蔽 prompt            统一尺子下 5.5475     差值 0.1879
```

不屏蔽时约 61% 的梯度被用在"预测用户会怎么提问"上——而提问在推理时是
给定的输入。

### 5.3 切到真实框架

```bash
pip install "transformers>=5.16" "datasets>=3.0" "accelerate>=1.0" "torch>=2.5"

# 生成脚本（参数与本课 SFTTrainingArgs 同源）
python -c "
from smart_research_agent.sft import SFTTrainingArgs, render_hf_sft_script
open('sft_train_transformers.py','w',encoding='utf-8').write(render_hf_sft_script(SFTTrainingArgs()))
"

python sft_train_transformers.py \
    --model_name Qwen/Qwen3-0.6B \
    --train_file data/finetune/out/train.jsonl \
    --eval_file data/finetune/out/eval.jsonl
```

走 TRL 路径则用 `render_trl_sft_script()`（依赖 `trl>=1.5`）。

生成的脚本里有三处与本课一一对应，改的时候不要动：

1. `kept_prompt = prompt_ids[-budget:]` —— 从左侧裁 prompt，保留靠近答案的指令；
2. `if budget < 1: return None` —— 答案放不下的样本丢弃，**不截断答案**；
3. `"labels": [IGNORE_INDEX] * len(kept_prompt) + answer_ids` —— label mask。

---

## 6. 检查点

`save_checkpoint` 固定写五个文件：

| 文件 | 内容 |
|------|------|
| `training_args.json` | 完整超参（含不属于 HF 的字段） |
| `tokenizer.json` | 词表（**没有它，`input_ids` 无法还原为文本**） |
| `model_state.json` | 权重 + 偏置 + 已更新次数 |
| `metrics.json` | 汇总指标 + 派生量 plan（附 `_meta` 自描述） |
| `train_log.jsonl` | 逐步的 `StepRecord`（画 loss 曲线、定位发散的那一步） |

末次检查点写在 `output_dir/final`，中间检查点写在
`output_dir/checkpoint-{step}`，按 `save_total_limit` 清理最旧的中间项
（`final` 永不删除）。实测 `save_total_limit=2`、`save_steps=10` 时，
写过 `checkpoint-10/20/30 + final`，落盘后保留 `checkpoint-20 / checkpoint-30 / final`。

---

## 7. 怎么跑

```bash
# 一键演示：数据 → 渲染 → 编码 → 长度分位 → 计划 → 训练 → 落盘 → 生成脚本
python scripts/sft_demo.py

# 单元测试（含本课的七个测试文件）
python -m pytest tests/test_sft_*.py -v

# 全量测试（覆盖率硬护栏 fail_under = 90）
python -m pytest -q
```

实测：全量 **1469 个用例通过、覆盖率 97.11%**，其中 `sft/` 包
（9 个文件、1016 语句、258 分支）**全部 100%**。

---

## 8. 常见错误对照表

| 现象 | 根因 | 修法 |
|------|------|------|
| loss 一直是 `ln(V)` 不动 | 学习率比模型规模小了数量级 | 按模型规模重新标定（见 4.2） |
| `SFTDataError: 答案需要 N 个 token…已超过 max_length` | `max_length` 按原始文本长度定的 | 用 `suggest_max_length()` 按渲染后分位取 |
| 训练"跑完了"但 loss 几乎不变 | `total_steps` 只有个位数 | 补数据，或调小 batch / 累积 |
| `drop_last 口径下每个 epoch 有 0 次参数更新` | 累积窗口比数据还大 | 调小 `gradient_accumulation_steps` |
| warmup 完全没效果 | `warmup_ratio < 1/total_steps` | 直接用 `warmup_steps`，或调大比例 |
| 显存远超预期 | `max_length` 过大，padding 参与计算 | 按分位数收小 `max_length` |
| 评估 loss 不降、训练 loss 降 | 过拟合（数据太少） | 先补数据；本题材下 37 条不可能训出可用模型 |
| 换模型后训练发散 | 沿用了上一个模型的学习率 | 学习率与批大小一起重新标定 |
