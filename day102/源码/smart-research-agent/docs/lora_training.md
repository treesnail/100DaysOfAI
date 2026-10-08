# LoRA / QLoRA 微调指南（day051）

本文件是 M5-D3 的实操说明。它回答三个问题：**要训多少参数**、**要占多少显存**、
**怎么在真实框架里跑起来**。所有数字都可以用本仓库的代码复算——本节末尾给出复算命令。

## 1. 什么时候用 LoRA，什么时候不用

| 情形 | 建议 | 理由 |
|------|------|------|
| 数据 < 500 条 | **先补数据** | day048 的选型规则；本课实测：30 条样本上秩与效果的关系是非单调的 |
| 单卡装不下全参微调的优化器状态 | 用 LoRA | 7B 全参 AdamW 需要 75.31 GiB，LoRA 只要 12.65 GiB |
| 单卡连 16-bit 基座都装不下 | 用 QLoRA | 7B 的 4-bit 基座是 3.33 GiB |
| 需要"一个基座 + N 个业务适配器" | 用 LoRA | 适配器可独立落盘（十几 MB）并热插拔 |
| 任务需要模型学到全新的知识/语言 | 不用 LoRA | 低秩增量的容量上限是 `min(r, in, out)`，装不下结构性的新能力 |

一句话：**LoRA 省的是"参数与显存"，不是"数据与算力"。** 它的前向反向仍要过整个基座。

## 2. 依赖与安装

本仓库的**参考实现（离线可跑）不需要任何额外依赖**：`smart_research_agent/peft/`
只用标准库。真实训练需要：

```bash
# LoRA
pip install "transformers>=5.16" "datasets>=3.0" "accelerate>=1.0" "torch>=2.5" "peft>=0.20"

# QLoRA 再加一个 4-bit 算子库
pip install "bitsandbytes>=0.50"
```

也可以让代码生成安装命令：

```python
from smart_research_agent.peft import peft_dependency_commands

peft_dependency_commands(use_qlora=True)["pip"]
```

## 3. 三条使用路径

### 3.1 离线参考实现（本课用，秒级、可手算）

```python
from smart_research_agent.peft import (
    LoRAReferenceModel,
    default_reference_lora_config,
)
from smart_research_agent.sft import ReferenceSFTModel, SFTTrainer, SFTTrainingArgs

model = LoRAReferenceModel(
    ReferenceSFTModel(vocab_size, seed=42),
    default_reference_lora_config(r=8, lora_alpha=16),
    seed=42,
)
trainer = SFTTrainer(model, tokenizer, SFTTrainingArgs(learning_rate=20.0, num_train_epochs=2.0))
report = trainer.fit(train_examples, eval_examples)
```

**`SFTTrainer` 一行都不用改**：`LoRAReferenceModel` 与 day050 的 `ReferenceSFTModel`
接口完全一致（`accumulate` / `apply_update` / `zero_grad` / `evaluate` / `state_dict`）。
LoRA 改变的只有"哪些参数有梯度"，渲染、编码、label mask、padding、梯度累积、
学习率调度、检查点落盘全都与它无关。

### 3.2 生成 LoRA 训练脚本（真实基座）

```python
from smart_research_agent.peft import LoRAConfig, render_lora_script
from smart_research_agent.sft import SFTTrainingArgs

script = render_lora_script(
    SFTTrainingArgs(learning_rate=2e-4, num_train_epochs=2.0, max_length=320),
    LoRAConfig(r=8, lora_alpha=16, target_modules="attention"),
)
open("lora_train.py", "w", encoding="utf-8").write(script)
```

生成的脚本里，LoRA 相对 day050 的全参脚本只多了两行：

```python
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, **LORA_CONFIG)
model = get_peft_model(model, peft_config)
```

### 3.3 生成 QLoRA 训练脚本（4-bit 基座）

```python
from smart_research_agent.peft import QLoRAConfig, render_qlora_script

script = render_qlora_script(
    SFTTrainingArgs(learning_rate=2e-4, num_train_epochs=2.0, max_length=320),
    LoRAConfig(r=8, lora_alpha=16, target_modules="attention"),
    QLoRAConfig(bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, block_size=64),
)
```

脚本比 LoRA 版多三处，每一处都有明确目的：

1. `BitsAndBytesConfig`：基座按 4-bit 载入（每参数 0.515869 字节，bf16 是 2 字节）；
2. `prepare_model_for_kbit_training(model)`：冻结基座、把 LayerNorm 提升到 fp32、
   启用输入梯度。**漏掉它就会得到一个"训练在跑但 loss 不降"的脚本**；
3. `bf16=True`：反量化后的计算精度是 bf16——**适配器自始至终是 bf16**。

## 4. 参数选择速查

| 参数 | 缺省 | 怎么定 |
|------|------|--------|
| `r` | 8 | 从 8 或 16 起步；`2r/d` 是单矩阵比例，`r=min(in,out)` 以上不再增加容量 |
| `lora_alpha` | 16 | 只与 `r` 的**比例**有意义（`alpha/r`）。`alpha=2r` 是社区惯例 |
| `lora_dropout` | 0.05 | 只作用在适配器分支的输入上；one-hot（查表）输入下退化为"样本级丢弃整条分支" |
| `target_modules` | `attention`（q/v） | 预设名或模块名元组。**没命中预期模块只会表现为"效果不如预期"，不会报错** |
| `bias` | `none` | 保持 `none`，「可训练参数 = r(in+out)」这条手算公式才成立 |
| `use_rslora` | `False` | 打算调大 `r` 时打开：把缩放换成 `alpha/sqrt(r)`，让不同秩之间可比 |
| `learning_rate` | — | **必须重新标定**，见下一节 |

## 5. 学习率：本课最重要的实测结论

在 30 条样本、30 步的同一预算下（评估 loss，越低越好）：

| 方法 | lr=2 | lr=8 | lr=20 | lr=32 | lr=100 | lr≥128 |
|------|------|------|-------|-------|--------|--------|
| 全参微调 | — | 5.3602 | 4.6473 | 4.2206 | **3.4344** | 发散（5.00 → 13.59） |
| LoRA r=8 | 6.2977 | 4.7815 | **3.8843** | 发散（3.4e15） | — | — |

两条结论：

1. **LoRA 的学习率区间比全参窄得多**：全参在 `[8, 100]` 都稳定（约 12 倍宽度），
   LoRA 在 `[8, 20]` 之后就爆掉（约 2.5 倍宽度）。把 day050 标定出的 `lr=8`
   照抄给 LoRA 会"能跑但学不动"（4.7815 vs 3.8843）。
2. **不要相信"LoRA 效果更好"的说法**：把两边都调到最优，全参（3.4344）优于
   LoRA r=8（3.8843）。这符合预期——LoRA 用秩 ≤ 8 的增量去逼近 310 249 维的
   全秩更新，容量差距是结构性的。**LoRA 的收益是显存与落盘体积（5.96× / 22.61×），
   不是效果。**

`LR_SOFT_RANGE_PEFT = (1e-4, 5e-4)` 是 **7B LoRA** 的经验区间；参考模型（557×557
bigram + 纯 SGD）的量级是 `1e0 ~ 2e1`，两者相差四个数量级。

## 6. 显存与落盘体积核算

```python
from smart_research_agent.peft import LoRAConfig, MODEL_SPECS, compare_strategies, savings_table

plans = compare_strategies(
    MODEL_SPECS["llama-2-7b"],
    lora_config=LoRAConfig(r=8, lora_alpha=16, target_modules="attention_all"),
)
for row in savings_table(plans):
    print(row["strategy"], row["as_gib"], row.get("savings_factor"))
```

实测（llama-2-7b / attention_all / r=8，不含激活显存）：

| 策略 | 可训练参数 | 总显存 | 相对全参 | 最少可用卡 |
|------|-----------|--------|---------|-----------|
| 全参微调 | 6 738 415 616 | 75.31 GiB（80.86 GB） | 1.00× | A100/H100 (80 GiB) |
| LoRA | 8 388 608 | 12.65 GiB（13.58 GB） | 5.96× | RTX 4090 (24 GiB) |
| QLoRA | 8 388 608 | 3.33 GiB（3.58 GB） | 22.61× | RTX 4090 (24 GiB) |

**这张表不含激活显存**——它取决于 `batch × sequence_length`，无法只由模型规格决定。
"放得下"不等于"跑得起来"。

**单位陷阱**：GiB 与 GB 相差 7.4%，"13.48 GiB"写成 GB 就是 14.47 GB。
在"能不能放进 24 GB 卡"这种判断上混用单位会给出错误答案，所以所有报告两个都给。

## 7. 4-bit 到底占多少

| block_size | 二级量化 | 单重量化 |
|-----------|---------|---------|
| 64 | 0.515869 B/参数 | 0.562500 B/参数 |
| 128 | 0.507935 B/参数 | 0.531250 B/参数 |
| 256 | 0.503967 B/参数 | 0.515625 B/参数 |
| 512 | 0.501984 B/参数 | 0.507812 B/参数 |

**4-bit 并没有把权重压到 0.5 字节**：常数开销随块变小而变大。方向容易记反——
块越大、常数开销越小、精度越低。

## 8. 常见故障与排查

| 现象 | 多半是 | 怎么查 |
|------|--------|--------|
| 可训练参数远少于预期 | `target_modules` 没命中 | 生成的脚本里有 `count_trainable()`，偏差 > 20% 会告警 |
| `loss` 一开始就爆炸 | 学习率照抄了全参的值 | 见第 5 节；LoRA 的可用区间更窄 |
| loss 曲线是一条水平线 | 学习率小了四个数量级 | `LR_SOFT_RANGE_PEFT` vs 参考模型的量级 |
| `loss` 正常但效果没变好 | 增量被"白算"（例如 `r` 超过 `min(in,out)`） | `plan_lora` 会给出对应告警 |
| 合并后输出全错 | 行/列约定用混了 | `ΔW=0` 时 `logits` 必须与基座逐位一致（本课踩过这个坑） |
| 适配器加载后效果不对 | `r` / `alpha` 变了却没重新合并 | `load_adapter_state` 会校验 `scaling` 是否一致 |

## 9. 复算本文件的全部数字

```bash
cd day051/源码/smart-research-agent
python scripts/lora_demo.py     # 约 40 秒，打印本文件的每一张表
python -m pytest -q             # 全量测试（含 peft 包）
```

三个 HTTP 端点也给出同样的算术（只读，不需要 GPU）：

```bash
curl http://localhost:8000/finetune/lora/defaults
curl -X POST http://localhost:8000/finetune/lora/plan -H "Content-Type: application/json" -d '{}'
curl -X POST http://localhost:8000/finetune/lora/memory -H "Content-Type: application/json" -d '{}'
```
