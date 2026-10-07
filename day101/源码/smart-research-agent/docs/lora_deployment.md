# LoRA 训练与部署指南（day052）

本文件是 M5-D4 的实操说明，回答四个问题：**适配器怎么存**、**多卡怎么配**、
**怎么合并上线**、**出问题先看哪里**。所有数字都可以用本仓库的代码复算——见第 7 节。

## 1. 适配器是一个三文件产物

一次 LoRA 训练结束后，交付目录里固定是三个文件：

| 文件 | 内容 | 缺了会怎样 |
|------|------|-----------|
| `adapter_config.json` | `r` / `alpha` / `dropout` / `target_modules` / 基座参数量 / 缩放公式 | 加载时用默认 `alpha`，**缩放悄悄变了一个倍数** |
| `adapter_model.json` | `A` 与 `B` 两个矩阵（只有它们） | 无权重可用 |
| `training_state.json` | 步数 / 学习率 / loss / 超参快照 / 词表大小 | 续训从第 0 步开始，学习率调度被重置 |

`load_adapter` 缺任何一个文件都抛 `AdapterCheckpointError`——**不返回"部分可用的对象"**。

参考模型（`V=557`、`r=8`、训 4 步）上的实测体量：

```text
adapter_config.json           435 B
adapter_model.json         272918 B
training_state.json          1325 B
合计                       274678 B（268.2 KiB）
```

### 内容哈希

```python
from smart_research_agent.peft import adapter_content_hash, build_manifest, write_manifest

manifest = build_manifest("outputs/lora/adapters/adapter-final", base_model="Qwen/Qwen3-0.6B")
write_manifest("outputs/lora/adapters/adapter-final", manifest)
print(manifest.summary_line())
# adapter-final | 基座 Qwen/Qwen3-0.6B | r=8 alpha=16 targets=['weight'] | step 4 | 274678 B | sha256:4aa1b146abb3 | 已合并 False
```

三个要点：

1. **哈希只覆盖配置与权重**，不含 `training_state.json`——它回答的是"权重是不是同一份"，把步数混进去会让答案随无关量漂移；
2. **序列化必须规范化**（排序键 + 紧凑分隔符），否则换一次键顺序就得到不同的哈希；
3. **清单随适配器一起走**（同目录的 `adapter_manifest.json`），里面有 `base_model`——**基座不匹配的适配器不会报错，只会给出莫名其妙的结果**。

### 清理与发布

```python
from smart_research_agent.peft import prune_adapters, repackage_adapter

prune_adapters("outputs/lora/adapters", limit=2)     # 中间副本保留最后 2 个
# adapter-final 永不删除；被删的是 adapter-step-2 这样的中间目录

repackage_adapter(
    "outputs/lora/adapters/adapter-final",
    "release/qa-adapter-v3",
    base_model="Qwen/Qwen3-0.6B",
    tags={"env": "staging"},
)
```

`repackage_adapter` 复制三个文件并**重新生成清单**，因此发布目录里不会留下训练过程中的临时文件。

## 2. 多卡与混合精度

### 2.1 先算账，再开机器

```python
from smart_research_agent.peft import MODEL_SPECS, LoRAConfig, plan_distributed
from smart_research_agent.sft import SFTTrainingArgs

plan = plan_distributed(
    MODEL_SPECS["llama-2-7b"],
    SFTTrainingArgs(per_device_train_batch_size=2, gradient_accumulation_steps=4, learning_rate=2e-4),
    devices=4,
    strategy="zero3",
    train_size=1000,
    lora_config=LoRAConfig(r=8, lora_alpha=16, target_modules="attention_all"),
)
print(plan.summary_line())
```

`plan` 给出：全局批、`steps_per_epoch` / `total_steps` / `warmup_steps`、缩放后的学习率、每设备显存、每步通信量、检查点体积，以及若干条风险告警。

### 2.2 分片策略：切哪几项决定省多少

```text
ddp     权重 + 梯度 + 优化器状态，每个设备一份完整副本
zero2   优化器状态 + 梯度按设备数分片，**参数不分片**
zero3   参数也分片（每层前向/反向各 all-gather 一次）
```

实测（`llama-2-7b` / 4 个设备 / bf16）：

| | ddp | zero2 | zero3 |
|---|---|---|---|
| 全参微调 | 75.31 GiB | **28.24 GiB**（2.67×） | **18.83 GiB**（4.00×） |
| LoRA（`attention_all` / `r=8`） | 12.65 GiB | **12.59 GiB**（1.00×） | **3.16 GiB**（4.00×） |
| 通信量/步（LoRA） | 0.025 GB | 0.025 GB | **26.98 GB** |

两条结论必须一起记：

- **LoRA 场景下 `zero2` 几乎不省**：它切的是优化器状态与梯度，而 LoRA 的优化器状态只有适配器那一百来 MB；占大头的是冻结的基座权重，只有 `zero3` 才会切；
- **`zero3` 的代价是通信量**：LoRA 上比 `zero2` 高三个数量级。**"显存够不够"与"跑得快不快"是两个问题。**

### 2.3 学习率要按法则缩放

```text
linear → lr × (new_batch / base_batch)      # 缺省，中小规模最常用
sqrt   → lr × sqrt(new_batch / base_batch)  # 更保守，批大小很大时更常见
none   → lr × 1                             # 会命中一条告警
```

实测（有效批 8 → 32，`lr=2e-4`）：`linear` → 8e-4、`sqrt` → 4e-4、`none` → 2e-4。

**放大批大小之后一定要重新看 warmup 与总步数**：两者都会变。

### 2.4 混合精度

| | 指数位 | 需要 loss scaling | 结论 |
|---|-------|------------------|------|
| `bf16` | 8 | 不需要 | Ampere 及之后优先 |
| `fp16` | 5 | **需要** | 精度略高但容易溢出为 `inf` |

**不要把 `TrainingArguments.bf16` 与 accelerate 的 `mixed_precision` 同时打开**：两套开关会各做一次 autocast 与缩放。

## 3. 生成配置与脚本

```python
from smart_research_agent.peft import (
    accelerate_config, deepspeed_zero_config, launch_command,
    render_config_yaml, render_accelerate_lora_script, render_merge_script,
)

config = accelerate_config(devices=4, strategy="zero2", mixed_precision="bf16",
                           deepspeed_config_file="ds_zero2.json")
print(render_config_yaml(config))
print(launch_command("train_lora_zero2.py", devices=4, strategy="zero2"))
# accelerate launch --config_file accelerate_config.yaml --num_processes 4 train_lora_zero2.py

open("deepspeed_zero2.json", "w").write(
    json.dumps(deepspeed_zero_config(stage=2, mixed_precision="bf16"), indent=2)
)
open("train_lora.py", "w").write(render_accelerate_lora_script(args, lora, devices=4, strategy="zero2"))
open("merge_adapter.py", "w").write(render_merge_script(lora))
```

配置字段名与官方一致，可以直接被 `accelerate launch --config_file` 消费。

**DeepSpeed 里的三个 `"auto"` 不要改成手写值**（`train_batch_size` / `train_micro_batch_size_per_gpu` / `gradient_accumulation_steps`）：它们由 accelerate 从 `TrainingArguments` 推导，写死会让"配置里的批大小"与"训练日志里的批大小"对不上。

## 4. 训练、续训、合并

### 4.1 训练与自动落盘

```python
from smart_research_agent.peft import LoRATrainer

trainer = LoRATrainer(model, tokenizer, args, adapter_save_steps=50, save_total_limit=2)
report = trainer.fit(train_examples, eval_examples)
print(report.summary_line())
# 可训练 8912（2.7875%）| 最终适配器 274678 B vs 全参 621612 B（省 2.3×）| 落盘 3 次
```

`adapter_save_steps=0`（缺省）表示只在训练结束时落盘一次。

### 4.2 续训

```python
trainer.restore("outputs/lora/adapters/adapter-final")   # 只加载适配器，不加载基座
```

`r` / `alpha` 与当前模型不一致时抛 `AdapterCheckpointError`。**续训前必须确认基座是同一个**——适配器可以换，基座不行。

### 4.3 合并前先过门禁

```python
from smart_research_agent.peft import verify_merge

verification = verify_merge(model, eval_batches)   # tolerance 缺省 0.0（逐位一致）
assert verification.passed, verification.summary_line()
# 检查 557 个上下文 | 逐位一致 True | 最大 logits 差 0.000e+00 | loss 6.3136913150 vs 6.3136913150 | 通过 True
```

`tolerance=0.0` 不是苛刻：两条路径做的是同一批浮点运算，结果本就应当相同（实测差 `0.00e+00`）。**如果传一个正数容差，那必须是显式的决定**——因为"合并错了"（O(1) 差异）与"浮点重排"（O(1e-7)）只差量级。

### 4.4 两种部署形态

```text
形态 A：一个基座 + N 个适配器（推理时 PeftModel.from_pretrained）
形态 B：merge_and_unload → save_pretrained（部署链路与普通模型一致）
```

实测（`llama-2-7b` 基座）：

| | 体积 |
|---|---|
| 基座（bf16） | 13 476 831 232 B（12.55 GiB） |
| 适配器 | 274 678 B（0.2620 MiB） |
| 10 个合并模型 | 134 768 312 320 B |
| 1 个基座 + 10 个适配器 | 13 479 578 392 B |
| 节省 | **10.00×** |

`merge_and_unload` **就地替换权重并丢弃适配器**，因此必须发生在所有需要适配器的动作之后；生成的合并脚本会在合并前打印适配器哈希——**不可逆动作的产物必须能指回它的来源**。

## 5. 出问题先看哪里

| 现象 | 多半是 | 怎么查 |
|------|--------|--------|
| 多卡脚本 `TypeError: got multiple values for keyword argument` | `gradient_accumulation_steps` 在 `TrainingArguments` 与内嵌 JSON 里各出现一次 | 渲染时从 JSON 里删掉它（`render_accelerate_lora_script` 已处理） |
| 配置里的 `mixed_precision: no` 被读成布尔 `False` | YAML 1.1 的布尔别名没加引号 | `_yaml_scalar` 里枚举保留字（已处理） |
| 多卡后训练明显变慢 | 全局批放大了但学习率没放大 | `lr_scaling_mode="linear"`；`plan_distributed` 有对应告警 |
| 每设备显存没降 | 策略选错了：LoRA 上 `zero2` 几乎不省 | 换 `zero3`，同时看通信量告警 |
| 适配器加载后效果不对 | `r` / `alpha` 与训练时不同，或基座不匹配 | 清单里的 `lora` 与 `base_model`；`load_adapter_state` 会校验 `scaling` |
| 落盘模型"半份" | ZeRO-3 未开 `stage3_gather_16bit_weights_on_model_save` | `deepspeed_zero_config(stage=3)` 已默认打开 |
| 四个进程写坏了适配器目录 | 没有只在主进程保存 | 生成的脚本里有 `if accelerator.is_main_process` |
| 训练时看到的 prompt 与推理不一致 | 显式传了 `system_prompt=None` 却被吞掉 | `LoRATrainer` 用哨兵区分"没传"与"显式 None"（已处理） |

## 6. 两个端点的等价查询

```bash
# 多卡计划 + accelerate 配置 + 启动命令 + 告警
curl -s http://127.0.0.1:8000/finetune/lora/distributed \
  -H 'Content-Type: application/json' \
  -d '{"model": "llama-2-7b", "devices": 4, "strategy": "zero3"}' | python -m json.tool

# 走一遍保存 → 清单 → 合并验证 → 体积对照（train_steps>0 时先在参考模型上真跑几步）
curl -s http://127.0.0.1:8000/finetune/lora/deploy \
  -H 'Content-Type: application/json' \
  -d '{"model": "llama-2-7b", "train_steps": 4}' | python -m json.tool
```

## 7. 复算本文件的全部数字

```bash
cd day052/源码/smart-research-agent
python scripts/lora_pipeline_demo.py     # 约 15 秒，打印本文件的每一张表
python -m pytest -q                      # 全量测试
```
