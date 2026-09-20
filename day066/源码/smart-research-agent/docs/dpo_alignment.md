# DPO 对齐实践：偏好数据、训练配置与损失函数（day055）

本文件是 day055 的工程说明，对应源码 `smart_research_agent/dpo/`。
教程见 [`../教程/教程.md`](../教程/教程.md)。

## 1. 包结构

| 模块 | 职责 | 是否触碰外部世界 |
|------|------|------------------|
| `dpo/errors.py` | `DPOError(ValueError)`：整包唯一的异常基类 | 否 |
| `dpo/config.py` | `DPOTrainingConfig`：TRL `DPOConfig` 的离线镜像 + 步数算术 | 否 |
| `dpo/losses.py` | 三种 DPO 损失的取值 / 梯度 / 数值核对 | 否 |
| `dpo/dataset.py` | 偏好数据构造（劣化算子 + 打分器 + 自洽体检）与 TRL 落盘 | 读 jsonl |

四个模块中只有 `dataset` 读磁盘，`losses` 是纯函数，`config` 是纯数据 + 纯算术——
所以整条链路**离线、确定性、零 GPU**，可以在没有一张显卡的机器上跑完整测试。

## 2. 偏好数据从哪来

工程上只有两条路，本模块用**两者的交集**：

| 路线 | 做法 | 代价 | 分布 |
|------|------|------|------|
| 拒绝采样 | 同一 prompt 采 K 个回答，打分取最高/最低 | 贵（K 倍推理） | 与当前策略一致（在线） |
| 规则劣化 | 以金标准为 chosen，用算子把答案"改坏" | 便宜、完全离线、可审计 | 人工构造，与真实错误不同 |

**交集**的读法：劣化算子负责**造出候选池**，规则打分器负责**排序**——
而"谁更好"的判据只有一份实现（`score_candidate`）。候选来源可以换（真实训练
换成模型采样即可），判据不能分家。

### 四个劣化算子

| 算子 | 对齐维度 | 把答案改坏的方式 |
|------|----------|------------------|
| `drop_tail` | `honesty` | 删掉最后一句：答了一半，信息不完整 |
| `vague_specifics` | `honesty` | 把具体数字/术语换成模糊说法 |
| `pad_verbose` | `conciseness` | 追加套话与铺垫：更长、更啰嗦、信息量不变 |
| `add_overclaim` | `honesty` | 追加无依据的效果承诺：把结论说过头 |

`drop_tail` 的维度归属是一个**真实的选择题**：它同时像"信息不全"与"答得太短"，
最终按"缺信息"归到 `honesty` 邻域——这类取舍必须写进数据里，而不是留在口头。

### 打分器：覆盖率 − 超长惩罚

`score_candidate` 只有两项：关键短语覆盖率与超长惩罚（`OVERLENGTH_PENALTY = 0.5`，
每超出金标准 1 倍扣 0.5 分）。**覆盖不了的维度就不打分**——`format` / `refusal`
需要结构化判定，本课把它们交给手写种子数据与安全偏好集，用一个字符级覆盖率
冒充"格式合规"比对未知更糟。

### 两条体检（不通过就丢弃并计数）

1. **自洽体检**：得分最高的候选必须是金标准，否则记 `gold_not_best` 丢弃。
   一条"chosen 并不更好"的样本在 DPO 里贡献的梯度方向是**反的**。
2. **区分度体检**：`score(gold) − score(rejected) >= MIN_SCORE_GAP`，否则记
   `insufficient_gap`。得分差接近 0 的一对几乎不产生梯度，却照样占一条样本位置。

### rejected 的挑选方式

- `worst`：取最低分候选。语义最像拒绝采样，但**实测会让四个算子退化成两个**
  （`drop_tail` 8 / `pad_verbose` 8）——"删掉半数句子"永远比"多说两句废话"错得更狠；
- `rotate`（默认）：按序号在适用算子之间轮转，牺牲"rejected 一定最差"，
  换来四个算子各占四分之一，于是"模型对哪类改坏方式区分度最差"才有数据可答。

**覆盖度比单条样本的极端性更重要。**

## 3. 训练配置：把 TRL DPOConfig 镜像成可测的算术

`to_dpo_config()` 的键名与 TRL `DPOConfig` **逐字对齐**，脚本里可直接
`DPOConfig(**cfg.to_dpo_config())`——不需要翻译表，翻译表就是漂移的来源。

三项固定值：`gradient_checkpointing=True`（显存换时间）、`bf16=True`、
`report_to="none"`（不依赖外部实验追踪）。

### 两个学习率

| 常量 | 值 | 用途 |
|------|-----|------|
| `DEFAULT_LEARNING_RATE` | `5e-6` | 真实 TRL 脚本的量级（比 SFT 小一个数量级） |
| `OFFLINE_LEARNING_RATE` | `0.5` | 离线参考模型上的标定值（沿用 day054 的量测） |

两个学习率差 5 个数量级不是笔误：真实模型上 DPO 的 lr 必须很小，否则
KL 约束会在几步内被冲垮；而离线参考模型是小规模数值模型，需要大步长才动得起来。

### 步数算术

```
micro_per_epoch   = train_pairs // per_device_train_batch_size
steps_per_epoch   = micro_per_epoch // gradient_accumulation_steps
total_steps       = steps_per_epoch * int(num_train_epochs)
warmup_steps      = int(total_steps * warmup_ratio)
```

以默认配置（batch 2 / 累积 2 / 3 epoch）跑 7 条偏好对：micro 3 → 每 epoch 1 步
→ 共 3 步，`warmup_steps = int(3 × 0.03) = 0`。**口径必须写清楚**：本课程按
"每个 epoch 各自取整再乘完整 epoch 数"计算；换成"先算总 micro-batch 再除以累积"
会得到不同结果——两种口径都有人用，关键是报告里写明用的是哪种。

三种"看似能跑、其实一次更新都不会发生"的配置会被提前拦下：

- `train_pairs` 不足一个 micro-batch；
- 每 epoch 的 micro-batch 数凑不满累积步数；
- `num_train_epochs < 1`（完整 epoch 数为 0）。

`warmup_floor(total_steps) = 1 / total_steps` 回答"warmup 是不是配置写错了"：
9 步对应的下限约 11.11%，默认 3% 必然退化。

## 4. 损失函数与梯度

| loss_type | 损失 | 梯度 | β 的角色 | 起点值（m=0） |
|-----------|------|------|----------|----------------|
| `sigmoid` | `−log σ(β·m)` | `−β·σ(−β·m)` | 温度：β 越大越激进 | `ln2` |
| `hinge` | `max(0, 1 − β·m)` | `−β`（`β·m<1`）否则 `0` | 间隔的倒数：超过 `1/β` 不再产生梯度 | `1` |
| `ipo` | `(m − 1/(2β))²` | `2·(m − 1/(2β))` | 目标 margin 的倒数 | `1/(4β²)` |

三条纪律：

1. **换损失函数，β 的语义会跟着换**——"随便换个 loss_type 试试"最容易被忽略的代价；
2. **起点值必须与损失函数配套**：`sigmoid` 是 `ln2`，`ipo` 是 `1/(4β²)`，
   沿用 ln2 做自检会漏掉真实的接线错误（`expected_zero_margin_tolerance` 就是判据）；
3. **解析梯度必须等于中心差分**（`numeric_gradient`，步长 `GRADIENT_CHECK_STEP = 1e-6`）。
   把数值核对放进生产代码而不只放进测试，是让"梯度确实是我写下的那个式子"
   这件事在学员眼前发生一次。

## 5. TRL 数据集的列约定

`TRL_COLUMNS = ("prompt", "chosen", "rejected")`——**只有这三列**。
多写的列会被 json 直读路径带进训练器；需要溯源时用 `include_meta=True`
追加 id / 维度 / 算子。

安全偏好集由 day031 的红队 payload 派生（`data/eval/safety_pairs.jsonl`），
覆盖 `prompt_injection` / `jailbreak` / `pii_leak` / `tool_abuse` 四个类别——
它提供的是"拒答边界"这一类**打分器覆盖不到**的偏好信号。

## 6. 运行方式

```bash
# 演示：五段数字，离线确定性
PYTHONPATH=. python scripts/dpo_demo.py

# 测试
python -m pytest tests/test_dpo_losses.py tests/test_dpo_config.py tests/test_dpo_dataset.py -q
```

> 注：本项目的 demo 脚本依赖 `PYTHONPATH=.`（或已安装的包），
> 与 `scripts/` 下其它脚本的运行方式一致。
