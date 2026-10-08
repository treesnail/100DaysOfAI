# PyTorch 高级与综合实践：数据 / 设备 / 检查点 / 推理（day096 / M8-D7）

> 一本可随手翻的手册，与 `smart_research_agent/torch_pipeline/` 一一对应。
> 每一节的每个结论都能在包内找到对应函数；每个读数都能现场重跑
> （`scripts/torch_pipeline_demo.py` 十一节打印的就是它们）。

## 1. 这一课要回答什么

day095 把训练技巧装成四个旋钮，并留下两个工程问题：**数据怎么喂、状态怎么存**。
今天把这两个问题连同一件"设备算术"讲完，并把整条链跑成一个端到端的小项目：

```text
数据       不可变样本 + 内容指纹 + 按比例的确定性切分          —— 本包新建
采样       打乱只依赖 (seed, epoch)；worker 分片取模且并集为全集  —— 本包新建
装载       按批切分、可选丢尾批；三个读数（批数 / 丢弃 / 分片）  —— 本包新建
设备       参数 + 激活 + 优化器状态 → 字节数，与预算对账         —— 本包新建（纯算术）
训练       DataLoader → 前向 / 反向 → 范数守卫 → 优化器 step     —— 复用 day095 / day094 / day092
检查点     五件套 + 清单哈希；缺文件点名、哈希不符当场拒绝        —— 本包新建（纪律承自 day050）
推理       只走推理相；可注入时钟让延迟读数逐位复现              —— 本包新建
```

一句话本质：**"中途停下再恢复"与"一口气跑完"必须给出同一批参数——
这不是性能问题，而是"检查点到底存没存对"的定义问题。**

七个阶段与它们各自的实现者（`study.pipeline_lines()` 打印的就是这张表）：

| 阶段 | 一句话 | 由谁实现 |
|------|--------|---------|
| `dataset` | 一批**不可变**的 (输入, 标签)，带内容指纹 | 本包新建（`datasets`） |
| `sampler` | 把 epoch 号映射成一串**确定**的访问顺序 | 本包新建（`sampler`） |
| `dataloader` | 按批切分并可选丢弃尾批；先分片再切批 | 本包新建（`dataloader`） |
| `device` | 把参数量换成字节数，并与设备预算对账 | 本包新建（`device`，纯算术） |
| `train` | 前向 / 反向 / 梯度守卫 / 优化器 step / 每轮推理相评估 | 本包新建（`train`，转发 day095） |
| `checkpoint` | 五件套落盘 + 清单哈希，可校验、可恢复 | 本包新建（`checkpoint`） |
| `inference` | 只走推理相，带可注入时钟 | 本包新建（`inference`） |

## 2. 数据集与切分

数据集把"这一批样本"变成一个有身份的对象：名字 + 内容指纹。
指纹与 day052 的 `suite_fingerprint` 同算法（规范化 JSON 的 `sha256` 前 16 位）：

```text
sign           | 32 条样本 | 正 16 / 负 16 | 指纹 22bc0a3f878c7012
sign/train     | 24 条样本 | 正 10 / 负 14 | 指纹 f8f70f5ccf76bc9c
sign/eval      |  8 条样本 | 正  6 / 负  2 | 指纹 0f2166367b3b3e58
```

切分（`datasets.split_dataset`）只依赖 `(seed, n)`：

```text
① 用复用的 LCG（math_foundations.probability.uniforms）做 Fisher–Yates 打乱
② 切点 = round(n × eval_ratio)，再**夹到** [1, n−1]（保证两侧都非空）
③ 训练集与评估集的内容指纹都由各自的样本决定
```

"夹到 `[1, n−1]`"是写进手册的口径、不是静默兜底：`eval_ratio` 取 0 会让评估集为空
（"每轮评估"变成"每轮评估一个空集"），取 1 会让训练集为空，两者都在 `validate()` 里被拒绝。

随机性**不自己造**：打乱、切分、worker 分片用到的随机源只有一个
（`uniforms` 的 LCG），因此"这一份切分"可以被逐位复核。

## 3. DataLoader 的三个读数（批数、丢弃、分区）

```text
批数       drop_last=True：n // b；drop_last=False：ceil(n / b) = (n + b − 1) // b
丢弃       dropped = max(0, n − b × 批数)
分区       worker k 的分片 = { i : i % workers == k }（互不相交、并集为全集）
```

真实读数（训练集 24 条、b=7）：

```text
装载器 | sign/train | b=7 | shuffle=True | drop_last=False | 批数 4 | 丢弃 0
装载器 | sign/train | b=7 | shuffle=True | drop_last=True  | 批数 3 | 丢弃 3
```

"丢弃"为什么用 `max(0, …)`：`drop_last=False` 时尾批是**被保留**的，
`n − b × ceil(n/b)` 会得到负数——那个 `max` 正是"保留尾批"与"丢弃尾批"的分界。

**顺序是"先分片、再切批"**：`Sampler.order(epoch)` 先给出全局访问顺序，
`shard_indices` 挑出属于本 worker 的**位置**，最后才按 `batch_size` 切块。
这样"一个样本只进一个 worker 的批"是结构性事实，而不是靠"块大小恰好整除"侥幸成立。

```text
worker 0：6 条 | 前 6 个下标 (0, 4, 8, 12, 16, 20)
worker 1：6 条 | 前 6 个下标 (1, 5, 9, 13, 17, 21)
...
并集覆盖 1.0，重叠项数 0
```

## 4. 设备与显存的算术

设备层**不探测硬件**：它一次 `import torch` 都不做、一次 `cuda` 查询都不发，
所有读数都是**算术**，因此可以在任何机器上逐位复算。

```text
tensor_bytes(shape, dtype)            = (Π shape_i) × DTYPE_BYTES[dtype]
parameter_bytes(count, dtype)         = count × DTYPE_BYTES[dtype]
activation_bytes(batch, width, dtype) = batch × width × DTYPE_BYTES[dtype]
plan_device(...)                      = params + activations + optimizer_multiplier × params
```

真实读数（宽度 4 的 lstm，114 个参数，批 8）：

```text
cpu/float32 | 参数 456 B + 激活 128 B + 优化器 1368 B = 1952 B / 8.0 GiB | fits=True
cuda/float32 | 参数 456 B + 激活 128 B + 优化器 1368 B = 1952 B / 4.0 GiB | fits=True
mps/float32 | 参数 456 B + 激活 128 B + 优化器 1368 B = 1952 B / 2.0 GiB | fits=True
```

**"装不下"不是异常，而是一个读数**：

```text
放不进预算 ⇒ device_plan.fits is False + 一行 warn_line()
          ⇒ **不抛异常**：它是可以换批大小 / 换精度解决的工程选择
```

只有"设备名 / 精度名不在白名单里"（`CUDA ` / `float32x`）才抛本包专属族的 `DeviceError`——
那是"这台机器上没有你要的东西"，既不是形状问题、也不是数值范围问题。

## 5. 检查点五件套

```text
meta.json        清单：step / 权重哈希 / 优化器哈希 / 文件表 / 参数个数
params.json      压平后的权重（flatten_params 的产物）
optimizer.json   优化器的跨步状态（存什么由 OPTIMIZER_STATE_KEYS 决定）
metrics.json     这一轮的汇总读数（含 running 统计量与 initial_loss，供恢复用）
history.jsonl    逐轮的 EpochRecord（一行一条，便于画曲线）
```

真实读数（一次最小检查点）：

```text
meta.json        |  345 字节 | 在场=True
params.json      | 2830 字节 | 在场=True
optimizer.json   |  426 字节 | 在场=True
metrics.json     |   34 字节 | 在场=True
history.jsonl    |    0 字节 | 在场=True
```

三条纪律：

```text
① meta.json **最后写**：它在，说明前面四件都写完了
② 缺文件必须**点名**（"缺少 params.json"而不是"检查点坏了"）
③ 清单里的哈希**只覆盖**权重与优化器状态：指标与历史是可读记录、允许被追加
```

第 ③ 条是"为什么哈希不覆盖 metrics"的答案：把可变记录纳入完整性校验，
会让"恢复训练时往 history.jsonl 追加一行"莫名其妙地把检查点判成损坏。

## 6. 恢复训练为什么必须逐位相等

这是本课最值钱的一条性质（第 ⑥ 条）。它一次性检验四件事：

```text
① 打乱只依赖 (seed, epoch)          ⇒ 第 k+1 轮的批与不中断时**逐位相同**
② 训练相的归一化只用本批统计量       ⇒ 训练轨迹与 running 统计量无关
③ dropout 恒等（本课不设丢弃率）     ⇒ 前向没有随机性
④ 检查点装回了**优化器的跨步状态**   ⇒ Adam 的一阶 / 二阶动量不从零开始
```

真实读数（数据集 32 条、批 8、6 轮；中途第 3 轮落检查点）：

```text
一口气训到第 6 轮：114 个参数
第 3 轮落盘后恢复：114 个参数
最大绝对差：0.0
```

其中第 ④ 条要求"把优化器状态装回去"：`day092` 的 `TrainingOptimizer` 把共享规则的
状态放在内部 `base`（day074 的对象）里，新规则（nesterov / rmsprop）的状态在顶层——
`train._restore_optimizer` 用 `OPTIMIZER_STATE_KEYS` 决定装哪些键，两处都装。
换优化器恢复会被当场拒绝（`CheckpointError`）：一个"看着能跑、其实动量对不上"的起点
比直接报错危险得多。

## 7. 推理封装与两个相

```text
phase="eval"   ⇒ BatchNorm 用 running 统计量、dropout 恒等   ← 推理只走这一相
phase="train"  ⇒ BatchNorm 用**这一批自己的**统计量、dropout 置零
```

推理时用训练相**不会报错**：它只让"单条样本的预测"随批里还有谁而变，
而所有形状检查都会通过。因此 `inference` 把 `phase` 写死成 `eval`，
不给调用方留一个可以填错的参数；同时它把 `running` 统计量作为构造参数带进来
（推理相的归一化读的就是它，宽度对不上当场抛 `ShapeError`）。

```text
predict_one          一条样本的预测（一条也是它）
predict_batch        一批样本的预测（一次前向；与逐条顺序调用逐位相同）
predict_dataset      整份数据集 → argmax 标签
```

## 8. 延迟读数怎么才可复现

```text
默认 clock = time.perf_counter（真实墙钟，读数**不可复现**）
测试 / 性质 注入一个假 clock：每次调用返回一个**预先写死或固定步进**的数
  ⇒ seconds_per_sample = seconds / samples 变成逐位可复算的读数
```

真实读数（注入每次前进 `1e-6` 秒的假 clock，数据集 32 条）：

```text
b=1  | 调用 32 次 | 样本 32 | 0.000032s | 每条 0.000001000s | 吞吐 1000000.0/s
b=4  | 调用  8 次 | 样本 32 | 0.000008s | 每条 0.000000250s | 吞吐 4000000.0/s
b=8  | 调用  4 次 | 样本 32 | 0.000004s | 每条 0.000000125s | 吞吐 8000000.0/s
```

把时钟做成**构造参数**而不是模块级全局，是"一个量只写一遍"在时间维度上的样子：
读数只有一个来源，而那个来源可以被替换成一个确定的替身。
倒退或非有限的时钟当场抛 `NumericError`：延迟读数的第一条前提是它非负。

## 9. 端到端小项目

`train.train_pipeline` 把七件东西按一个**写死**的回路串起来：

```text
构造 DataLoader → 每个 epoch 打乱 → 逐批前向 / 反向（复用 day095 的 loss_and_grad）
  → 调用 day094 的 check_gradient_norm 守卫 → 优化器 step（make_train_optimizer）
  → 每 epoch 用推理相评一次 → 若给了 checkpoint_dir 则落一次五件套检查点
```

真实读数（数据集 32 条、批 8、10 轮）：

```text
10 轮 × 3 批 | 训练 2.026e+00 → 3.892e-02（↓98.1%）| 推理 2.447e-02（最好 2.447e-02）| 30 步 | 丢弃 0
epoch   1 | 训练 1.244526 | 推理 0.726792 | 准确率 25.0% | lr 0.050000
epoch   2 | 训练 0.352426 | 推理 0.773981 | 准确率 25.0% | lr 0.050000
```

训练损失与评估损失是**两把尺子**：训练损失用本批统计量、评估损失用 running 统计量。
只报其中一把会把"表示会漂移"这件事看反，因此报告把初始损失、训练损失、评估损失、
最好评估损失四个数都给出来。

## 10. 七条性质与失败族

```text
① shuffle_order_is_deterministic       相等   同 seed 两次 order 的差异项数 = 0
② worker_shards_partition_the_dataset  下界   并集覆盖 = 1.0 且不相交项数 = 0
③ drop_last_matches_batch_formula      相等   批数与 n // b 的差 = 0
④ device_bytes_match_formula           上界   total_bytes 与手算的相对差 <= 1e-9
⑤ checkpoint_round_trip_is_bitwise     相等   save → load 之后压平参数逐位相同
⑥ resume_matches_uninterrupted         相等   中途恢复与一口气训到第 N 轮参数差 = 0   ← 核心
⑦ predict_batch_equals_sequential      相等   批量推理与逐条推理的输出差 = 0
```

判据三类：**相等**（`upper_bound=0.0`）、**不超过上界**（`upper_bound>0`）、
**不低于下界**（`lower_bound`）。第 ② 条是唯一的下界——

```text
把取模分片改写成切块分片（range(k·b, (k+1)·b)）
  ⇒ 换个 workers 就会漏样本 / 重叠
  ⇒ 而它**不会报错**：只是少数样本被喂两遍、或几条从来没被训过
  ⇒ 只有"并集覆盖 = 1.0"这条下界能把它钉死
```

五个失败族（`errors.FAMILY_OUTCOMES`）：

| 族 | 判据 | 该谁去修 |
|----|------|---------|
| `ShapeError` | 索引 / 批的列数 / 压平长度对不上 | 改调用 |
| `ParameterError` | batch_size / eval_ratio / workers 取值非法 | 改调用 |
| `NumericError` | 空数据集 / 某一侧被切空 / 比例非法 | 改数据或改实现 |
| `DeviceError` | 未知设备名 / 精度名 | 改调用（换白名单内的取值） |
| `CheckpointError` | 五件套缺一个 / 哈希不符 / JSON 损坏 | 改文件或改流程 |

本课请回了 `CheckpointError`（day050 首次命名、此后缺席），
而 `GradientError` 继续缺席：训练回路里唯一的梯度控制流事件仍是 day094 的
`check_gradient_norm`，本课只是**调用**它、不重新抛。

## 11. 五条边界（这一课明确不承诺的事）

```text
1. 不真的探测硬件：设备与预算是**算术表**，不 import torch、不做 cuda 查询。
2. 不做真正的多进程：worker 分片是**索引层的划分**，为的是可复算的读数。
3. 不做异步 IO / 预取：DataLoader 是纯同步的批切分器，延迟读数由假 clock 钉住。
4. 不新增第三方依赖：全部纯标准库实现，不 import torch / numpy。
5. 不改动 day095：网络、损失与梯度全部**调用** regularization 的那一份。
```

## 12. 与既有包的接缝（M8-D6 → M8-D7 的过渡）

- **上游**：`regularization`（day095：网络、损失、梯度、压平契约）、
  `sequence_models`（day094：梯度范数守卫、±1 符号数据集）、
  `optimizers`（day092：更新规则与 `OPTIMIZER_STATE_KEYS`）、
  `math_foundations`（day073：LCG）、`sft.checkpoint`（day050：五件套纪律的来源）；
- **不改动任何既有模块**：day095 的 `regularization` 一行未改——本课是它的**使用者**；
- **脚下**：`config` **没有**新增配置项——设备、批大小、检查点目录全部进
  `train.PipelineConfig`（它是"这一次训练"的判据，而不是服务级默认值）；
- **下游**：day097（R3 复盘）会把 M7～M8 串成一条线，本课的读数就是它的素材。

一次跑完六张表 + 阶段线：`study.study_lines()`（演示脚本第 11 节打印的消融读数来自
`study.ablation_rows()`，其中"丢尾批"比"基准"少一批、丢 3 条）。
