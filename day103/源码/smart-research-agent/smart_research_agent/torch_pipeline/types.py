"""``torch_pipeline`` 的口径表（day096 / M8-D7）.

一次性把这一课的"名词表"写全：**七个阶段 / 四条读数轴 / 三个设备 / 四种精度 /
七条性质 / 十条笔记 / 五条边界 / 十张公式**。全部是常量，因此可以被测试逐键检查。

```text
七个阶段   dataset → sampler → dataloader → device → train → checkpoint → inference
四条读数轴 batch（批数 / 丢弃）/ worker（多进程分片）/ device（显存字节）/ time（延迟）
三个设备   cpu / cuda / mps（一张白名单，写错了抛 DeviceError）
四种精度   float32 / float64 / float16 / bfloat16（每种一个字节数）
```

## 这一课的位置：把 day095 的链"搬进工程流程"，而不是重写它

```text
day089  最小的零件（Dense / 激活 / 交叉熵）
day090  梯度怎么回传（链式法则 + 累加）
day092  拿到梯度后一步走多远（优化器）
day093  空间的权重共享（卷积）
day094  时间的权重共享（RNN / LSTM，BPTT）
day095  训练技巧：归一化 / 随机丢 / 学习率衰减 / 早停
day096  **工程流程**：DataLoader / 设备与显存算术 / 检查点 / 推理封装 / 端到端小项目
```

因此本课的**不变量**是"不改 day095 的一行"：``regularization`` 既是上游
（提供网络、损失与梯度），也是被实验的对象——本课只把"数据怎么喂、状态怎么存、
结果怎么读"这三件事写清楚。

## 一条纪律：读数要能复算

```text
批数          n // b（drop_last）或 ceil(n / b）
丢弃样本      n − b × 批数
设备字节      params + activations + optimizer_multiplier × params
延迟          seconds / samples（假 clock 之下逐位可复现）
```

上面四个式子各只有一处实现（``dataloader`` / ``device`` / ``inference``），
本模块只把它们的**口径**写成常量，供报告与测试逐键核对。
"""

from __future__ import annotations

from smart_research_agent.torch_pipeline.errors import TorchPipelineError

# --------------------------------------------------------------------------- #
# 闭合表 1：七个阶段（含"谁实现了它"）
# --------------------------------------------------------------------------- #

#: 七个阶段（顺序 = 数据流动的顺序）.
STAGES: tuple[str, ...] = (
    "dataset",
    "sampler",
    "dataloader",
    "device",
    "train",
    "checkpoint",
    "inference",
)

#: 每个阶段的一句话解释.
STAGE_DESCRIPTIONS: dict[str, str] = {
    "dataset": "数据集：一批**不可变**的 (输入, 标签)，带内容指纹",
    "sampler": "采样器：把 epoch 号映射成一串**确定**的访问顺序（同种子同顺序）",
    "dataloader": "装载器：按批切分并可选丢弃尾批；多进程时先分片再切批",
    "device": "设备：把参数量换成字节数，并与设备预算对账（不足由 fits 暴露）",
    "train": "训练：前向 / 反向 / 梯度守卫 / 优化器 step / 每轮推理相评估",
    "checkpoint": "检查点：五件套落盘 + 清单哈希，可校验、可恢复",
    "inference": "推理：**只走推理相**（BN 用 running、dropout 恒等），带可注入时钟",
}

#: 每个阶段**由谁实现**（"转发而不重写"纪律的可核对形式）.
STAGE_OWNERS: dict[str, str] = {
    "dataset": "本包新建（datasets.TabularDataset / make_dataset / split_dataset）",
    "sampler": "本包新建（sampler.Sampler / shard_indices）",
    "dataloader": "本包新建（dataloader.DataLoader / collate）",
    "device": "本包新建（device.plan_device / DevicePlan，纯算术）",
    "train": "本包新建（train.train_pipeline，梯度转发 day095 的 loss_and_grad）",
    "checkpoint": "本包新建（checkpoint.save_checkpoint / load_checkpoint）",
    "inference": "本包新建（inference.predict_one / predict_batch，前向转发 day095）",
}

if not (
    set(STAGES) == set(STAGE_DESCRIPTIONS) == set(STAGE_OWNERS)
):  # pragma: no cover - 导入期不变式
    raise TorchPipelineError(
        "七个阶段的三张表不一致：STAGES / STAGE_DESCRIPTIONS / STAGE_OWNERS 必须逐键对齐——"
        "少一个键的那个阶段在报告里只有名字、没有'它由谁实现'。"
    )

# --------------------------------------------------------------------------- #
# 闭合表 2：四条读数轴
# --------------------------------------------------------------------------- #

#: 四条读数轴（顺序 = 从"一批多少条"到"一步多少秒"）.
PIPELINE_AXES: tuple[str, ...] = ("batch", "worker", "device", "time")

#: 每条轴的一句话解释.
AXIS_DESCRIPTIONS: dict[str, str] = {
    "batch": "批：一个 epoch 有多少批、尾批丢了多少样本",
    "worker": "工作进程：多进程加载时数据怎么分片（互不相交、并集为全集）",
    "device": "设备：参数 / 激活 / 优化器状态在预算里占多少字节",
    "time": "时间：推理延迟与吞吐（每样本秒数、每秒样本数）",
}

#: 每条轴的一句话公式.
AXIS_FORMULAS: dict[str, str] = {
    "batch": "批数 = n // b 或 ceil(n / b)；丢弃 = n − b × 批数",
    "worker": "worker k 的分片 = { i : i % workers == k }",
    "device": "total = params + activations + optimizer_multiplier × params",
    "time": "seconds_per_sample = seconds / samples；throughput = samples / seconds",
}

if not (
    set(PIPELINE_AXES) == set(AXIS_DESCRIPTIONS) == set(AXIS_FORMULAS)
):  # pragma: no cover - 导入期不变式
    raise TorchPipelineError(
        "四条读数轴的三张表不一致：PIPELINE_AXES / AXIS_DESCRIPTIONS / AXIS_FORMULAS "
        "必须逐键对齐，否则某条轴在报告里只有名字、没有它的公式。"
    )

# --------------------------------------------------------------------------- #
# 闭合表 3：三个设备与四种精度
# --------------------------------------------------------------------------- #

DEVICE_CPU = "cpu"
DEVICE_CUDA = "cuda"
DEVICE_MPS = "mps"

#: 三个设备（顺序 = 从"一定在"到"不一定在"）.
DEVICE_KINDS: tuple[str, ...] = (DEVICE_CPU, DEVICE_CUDA, DEVICE_MPS)

#: 每个设备的一句话解释（**本课不真的探测硬件**——它只做算术）.
DEVICE_DESCRIPTIONS: dict[str, str] = {
    DEVICE_CPU: "CPU：内存预算，永远可用（本课所有读数都在这里可复算）",
    DEVICE_CUDA: "CUDA：GPU 显存，参数与激活都要从主机搬过去",
    DEVICE_MPS: "MPS：Apple 的统一内存后端，口径与 CUDA 相同、预算更低",
}

#: 每个设备的搬运方向前缀（``move_report`` 用它拼出一行可读文案）.
DEVICE_PREFIXES: dict[str, str] = {
    DEVICE_CPU: "CPU",
    DEVICE_CUDA: "CPU → CUDA:0",
    DEVICE_MPS: "CPU → MPS:0",
}

if not (
    set(DEVICE_KINDS) == set(DEVICE_DESCRIPTIONS) == set(DEVICE_PREFIXES)
):  # pragma: no cover - 导入期不变式
    raise TorchPipelineError(
        "三个设备的三张表不一致：DEVICE_KINDS / DEVICE_DESCRIPTIONS / DEVICE_PREFIXES "
        "必须逐键对齐。"
    )

DTYPE_FLOAT32 = "float32"

#: 四种精度 -> 每个元素的字节数（口径表，**不是运行时探测**）.
DTYPE_BYTES: dict[str, int] = {
    "float32": 4,
    "float64": 8,
    "float16": 2,
    "bfloat16": 2,
}

#: 每种精度的一句话解释.
DTYPE_DESCRIPTIONS: dict[str, str] = {
    "float32": "单精度（默认）：4 字节，可训练链的标准精度",
    "float64": "双精度：8 字节，只在数值对账时用（训练太贵）",
    "float16": "半精度：2 字节，推理常用；训练需要损失缩放",
    "bfloat16": "脑浮点：2 字节，动态范围与 float32 相同、尾数更短",
}

if set(DTYPE_BYTES) != set(DTYPE_DESCRIPTIONS):  # pragma: no cover - 导入期不变式
    raise TorchPipelineError(
        "四种精度的两张表不一致：DTYPE_BYTES / DTYPE_DESCRIPTIONS 必须逐键对齐。"
    )

# --------------------------------------------------------------------------- #
# 闭合表 4：七条性质
# --------------------------------------------------------------------------- #

PROPERTY_SHUFFLE_ORDER_IS_DETERMINISTIC = "shuffle_order_is_deterministic"
PROPERTY_WORKER_SHARDS_PARTITION_THE_DATASET = "worker_shards_partition_the_dataset"
PROPERTY_DROP_LAST_MATCHES_BATCH_FORMULA = "drop_last_matches_batch_formula"
PROPERTY_DEVICE_BYTES_MATCH_FORMULA = "device_bytes_match_formula"
PROPERTY_CHECKPOINT_ROUND_TRIP_IS_BITWISE = "checkpoint_round_trip_is_bitwise"
PROPERTY_RESUME_MATCHES_UNINTERRUPTED = "resume_matches_uninterrupted"
PROPERTY_PREDICT_BATCH_EQUALS_SEQUENTIAL = "predict_batch_equals_sequential"

#: 七条性质（顺序 = 数据 → 分片 → 批次 → 设备 → 检查点 → 恢复 → 推理）.
PIPELINE_PROPERTIES: tuple[str, ...] = (
    PROPERTY_SHUFFLE_ORDER_IS_DETERMINISTIC,
    PROPERTY_WORKER_SHARDS_PARTITION_THE_DATASET,
    PROPERTY_DROP_LAST_MATCHES_BATCH_FORMULA,
    PROPERTY_DEVICE_BYTES_MATCH_FORMULA,
    PROPERTY_CHECKPOINT_ROUND_TRIP_IS_BITWISE,
    PROPERTY_RESUME_MATCHES_UNINTERRUPTED,
    PROPERTY_PREDICT_BATCH_EQUALS_SEQUENTIAL,
)

#: 每条性质在讲什么（一句话），以及它的判据属于哪一类.
PROPERTY_DESCRIPTIONS: dict[str, str] = {
    PROPERTY_SHUFFLE_ORDER_IS_DETERMINISTIC: "相等：同 seed 两次 Sampler.order(epoch) 的差异项数 = 0",
    PROPERTY_WORKER_SHARDS_PARTITION_THE_DATASET: "下界：worker 分片的并集覆盖 = 1.0 且不相交项数 = 0",
    PROPERTY_DROP_LAST_MATCHES_BATCH_FORMULA: "相等：批数与 n // b 的差 = 0（drop_last）",
    PROPERTY_DEVICE_BYTES_MATCH_FORMULA: "上界：DevicePlan.total_bytes 与手算的相对差 <= 1e-9",
    PROPERTY_CHECKPOINT_ROUND_TRIP_IS_BITWISE: "相等：save → load 之后压平参数逐位相同（差 = 0）",
    PROPERTY_RESUME_MATCHES_UNINTERRUPTED: "相等：中途恢复训到第 N 轮与一口气训到第 N 轮，参数最大绝对差 = 0",
    PROPERTY_PREDICT_BATCH_EQUALS_SEQUENTIAL: "相等：predict_batch 与逐条 predict_one 的输出最大绝对差 = 0",
}

#: 每条性质"失败意味着什么"（**不通过时要去看哪里**）.
PROPERTY_FAILURE: dict[str, str] = {
    PROPERTY_SHUFFLE_ORDER_IS_DETERMINISTIC: "打乱用了全局随机源，而不是复用的 LCG（uniforms）",
    PROPERTY_WORKER_SHARDS_PARTITION_THE_DATASET: "分片写成了固定块（range(k*b, (k+1)*b)）⇒ 会漏样本或重叠",
    PROPERTY_DROP_LAST_MATCHES_BATCH_FORMULA: "循环上界写成 (n + b) // b（多算一批）或漏了 ceil 分支",
    PROPERTY_DEVICE_BYTES_MATCH_FORMULA: "优化器状态按 1× 参数算（漏了 optimizer_multiplier）",
    PROPERTY_CHECKPOINT_ROUND_TRIP_IS_BITWISE: "落盘时先 round 了权重（json 的默认 repr 本来就是精确往返）",
    PROPERTY_RESUME_MATCHES_UNINTERRUPTED: "恢复时没装回优化器跨步状态，或 shuffle 的 epoch 号接错了",
    PROPERTY_PREDICT_BATCH_EQUALS_SEQUENTIAL: "批量推理走了训练相（BN 用本批统计、dropout 没关）",
}

if not (
    set(PIPELINE_PROPERTIES) == set(PROPERTY_DESCRIPTIONS) == set(PROPERTY_FAILURE)
):  # pragma: no cover - 导入期不变式
    raise TorchPipelineError(
        "七条性质的三张表不一致：名单 / 说明 / 失败意味着什么必须逐键对齐。"
    )

# --------------------------------------------------------------------------- #
# 闭合表 5：十条笔记 / 五条边界
# --------------------------------------------------------------------------- #

#: 十条笔记的顺序（键集合必须与 :data:`PIPELINE_NOTES` 一致）.
NOTES_ORDER: tuple[str, ...] = (
    "immutable_samples",
    "fingerprint",
    "shuffle_by_epoch",
    "drop_last_tail",
    "worker_shard",
    "bytes_arithmetic",
    "budget_is_not_error",
    "five_files",
    "hash_covers_weights",
    "eval_phase_only",
)

#: 十条笔记（键 -> 一句话）.
PIPELINE_NOTES: dict[str, str] = {
    "immutable_samples": "样本一旦进数据集就**不再改**：改一条样本等于换了一份数据，指纹必须随之变。",
    "fingerprint": "内容指纹（sha256 前 16 位）让'这两个分数能比吗'有一个可以查证的答案。",
    "shuffle_by_epoch": "打乱必须**只**依赖 (seed, epoch)：否则'第 3 轮的批'在两台机器上是两回事。",
    "drop_last_tail": "drop_last 丢的只有**尾批**：批数公式与丢弃数是同一个式子的两面。",
    "worker_shard": "多进程先分片再切批：分片必须互不相交、并集为全集，否则有的样本被喂两遍。",
    "bytes_arithmetic": "显存是**算**出来的：参数 + 激活 + 优化器状态（Adam 约 3× 参数），不是'感觉够用'。",
    "budget_is_not_error": "放不进预算**不抛异常**：它由 fits 暴露，因为'装不下'是可以换批大小解决的。",
    "five_files": "检查点是**五件套**：缺一个就抛 CheckpointError 并点名是哪一个。",
    "hash_covers_weights": "清单里的哈希只覆盖权重与优化器状态——指标与历史是可读记录，不参与校验。",
    "eval_phase_only": "推理只走推理相：BN 用 running 统计、dropout 恒等；用训练相会让结果随批抖动。",
}

if set(NOTES_ORDER) != set(PIPELINE_NOTES):  # pragma: no cover - 导入期不变式
    raise TorchPipelineError("十条笔记的顺序表与键集合不一致。")

#: 五条边界（**这一课明确不承诺的事**）.
PIPELINE_BOUNDARIES: tuple[str, ...] = (
    "不真的探测硬件：设备与预算是**算术表**，不 import torch、不做 cuda 查询。",
    "不做真正的多进程：worker 分片是**索引层的划分**，为的是可复算的读数。",
    "不做异步 IO / 预取：DataLoader 是纯同步的批切分器，延迟读数由假 clock 钉住。",
    "不新增第三方依赖：全部纯标准库实现，不 import torch / numpy。",
    "不改动 day095：网络、损失与梯度全部**调用** regularization 的那一份。",
)

# --------------------------------------------------------------------------- #
# 闭合表 6：十张公式与四个容差
# --------------------------------------------------------------------------- #

#: 批数公式（drop_last 的两个分支）。
BATCH_COUNT_FORMULA = "drop_last=True：批数 = n // b；drop_last=False：批数 = ceil(n / b) = (n + b − 1) // b"

#: 丢弃样本公式。
DROPPED_SAMPLES_FORMULA = "dropped = n − b × 批数（drop_last=True 时它是被整批丢掉的尾批）"

#: 工作进程分片公式。
SHARD_FORMULA = "worker k 的分片 = { i : i % workers == k }（互不相交、并集为全集）"

#: 张量字节公式。
TENSOR_BYTES_FORMULA = "bytes = (Π shape_i) × DTYPE_BYTES[dtype]"

#: 激活字节公式（一层 N×width 的中间结果）。
ACTIVATION_BYTES_FORMULA = "bytes = batch_size × width × DTYPE_BYTES[dtype]"

#: 总字节公式（优化器状态按参数量的倍数算）。
TOTAL_BYTES_FORMULA = "total = params + activations + optimizer_multiplier × params"

#: 是否装得下与余量的口径。
DEVICE_FITS_FORMULA = "fits ⇔ total <= budget_bytes；headroom = budget_bytes − total"

#: 五件套的口径。
CHECKPOINT_FILES_FORMULA = "五件套 = meta.json + params.json + optimizer.json + metrics.json + history.jsonl"

#: 恢复训练的口径。
RESUME_FORMULA = "恢复 = 载入 params + 装回优化器跨步状态 + 从 epoch k+1 继续（order(k+1) 逐位相同）"

#: 延迟读数的口径。
LATENCY_FORMULA = "seconds_per_sample = seconds / samples；throughput = samples / seconds"

#: "逐位相等"的容差（写成 0.0：这一类判据不接受任何浮点自由）.
EXACT_TOLERANCE = 0.0

#: 参数逐位往返的容差（JSON 的 repr 精确往返，用 1e-12 只为承认"两次求值"这一事实）.
PARAMS_TOLERANCE = 1e-12

#: 设备字节算术的容差（整数运算，用 1e-9 的**相对差**作为上界判据）.
DEVICE_BYTES_TOLERANCE = 1e-9

#: 分片覆盖的下界（1.0 = 一条样本都不许漏）.
SHARD_COVERAGE_LOWER_BOUND = 1.0

#: 这一课的性质名单（供报告核对：表里的行数必须等于它）.
PROPERTIES = PIPELINE_PROPERTIES

__all__ = [
    "ACTIVATION_BYTES_FORMULA",
    "AXIS_DESCRIPTIONS",
    "AXIS_FORMULAS",
    "BATCH_COUNT_FORMULA",
    "CHECKPOINT_FILES_FORMULA",
    "DEVICE_CPU",
    "DEVICE_CUDA",
    "DEVICE_DESCRIPTIONS",
    "DEVICE_FITS_FORMULA",
    "DEVICE_KINDS",
    "DEVICE_MPS",
    "DEVICE_PREFIXES",
    "DROPPED_SAMPLES_FORMULA",
    "DTYPE_BYTES",
    "DTYPE_DESCRIPTIONS",
    "DTYPE_FLOAT32",
    "DEVICE_BYTES_TOLERANCE",
    "EXACT_TOLERANCE",
    "LATENCY_FORMULA",
    "NOTES_ORDER",
    "PARAMS_TOLERANCE",
    "PIPELINE_AXES",
    "PIPELINE_BOUNDARIES",
    "PIPELINE_NOTES",
    "PIPELINE_PROPERTIES",
    "PROPERTIES",
    "PROPERTY_CHECKPOINT_ROUND_TRIP_IS_BITWISE",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_DEVICE_BYTES_MATCH_FORMULA",
    "PROPERTY_DROP_LAST_MATCHES_BATCH_FORMULA",
    "PROPERTY_FAILURE",
    "PROPERTY_PREDICT_BATCH_EQUALS_SEQUENTIAL",
    "PROPERTY_RESUME_MATCHES_UNINTERRUPTED",
    "PROPERTY_SHUFFLE_ORDER_IS_DETERMINISTIC",
    "PROPERTY_WORKER_SHARDS_PARTITION_THE_DATASET",
    "RESUME_FORMULA",
    "SHARD_COVERAGE_LOWER_BOUND",
    "SHARD_FORMULA",
    "STAGES",
    "STAGE_DESCRIPTIONS",
    "STAGE_OWNERS",
    "TENSOR_BYTES_FORMULA",
    "TOTAL_BYTES_FORMULA",
]
