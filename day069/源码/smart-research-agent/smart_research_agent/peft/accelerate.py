"""多卡与混合精度算术：Accelerate / DeepSpeed 的那几个数字（M5-D4）.

day051 回答了"一个设备要多少显存"。这一天回答它的下一个问题：**`D` 个设备呢？**
答案不是"除以 D"——只有**分片（sharding）**才让显存随设备数下降，而"数据并行
（DDP）"在每个设备上放一份完整副本：

.. code-block:: text

    策略        每设备的显存
    ddp         完整模型 + 完整梯度 + 完整优化器状态（不变）
    zero2       完整模型 + 完整梯度 + 优化器状态 / D      ← 只切优化器状态
    zero3       模型 / D + 梯度 / D + 优化器状态 / D      ← 全切

实测（`llama-2-7b` / 全参微调 / bf16 权重与梯度 + AdamW fp32 状态 / 4 个设备）:

.. code-block:: text

    单项（每参数）        全模型（7B）
    权重 2 B              12.55 GiB
    梯度 2 B              12.55 GiB
    优化器状态 8 B        50.20 GiB
    ------------------------------------
    合计 12 B/参数        75.30 GiB

    策略     每设备显存      算法
    ddp      75.30 GiB    一份完整副本
    zero2    28.24 GiB    12.55 + 12.55/4 + 50.20/4
    zero3    18.83 GiB    12.55/4 + 12.55/4 + 50.20/4

于是出现了"同一份配置，DDP 要 80 GB 卡、ZeRO-3 只要 24 GB 卡"这种跨度——
**"能不能多卡跑"的答案取决于策略，而不是设备数**。

LoRA 上的结论恰好相反，而且反直觉（本课实测）：

.. code-block:: text

    策略     每设备显存（LoRA / 4 卡）   省多少
    ddp      12.65 GiB                  1.00×
    zero2    12.59 GiB                  1.00×   ← 几乎没省
    zero3     3.16 GiB                  4.00×

**zero2 对 LoRA 无效**：它切的是优化器状态与梯度，而 LoRA 的优化器状态只有
适配器那一百来 MB——占大头的是**冻结的基座权重**，它只有 zero3 才会切。
代价是 zero3 每步要多花三个数量级的通信量（0.025 GB → 26.98 GB），
**因为每层前向都要把分片的参数 all-gather 回完整的一份**。
这张对照表是"分片策略的收益取决于'哪一项占用最大'"的最好例证。

本模块还负责另外两件容易写错的事：

1. **有效批大小的乘数**：``per_device × accum × devices``。单卡调到 8 的有效批，
   在 4 卡上默认变成 32——**学习率、warmup、总步数都会跟着变**；
2. **学习率的线性缩放**：批大小翻倍时梯度噪声变小，通常配 `lr` 同倍放大
   （并重新 warmup）。这个规则有四条边界，本模块把它们写成显式告警。

依赖说明：``accelerate`` / ``deepspeed`` **不装进本课的日常环境**。本模块只用
标准库做算术与配置生成，生成物是 Accelerate 实际接受的配置格式（`accelerate
config` 产出的那份 YAML）——测试会把它解析回来逐键核对。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from smart_research_agent.peft.config import LoRAConfig, PEFTConfigError, QLoRAConfig
from smart_research_agent.peft.memory import (
    COMPUTE_DTYPE_BYTES,
    MemoryBreakdown,
    plan_memory,
)
from smart_research_agent.peft.targets import DecoderSpec, plan_lora
from smart_research_agent.sft.args import SFTTrainingArgs

#: 分片策略（顺序即"每设备显存"从大到小）
SHARDING_STRATEGIES: tuple[str, ...] = ("ddp", "zero2", "zero3")

#: 混合精度配置文件的取值（``accelerate config`` 的 ``mixed_precision`` 字段）
MIXED_PRECISION_VALUES: tuple[str, ...] = ("no", "fp16", "bf16")


@dataclass(frozen=True)
class MixedPrecisionProfile:
    """一种混合精度配置的画像.

    ``fp16`` 与 ``bf16`` 的字节数完全一样（都是 2 字节），差别在**指数位**：
    bf16 有 8 位指数（与 fp32 相同，不会溢出）与 7 位尾数（精度低）；
    fp16 有 5 位指数（动态范围窄，容易溢出为 inf）与 10 位尾数（精度高）。
    直接后果是：**fp16 必须配梯度缩放（loss scaling），bf16 不需要**——
    这也是"Ampere 及之后优先用 bf16"的全部理由。
    """

    name: str
    torch_dtype: str
    bytes_per_parameter: int
    needs_loss_scaling: bool
    notes: list[str] = field(default_factory=list)


#: 混合精度画像表.
#:
#: ``fp32`` 保留在这里不是为了"用它训练"（它比 bf16 贵一倍），而是为了让
#: "权重的存储精度"与"计算精度"分开讨论时有第三个参照点——QLoRA 的基座
#: 存储是 4-bit、计算是 bf16 就是这种分离的极端形式。
MIXED_PRECISION_PROFILES: dict[str, MixedPrecisionProfile] = {
    "bf16": MixedPrecisionProfile(
        name="bf16",
        torch_dtype="torch.bfloat16",
        bytes_per_parameter=2,
        needs_loss_scaling=False,
        notes=["8 位指数与 fp32 相同：动态范围大，几乎不会溢出", "Ampere 及之后的缺省选择"],
    ),
    "fp16": MixedPrecisionProfile(
        name="fp16",
        torch_dtype="torch.float16",
        bytes_per_parameter=2,
        needs_loss_scaling=True,
        notes=["5 位指数：动态范围窄，梯度下溢需要 loss scaling", "10 位尾数：精度略高于 bf16"],
    ),
    "fp32": MixedPrecisionProfile(
        name="fp32",
        torch_dtype="torch.float32",
        bytes_per_parameter=4,
        needs_loss_scaling=False,
        notes=["没有混合精度收益，只作为对照口径"],
    ),
}


def mixed_precision_profile(name: str) -> MixedPrecisionProfile:
    """按名字取混合精度画像（未知名字抛 ``PEFTConfigError``）."""
    profile = MIXED_PRECISION_PROFILES.get(name)
    if profile is None:
        raise PEFTConfigError(
            f"未知的混合精度 {name!r}，可选：{', '.join(sorted(MIXED_PRECISION_PROFILES))}"
        )
    return profile


def scale_learning_rate(
    base_learning_rate: float,
    *,
    base_global_batch_size: int,
    new_global_batch_size: int,
    mode: str = "linear",
) -> float:
    """按批大小的变化缩放学习率（线性法则 / 平方根法则）.

    .. code-block::

        linear（缺省）：lr_new = lr_base × (new_batch / base_batch)
        sqrt        ：lr_new = lr_base × sqrt(new_batch / base_batch)
        none        ：lr_new = lr_base（不缩放）

    为什么要缩放：批大小变大意味着梯度是更多样本的平均，**梯度噪声变小、
    步长可以更大**。线性法则来自"批大小翻倍时梯度方差减半"的直觉，工程上
    在中小规模最常用；平方根法则更保守，在批大小很大（上千）时更常见。

    ``none`` 不是"错"的选项，但它会命中 ``plan_distributed`` 的一条告警
    （梯度被多卡平均后变小而步长不变，训练会明显变慢）——**把"不缩放"
    显式化，比让它悄悄发生好**。

    三处边界（都在测试里被钉住）：

    1. 只支持这三条法则——`mode="none"` 的比例因子恒为 1；
    2. 批大小必须为正——0 或负数会让比例无定义；
    3. 结果**不是**建议值而是公式值：真实学习率仍要靠评估集标定（day051 第
       4.3 节实测过，参考模型上最优 `lr` 与 7B 的经验区间差四个数量级）。
    """
    if base_global_batch_size <= 0 or new_global_batch_size <= 0:
        raise PEFTConfigError(
            f"批大小必须为正整数，收到 {base_global_batch_size} / {new_global_batch_size}"
        )
    if base_learning_rate <= 0:
        raise PEFTConfigError(f"学习率必须为正数，收到 {base_learning_rate}")
    ratio = new_global_batch_size / base_global_batch_size
    if mode == "linear":
        factor = ratio
    elif mode == "sqrt":
        factor = ratio**0.5
    elif mode == "none":
        factor = 1.0
    else:
        raise PEFTConfigError(
            f"未知的缩放法则 {mode!r}，可选：linear（线性）/ sqrt（平方根）/ none（不缩放）"
        )
    return base_learning_rate * factor


def per_device_memory(
    spec: DecoderSpec,
    *,
    strategy: str,
    devices: int,
    lora_config: LoRAConfig | None = None,
    qlora_config: QLoRAConfig | None = None,
    optimizer: str = "adamw_torch",
    compute_dtype: str = "bfloat16",
    full_finetune: bool = False,
) -> MemoryBreakdown:
    """按分片策略算出**每设备**的显存.

    ``ddp`` 直接复用 day051 的 ``plan_memory``——数据并行在每个设备上放一份
    完整副本，因此单设备预算与单卡完全相同。``zero2`` / ``zero3`` 在这个
    基准上把可切分的那几项除以设备数：

    - ZeRO-2：切**优化器状态**（与梯度）。实测里它是"最划算的一步"——
      优化器状态常常是最大的一项（8 字节/参数），切掉它就能让 7B 全参
      从 75.31 GiB 降到 33.65 GiB；
    - ZeRO-3：再切**参数与梯度**。代价是每层前向/反向都要 all-gather 一次
      参数，通信量随设备数上升、计算利用率下降（"能跑"不等于"跑得快"）。
    """
    if strategy not in SHARDING_STRATEGIES:
        raise PEFTConfigError(
            f"未知的分片策略 {strategy!r}，可选：{', '.join(SHARDING_STRATEGIES)}"
        )
    if devices <= 0:
        raise PEFTConfigError(f"设备数必须为正整数，收到 {devices}")

    if full_finetune:
        baseline = plan_memory(
            spec, strategy="full", optimizer=optimizer, compute_dtype=compute_dtype
        )
    else:
        if lora_config is None:
            raise PEFTConfigError("LoRA / QLoRA 策略必须提供 lora_config")
        baseline = plan_memory(
            spec,
            strategy="qlora" if qlora_config is not None else "lora",
            lora_config=lora_config,
            qlora_config=qlora_config,
            optimizer=optimizer,
            compute_dtype=compute_dtype,
        )

    if strategy == "ddp" or devices == 1:
        return replace(
            baseline, notes=[*baseline.notes, f"{strategy}：每个设备一份完整副本"]
        )

    # 分片规则（DeepSpeed ZeRO 的定义）：
    #   zero2：切优化器状态 + 梯度，**参数不切**（每个设备仍持有完整权重）
    #   zero3：参数也切（每层前向/反向各 all-gather 一次）
    # 这条规则决定了"哪种策略对哪类模型有效"：
    #   全参微调里最大的一项是优化器状态（8 字节/参数）→ zero2 就砍掉一大块；
    #   LoRA 里最大的一项是**冻结的基座权重**（优化器状态只有适配器那一百来 MB）
    #   → **zero2 几乎不省，只有 zero3 有用**。
    shard_weight = strategy == "zero3"
    shard_gradient = True
    shard_optimizer = True

    def _shard(value: int, *, enable: bool) -> int:
        return value // devices if enable else value

    base_weight = _shard(baseline.base_weight_bytes, enable=shard_weight)
    adapter_weight = _shard(baseline.adapter_weight_bytes, enable=shard_weight)
    notes = [
        *baseline.notes,
        f"{strategy}：优化器状态与梯度按 {devices} 个设备分片"
        + ("；参数也分片（每层前向需要 all-gather）" if shard_weight else "；参数不分片"),
    ]
    return MemoryBreakdown(
        model=baseline.model,
        strategy=f"{baseline.strategy}+{strategy}",
        optimizer=baseline.optimizer,
        compute_dtype=baseline.compute_dtype,
        base_parameters=baseline.base_parameters,
        trainable_parameters=baseline.trainable_parameters,
        base_weight_bytes=base_weight,
        base_gradient_bytes=_shard(baseline.base_gradient_bytes, enable=shard_gradient),
        base_optimizer_bytes=_shard(baseline.base_optimizer_bytes, enable=shard_optimizer),
        adapter_weight_bytes=adapter_weight,
        adapter_gradient_bytes=_shard(baseline.adapter_gradient_bytes, enable=shard_gradient),
        adapter_optimizer_bytes=_shard(baseline.adapter_optimizer_bytes, enable=shard_optimizer),
        base_bits_per_parameter=baseline.base_bits_per_parameter,
        notes=notes,
    )


def communication_bytes_per_step(
    *,
    gradient_parameters: int,
    parameter_parameters: int | None = None,
    devices: int,
    bytes_per_parameter: int = 2,
    stage: str = "zero2",
) -> int:
    """每步的跨设备通信量（字节）——"能跑"之外的第二个成本.

    两种策略的通信模式完全不同：

    - **DDP / ZeRO-2**：每步一次梯度的 all-reduce。通信量与**需要梯度的参数量**
      （``gradient_parameters``）成正比，与设备数关系不大（ring all-reduce 的
      每卡流量约 ``2(N-1)/N × 参数字节``）；
    - **ZeRO-3**：除梯度通信外，被分片的**参数**还要在前向与反向各
      all-gather 一次，因此第二项按参数规模（``parameter_parameters``，缺省
      取梯度参数）算两倍。

    LoRA 上这两项的大小关系很值得看：适配器只有几百万个参数（梯度项很小），
    而 ZeRO-3 要 gather 的是**整个基座**（参数项很大）——**所以 LoRA + ZeRO-3
    的通信量几乎全花在"把一个本来就不大的模型搬来搬去"上**。这正是"LoRA 场景下
    ZeRO-3 收益有限"的量化解释。
    """
    if gradient_parameters <= 0 or devices <= 0 or bytes_per_parameter <= 0:
        raise PEFTConfigError("gradient_parameters / devices / bytes_per_parameter 必须为正整数")
    if stage not in ("zero2", "zero3"):
        raise PEFTConfigError(f"通信量只对 zero2 / zero3 建模，收到 {stage!r}")
    if devices == 1:
        # 单卡没有跨设备通信：gradient 项本来就为 0，参数 all-gather 也只发生在
        # 多卡分片之后。**不能给 zero3 在单卡下算出一个非零值**——那会让
        # "通信量随策略与设备数变化"的对照表在 devices=1 这一列自相矛盾。
        return 0
    gathered = (
        parameter_parameters if parameter_parameters is not None else gradient_parameters
    )
    if gathered <= 0:
        raise PEFTConfigError(f"parameter_parameters 必须为正整数，收到 {gathered}")
    gradient = 2 * (devices - 1) / devices * gradient_parameters * bytes_per_parameter
    if stage == "zero2":
        return int(gradient)
    return int(gradient + 2 * gathered * bytes_per_parameter)


@dataclass
class DistributedPlan:
    """一次多卡 LoRA 训练的全部派生量."""

    model: str
    devices: int
    strategy: str
    mixed_precision: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    global_batch_size: int
    global_micro_batch_size: int
    train_size: int
    trainable_parameters: int
    micro_batches_per_epoch: int
    steps_per_epoch: int
    total_steps: int
    warmup_steps: int
    base_learning_rate: float
    scaled_learning_rate: float
    scaling_mode: str
    per_device_gib: float
    single_device_gib: float
    memory_saving_factor: float
    communication_bytes_per_step: int
    communication_bytes_total: int
    checkpoint_bytes: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典."""
        return asdict(self)

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"{self.devices}× {self.strategy} | 全局批 {self.global_batch_size} | "
            f"每设备 {self.per_device_gib:.2f} GiB（单卡 {self.single_device_gib:.2f}，"
            f"省 {self.memory_saving_factor:.2f}×）| "
            f"lr {self.base_learning_rate:g} → {self.scaled_learning_rate:g} | "
            f"总步数 {self.total_steps}"
        )


def plan_distributed(
    spec: DecoderSpec,
    args: SFTTrainingArgs,
    *,
    devices: int,
    strategy: str = "ddp",
    train_size: int,
    eval_size: int = 0,
    lora_config: LoRAConfig | None = None,
    qlora_config: QLoRAConfig | None = None,
    full_finetune: bool = False,
    mixed_precision: str = "bf16",
    lr_scaling_mode: str = "linear",
    optimizer: str = "adamw_torch",
) -> DistributedPlan:
    """算出多卡训练的步数、学习率、每设备显存与通信量，并生成告警.

    五类告警每一条都对应一个真实的坑：

    1. **全局批被放大但没有放大学习率**——所有梯度都小了 `D` 倍，训练变慢
       而看不出错；
    2. **全局批被放大但没重新 warmup**——warmup 步数是按旧批大小定的；
    3. **全局批超过训练集**——一个 epoch 连一步都走不完（`steps_per_epoch = 0`）；
    4. **ZeRO-3 但每设备显存没降**（例如 LoRA 场景下基座本来就小）——
       额外的通信开销换不到什么；
    5. **fp16 但没有 loss scaling**——梯度下溢会让部分参数停止更新。
    """
    if strategy not in SHARDING_STRATEGIES:
        raise PEFTConfigError(
            f"未知的分片策略 {strategy!r}，可选：{', '.join(SHARDING_STRATEGIES)}"
        )
    if devices <= 0:
        raise PEFTConfigError(f"设备数必须为正整数，收到 {devices}")
    args.validate()
    profile = mixed_precision_profile(mixed_precision)
    compute_dtype = mixed_precision if mixed_precision in COMPUTE_DTYPE_BYTES else "bfloat16"

    global_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * devices
    global_micro = args.per_device_train_batch_size * devices
    micro_batches = train_size // global_micro
    steps_per_epoch = micro_batches // args.gradient_accumulation_steps
    total_steps = max(1, int(steps_per_epoch * args.num_train_epochs))
    warmup_steps = int(total_steps * args.warmup_ratio)

    base_lr = args.learning_rate
    scaled_lr = scale_learning_rate(
        base_lr,
        base_global_batch_size=args.effective_batch_size,
        new_global_batch_size=global_batch,
        mode=lr_scaling_mode,
    )

    single = plan_memory(
        spec,
        strategy="full" if full_finetune else ("qlora" if qlora_config else "lora"),
        lora_config=lora_config,
        qlora_config=qlora_config,
        optimizer=optimizer,
        compute_dtype=compute_dtype,
    )
    per_device = per_device_memory(
        spec,
        strategy=strategy,
        devices=devices,
        lora_config=lora_config,
        qlora_config=qlora_config,
        optimizer=optimizer,
        compute_dtype=compute_dtype,
        full_finetune=full_finetune,
    )

    parameter_count = spec.total_parameters()
    gradient_parameters = (
        parameter_count if full_finetune else (plan_lora(spec, lora_config).trainable_parameters
                                               if lora_config else 0)
    )
    comm_per_step = communication_bytes_per_step(
        gradient_parameters=gradient_parameters,
        # ZeRO-3 要 all-gather 的是被分片的**参数**：全参即全部参数，LoRA 则是
        # 整个基座（适配器太小、切它没有意义）
        parameter_parameters=parameter_count,
        devices=devices,
        bytes_per_parameter=profile.bytes_per_parameter,
        stage="zero3" if strategy == "zero3" else "zero2",
    )

    warnings: list[str] = []
    if devices > 1 and lr_scaling_mode == "none":
        warnings.append(
            f"全局批从 {args.effective_batch_size} 放大到 {global_batch}（{devices} 个设备），"
            "但 lr_scaling_mode=none：梯度噪声变小而步长不变，训练会明显变慢。"
            "批大小翻倍时通常把学习率同倍放大（线性法则），并重新标定。"
        )
    if devices > 1 and warmup_steps == 0:
        warnings.append(
            f"放大批大小后 warmup_steps 仍是 0（total_steps={total_steps}，"
            f"warmup_ratio={args.warmup_ratio}）：大 batch 的前几步最容易发散，"
            "建议把 warmup 至少设到 total_steps 的 5%。"
        )
    if steps_per_epoch == 0:
        warnings.append(
            f"全局批 {global_batch} 超过训练集 {train_size} 条："
            f"每个 epoch 走不出一步，训练不会发生。请调小批大小或补充数据。"
        )
    saving = single.as_gib / per_device.as_gib if per_device.as_gib else 1.0
    warning_comm_ratio = 0.0
    if strategy == "zero3" and devices > 1:
        zero2_comm = communication_bytes_per_step(
            gradient_parameters=gradient_parameters,
            devices=devices,
            bytes_per_parameter=profile.bytes_per_parameter,
            stage="zero2",
        )
        if zero2_comm > 0:
            warning_comm_ratio = comm_per_step / zero2_comm
    if warning_comm_ratio > 10:
        warnings.append(
            f"zero3 的每步通信量是 zero2 的 {warning_comm_ratio:.0f} 倍"
            f"（{comm_per_step / 1e9:.2f} GB vs {zero2_comm / 1e9:.3f} GB），"
            f"而每设备显存只从 {single.as_gib:.2f} 降到 {per_device.as_gib:.2f} GiB。"
            "**zero3 的显存收益来自切分参数，而它的通信代价正是每层前向都要"
            "all-gather 一遍参数**——这笔交换是否值得要用实测吞吐来判断，"
            "不能只看显存表。"
        )
    if mixed_precision == "fp16":
        warnings.append(
            "fp16 的动态范围窄（5 位指数），在 batch 很大或学习率偏高时容易溢出为 inf，"
            "且需要 loss scaling（accelerate 会自动启用）；Ampere 及之后的硬件优先用 bf16。"
        )

    return DistributedPlan(
        model=spec.name,
        devices=devices,
        strategy=strategy,
        mixed_precision=mixed_precision,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        global_batch_size=global_batch,
        global_micro_batch_size=global_micro,
        train_size=train_size,
        trainable_parameters=gradient_parameters,
        micro_batches_per_epoch=micro_batches,
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        base_learning_rate=base_lr,
        scaled_learning_rate=scaled_lr,
        scaling_mode=lr_scaling_mode,
        per_device_gib=per_device.as_gib,
        single_device_gib=single.as_gib,
        memory_saving_factor=saving,
        communication_bytes_per_step=comm_per_step,
        communication_bytes_total=comm_per_step * total_steps,
        # 检查点体积 = 需要落盘的参数量 × 存储精度字节数。LoRA 只需存适配器，
        # 因此这一项比全参小三个数量级——"能多卡跑"之外，落盘也是 LoRA 的收益。
        checkpoint_bytes=gradient_parameters * profile.bytes_per_parameter,
        warnings=warnings,
    )


def accelerate_config(
    *,
    devices: int,
    strategy: str = "ddp",
    mixed_precision: str = "bf16",
    deepspeed_config_file: str | None = None,
    machine_rank: int = 0,
) -> dict[str, Any]:
    """生成一份 ``accelerate config`` 产出的配置（字段名与官方一致）.

    ``accelerate config`` 交互式地问一堆问题，最后写出这样一份 YAML：

    .. code-block:: yaml

        compute_environment: LOCAL_MACHINE
        distributed_type: MULTI_GPU
        num_processes: 4
        machine_rank: 0
        main_training_function: main
        mixed_precision: bf16
        use_cpu: false

    ``ddp`` 对应 ``MULTI_GPU``；ZeRO-2 / ZeRO-3 对应 ``DEEPSPEED`` 并额外
    需要一个 DeepSpeed 配置文件。**本模块把它做成数据**，理由与 LoRA 配置
    一样：交互式问答产出的配置没法被测试、也没法被 review。
    """
    profile = mixed_precision_profile(mixed_precision)
    if strategy not in SHARDING_STRATEGIES:
        raise PEFTConfigError(
            f"未知的分片策略 {strategy!r}，可选：{', '.join(SHARDING_STRATEGIES)}"
        )
    if devices <= 0:
        raise PEFTConfigError(f"设备数必须为正整数，收到 {devices}")
    if machine_rank < 0:
        raise PEFTConfigError(f"machine_rank 不能为负数，收到 {machine_rank}")
    distributed_type = "MULTI_GPU" if strategy == "ddp" else "DEEPSPEED"
    if strategy != "ddp" and not deepspeed_config_file:
        raise PEFTConfigError(
            f"{strategy} 需要 deepspeed_config_file：DeepSpeed 的 ZeRO 阶段由那份 JSON 决定"
        )
    if strategy == "ddp" and deepspeed_config_file:
        raise PEFTConfigError(
            "ddp 不该带 deepspeed_config_file：accelerate 会因此切到 DEEPSPEED 路径"
        )
    payload: dict[str, Any] = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": distributed_type,
        "num_processes": devices,
        "machine_rank": machine_rank,
        "main_training_function": "main",
        "mixed_precision": profile.name if profile.name != "fp32" else "no",
        "use_cpu": False,
        "gpu_ids": "all",
    }
    if strategy != "ddp":
        payload["deepspeed_config"] = {
            "deepspeed_config_file": deepspeed_config_file,
            "zero3_init_flag": strategy == "zero3",
        }
    return payload


def deepspeed_zero_config(
    *,
    stage: int,
    mixed_precision: str = "bf16",
    offload_optimizer: bool = False,
    offload_parameters: bool = False,
) -> dict[str, Any]:
    """生成一份 DeepSpeed ZeRO 配置（``stage`` 取 2 / 3）.

    四个字段值得解释：

    - ``train_batch_size`` / ``gradient_accumulation_steps`` 写 ``"auto"``：
      这两项由 ``accelerate`` 在运行时从 ``TrainingArguments`` 推导，
      **手写死值会让"配置里的批大小"与"训练日志里的批大小"对不上**；
    - ``offload_optimizer`` / ``offload_param`` 把状态放到 CPU 内存上，
      用带宽换显存（PCIe 比 HBM 慢一个数量级，实测常常慢 2~4 倍）；
    - ``stage3_gather_16bit_weights_on_model_save`` 只在 ZeRO-3 下有意义：
      权重被分片了，保存时必须先聚合——**不开这一项，落盘的模型是"半份"的**。

    本课的实现只做参数校验与结构生成，不做任何 GPU 侧假设——它是一份
    可以被 review 的配置，而不是一段需要跑起来才知道对不对的代码。
    """
    if stage not in (2, 3):
        raise PEFTConfigError(f"ZeRO 只支持 stage 2 / 3，收到 {stage}")
    profile = mixed_precision_profile(mixed_precision)
    zero_optimization: dict[str, Any] = {"stage": stage}
    if stage == 2:
        zero_optimization["offload_optimizer"] = {
            "device": "cpu" if offload_optimizer else "none"
        }
    else:
        zero_optimization["offload_optimizer"] = {
            "device": "cpu" if offload_optimizer else "none"
        }
        zero_optimization["offload_param"] = {
            "device": "cpu" if offload_parameters else "none"
        }
        zero_optimization["stage3_gather_16bit_weights_on_model_save"] = True
    payload: dict[str, Any] = {
        "zero_optimization": zero_optimization,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
    }
    if profile.name != "fp32":
        payload[profile.name] = {"enabled": True}
    else:
        payload["fp32"] = {"enabled": True}
    return payload


def launch_command(
    script: str,
    *,
    devices: int,
    strategy: str = "ddp",
    config_file: str = "accelerate_config.yaml",
) -> str:
    """生成 ``accelerate launch`` 命令行（把"怎么启动"也变成可复制的产物）.

    三种写法对应三种策略，差别只在多出来的参数：

    .. code-block:: text

        ddp     accelerate launch --config_file accelerate_config.yaml train.py
        zero2   accelerate launch --config_file accelerate_config.yaml --num_processes 4 train.py
        zero3   accelerate launch --config_file accelerate_config.yaml --num_processes 4 train.py

    ``--num_processes`` 显式给出，是为了让命令**自解释**（配置文件里也有，
    但一条命令里能看到几个设备更好）。命令里刻意的顺序是"配置文件在前、
    脚本在后"——与 `accelerate` 的官方用法一致。
    """
    if not script.strip():
        raise PEFTConfigError("script 不能为空")
    if devices <= 0:
        raise PEFTConfigError(f"设备数必须为正整数，收到 {devices}")
    if strategy not in SHARDING_STRATEGIES:
        raise PEFTConfigError(
            f"未知的分片策略 {strategy!r}，可选：{', '.join(SHARDING_STRATEGIES)}"
        )
    if not config_file.strip():
        raise PEFTConfigError("config_file 不能为空")
    parts = ["accelerate", "launch", "--config_file", config_file]
    if devices > 1:
        parts += ["--num_processes", str(devices)]
    parts.append(script)
    return " ".join(parts)


def render_config_yaml(payload: dict[str, Any]) -> str:
    """把配置字典渲染成 YAML 文本（只支持本模块用到的两种类型）.

    手写而不是引入 PyYAML：配置里只有标量、字符串与一层嵌套字典，而
    **多一个依赖就要多一次版本核对**。渲染出的文本会被测试 `yaml.safe_load`
    解析回来逐键比对（环境里恰好有 PyYAML——它随项目依赖间接装上了）。
    """
    lines: list[str] = []
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for inner_key, inner_value in value.items():
                lines.append(f"  {inner_key}: {_yaml_scalar(inner_value)}")
        else:
            lines.append(f"{key}: {_yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


def _yaml_scalar(value: Any) -> str:
    """把标量渲染成 YAML 字面量（布尔小写、字符串按需加引号）.

    除了 YAML 语法字符之外，还有一类必须加引号的值：**YAML 1.1 的布尔别名**
    （``no`` / ``yes`` / ``on`` / ``off`` / ``true`` / ``false`` / ``null`` / ``~``）。
    ``mixed_precision: no`` 在语义上是一个字符串，但 `yaml.safe_load` 会把它读成
    布尔 ``False``——**渲染出去的配置与读回来的配置不一致，而两边都"没报错"**。
    本课实现时踩过这一次，所以这里显式枚举它们。
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    reserved = {"no", "yes", "on", "off", "true", "false", "null", "~", "none", ""}
    if text.lower() in reserved or any(char in text for char in ":#{}[],&*?|>'\"%@`"):
        return json.dumps(text, ensure_ascii=False)
    return text


def parse_gib(budget_gib: float) -> int:
    """把 GiB 预算换算成字节（供"多卡后能不能放下"的判定使用）."""
    if budget_gib <= 0:
        raise PEFTConfigError(f"预算必须为正数，收到 {budget_gib}")
    return int(budget_gib * 1024**3)


def device_fit_summary(plan: DistributedPlan, *, budget_gib: float) -> dict[str, Any]:
    """多卡计划在给定单卡预算下是否放得下（含"能放下"与"跑得起来"的区分）."""
    per_device_bytes = parse_gib(plan.per_device_gib)
    budget_bytes = parse_gib(budget_gib)
    return {
        "devices": plan.devices,
        "strategy": plan.strategy,
        "per_device_gib": round(plan.per_device_gib, 4),
        "budget_gib": budget_gib,
        "fits": per_device_bytes <= budget_bytes,
        "headroom_gib": round((budget_bytes - per_device_bytes) / 1024**3, 4),
        "note": "只比较权重 + 梯度 + 优化器状态，不含激活显存",
    }


__all__ = [
    "MIXED_PRECISION_PROFILES",
    "MIXED_PRECISION_VALUES",
    "SHARDING_STRATEGIES",
    "DistributedPlan",
    "MixedPrecisionProfile",
    "accelerate_config",
    "communication_bytes_per_step",
    "deepspeed_zero_config",
    "device_fit_summary",
    "launch_command",
    "mixed_precision_profile",
    "parse_gib",
    "per_device_memory",
    "plan_distributed",
    "render_config_yaml",
    "scale_learning_rate",
]
