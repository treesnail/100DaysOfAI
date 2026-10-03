"""显存算术：全参微调 / LoRA / QLoRA 到底各需要多少（M5-D3）.

这一节回答的是**"为什么需要 LoRA"**这个问题的量化版本。把训练显存拆成
六项之后，三种方法的差别一眼可见：

===========  ====================  ====================  ==================
项目          全参微调                LoRA                  QLoRA
===========  ====================  ====================  ==================
基座权重      ``2 B/参数``            ``2 B/参数``            ``0.515869 B/参数``
基座梯度      ``2 B/参数``            **0**                  **0**
基座优化器状态 ``8 B/参数``            **0**                  **0**
适配器权重    —                      ``2 B/可训练参数``       ``2 B/可训练参数``
适配器梯度    —                      ``2 B/可训练参数``       ``2 B/可训练参数``
适配器优化器  —                      ``8 B/可训练参数``       ``8 B/可训练参数``
===========  ====================  ====================  ==================

三处数字需要解释，因为它们决定了这张表能不能被信任：

1. **优化器状态是 8 字节/参数**：AdamW 保存一阶与二阶矩两个 fp32 张量，
   ``2 × 4 = 8``。这一项常常是训练显存里最大的一块，也是"7B 全参微调
   需要 80 GB"这个说法的来源：``6.738e9 × (2 + 2 + 8) = 80.86 GB``；
2. **LoRA 把"基座梯度 + 基座优化器状态"整块删掉了**（10 字节/参数），
   所以 LoRA 在 7B 上的基座部分是 ``13.48 GB``——**这才是 24 GB 卡
   能跑 7B 微调的原因**；
3. **QLoRA 只动"基座权重"这一项**：``2 → 0.515869`` 字节/参数。它不改变
   适配器的精度，也不改变优化器状态——**"4-bit 训练"这个说法本身就是错的**。

本模块**不估算激活显存**，并在报告里显式写出这一点。激活大小取决于
``batch × sequence_length × hidden × layers``，在 ``batch=2`` /
``seq=384`` 上是几百 MB，在长上下文 + 大 batch 上可以是十几 GB；
把它塞进一个"看起来很精确"的汇总数字里，只会让这个数字变得不可信。
**知道一份预算没算什么，比多一个看起来完整的表格重要。**
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from smart_research_agent.peft.config import LoRAConfig, PEFTConfigError, QLoRAConfig
from smart_research_agent.peft.targets import DecoderSpec, plan_lora

#: 1 GiB（1024³）；与硬盘厂商的 1 GB（10⁹）不同，见 ``format_bytes`` 的说明
GIB = 1024**3

#: 计算精度对应的字节数（训练时的权重/梯度都用它）
COMPUTE_DTYPE_BYTES: dict[str, int] = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
}

#: 优化器的状态开销（字节/参数）.
#:
#: ``adamw_torch`` 的 8 字节不是"经验值"：AdamW 为**每个**可训练参数保存
#: 一阶矩与二阶矩两个 fp32 张量，``2 × 4 = 8``。8-bit Adam 把这些状态压到
#: 2 字节/参数（4 倍节省），代价是收敛略慢且对超参更敏感；SGD 不需要状态，
#: 但收敛通常更慢——**没有免费的显存**。
OPTIMIZER_STATES: dict[str, int] = {
    "adamw_torch": 8,
    "adamw_bnb_8bit": 2,
    "sgd": 0,
    "sgd_momentum": 4,
    "adafactor": 2,
}

#: 常见单卡预算（GiB）。**用 GiB 而不是 GB**：24 GB 的卡实际是 24 GiB
#: （25.77 GB），而权重体积按字节算出来的数字用 GB 表示会显得更小，
#: 容易在"差一点点放不下"的时候得出错误结论。
DEVICE_BUDGETS: dict[str, float] = {
    "RTX 4090 (24 GiB)": 24.0,
    "RTX 6000 Ada (48 GiB)": 48.0,
    "A100 (40 GiB)": 40.0,
    "A100/H100 (80 GiB)": 80.0,
}

#: 三种策略的名字（顺序即报告的展示顺序）
STRATEGIES: tuple[str, ...] = ("full", "lora", "qlora")


def parse_size(value: str) -> float:
    """把 ``"24GiB"`` / ``"80GB"`` 解析成 GiB 数值（供 CLI 与测试使用）.

    单位后缀大小写不敏感；``GiB``/``G`` 都按 **1024³** 解释，``GB`` 按
    ``10⁹`` 解释后换算成 GiB。这个函数存在的理由很实际：报错信息里
    "要 13.5 GiB，可用 12.0 GiB"比"要 14495514624 字节"有用得多。
    """
    text = value.strip().lower()
    suffixes = (
        ("gib", 1.0),
        ("gb", 1000**3 / GIB),
        ("g", 1.0),
        ("mib", 1 / 1024),
        ("m", 1048576 / GIB),
    )
    for suffix, factor in suffixes:
        if text.endswith(suffix) and len(text) > len(suffix):
            body = text[: -len(suffix)].strip()
            try:
                return float(body) * factor
            except ValueError as exc:
                raise PEFTConfigError(f"无法解析容量 {value!r}") from exc
    raise PEFTConfigError(f"无法解析容量 {value!r}：请带单位（如 24GiB / 80GB）")


def format_bytes(value: float) -> str:
    """把字节数同时格式化成 GiB 与 GB 两个数字.

    一起给出的理由是它们**相差 7.4%**：一个 13.48 GiB 的 LoRA 基座写出来是
    "14.5 GB"。在"能不能放进 24 GB 卡"这种判断上，混用单位会给出错误答案，
    所以本模块的所有报告都两个都给。
    """
    if value < 0:
        raise PEFTConfigError(f"字节数不能为负，收到 {value}")
    return f"{value / GIB:.2f} GiB（{value / 1e9:.2f} GB）"


@dataclass
class MemoryBreakdown:
    """一次训练（某一种策略）的显存预算."""

    model: str
    strategy: str
    optimizer: str
    compute_dtype: str
    base_parameters: int
    trainable_parameters: int
    base_weight_bytes: int
    base_gradient_bytes: int
    base_optimizer_bytes: int
    adapter_weight_bytes: int
    adapter_gradient_bytes: int
    adapter_optimizer_bytes: int
    base_bits_per_parameter: float
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ 派生量
    @property
    def total_bytes(self) -> int:
        """总字节数（不含激活显存）."""
        return (
            self.base_weight_bytes
            + self.base_gradient_bytes
            + self.base_optimizer_bytes
            + self.adapter_weight_bytes
            + self.adapter_gradient_bytes
            + self.adapter_optimizer_bytes
        )

    @property
    def as_gib(self) -> float:
        """总字节数换算成 GiB."""
        return self.total_bytes / GIB

    @property
    def items(self) -> dict[str, int]:
        """六项明细（键名与模块文档里的表格一致）."""
        return {
            "base_weight_bytes": self.base_weight_bytes,
            "base_gradient_bytes": self.base_gradient_bytes,
            "base_optimizer_bytes": self.base_optimizer_bytes,
            "adapter_weight_bytes": self.adapter_weight_bytes,
            "adapter_gradient_bytes": self.adapter_gradient_bytes,
            "adapter_optimizer_bytes": self.adapter_optimizer_bytes,
        }

    def fits(self, budget_gib: float) -> bool:
        """给定单卡预算（GiB）能否放下（不含激活）."""
        if budget_gib <= 0:
            raise PEFTConfigError(f"预算必须为正数，收到 {budget_gib}")
        return self.as_gib <= budget_gib

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"{self.strategy:5s} | 可训练 {self.trainable_parameters:,} | "
            f"总计 {format_bytes(self.total_bytes)} | "
            f"基座权重 {self.base_bits_per_parameter:.3f} bit/参数"
        )

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典（附派生量）."""
        payload = asdict(self)
        payload.update(
            {
                "total_bytes": self.total_bytes,
                "as_gib": round(self.as_gib, 4),
                "as_gb": round(self.total_bytes / 1e9, 4),
                "items": self.items,
            }
        )
        return payload


def plan_memory(
    spec: DecoderSpec,
    *,
    strategy: str = "lora",
    lora_config: LoRAConfig | None = None,
    qlora_config: QLoRAConfig | None = None,
    optimizer: str = "adamw_torch",
    compute_dtype: str = "bfloat16",
    full_finetune_master_weights_fp32: bool = False,
) -> MemoryBreakdown:
    """算出一种策略在给定模型规格上的显存预算.

    ``strategy`` 取 ``full`` / ``lora`` / ``qlora``：

    - ``full``：所有权重都参与训练。``full_finetune_master_weights_fp32=True``
      时按"fp32 主权重 + bf16 计算"算（4+4+8 = 16 字节/参数），否则按
      "bf16 权重 + bf16 梯度 + fp32 优化器状态"算（2+2+8 = 12 字节/参数）。
      **这是同一个模型的两个相差 33% 的数字**，很多"7B 要多少显存"的争论
      其实是在争这个口径；
    - ``lora``：基座 bf16 冻结，只训适配器；
    - ``qlora``：基座 4-bit 冻结，只训适配器。``qlora_config`` 决定
      每参数存储（``nf4`` + 二级量化为 0.515869 字节）。
    """
    if strategy not in STRATEGIES:
        raise PEFTConfigError(
            f"未知的策略 {strategy!r}，可选：{', '.join(STRATEGIES)}"
        )
    if optimizer not in OPTIMIZER_STATES:
        raise PEFTConfigError(
            f"未知的优化器 {optimizer!r}，可选：{', '.join(sorted(OPTIMIZER_STATES))}"
        )
    if compute_dtype not in COMPUTE_DTYPE_BYTES:
        raise PEFTConfigError(
            f"未知的计算精度 {compute_dtype!r}，可选：{', '.join(sorted(COMPUTE_DTYPE_BYTES))}"
        )
    compute_bytes = COMPUTE_DTYPE_BYTES[compute_dtype]
    optimizer_bytes = OPTIMIZER_STATES[optimizer]
    total = spec.total_parameters()
    notes: list[str] = [
        "不含激活显存：它取决于 batch × sequence_length，无法只由模型规格决定",
    ]

    if strategy == "full":
        weight_bytes = 4 if full_finetune_master_weights_fp32 else compute_bytes
        gradient_bytes = 4 if full_finetune_master_weights_fp32 else compute_bytes
        if full_finetune_master_weights_fp32:
            weight_notes = "fp32 主权重（+ fp32 梯度）"
        else:
            weight_notes = f"{compute_dtype} 直接训练（+ {compute_dtype} 梯度）"
        notes.append(
            f"全参微调：权重与梯度都按 {weight_bytes} 字节/参数（{weight_notes}）"
        )
        if optimizer == "adamw_torch":
            notes.append(
                "AdamW 的一阶/二阶矩占 8 字节/参数，是全参微调里最大的一项；"
                "换成 adamw_bnb_8bit 可降到 2 字节/参数"
            )
        return MemoryBreakdown(
            model=spec.name,
            strategy=strategy,
            optimizer=optimizer,
            compute_dtype=compute_dtype,
            base_parameters=total,
            trainable_parameters=total,
            base_weight_bytes=total * weight_bytes,
            base_gradient_bytes=total * gradient_bytes,
            base_optimizer_bytes=total * optimizer_bytes,
            adapter_weight_bytes=0,
            adapter_gradient_bytes=0,
            adapter_optimizer_bytes=0,
            base_bits_per_parameter=weight_bytes * 8,
            notes=notes,
        )

    if lora_config is None:
        raise PEFTConfigError(f"strategy={strategy!r} 必须提供 lora_config")
    plan = plan_lora(spec, lora_config)
    trainable = plan.trainable_parameters
    if strategy == "qlora":
        effective_qlora = qlora_config or QLoRAConfig()
        effective_qlora.validate()
        base_bits = effective_qlora.bits_per_parameter
        base_weight_bytes = effective_qlora.base_weight_bytes(total)
        notes.append(
            f"QLoRA 基座：{effective_qlora.bnb_4bit_quant_type} "
            f"block={effective_qlora.block_size} "
            f"double_quant={effective_qlora.bnb_4bit_use_double_quant} → "
            f"{base_bits:.6f} bit/参数"
        )
        notes.append(
            "适配器仍是 16-bit：QLoRA 训练的是 fp16/bf16 的 A/B，4-bit 只用于存基座"
        )
    else:
        base_bits = compute_bytes * 8
        base_weight_bytes = total * compute_bytes
        notes.append(f"LoRA 基座：{compute_dtype} 冻结存储（{compute_bytes} 字节/参数）")

    notes.append(
        f"可训练参数 {trainable:,}（占 {plan.trainable_ratio:.4%}）："
        "梯度与优化器状态只按这个规模计费"
    )
    return MemoryBreakdown(
        model=spec.name,
        strategy=strategy,
        optimizer=optimizer,
        compute_dtype=compute_dtype,
        base_parameters=total,
        trainable_parameters=trainable,
        base_weight_bytes=base_weight_bytes,
        base_gradient_bytes=0,
        base_optimizer_bytes=0,
        adapter_weight_bytes=trainable * compute_bytes,
        adapter_gradient_bytes=trainable * compute_bytes,
        adapter_optimizer_bytes=trainable * optimizer_bytes,
        base_bits_per_parameter=base_bits,
        notes=notes,
    )


def compare_strategies(
    spec: DecoderSpec,
    *,
    lora_config: LoRAConfig,
    qlora_config: QLoRAConfig | None = None,
    optimizer: str = "adamw_torch",
    compute_dtype: str = "bfloat16",
) -> list[MemoryBreakdown]:
    """三种策略的对照（顺序：full / lora / qlora）."""
    return [
        plan_memory(
            spec,
            strategy=strategy,
            lora_config=lora_config,
            qlora_config=qlora_config,
            optimizer=optimizer,
            compute_dtype=compute_dtype,
        )
        for strategy in STRATEGIES
    ]


def savings_table(plans: list[MemoryBreakdown]) -> list[dict[str, Any]]:
    """把三种策略的对照整理成表（含相对全参微调的节省倍数）.

    节省倍数用**同一份预算**做分母：``full`` 策略的总字节。这个数字容易被
    误读——"LoRA 比全参省 5.9 倍"说的是**这份预算**（不含激活），而不是
    "训练总能快 5.9 倍"。算力与显存不是同一回事：LoRA 的前向反向仍然要
    过整个基座，它的加速主要来自"优化器状态更少"和"可以用更大的 batch"。
    """
    if not plans:
        raise PEFTConfigError("savings_table 需要至少一份预算")
    baseline = next((plan for plan in plans if plan.strategy == "full"), None)
    base_bytes = baseline.total_bytes if baseline else None
    rows: list[dict[str, Any]] = []
    for plan in plans:
        row = {
            "strategy": plan.strategy,
            "trainable_parameters": plan.trainable_parameters,
            "total_bytes": plan.total_bytes,
            "as_gib": round(plan.as_gib, 4),
            "as_gb": round(plan.total_bytes / 1e9, 4),
            "base_bits_per_parameter": round(plan.base_bits_per_parameter, 6),
        }
        if base_bytes:
            row["savings_factor"] = round(base_bytes / plan.total_bytes, 4)
        rows.append(row)
    return rows


def device_fit_table(
    plans: list[MemoryBreakdown],
    *,
    budgets: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """每种策略在每个单卡预算下能否放下（不含激活）.

    它回答的是那个最实际的问题："这张卡能不能跑"。判据只有一条
    ``总字节 ≤ 预算``，并且**明确排除了激活**——所以"能放下"不等于
    "能跑起来"。把这句话写进表格里，比在报告末尾加一句免责声明有效。
    """
    effective = budgets if budgets is not None else DEVICE_BUDGETS
    rows: list[dict[str, Any]] = []
    for name, budget_gib in effective.items():
        row: dict[str, Any] = {
            "device": name,
            "budget_gib": budget_gib,
            "fits": {
                plan.strategy: plan.fits(budget_gib) for plan in plans
            },
        }
        row["note"] = "仅比较「权重 + 梯度 + 优化器状态」，不含激活显存"
        rows.append(row)
    return rows


def adamw_state_bytes(parameters: int, *, bits: int = 32) -> int:
    """AdamW 的状态字节数（``2 × parameters × bits/8``）.

    单独抽出来是因为它出现的频率最高，而且公式里那个 ``2``（一阶矩 + 二阶矩）
    很容易被漏掉。省掉它的估算会让显存预算偏低 30% 左右。
    """
    if parameters <= 0:
        raise PEFTConfigError(f"parameters 必须为正整数，收到 {parameters}")
    if bits <= 0 or bits % 8 != 0:
        raise PEFTConfigError(f"bits 必须为 8 的正整数倍，收到 {bits}")
    return 2 * parameters * (bits // 8)


def minimal_device(budget_gib: float) -> str:
    """给定需要的显存（GiB），返回**够用的最小常见卡**型号名.

    用途是把"13.48 GiB"翻译成"一张 24 GiB 的卡就够了"。返回最小的那个
    满足条件的型号；没有任何型号满足时返回
    ``"需要多卡或更小的模型"``，而不是抛异常——这是一条信息性查询。
    """
    if budget_gib <= 0:
        raise PEFTConfigError(f"预算必须为正数，收到 {budget_gib}")
    candidates = sorted(DEVICE_BUDGETS.items(), key=lambda item: item[1])
    for name, size in candidates:
        if budget_gib <= size:
            return name
    return "需要多卡或更小的模型"


def quantization_memory_note(config: QLoRAConfig) -> str:
    """QLoRA 存储的一句话说明（供 API 与文档直接展示）."""
    constant_bits = config.constant_bits_per_parameter
    parts = [
        f"码点 {config.code_bits} bit",
        f"常数 {constant_bits:.6f} bit/参数",
        f"合计 {config.bits_per_parameter:.6f} bit = {config.bytes_per_parameter:.6f} 字节/参数",
    ]
    if config.bnb_4bit_use_double_quant:
        single = QLoRAConfig(
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            block_size=config.block_size,
            bnb_4bit_use_double_quant=False,
        )
        saved = single.bytes_per_parameter - config.bytes_per_parameter
        parts.append(f"二级量化比单重量化省 {saved:.6f} 字节/参数")
    else:
        parts.append("未启用二级量化：常数仍是 fp32")
    return " | ".join(parts)


def bytes_per_parameter_of_16bit(dtype: str = "bfloat16") -> int:
    """16-bit 基座的每参数字节数（QLoRA 的对照基线）."""
    if dtype not in COMPUTE_DTYPE_BYTES:
        raise PEFTConfigError(
            f"未知的精度 {dtype!r}，可选：{', '.join(sorted(COMPUTE_DTYPE_BYTES))}"
        )
    return COMPUTE_DTYPE_BYTES[dtype]


def parameter_scale(model_parameters: int) -> float:
    """把参数量换算成"十亿（B）"单位的显示值."""
    if model_parameters <= 0:
        raise PEFTConfigError(f"参数量必须为正整数，收到 {model_parameters}")
    return model_parameters / 1e9


def logistic_estimate_hours(
    *, total_steps: int, seconds_per_step: float, devices: int = 1
) -> float:
    """给定每步耗时与设备数，估算训练小时数（线性外推）.

    这个函数刻意只有一行算术，而且**不做非线性外推**：真实训练里
    "步数翻倍、耗时也大致翻倍"是可靠的，而"设备数翻倍、耗时减半"只在
    通信不成为瓶颈时才成立。把不确定的东西乘进一个精确公式，得到的是
    精确的错。
    """
    if total_steps <= 0 or seconds_per_step <= 0 or devices <= 0:
        raise PEFTConfigError("total_steps / seconds_per_step / devices 必须为正数")
    return total_steps * seconds_per_step / devices / 3600


__all__ = [
    "COMPUTE_DTYPE_BYTES",
    "DEVICE_BUDGETS",
    "GIB",
    "OPTIMIZER_STATES",
    "STRATEGIES",
    "MemoryBreakdown",
    "adamw_state_bytes",
    "bytes_per_parameter_of_16bit",
    "compare_strategies",
    "device_fit_table",
    "format_bytes",
    "logistic_estimate_hours",
    "minimal_device",
    "parameter_scale",
    "parse_size",
    "plan_memory",
    "quantization_memory_note",
    "savings_table",
]
