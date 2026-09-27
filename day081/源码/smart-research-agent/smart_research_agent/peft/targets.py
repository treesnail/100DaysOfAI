"""LoRA 的架构算术：在真实模型上算清"要训多少参数"（M5-D3）.

day050 的 ``plan_training`` 回答了"会跑多少步"；本模块回答它的姊妹问题：
**"要训多少参数、占多少显存"**。两个问题都必须在提交任务之前算完，而且
都不需要 GPU。

LoRA 的参数量公式简单到可以写在手心里：

.. code-block:: text

    一个被适配的线性层（in × out）：
        原参数     = in × out
        新增参数   = r × (in + out)          ← A 是 r×in，B 是 out×r
        单层比例   = r(in+out) / (in·out)

    当 in = out = d 时，比例退化为 **2r/d**——与 d 无关的那个漂亮式子。

day049 预先算过四个数字，本模块把它们做成可执行的断言：

.. code-block:: text

    d = 4096,  r = 8   -> 2r/d = 0.3906%
    d = 4096,  r = 16  -> 2r/d = 0.7812%
    d = 2048,  r = 8   -> 2r/d = 0.7812%
    d = 11008, r = 16  -> 2r/d = 0.2907%

**注意这四个数字是"单个矩阵"的比例，不是整个模型的比例。** 这是本节最容易
被误读的地方：真实模型上有 32 层、每层 7 个投影，因此模型级比例还取决于
``target_modules`` 的选择——在本模块里，``llama-2-7b`` + 只适配 q/v +
``r=8`` 的模型级比例是 **0.0622%**，而适配全部 7 个投影时是 **0.2958%**，
相差 **4.75 倍**。这正是"``target_modules`` 不是随手写的"的原因。

模型规格（``DecoderSpec``）里的每个数字都取自公开配置，并且**参数量会被
逐项核对**：``Qwen3-0.6B`` 的规格算出 **596 049 920** 个参数（28 层 ×
15 730 944 + 词嵌入 155 582 464 + 最终 norm 1024），与公开技术报告的
数字逐位一致；``Llama-2-7B`` 算出 **6 738 415 616**，也是公开的参数量。
**规格写错一个字段，这两个总数立刻对不上**——这是本模块最重要的一道自检。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from smart_research_agent.peft.config import LoRAConfig, PEFTConfigError

#: 目标模块名 → 它在 ``DecoderSpec.linear_modules`` 里的位置说明（供文档与校验）.
TARGET_LAYER_DESCRIPTIONS: dict[str, str] = {
    "q_proj": "查询投影（hidden → num_heads × head_dim）",
    "k_proj": "键投影（hidden → num_kv_heads × head_dim，GQA 时比 q 小）",
    "v_proj": "值投影（hidden → num_kv_heads × head_dim）",
    "o_proj": "输出投影（num_heads × head_dim → hidden）",
    "gate_proj": "MLP 门控投影（hidden → intermediate）",
    "up_proj": "MLP 升维投影（hidden → intermediate）",
    "down_proj": "MLP 降维投影（intermediate → hidden）",
    "weight": "参考模型的 bigram 权重矩阵（V → V，本课用它跑离线实验）",
}


@dataclass(frozen=True)
class ModuleShape:
    """一个线性层的形状（``in_features`` / ``out_features``）."""

    name: str
    in_features: int
    out_features: int

    def __post_init__(self) -> None:
        if self.in_features <= 0 or self.out_features <= 0:
            raise PEFTConfigError(
                f"模块 {self.name} 的形状必须为正整数，收到 "
                f"{self.in_features}×{self.out_features}"
            )

    @property
    def parameters(self) -> int:
        """该层原有参数个数."""
        return self.in_features * self.out_features

    def adapter_parameters(self, r: int) -> int:
        """LoRA 在该层新增的参数个数 ``r(in + out)``."""
        if r <= 0:
            raise PEFTConfigError(f"r 必须为正整数，收到 {r}")
        return r * (self.in_features + self.out_features)

    def ratio(self, r: int) -> float:
        """该层的 LoRA 参数量占比（**单层比例**，不是模型级比例）."""
        return self.adapter_parameters(r) / self.parameters

    def square_shortcut_ratio(self, r: int) -> float | None:
        """``in == out`` 时的简化比例 ``2r/d``（day049 用的就是它）.

        非方阵返回 ``None``——刻意不给出"近似值"：一个 ``4096×11008`` 的
        MLP 投影用 ``2r/11008`` 去估计，会得到 0.2907%，而精确值是
        **0.5360%**（少了 46%）。**近似值只在它成立的条件下才叫近似。**
        """
        if self.in_features != self.out_features:
            return None
        return 2 * r / self.in_features


@dataclass(frozen=True)
class DecoderSpec:
    """一个 decoder-only 模型的架构规格（逐字段取自公开配置）.

    ``head_dim`` 单独给出而不是用 ``hidden_size // num_heads`` 算：Qwen3 系列
    就属于"两者不相等"的情况（``0.6B`` 的 ``hidden=1024``、``heads=16``，
    但 ``head_dim=128``），用除法算会得到 64，于是 q/o 投影的形状全部算错，
    整份参数量核对随之失效。**能用公开配置核对的东西，不要用公式猜。**
    """

    name: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    #: 词嵌入与输出层是否共享权重（Qwen3 小模型为 True，Llama-2 为 False）
    tie_word_embeddings: bool
    #: 是否在注意力内部对 q/k 做 RMSNorm（Qwen3 为 True，Llama-2 为 False）
    qk_norm: bool = False
    #: 线性层是否有偏置（Llama-2 / Qwen3 均为 False）
    linear_bias: bool = False

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads != 0:
            raise PEFTConfigError(
                f"{self.name}: num_heads({self.num_heads}) 必须能被 "
                f"num_kv_heads({self.num_kv_heads}) 整除（GQA 的分组要求）"
            )
        if self.head_dim <= 0 or self.hidden_size <= 0:
            raise PEFTConfigError(f"{self.name}: hidden_size / head_dim 必须为正整数")

    @property
    def attention_width(self) -> int:
        """查询投影的输出宽度（= ``num_heads × head_dim``；GQA 时常大于 hidden）."""
        return self.num_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        """键/值投影的输出宽度（= ``num_kv_heads × head_dim``）."""
        return self.num_kv_heads * self.head_dim

    def linear_modules(self) -> tuple[ModuleShape, ...]:
        """每层的七类线性投影（顺序固定，便于逐项对照）."""
        hidden = self.hidden_size
        intermediate = self.intermediate_size
        shapes = (
            ModuleShape("q_proj", hidden, self.attention_width),
            ModuleShape("k_proj", hidden, self.kv_width),
            ModuleShape("v_proj", hidden, self.kv_width),
            ModuleShape("o_proj", self.attention_width, hidden),
            ModuleShape("gate_proj", hidden, intermediate),
            ModuleShape("up_proj", hidden, intermediate),
            ModuleShape("down_proj", intermediate, hidden),
        )
        return shapes

    def layer_parameters(self) -> int:
        """单层参数量（线性投影 + LayerNorm 权重）.

        LayerNorm 是必须算进去的小项：单层 4096 隐藏维的模型有 2 个 norm，
        32 层就是 262 144 个参数。它不影响"LoRA 省了多少"的量级，但**少了
        它，模型总参数量就对不上公开数字**，而"对不上"正是发现规格写错的
        唯一手段。
        """
        linear = sum(shape.parameters for shape in self.linear_modules())
        norms = 2 * self.hidden_size
        if self.qk_norm:
            norms += 2 * self.head_dim
        if self.linear_bias:
            linear += sum(shape.out_features for shape in self.linear_modules())
        return linear + norms

    def embedding_parameters(self) -> int:
        """词嵌入参数量（共享权重时只算一份）."""
        if self.tie_word_embeddings:
            return self.vocab_size * self.hidden_size
        return 2 * self.vocab_size * self.hidden_size

    def total_parameters(self) -> int:
        """模型总参数量（= 层 × num_layers + 词嵌入 + 最终 norm）."""
        return (
            self.num_layers * self.layer_parameters()
            + self.embedding_parameters()
            + self.hidden_size
        )

    def linear_parameters(self) -> int:
        """全部线性层的参数量（不含嵌入与 norm）.

        LoRA 的"分母"用哪个量值得说清楚：报告里说"训了模型的 0.06%"时，
        分母是**总参数**（含嵌入）；而"适配了线性层的 0.5%"时，分母是
        线性层参数。两者相差一倍以上（嵌入占了 7B 模型的 3.9%），
        所以数字必须带着分母一起说。
        """
        return sum(
            shape.parameters for shape in self.linear_modules()
        ) * self.num_layers

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典（含派生量）."""
        payload = asdict(self)
        payload.update(
            {
                "attention_width": self.attention_width,
                "kv_width": self.kv_width,
                "layer_parameters": self.layer_parameters(),
                "embedding_parameters": self.embedding_parameters(),
                "linear_parameters": self.linear_parameters(),
                "total_parameters": self.total_parameters(),
                "linear_modules": [shape.name for shape in self.linear_modules()],
            }
        )
        return payload


#: 公开配置的模型规格（每个数字都能在对应模型的 ``config.json`` 里核对）.
#:
#: - ``llama-2-7b``：``hidden=4096 / intermediate=11008 / layers=32 /
#:   heads=32 / kv_heads=32（MHA）/ head_dim=128 / vocab=32000 /
#:   tie=False``，算出 **6 738 415 616** 个参数；
#: - ``qwen3-0.6b``：``hidden=1024 / intermediate=3072 / layers=28 /
#:   heads=16 / kv_heads=8（GQA）/ head_dim=128 / vocab=151936 /
#:   tie=True / qk_norm=True``，算出 **596 049 920** 个参数。它也是
#:   day050 生成脚本里的缺省基座，因此那天写下的"单卡即可 SFT"这句话，
#:   在这里有了参数量依据。
MODEL_SPECS: dict[str, DecoderSpec] = {
    "llama-2-7b": DecoderSpec(
        name="llama-2-7b",
        hidden_size=4096,
        intermediate_size=11008,
        num_layers=32,
        num_heads=32,
        num_kv_heads=32,
        head_dim=128,
        vocab_size=32000,
        tie_word_embeddings=False,
    ),
    "qwen3-0.6b": DecoderSpec(
        name="qwen3-0.6b",
        hidden_size=1024,
        intermediate_size=3072,
        num_layers=28,
        num_heads=16,
        num_kv_heads=8,
        head_dim=128,
        vocab_size=151936,
        tie_word_embeddings=True,
        qk_norm=True,
    ),
}


def single_matrix_ratio(in_features: int, out_features: int, r: int) -> float:
    """单个线性层的 LoRA 参数量占比 ``r(in+out)/(in·out)``（精确值）."""
    if in_features <= 0 or out_features <= 0 or r <= 0:
        raise PEFTConfigError("in_features / out_features / r 必须为正整数")
    return r * (in_features + out_features) / (in_features * out_features)


def square_ratio_shortcut(dimension: int, r: int) -> float:
    """方阵的简化比例 ``2r/d``（= ``single_matrix_ratio(d, d, r)``）.

    day049 算出的那四个比例用的就是这个式子。把它单独做成函数，是为了让
    "简化式与精确式在方阵上等价"这条性质可以被一条断言守住
    （``square_ratio_shortcut(d, r) == single_matrix_ratio(d, d, r)``）。
    """
    if dimension <= 0 or r <= 0:
        raise PEFTConfigError("dimension / r 必须为正整数")
    return 2 * r / dimension


def reference_matrix_ratios() -> list[dict[str, float]]:
    """day049 预先算出的四个"单矩阵比例"，用同一套算术在运行时复算.

    四个数字是 day049 教程里的结论，本课把它们变成函数输出，从而可以被
    测试断言：只要有人改了公式，这四个数字立刻变红。
    """
    cases = ((4096, 8), (4096, 16), (2048, 8), (11008, 16))
    return [
        {
            "dimension": float(dimension),
            "r": float(r),
            "ratio": square_ratio_shortcut(dimension, r),
            "ratio_percent": square_ratio_shortcut(dimension, r) * 100,
        }
        for dimension, r in cases
    ]


@dataclass
class LoRAPlan:
    """一次 LoRA 的参数量画像（模型级 + 单层级）."""

    model: str
    targets: tuple[str, ...]
    r: int
    lora_alpha: int
    scaling: float
    base_parameters: int
    adapter_parameters: int
    trainable_parameters: int
    frozen_parameters: int
    trainable_ratio: float
    rank_upper_bound: int
    adapter_parameters_per_layer: int
    matched_modules_per_layer: tuple[str, ...]
    per_module_ratio: dict[str, float]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典."""
        payload = asdict(self)
        payload["targets"] = list(self.targets)
        payload["matched_modules_per_layer"] = list(self.matched_modules_per_layer)
        return payload

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"{self.model} | LoRA r={self.r} alpha={self.lora_alpha} "
            f"targets={list(self.targets)} | 可训练 {self.trainable_parameters} / "
            f"总 {self.base_parameters} = {self.trainable_ratio:.4%} | "
            f"每层新增 {self.adapter_parameters_per_layer}"
        )


def plan_lora(spec: DecoderSpec, config: LoRAConfig) -> LoRAPlan:
    """算出 LoRA 在给定模型规格上的参数量，并生成风险告警.

    ``warnings`` 里每一条都对应一个**真实会发生的问题**：

    1. 目标预设**没有命中任何模块**——配置看着正常，实际一个适配器都没插上；
    2. ``r`` 超过被适配层的最小维度——此时秩上界由维度而非 ``r`` 决定，
       再加秩不会再增加容量，只是白花显存；
    3. 可训练参数占比超过 1%——"参数高效"的收益在这个量级上已经开始消失
       （全参微调在 7B 上是 12 字节/参数的内存，LoRA 若训 1% 的参数，
       优化器状态就只有 1% 的规模，节省比例降到 10 倍以内）；
    4. 只适配了 ``q_proj`` 或只适配了 ``v_proj`` 之一——LoRA 原论文的消融
       实验比较的是"q/v 一起适配"，单边适配不是同一件事。
    """
    config.validate()
    targets = config.resolved_targets
    modules = spec.linear_modules()
    matched = tuple(shape for shape in modules if shape.name in targets)
    if not matched:
        raise PEFTConfigError(
            f"{spec.name} 的线性层里没有匹配 target_modules={list(targets)} 的模块；"
            f"可用模块：{', '.join(shape.name for shape in modules)}"
        )

    adapter_per_layer = sum(shape.adapter_parameters(config.r) for shape in matched)
    adapter_total = adapter_per_layer * spec.num_layers
    bias_parameters = 0
    if config.bias in ("all", "lora_only"):
        # bias=all 时全部线性层偏置都训；lora_only 只训被适配层的偏置。
        # Llama-2 / Qwen3 都没有线性偏置，因此这个分支在真实规格上是 0——
        # 但代码必须算它，否则规范一旦换成带偏置的模型就静默算错。
        if config.bias == "all":
            bias_parameters = sum(shape.out_features for shape in modules) * spec.num_layers
        else:
            bias_parameters = sum(shape.out_features for shape in matched) * spec.num_layers
        if config.bias == "all":
            bias_parameters += 2 * spec.hidden_size * spec.num_layers
    trainable = adapter_total + bias_parameters
    base_parameters = spec.total_parameters()
    # 占比的**分母含适配器自己**：这与 peft ``print_trainable_parameters()``
    # 的口径一致（``trainable / all params``），因此报告里的数字与训练日志里
    # 打印的数字可以直接对照。两种口径在小模型上差得很明显——参考模型
    # （V=557）上 8912/310806 = 2.867%，而 8912/319718 = **2.787%**。
    total_with_adapter = base_parameters + trainable

    warnings: list[str] = []
    minimum_dimension = min(min(shape.in_features, shape.out_features) for shape in matched)
    if config.r > minimum_dimension:
        warnings.append(
            f"r={config.r} 超过被适配层的最小维度 {minimum_dimension}："
            f"秩上界由维度决定（={minimum_dimension}）而不是 r，继续加大 r 不会"
            "增加容量，只是线性地增加显存。"
        )
    ratio = trainable / total_with_adapter
    if ratio > 0.01:
        warnings.append(
            f"可训练参数占比 {ratio:.4%} 超过 1%：此时优化器状态的规模已达到全参的 "
            f"{ratio:.2%}，LoRA 的显存优势被大幅削弱（全参 AdamW 是 12 字节/参数，"
            "LoRA 若要训 1% 的参数，节省比例就只剩 10 倍以内）。"
        )
    if {"q_proj", "v_proj"} - set(targets) and ({"q_proj", "v_proj"} & set(targets)):
        warnings.append(
            "只适配了 q_proj / v_proj 中的一个：LoRA 原论文的消融实验比较的是"
            "「q/v 一起适配」，单边适配的可训练参数少一半，不是同一件事。"
        )
    if spec.linear_bias and config.bias != "none":
        warnings.append(
            f"{spec.name} 的线性层带偏置且 bias={config.bias}：偏置参数与秩无关，"
            "会让「可训练参数 = 2·r·d」这条手算公式不再成立。"
        )

    return LoRAPlan(
        model=spec.name,
        targets=targets,
        r=config.r,
        lora_alpha=config.lora_alpha,
        scaling=config.scaling,
        base_parameters=base_parameters,
        adapter_parameters=adapter_total,
        trainable_parameters=trainable,
        frozen_parameters=base_parameters,
        trainable_ratio=ratio,
        rank_upper_bound=config.rank_upper_bound(
            max(shape.in_features for shape in matched),
            max(shape.out_features for shape in matched),
        ),
        adapter_parameters_per_layer=adapter_per_layer,
        matched_modules_per_layer=tuple(shape.name for shape in matched),
        per_module_ratio={
            shape.name: shape.ratio(config.r) for shape in matched
        },
        warnings=warnings,
    )


def rank_comparison(
    spec: DecoderSpec, *, ranks: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
) -> list[dict[str, float | int]]:
    """给出一张"秩 → 参数量/占比"的对照表（默认预设 ``attention_all``）.

    这张表回答的是"``r`` 该取多少"：参数量对 ``r`` 是**严格线性**的
    （每个秩 +1，每层多 ``in+out`` 个参数），因此"加大秩"从来不是免费的；
    而容量对 ``r`` 的收益是递减的——这是 LoRA 里最需要经验的地方，
    也是"先看参数量表、再决定要不要加秩"这条纪律的来源。
    """
    config_base = LoRAConfig(target_modules="attention_all")
    rows: list[dict[str, float | int]] = []
    for r in ranks:
        plan = plan_lora(spec, config_base.with_overrides(r=r, lora_alpha=2 * r))
        rows.append(
            {
                "r": r,
                "scaling": plan.scaling,
                "adapter_parameters": plan.adapter_parameters,
                "parameters_per_layer": plan.adapter_parameters_per_layer,
                "trainable_ratio": plan.trainable_ratio,
                "rank_upper_bound": plan.rank_upper_bound,
            }
        )
    return rows


def target_preset_table(spec: DecoderSpec, *, r: int = 8) -> list[dict[str, Any]]:
    """五种预设在同一模型上的参数量对照（``r`` 固定，只变目标模块）.

    实测（``llama-2-7b`` / ``r=8``，占比的分母含适配器自己）::

        attention       q,v          4 194 304    0.0622%
        attention_all   q,k,v,o      8 388 608    0.1243%
        mlp             gate,up,down 11 599 872   0.1718%
        all_linear      七类          19 988 480   0.2958%

    同一份数据、同一个 ``r``，参数量相差 **4.75 倍**——这四行就是"为什么
    ``target_modules`` 必须显式写出来"的最好说明。
    """
    rows: list[dict[str, Any]] = []
    for preset in ("attention", "attention_all", "mlp", "all_linear"):
        try:
            plan = plan_lora(spec, LoRAConfig(r=r, lora_alpha=2 * r, target_modules=preset))
        except PEFTConfigError:  # pragma: no cover - 预设与规格永远匹配得上
            continue
        rows.append(
            {
                "preset": preset,
                "targets": list(plan.targets),
                "adapter_parameters": plan.adapter_parameters,
                "parameters_per_layer": plan.adapter_parameters_per_layer,
                "trainable_ratio": plan.trainable_ratio,
                "trainable_ratio_percent": plan.trainable_ratio * 100,
            }
        )
    return rows


def adapter_footprint(
    *, r: int, in_features: int, out_features: int, dtype_bytes: int = 2
) -> dict[str, int | float]:
    """单个适配器的落盘体积（``A`` + ``B`` 两个矩阵）.

    这是 LoRA 最实用的性质之一：**适配器是可以被"带走"的**。``llama-2-7b``
    上的 ``attention_all`` / ``r=8`` 适配器是 8 388 608 个参数，bf16 落盘
    约 **16 MB**——而基座是 13.5 GB。于是"一个基座 + N 个小适配器"成为可能，
    这正是 day052 要落地的部署形态。
    """
    if r <= 0 or in_features <= 0 or out_features <= 0 or dtype_bytes <= 0:
        raise PEFTConfigError("r / in_features / out_features / dtype_bytes 必须为正整数")
    parameters = r * (in_features + out_features)
    return {
        "parameters": parameters,
        "bytes": parameters * dtype_bytes,
        "mebibytes": parameters * dtype_bytes / (1024 * 1024),
        "equivalent_full_layer_parameters": in_features * out_features,
        "compression_ratio": (in_features * out_features) / parameters,
    }


def describe_model(spec: DecoderSpec) -> str:
    """人类可读的模型规格摘要（含参数量自检值）."""
    total = spec.total_parameters()
    return (
        f"{spec.name} | hidden={spec.hidden_size} intermediate={spec.intermediate_size} "
        f"layers={spec.num_layers} heads={spec.num_heads}/{spec.num_kv_heads} "
        f"head_dim={spec.head_dim} vocab={spec.vocab_size} "
        f"tie={spec.tie_word_embeddings} qk_norm={spec.qk_norm} | "
        f"总参数 {total:,}（{total / 1e9:.3f} B）| 单层 {spec.layer_parameters():,} | "
        f"线性层 {spec.linear_parameters():,}（占 {spec.linear_parameters() / total:.2%}）"
    )


def parameter_check(spec: DecoderSpec) -> tuple[int, float]:
    """返回 ``(总参数, 十亿为单位的参数)``——教程与测试的核对入口."""
    total = spec.total_parameters()
    return total, total / 1e9


def theoretical_adapter_parameter_cap(spec: DecoderSpec) -> int:
    """理论上的上限：把每个线性层都换成"全秩" LoRA 会多出多少参数.

    ``r = min(in, out)`` 时 LoRA 与全参微调的可训练参数量同阶，此时
    "参数高效"完全失效。给出这个数字是为了让"r 不能乱加"有一条可量化的
    边界，而不是一句经验之谈。
    """
    total = 0
    for shape in spec.linear_modules():
        rank = min(shape.in_features, shape.out_features)
        total += shape.adapter_parameters(rank)
    return total * spec.num_layers


def ratio_curve(spec: DecoderSpec, *, r_max: int = 128) -> list[tuple[int, float]]:
    """``(r, 模型级可训练占比)`` 曲线（2 的幂次采样）.

    它的形状是**严格线性**的：占位比例对 ``r`` 没有拐点，因此"秩的收益递减"
    只能通过评估指标观察，不能通过参数量观察。这条区分很重要——它意味着
    **调秩必须配一个评估集**，否则你只是在花算力。
    """
    if r_max < 1:
        raise PEFTConfigError(f"r_max 必须为正整数，收到 {r_max}")
    ranks = [2**power for power in range(0, int(math.log2(r_max)) + 1)] if r_max >= 1 else []
    ranks = [r for r in ranks if r <= r_max]
    return [(r, plan_lora(spec, LoRAConfig(r=r, lora_alpha=2 * r)).trainable_ratio) for r in ranks]


__all__ = [
    "MODEL_SPECS",
    "TARGET_LAYER_DESCRIPTIONS",
    "DecoderSpec",
    "LoRAPlan",
    "ModuleShape",
    "adapter_footprint",
    "describe_model",
    "parameter_check",
    "plan_lora",
    "rank_comparison",
    "ratio_curve",
    "reference_matrix_ratios",
    "single_matrix_ratio",
    "square_ratio_shortcut",
    "target_preset_table",
    "theoretical_adapter_parameter_cap",
]
