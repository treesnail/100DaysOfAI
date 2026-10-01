"""实验读数：**把"两个默认值不一样"变成一行可以复算的数**（day085 / M7-D9）.

源码精读最容易停在"我读懂了"，而"读懂"不可反驳。本模块因此把五个问题各做成一张表：

```text
filter_study       四个 warper 的差别：同一个分布上，top_k / top_p / 温度各留下几个候选
sampling_study     四种策略在同一个玩具模型上生成什么（**同一个种子 = 同一串 token**）
beam_study         beam 宽度与长度惩罚怎么改变最终序列（以及"不惩罚会偏向短序列"）
activation_study   gelu 与 gelu_new 到底差多少（逐点最大绝对差）
layer_norm_study   三个 eps 默认值（1e-5 / 1e-12 / 1e-6）各自让"方差离 1"多远
```

三条整理纪律（沿用 day070 与 day084 的传统）：

```text
① 每个数都标注来源（哪个函数、哪个种子）
② 不同口径不混列（保留个数 / 熵 / 文本长度各自一列）
③ 只有一次观测的表只能读成"一次观测"——本模块的每张表都写死输入与种子，
   因此它是一条**可以被重新跑出来**的读数，而不是一次运气
```
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from smart_research_agent.encoder_decoder.layers import layer_norm
from smart_research_agent.hf_source.blocks import gelu_exact, gelu_new, make_hf_block
from smart_research_agent.hf_source.errors import ParameterError
from smart_research_agent.hf_source.generation import (
    FILTERED_LOGIT,
    FILTER_LOGITS,
    distribution_entropy,
    generate,
    softmax_of,
    top_k_filter,
    top_p_filter,
    transform_logits,
    validate_settings,
)
from smart_research_agent.hf_source.types import (
    LN_EPS_BERT,
    LN_EPS_DEFAULTS,
    LN_EPS_GPT2,
    LN_EPS_T5,
    GenerationSettings,
)
from smart_research_agent.math_foundations.types import Matrix, Vector, validate_matrix

#: 四个"策略"在过滤器实验里的名字（与 :data:`generation.strategy_of` 的返回值一致）.
FILTER_CASES: tuple[tuple[str, GenerationSettings], ...] = (
    ("raw", GenerationSettings(do_sample=True, temperature=1.0)),
    ("temperature=0.5", GenerationSettings(do_sample=True, temperature=0.5)),
    ("temperature=2.0", GenerationSettings(do_sample=True, temperature=2.0)),
    ("top_k=2", GenerationSettings(do_sample=True, top_k=2)),
    ("top_k=4", GenerationSettings(do_sample=True, top_k=4)),
    ("top_p=0.5", GenerationSettings(do_sample=True, top_p=0.5)),
    ("top_p=0.9", GenerationSettings(do_sample=True, top_p=0.9)),
)

#: 激活实验的取点（**含 0**——两式在那里的函数值必须同为 0、导数同为 0.5）.
ACTIVATION_POINTS: tuple[float, ...] = (-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0)

#: 玩具模型的词表大小。
TOY_VOCAB = 8

#: 玩具模型的"尖度"（越大越尖；它只是让策略之间的差别可读）。
TOY_SHARPNESS = 1.6


def toy_logits(
    tokens: tuple[int, ...],
    *,
    vocab: int = TOY_VOCAB,
    sharpness: float = TOY_SHARPNESS,
) -> Vector:
    """一个**确定性**的玩具"模型"：给定 token 序列，返回下一个 token 的 logits.

    它刻意不是随机数：这样"同一个种子给出同样的文本"才是**策略**的性质，
    而不是"模型的随机性恰好一样"。它的取值由最后两个 token 决定，
    因此不同策略会走出不同的轨迹，而每条轨迹都能被重新跑出来。
    """
    if not tokens:
        raise ParameterError("toys 模型需要至少一个 token。")
    last = tokens[-1]
    previous = tokens[-2] if len(tokens) > 1 else last
    return tuple(
        math.cos(0.7 * index + 0.3 * last)
        - sharpness * ((index - (last + previous + 1) % vocab) / vocab) ** 2
        for index in range(vocab)
    )


@dataclass(frozen=True)
class FilterRow:
    """一个过滤配置的读数：留下几个候选、最尖的概率、熵."""

    label: str
    kept: int
    top_probability: float
    entropy: float

    def line(self) -> str:
        """一行可读的读数."""
        return (
            f"{self.label:>16} | 保留 {self.kept}/{len(FILTER_LOGITS)} | "
            f"峰值 {self.top_probability:.4f} | 熵 {self.entropy:.4f}"
        )

    def to_dict(self) -> dict[str, object]:
        """摊平成字段."""
        return {
            "label": self.label,
            "kept": self.kept,
            "top_probability": self.top_probability,
            "entropy": self.entropy,
        }


def filter_study() -> tuple[FilterRow, ...]:
    """四个 warper 在**同一个分布**上各留下几个候选（这是"预算即过滤器"的读数）.

    它同时回答了"为什么 top_k 与 top_p 不能混着读"：
    `top_k` 的保留个数是**常数**（由参数决定），
    而 `top_p` 的保留个数随**温度**变化（由数据决定）。
    """
    rows: list[FilterRow] = []
    for label, settings in FILTER_CASES:
        validate_settings(settings, len(FILTER_LOGITS))
        transformed, kept = transform_logits(FILTER_LOGITS, settings, ())
        probabilities = softmax_of(transformed)
        rows.append(
            FilterRow(
                label=label,
                kept=kept,
                top_probability=max(probabilities),
                entropy=distribution_entropy(probabilities),
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class NucleusRow:
    """top-p 的一行：`p`、保留个数、以及"累积概率刚好超过 p"的那一格."""

    top_p: float
    kept: int
    cumulative: float
    removed: int

    def line(self) -> str:
        """一行可读的读数."""
        return (
            f"top_p={self.top_p:.2f} | 保留 {self.kept} | 删掉 {self.removed} | "
            f"留下的最小累积概率 {self.cumulative:.4f}"
        )

    def to_dict(self) -> dict[str, object]:
        """摊平成字段."""
        return {
            "top_p": self.top_p,
            "kept": self.kept,
            "cumulative": self.cumulative,
            "removed": self.removed,
        }


def nucleus_study(p_values: tuple[float, ...] = (0.1, 0.3, 0.5, 0.8, 0.95)) -> tuple[NucleusRow, ...]:
    """扫一遍 ``top_p``：**保留个数由数据决定**，因此这张表逐行都可能不同.

    ``cumulative`` 是"被留下的那些 token 的总概率"——它恰好是"刚好超过 p"的那个数，
    也就是 HF 那行 `sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0`
    所保证的东西：**核永不为空**。
    """
    probabilities = softmax_of(FILTER_LOGITS)
    rows: list[NucleusRow] = []
    for value in p_values:
        filtered, kept = top_p_filter(FILTER_LOGITS, value)
        live = [
            probability
            for probability, logit in zip(probabilities, filtered)
            if logit != FILTERED_LOGIT
        ]
        rows.append(
            NucleusRow(
                top_p=value,
                kept=kept,
                cumulative=math.fsum(live),
                removed=len(FILTER_LOGITS) - kept,
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class TopKRow:
    """top-k 的一行：`k`、实际保留个数（`k` 大于词表时被夹住）."""

    top_k: int
    kept: int
    top_probability: float

    def line(self) -> str:
        """一行可读的读数."""
        return f"top_k={self.top_k} | 保留 {self.kept} | 峰值 {self.top_probability:.4f}"

    def to_dict(self) -> dict[str, object]:
        """摊平成字段."""
        return {"top_k": self.top_k, "kept": self.kept, "top_probability": self.top_probability}


def top_k_study(k_values: tuple[int, ...] = (1, 2, 3, 6)) -> tuple[TopKRow, ...]:
    """扫一遍 ``top_k``：**保留个数是常数**（与 top-p 正好相反）."""
    rows: list[TopKRow] = []
    for value in k_values:
        filtered, kept = top_k_filter(FILTER_LOGITS, value)
        rows.append(
            TopKRow(top_k=value, kept=kept, top_probability=max(softmax_of(filtered)))
        )
    return tuple(rows)


@dataclass(frozen=True)
class SamplingRow:
    """一次生成的读数：策略、设置、生成的 token、以及逐步保留个数."""

    strategy: str
    settings: str
    generated: tuple[int, ...]
    kept: tuple[int, ...]
    entropy: tuple[float, ...]

    def line(self) -> str:
        """一行可读的读数（token 序列用逗号连接，便于逐行比）."""
        tokens = ",".join(str(token) for token in self.generated)
        return (
            f"{self.strategy:>8} | {self.settings:<26} | 生成 [{tokens}] | "
            f"逐步保留 {list(self.kept)}"
        )

    def to_dict(self) -> dict[str, object]:
        """摊平成字段."""
        return {
            "strategy": self.strategy,
            "settings": self.settings,
            "generated": list(self.generated),
            "kept": list(self.kept),
            "entropy": list(self.entropy),
        }


#: 采样实验的四个配置（**同一个种子**，因此差别只来自策略）.
SAMPLING_CASES: tuple[tuple[str, GenerationSettings], ...] = (
    (
        "greedy",
        GenerationSettings(max_new_tokens=6, do_sample=False, seed=7),
    ),
    (
        "sample/T=1",
        GenerationSettings(max_new_tokens=6, do_sample=True, temperature=1.0, seed=7),
    ),
    (
        "top_k=2",
        GenerationSettings(max_new_tokens=6, do_sample=True, top_k=2, seed=7),
    ),
    (
        "top_p=0.7",
        GenerationSettings(max_new_tokens=6, do_sample=True, top_p=0.7, seed=7),
    ),
)


def sampling_study(
    prompt: tuple[int, ...] = (1, 3),
    cases: tuple[tuple[str, GenerationSettings], ...] = SAMPLING_CASES,
) -> tuple[SamplingRow, ...]:
    """四种策略在同一个玩具模型上生成什么（**同一个种子**）.

    `top_k` / `top_p` 两条在最前面会与 `sample` 分道扬镳：它们把"不可能的 token"
    提前置成 `-1e30`，于是逆变换采样落在尾部的概率被压成 0——
    而 `sample` 会真的偶尔走到尾部（这就是"长尾"的代价）。
    """
    rows: list[SamplingRow] = []
    for label, settings in cases:
        validate_settings(settings, TOY_VOCAB)
        result = generate(toy_logits, prompt, settings, vocab=TOY_VOCAB)
        rows.append(
            SamplingRow(
                strategy=label,
                settings=f"T={settings.temperature:g} k={settings.top_k} p={settings.top_p:g}",
                generated=result.generated,
                kept=tuple(step.kept for step in result.steps),
                entropy=tuple(round(step.entropy, 4) for step in result.steps),
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class BeamRow:
    """一次 beam 的读数：宽度、长度惩罚、生成的 token、归一化分数."""

    beams: int
    length_penalty: float
    generated: tuple[int, ...]
    note: str

    def line(self) -> str:
        """一行可读的读数."""
        tokens = ",".join(str(token) for token in self.generated)
        return (
            f"beams={self.beams} lp={self.length_penalty:g} | 生成 [{tokens}] | {self.note}"
        )

    def to_dict(self) -> dict[str, object]:
        """摊平成字段."""
        return {
            "beams": self.beams,
            "length_penalty": self.length_penalty,
            "generated": list(self.generated),
            "note": self.note,
        }


#: beam 实验的配置：贪心一行 + 四行 beam（宽度与长度惩罚各扫两次）.
BEAM_CASES: tuple[GenerationSettings, ...] = (
    GenerationSettings(max_new_tokens=5, do_sample=False, seed=7),
    GenerationSettings(max_new_tokens=5, do_sample=False, num_beams=2, length_penalty=1.0, seed=7),
    GenerationSettings(max_new_tokens=5, do_sample=False, num_beams=2, length_penalty=0.5, seed=7),
    GenerationSettings(max_new_tokens=5, do_sample=False, num_beams=3, length_penalty=2.0, seed=7),
)


def beam_study(prompt: tuple[int, ...] = (2, 5)) -> tuple[BeamRow, ...]:
    """贪心与 beam 的对照：**长度惩罚改变的是选哪条前缀，而不是模型**."""
    rows: list[BeamRow] = []
    for settings in BEAM_CASES:
        validate_settings(settings, TOY_VOCAB)
        result = generate(toy_logits, prompt, settings, vocab=TOY_VOCAB)
        note = "；".join(result.notes)
        rows.append(
            BeamRow(
                beams=settings.num_beams,
                length_penalty=settings.length_penalty,
                generated=result.generated,
                note=note,
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class ActivationRow:
    """一个取点上的三种激活值（含两式之间的差）."""

    point: float
    gelu: float
    gelu_new: float
    gap: float

    def line(self) -> str:
        """一行可读的读数."""
        return (
            f"x={self.point:>5.1f} | gelu {self.gelu:+.6f} | gelu_new {self.gelu_new:+.6f} | "
            f"差 {self.gap:.3e}"
        )

    def to_dict(self) -> dict[str, object]:
        """摊平成字段."""
        return {
            "point": self.point,
            "gelu": self.gelu,
            "gelu_new": self.gelu_new,
            "gap": self.gap,
        }


def activation_study(
    points: tuple[float, ...] = ACTIVATION_POINTS,
) -> tuple[ActivationRow, ...]:
    """``gelu`` 与 ``gelu_new`` 逐点对照——**这一课说的"两个默认值不一样"就是这个数**.

    `x = 0` 那一行是唯一允许被手算断言的一行：两式的**函数值同为 0**
    （`0.5·0·(…) = 0`），而**导数同为 0.5**（对称性保证，day079 第 4.3 节量过同一个数）。
    """
    rows: list[ActivationRow] = []
    for point in points:
        exact = gelu_exact(point)
        approximate = gelu_new(point)
        rows.append(
            ActivationRow(
                point=point,
                gelu=exact,
                gelu_new=approximate,
                gap=abs(exact - approximate),
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class LayerNormRow:
    """一个 eps 的读数：实测方差、理论方差、以及两者之间的最大差."""

    name: str
    epsilon: float
    measured_variance: float
    theory_variance: float
    max_gap: float

    @property
    def distance_to_one(self) -> float:
        """"方差离 1 多远"——**它不是误差**，而是 eps 决定的一个定义量（day079 第 3.2 节）."""
        return 1.0 - self.measured_variance

    def line(self) -> str:
        """一行可读的读数."""
        return (
            f"{self.name:<7} eps={self.epsilon:.0e} | 实测方差 {self.measured_variance:.8f} | "
            f"理论 σ²/(σ²+eps) {self.theory_variance:.8f} | 差 {self.max_gap:.3e} | "
            f"离 1 {self.distance_to_one:.2e}"
        )

    def to_dict(self) -> dict[str, object]:
        """摊平成字段."""
        return {
            "name": self.name,
            "epsilon": self.epsilon,
            "measured_variance": self.measured_variance,
            "theory_variance": self.theory_variance,
            "max_gap": self.max_gap,
            "distance_to_one": self.distance_to_one,
        }


def _row_variance(matrix: Matrix) -> tuple[float, ...]:
    """逐行的**有偏**方差（除以 d，与 PyTorch 一致）."""
    result: list[float] = []
    for row in matrix:
        mean = math.fsum(row) / len(row)
        result.append(math.fsum((value - mean) ** 2 for value in row) / len(row))
    return tuple(result)


def layer_norm_study(
    inputs: Matrix | None = None,
    *,
    epsilons: tuple[float, ...] = (LN_EPS_GPT2, LN_EPS_BERT, LN_EPS_T5),
) -> tuple[LayerNormRow, ...]:
    """三个 eps 默认值各自让"方差离 1"多远——**同时对账 day079 的那条公式**.

    input 取 day079 样本里那四个数（`(1, 2, 3, 4)` 的推广），
    于是 `σ²` 手算得出来，`σ²/(σ²+eps)` 也手算得出来。
    """
    source = (
        (
            (1.0, 2.0, 3.0, 4.0),
            (2.0, 4.0, 6.0, 8.0),
        )
        if inputs is None
        else inputs
    )
    checked = validate_matrix(source, name="inputs")
    variances = _row_variance(checked)
    names = {LN_EPS_GPT2: "gpt2", LN_EPS_BERT: "bert", LN_EPS_T5: "t5"}
    rows: list[LayerNormRow] = []
    for epsilon in epsilons:
        normalized, _cache = layer_norm(checked, epsilon=epsilon)
        measured = math.fsum(_row_variance(normalized)) / len(normalized)
        theory = math.fsum(sigma / (sigma + epsilon) for sigma in variances) / len(variances)
        rows.append(
            LayerNormRow(
                name=names.get(epsilon, "custom"),
                epsilon=epsilon,
                measured_variance=measured,
                theory_variance=theory,
                max_gap=abs(measured - theory),
            )
        )
    return tuple(rows)


def study_lines() -> tuple[str, ...]:
    """把五张表渲染成文本（演示脚本直接打印它）."""
    lines: list[str] = ["== 1. 四个 warper 在同一个分布上（logits = (2, 1, 0.5, 0, -1, -2)）"]
    lines.extend(row.line() for row in filter_study())
    lines.append("== 2. top-k：保留个数是常数")
    lines.extend(row.line() for row in top_k_study())
    lines.append("== 3. top-p：保留个数由数据决定（核永不为空）")
    lines.extend(row.line() for row in nucleus_study())
    lines.append("== 4. 四种策略在同一个玩具模型上（同一个种子 seed=7）")
    lines.extend(row.line() for row in sampling_study())
    lines.append("== 5. 贪心 vs beam（长度惩罚改变的是选哪条前缀）")
    lines.extend(row.line() for row in beam_study())
    lines.append("== 6. gelu 与 gelu_new（两个模型的默认激活）")
    lines.extend(row.line() for row in activation_study())
    lines.append("== 7. LayerNorm 的三个 eps 默认值（同一个算子、三个默认值）")
    lines.extend(row.line() for row in layer_norm_study())
    return tuple(lines)


#: 默认的（也是演示脚本用的）形状。
DEFAULT_HIDDEN = 6
DEFAULT_TOKENS = 4


@dataclass(frozen=True)
class BlockRow:
    """一个模型的块的一行读数：摆放、激活、输出范数比."""

    name: str
    placement: str
    activation: str
    gain: float

    def line(self) -> str:
        """一行可读的读数."""
        return (
            f"{self.name} | 摆放 {self.placement} | 激活 {self.activation} | "
            f"增益（‖y‖/‖x‖）{self.gain:.6f}"
        )

    def to_dict(self) -> dict[str, object]:
        """摊平成字段."""
        return {
            "name": self.name,
            "placement": self.placement,
            "activation": self.activation,
            "gain": self.gain,
        }


def block_study(
    names: tuple[str, ...] = ("gpt2", "bert"),
    *,
    hidden: int = DEFAULT_HIDDEN,
    tokens: int = DEFAULT_TOKENS,
) -> tuple[BlockRow, ...]:
    """两个块在**同一份参数**下的输出增益（读数必须是同口径的）."""
    from smart_research_agent.hf_source.blocks import block_of, source_shape_of
    from smart_research_agent.hf_source.verify import default_inputs

    rows: list[BlockRow] = []
    for name in names:
        block_params, attention_params, _shape = make_hf_block(name, hidden=hidden, tokens=tokens)
        shape = source_shape_of(name, hidden=hidden, tokens=tokens)
        inputs = default_inputs(tokens, hidden)
        forward = block_of(name, block_params, attention_params, inputs, shape)
        before = math.sqrt(math.fsum(value * value for row in forward.inputs for value in row))
        after = math.sqrt(math.fsum(value * value for row in forward.output for value in row))
        rows.append(
            BlockRow(
                name=name,
                placement=forward.placement,
                activation=forward.activation,
                gain=after / before if before else 0.0,
            )
        )
    return tuple(rows)


__all__ = [
    "ACTIVATION_POINTS",
    "BEAM_CASES",
    "DEFAULT_HIDDEN",
    "DEFAULT_TOKENS",
    "FILTER_CASES",
    "FILTER_LOGITS",
    "SAMPLING_CASES",
    "TOY_SHARPNESS",
    "TOY_VOCAB",
    "ActivationRow",
    "BeamRow",
    "BlockRow",
    "FilterRow",
    "LayerNormRow",
    "NucleusRow",
    "SamplingRow",
    "TopKRow",
    "activation_study",
    "beam_study",
    "block_study",
    "filter_study",
    "layer_norm_study",
    "nucleus_study",
    "sampling_study",
    "study_lines",
    "top_k_study",
    "toy_logits",
]
