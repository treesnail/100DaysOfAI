"""``arch_variants`` 的类型与口径表：**三个变体的差别只有三处**（day082 / M7-D7）.

昨日（day081）交出的是“怎么把一条链变好”；今天回答的是**同一个块该怎么接**。
三篇论文（BERT 2019 / GPT 2018-2020 / T5 2020）在结构上只差三件事：

```text
                 ┌── 自注意力的掩码 ──┬── 有没有第二路（交叉注意力） ── 训练目标 ──┐
encoder_only     │ 全开（双向）        │ 无                            │ MLM（掩码语言建模）
decoder_only     │ 因果（只看过去）     │ 无                            │ CLM（自回归下一个词）
encoder_decoder  │ 编码器全开 + 解码器因果│ 有（Q 来自解码器、K/V 来自编码器）│ 去噪（文本到文本）
                 └────────────────────┴───────────────────────────────┴────────────┘
```

## 一、这个包**不写**一条新的前向公式

三个变体的每一层都是 day079/day080 已经验过的东西：

```text
encoder_only      self_attention(mask=全开)  →  encoder_block        （day079 的两种部件）
decoder_only      self_attention(mask=因果)  →  encoder_block        （**同一个块，只换掩码**）
encoder_decoder   上一条 + decoder_block（自注意力因果 + 交叉注意力 + 前馈）
```

因此本包**没有一行注意力算术**，它做的是四件“接线”的事：

```text
① 造掩码     三种（全开 / 因果 / 填充）以及它们的组合（masks.py）
② 接线       把掩码传给正确的流、把两路接起来、把参数按变体造齐（stacks.py）
③ 量泄漏     扰动第 j 个 token，看第 i 个输出动不动 —— 掩码的**实测**证据（probe.py）
④ 数参数     每一个变体的参数量、子层数、块数（census）
```

## 二、今天最值钱的一句话：**因果性不是读代码读出来的**

“这个模型是不是因果的”通常靠看代码（有没有 `causal=True`）来确认，而本课换一条路：

```text
扰动输入的第 j 个 token（乘 1.5），重新前向，看输出第 i 行变了多少
  被掩码挡掉的位置   Δ **恰好是 0.0**（不是 1e-16 —— 被掩码的打分**从来没有被读过**）
  允许的位置        Δ 明显大于 0
```

于是“因果”变成一张**可以被打印出来的 0/非 0 表**，而它有两条判据：

```text
逐位判据   被挡掉的位置必须**逐位相等**（`== 0.0`）——这一条不留容差
反证判据   GPT 用全开掩码时，同一张表上出现非零 ⇒ 这张表真的有分辨力
```

## 三、六条性质

```text
mask_zeroes_are_exact              被掩码的位置权重恰好 0.0（且不计入 softmax 分母）
decoder_reads_only_past            解码器第 i 行只依赖 j <= i（扰动实测为**逐位** 0）
encoder_reads_every_position       编码器每一行都依赖**所有**位置
cross_spans_all_sources            交叉注意力的每一行都覆盖全部源位置（长方形，无掩码）
stack_preserves_shape              变体保形（同一条流上进出一致）
deterministic                      同输入同参数 ⇒ 输出逐位相同
```

第 2、3、4 条构成一组对照：**同一个解码器里，自注意力因果、交叉注意力不因果**——
这正是 day079 “交叉注意力不能加因果掩码”那句拒绝的正面读法。

## 四、记录口径

```text
VariantShape         tokens / hidden / ffn / layers / sources（源序列长度）
BlockStackParameters blocks + attentions              （编码器式块链，三个变体都有）
DecoderStackParameters attentions + crosses + ffns + γ/β（只有 encoder_decoder 有）
VariantParameters    上面三者 + variant 名字（**变体名与部件的搭配在这里被校验**）
VariantForward       输出、两条流的输出、每一层的注意力权重、用的掩码
VariantCensus        块数 / 子层数 / 参数量（**逐个数出来**，另有一条解析式对照）
```
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from smart_research_agent.arch_variants.errors import (
    AssemblyError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.encoder_decoder.types import FFNWeights
from smart_research_agent.math_foundations.types import (
    Matrix,
    Vector,
    matrix_shape,
    validate_matrix,
)
from smart_research_agent.transformer_core.types import AttentionParams

# ---------------------------------------------------------------------- 三个变体

#: 只有编码器（`BERT <https://arxiv.org/abs/1810.04805>`_ 一类）.
VARIANT_ENCODER_ONLY = "encoder_only"
#: 只有解码器（`GPT <https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf>`_ 一类）.
VARIANT_DECODER_ONLY = "decoder_only"
#: 编码器 + 解码器（`T5 <https://arxiv.org/abs/1910.10683>`_ 一类）.
VARIANT_ENCODER_DECODER = "encoder_decoder"

VARIANTS: tuple[str, ...] = (
    VARIANT_ENCODER_ONLY,
    VARIANT_DECODER_ONLY,
    VARIANT_ENCODER_DECODER,
)

VARIANT_DESCRIPTIONS: dict[str, str] = {
    VARIANT_ENCODER_ONLY: "只有编码器：自注意力**全开**，每个位置都看得到所有位置（BERT 一类）",
    VARIANT_DECODER_ONLY: "只有解码器：自注意力**因果**，第 i 个位置只看得到 j <= i（GPT 一类）",
    VARIANT_ENCODER_DECODER: "编码器 + 解码器：**同一个解码器里因果与不因果各占一层**（T5 一类）",
}

#: 每个变体最常被点名的那篇论文里的模型（**只作为对照，本包的实现与它同构、不同规模**）.
VARIANT_EXAMPLES: dict[str, str] = {
    VARIANT_ENCODER_ONLY: "BERT",
    VARIANT_DECODER_ONLY: "GPT",
    VARIANT_ENCODER_DECODER: "T5",
}

#: 三种训练目标的名字与一句话（本包**不实现**训练目标，但它解释“为什么掩码这么选”）.
VARIANT_OBJECTIVES: dict[str, str] = {
    VARIANT_ENCODER_ONLY: "掩码语言建模（MLM）：挖掉若干位置、用左右两边一起预测它——因此必须双向",
    VARIANT_DECODER_ONLY: "自回归语言建模（CLM）：预测下一个 token——因此必须因果，否则答案就在输入里",
    VARIANT_ENCODER_DECODER: "去噪（文本到文本）：源序列被破坏、目标序列完整生成——源侧双向、目标侧因果",
}

#: 每个变体在两条流上各自用哪种掩码（**这张表要被测试逐键检查**）.
VARIANT_MASKS: dict[str, tuple[str, ...]] = {
    VARIANT_ENCODER_ONLY: ("full",),
    VARIANT_DECODER_ONLY: ("causal",),
    VARIANT_ENCODER_DECODER: ("full", "causal"),
}

# ---------------------------------------------------------------------- 六条性质

PROPERTY_MASK_ZEROES_ARE_EXACT = "mask_zeroes_are_exact"
PROPERTY_DECODER_READS_ONLY_PAST = "decoder_reads_only_past"
PROPERTY_ENCODER_READS_EVERY_POSITION = "encoder_reads_every_position"
PROPERTY_CROSS_SPANS_ALL_SOURCES = "cross_spans_all_sources"
PROPERTY_STACK_PRESERVES_SHAPE = "stack_preserves_shape"
PROPERTY_DETERMINISTIC = "deterministic"

ARCH_PROPERTIES: tuple[str, ...] = (
    PROPERTY_MASK_ZEROES_ARE_EXACT,
    PROPERTY_DECODER_READS_ONLY_PAST,
    PROPERTY_ENCODER_READS_EVERY_POSITION,
    PROPERTY_CROSS_SPANS_ALL_SOURCES,
    PROPERTY_STACK_PRESERVES_SHAPE,
    PROPERTY_DETERMINISTIC,
)

PROPERTY_DESCRIPTIONS: dict[str, str] = {
    PROPERTY_MASK_ZEROES_ARE_EXACT: "被掩码的位置权重恰好是 0.0，且不计入 softmax 的分母（逐位）",
    PROPERTY_DECODER_READS_ONLY_PAST: "解码器第 i 行对 j > i 的依赖**逐位为 0**（扰动实测）",
    PROPERTY_ENCODER_READS_EVERY_POSITION: "编码器每一行对**所有**位置都有非零依赖（扰动实测）",
    PROPERTY_CROSS_SPANS_ALL_SOURCES: "交叉注意力每一行都覆盖全部源位置（**长方形、无掩码**）",
    PROPERTY_STACK_PRESERVES_SHAPE: "变体把 (n, d) 映射成 (n, d)：不保形就没法堆叠",
    PROPERTY_DETERMINISTIC: "同一份参数与输入跑两次逐位相同（没有随机数）",
}

ARCH_NOTES: tuple[str, ...] = (
    "本包不写新的注意力算术：三个变体都由 day079/day080 的 self_attention 与 encoder_block / "
    "decoder_block 拼出来。",
    "因果与不因果**不是两条代码路径**：它们是同一个算子的两张掩码——decoder_only 用的就是 "
    "encoder_block（只换掩码）。",
    "实测依赖表是这一课的判据：被挡掉的格子必须逐位为 0.0，而它有一条反证（用错掩码时非零）。",
)

# ---------------------------------------------------------------------- 默认规模

#: 默认序列长度（与 day073~081 的样本同值，便于跨天对照）.
DEFAULT_TOKENS = 4
#: 默认隐藏维（同上）.
DEFAULT_HIDDEN = 6
#: 默认前馈倍数（原论文的取值）.
DEFAULT_FFN_RATIO = 4
#: 默认层数（day080 的取值）.
DEFAULT_LAYERS = 3
#: 默认源序列长度（T5 的样本用 5，与目标序列不同——**两路长度不同才看得出长方形**）.
DEFAULT_SOURCES = 5
#: 参数初始化的幅度（与 day075/079 的 ``DEFAULT_INIT_SCALE`` 同值）.
DEFAULT_INIT_SCALE = 0.25


def validate_variant(name: Any) -> str:
    """校验变体名（**不做默认值兜底**：拼错的名字必须当场失败）."""
    if not isinstance(name, str):
        raise ParameterError(f"变体名必须是字符串，收到 {type(name).__name__}。")
    if name not in VARIANTS:
        raise ParameterError(
            f"未知的变体名 {name!r}：可选 {', '.join(VARIANTS)}——"
            "本包不接受'拼错就当成默认变体'，否则'到底跑的是哪一个'只能靠读代码才知道。"
        )
    return name


def _checked_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParameterError(f"{name} 必须是整数，收到 {value!r}。")
    if value < 1:
        raise ParameterError(f"{name} 必须 >= 1，收到 {value}。")
    return value


@dataclass(frozen=True)
class VariantShape:
    """一次变体前向的形状：``(tokens, hidden, ffn, layers)`` 与源序列长度.

    ``tokens`` 是**目标那一流**（或唯一那条流）的长度，``sources`` 是源序列长度。
    两个长度分开写出来是有代价的：它让 ``encoder_decoder`` 的权重矩阵**不是方阵**——
    而这正是 day079 “交叉注意力不能加因果掩码”那条拒绝的几何来源。
    """

    tokens: int = DEFAULT_TOKENS
    hidden: int = DEFAULT_HIDDEN
    ffn: int = DEFAULT_FFN_RATIO * DEFAULT_HIDDEN
    layers: int = DEFAULT_LAYERS
    sources: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tokens", _checked_positive_int(self.tokens, name="tokens"))
        object.__setattr__(self, "hidden", _checked_positive_int(self.hidden, name="hidden"))
        object.__setattr__(self, "ffn", _checked_positive_int(self.ffn, name="ffn"))
        object.__setattr__(self, "layers", _checked_positive_int(self.layers, name="layers"))
        if self.sources is not None:
            object.__setattr__(
                self, "sources", _checked_positive_int(self.sources, name="sources")
            )

    @property
    def source_length(self) -> int:
        """源序列长度（``sources`` 为 ``None`` 时与 ``tokens`` 相同）."""
        return self.tokens if self.sources is None else self.sources

    @property
    def hidden_dimension(self) -> int:
        """隐藏维（与 day079 的 ``BlockShape.hidden`` 同义）."""
        return self.hidden

    def summary_line(self) -> str:
        """一行说明：``tokens=4 sources=5 hidden=6 ffn=24 layers=3``."""
        return (
            f"tokens={self.tokens} sources={self.source_length} "
            f"hidden={self.hidden} ffn={self.ffn} layers={self.layers}"
        )


@dataclass(frozen=True)
class BlockStackParameters:
    """一条**编码器式块链**的参数：每层一个块 + 一层自注意力.

    三个变体都有这一份——``decoder_only`` 用的也是它（只是掩码因果），
    这是本课“因果性不是一条新代码路径”这句话在类型上的样子。
    """

    blocks: tuple[Any, ...]
    attentions: tuple[AttentionParams, ...]

    def __post_init__(self) -> None:
        blocks = tuple(self.blocks)
        attentions = tuple(self.attentions)
        if not blocks:
            raise ParameterError("一条块链至少要有一层。")
        if len(blocks) != len(attentions):
            raise ShapeError(
                f"块的数量 {len(blocks)} 与注意力的数量 {len(attentions)} 不一致："
                "每一层各有一个块与一层注意力。"
            )
        for item in attentions:
            if not isinstance(item, AttentionParams):
                raise ParameterError(
                    f"attentions 里必须是 AttentionParams，收到 {type(item).__name__}。"
                )
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "attentions", attentions)

    @property
    def layers(self) -> int:
        """层数."""
        return len(self.blocks)

    def parameter_count(self) -> int:
        """参数量（逐个数出来：块的八块 + 注意力的四块）."""
        total = 0
        for block in self.blocks:
            total += block.parameter_count
        for attention in self.attentions:
            total += attention.parameter_count()
        return total


@dataclass(frozen=True)
class DecoderStackParameters:
    """**解码器式块链**的参数：每层一层因果自注意力 + 交叉注意力 + 前馈 + 三个 LN.

    与 :class:`BlockStackParameters` 的差别只有一处，而它正是 ``encoder_decoder``
    与另两个变体的分水岭：**多一层作用在“另一路”上的注意力**。

    参数刻意分成四份（与 day079 的 ``decoder_block`` 签名逐字对应）：
    ``attentions``（自己的流）、``crosses``（两路之间）、``ffns``（自己的流）、
    ``gammas``/``betas``（三个 LN 各一组）——
    “哪一个参数属于哪一路”是这一层最容易搞混的地方。
    """

    attentions: tuple[AttentionParams, ...]
    crosses: tuple[Any, ...]
    ffns: tuple[FFNWeights, ...]
    gammas: tuple[tuple[Vector, Vector, Vector], ...]
    betas: tuple[tuple[Vector, Vector, Vector], ...]

    def __post_init__(self) -> None:
        lengths = {
            "attentions": len(self.attentions),
            "crosses": len(self.crosses),
            "ffns": len(self.ffns),
            "gammas": len(self.gammas),
            "betas": len(self.betas),
        }
        if not self.attentions:
            raise ParameterError("一条解码器链至少要有一层。")
        if len(set(lengths.values())) != 1:
            raise ShapeError(
                f"解码器链的五份参数层数不一致：{lengths}——每一层都要有自注意力、"
                "交叉注意力、前馈与三个 LN 的 γ/β。"
            )
        for group, expected in (
            (self.gammas, 3),
            (self.betas, 3),
        ):
            for index, item in enumerate(group):
                if len(item) != expected:
                    raise AssemblyError(
                        f"第 {index} 层的 γ/β 有 {len(item)} 组，解码器块有三个 LN "
                        f"（自注意力前、交叉注意力前、前馈前）——因此必须是 {expected} 组。"
                    )

    @property
    def layers(self) -> int:
        """层数."""
        return len(self.attentions)

    def parameter_count(self) -> int:
        """参数量（自注意力 + 交叉注意力 + 前馈 + 六个 γ/β 向量）."""
        total = 0
        for attention in self.attentions:
            total += attention.parameter_count()
        for cross in self.crosses:
            total += cross.parameter_count
        for ffn in self.ffns:
            total += ffn.parameter_count
        for group in (*self.gammas, *self.betas):
            for vector in group:
                total += len(vector)
        return total


@dataclass(frozen=True)
class VariantParameters:
    """一个变体的全部参数：**一份块链 + 可选的解码器链**.

    变体名与部件的搭配在 ``__post_init__`` 里被校验（三条规则）：

    ```text
    decoder 非空            ⇒ variant 必须是 encoder_decoder（只有它有两路）
    variant == encoder_decoder ⇒ decoder 必须非空（只有编码器就不叫 encoder_decoder 了）
    decoder 的层数            ⇒ 与 shape.layers 一致（层数是**同一个**旋钮）
    ```
    """

    shape: VariantShape
    variant: str
    blocks: BlockStackParameters
    decoder: DecoderStackParameters | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.shape, VariantShape):
            raise ParameterError(
                f"shape 必须是 VariantShape，收到 {type(self.shape).__name__}。"
            )
        variant = validate_variant(self.variant)
        if not isinstance(self.blocks, BlockStackParameters):
            raise ParameterError(
                f"blocks 必须是 BlockStackParameters，收到 {type(self.blocks).__name__}。"
            )
        if variant == VARIANT_ENCODER_DECODER:
            if self.decoder is None:
                raise AssemblyError(
                    "encoder_decoder 变体必须有解码器链：只有一条块链时它其实是 "
                    "encoder_only 或 decoder_only——而这两个名字的失败方式完全不同"
                    "（一个漏看未来、一个看不到全部上下文）。"
                )
        elif self.decoder is not None:
            raise AssemblyError(
                f"{variant} 变体不该带解码器链：它只有一条流，而交叉注意力要求**第二路**"
                "的 K/V——两路里少了一路时，那个乘法根本没有定义。"
            )
        if self.decoder is not None and self.decoder.layers != self.shape.layers:
            raise ShapeError(
                f"解码器链的层数 {self.decoder.layers} 与 shape.layers "
                f"{self.shape.layers} 不一致：层数是同一个旋钮，两处给不同的值只会"
                "让报告里出现两个都'对'的层数。"
            )
        if self.blocks.layers != self.shape.layers:
            raise ShapeError(
                f"块链的层数 {self.blocks.layers} 与 shape.layers "
                f"{self.shape.layers} 不一致。"
            )
        object.__setattr__(self, "variant", variant)

    @property
    def layers(self) -> int:
        """层数."""
        return self.shape.layers

    def parameter_count(self) -> int:
        """参数量（逐个数出来；与解析式的对照在 ``census_of`` 里）."""
        total = self.blocks.parameter_count()
        if self.decoder is not None:
            total += self.decoder.parameter_count()
        return total

    def flatten(self) -> tuple[float, ...]:
        """按**固定顺序**压成一串浮点（顺序见 :data:`PARAMETER_BLOCK_ORDER`）.

        顺序被写进一个常量而不是散落在压缩代码里：``census`` 的数数、
        逐位比较与“换一个变体”的对照都靠同一个顺序，
        而两处顺序一旦分开，误差不会出现在数数上——**只会出现在某一条断言里**。
        """
        values: list[float] = []
        for block in self.blocks.blocks:
            flat, _shapes = block.flatten()
            values.extend(flat)
        for attention in self.blocks.attentions:
            for matrix in attention.matrices():
                for row in matrix:
                    values.extend(row)
        if self.decoder is not None:
            for attention in self.decoder.attentions:
                for matrix in (
                    attention.w_query,
                    attention.w_key,
                    attention.w_value,
                    attention.w_output,
                ):
                    for row in matrix:
                        values.extend(row)
            for cross in self.decoder.crosses:
                for matrix in cross.matrices():
                    for row in matrix:
                        values.extend(row)
            for ffn in self.decoder.ffns:
                for matrix in (ffn.w_in, ffn.w_out):
                    for row in matrix:
                        values.extend(row)
                values.extend(ffn.b_in)
                values.extend(ffn.b_out)
            for group in (*self.decoder.gammas, *self.decoder.betas):
                for vector in group:
                    values.extend(vector)
        return tuple(values)


#: 压平参数时的块顺序（**写下来是为了让两个数数的地方不可能走散**）.
PARAMETER_BLOCK_ORDER: tuple[str, ...] = (
    "blocks",
    "self_attentions",
    "decoder_self_attentions",
    "cross_attentions",
    "decoder_ffns",
    "decoder_norms",
)


@dataclass(frozen=True)
class VariantCensus:
    """一次变体的“家底”：块数、子层数、参数量（**逐个数出来的**）.

    ``analytic_parameters`` 是用公式算出来的那一份：

    ```text
    块（前馈 + 两个 LN）      2·d·f + f + 5·d
    自注意力（四个 d×d 投影）   4·d²
    交叉注意力                4·d²
    解码器前馈 + 三个 LN       2·d·f + f + d + 6·d
    ```

    两个数必须相等——这条断言与 day080 的“解析参数量 == 逐块数出来的参数量”同源：
    **它挡住的是“忘了数某一类参数”这种不会报错的错误**（参数少了一块，前向照样跑）。
    """

    variant: str
    layers: int
    block_layers: int
    decoder_layers: int
    self_attention_layers: int
    cross_attention_layers: int
    sub_layers: int
    parameters: int
    analytic_parameters: int
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant", validate_variant(self.variant))
        if self.parameters <= 0:
            raise NumericError(f"参数量必须为正，收到 {self.parameters}。")
        if self.sub_layers < self.layers:
            raise NumericError(
                f"子层数 {self.sub_layers} 少于层数 {self.layers}：每个块至少有两个子层。"
            )
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def matches_analytic(self) -> bool:
        """两条路径数出来的参数量是否一致."""
        return self.parameters == self.analytic_parameters

    @property
    def layers_per_block(self) -> float:
        """平均每个块有几个子层（``encoder_only`` 是 2，解码器是 3）."""
        return self.sub_layers / self.layers

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "variant": self.variant,
            "layers": self.layers,
            "block_layers": self.block_layers,
            "decoder_layers": self.decoder_layers,
            "self_attention_layers": self.self_attention_layers,
            "cross_attention_layers": self.cross_attention_layers,
            "sub_layers": self.sub_layers,
            "parameters": self.parameters,
            "analytic_parameters": self.analytic_parameters,
            "matches_analytic": self.matches_analytic,
        }

    def summary_line(self) -> str:
        """一行说明（实验表里用的就是这一行）."""
        return (
            f"{self.variant:<16} | 块 {self.block_layers} + 解码器 {self.decoder_layers} | "
            f"自注意力 {self.self_attention_layers} · 交叉 {self.cross_attention_layers} | "
            f"子层 {self.sub_layers} | 参数 {self.parameters}"
            f"（解析 {self.analytic_parameters}）"
        )


@dataclass(frozen=True)
class VariantForward:
    """一次变体前向的账：输入、输出、每层的注意力权重、用过的掩码与三个摆放旋钮.

    权重留在这里而不是丢掉，是 day083 的伏笔：**可解释性要的正是这张表**。
    三份权重各管一条路，而它们的形状说明了三条路的差别：

    ```text
    block_weights          编码器（或唯一那条）流的自注意力权重  (n, n)
    decoder_self_weights   解码器那一侧的自注意力权重            (n_tgt, n_tgt)  只有 enc/dec 有
    cross_weights          交叉注意力权重                        (n_tgt, n_src)  **长方形**
    ```

    ``inputs`` / ``source`` / ``placement`` / ``use_residual`` / ``activation``
    五个字段是为**反向**留的：本包的反向与 day079/080 同一条纪律——
    **重放一遍前向**（而不是把每一层的中间账都塞进记录里），
    而重放要求“输入与旋钮”都还在手上。少留一个字段的后果很具体：
    反向会在一份**不同配置**上前进，而它的形状完全合法。
    """

    variant: str
    shape: VariantShape
    inputs: Matrix
    output: Matrix
    mask: tuple[tuple[bool, ...], ...]
    placement: str = "pre"
    use_residual: bool = True
    activation: str = "relu"
    source: Matrix | None = None
    source_mask: tuple[tuple[bool, ...], ...] | None = None
    encoder_output: Matrix | None = None
    block_weights: tuple[Matrix, ...] = ()
    decoder_self_weights: tuple[Matrix, ...] = ()
    cross_weights: tuple[Matrix, ...] = ()
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant", validate_variant(self.variant))
        checked_inputs = validate_matrix(self.inputs, name="inputs")
        checked = validate_matrix(self.output, name="output")
        object.__setattr__(self, "inputs", checked_inputs)
        object.__setattr__(self, "output", checked)
        expected = (self.shape.tokens, self.shape.hidden)
        if matrix_shape(checked_inputs) != expected:
            raise ShapeError(
                f"输入形状 {matrix_shape(checked_inputs)} 与 shape 声明的 {expected} 不一致。"
            )
        if matrix_shape(checked) != expected:
            raise ShapeError(
                f"输出形状 {matrix_shape(checked)} 与 shape 声明的 {expected} 不一致。"
            )
        if len(self.block_weights) != self.shape.layers:
            raise ShapeError(
                f"记录了 {len(self.block_weights)} 层的自注意力权重，shape 声明 "
                f"{self.shape.layers} 层。"
            )
        if self.source is not None:
            checked_source = validate_matrix(self.source, name="source")
            object.__setattr__(self, "source", checked_source)
            if matrix_shape(checked_source) != (
                self.shape.source_length,
                self.shape.hidden,
            ):
                raise ShapeError(
                    f"源序列形状 {matrix_shape(checked_source)} 与 shape 声明的 "
                    f"({self.shape.source_length}, {self.shape.hidden}) 不一致。"
                )
        if self.encoder_output is not None:
            encoder_shape = matrix_shape(self.encoder_output)
            if encoder_shape != (self.shape.source_length, self.shape.hidden):
                raise ShapeError(
                    f"编码器输出形状 {encoder_shape} 与源长度不一致：应当是 "
                    f"({self.shape.source_length}, {self.shape.hidden})。"
                )
            if len(self.cross_weights) != self.shape.layers:
                raise ShapeError(
                    f"encoder_decoder 变体每一层都有一张交叉注意力权重表，收到 "
                    f"{len(self.cross_weights)} 张。"
                )
            if len(self.decoder_self_weights) != self.shape.layers:
                raise ShapeError(
                    f"encoder_decoder 变体的解码器每一层都有一张自注意力权重表，收到 "
                    f"{len(self.decoder_self_weights)} 张。"
                )
        elif self.cross_weights or self.decoder_self_weights:
            raise NumericError(
                "只有一条流时不该有交叉注意力（或解码器自注意力）权重："
                "那两张表都要**第二路**才有定义。"
            )
        object.__setattr__(self, "block_weights", tuple(self.block_weights))
        object.__setattr__(self, "decoder_self_weights", tuple(self.decoder_self_weights))
        object.__setattr__(self, "cross_weights", tuple(self.cross_weights))
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def layers(self) -> int:
        """层数."""
        return self.shape.layers

    @property
    def has_cross(self) -> bool:
        """这一份前向里有没有交叉注意力（即是不是 encoder_decoder）."""
        return bool(self.cross_weights)

    def weights_of(self, layer: int) -> Matrix:
        """第 ``layer`` 层的**编码器（或唯一那条）流**的自注意力权重."""
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise ParameterError(f"层号必须是整数，收到 {layer!r}。")
        if not 0 <= layer < self.layers:
            raise ParameterError(f"层号 {layer} 越界：可选 0..{self.layers - 1}。")
        return self.block_weights[layer]

    def decoder_weights_of(self, layer: int) -> Matrix:
        """第 ``layer`` 层的**解码器自注意力**权重（没有解码器时当场拒绝）."""
        if not self.decoder_self_weights:
            raise AssemblyError(
                f"{self.variant} 变体没有解码器那一流：它的自注意力只有一条——"
                "解码器自注意力要的是第 2 条流上的因果自注意力。"
            )
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise ParameterError(f"层号必须是整数，收到 {layer!r}。")
        if not 0 <= layer < len(self.decoder_self_weights):
            raise ParameterError(
                f"层号 {layer} 越界：可选 0..{len(self.decoder_self_weights) - 1}。"
            )
        return self.decoder_self_weights[layer]

    def cross_weights_of(self, layer: int) -> Matrix:
        """第 ``layer`` 层的交叉注意力权重（没有交叉注意力时当场拒绝）."""
        if not self.cross_weights:
            raise AssemblyError(
                f"{self.variant} 变体没有交叉注意力：它的 K/V 只来自自己的那一条流——"
                "而交叉注意力要的是**第二路**。"
            )
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise ParameterError(f"层号必须是整数，收到 {layer!r}。")
        if not 0 <= layer < len(self.cross_weights):
            raise ParameterError(
                f"层号 {layer} 越界：可选 0..{len(self.cross_weights) - 1}。"
            )
        return self.cross_weights[layer]

    def summary_line(self) -> str:
        """一行说明."""
        cross = "有交叉注意力" if self.has_cross else "无交叉注意力"
        return (
            f"{self.variant:<16} | {self.shape.summary_line()} | 输出 "
            f"{matrix_shape(self.output)} | {cross}"
        )


# ---------------------------------------------------------------------- 读数工具


def frobenius(matrix: Matrix) -> float:
    """Frobenius 范数 ``√Σa²``（用 ``math.fsum``：跨表比较的两个数必须同口径）."""
    checked = validate_matrix(matrix, name="matrix")
    return math.sqrt(math.fsum(value * value for row in checked for value in row))


def max_absolute(matrix: Matrix) -> float:
    """最大绝对值（读数表里那个“这一层的量级”）."""
    checked = validate_matrix(matrix, name="matrix")
    return max((abs(value) for row in checked for value in row), default=0.0)


def relative_matrix_error(approximate: Matrix, reference: Matrix) -> float:
    """两个矩阵的相对误差 ``max|a−b| / max(1, max|b|)``.

    分母带一个 ``max(1, ·)`` 是因为参考矩阵可能整体很小——
    纯相对误差在“两边都接近 0”时会放大到无意义的量级，
    而本课的读数里确实有那样的矩阵（被掩码抹平的那些）。
    """
    left = validate_matrix(approximate, name="approximate")
    right = validate_matrix(reference, name="reference")
    if matrix_shape(left) != matrix_shape(right):
        raise ShapeError(
            f"两个矩阵形状不同：{matrix_shape(left)} 与 {matrix_shape(right)}。"
        )
    gap = max(
        (abs(a - b) for left_row, right_row in zip(left, right, strict=True)
         for a, b in zip(left_row, right_row, strict=True)),
        default=0.0,
    )
    return gap / max(1.0, max_absolute(right))


def ones_vector(length: int) -> Vector:
    """全 1 向量（LayerNorm 的 γ 初始值）."""
    return tuple(1.0 for _ in range(_checked_positive_int(length, name="length")))


def zeros_vector(length: int) -> Vector:
    """全 0 向量（LayerNorm 的 β 初始值）."""
    return tuple(0.0 for _ in range(_checked_positive_int(length, name="length")))


__all__ = [
    "ARCH_NOTES",
    "ARCH_PROPERTIES",
    "DEFAULT_FFN_RATIO",
    "DEFAULT_HIDDEN",
    "DEFAULT_INIT_SCALE",
    "DEFAULT_LAYERS",
    "DEFAULT_SOURCES",
    "DEFAULT_TOKENS",
    "PARAMETER_BLOCK_ORDER",
    "PROPERTY_CROSS_SPANS_ALL_SOURCES",
    "PROPERTY_DECODER_READS_ONLY_PAST",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_DETERMINISTIC",
    "PROPERTY_ENCODER_READS_EVERY_POSITION",
    "PROPERTY_MASK_ZEROES_ARE_EXACT",
    "PROPERTY_STACK_PRESERVES_SHAPE",
    "VARIANTS",
    "VARIANT_DESCRIPTIONS",
    "VARIANT_DECODER_ONLY",
    "VARIANT_ENCODER_DECODER",
    "VARIANT_ENCODER_ONLY",
    "VARIANT_EXAMPLES",
    "VARIANT_MASKS",
    "VARIANT_OBJECTIVES",
    "BlockStackParameters",
    "DecoderStackParameters",
    "VariantCensus",
    "VariantForward",
    "VariantParameters",
    "VariantShape",
    "frobenius",
    "max_absolute",
    "ones_vector",
    "relative_matrix_error",
    "validate_variant",
    "zeros_vector",
]
