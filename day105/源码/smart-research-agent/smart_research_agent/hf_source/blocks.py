"""块：**同一个块，两种摆放，两个激活**（day085 / M7-D9）.

day079 交出了"一个块"的六个阶段与九项梯度；今天要读的是**HF 里那个块的接线**：

```text
GPT-2（pre-LN）    x = x + attn(ln_1(x))
                   x = x + mlp(ln_2(x))            ← 栈尾还有一次 ln_f
BERT（post-LN）    x = ln_1(x + attn(x))
                   x = ln_2(x + mlp(x))
```

两段代码只差 LN 的位置，而它们在源码里长得**很像**——
都写着 `residual = hidden_states; hidden_states = self.xxx(hidden_states); hidden_states = residual + hidden_states`，
差别只在 `layer_norm` 被乘在哪一步。这正是 day079 那句"pre/post 是同一条代码路径上
的一个位置差"在真实源码里的样子。

## 两个不可互换的激活

```text
BERT   hidden_act = "gelu"       0.5x(1 + erf(x/√2))                     精确式
GPT-2  activation_function = "gelu_new"
                                 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))  tanh 近似
```

`gelu_new` 是那条 erf 式的**多项式—双曲近似**：它在 ``x = 0`` 处与精确式
**函数值同为 `0`、导数同为 `0.5`**（对称性保证），而在两侧各有 O(1e-4) 量级的偏差。
本包把它写成一张表（:mod:`study`），而不是一句"两者差不多"。

## 与 day079 的接缝

本模块**不新造参数形状**：块的八块参数直接用 day079 的
:class:`~smart_research_agent.encoder_decoder.types.BlockParameters`，
前馈直接用它的 :class:`~smart_research_agent.encoder_decoder.types.FFNWeights`。
因此"`gpt2_block` 与 `encoder_block(placement="pre")` 落在同一个值上"这条对账
是一条**真的**对账，而不是两份格式不同的东西互相看一眼。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from smart_research_agent.encoder_decoder.depth import make_block_parameters
from smart_research_agent.encoder_decoder.layers import add_residual, layer_norm
from smart_research_agent.encoder_decoder.types import (
    ACTIVATION_GELU as DAY079_GELU,
)
from smart_research_agent.encoder_decoder.types import (
    ACTIVATION_RELU as DAY079_RELU,
)
from smart_research_agent.encoder_decoder.types import (
    BlockParameters,
    BlockShape,
    FFNWeights,
)
from smart_research_agent.hf_source.attention import HfAttentionForward, hf_attention
from smart_research_agent.hf_source.errors import ParameterError, ShapeError
from smart_research_agent.hf_source.types import (
    ACTIVATION_BERT,
    ACTIVATION_GPT2,
    ACTIVATION_RELU,
    ACTIVATIONS,
    NORM_POST,
    NORM_PRE,
    SourceShape,
    profile_of,
)
from smart_research_agent.math_foundations.types import Matrix, validate_matrix
from smart_research_agent.transformer_core.train import default_parameters
from smart_research_agent.transformer_core.types import AttentionParams

#: ``gelu_new`` 的三次项系数（这是那个近似式的**全部魔法**）.
GELU_NEW_CUBIC = 0.044715

#: ``gelu_new`` 里 ``tanh`` 的内系数 ``√(2/π)``.
GELU_NEW_INNER = math.sqrt(2.0 / math.pi)

#: 与 day079 对账时的容差（两份独立实现之间的"刻意近似"）.
BLOCK_TOLERANCE = 1e-12

#: 前馈的扩张比（两个模型的默认值都是 4）。
FFN_RATIO = 4


def gelu_exact(value: float) -> float:
    """精确 GELU：``0.5x(1 + erf(x/√2))``（BERT 的默认激活）."""
    return 0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))


def gelu_new(value: float) -> float:
    """``gelu_new``：``0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))``（GPT-2 的默认激活）.

    它不是"换个写法"：``x = 0`` 处两式的函数值同为 `0`、**导数同为 `0.5`**
    （后者由 `0.5·(1 + tanh(0)) = 0.5` 与精确式在 0 处的导数一致得到），
    而两侧各有可量出的偏差。本包把它量出来，因为"两个默认值不一样"这句话
    如果不带一个数，它就是一句不可反驳的话。
    """
    inner = GELU_NEW_INNER * (value + GELU_NEW_CUBIC * value**3)
    return 0.5 * value * (1.0 + math.tanh(inner))


def activation_of(name: str, value: float) -> float:
    """按名字算一个激活值（未知名字当场拒绝，绝不回退到 relu）."""
    if name == ACTIVATION_BERT:
        return gelu_exact(value)
    if name == ACTIVATION_GPT2:
        return gelu_new(value)
    if name == ACTIVATION_RELU:
        return value if value > 0.0 else 0.0
    raise ParameterError(
        f"未知的激活名 {name!r}：本包只认 {list(ACTIVATIONS)}。"
        "回退到 relu 的后果是——一份 GPT-2 的读数会被印成 BERT 的"
        "（day081 第 10.2 节的'漏传'在这里换了一个形式）。"
    )


def activate_rows(matrix: Matrix, activation: str) -> Matrix:
    """逐元素激活（因此**逐行**，与 day079 的 ``activate`` 是同一件事）."""
    checked = validate_matrix(matrix, name="matrix")
    return tuple(tuple(activation_of(activation, value) for value in row) for row in checked)


def gpt2_mlp(
    inputs: Matrix,
    weights: FFNWeights,
    *,
    activation: str = ACTIVATION_GPT2,
) -> Matrix:
    """GPT-2 的 MLP：``c_fc → act → c_proj``（HF 里就是两层 ``Linear`` 加一个激活）.

    形状约定与 day079 的 ``feed_forward`` **完全相同**（``d_ff = 4·hidden``、
    ``y = act(x·W_inᵀ + b_in)·W_outᵀ + b_out``），因此两者的差别只剩下激活函数
    与参数名字——这也正是"同一件事的两种写法"该有的样子。
    """
    checked = validate_matrix(inputs, name="inputs")
    hidden = tuple(
        tuple(
            math.fsum(value * weight for value, weight in zip(row, unit)) + offset
            for unit, offset in zip(weights.w_in, weights.b_in)
        )
        for row in checked
    )
    activated = activate_rows(hidden, activation)
    return tuple(
        tuple(
            math.fsum(value * weight for value, weight in zip(row, unit)) + offset
            for unit, offset in zip(weights.w_out, weights.b_out)
        )
        for row in activated
    )


@dataclass(frozen=True)
class BlockForward:
    """一次块前向的账（两种摆放共用这一个记录，因为**字段名必须一样**）."""

    inputs: Matrix
    first_normed: Matrix
    attention: HfAttentionForward
    after_attention: Matrix
    second_normed: Matrix
    mlp_output: Matrix
    output: Matrix
    placement: str
    activation: str

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段（**不含**张量本体）."""
        return {
            "tokens": len(self.inputs),
            "hidden": len(self.inputs[0]),
            "placement": self.placement,
            "activation": self.activation,
            "heads": self.attention.heads,
        }


def gpt2_block(
    params: BlockParameters,
    attention_params: AttentionParams,
    inputs: Matrix,
    shape: SourceShape,
    *,
    activation: str = ACTIVATION_GPT2,
    causal: bool = True,
) -> BlockForward:
    """GPT-2 的块（pre-LN，两个子层）.

    接线是 **先归一化再入子层**：``x = x + attn(ln_1(x)); x = x + mlp(ln_2(x))``。
    ``causal`` 默认 ``True``——GPT-2 是**解码器**，每一步只能看过去
    （day082 已经用扰动量过这件事，本包在性质里再量一次）。
    """
    profile = profile_of("gpt2")
    if profile.norm_placement != NORM_PRE:  # pragma: no cover - 画像被改时触发
        raise ParameterError("画像与实现不一致：gpt2_block 假定 GPT-2 是 pre-LN 摆放。")
    checked = validate_matrix(inputs, name="inputs")
    normed, _cache = layer_norm(
        checked, gamma=params.norm1_gamma, beta=params.norm1_beta, epsilon=profile.ln_eps
    )
    attention = hf_attention(attention_params, normed, shape, causal=causal)
    after_attention = add_residual(checked, attention.output)
    normed_two, _cache_two = layer_norm(
        after_attention, gamma=params.norm2_gamma, beta=params.norm2_beta, epsilon=profile.ln_eps
    )
    mlp = gpt2_mlp(normed_two, _ffn_weights(params), activation=activation)
    return BlockForward(
        inputs=checked,
        first_normed=normed,
        attention=attention,
        after_attention=after_attention,
        second_normed=normed_two,
        mlp_output=mlp,
        output=add_residual(after_attention, mlp),
        placement=NORM_PRE,
        activation=activation,
    )


def bert_layer(
    params: BlockParameters,
    attention_params: AttentionParams,
    inputs: Matrix,
    shape: SourceShape,
    *,
    activation: str = ACTIVATION_BERT,
    causal: bool = False,
) -> BlockForward:
    """BERT 的层（post-LN，两个子层）.

    接线是 **先入子层再归一化**：``x = ln_1(x + attn(x)); x = ln_2(x + mlp(x))``。
    ``causal`` 默认 ``False``——BERT 是**编码器**，双向看整条序列。

    它与 :func:`gpt2_block` 共用同一个 :class:`BlockForward` 记录，而这一点是刻意的：
    两份记录字段一致，才能把"两种摆放"放进同一张表里比。
    注意 `first_normed` / `second_normed` 在 post 摆放下的含义：
    它们**就是**残差之后的归一化结果（因此 `second_normed` 与 `after_attention` 同值）。
    """
    profile = profile_of("bert")
    if profile.norm_placement != NORM_POST:  # pragma: no cover - 画像被改时触发
        raise ParameterError("画像与实现不一致：bert_layer 假定 BERT 是 post-LN 摆放。")
    checked = validate_matrix(inputs, name="inputs")
    attention = hf_attention(attention_params, checked, shape, causal=causal)
    after_attention, _cache = layer_norm(
        add_residual(checked, attention.output),
        gamma=params.norm1_gamma,
        beta=params.norm1_beta,
        epsilon=profile.ln_eps,
    )
    mlp = gpt2_mlp(after_attention, _ffn_weights(params), activation=activation)
    output, _cache_two = layer_norm(
        add_residual(after_attention, mlp),
        gamma=params.norm2_gamma,
        beta=params.norm2_beta,
        epsilon=profile.ln_eps,
    )
    return BlockForward(
        inputs=checked,
        first_normed=after_attention,
        attention=attention,
        after_attention=after_attention,
        second_normed=after_attention,
        mlp_output=mlp,
        output=output,
        placement=NORM_POST,
        activation=activation,
    )


def _ffn_weights(params: BlockParameters) -> FFNWeights:
    """把 day079 块参数里的前馈那四块单独取出来（**只换容器，不换值**）."""
    return FFNWeights(
        w_in=params.ffn_w_in,
        b_in=params.ffn_b_in,
        w_out=params.ffn_w_out,
        b_out=params.ffn_b_out,
    )


def block_of(
    name: str,
    params: BlockParameters,
    attention_params: AttentionParams,
    inputs: Matrix,
    shape: SourceShape,
    *,
    activation: str | None = None,
    causal: bool | None = None,
) -> BlockForward:
    """按模型名分派到两个块之一（**画像驱动**：摆放与激活都从画像取默认值）."""
    profile = profile_of(name)
    resolved_activation = profile.activation if activation is None else activation
    if profile.norm_placement == NORM_PRE:
        resolved_causal = True if causal is None else causal
        return gpt2_block(
            params,
            attention_params,
            inputs,
            shape,
            activation=resolved_activation,
            causal=resolved_causal,
        )
    resolved_causal = False if causal is None else causal
    return bert_layer(
        params,
        attention_params,
        inputs,
        shape,
        activation=resolved_activation,
        causal=resolved_causal,
    )


def make_hf_block(
    name: str,
    *,
    hidden: int = 6,
    tokens: int = 4,
    seed: int = 7,
) -> tuple[BlockParameters, AttentionParams, BlockShape]:
    """造一份确定性的块参数 + 注意力参数 + 形状（**两个模型的形状约定相同**）.

    参数由 LCG（day073 的 ``uniforms``，经 day079/day075 的两个生产函数）唯一决定，
    因此"同样的一次前向"是一条可以被复核的结论——而不是"上次跑出来的值"。
    """
    profile_of(name)  # 未知模型名在这里就被拒绝
    shape = BlockShape(hidden=hidden, ffn=FFN_RATIO * hidden, tokens=tokens)
    block = make_block_parameters(shape, seed=seed)
    attention = default_parameters(hidden, seed=seed)
    return block, attention, shape


def source_shape_of(
    name: str,
    *,
    hidden: int = 6,
    tokens: int = 4,
    heads: int = 1,
    vocab: int = 12,
) -> SourceShape:
    """造一个 :class:`~smart_research_agent.hf_source.types.SourceShape`（画像驱动因果默认值）."""
    profile = profile_of(name)
    return SourceShape(
        tokens=tokens,
        hidden=hidden,
        heads=heads,
        vocab=vocab,
        causal_default=profile.norm_placement == NORM_PRE,
    )


def max_abs_gap(left: Matrix, right: Matrix) -> float:
    """两份矩阵之间的**最大绝对差**（两条独立实现之间的读数）."""
    first = validate_matrix(left, name="left")
    second = validate_matrix(right, name="right")
    if len(first) != len(second) or len(first[0]) != len(second[0]):
        raise ShapeError(
            f"两张矩阵形状不一致：{len(first)}x{len(first[0])} "
            f"vs {len(second)}x{len(second[0])}。"
        )
    return max(
        (abs(a - b) for row_a, row_b in zip(first, second) for a, b in zip(row_a, row_b)),
        default=0.0,
    )


#: 与 day079 对账时使用的激活名（day079 只有 relu 与精确 gelu）。
#:
#: 两个名字、两个值都与 day079 相同，因此两个块的对账可以**绕开**
#: `gelu_new` 这个 GPT-2 专有项。
DAY079_ACTIVATIONS: tuple[str, ...] = (DAY079_RELU, DAY079_GELU)


__all__ = [
    "BLOCK_TOLERANCE",
    "DAY079_ACTIVATIONS",
    "FFN_RATIO",
    "GELU_NEW_CUBIC",
    "GELU_NEW_INNER",
    "BlockForward",
    "activate_rows",
    "activation_of",
    "bert_layer",
    "block_of",
    "gelu_exact",
    "gelu_new",
    "gpt2_block",
    "gpt2_mlp",
    "make_hf_block",
    "max_abs_gap",
    "source_shape_of",
]
