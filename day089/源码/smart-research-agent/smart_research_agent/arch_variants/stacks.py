"""``stacks.py``：把三个变体接起来，而**没有一行新的注意力算术**（day082）.

## 一、接线的全部内容

```text
encoder_only      self_attention(mask=全开)   → encoder_block   （原样复用 day079）
decoder_only      self_attention(mask=因果)   → encoder_block   （**同一个块，只换掩码**）
encoder_decoder   上一行 → 再叠 decoder_block（自注意力因果 + 交叉注意力 + 前馈）
```

第 2 行是本课最想让读者记住的一句：**GPT 与 BERT 的每一层是同一个块**，
差别只在递给它的那张掩码。于是“换一个变体”在本包里的代价是**换一个字符串**，
而不是换一份实现——这也是为什么第 5 章的实测表能那么干净地只量“能看到谁”。

## 二、三个变体的掩码从哪来（``mask_for_streams``）

```text
encoder_only      主流 = 全开（可选 ∧ 填充）        源流：无
decoder_only      主流 = 因果（可选 ∧ 填充）        源流：无
encoder_decoder   主流 = **因果**（由 decoder_block 强制）
                  源流 = 全开（``mask_kind`` 控制的就是这一条）
```

第 3 行里那个“由 ``decoder_block`` 强制”不是描述，而是一条**接口事实**：
day079 的 ``decoder_block`` 要求 ``self_attention_forward.causal`` 为真，
而 day075 的 ``resolve_mask`` 拒绝 ``causal=True`` 与显式掩码同时给。
两条加起来的后果是一条被写下来的边界：

```text
T5 的解码器**读不到填充信息**（编码器那一侧可以）
⇒ 本包对 `encoder_decoder + pads` 直接抛 AssemblyError，而不是悄悄忽略 pads
```

“悄悄忽略”在这里格外危险：填充掩码被丢掉之后，**前向照样跑、形状一模一样、
读数只是略有不同**——它不会以任何异常的形式露面（day081 第 10.2 节那类失败）。

## 三、反向：重放一遍前向

与 day079/080 同一条纪律——反向不依赖“记录里的中间账”，而是**用记录里的输入与
三个旋钮重放一遍前向**。代价是几毫秒，换来的是“记录可以说清”的大小。
而它要求记录里留下 ``placement`` / ``use_residual`` / ``activation`` /
``inputs`` / ``source`` / ``mask``：少留任何一个，反向就会在一份**不同配置**上前进，
而它的形状完全合法。

## 四、``census_of``：两条路径数出来的参数量必须相等

```text
逐个数        params.parameter_count()（把每一块参数的长度加起来）
公式算        2·d·f + f + 5·d + 4·d²（每个块）与 4·d² + 2·d·f + f + 7·d（解码器多出来的那部分）
```

它们必须相等——这条断言挡住的是“忘了数某一类参数”这种**不会报错**的错误
（参数少了一块，前向照样跑，只是那一块永远停在初始化）。
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
from smart_research_agent.arch_variants.masks import (
    MASK_CAUSAL,
    MASK_FULL,
    MASK_KINDS,
    Mask,
    combine_masks,
    mask_allowed_pairs,
    mask_fits_sequence,
    mask_is_causal,
    mask_of,
    mask_summary,
    padding_mask,
)
from smart_research_agent.arch_variants.types import (
    DEFAULT_FFN_RATIO,
    DEFAULT_HIDDEN,
    DEFAULT_INIT_SCALE,
    DEFAULT_LAYERS,
    DEFAULT_TOKENS,
    VARIANT_DECODER_ONLY,
    VARIANT_ENCODER_DECODER,
    BlockStackParameters,
    DecoderStackParameters,
    VariantCensus,
    VariantForward,
    VariantParameters,
    VariantShape,
    ones_vector,
    validate_variant,
    zeros_vector,
)
from smart_research_agent.encoder_decoder.depth import (
    make_block_parameters,
    make_stack_parameters,
)
from smart_research_agent.encoder_decoder.layers import (
    decoder_block,
    decoder_block_backward,
    encoder_block,
    encoder_block_backward,
    layer_norm,
)
from smart_research_agent.encoder_decoder.types import (
    ACTIVATION_RELU,
    NORM_PRE,
    BlockForward,
    BlockGradients,
    BlockShape,
    CrossParameters,
    DecoderGradients,
    FFNWeights,
)
from smart_research_agent.encoder_decoder.types import (
    _checked_activation as _checked_activation,
)
from smart_research_agent.encoder_decoder.types import (
    _checked_placement as _checked_placement,
)
from smart_research_agent.math_foundations.probability import uniforms
from smart_research_agent.math_foundations.types import (
    Matrix,
    Vector,
    matrix_shape,
    validate_matrix,
)
from smart_research_agent.transformer_core.layers import self_attention
from smart_research_agent.transformer_core.train import default_parameters
from smart_research_agent.transformer_core.types import AttentionParams

#: 源序列的默认种子（与参数的种子分开：换源序列与换参数是两件事）.
DEFAULT_SOURCE_SEED = 13

#: 层与层之间参数的种子步长（与 day079 的 ``make_stack_parameters`` 逐字相同）.
BLOCK_SEED_STRIDE = 13
ATTENTION_SEED_STRIDE = 17
CROSS_SEED_STRIDE = 23


def _random_matrix(rows: int, columns: int, *, scale: float, seed: int) -> Matrix:
    """确定性随机矩阵（``[-scale, scale)``，用 day073 那串 LCG 随机数）.

    与 day079 的 ``_random_matrix`` 同一条实现、同一串随机数、同一幅度——
    交叉注意力那四个矩阵因此与自注意力那四个**同分布**，
    “换一个变体”与“换一次初始化”不会混在一起。
    """
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ParameterError(f"行数必须 >= 1 的整数，收到 {rows!r}。")
    if isinstance(columns, bool) or not isinstance(columns, int) or columns < 1:
        raise ParameterError(f"列数必须 >= 1 的整数，收到 {columns!r}。")
    if not math.isfinite(scale) or scale <= 0:
        raise ParameterError(f"scale 必须是正的有限数，收到 {scale!r}。")
    raw = uniforms(rows * columns, seed=seed)
    out: list[Vector] = []
    cursor = 0
    for _ in range(rows):
        out.append(
            tuple((value * 2.0 - 1.0) * scale for value in raw[cursor : cursor + columns])
        )
        cursor += columns
    return tuple(out)


def sample_matrix(
    rows: int,
    columns: int,
    *,
    scale: float = DEFAULT_INIT_SCALE,
    seed: int = DEFAULT_SOURCE_SEED,
) -> Matrix:
    """一段确定性的样本矩阵（源序列、目标序列都用它——**换种子重跑一遍很容易**）."""
    return _random_matrix(rows, columns, scale=scale, seed=seed)


def make_variant_shape(
    tokens: int = DEFAULT_TOKENS,
    hidden: int = DEFAULT_HIDDEN,
    *,
    ffn_ratio: int = DEFAULT_FFN_RATIO,
    layers: int = DEFAULT_LAYERS,
    sources: int | None = None,
) -> VariantShape:
    """按“序列长度 + 隐藏维 + 倍数 + 层数”造一个形状（``ffn = ratio × hidden``）."""
    if isinstance(ffn_ratio, bool) or not isinstance(ffn_ratio, int) or ffn_ratio < 1:
        raise ParameterError(f"ffn_ratio 必须是 >= 1 的整数，收到 {ffn_ratio!r}。")
    return VariantShape(
        tokens=tokens,
        hidden=hidden,
        ffn=ffn_ratio * hidden,
        layers=layers,
        sources=sources,
    )


def block_shape_of(shape: VariantShape) -> BlockShape:
    """从变体形状取出 day079 的 ``BlockShape``（**只有一处换算**）."""
    if not isinstance(shape, VariantShape):
        raise ParameterError(f"shape 必须是 VariantShape，收到 {type(shape).__name__}。")
    return BlockShape(hidden=shape.hidden, ffn=shape.ffn, tokens=shape.tokens)


def make_cross_parameters(
    shape: VariantShape,
    *,
    seed: int = 7,
    scale: float = DEFAULT_INIT_SCALE,
) -> CrossParameters:
    """交叉注意力的四个投影（每个都是 ``d × d``，因此**矩阵是方的、权重表是长的**）.

    这里有一处值得单独指出：四个投影**都是方阵**，而它们产生的权重表
    ``(n_tgt, n_src)`` 是**长方形**。“方的参数、长的表”正是交叉注意力的几何特征——
    day079 那条“不能加因果掩码”的拒绝就来自这里（一张方阵掩码配长方形的表，
    只有在两边长度相等时才不报错）。
    """
    hidden = shape.hidden
    return CrossParameters(
        w_query=_random_matrix(hidden, hidden, scale=scale, seed=seed),
        w_key=_random_matrix(hidden, hidden, scale=scale, seed=seed + 1),
        w_value=_random_matrix(hidden, hidden, scale=scale, seed=seed + 2),
        w_output=_random_matrix(hidden, hidden, scale=scale, seed=seed + 3),
    )


def make_block_stack(
    shape: VariantShape,
    *,
    seed: int = 7,
    scale: float = DEFAULT_INIT_SCALE,
) -> BlockStackParameters:
    """一条编码器式块链（**转发 day079 的 ``make_stack_parameters``**）.

    转发而不是重写：一条块链的初始化在 day079 已经被验证过（`uniform` 方案与
    day080 的 ``make_stack_parameters`` 逐位一致），本课只需要它作为**基线**。
    """
    blocks, attentions = make_stack_parameters(
        block_shape_of(shape), shape.layers, seed=seed, scale=scale
    )
    return BlockStackParameters(blocks=blocks, attentions=attentions)


def make_decoder_stack(
    shape: VariantShape,
    *,
    seed: int = 7,
    scale: float = DEFAULT_INIT_SCALE,
) -> DecoderStackParameters:
    """一条解码器式块链：每层自注意力 + 交叉注意力 + 前馈 + 三个 LN.

    三处初始化都刻意与编码器那一侧**同源**：

    ```text
    自注意力    default_parameters(seed + i·17)     与块链的注意力同一个来源
    前馈        make_block_parameters(seed + i·13).ffn  与编码器块的前馈同源
    交叉注意力  新造的四个 d×d（种子步长 23）
    γ/β        γ = 1、β = 0（day079 的口径：让初始时 LN 就是标准化本身）
    ```

    “同源”是为了让对照只有一个变量：编码器与解码器在同一颗种子下拿到**同一个前馈**，
    于是“多了一层交叉注意力”就是两者唯一的差别。
    """
    attentions: list[AttentionParams] = []
    crosses: list[CrossParameters] = []
    ffns: list[FFNWeights] = []
    gammas: list[tuple[Vector, Vector, Vector]] = []
    betas: list[tuple[Vector, Vector, Vector]] = []
    for index in range(shape.layers):
        attentions.append(
            default_parameters(shape.hidden, seed=seed + index * ATTENTION_SEED_STRIDE)
        )
        crosses.append(
            make_cross_parameters(
                shape, seed=seed + index * CROSS_SEED_STRIDE, scale=scale
            )
        )
        block = make_block_parameters(
            block_shape_of(shape),
            seed=seed + index * BLOCK_SEED_STRIDE,
            scale=scale,
        )
        ffns.append(block.ffn)
        gammas.append(
            (ones_vector(shape.hidden), ones_vector(shape.hidden), ones_vector(shape.hidden))
        )
        betas.append(
            (zeros_vector(shape.hidden), zeros_vector(shape.hidden), zeros_vector(shape.hidden))
        )
    return DecoderStackParameters(
        attentions=tuple(attentions),
        crosses=tuple(crosses),
        ffns=tuple(ffns),
        gammas=tuple(gammas),
        betas=tuple(betas),
    )


def make_variant_parameters(
    shape: VariantShape,
    variant: str,
    *,
    seed: int = 7,
    scale: float = DEFAULT_INIT_SCALE,
) -> VariantParameters:
    """一个变体的全套参数（``encoder_decoder`` 会多造一条解码器链）."""
    resolved = validate_variant(variant)
    blocks = make_block_stack(shape, seed=seed, scale=scale)
    decoder = (
        make_decoder_stack(shape, seed=seed, scale=scale)
        if resolved == VARIANT_ENCODER_DECODER
        else None
    )
    return VariantParameters(
        shape=shape, variant=resolved, blocks=blocks, decoder=decoder
    )


# ---------------------------------------------------------------------- 掩码的分配


def _checked_pads(pads: Sequence[bool] | None, *, tokens: int) -> tuple[bool, ...] | None:
    if pads is None:
        return None
    resolved = tuple(pads)
    if not resolved:
        raise ParameterError("pads 不能是空序列：要么不给，要么给出一整条序列的标记。")
    if len(resolved) != tokens:
        raise ShapeError(
            f"填充标记有 {len(resolved)} 个，序列长度是 {tokens}："
            "两者不一致时掩码会错位，而错位的掩码**不会报错**（它只会让某几行少看一个位置）。"
        )
    for index, value in enumerate(resolved):
        if not isinstance(value, bool):
            raise ParameterError(f"第 {index} 个填充标记必须是 bool，收到 {value!r}。")
    return resolved


def mask_for_streams(
    variant: str,
    shape: VariantShape,
    *,
    pads: Sequence[bool] | None = None,
    mask_kind: str | None = None,
) -> tuple[Mask, Mask | None]:
    """按变体给出**两条流**各自的掩码：``(主流, 源流或 None)``.

    ``mask_kind`` 覆盖的是“**单流那一侧**”的掩码：单流变体就是唯一那条流，
    ``encoder_decoder`` 则是**编码器**那条流（解码器那一侧被 day079 的接口钉死为因果）。
    """
    resolved_variant = validate_variant(variant)
    if mask_kind is not None and mask_kind not in MASK_KINDS:
        raise ParameterError(
            f"未知的掩码名 {mask_kind!r}：可选 {', '.join(MASK_KINDS)}。"
        )
    default_kind = MASK_CAUSAL if resolved_variant == VARIANT_DECODER_ONLY else MASK_FULL
    kind = default_kind if mask_kind is None else mask_kind
    checked_pads = _checked_pads(pads, tokens=shape.tokens)
    if checked_pads is not None and resolved_variant == VARIANT_ENCODER_DECODER:
        raise AssemblyError(
            "encoder_decoder 变体的解码器**读不到填充信息**：day079 的 decoder_block "
            "要求自注意力是因果的，而 day075 的 resolve_mask 拒绝因果与显式掩码同时给——"
            "因此本包拒绝 `pads`，而不是悄悄把它丢掉（丢掉的后果是前向照跑、读数略有不同）。"
        )
    base = mask_of(kind, shape.tokens)
    main = (
        combine_masks(base, padding_mask(checked_pads))
        if checked_pads is not None
        else base
    )
    if resolved_variant == VARIANT_ENCODER_DECODER:
        return mask_of(MASK_CAUSAL, shape.tokens), mask_of(kind, shape.source_length)
    return main, None


# ---------------------------------------------------------------------- 前向


def _first_sublayer_input(
    inputs: Matrix,
    *,
    placement: str,
    gamma: Vector,
    beta: Vector,
) -> Matrix:
    """**第一个子层看到的输入**：pre-LN 是 ``LN(x)``、post-LN 是 ``x``.

    这四行是本课踩到的第二个真实的坑，而它的表现只有一处：**梯度对不上**。

    ```text
    pre-LN 时，块里的注意力作用在 LN(x) 上   ⇒ 传进来的注意力账也必须是 LN(x) 上的
    若图省事把注意力作用在原始 x 上            ⇒ 前向看起来完全正常（形状、读数都对）
                                             ，而 encoder_block_backward 会按“注意力在
                                             LN(x) 上”那一条链把梯度分给 γ₁/β₁ ——
                                             两边算的**不是同一个函数**，误差在 1e-1 量级
    ```

    day079 的 ``block_attention`` 存在的理由就是这件事（它的文档里写了同一句话）。
    本包没有直接调用它，只因为这里还需要把**显式掩码**传下去——因此这四行
    是 ``block_attention`` 的“带掩码版本”，而两处必须保持同一个口径。
    """
    if placement == NORM_PRE:
        normed, _cache = layer_norm(inputs, gamma=gamma, beta=beta)
        return normed
    return inputs


@dataclass(frozen=True)
class _Replay:
    """一次前向的全部“反向要用的东西”（**私有**：它不是给读者看的记录）."""

    output: Matrix
    block_forwards: tuple[BlockForward, ...]
    block_weights: tuple[Matrix, ...]
    encoder_output: Matrix | None
    decoder_parts: tuple[tuple[Any, ...], ...]
    decoder_self_weights: tuple[Matrix, ...]
    cross_weights: tuple[Matrix, ...]


def _replay(
    params: VariantParameters,
    inputs: Matrix,
    source: Matrix | None,
    *,
    stream_mask: Mask,
    placement: str,
    use_residual: bool,
    activation: str,
) -> _Replay:
    """把变体的前向重放一遍（前向与反向各自调用它一次，因此两边一定是同一个函数）."""
    shape = params.shape
    two_streams = params.variant == VARIANT_ENCODER_DECODER
    if two_streams:
        if source is None:  # pragma: no cover - variant_forward 已经保证过
            raise AssemblyError("encoder_decoder 变体的前向必须有源序列。")
        stream_input = source
    else:
        stream_input = inputs
    current = stream_input
    block_forwards: list[BlockForward] = []
    block_weights: list[Matrix] = []
    for index in range(shape.layers):
        block_params = params.blocks.blocks[index]
        normed = _first_sublayer_input(
            current,
            placement=placement,
            gamma=block_params.norm1_gamma,
            beta=block_params.norm1_beta,
        )
        attention = self_attention(params.blocks.attentions[index], normed, mask=stream_mask)
        block = encoder_block(
            block_params,
            current,
            attention,
            placement=placement,
            use_residual=use_residual,
            activation=activation,
        )
        block_forwards.append(block)
        block_weights.append(attention.weights)
        current = block.output
    encoder_output: Matrix | None = None
    decoder_parts: list[tuple[Any, ...]] = []
    decoder_self_weights: list[Matrix] = []
    cross_weights: list[Matrix] = []
    if two_streams:
        decoder_stack = params.decoder
        if decoder_stack is None:  # pragma: no cover - VariantParameters 已经保证过
            raise AssemblyError("encoder_decoder 变体必须有解码器链。")
        encoder_output = current
        current = inputs
        for index in range(shape.layers):
            normed = _first_sublayer_input(
                current,
                placement=placement,
                gamma=decoder_stack.gammas[index][0],
                beta=decoder_stack.betas[index][0],
            )
            self_forward = self_attention(
                decoder_stack.attentions[index], normed, causal=True
            )
            output, cross_forward, caches, ffn_cache = decoder_block(
                self_forward,
                decoder_stack.crosses[index],
                decoder_stack.gammas[index],
                decoder_stack.betas[index],
                decoder_stack.ffns[index],
                current,
                encoder_output,
                use_residual=use_residual,
                activation=activation,
            )
            decoder_parts.append((self_forward, cross_forward, caches, ffn_cache))
            decoder_self_weights.append(self_forward.weights)
            cross_weights.append(cross_forward.weights)
            current = output
    return _Replay(
        output=current,
        block_forwards=tuple(block_forwards),
        block_weights=tuple(block_weights),
        encoder_output=encoder_output,
        decoder_parts=tuple(decoder_parts),
        decoder_self_weights=tuple(decoder_self_weights),
        cross_weights=tuple(cross_weights),
    )


def variant_forward(
    params: VariantParameters,
    inputs: Matrix,
    *,
    source: Matrix | None = None,
    pads: Sequence[bool] | None = None,
    mask_kind: str | None = None,
    placement: str = NORM_PRE,
    use_residual: bool = True,
    activation: str = ACTIVATION_RELU,
    source_seed: int = DEFAULT_SOURCE_SEED,
) -> VariantForward:
    """一个变体的前向：掩码按变体分配、两条流按变体接上，其余全部复用 day079/080.

    三处“不替调用方猜”：

    ```text
    单流变体给 source    → AssemblyError（它只有一条流，交叉注意力没有第二路）
    enc/dec 不给 source  → 造一段确定性样本（**同一颗种子 ⇒ 可复现**）
    enc/dec 给 pads      → AssemblyError（见模块说明第二节）
    ```
    """
    if not isinstance(params, VariantParameters):
        raise ParameterError(f"params 必须是 VariantParameters，收到 {type(params).__name__}。")
    checked_inputs = validate_matrix(inputs, name="inputs")
    expected = (params.shape.tokens, params.shape.hidden)
    if matrix_shape(checked_inputs) != expected:
        raise ShapeError(f"输入形状 {matrix_shape(checked_inputs)} 与 {expected} 不一致。")
    resolved_placement = _checked_placement(placement)
    resolved_activation = _checked_activation(activation)
    stream_mask, source_mask = mask_for_streams(
        params.variant, params.shape, pads=pads, mask_kind=mask_kind
    )
    resolved_source: Matrix | None
    if params.variant == VARIANT_ENCODER_DECODER:
        if source is None:
            resolved_source = sample_matrix(
                params.shape.source_length, params.shape.hidden, seed=source_seed
            )
        else:
            resolved_source = validate_matrix(source, name="source")
            if matrix_shape(resolved_source) != (
                params.shape.source_length,
                params.shape.hidden,
            ):
                raise ShapeError(
                    f"源序列形状 {matrix_shape(resolved_source)} 与 shape 声明的 "
                    f"({params.shape.source_length}, {params.shape.hidden}) 不一致。"
                )
    else:
        if source is not None:
            raise AssemblyError(
                f"{params.variant} 变体只有一条流，不该收到 source：交叉注意力要的是"
                "**第二路**的 K/V——两路里多给一路时，那个乘法没有定义。"
            )
        resolved_source = None
    replay = _replay(
        params,
        checked_inputs,
        resolved_source,
        stream_mask=source_mask if source_mask is not None else stream_mask,
        placement=resolved_placement,
        use_residual=use_residual,
        activation=resolved_activation,
    )
    main_mask = mask_of(MASK_CAUSAL, params.shape.tokens) if source_mask is not None else stream_mask
    notes = [
        f"掩码（主流）：{mask_summary(main_mask)}",
        f"掩码（源流）：{mask_summary(source_mask)}" if source_mask is not None else "源流：无",
        f"摆放 {resolved_placement} | 残差 {'开' if use_residual else '关'} | 激活 {resolved_activation}",
    ]
    if source_mask is not None:
        notes.append(
            "解码器那一侧的自注意力由 decoder_block 强制为因果（day079 的组装拒绝）"
        )
    return VariantForward(
        variant=params.variant,
        shape=params.shape,
        inputs=checked_inputs,
        output=replay.output,
        mask=main_mask,
        placement=resolved_placement,
        use_residual=use_residual,
        activation=resolved_activation,
        source=resolved_source,
        source_mask=source_mask,
        encoder_output=replay.encoder_output,
        block_weights=replay.block_weights,
        decoder_self_weights=replay.decoder_self_weights,
        cross_weights=replay.cross_weights,
        notes=tuple(notes),
    )


def variant_loss(
    params: VariantParameters,
    inputs: Matrix,
    target: Matrix,
    **kwargs: Any,
) -> float:
    """``0.5 · Σ (输出 − 目标)²``（梯度是 ``输出 − 目标``，**一行就能手算**）.

    损失取得这么平凡是刻意的（与 day075 用 MSE 验证注意力、day079 用同样手法一样）：
    **一次只留一个不确定性**——外部输入的梯度是平凡的，任何梯度校验失败
    都只可能来自“掩码有没有被穿进反向”这件事。
    """
    forward = variant_forward(params, inputs, **kwargs)
    checked_target = validate_matrix(target, name="target")
    if matrix_shape(checked_target) != matrix_shape(forward.output):
        raise ShapeError(
            f"目标形状 {matrix_shape(checked_target)} 与输出 "
            f"{matrix_shape(forward.output)} 不一致。"
        )
    total = math.fsum(
        (a - b) * (a - b)
        for left, right in zip(forward.output, checked_target, strict=True)
        for a, b in zip(left, right, strict=True)
    )
    return 0.5 * total


def loss_gradient(output: Matrix, target: Matrix) -> Matrix:
    """``0.5·Σ(y − t)²`` 对 ``y`` 的梯度：``y − t``."""
    checked_output = validate_matrix(output, name="output")
    checked_target = validate_matrix(target, name="target")
    if matrix_shape(checked_output) != matrix_shape(checked_target):
        raise ShapeError(
            f"输出形状 {matrix_shape(checked_output)} 与目标 "
            f"{matrix_shape(checked_target)} 不一致。"
        )
    return tuple(
        tuple(a - b for a, b in zip(left, right, strict=True))
        for left, right in zip(checked_output, checked_target, strict=True)
    )


# ---------------------------------------------------------------------- 反向


@dataclass(frozen=True)
class VariantGradients:
    """一个变体的反向：**两条流各自的输入梯度** + 每一层入口的梯度序列.

    ```text
    grad_inputs         目标流（或唯一那条流）输入的梯度
    grad_source_inputs  源序列输入的梯度（只有 encoder_decoder；**穿过编码器那一条链**）
    grad_block_inputs   每一层入口的梯度范数对应的矩阵（逐层读数的来源，day080 的伏笔）
    decoder_gradients   解码器每一层的九块梯度（只有 encoder_decoder）
    ```

    ``grad_source_inputs`` 是本课与 day079 那条 ``dSource = dK·W_k + dV·W_v`` 的合流点：
    它必须**穿过整条编码器链**才能回到源序列的输入上，而漏掉那一段的写法
    形状完全合法（只是梯度停在了编码器的输出上）。
    """

    variant: str
    grad_inputs: Matrix
    grad_source_inputs: Matrix | None = None
    grad_block_inputs: tuple[Matrix, ...] = ()
    decoder_gradients: tuple[DecoderGradients, ...] = ()
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        object.__setattr__(self, "variant", validate_variant(self.variant))
        object.__setattr__(self, "grad_inputs", validate_matrix(self.grad_inputs, name="grad_inputs"))
        if self.grad_source_inputs is not None:
            object.__setattr__(
                self,
                "grad_source_inputs",
                validate_matrix(self.grad_source_inputs, name="grad_source_inputs"),
            )
        elif self.decoder_gradients:
            raise NumericError(
                "有解码器梯度时必须有源序列的梯度：交叉注意力把梯度**送回了另一路**，"
                "而丢掉它不会以任何形状错误的形式暴露出来。"
            )
        object.__setattr__(self, "grad_block_inputs", tuple(self.grad_block_inputs))
        object.__setattr__(self, "decoder_gradients", tuple(self.decoder_gradients))
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))


def variant_backward(
    forward: VariantForward,
    params: VariantParameters,
    grad_output: Matrix,
    *,
    activation: str | None = None,
) -> VariantGradients:
    """一个变体的反向：**重放一遍前向**，再从最后一层倒着走回两条流的入口.

    三处顺序不能反（都不报错，只会让梯度少走一段）：

    ```text
    ① 单流：第 i 层拿到的 grad_output 必须是第 i+1 层的 grad_inputs
    ② 两条流：解码器每一层的 dEncoder 要**累加**，而不是被最后一层覆盖
    ③ 两条流：拿到 dEncoder 之后还要**穿过整条编码器链**才回到 source
    ```
    """
    if not isinstance(forward, VariantForward):
        raise ParameterError(f"forward 必须是 VariantForward，收到 {type(forward).__name__}。")
    if not isinstance(params, VariantParameters):
        raise ParameterError(f"params 必须是 VariantParameters，收到 {type(params).__name__}。")
    if forward.variant != params.variant:
        raise AssemblyError(
            f"记录里的变体是 {forward.variant}，参数是 {params.variant}："
            "两份不同变体的账拼在一起时，形状**完全合法**——这正是它危险的地方。"
        )
    checked_grad = validate_matrix(grad_output, name="grad_output")
    if matrix_shape(checked_grad) != matrix_shape(forward.output):
        raise ShapeError(
            f"回传梯度 {matrix_shape(checked_grad)} 与输出 "
            f"{matrix_shape(forward.output)} 不一致。"
        )
    resolved_activation = (
        forward.activation if activation is None else _checked_activation(activation)
    )
    replay = _replay(
        params,
        forward.inputs,
        forward.source,
        stream_mask=forward.source_mask if forward.source_mask is not None else forward.mask,
        placement=forward.placement,
        use_residual=forward.use_residual,
        activation=resolved_activation,
    )
    if forward.source_mask is not None:
        return _backward_two_streams(forward, params, replay, checked_grad, resolved_activation)
    return _backward_single_stream(forward, params, replay, checked_grad, resolved_activation)


def _backward_single_stream(
    forward: VariantForward,
    params: VariantParameters,
    replay: _Replay,
    grad_output: Matrix,
    activation: str,
) -> VariantGradients:
    """单流变体的反向：一条链倒着走，逐层记下入口梯度."""
    current = grad_output
    entry_gradients: list[Matrix] = [grad_output] * forward.layers
    block_gradients: list[BlockGradients] = []
    for index in range(forward.layers - 1, -1, -1):
        entry_gradients[index] = current
        grads = encoder_block_backward(
            replay.block_forwards[index],
            params.blocks.blocks[index],
            current,
            activation=activation,
        )
        block_gradients.append(grads)
        current = grads.grad_inputs
    return VariantGradients(
        variant=forward.variant,
        grad_inputs=current,
        grad_block_inputs=tuple(entry_gradients),
        notes=(
            f"单流反向：{forward.layers} 层，逐层记下入口梯度（共 {len(entry_gradients)} 个入口）",
            "最底层入口的梯度由最后一步给出——它与输入的形状相同",
        ),
    )


def _backward_two_streams(
    forward: VariantForward,
    params: VariantParameters,
    replay: _Replay,
    grad_output: Matrix,
    activation: str,
) -> VariantGradients:
    """两条流的反向：先解码器（**累加** dEncoder），再穿过整条编码器链."""
    decoder_stack = params.decoder
    if decoder_stack is None:  # pragma: no cover - VariantParameters 已经保证过
        raise AssemblyError("encoder_decoder 变体必须有解码器链。")
    d_target = grad_output
    d_encoder: Matrix | None = None
    decoder_gradients: list[DecoderGradients] = []
    for index in range(forward.layers - 1, -1, -1):
        self_forward, cross_forward, caches, ffn_cache = replay.decoder_parts[index]
        grads = decoder_block_backward(
            self_forward,
            cross_forward,
            caches,
            ffn_cache,
            d_target,
            use_residual=forward.use_residual,
        )
        decoder_gradients.append(grads)
        d_target = grads.grad_decoder_inputs
        d_encoder = (
            grads.grad_encoder_outputs
            if d_encoder is None
            else tuple(
                tuple(a + b for a, b in zip(left, right, strict=True))
                for left, right in zip(d_encoder, grads.grad_encoder_outputs, strict=True)
            )
        )
    if d_encoder is None:  # pragma: no cover - 层数 >= 1，构造上不可能
        raise NumericError("解码器链至少要有一层。")
    current = d_encoder
    entry_gradients: list[Matrix] = [d_encoder] * forward.layers
    for index in range(forward.layers - 1, -1, -1):
        entry_gradients[index] = current
        block_grads = encoder_block_backward(
            replay.block_forwards[index],
            params.blocks.blocks[index],
            current,
            activation=activation,
        )
        current = block_grads.grad_inputs
    return VariantGradients(
        variant=forward.variant,
        grad_inputs=d_target,
        grad_source_inputs=current,
        grad_block_inputs=tuple(entry_gradients),
        decoder_gradients=tuple(reversed(decoder_gradients)),
        notes=(
            f"两条流反向：解码器 {forward.layers} 层（逐层累加 dEncoder）+ 编码器 "
            f"{forward.layers} 层",
            "grad_source_inputs 必须穿过整条编码器链才能回到源序列的输入上",
        ),
    )


# ---------------------------------------------------------------------- 数参数


def census_of(params: VariantParameters) -> VariantCensus:
    """数一遍这个变体的家底（**逐个数**与**公式算**两条路径必须相等）."""
    if not isinstance(params, VariantParameters):
        raise ParameterError(f"params 必须是 VariantParameters，收到 {type(params).__name__}。")
    shape = params.shape
    hidden, ffn, layers = shape.hidden, shape.ffn, shape.layers
    block_cost = 2 * hidden * ffn + ffn + hidden + 4 * hidden
    attention_cost = 4 * hidden * hidden
    cross_cost = 4 * hidden * hidden
    decoder_ffn_cost = 2 * hidden * ffn + ffn + hidden
    decoder_norm_cost = 6 * hidden
    if params.variant == VARIANT_ENCODER_DECODER:
        analytic = layers * (block_cost + attention_cost) + layers * (
            attention_cost + cross_cost + decoder_ffn_cost + decoder_norm_cost
        )
        decoder_layers = layers
        cross_layers = layers
        sub_layers = 2 * layers + 3 * layers
    else:
        analytic = layers * (block_cost + attention_cost)
        decoder_layers = 0
        cross_layers = 0
        sub_layers = 2 * layers
    censused = params.parameter_count()
    visible = mask_allowed_pairs(params_mask_of(params))
    notes = [
        f"每个块 {block_cost} 个参数（前馈 2df + f + d，两个 LN 的 γ/β 共 4d）",
        f"每层自注意力 {attention_cost} 个参数（四个 d×d 投影，无偏置）",
        f"主流掩码允许 {visible}/{shape.tokens * shape.tokens} 个位置对",
    ]
    if cross_layers:
        notes.append(f"每层交叉注意力 {cross_cost} 个参数：**方阵参数、长方形权重表**")
    return VariantCensus(
        variant=params.variant,
        layers=layers,
        block_layers=layers,
        decoder_layers=decoder_layers,
        self_attention_layers=layers + decoder_layers,
        cross_attention_layers=cross_layers,
        sub_layers=sub_layers,
        parameters=censused,
        analytic_parameters=analytic,
        notes=tuple(notes),
    )


def params_mask_of(params: VariantParameters) -> Mask:
    """这个变体主流的默认掩码（**只给计数用**：它不依赖输入数据）."""
    main, _source = mask_for_streams(params.variant, params.shape)
    return main


def visible_positions(params: VariantParameters) -> tuple[int, ...]:
    """主流每一行允许看到的位置数（因果掩码下是 ``1, 2, ..., n``）."""
    from smart_research_agent.arch_variants.masks import mask_allowed_counts

    return mask_allowed_counts(params_mask_of(params))


def stream_is_causal(params: VariantParameters) -> bool:
    """这个变体的主流是不是因果的（``mask_is_causal`` 只看掩码，不看变体名）."""
    return mask_is_causal(params_mask_of(params))


def encoder_stream_is_causal(params: VariantParameters) -> bool:
    """编码器那一条流是不是因果的（单流变体时它就是主流）."""
    if params.variant != VARIANT_ENCODER_DECODER:
        return stream_is_causal(params)
    _main, source = mask_for_streams(params.variant, params.shape)
    if source is None:  # pragma: no cover - 构造上不可能
        raise NumericError("encoder_decoder 变体必须有源流的掩码。")
    return mask_is_causal(source)


def check_mask_fits(mask: Mask, length: int) -> Mask:
    """掩码与序列长度一致性的唯一入口（转发 ``masks.mask_fits_sequence``）."""
    return mask_fits_sequence(mask, length)


__all__ = [
    "ATTENTION_SEED_STRIDE",
    "BLOCK_SEED_STRIDE",
    "CROSS_SEED_STRIDE",
    "DEFAULT_SOURCE_SEED",
    "VariantGradients",
    "block_shape_of",
    "census_of",
    "check_mask_fits",
    "encoder_stream_is_causal",
    "loss_gradient",
    "make_block_stack",
    "make_cross_parameters",
    "make_decoder_stack",
    "make_variant_parameters",
    "make_variant_shape",
    "mask_for_streams",
    "params_mask_of",
    "sample_matrix",
    "stream_is_causal",
    "variant_backward",
    "variant_forward",
    "variant_loss",
    "visible_positions",
]
