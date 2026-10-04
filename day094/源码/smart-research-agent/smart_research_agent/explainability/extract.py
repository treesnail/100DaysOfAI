"""``extract.py``：把权重**读出来**——两条读法，一个刻意的区分（day083）.

## 一、两种读法，而它们不是同一条前向

```text
self_records    模型**真实**看到的权重：直接取自 day082 的前向记录
                （block_weights / decoder_self_weights / cross_weights）
head_records    **同一组投影**在多头划分下的读法：用 day076 的 multi_head_attention
                在同一个输入上重算一遍
```

为什么必须分开？因为 day076 的一条硬事实：

```text
单头（day075）    缩放 = 1/√d_k         每一头看到 d_k 个维度
多头（day076）    缩放 = 1/√(d_k/heads)  每一头看到 d_k/heads 个维度
                  ⇒ 打分尺度差 √heads 倍（day076 的 scale_ratio 就是这个数）
```

于是"把单头的权重按头切开"这件事**根本不存在**：多头不是"单头切成几份"，
而是"投影划分 + 缩放都不同"的另一个算子。因此本包：

```text
① 不把两者混着比（两个函数、两批记录）
② 给出一条**跨口径的接缝**：head_records(heads=1) 必须**逐位**等于 self_records
   ——这条断言把"两处口径走散"变成一次可以被抓到的失败
③ 有一条边界：day079 的 encoder_block / decoder_block 要求 day075 的 AttentionForward，
   而多头的那份是**另一种类型**（MultiHeadForward）⇒ 两处不能直接拼。
   因此本包的多头读法只作用在"某一层的输入"上，而不是"整条链都换多头"。
```

第 ③ 条是这一课**发现的接口事实**，值得写下来：day085（源码精读）会看到
Hugging Face 的 `T5Attention` 同时支持 `n_heads` 与因果掩码——
那是因为它把"分头"放在**投影内部**，而不是换一个块类型。

## 二、`self_records` 里三条流各自的名字与掩码

```text
encoder_only       "层 i · 头 0"                     掩码 = 主流（全开）
decoder_only       "层 i · 头 0"                     掩码 = 主流（因果）
encoder_decoder    "编码器 i · 头 0"                 掩码 = 源流（全开）
                   "解码器 i · 头 0"                 掩码 = **因果**（day079 强制）
                   "交叉 i"                          掩码 = **全开的长方形**（无掩码）
```

第三条（交叉）的掩码是全开的**长方形**——它正好是 day082 第 4 条性质
（``cross_spans_all_sources``）读的那张表。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

from smart_research_agent.arch_variants.stacks import (
    VariantParameters,
    mask_for_streams,
    variant_forward,
)
from smart_research_agent.arch_variants.types import (
    VARIANT_ENCODER_DECODER,
    VariantForward,
    validate_variant,
)
from smart_research_agent.encoder_decoder.layers import encoder_block, layer_norm
from smart_research_agent.encoder_decoder.types import NORM_PRE
from smart_research_agent.encoder_decoder.types import (
    _checked_placement as _checked_placement,
)
from smart_research_agent.explainability.errors import (
    AssemblyError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.explainability.types import (
    DEFAULT_HEADS,
    STREAM_CROSS,
    STREAM_SELF,
    AttentionRecord,
    causal_mask,
    full_mask,
)
from smart_research_agent.math_foundations.types import (
    Matrix,
    matrix_shape,
    validate_matrix,
)
from smart_research_agent.multi_head.layers import multi_head_attention
from smart_research_agent.transformer_core.layers import self_attention

#: 三条流的标签前缀（**只有一处实现**：报告与热力图都用它）.
LABEL_LAYER = "层"
LABEL_ENCODER = "编码器"
LABEL_DECODER = "解码器"
LABEL_CROSS = "交叉"


def _checked_params(params: Any) -> VariantParameters:
    if not isinstance(params, VariantParameters):
        raise ParameterError(
            f"params 必须是 VariantParameters，收到 {type(params).__name__}。"
        )
    return params


def _checked_labels(labels: Sequence[str] | None, *, length: int, name: str) -> tuple[str, ...]:
    if labels is None:
        return tuple(str(index) for index in range(length))
    resolved = tuple(str(item) for item in labels)
    if len(resolved) != length:
        raise ShapeError(f"{name} 有 {len(resolved)} 个标签，需要 {length} 个。")
    return resolved


def _rectangle(rows: int, columns: int) -> tuple[tuple[bool, ...], ...]:
    """全开的长方形掩码（交叉注意力**没有**掩码：这是 day079 那条拒绝的正面读法）."""
    return tuple(tuple(True for _ in range(columns)) for _ in range(rows))


def _record(
    label: str,
    weights: Matrix,
    mask: tuple[tuple[bool, ...], ...],
    *,
    stream: str,
    tokens: tuple[str, ...],
    source_tokens: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
) -> AttentionRecord:
    return AttentionRecord(
        label=label,
        weights=weights,
        mask=mask,
        stream=stream,
        tokens=tokens,
        source_tokens=source_tokens,
        notes=notes,
    )


def self_records(
    params: VariantParameters,
    inputs: Matrix,
    *,
    source: Matrix | None = None,
    tokens: Sequence[str] | None = None,
    source_tokens: Sequence[str] | None = None,
) -> tuple[AttentionRecord, ...]:
    """模型**真实**的注意力权重（每一层一份，头数为 1）.

    它不做任何重放：权重直接来自 :func:`arch_variants.variant_forward` 的记录。
    因此"读出来的东西"与"模型实际算的东西"**逐位相同**——
    这条性质由 ``test_explain_extract.py`` 里的一条断言钉住。
    """
    checked = _checked_params(params)
    forward = variant_forward(checked, inputs, source=source)
    row_labels = _checked_labels(tokens, length=checked.shape.tokens, name="tokens")
    column_labels = _checked_labels(
        source_tokens, length=checked.shape.source_length, name="source_tokens"
    )
    records: list[AttentionRecord] = []
    if checked.variant != VARIANT_ENCODER_DECODER:
        for index, weights in enumerate(forward.block_weights):
            records.append(
                _record(
                    f"{LABEL_LAYER} {index} · 头 0",
                    weights,
                    forward.mask,
                    stream=STREAM_SELF,
                    tokens=row_labels,
                    notes=(f"主流（唯一那条流），掩码 {_mask_kind_label(forward)}",),
                )
            )
        return tuple(records)
    source_mask = forward.source_mask
    if source_mask is None:  # pragma: no cover - VariantForward 已经保证过
        raise NumericError("encoder_decoder 变体必须有源流的掩码。")
    for index, weights in enumerate(forward.block_weights):
        records.append(
            _record(
                f"{LABEL_ENCODER} {index} · 头 0",
                weights,
                source_mask,
                stream=STREAM_SELF,
                tokens=column_labels,
                notes=("编码器那一流（吃的是源序列）",),
            )
        )
    causal = causal_mask(checked.shape.tokens)
    for index, weights in enumerate(forward.decoder_self_weights):
        records.append(
            _record(
                f"{LABEL_DECODER} {index} · 头 0",
                weights,
                causal,
                stream=STREAM_SELF,
                tokens=row_labels,
                notes=("解码器那一流：因果（day079 的 decoder_block 强制）",),
            )
        )
    for index, weights in enumerate(forward.cross_weights):
        rows, columns = matrix_shape(weights)
        records.append(
            _record(
                f"{LABEL_CROSS} {index}",
                weights,
                _rectangle(rows, columns),
                stream=STREAM_CROSS,
                tokens=row_labels,
                source_tokens=column_labels,
                notes=("交叉那一路：**没有掩码**（长方形，每一行覆盖全部源位置）",),
            )
        )
    return tuple(records)


def _mask_kind_label(forward: VariantForward) -> str:
    """给掩码起一个人读的名字（因果 / 全开）."""
    from smart_research_agent.arch_variants.masks import mask_is_causal

    return "因果" if mask_is_causal(forward.mask) else "全开"


def self_stream_records(
    params: VariantParameters,
    inputs: Matrix,
    *,
    source: Matrix | None = None,
    tokens: Sequence[str] | None = None,
) -> tuple[AttentionRecord, ...]:
    """只要**自注意力**那几份（把交叉那一路筛掉：本包不做"按列对比两路"这种事）."""
    return tuple(
        record
        for record in self_records(params, inputs, source=source, tokens=tokens)
        if record.stream == STREAM_SELF
    )


def cross_records(
    params: VariantParameters,
    inputs: Matrix,
    source: Matrix,
    *,
    tokens: Sequence[str] | None = None,
    source_tokens: Sequence[str] | None = None,
) -> tuple[AttentionRecord, ...]:
    """只要交叉那几份（**只有** ``encoder_decoder`` 才有；否则当场拒绝）."""
    checked = _checked_params(params)
    if checked.variant != VARIANT_ENCODER_DECODER:
        raise AssemblyError(
            f"{checked.variant} 变体没有交叉注意力：它的 K/V 只来自自己那一条流——"
            "交叉注意力要的是**第二路**。"
        )
    return tuple(
        record
        for record in self_records(
            checked, inputs, source=source, tokens=tokens, source_tokens=source_tokens
        )
        if record.stream == STREAM_CROSS
    )


def _stream_input_at_layer(
    params: VariantParameters,
    inputs: Matrix,
    layer: int,
    *,
    source: Matrix | None,
    placement: str,
) -> Matrix:
    """第 ``layer`` 层**第一个子层看到的原始输入**（按 day082 的单头路径重放）.

    这件事必须做对，否则第 1 层之后的读法就全是错的——而它**不会报错**：
    形状对、行随机、熵也算得出来，只是那些数属于"把输入直接喂给第 k 层"这件事。
    第一次运行第 :func:`single_head_matches` 时就抓到了它：
    ``heads = 1`` 在第 0 层逐位相等、在第 2 层不相等 ⇒ 说明读的不是那一层的输入。

    重放用的是**单头**路径（与 day082 的前向逐字相同），因此 ``heads = 1`` 的
    多头读法与模型真实权重逐位相同；``heads > 1`` 时它才是"另一种读法"。
    """
    main_mask, source_mask = mask_for_streams(params.variant, params.shape)
    stream_input = inputs
    if params.variant == VARIANT_ENCODER_DECODER:
        if source is None:
            raise AssemblyError(
                "encoder_decoder 变体的逐层读法要显式给出源序列（编码器那一流吃的是它）。"
            )
        stream_input = source
    mask = source_mask if source_mask is not None else main_mask
    current = stream_input
    for index in range(layer):
        block_params = params.blocks.blocks[index]
        if placement == NORM_PRE:
            normed, _cache = layer_norm(
                current, gamma=block_params.norm1_gamma, beta=block_params.norm1_beta
            )
        else:
            normed = current
        attention = self_attention(params.blocks.attentions[index], normed, mask=mask)
        block = encoder_block(block_params, current, attention, placement=placement)
        current = block.output
    return current


def head_records(
    params: VariantParameters,
    inputs: Matrix,
    *,
    layer: int = 0,
    heads: int = DEFAULT_HEADS,
    source: Matrix | None = None,
    placement: str = NORM_PRE,
    tokens: Sequence[str] | None = None,
) -> tuple[AttentionRecord, ...]:
    """**同一组投影**在多头划分下的读法（每一头一份）.

    与 :func:`self_records` 的三处接口差别：

    ```text
    ① 它作用在**某一层的输入**上（``layer``）—— 那一层的输入由"单头路径重放"得到
       （见 _stream_input_at_layer），而不是"整条链都换多头"：
       因为 day079 的块要求 day075 的 AttentionForward，两者类型不同（模块说明第 ③ 条）
    ② ``heads > 1`` 时缩放变了（1/√d_head），因此它的权重**不是**模型的权重
    ③ ``heads = 1`` 时它必须**逐位**等于 :func:`self_records` 那一条记录
       ——这条断言把"读错了层"与"两处口径走散"都变成一次可以被抓到的失败
    """
    checked = _checked_params(params)
    resolved_placement = _checked_placement(placement)
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise ParameterError(f"层号必须是整数，收到 {layer!r}。")
    if not 0 <= layer < checked.shape.layers:
        raise ParameterError(
            f"层号 {layer} 越界：可选 0..{checked.shape.layers - 1}。"
        )
    validate_heads(checked.shape.hidden, heads)
    checked_inputs = validate_matrix(inputs, name="inputs")
    row_labels = _checked_labels(tokens, length=checked.shape.tokens, name="tokens")
    main_mask, source_mask = mask_for_streams(checked.variant, checked.shape)
    if checked.variant == VARIANT_ENCODER_DECODER:
        if source is None:
            raise AssemblyError(
                "encoder_decoder 变体的多头读法要显式给出源序列："
                "编码器那一流吃的是源序列，用默认样本会量出一段与调用方无关的表。"
            )
        stream_input = validate_matrix(source, name="source")
        expected = (checked.shape.source_length, checked.shape.hidden)
        if matrix_shape(stream_input) != expected:
            raise ShapeError(f"源序列形状 {matrix_shape(stream_input)} 与 {expected} 不一致。")
        mask = source_mask if source_mask is not None else main_mask
    else:
        stream_input = checked_inputs
        mask = main_mask
    block_params = checked.blocks.blocks[layer]
    current = _stream_input_at_layer(
        checked,
        checked_inputs,
        layer,
        source=source,
        placement=resolved_placement,
    )
    if resolved_placement == NORM_PRE:
        normed, _cache = layer_norm(
            current, gamma=block_params.norm1_gamma, beta=block_params.norm1_beta
        )
    else:
        normed = current
    forward = multi_head_attention(
        checked.blocks.attentions[layer], normed, heads=heads, mask=mask
    )
    labels = (
        row_labels
        if len(stream_input) == checked.shape.tokens
        else _checked_labels(None, length=len(stream_input), name="tokens")
    )
    return tuple(
        _record(
            f"{LABEL_LAYER} {layer} · 头 {head}",
            forward.head_weights[head],
            mask,
            stream=STREAM_SELF,
            tokens=labels,
            notes=(
                f"{heads} 头读法（缩放 1/√{checked.shape.hidden // heads}）",
                f"作用在第 {layer} 层的输入上（单头路径重放到那一层）",
                "与整条链的单头前向**不是**同一条前向：多头会改变投影划分与缩放",
            ),
        )
        for head in range(heads)
    )


def single_head_matches(
    params: VariantParameters,
    inputs: Matrix,
    *,
    layer: int = 0,
    source: Matrix | None = None,
) -> bool:
    """``heads=1`` 的多头读法与模型真实权重是否**逐位**相同（跨口径的那条接缝）.

    只对**单流变体**有定义：``encoder_decoder`` 上模型真实权重那几条记录按
    "编码器 i · 头 0" / "解码器 i · 头 0" 命名，而多头读法作用在编码器那一流上——
    两套命名不同，因此这里显式拒绝，而不是"取第 layer 条碰碰运气"。
    """
    checked = _checked_params(params)
    if checked.variant == VARIANT_ENCODER_DECODER:
        raise AssemblyError(
            "single_head_matches 只对单流变体有定义：encoder_decoder 上模型真实权重"
            "分属两条流（编码器 / 解码器），而多头读法只作用在编码器那一流上。"
        )
    if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer < checked.shape.layers:
        raise ParameterError(
            f"层号必须是 0..{checked.shape.layers - 1} 的整数，收到 {layer!r}。"
        )
    records = head_records(checked, inputs, layer=layer, heads=1, source=source)
    baseline = self_records(checked, inputs, source=source)
    left = records[0].weights
    right = baseline[layer].weights
    if matrix_shape(left) != matrix_shape(right):
        return False
    return all(
        a == b
        for left_row, right_row in zip(left, right, strict=True)
        for a, b in zip(left_row, right_row, strict=True)
    )


def records_of_layer(
    records: Sequence[AttentionRecord], layer: int
) -> tuple[AttentionRecord, ...]:
    """同一个层的所有头（按标签里的"层 i"匹配）."""
    if isinstance(layer, bool) or not isinstance(layer, int):
        raise ParameterError(f"层号必须是整数，收到 {layer!r}。")
    matched = tuple(
        record for record in records if record.label.startswith(f"{LABEL_LAYER} {layer} ·")
    )
    if not matched:
        raise ParameterError(
            f"记录里没有层 {layer}：已有 {sorted({record.label for record in records})}。"
        )
    return matched


def record_of(records: Sequence[AttentionRecord], label: str) -> AttentionRecord:
    """按标签取一条（标签不认识时当场拒绝）."""
    for record in records:
        if record.label == label:
            return record
    raise ParameterError(
        f"记录里没有 {label!r}：已有 {sorted(record.label for record in records)}。"
    )


def aggregate_heads(records: Sequence[AttentionRecord]) -> AttentionRecord:
    """把同一个层的几个头**按逐格平均**合成一条（滚动要用它）.

    逐格平均而不是"取第 0 头"：后者会让"层与层之间换了哪一头"变成噪声。
    平均之后每一行仍然是分布（``α·Σ + (1−α)`` 的凸组合性质），
    而这一点被测试钉住（行和仍为 1）。
    """
    resolved = tuple(records)
    if not resolved:
        raise ParameterError("合成需要至少一条记录。")
    first = resolved[0]
    rows, columns = first.rows, first.columns
    for index, record in enumerate(resolved):
        if (record.rows, record.columns) != (rows, columns):
            raise ShapeError(
                f"第 {index} 条记录的形状 ({record.rows}, {record.columns}) 与第 1 条 "
                f"({rows}, {columns}) 不一致：只有同形状的几张表才能逐格平均。"
            )
        if record.mask != first.mask:
            raise AssemblyError(
                f"第 {index} 条记录的掩码与第 1 条不同：不同掩码的两张表逐格平均之后，"
                "'哪一格被挡住'这件事就没有意义了。"
            )
    count = len(resolved)
    averaged = tuple(
        tuple(
            sum(record.weights[row][column] for record in resolved) / count
            for column in range(columns)
        )
        for row in range(rows)
    )
    return replace(
        first,
        label=f"{first.label.split(' ·')[0]} · 平均 {count} 头",
        weights=averaged,
        notes=("逐格平均（每一行仍然是分布）",),
    )


def layer_inputs(
    params: VariantParameters,
    inputs: Matrix,
    *,
    source: Matrix | None = None,
    placement: str = NORM_PRE,
) -> tuple[Matrix, ...]:
    """每一层"第一个子层看到的输入"（pre-LN 是 ``LN(x)``、post 是 ``x``）.

    它对每一层各输出一份矩阵，用于核对"层与层之间表示变了多少"——
    而它按 day082 的同一条口径算（pre-LN 先过一次 LN），并且**复用同一个重放**
    （:func:`_stream_input_at_layer`）：两处若各写一份，"哪一层读的是哪一份输入"
    会在某一天悄悄分开，而那时两张表都还是行随机的。
    """
    checked = _checked_params(params)
    resolved_placement = _checked_placement(placement)
    checked_inputs = validate_matrix(inputs, name="inputs")
    out: list[Matrix] = []
    for index in range(checked.shape.layers):
        current = _stream_input_at_layer(
            checked, checked_inputs, index, source=source, placement=resolved_placement
        )
        if resolved_placement == NORM_PRE:
            block_params = checked.blocks.blocks[index]
            normed, _cache = layer_norm(
                current, gamma=block_params.norm1_gamma, beta=block_params.norm1_beta
            )
        else:
            normed = current
        out.append(normed)
    return tuple(out)


def summary_lines(records: Sequence[AttentionRecord]) -> tuple[str, ...]:
    """逐条一行（演示脚本直接印它）."""
    return tuple(record.summary_line() for record in records)


def layer_count(records: Sequence[AttentionRecord]) -> int:
    """记录涉及多少个"层"（按标签去重）."""
    layers: set[str] = set()
    for record in records:
        prefix = record.label.split(" ·")[0]
        layers.add(prefix)
    return len(layers)


def validate_heads(shape_hidden: int, heads: int) -> int:
    """头数的唯一校验入口（隐藏维必须能被整除）."""
    if isinstance(shape_hidden, bool) or not isinstance(shape_hidden, int) or shape_hidden < 1:
        raise ParameterError(f"隐藏维必须是 >= 1 的整数，收到 {shape_hidden!r}。")
    if isinstance(heads, bool) or not isinstance(heads, int) or heads < 1:
        raise ParameterError(f"heads 必须是 >= 1 的整数，收到 {heads!r}。")
    if shape_hidden % heads != 0:
        raise ParameterError(
            f"隐藏维 {shape_hidden} 不能被 heads={heads} 整除："
            "多头把 d 切成等份，因此每一份的长度必须相等。"
        )
    return heads


def variant_of(params: Any) -> str:
    """取变体名（顺手做一次校验：这一步让"拼错的变体名"在这里就失败）."""
    return validate_variant(_checked_params(params).variant)


__all__ = [
    "LABEL_CROSS",
    "LABEL_DECODER",
    "LABEL_ENCODER",
    "LABEL_LAYER",
    "aggregate_heads",
    "cross_records",
    "full_mask",
    "head_records",
    "layer_count",
    "layer_inputs",
    "record_of",
    "records_of_layer",
    "self_records",
    "self_stream_records",
    "single_head_matches",
    "summary_lines",
    "validate_heads",
    "variant_of",
]
