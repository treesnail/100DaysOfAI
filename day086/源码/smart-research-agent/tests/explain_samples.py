"""``explainability`` 的共用样本：参数、输入与几条**手算过的**权重表（day083）.

两条纪律与前八天的 ``*_samples.py`` 相同：

```text
① 样本可复现：参数 / 输入 / 源序列各一颗种子（换模型与换数据不混）
② 手算样本进样本模块：几张**能一眼算清**的表（均匀、独热、两个头完全相同）
   ——读数的判据要落在能被手算复核的数上，而不是"跑出来的那个数"
```
"""

from __future__ import annotations

from typing import Sequence

from smart_research_agent.arch_variants.stacks import (
    make_variant_parameters,
    make_variant_shape,
    sample_matrix,
)
from smart_research_agent.arch_variants.types import (
    VARIANT_DECODER_ONLY,
    VARIANT_ENCODER_DECODER,
    VARIANT_ENCODER_ONLY,
    VARIANTS,
    VariantParameters,
    VariantShape,
)
from smart_research_agent.explainability import extract
from smart_research_agent.explainability.types import (
    STREAM_CROSS,
    STREAM_SELF,
    AttentionRecord,
    causal_mask,
    full_mask,
)

#: 与 day082 的样本同值（跨天对照）。
DEFAULT_TOKENS = 4
DEFAULT_SOURCES = 5
DEFAULT_HIDDEN = 6
DEFAULT_LAYERS = 3

PARAMETER_SEED = 7
INPUT_SEED = 21
SOURCE_SEED = 13


def sample_shape(
    *,
    tokens: int = DEFAULT_TOKENS,
    hidden: int = DEFAULT_HIDDEN,
    layers: int = DEFAULT_LAYERS,
    sources: int = DEFAULT_SOURCES,
) -> VariantShape:
    """样本形状（``d = 6`` 时 2 头、每头 3 维：能整除）."""
    return make_variant_shape(tokens=tokens, hidden=hidden, layers=layers, sources=sources)


def sample_inputs(shape: VariantShape | None = None):
    """样本输入."""
    resolved = shape if shape is not None else sample_shape()
    return sample_matrix(resolved.tokens, resolved.hidden, seed=INPUT_SEED)


def sample_source(shape: VariantShape | None = None):
    """样本源序列."""
    resolved = shape if shape is not None else sample_shape()
    return sample_matrix(resolved.source_length, resolved.hidden, seed=SOURCE_SEED)


def sample_params(variant: str = VARIANT_DECODER_ONLY, shape: VariantShape | None = None):
    """某个变体的参数."""
    resolved = shape if shape is not None else sample_shape()
    return make_variant_parameters(resolved, variant, seed=PARAMETER_SEED)


def sample_self_records(variant: str = VARIANT_DECODER_ONLY, shape: VariantShape | None = None):
    """模型真实的自注意力记录（``encoder_decoder`` 会自动带上源序列）."""
    resolved = shape if shape is not None else sample_shape()
    params = sample_params(variant, resolved)
    if variant == VARIANT_ENCODER_DECODER:
        return extract.self_records(params, sample_inputs(resolved), source=sample_source(resolved))
    return extract.self_records(params, sample_inputs(resolved))


def sample_head_records(
    variant: str = VARIANT_DECODER_ONLY,
    shape: VariantShape | None = None,
    *,
    layer: int = 0,
    heads: int = 2,
):
    """多头读法（默认 2 头、第 0 层）."""
    resolved = shape if shape is not None else sample_shape()
    params = sample_params(variant, resolved)
    if variant == VARIANT_ENCODER_DECODER:
        return extract.head_records(
            params, sample_inputs(resolved), layer=layer, heads=heads,
            source=sample_source(resolved),
        )
    return extract.head_records(params, sample_inputs(resolved), layer=layer, heads=heads)


def sample_layer_records(
    variant: str = VARIANT_DECODER_ONLY, shape: VariantShape | None = None, *, heads: int = 2
):
    """逐层记录（标签是"层 i · 头 h"，可以直接进滚动）."""
    from smart_research_agent.explainability.study import all_layer_records

    resolved = shape if shape is not None else sample_shape()
    params = sample_params(variant, resolved)
    source = sample_source(resolved) if variant == VARIANT_ENCODER_DECODER else None
    return all_layer_records(params, sample_inputs(resolved), heads=heads, source=source)


def uniform_record(
    size: int = 3, *, label: str = "均匀", mask=None, stream: str = STREAM_SELF
) -> AttentionRecord:
    """**手算样本**：每一行完全均匀（每一格 ``1/size``）.

    它的读数是可手算的：熵 = ln(size)、Frobenius = 1（因为 ``Σ(1/n)² × n² = 1``）、
    对角质量 = ``1/size``、支撑集 = 全部格子。
    """
    weights = tuple(tuple(1.0 / size for _ in range(size)) for _ in range(size))
    resolved_mask = mask if mask is not None else full_mask(size)
    return AttentionRecord(
        label=label, weights=weights, mask=resolved_mask, stream=stream
    )


def one_hot_record(
    size: int = 3, *, label: str = "独热", mask=None
) -> AttentionRecord:
    """**手算样本**：每一行把全部质量放在对角线上（``口口口`` → 熵 0）.

    熵 = 0、Frobenius = √size、对角质量 = 1、支撑集 = size。
    """
    weights = tuple(
        tuple(1.0 if row == column else 0.0 for column in range(size))
        for row in range(size)
    )
    resolved_mask = mask if mask is not None else full_mask(size)
    return AttentionRecord(
        label=label, weights=weights, mask=resolved_mask, stream=STREAM_SELF
    )


def causal_uniform_record(size: int = 4, *, label: str = "因果均匀") -> AttentionRecord:
    """**手算样本**：因果掩码下每一行在允许的位置上均匀.

    第 i 行有 ``i + 1`` 个允许的位置 ⇒ 熵 = ``(1/n)Σln(i+1)``（正好贴到天花板）。
    """
    mask = causal_mask(size)
    weights = tuple(
        tuple(
            (1.0 / (row + 1)) if column <= row else 0.0
            for column in range(size)
        )
        for row in range(size)
    )
    return AttentionRecord(label=label, weights=weights, mask=mask)


def cross_record(
    rows: int = 3, columns: int = 5, *, label: str = "交叉均匀"
) -> AttentionRecord:
    """一张长方形的交叉记录（每一行在全部源位置上均匀）."""
    weights = tuple(tuple(1.0 / columns for _ in range(columns)) for _ in range(rows))
    mask = tuple(tuple(True for _ in range(columns)) for _ in range(rows))
    return AttentionRecord(
        label=label, weights=weights, mask=mask, stream=STREAM_CROSS
    )


def near_uniform_record(
    size: int = 3, *, label: str = "近似均匀", epsilon: float = 1e-3
) -> AttentionRecord:
    """**手算样本**：均匀 + 对角线上多一点点（支撑集仍是全部格子）.

    ``weights = (1 − ε)/size + ε``（对角线）与 ``(1 − ε)/size``（其他），
    它的熵严格小于 ``ln size``（**更尖**），因此可以用来看归一化熵的方向。
    """
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon 必须落在 [0, 1]")
    base = (1.0 - epsilon) / size
    weights = tuple(
        tuple(base + (epsilon if row == column else 0.0) for column in range(size))
        for row in range(size)
    )
    return AttentionRecord(
        label=label, weights=weights, mask=full_mask(size)
    )


def all_variant_labels() -> Sequence[str]:
    """三个变体名（供参数化测试）."""
    return tuple(VARIANTS)


def encoder_only_params() -> VariantParameters:
    """``encoder_only`` 的参数（写在这里，省得每个测试各写一遍）."""
    return sample_params(VARIANT_ENCODER_ONLY)
