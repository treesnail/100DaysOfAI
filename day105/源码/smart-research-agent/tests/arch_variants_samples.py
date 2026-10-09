"""``arch_variants`` 的共用样本：形状、参数、输入与脚本（day082）.

与前几天的 ``*_samples.py`` 同一条纪律：**样本本身要可复现**——
参数、输入、源序列、目标各自一颗种子，于是“换一个旋钮”与“换一次随机性”
不会混在一起（第 8 章四组实验的对照就靠这件事）。
"""

from __future__ import annotations

from smart_research_agent.arch_variants.stacks import (
    make_variant_parameters,
    make_variant_shape,
    sample_matrix,
    variant_forward,
)
from smart_research_agent.arch_variants.types import (
    VARIANT_ENCODER_DECODER,
    VARIANTS,
    VariantForward,
    VariantParameters,
    VariantShape,
)

#: 默认样本的形状（目标 4 个 token、源 5 个、d=6、d_ff=24、3 层）.
DEFAULT_TOKENS = 4
DEFAULT_SOURCES = 5
DEFAULT_HIDDEN = 6
DEFAULT_LAYERS = 3

#: 三颗互不相干的种子（参数 / 输入 / 源序列）.
PARAMETER_SEED = 7
INPUT_SEED = 21
SOURCE_SEED = 17
TARGET_SEED = 31


def sample_shape(
    *,
    tokens: int = DEFAULT_TOKENS,
    hidden: int = DEFAULT_HIDDEN,
    layers: int = DEFAULT_LAYERS,
    sources: int = DEFAULT_SOURCES,
) -> VariantShape:
    """样本形状（``ffn`` 由 4 倍关系给出，与原论文同口径）."""
    return make_variant_shape(tokens=tokens, hidden=hidden, layers=layers, sources=sources)


def sample_inputs(shape: VariantShape | None = None, *, seed: int = INPUT_SEED):
    """目标流（或唯一那条流）的输入样本."""
    resolved = shape if shape is not None else sample_shape()
    return sample_matrix(resolved.tokens, resolved.hidden, seed=seed)


def sample_source(shape: VariantShape | None = None, *, seed: int = SOURCE_SEED):
    """源序列样本."""
    resolved = shape if shape is not None else sample_shape()
    return sample_matrix(resolved.source_length, resolved.hidden, seed=seed)


def sample_target(shape: VariantShape | None = None, *, seed: int = TARGET_SEED):
    """目标样本（与输入**不同**：否则“损失在下降”这件事看不出来）."""
    resolved = shape if shape is not None else sample_shape()
    return sample_matrix(resolved.tokens, resolved.hidden, seed=seed)


def sample_parameters(
    variant: str,
    shape: VariantShape | None = None,
    *,
    seed: int = PARAMETER_SEED,
) -> VariantParameters:
    """某个变体的全套参数."""
    resolved = shape if shape is not None else sample_shape()
    return make_variant_parameters(resolved, variant, seed=seed)


def sample_forward(variant: str, shape: VariantShape | None = None, **kwargs):
    """某个变体的一次前向（``encoder_decoder`` 会自动带上源序列）."""
    resolved = shape if shape is not None else sample_shape()
    params = sample_parameters(variant, resolved)
    inputs = sample_inputs(resolved)
    if variant == VARIANT_ENCODER_DECODER:
        kwargs.setdefault("source", sample_source(resolved))
    return variant_forward(params, inputs, **kwargs)


def all_sample_forwards(**kwargs) -> dict[str, VariantForward]:
    """三个变体各跑一次前向（用来核对“同一份配置下三个都能跑”）."""
    resolved = sample_shape()
    return {variant: sample_forward(variant, resolved, **kwargs) for variant in VARIANTS}
