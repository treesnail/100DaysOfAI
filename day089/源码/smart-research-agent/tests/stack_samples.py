"""``transformer_stack`` 的共用样本（day080）.

与 day073~079 的样本模块同一条纪律：**所有测试共用同一组固定样本**，
因此“某一条测试失败”与“样本变了”两件事不会混在一起。
"""

from __future__ import annotations

from typing import Sequence

from smart_research_agent.encoder_decoder.types import NORM_POST, NORM_PRE
from smart_research_agent.math_foundations.types import Matrix, matrix_shape
from smart_research_agent.transformer_stack.assembly import assembly_script
from smart_research_agent.transformer_stack.layers import (
    make_shape,
    make_stack_parameters,
)
from smart_research_agent.transformer_stack.types import StackParameters, StackShape

#: 样本的四个维度（与 day073~079 的 d=6 / d_ff=24 / n=4 对齐，便于跨天对照）.
HIDDEN = 6
FFN_RATIO = 4
TOKENS = 4
LAYERS = 4

#: 两个摆放位置（测试里成对地遍历它们）.
PLACEMENTS: tuple[str, ...] = (NORM_PRE, NORM_POST)


def shape(
    *,
    layers: int = LAYERS,
    hidden: int = HIDDEN,
    tokens: int = TOKENS,
    ffn_ratio: int = FFN_RATIO,
) -> StackShape:
    """样本形状（默认 4 层 / d=6 / n=4 / d_ff=4d）."""
    return make_shape(layers=layers, hidden=hidden, tokens=tokens, ffn_ratio=ffn_ratio)


def parameters(
    *,
    layers: int = LAYERS,
    hidden: int = HIDDEN,
    tokens: int = TOKENS,
    seed: int = 7,
    scale: float = 0.25,
) -> StackParameters:
    """样本参数（第 i 层用 ``seed + i*13`` / ``seed + i*17`` 两份种子）."""
    return make_stack_parameters(
        shape(layers=layers, hidden=hidden, tokens=tokens), seed=seed, scale=scale
    )


def inputs(*, tokens: int = TOKENS, hidden: int = HIDDEN) -> Matrix:
    """样本输入：确定性的小矩阵（与 ``study._sample_inputs`` 同值）."""
    return tuple(
        tuple(0.1 * (row + 1) - 0.05 * (column + 1) for column in range(hidden))
        for row in range(tokens)
    )


def target(*, tokens: int = TOKENS, hidden: int = HIDDEN) -> Matrix:
    """样本目标：确定性的小矩阵（**固定**，所有配置共用同一个）."""
    return tuple(
        tuple(0.2 - 0.03 * (row * hidden + column) for column in range(hidden))
        for row in range(tokens)
    )


def script(
    *,
    layers: int = LAYERS,
    hidden: int = HIDDEN,
    tokens: int = TOKENS,
    placement: str = NORM_PRE,
    activation: str = "relu",
    num_heads: int = 2,
) -> str:
    """样本 PyTorch 组装脚本."""
    return assembly_script(
        shape(layers=layers, hidden=hidden, tokens=tokens),
        placement=placement,
        activation=activation,
        num_heads=num_heads,
    )


def matrix_norm(matrix: Matrix) -> float:
    """样本里常用的范数（避免测试里到处手写平方和）."""
    return sum(value * value for row in matrix for value in row) ** 0.5


def shape_of(matrix: Matrix) -> tuple[int, int]:
    """矩阵形状（转发 ``matrix_shape``，让测试读起来短一些）."""
    return matrix_shape(matrix)


def flat(values: Sequence[float]) -> tuple[float, ...]:
    """把一串数转成 tuple（比对时用）。"""
    return tuple(float(item) for item in values)
