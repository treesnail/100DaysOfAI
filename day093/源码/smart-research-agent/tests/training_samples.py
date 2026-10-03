"""``training_optim`` 的共用样本（day081）.

与 day073~080 的样本模块同一条纪律：**所有测试共用同一组固定样本**，
因此"某一条测试失败"与"样本变了"两件事不会混在一起。

样本与 ``scripts/training_optim_demo.py`` **逐位同值**，因此演示里的每一个读数
都能在测试里被复核。
"""

from __future__ import annotations

from smart_research_agent.encoder_decoder.types import NORM_POST, NORM_PRE
from smart_research_agent.math_foundations.types import Matrix
from smart_research_agent.transformer_stack import make_shape, make_stack_parameters
from smart_research_agent.training_optim import TrainingConfig, initialize_parameters

#: 样本的四个维度（与 day073~080 的 d=6 / d_ff=24 / n=4 对齐，便于跨天对照）.
HIDDEN = 6
FFN_RATIO = 4
TOKENS = 4
LAYERS = 4
SEED = 7

#: 样本的学习率与步数（演示脚本用的是同一对值）.
LEARNING_RATE = 0.1
STEPS = 40
WARMUP_STEPS = 5

#: 两个摆放位置（测试里成对地遍历它们）.
PLACEMENTS: tuple[str, ...] = (NORM_PRE, NORM_POST)


def shape(layers: int = LAYERS, hidden: int = HIDDEN, tokens: int = TOKENS):
    """样本形状（默认 4 层 / d=6 / n=4 / d_ff=4d）."""
    return make_shape(layers=layers, hidden=hidden, tokens=tokens, ffn_ratio=FFN_RATIO)


def inputs(tokens: int = TOKENS, hidden: int = HIDDEN) -> Matrix:
    """样本输入（与演示脚本同值）."""
    return tuple(
        tuple(0.1 * (row + 1) - 0.05 * (column + 1) for column in range(hidden))
        for row in range(tokens)
    )


def target(tokens: int = TOKENS, hidden: int = HIDDEN) -> Matrix:
    """样本目标（与演示脚本同值）."""
    return tuple(
        tuple(0.2 - 0.03 * (row * hidden + column) for column in range(hidden))
        for row in range(tokens)
    )


def config(**changes) -> TrainingConfig:
    """样本配置（一次只改一个旋钮时用它 ``replaced``）."""
    base = TrainingConfig(learning_rate=LEARNING_RATE, steps=STEPS, seed=SEED)
    return base if not changes else base.replaced(**changes)


def params(scheme: str = "uniform", seed: int = SEED):
    """样本参数（默认与 day080 的 ``make_stack_parameters`` 逐位一致）."""
    return initialize_parameters(shape(), scheme=scheme, seed=seed)


def day080_params(seed: int = SEED):
    """day080 的那一摞参数（用来核对"uniform 方案与它逐位一致"）."""
    return make_stack_parameters(shape(), seed=seed)


def matrix_shape_of(matrix: Matrix) -> tuple[int, int]:
    """矩阵形状（让测试读起来短一些）."""
    return (len(matrix), len(matrix[0]))


def norm_of(matrix: Matrix) -> float:
    """Frobenius 范数（测试里核对读数用）."""
    return sum(value * value for row in matrix for value in row) ** 0.5
