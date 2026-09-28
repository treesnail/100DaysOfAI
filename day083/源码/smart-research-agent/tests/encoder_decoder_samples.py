"""``encoder_decoder`` 的共享样本：写死的形状、输入与手算锚点（day079 / M7-D4）.

三条约定（与 ``position_samples`` / ``multihead_samples`` 逐字相同）：

```text
① 全部离线      纯 Python 算术，零依赖、零网络，任何环境都能跑
② 输入写死      不用随机输入（除非种子钉住），随机输入会让"这次对上了"不可复现
③ 期望值手算    关键断言给出**手算结果**，而不是"上次跑出来的值"
```

本模块只做两件事：**把生产函数的入参写死**、**把锚点写成常量**。
工厂函数一律**委托生产函数**（``make_block_parameters`` / ``default_parameters``），
绝不另写一份公式——两份公式迟早在某一处悄悄分家。

一个必须写下来的手算锚点：``x = (1, 2, 3, 4)`` 的 LayerNorm。

```text
μ = 2.5、σ² = 1.25            （四个偏差是 −1.5 / −0.5 / 0.5 / 1.5）
x̂ = (x − μ)/√(σ² + eps)      eps → 0 时是 (−1.341641, −0.447214, 0.447214, 1.341641)
而 eps = 1e-5 让"x̂ 的方差"变成 σ²/(σ² + eps) = 0.999992 → 离 1 有 8.00e-06
```

后一句话说明：**"方差离 1 多远" 是 eps 的定义，不是误差**（``check_rows_are_standardized``
对每一行分别算期望值 ``σ²/(σ²+eps)`` 再比，正是为了不把这件事说成 bug）。
"""

from __future__ import annotations

from smart_research_agent.encoder_decoder import (
    BlockParameters,
    BlockShape,
    CrossParameters,
    make_block_parameters,
)
from smart_research_agent.math_foundations.types import Matrix
from smart_research_agent.transformer_core.train import default_parameters
from smart_research_agent.transformer_core.types import AttentionParams

#: 隐藏维 ``d``（与 day075 的小样本一致，四个投影是 6×6）.
HIDDEN = 6

#: 前馈的中间维 ``d_ff = 4d``（原论文的扩张比）.
FFN = 24

#: 编码器块的行数（一个序列 4 个 token）.
TOKENS = 4

#: 交叉注意力的源序列长度（**与 target 不同**，于是权重是长方形）.
SOURCE_TOKENS = 6

#: 解码器块的行数（**可以**与编码器输出不同宽——只要求隐藏维相同）.
DECODER_TOKENS = 3

#: 参数初始化的种子与幅度（与 ``make_block_parameters`` 的缺省值一致）.
SEED = 7
SCALE = 0.25

#: 手算 LayerNorm 的那一行输入 ``x = (1, 2, 3, 4)``.
HAND_ROW: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)

#: 上面那一行的手算均值与（有偏）方差.
HAND_MEAN = 2.5
HAND_VARIANCE = 1.25

#: ``eps → 0`` 的闭式标准化结果（**教科书写法**，第 6 位小数已经四舍五入）.
HAND_STANDARDIZED: tuple[float, ...] = (-1.341641, -0.447214, 0.447214, 1.341641)

#: ``eps = 1e-5`` 时 x̂ 的方差 ``σ²/(σ² + eps) = 0.999992``，因此"离 1"有这么多——**它是 eps 的定义**.
HAND_VARIANCE_DEFICIT = 8.00e-06

#: 逐位的差值：``eps`` 让 x̂ 的第 6 位小数从 1.341641 变到 1.341635（量级 5e-6）.
HAND_EPSILON_SHIFT = 5e-6

#: 残差那条 ``+1`` 路：把前馈输出层整体压到 ``1/16``（``scale = 0.0625``）.
RESIDUAL_SCALE = 0.0625

#: 实测的"有残差 / 无残差"比值（本样本）——它是第 8.7 条性质的全部说服力.
RESIDUAL_GAP = 18.2

#: 上游取全 1 时 ``‖dx‖/‖dy‖`` 的手算值（两个分支为 0 时，两条 LN 链仍然在贡献）.
RESIDUAL_IDENTITY_NORM_RATIO = 1.0

#: 参数量为 1 的 LN（``gamma=1``、``beta=0``），供"手算 LN"的用例使用.
IDENTITY_GAMMA: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
ZERO_BETA: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)


def approx(left: float, right: float, *, tolerance: float = 1e-9) -> bool:
    """手算期望值用的比较（相对 + 绝对混合，与 ``types.close`` 同口径）."""
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def approx_matrix(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
    *,
    tolerance: float = 1e-9,
) -> bool:
    """逐元素比较两个矩阵（形状不同直接 False）."""
    if len(left) != len(right):
        return False
    return all(
        len(left_row) == len(right_row)
        and all(approx(a, b, tolerance=tolerance) for a, b in zip(left_row, right_row))
        for left_row, right_row in zip(left, right)
    )


def ramp(rows: int = TOKENS, columns: int = HIDDEN, *, step: float = 0.2, drift: float = 0.1) -> Matrix:
    """写死的非平凡输入：``x[i][j] = step·(i+1) + drift·(j+1)``.

    为什么不用全 1 / one-hot：**每一行必须有非零方差**，
    否则 LayerNorm 那一行会退化成全零，而 ``self_attention`` 会正确地拒绝"全零行"
    （全零行让打分全是 0、softmax 给出均匀分布，看起来像"模型还没学到"）。
    """
    return tuple(
        tuple(step * (i + 1) + drift * (j + 1) for j in range(columns)) for i in range(rows)
    )


def block_shape(tokens: int = TOKENS) -> BlockShape:
    """写死的块形状：``hidden=6、ffn=24、tokens=4``（另一处只改行数）."""
    return BlockShape(hidden=HIDDEN, ffn=FFN, tokens=tokens)


def block_parameters(tokens: int = TOKENS) -> BlockParameters:
    """一个块的初始参数（**委托生产函数**，种子与幅度钉住）."""
    return make_block_parameters(block_shape(tokens), seed=SEED, scale=SCALE)


def attention_parameters(size: int = HIDDEN) -> AttentionParams:
    """确定性的小随机注意力参数（**调用生产代码的** ``default_parameters``）."""
    return default_parameters(size, seed=SEED, scale=SCALE)


def inputs(rows: int = TOKENS) -> Matrix:
    """梯度校验用的输入（写死的斜坡）."""
    return ramp(rows)


def target(rows: int = TOKENS, columns: int = HIDDEN, value: float = 0.4) -> Matrix:
    """梯度校验用的目标（常数矩阵——损失对每一列都非零）."""
    return tuple(tuple(value for _ in range(columns)) for _ in range(rows))


def _scaled_matrix(rows: int, rows_of_columns: int, scale: float) -> Matrix:
    """``scale·(j+1)`` 的满矩阵（每一行相同，但**行与行不必不同**）."""
    return tuple(tuple(scale * (j + 1) for j in range(rows_of_columns)) for _ in range(rows))


def cross_parameters() -> CrossParameters:
    """一般（非饱和）的交叉注意力参数：权重是 ``(TOKENS, SOURCE_TOKENS) = (4, 6)`` 的长方形."""
    return CrossParameters(
        w_query=_scaled_matrix(HIDDEN, HIDDEN, 0.02),
        w_key=_scaled_matrix(HIDDEN, HIDDEN, 0.03),
        w_value=_scaled_matrix(HIDDEN, HIDDEN, 0.04),
        w_output=_scaled_matrix(HIDDEN, HIDDEN, 0.05),
    )


def saturated_cross_parameters() -> CrossParameters:
    """**饱和**的交叉注意力参数（``10·I``）——用来把"误加因果掩码"的代价量到 ``1.00e+00``.

    饱和之后每一行的权重几乎全押在一个位置上，因此"遮掉后半段"的改动是**整行重写**，
    相对偏差达到上界 1.0（分母取 ``max(1, |ref|)`` 时，权重都在 [0, 1] 内）。
    """
    def identity(size: int, scale: float) -> Matrix:
        return tuple(tuple(scale if i == j else 0.0 for j in range(size)) for i in range(size))

    return CrossParameters(
        w_query=identity(HIDDEN, 10.0),
        w_key=identity(HIDDEN, 10.0),
        w_value=identity(HIDDEN, 10.0),
        w_output=identity(HIDDEN, 1.0),
    )


def source_inputs(rows: int = SOURCE_TOKENS) -> Matrix:
    """交叉注意力 source 那一侧的输入（写死的斜坡，行数与 target 不同）."""
    return ramp(rows, HIDDEN, step=0.2, drift=0.01)


def decoder_inputs(rows: int = DECODER_TOKENS) -> Matrix:
    """解码器块自己的输入（行数可以与编码器输出不同）."""
    return ramp(rows, HIDDEN, step=0.1, drift=0.02)


def encoder_outputs(rows: int = SOURCE_TOKENS) -> Matrix:
    """编码器的输出（与解码器**同宽**，行数可以不同）."""
    return ramp(rows, HIDDEN, step=0.3, drift=0.02)


def decoder_gammas(size: int = HIDDEN) -> tuple[tuple[float, ...], ...]:
    """解码器块三个 LN 的 γ（都是全 1）."""
    return tuple(tuple(1.0 for _ in range(size)) for _ in range(3))


def decoder_betas(size: int = HIDDEN) -> tuple[tuple[float, ...], ...]:
    """解码器块三个 LN 的 β（都是全 0）."""
    return tuple(tuple(0.0 for _ in range(size)) for _ in range(3))


def decoder_target(rows: int = DECODER_TOKENS, columns: int = HIDDEN, value: float = 0.5) -> Matrix:
    """解码器块的梯度校验目标（**行数必须等于解码器输入的行数**）."""
    return tuple(tuple(value for _ in range(columns)) for _ in range(rows))


__all__ = [
    "DECODER_TOKENS",
    "FFN",
    "HAND_EPSILON_SHIFT",
    "HAND_MEAN",
    "HAND_ROW",
    "HAND_STANDARDIZED",
    "HAND_VARIANCE",
    "HAND_VARIANCE_DEFICIT",
    "HIDDEN",
    "IDENTITY_GAMMA",
    "RESIDUAL_GAP",
    "RESIDUAL_IDENTITY_NORM_RATIO",
    "RESIDUAL_SCALE",
    "SCALE",
    "SEED",
    "SOURCE_TOKENS",
    "TOKENS",
    "ZERO_BETA",
    "approx",
    "approx_matrix",
    "attention_parameters",
    "block_parameters",
    "block_shape",
    "cross_parameters",
    "decoder_betas",
    "decoder_gammas",
    "decoder_inputs",
    "decoder_target",
    "encoder_outputs",
    "inputs",
    "ramp",
    "saturated_cross_parameters",
    "source_inputs",
    "target",
]
