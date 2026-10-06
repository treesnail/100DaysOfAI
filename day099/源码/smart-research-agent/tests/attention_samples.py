"""``transformer_core`` 的共享样本：写死的参数、输入与目标（day075 / M7-D1）.

三个约定（与 ``math_samples`` / ``rag_ops_samples`` 同一取向）：

```text
① 全部离线      纯 Python 算术，零依赖、零网络，任何环境都能跑
② 输入写死      不用随机输入（随机输入会让"这次对上了"不可复现）
③ 期望值手算    关键断言给出**手算结果**（例如 softmax([0, 1/√2]) 的两个分量）
```

第 ③ 条在这一课格外重要：**梯度校验的期望值来自推导，而不是来自上一次跑出来的值**。
"上一次跑出来的值"是实现的影子——它会在实现写错时跟着一起错。
"""

from __future__ import annotations

import math

from smart_research_agent.transformer_core.train import (
    InductionTask,
    make_induction_batch,
)
from smart_research_agent.transformer_core.types import AttentionParams

#: 两个正交的 one-hot token（d = 2）——用来手算整条前向.
ONE_HOT_INPUTS: tuple[tuple[float, ...], ...] = ((1.0, 0.0), (0.0, 1.0))

#: 三个一般的输入行（d = 3），行列都不正交——用来验证"不是碰巧对"的场景.
SMALL_INPUTS: tuple[tuple[float, ...], ...] = (
    (1.0, 0.0, 0.5),
    (0.0, 1.0, 0.25),
    (0.5, 0.5, 0.0),
)

#: 与 ``SMALL_INPUTS`` 配套的目标（三个单位向量）.
SMALL_TARGET: tuple[tuple[float, ...], ...] = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

#: 缩放系数 ``1/√2``（手算 softmax 时反复用到）.
SCALE_SQRT2 = 1.0 / math.sqrt(2.0)

#: ``softmax([0, 1/√2])`` 的第一个分量 ``1/(1+e^{1/√2})``——手算的锚点.
P_SQRT2 = 1.0 / (1.0 + math.exp(SCALE_SQRT2))


def identity_parameters(size: int) -> AttentionParams:
    """四个投影都是单位矩阵（**最容易手算的配置**）.

    此时 ``Q = K = V = x``，于是
    ``raw = x·xᵀ``（不同 token 之间是 0）、``context`` 是输入的凸组合、
    ``output = context``——每一项都能在纸上算出来。
    """
    identity = tuple(
        tuple(1.0 if row == column else 0.0 for column in range(size))
        for row in range(size)
    )
    return AttentionParams(
        w_query=identity, w_key=identity, w_value=identity, w_output=identity
    )


def scaled_identity(value: float, size: int) -> tuple[tuple[float, ...], ...]:
    """``value · I``（用于构造"只改一个矩阵"的对照）."""
    return tuple(
        tuple(value if row == column else 0.0 for column in range(size))
        for row in range(size)
    )


def small_parameters() -> AttentionParams:
    """三个手算过的 3×3 矩阵：``W_q = 0.5I``、``W_k`` 与 ``W_v = I``、``W_o = I``."""
    return AttentionParams(
        w_query=scaled_identity(0.5, 3),
        w_key=(
            (0.4, 0.1, 0.0),
            (0.0, 0.4, 0.1),
            (0.1, 0.0, 0.4),
        ),
        w_value=scaled_identity(1.0, 3),
        w_output=scaled_identity(1.0, 3),
    )


def toy_parameters(size: int = 6, *, seed: int = 7) -> AttentionParams:
    """确定性的小随机参数（调用 ``train.default_parameters``，**不另写一份**）.

    与 day073 的 ``QUERIES/KEYS/VALUES`` 同一条纪律：
    输入只在一个地方生成，避免"测试验的那组"与"演示用的那组"悄悄分家。
    """
    from smart_research_agent.transformer_core.train import default_parameters

    return default_parameters(size, seed=seed)


def induction_tasks(count: int = 4, *, seed: int = 42) -> tuple[InductionTask, ...]:
    """一批 induction 样本（同样来自生产代码里的构造函数）."""
    return make_induction_batch(count, seed=seed)


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


__all__ = [
    "ONE_HOT_INPUTS",
    "P_SQRT2",
    "SCALE_SQRT2",
    "SMALL_INPUTS",
    "SMALL_TARGET",
    "approx",
    "approx_matrix",
    "identity_parameters",
    "induction_tasks",
    "scaled_identity",
    "small_parameters",
    "toy_parameters",
]
