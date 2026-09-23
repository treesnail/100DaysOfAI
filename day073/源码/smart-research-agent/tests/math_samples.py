"""``math_foundations`` 的共享样本：固定的向量、矩阵、分布与注意力输入.

三个约定（与 ``rag_ops_samples`` / ``rag_debug_samples`` 同一取向）：

```text
① 全部离线      纯 Python 算术，零依赖、零网络，任何环境都能跑
② 输入写死      不用随机输入（随机输入会让"这次对上了"不可复现）
③ 期望值手算    关键断言给出**手算结果**（例如 cos(0°) = 1），而不是"跑出来的值"
```

第 ③ 条是这一课与工程课最大的差别：工程课的期望值来自"上一次跑通的结果"，
而数学课的期望值应当能在一张纸上推出来——**测试的价值正在于它独立于实现**。
"""

from __future__ import annotations

import math

from smart_research_agent.math_foundations import (
    Distribution,
    JointTable,
    Matrix,
    Vector,
    sample_outputs,
)

#: 3 维单位基向量（正交、长度 1——手算余弦最方便的一组）。
E1: Vector = (1.0, 0.0, 0.0)
E2: Vector = (0.0, 1.0, 0.0)
E3: Vector = (0.0, 0.0, 1.0)

#: 一般向量（长度 √14 ≈ 3.741657）。
A3: Vector = (1.0, 2.0, 3.0)

#: 与 A3 同向的放大版本（余弦应为 1，点积会变）。
A3_SCALED: Vector = (2.0, 4.0, 6.0)

#: 与 A3 正交的非零向量（余弦应为 0：1·0 + 2·3 + 3·(−2) = 0）。
A3_ORTHOGONAL: Vector = (0.0, 3.0, -2.0)

#: 一个 2×2 的对角矩阵（奇异值就是对角线上的两个数，手算可验证）。
DIAGONAL_2X2: Matrix = ((3.0, 0.0), (0.0, 2.0))

#: 3×3 的对角矩阵（奇异值 2、1、0.5；秩 2 近似应当只剩前两个方向）。
DIAGONAL_3X3: Matrix = ((2.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.5))

#: 一个 2×3 的非方阵（用于"转置、乘法、非方阵拒绝"三条判据）。
RECT_2X3: Matrix = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))

#: 注意力用的 Q / K / V（3 个 query、3 个 key、每头 4 维 / 2 维）。
#: **直接取自** ``bridge.sample_outputs()``（生产代码里那一份），而不是在这里再写一份：
#: 两处各写一份的后果是"测试验的那组输入"与"演示用的那组输入"悄悄分家。
QUERIES, KEYS, VALUES = sample_outputs()


def uniform(size: int) -> Vector:
    """``size`` 个等可能事件的概率（每个 ``1/size``）."""
    return tuple(1.0 / size for _ in range(size))


def one_hot(index: int, size: int) -> Vector:
    """独热分布（第 ``index`` 位是 1）."""
    return tuple(1.0 if position == index else 0.0 for position in range(size))


def distribution(probabilities: Vector | None = None, *, name: str = "sample") -> Distribution:
    """造一个分布（缺省是 3 个事件的均匀分布）."""
    resolved = probabilities if probabilities is not None else uniform(3)
    return Distribution(probabilities=resolved, name=name)


def joint(
    cells: Matrix | None = None,
    *,
    names_a: tuple[str, ...] = ("下雨", "不下雨"),
    names_b: tuple[str, ...] = ("带伞", "不带伞"),
) -> JointTable:
    """造一张联合分布表（缺省是演示脚本里那张 2×2 的"下雨与带伞"）.

    它**不**独立（互信息 0.0863 nats，最大偏差 0.1），因此适合用来验证
    条件概率、贝叶斯后验与互信息这三件事真的算了东西。
    """
    resolved = cells if cells is not None else ((0.30, 0.10), (0.20, 0.40))
    return JointTable(cells=resolved, names_a=names_a, names_b=names_b)


def independent_joint() -> JointTable:
    """造一张**独立**的联合分布表（互信息应为 0）.

    取 ``P(A) = (0.4, 0.6)``、``P(B) = (0.25, 0.75)``，外积得到
    ``((0.1, 0.3), (0.15, 0.45))``——每一格都恰好等于两个边缘之积。
    """
    return JointTable(
        cells=((0.10, 0.30), (0.15, 0.45)),
        names_a=("A1", "A2"),
        names_b=("B1", "B2"),
    )


def approx(left: float, right: float, *, tolerance: float = 1e-9) -> bool:
    """手算期望值用的比较（相对 + 绝对混合，与 ``types.close`` 同口径）."""
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def approx_vector(left: Vector, right: Vector, *, tolerance: float = 1e-9) -> bool:
    """逐分量比较两个向量（长度不同直接 False）."""
    if len(left) != len(right):
        return False
    return all(approx(x, y, tolerance=tolerance) for x, y in zip(left, right))


#: 手算过的常量，多处断言共用（避免每个测试各写一遍魔数）。
SQRT_14 = math.sqrt(14.0)
SQRT_3 = math.sqrt(3.0)
LN_3 = math.log(3.0)


__all__ = [
    "A3",
    "A3_ORTHOGONAL",
    "A3_SCALED",
    "DIAGONAL_2X2",
    "DIAGONAL_3X3",
    "E1",
    "E2",
    "E3",
    "KEYS",
    "LN_3",
    "QUERIES",
    "RECT_2X3",
    "SQRT_3",
    "SQRT_14",
    "VALUES",
    "approx",
    "approx_vector",
    "distribution",
    "independent_joint",
    "joint",
    "one_hot",
    "uniform",
]
