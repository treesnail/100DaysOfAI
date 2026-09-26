"""``multi_head`` 的共享样本：写死的参数、输入与期望值（day076 / M7-D2）.

三条约定（与 ``attention_samples`` / ``math_samples`` 逐字相同）：

```text
① 全部离线      纯 Python 算术，零依赖、零网络，任何环境都能跑
② 输入写死      不用随机输入（随机输入会让"这次对上了"不可复现）
③ 期望值手算    关键断言给出**手算结果**，而不是"上次跑出来的值"
```

第 ③ 条在这一课多了一处必须交代的地方：**边界情形的期望值来自两条不同的路径**
（一条公式、一段算法），它们在浮点下可能差最后一位。样本里因此同时给出
"手算值"与"允许的比较方式"，并把这件事写在注释里：

```text
WITNESS_ONE_HEAD_DISTANCE   1/√2 = 0.7071067811865475（公式）
实测距离                     0.7071067811865476（点到线段距离的算法）
```

把两者写成 ``==`` 会让某一次无关的重排变成一次假失败——**该近似的地方就说近似**。
"""

from __future__ import annotations

import math

from smart_research_agent.multi_head.reachability import (
    WITNESS_HEAD_ONE_PULLS,
    WITNESS_HEAD_ZERO_PULLS,
    WITNESS_ONE_HEAD_DISTANCE,
    WITNESS_TARGET,
    WITNESS_WEIGHTS,
    designed_witness,
)
from smart_research_agent.multi_head.types import HeadPartition, MultiHeadShape
from smart_research_agent.transformer_core.train import (
    InductionTask,
    default_parameters,
    make_induction_batch,
)
from smart_research_agent.transformer_core.types import AttentionParams, AttentionShape

#: 头数取值：**全部能整除 6**（6 的因数有 1 / 2 / 3 / 6）——因此都可以作用在
#: ``default_parameters(6)`` 上。这是"同参数量对照"的实验前提。
HEAD_COUNTS: tuple[int, ...] = (1, 2, 3)

#: induction 任务的默认词表大小（与 day075 的 ``default_parameters`` 默认值一致）.
VOCABULARY = 6

#: 手算锚点 1：``softmax([0, 1/√2])`` 的第一个分量（与 day075 同一个数）.
#: 0.330238...
P_SQRT2 = 1.0 / (1.0 + math.exp(1.0 / math.sqrt(2.0)))

#: 手算锚点 2：``softmax([2/√2, 5/√2])`` 的第一个分量 = ``1/(1+e^{3/√2})``.
#: 0.107008...（差值是 3/√2 而不是 1/√2，因此这个数与锚点 1 明显不同）
P_THREE_SQRT2_HALF = 1.0 / (1.0 + math.exp(3.0 / math.sqrt(2.0)))

#: 手算配置的输入：两个 4 维 token，**两个头看到的打分差不同**（1/√2 与 3/√2）.
#: 于是"各头看到的东西不一样"这件事在这条样本上是可手算的。
#:
#: ```text
#: 第 0 头（维度 0,1）  第 1 行 raw = (0, 1)  → 打分差 1/√2
#: 第 1 头（维度 2,3）  第 1 行 raw = (2, 5)  → 打分差 3/√2
#: ```
IDENTITY_INPUTS: tuple[tuple[float, ...], ...] = (
    (1.0, 0.0, 1.0, 0.0),
    (0.0, 1.0, 2.0, 1.0),
)

#: 上面的输入在 ``identity_parameters(4)`` + ``heads=2`` 下**手算出来**的输出：
#:
#: ```text
#: 第 0 行  每一头都只看自己 → 输出 = 输入本身
#: 第 1 头  weights = (q, 1−q)、values = ((1,0), (2,1))
#:          context = q·(1,0) + (1−q)·(2,1) = (2−q, 1−q)   ← 那个 2 来自输入的第三个分量
#: 第 0 头  weights = (p, 1−p)、values = ((1,0), (0,1))
#:          context = (p, 1−p)
#: ```
#:
#: 四个数**全部由两个锚点表达**（``P_SQRT2`` 与 ``P_THREE_SQRT2_HALF``），
#: 而不是写成上一轮跑出来的小数——"上一次跑出来的值"是实现的影子。
IDENTITY_OUTPUT: tuple[tuple[float, ...], ...] = (
    (1.0, 0.0, 1.0, 0.0),
    (
        P_SQRT2,
        1.0 - P_SQRT2,
        2.0 - P_THREE_SQRT2_HALF,
        1.0 - P_THREE_SQRT2_HALF,
    ),
)

#: 手算的头间差异（两行）：
#:
#: ```text
#: 第 0 行  两份分布相同（都是 (1,0)）        → TV = 0
#: 第 1 行  两个分量差的绝对值相同            → TV = ½·2·|p − q| = |p − q|
#: ```
IDENTITY_ROW_TOTAL_VARIATIONS: tuple[float, float] = (
    0.0,
    P_SQRT2 - P_THREE_SQRT2_HALF,
)

#: 手算锚点 3/4：``softmax([0, 1])`` 与 ``softmax([2, 4])`` 的第一个分量.
#: 它们只在 ``heads=4`` 时用得上（那时 ``head_dim = 1``、缩放系数恰好是 1）——
#: **边界情形**因此也是可手算的，而不是只能"跑一遍看看"。
P_E = 1.0 / (1.0 + math.exp(1.0))
P_E_SQUARED = 1.0 / (1.0 + math.exp(2.0))

#: ``heads=4``（``head_dim = head_value = 1``）下第 1 行的**手算输出**：
#:
#: ```text
#: head 0  打分 (0, 0)    → 权重 (0.5, 0.5)、value (1,0)   → context 0.5
#: head 1  打分 (0, 1)    → 权重 (P_E, 1−P_E)、value (0,1) → context 1−P_E
#: head 2  打分 (2, 4)    → 权重 (P_E², 1−P_E²)、value (1,2) → context 2−P_E²
#: head 3  打分 (0, 1)    → 同 head 1                       → context 1−P_E
#: ```
IDENTITY_OUTPUT_FOUR_HEADS_ROW_ONE: tuple[float, ...] = (
    0.5,
    1.0 - P_E,
    2.0 - P_E_SQUARED,
    1.0 - P_E,
)

#: 见证样本的四个 pull 与目标（从生产代码里再导出一份，测试只读一份事实）.
WITNESS_ZERO_PULLS = WITNESS_HEAD_ZERO_PULLS
WITNESS_ONE_PULLS = WITNESS_HEAD_ONE_PULLS
WITNESS_DISTANCE = WITNESS_ONE_HEAD_DISTANCE
WITNESS_TARGET_2D = WITNESS_TARGET
WITNESS_DISTRIBUTIONS = WITNESS_WEIGHTS


def identity_parameters(size: int) -> AttentionParams:
    """四个投影都是单位矩阵（**最容易手算的配置**，与 day075 同构）.

    此时 ``Q = K = V = x``，于是每一头的打分就是"它那几维上输入自己的点积"，
    每一头的 context 就是输入的凸组合——每一项都能在纸上算出来。
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


def repeated_block_parameters(size: int, heads: int) -> AttentionParams:
    """**每一头完全相同**的参数（用来构造"多头退化"的对照）.

    做法是把一个 ``(size/heads, size)`` 的块**重复 heads 遍**填满四个矩阵：
    于是每一头拿到的行块逐位相同 → 每一头的打分、softmax、context 全部相同
    → 头间差异恰好是 0.0。这不是一个"病态输入"，而是一个**必须能被看见**的读数：
    多头最大的失败模式就是所有头学到同一个分布。
    """
    partition = HeadPartition(size, heads)
    block = tuple(
        tuple(1.0 if (index + column) % size in (0, 1) else 0.0 for column in range(size))
        for index in range(partition.width)
    )
    stacked = tuple(row for _ in range(partition.heads) for row in block)
    return AttentionParams(
        w_query=stacked, w_key=stacked, w_value=stacked, w_output=scaled_identity(1.0, size)
    )


def toy_parameters(size: int = 6, *, seed: int = 7) -> AttentionParams:
    """确定性的小随机参数（调用生产代码的 ``default_parameters``，**不另写一份**）."""
    return default_parameters(size, seed=seed)


def induction_tasks(count: int = 4, *, seed: int = 42) -> tuple[InductionTask, ...]:
    """一批 induction 样本（同样来自生产代码里的构造函数）."""
    return make_induction_batch(count, seed=seed)


def head_shape(size: int, heads: int) -> MultiHeadShape:
    """``size × size`` 的四个投影配上给定头数的形状记录（**参数必须能整除**）."""
    return MultiHeadShape(
        AttentionShape(inputs=size, keys=size, values=size, outputs=size), heads
    )


def witness_sample():
    """手工设计的分离见证（直接调用生产代码，避免测试与演示分家）."""
    return designed_witness()


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
    "HEAD_COUNTS",
    "IDENTITY_INPUTS",
    "IDENTITY_OUTPUT",
    "IDENTITY_OUTPUT_FOUR_HEADS_ROW_ONE",
    "IDENTITY_ROW_TOTAL_VARIATIONS",
    "P_E",
    "P_E_SQUARED",
    "P_SQRT2",
    "P_THREE_SQRT2_HALF",
    "VOCABULARY",
    "WITNESS_DISTANCE",
    "WITNESS_DISTRIBUTIONS",
    "WITNESS_ONE_PULLS",
    "WITNESS_TARGET_2D",
    "WITNESS_ZERO_PULLS",
    "approx",
    "approx_matrix",
    "head_shape",
    "identity_parameters",
    "induction_tasks",
    "repeated_block_parameters",
    "scaled_identity",
    "toy_parameters",
    "witness_sample",
]
