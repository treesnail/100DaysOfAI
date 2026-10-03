"""``positional_encoding`` 的共享样本：写死的参数、表与手算锚点（day078 / M7-D3）.

三条约定（与 ``multihead_samples`` / ``attention_samples`` 逐字相同）：

```text
① 全部离线      纯 Python 算术，零依赖、零网络，任何环境都能跑
② 输入写死      不用随机输入（随机输入会让"这次对上了"不可复现）
③ 期望值手算    关键断言给出**手算结果**，而不是"上次跑出来的值"
```

第 ③ 条在这一课有一个额外的好运气：``d = 4``、``base = 10000`` 时两个频率周期恰好是
``1`` 与 ``100``（``√10000 = 100``）——于是 ``PE(1)`` 的四个分量**不需要计算器**就能写出来：

```text
PE(0) = (sin 0, cos 0, sin 0, cos 0)          = (0, 1, 0, 1)
PE(1) = (sin 1, cos 1, sin 0.01, cos 0.01)    = (0.841471, 0.540302, 0.00999983, 0.99995)
每行范数 = √(d/2) = √2                        每一对贡献 sin² + cos² = 1
<PE(p), PE(p+1)> = cos(1) + cos(0.01)         = 1.540252306284805
```

本模块只做两件事：**把生产函数的入参写死**、**把锚点写成常量**。
工厂函数一律**委托生产函数**，绝不另写一份公式——两份公式迟早在某一处悄悄分家。
"""

from __future__ import annotations

import math

from smart_research_agent.positional_encoding.layers import (
    POSITIONAL_BASE,
    learnable_table,
    sinusoidal_table,
    staggered_table,
    zero_table,
)
from smart_research_agent.positional_encoding.symmetry import (
    bag_predictor,
    make_readout_task,
)
from smart_research_agent.positional_encoding.types import EncodingTable
from smart_research_agent.positional_encoding.verify import default_permutation
from smart_research_agent.transformer_core.train import default_parameters
from smart_research_agent.transformer_core.types import AttentionParams

#: 位置选择任务的默认词表与长度（``make_readout_task`` 的缺省值，写在这里以免各处再写一遍）.
VOCABULARY = 6
LENGTH = 4

#: 手算正弦表用的**偶数维**（``d = 4`` → 两个频率对，周期 1 与 100）.
DIMENSION = 4

#: 频率基数（与生产常量同值，但**另立一份**并由断言钉住——见测试里的 ``POSITIONAL_BASE`` 一致性）.
POSITIONAL_BASE_VALUE = POSITIONAL_BASE

#: 两个频率对的周期：``base^(2i/d)``，``i = 0`` 是 1、``i = 1`` 是 ``√10000 = 100``.
FREQUENCY_PAIR_ZERO = 1.0
FREQUENCY_PAIR_ONE = 100.0

#: ``sinusoidal_table(4, 4)`` 的第 0 行（对齐频率）：``(sin 0, cos 0, sin 0, cos 0)``.
PE_ROW_ZERO: tuple[float, ...] = (0.0, 1.0, 0.0, 1.0)

#: 第 1 行：``(sin 1, cos 1, sin 0.01, cos 0.01)``——四个数全部手算.
PE_ROW_ONE: tuple[float, ...] = (
    0.8414709848078965,
    0.5403023058681398,
    0.009999833334166664,
    0.9999500004166653,
)

#: 错位表（``pairing="staggered"``，day073 的写法）的第 1 行：
#: ``(sin 1, cos 0.1, sin 0.01, cos 0.001)``——它**不再**与 PE(0) 同频配对.
STAGGERED_ROW_ONE: tuple[float, ...] = (
    0.8414709848078965,
    0.9950041652780258,
    0.009999833334166664,
    0.9999995000000417,
)

#: 对齐正弦表每一行的 L2 范数 ``√(d/2) = √2``（每一对贡献 ``sin² + cos² = 1``）.
ROW_NORM_TWO = math.sqrt(2.0)

#: 位移律的闭式在 ``δ = 1`` 处的值 ``cos(1) + cos(0.01) = 1.540252306284805``.
OFFSET_INNER_ONE = 1.540252306284805

#: 一列全 1 的输入（``4 × 4``）：注入之后的第 0 行可以手算成 ``(1, 1, 1, 1) + PE(0)``.
ONES_INPUTS: tuple[tuple[float, ...], ...] = ((1.0, 1.0, 1.0, 1.0),) * 4

#: ``ONES_INPUTS`` 注入 ``sinusoidal_table(4, 4)`` 之后第 0 行：``(1 + 0, 1 + 1, 1 + 0, 1 + 1)``.
ONES_INJECTED_ROW_ZERO: tuple[float, ...] = (1.0, 2.0, 1.0, 2.0)

#: 上面那一行的“位置/内容”比 ``‖PE(0)‖ / ‖(1,1,1,1)‖ = √2 / 2``.
ONES_MEAN_INJECTION_RATIO = math.sqrt(2.0) / 2.0

#: 位置选择任务的基序列（``seed=42``、``V=6``、``n=4``）——**互不相同**的四个 token.
READOUT_TOKENS: tuple[int, ...] = (1, 0, 3, 2)

#: 它的整条循环位移轨道（``n`` 个位移，含自身）.
READOUT_ORBIT_TOKENS: tuple[tuple[int, ...], ...] = (
    (1, 0, 3, 2),
    (0, 3, 2, 1),
    (3, 2, 1, 0),
    (2, 1, 0, 3),
)

#: 闭式下界 ``(n − 1) / (n·V) = 3 / 24``——等变模型在轨道上的平均损失不可能更低.
READOUT_FLOOR = 3 / 24

#: 袋子预测器（最优等变预测器）的一行：``n`` 个 token 上各 ``1/n``，其余为 0.
BAG_PREDICTOR_ROW: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25, 0.0, 0.0)

#: 缺省置换是**倒序**（它同时也是"最不像恒等"的那个置换）.
REVERSAL_PERMUTATION: tuple[int, ...] = (3, 2, 1, 0)

#: 训练类用例统一用 20 步（够让损失明显下降，又能让整套几分钟内跑完）.
TRAIN_STEPS = 20

#: 训练类用例统一的学习率（``AdamOptimizer``，与 day075/076 的小训练同配置）.
LEARNING_RATE = 0.05


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


def position_parameters(size: int = VOCABULARY) -> AttentionParams:
    """确定性的小随机参数（**调用生产代码的** ``default_parameters``，不另写一份）."""
    return default_parameters(size)


def readout_task():
    """位置选择任务的基序列（``seed=42``、``V=6``、``n=4``）——同样委托生产函数."""
    return make_readout_task(seed=42, vocabulary=VOCABULARY, length=LENGTH)


def readout_orbit() -> tuple:
    """整条循环位移轨道（``n`` 个 ``ReadoutTask``）."""
    return readout_task().orbit()


def readout_bag_row() -> tuple[float, ...]:
    """袋子预测器的一行（委托生产函数，避免手抄一份分布）."""
    return bag_predictor(readout_task())[0]


def sinusoidal(
    positions: int = LENGTH, dimension: int = DIMENSION, *, pairing: str | None = None
) -> EncodingTable:
    """正弦表（缺省**对齐频率**；给 ``pairing`` 时原样透传）."""
    if pairing is None:
        return sinusoidal_table(positions, dimension, base=POSITIONAL_BASE)
    return sinusoidal_table(positions, dimension, base=POSITIONAL_BASE, pairing=pairing)


def staggered(positions: int = LENGTH, dimension: int = DIMENSION) -> EncodingTable:
    """错位频率的正弦表（day073 的写法）——两条性质一起失效的那一张."""
    return staggered_table(positions, dimension, base=POSITIONAL_BASE)


def learnable(
    positions: int = LENGTH, dimension: int = DIMENSION, *, initializer: str = "sinusoidal"
) -> EncodingTable:
    """可学习表（缺省从对齐正弦表出发；``initializer="random"`` 时随机初始化）."""
    return learnable_table(positions, dimension, initializer=initializer)


def zeros(positions: int = LENGTH, dimension: int = DIMENSION) -> EncodingTable:
    """全零表（注入退化成恒等映射，也是最容易手算的那一张）."""
    return zero_table(positions, dimension)


def reversal_permutation(rows: int) -> tuple[int, ...]:
    """倒序置换（委托生产代码的 ``default_permutation``）."""
    return default_permutation(rows)


__all__ = [
    "BAG_PREDICTOR_ROW",
    "DIMENSION",
    "FREQUENCY_PAIR_ONE",
    "FREQUENCY_PAIR_ZERO",
    "LEARNING_RATE",
    "LENGTH",
    "OFFSET_INNER_ONE",
    "ONES_INJECTED_ROW_ZERO",
    "ONES_INPUTS",
    "ONES_MEAN_INJECTION_RATIO",
    "PE_ROW_ONE",
    "PE_ROW_ZERO",
    "POSITIONAL_BASE_VALUE",
    "READOUT_FLOOR",
    "READOUT_ORBIT_TOKENS",
    "READOUT_TOKENS",
    "REVERSAL_PERMUTATION",
    "ROW_NORM_TWO",
    "STAGGERED_ROW_ONE",
    "TRAIN_STEPS",
    "VOCABULARY",
    "approx",
    "approx_matrix",
    "learnable",
    "position_parameters",
    "readout_bag_row",
    "readout_orbit",
    "readout_task",
    "reversal_permutation",
    "sinusoidal",
    "staggered",
    "zeros",
]
