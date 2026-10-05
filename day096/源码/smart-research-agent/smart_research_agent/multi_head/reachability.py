"""多头的表达能力：**每一行能到达哪些输出**（day076 / M7-D2）.

这一课到这里为止都在回答"多头怎么算、怎么反向、怎么写对"。
但还有一个更基本的问题没有回答：

> **多头到底多在哪？**（同一批参数、同一个参数量，为什么会更强？）

## 一、把"输出"写成一份混合

先看单头。第 ``i`` 行的输出是（``W_o`` 在最后统一作用）：

```text
output_i = Σ_j w_j · (W_o · V_j)          w 是那份分布（和为 1、非负）
```

括号里那一项只依赖**位置**，不依赖分布——把它记作

```text
contrib_j = W_o · V_j                     第 j 个位置的"贡献向量"
```

于是单头第 ``i`` 行的可达集合是

```text
单头   { Σ_j w_j · contrib_j : w 是 A_i 上的任意分布 }  =  conv{ contrib_j : j ∈ A_i }
```

**一个凸包**：顶点的数量最多等于可见位置的个数（``A_i`` 是第 i 行允许看的位置）。

## 二、多头把它变成一个"闵可夫斯基和"

多头把 value 维切成 heads 份，每一份有自己的分布 ``w^h``：

```text
output_i = Σ_h Σ_j w^h_j · contrib^h_j        contrib^h_j = W_o^h · V^h_j
```

因此多头第 ``i`` 行的可达集合是

```text
多头   Σ_h conv{ contrib^h_j : j ∈ A_i }     ← heads 个凸包的**闵可夫斯基和**
```

而"凸包的闵可夫斯基和"有一条干净的等价形式：

```text
conv(A) + conv(B) = conv{a + b : a ∈ A, b ∈ B}
```

于是可达集合也可以读成"**网格点集的凸包**"：

```text
conv{ Σ_h contrib^h_{j_h} : 每个 j_h 都落在 A_i 里 }
```

## 三、一个能手工算完的例子

设 ``d_in = d_v = 4``、``d_k = 8``、``d_out = 2``、``heads = 2``
（于是 ``head_dim = 4``、``head_value = 2``），四个位置、value 行是

```text
位置 j    0        1        2        3
t_j       1        0        0.5      0.5        ← head 0 的 pull
s_j       0        1        0.5      0.5        ← head 1 的 pull
V_j     (1,0,0,0) (0,0,0,1) (0.5,0,0,0.5) (0.5,0,0,0.5)
```

``W_o = [[1,0,0,0],[0,0,0,1]]``：第 0 头从 value 的前两维里取第 0 维、
第 1 头从后两维里取第 1 维。于是每一头的贡献是**沿一条轴**的：

```text
contrib⁰_j = (t_j, 0)          contrib¹_j = (0, s_j)
```

**单头**（同一批参数，只把 heads 改成 1）：

```text
contrib_j  = (t_j, s_j) = (1,0)、(0,1)、(0.5,0.5)（第三、四个位置重合）
conv       = 那条线段 x + y = 1     ← 因为 (0.5,0.5) 恰好是前两点的**中点**
```

**双头**：

```text
网格点  (t_a, s_b) 共 4×4 = 16 个（去重后 9 个）
conv    = 正方形 [0,1] × [0,1]      ← **张开了一整个面**
```

取目标 ``(1, 1)``（正方形的顶点）：

```text
它到那条线段的距离    |1 + 1 − 1| / √2 = 1/√2 = 0.7071067811865475
它在正方形内部        → 距离 0
```

兑现它的那两份分布也写清楚了：

```text
head 0  把全部权重押在位置 0   （那里 t = 1）
head 1  把全部权重押在位置 1   （那里 s = 1）
=>  contrib⁰₀ + contrib¹₁ = (1,0) + (0,1) = (1,1)
```

于是这一课的核心结论可以被一句话说完，而且**每个数字都能在纸上算出来**：

> **同一批参数（参数量一字不差），单头只能到达一条线段，双头能到达一个正方形。**

## 四、这份结论的**前提**，必须写下来

"任意分布都能被某组参数实现"不是免费的。第 ``i`` 行的打分是 ``q_i·k_j``，
它的取值落在 ``span{k_j}`` 里，而那个空间的维数最多是 ``head_dim``。因此：

```text
head_dim >= |A_i|      这一行的可达集合才等于上面那两个凸包
head_dim <  |A_i|      可达集合是那两个凸包的**子集**——
                       此时"单头到不了"仍然成立（它的可达集合只会更小），
                       但"双头一定能到"就**不能**由这条路证明
```

所以本模块在 ``head_dim < |A_i|`` 时**直接拒绝计算**（抛 ``ParameterError``），
而不是给出一个偏大的可达集合：**一个偏大的可达集合会把"到不了"的结论说成"到得了"**，
而那是这一课唯一真正值钱的结论。

上面那个例子里 ``head_dim = 4`` 而可见位置恰好是 4 个——**边界情形**
（``>=`` 取等号），因此它同时也是这条前提的一次极限测试。

## 五、还有一个"到底能不能到"的问题，本文档只回答一半

```text
本模块回答    "输出**空间**里哪些点是可以到达的"（一个几何问题）
本模块不回答  "训练**能不能**找到那组参数"（一个优化问题）
```

两者完全不同：可达集合是"存在性"的陈述，它不承诺梯度下降会找到它。
演示脚本里那个 ``(1, 1)`` 是被**构造**出来的（权重写死、逐位核对），
而不是被训练出来的——教程里会明确区分这两件事。

## 六、一处实现细节：为什么这一课只做二维

可达集合在任意 ``d_out`` 上都算得出来（网格点集与它是一样的），
但"距离"这个读数需要一个**凸包**工具，而通用维度的凸包/线性可行性判定
与这一课要讲的东西无关。因此本模块只支持 ``d_out = 2``：

```text
二维    凸包是一条折线/多边形，距离可以手算、可以画出来
高维    需要一个线性规划求解器——那会把"可核对的几何"变成"信任一个求解器"
```

它同样是**显式拒绝**而不是近似：二维以外的目标维度当场抛 ``ParameterError``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from smart_research_agent.math_foundations.linalg import matmul, transpose
from smart_research_agent.math_foundations.types import Matrix, Vector
from smart_research_agent.multi_head.errors import (
    MultiHeadError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.multi_head.layers import multi_head_attention
from smart_research_agent.multi_head.types import MultiHeadForward, MultiHeadShape
from smart_research_agent.transformer_core.types import AttentionParams

#: 网格点的个数上限（``|A|^heads``）。超过它就拒绝计算而不是把内存吃光：
#: ``n = 12``、``heads = 6`` 已经是 300 万个点了——
#: 而这个读数只在"手工可核对"的规模上才有意义。
MAX_VERTEX_COMBINATIONS = 200_000

#: 二维凸包判定的容差（``1e-9``：足够吸收浮点噪声，又远小于样本之间的间距）.
HULL_TOLERANCE = 1e-9

#: 见证样本里每一头在"可见位置"上的 pull（**这就是凸包的坐标**）.
#:
#: ```text
#: 位置          0      1      2      3
#: head 0 的 t   1.0    0.0    0.5    0.5
#: head 1 的 s   0.0    1.0    0.5    0.5
#: ```
#:
#: 位置 2、3 的 pull 相同（两个不同的 token 可以有相同的 value 向量）：
#: 它不影响任何一个凸包，但让样本长度是 4——恰好顶到 ``head_dim = 4`` 这条前提上。
WITNESS_HEAD_ZERO_PULLS: Vector = (1.0, 0.0, 0.5, 0.5)
WITNESS_HEAD_ONE_PULLS: Vector = (0.0, 1.0, 0.5, 0.5)

#: 手工设计的见证目标：正方形的一个顶点 ``(1, 1)``.
WITNESS_TARGET: Vector = (1.0, 1.0)

#: 兑现那个目标的两份分布（**两头各一份，而且它们完全不一样**）.
#:
#: ```text
#: head 0  全部权重押在位置 0（t = 1）
#: head 1  全部权重押在位置 1（s = 1）
#: => (1,0) + (0,1) = (1,1)
#: ```
#:
#: 两份分布的 TV 距离是 **1.0**（最大值）——这是"多头真的分了两路"的最强形态。
WITNESS_WEIGHTS: tuple[Vector, ...] = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
)

#: 单头到目标的距离 ``|1+1−1|/√2``（**手算值**，测试直接断言它）.
WITNESS_ONE_HEAD_DISTANCE = 1.0 / math.sqrt(2.0)


def allowed_positions(forward: MultiHeadForward, row: int) -> tuple[int, ...]:
    """第 ``row`` 行允许看的位置（掩码为 ``True`` 的那些列）."""
    if isinstance(row, bool) or not isinstance(row, int):
        raise ParameterError(f"行号必须是整数，收到 {row!r}。")
    if not 0 <= row < forward.tokens:
        raise ParameterError(f"行号 {row} 落在 [0, {forward.tokens}) 之外。")
    positions = tuple(
        column for column in range(forward.tokens) if forward.mask[row][column]
    )
    if not positions:
        raise ShapeError(
            f"第 {row} 行没有任何允许的位置：这一行的注意力分布没有定义。"
        )
    return positions


def head_contributions(
    forward: MultiHeadForward,
    row: int,
) -> tuple[tuple[Vector, ...], ...]:
    """每一头在 ``A_i`` 各位置上的**贡献向量** ``W_o^h · V^h_j``（宽度 ``d_out``）.

    "贡献"这个词是精确的：如果第 ``h`` 头把整份权重押在位置 ``j`` 上，
    它交给输出投影的就是这条向量，于是

    ```text
    output_i = Σ_h Σ_j w^h_j · contrib^h_j
    ```
    """
    shape = forward.shape
    positions = allowed_positions(forward, row)
    output_blocks = shape.values_partition.split_columns(forward.params.w_output)
    per_head: list[tuple[Vector, ...]] = []
    for head in range(shape.heads):
        vertices: list[Vector] = []
        for position in positions:
            value_row = forward.head_values[head][position]
            contributed = matmul((value_row,), transpose(output_blocks[head]))[0]
            vertices.append(tuple(contributed))
        per_head.append(tuple(vertices))
    return tuple(per_head)


def single_head_contributions(
    params: AttentionParams,
    inputs: Matrix,
    row: int,
    *,
    causal: bool = True,
) -> tuple[Vector, ...]:
    """**同一批参数**下 ``heads=1`` 的位置贡献 ``W_o · V_j``（对照的基准）.

    为什么基准必须是"同一批参数、只改 heads"：
    换一批参数就变成了"两个模型的对比"，而那回答不了"多头结构本身多了什么"。
    参数量在这里一字不差——``W_q``/``W_k``/``W_v``/``W_o`` 的形状与数值完全相同，
    唯一变化的是"这些维度被分成几段"。
    """
    forward = multi_head_attention(params, inputs, heads=1, causal=causal)
    positions = allowed_positions(forward, row)
    return tuple(
        tuple(matmul((forward.values[position],), transpose(params.w_output))[0])
        for position in positions
    )


def minkowski_vertices(sets: tuple[tuple[Vector, ...], ...]) -> tuple[Vector, ...]:
    """网格点集 ``{Σ_h v^h_{j_h}}``——**凸包顶点的候选**（去重、保序）.

    用它而不是"逐点相加再算凸包"的理由是那条定理：

    ```text
    conv(A) + conv(B) = conv{a + b : a ∈ A, b ∈ B}
    ```

    左边的闵可夫斯基和是"两个无限点集的和"，右边只是一个有限的网格点集——
    于是"多头可达集合 = heads 个凸包之和"这句话可以被一台计算机在毫秒内算完。
    """
    if not sets:
        raise ShapeError("至少要有一头：零头的和需要一条「空集合 = 0」的约定。")
    width = len(sets[0][0]) if sets[0] and sets[0][0] else 0
    if width == 0:
        raise ShapeError("每一头至少要有一个贡献向量，而且它的宽度不能为 0。")
    total = 1
    for group in sets:
        if not group:
            raise ShapeError("某一头没有任何位置可看：它的可达集合是空集。")
        if len(group[0]) != width:
            raise ShapeError(
                f"各头贡献向量的宽度不一致：{len(group[0])} 与 {width}——"
                "它们必须都是 d_out 维（同一层输出的不同头）。"
            )
        total *= len(group)
    if total > MAX_VERTEX_COMBINATIONS:
        raise ParameterError(
            f"网格点有 {total} 个（上限 {MAX_VERTEX_COMBINATIONS}）："
            "|A|^heads 随序列长度与头数迅速爆炸，"
            "而「手工可核对」的规模才是这个读数的意义所在。"
        )
    seen: list[Vector] = []
    for combination in product(*sets):
        summed = tuple(
            math.fsum(part[column] for part in combination) for column in range(width)
        )
        if summed not in seen:
            seen.append(summed)
    return tuple(seen)


def _cross(origin: Vector, first: Vector, second: Vector) -> float:
    """叉积 ``(a−o) × (b−o)``（二维，符号即转向）."""
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (
        first[1] - origin[1]
    ) * (second[0] - origin[0])


def _checked_two_dimensional(
    points: tuple[Vector, ...] | list[Vector],
    *,
    what: str,
) -> None:
    """校验一批点都是二维（**显式拒绝而不是近似**）."""
    if not points:
        raise ShapeError(f"{what}至少要有一个点。")
    for index, point in enumerate(points):
        if len(point) != 2:
            raise ParameterError(
                f"{what}的第 {index} 个点是 {len(point)} 维，而这一课只做二维："
                "高维可达集合还能算（网格点是一样的），但**画不出来**——"
                "而画不出来就没有「距离」这个读数。"
            )
        if not (math.isfinite(point[0]) and math.isfinite(point[1])):
            raise ParameterError(f"{what}的第 {index} 个点不是有限数：{point}。")


def convex_hull_2d(points: tuple[Vector, ...] | list[Vector]) -> tuple[Vector, ...]:
    """二维凸包（Andrew 单调链，**逆时针**、不含重复闭合点）.

    三个约定写下来，因为它们决定了"距离 = 0"这件事的判据：

    ```text
    共线点被丢掉     (1,0)、(0,1)、(0.5,0.5) 的凸包是**两个**端点
                     ——留下中点会让"点在多边形内"的判定面对一个退化多边形
    顺序逆时针       单调链天然给出逆时针，于是"点在内部" = 所有叉积 >= 0
    重复点先去重     同一对 (x, y) 只保留一份（不去重会让共线判定退化）
    ```
    """
    _checked_two_dimensional(points, what="凸包")
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) <= 2:
        return tuple(unique)

    def _chain(ordered: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """单调链的一段（``<= 0`` 意味着丢掉共线点）."""
        chain: list[tuple[float, float]] = []
        for point in ordered:
            while len(chain) >= 2 and _cross(chain[-2], chain[-1], point) <= 0:
                chain.pop()
            chain.append(point)
        return chain

    lower = _chain(unique)
    upper = _chain(list(reversed(unique)))
    return tuple(lower[:-1] + upper[:-1])


def _point_in_convex_polygon_2d(
    point: Vector,
    hull: tuple[Vector, ...],
    *,
    tolerance: float = HULL_TOLERANCE,
) -> bool:
    """点是否落在**逆时针凸多边形**内（含边界）."""
    for index in range(len(hull)):
        origin = hull[index]
        following = hull[(index + 1) % len(hull)]
        if _cross(origin, following, point) < -tolerance:
            return False
    return True


def _point_to_segment_distance(point: Vector, start: Vector, end: Vector) -> float:
    """点到线段的欧氏距离（**投影落在线段外时取端点的距离**）."""
    span_x = end[0] - start[0]
    span_y = end[1] - start[1]
    length_squared = span_x * span_x + span_y * span_y
    if length_squared == 0.0:
        return math.dist(point, start)
    ratio = ((point[0] - start[0]) * span_x + (point[1] - start[1]) * span_y) / (
        length_squared
    )
    ratio = min(1.0, max(0.0, ratio))
    foot = (start[0] + ratio * span_x, start[1] + ratio * span_y)
    return math.dist(point, foot)


def distance_to_convex_hull_2d(
    point: Vector,
    points: tuple[Vector, ...] | list[Vector],
    *,
    tolerance: float = HULL_TOLERANCE,
) -> float:
    """点到"这些点的凸包"的距离（**在凸包内时为 0.0**）.

    这是这一课最重要的一个读数：

    ```text
    单头的距离 > 0    → 这个目标**结构上**到不了（不是"训练没找到"，是"不存在"）
    单头的距离 = 0    → 处在这条路上，剩下的问题才是优化
    ```

    三种退化情形各有各的算法：一个点（退化成点距）、两个点（线段距离）、
    三个及以上（凸多边形：先判内部，再取到各边的最小距离）。
    """
    _checked_two_dimensional([tuple(point)], what="目标点")
    hull = convex_hull_2d(points)
    if len(hull) == 1:
        return math.dist(tuple(point), hull[0])
    if len(hull) == 2:
        return _point_to_segment_distance(tuple(point), hull[0], hull[1])
    if _point_in_convex_polygon_2d(tuple(point), hull, tolerance=tolerance):
        return 0.0
    return min(
        _point_to_segment_distance(tuple(point), hull[index], hull[(index + 1) % len(hull)])
        for index in range(len(hull))
    )


def _checked_head_sets(
    sets: tuple[tuple[Vector, ...], ...],
) -> tuple[tuple[Vector, ...], ...]:
    """校验每一头都有贡献向量（空集会让"可达集合"变成空集）."""
    for index, group in enumerate(sets):
        if not group:
            raise ShapeError(f"第 {index} 头没有任何位置可看：它的可达集合是空集。")
    return sets


@dataclass(frozen=True)
class ReachabilityReport:
    """「同一行、两种结构」的可达集合对照（:func:`reachability_report` 的产物）.

    ```text
    one_head_vertices     单头的位置贡献（最多 |A_i| 个点）
    multi_head_vertices   多头网格点（最多 |A_i|^heads 个点）
    one_head_hull         上面那批点的凸包
    multi_head_hull       同上
    target                被问「到不到得了」的那个点
    one_head_distance     到单头凸包的距离（> 0 即结构上到不了）
    multi_head_distance   到多头凸包的距离
    witness               兑现目标的 heads 份分布（**每头一份**）
    ```

    ``witness`` 不是"训练出来的解"，而是**构造出来的**存在性证据：
    它把"到得了"从一句几何断言变成一组可以直接代进前向核对的权重。
    """

    heads: int
    row: int
    allowed: tuple[int, ...]
    head_dim: int
    one_head_vertices: tuple[Vector, ...]
    multi_head_vertices: tuple[Vector, ...]
    one_head_hull: tuple[Vector, ...]
    multi_head_hull: tuple[Vector, ...]
    target: Vector
    one_head_distance: float
    multi_head_distance: float
    witness: tuple[Vector, ...]
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.heads < 1:
            raise ParameterError(f"头数必须 >= 1，收到 {self.heads}。")
        if self.row < 0:
            raise ParameterError(f"行号必须 >= 0，收到 {self.row}。")
        if not self.allowed:
            raise ShapeError("允许的位置不能为空。")
        if len(self.target) != 2:
            raise ParameterError(f"目标点必须是 2 维，收到 {len(self.target)} 维。")
        if self.head_dim < len(self.allowed):
            raise ParameterError(
                f"head_dim={self.head_dim} 小于可见位置数 {len(self.allowed)}："
                "此时「任意分布都能被实现」不再成立，而一个**偏大的**可达集合"
                "会把「到不了」的结论说成「到得了」——那正是这一课唯一值钱的结论。"
            )
        if len(self.witness) != self.heads:
            raise ShapeError(
                f"见证给出了 {len(self.witness)} 份分布，而头数是 {self.heads}："
                "每一头都要有自己的那一份。"
            )
        for index, distribution in enumerate(self.witness):
            if len(distribution) != len(self.allowed):
                raise ShapeError(
                    f"第 {index} 份分布有 {len(distribution)} 项，"
                    f"而可见位置有 {len(self.allowed)} 个。"
                )
            total = math.fsum(distribution)
            if abs(total - 1.0) > 1e-9:
                raise MultiHeadError(
                    f"第 {index} 份分布的和是 {total!r}：见证必须是一份**合法的分布**，"
                    "否则它证明的是「某个非概率的东西能到目标」。"
                )
            if min(distribution) < 0.0:
                raise MultiHeadError(f"第 {index} 份分布出现了负权重。")
        for name, value in (
            ("单头距离", self.one_head_distance),
            ("多头距离", self.multi_head_distance),
        ):
            if math.isnan(value) or value < 0:
                raise ParameterError(f"{name}不能为负或 NaN：{value}。")
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def one_head_reachable(self) -> bool:
        """单头结构上到得了吗（距离为 0）."""
        return self.one_head_distance <= HULL_TOLERANCE

    @property
    def multi_head_reachable(self) -> bool:
        """多头结构上到得了吗."""
        return self.multi_head_distance <= HULL_TOLERANCE

    @property
    def gap(self) -> float:
        """单头距离（"多头多出来的那一块"的度量）."""
        return self.one_head_distance

    @property
    def separates(self) -> bool:
        """这一行是否**分离**了两种结构：单头到不了而多头到得了."""
        return (not self.one_head_reachable) and self.multi_head_reachable

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "heads": self.heads,
            "row": self.row,
            "allowed": list(self.allowed),
            "head_dim": self.head_dim,
            "one_head_vertices": [list(point) for point in self.one_head_vertices],
            "multi_head_vertices": [list(point) for point in self.multi_head_vertices],
            "one_head_hull": [list(point) for point in self.one_head_hull],
            "multi_head_hull": [list(point) for point in self.multi_head_hull],
            "target": list(self.target),
            "one_head_distance": self.one_head_distance,
            "multi_head_distance": self.multi_head_distance,
            "one_head_reachable": self.one_head_reachable,
            "multi_head_reachable": self.multi_head_reachable,
            "separates": self.separates,
            "witness": [list(distribution) for distribution in self.witness],
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``row 3 | 目标 (1.0, 1.0) | 单头距 0.7071（到不了）| 多头距 0.0000``."""
        verdict = "分离" if self.separates else "未分离"
        return (
            f"row {self.row} | 可见 {len(self.allowed)} 个位置 | "
            f"目标 {tuple(round(value, 4) for value in self.target)} | "
            f"单头距 {self.one_head_distance:.4f}"
            f"（{'到得了' if self.one_head_reachable else '到不了'}）| "
            f"多头距 {self.multi_head_distance:.4f}"
            f"（{'到得了' if self.multi_head_reachable else '到不了'}）| {verdict}"
        )


def reachability_report(
    params: AttentionParams,
    inputs: Matrix,
    row: int,
    *,
    heads: int,
    target: Vector,
    causal: bool = True,
    witness: tuple[Vector, ...] | None = None,
) -> ReachabilityReport:
    """算一行在"同一批参数"下两种结构的可达集合，并回答目标点到不到得了.

    ``witness`` 可以显式给出（用于手工构造的样本），也可以让本函数自己找：
    缺省的做法是"每一头各取第一个可见位置"——那是一个合法分布，
    但**不保证**兑现 ``target``；报告里会写清这一点。

    为什么不在内部解一个小线性规划去**找**见证：

    ```text
    这一课要证明的是"分离"（单头到不了、多头到得了）
    而"到得了"的证据最好由**构造**给出（一组可以代进去核对的权重），
    而不是由一个求解器说"我找到了"——前者可复核，后者多一层信任
    ```
    """
    forward = multi_head_attention(params, inputs, heads=heads, causal=causal)
    shape = forward.shape
    positions = allowed_positions(forward, row)
    if shape.head_dim < len(positions):
        raise ParameterError(
            f"head_dim={shape.head_dim} 小于可见位置数 {len(positions)}（第 {row} 行）："
            "打分矩阵的秩不足以张出任意 logits，因此「任意分布都能被实现」不成立，"
            "而基于它的可达集合会**偏大**。请换一个更宽的 d_k 或更短的序列。"
        )
    if shape.attention.outputs != 2:
        raise ParameterError(
            f"本函数只支持 d_out = 2（收到 {shape.attention.outputs}）："
            "可达集合在任意维度上都算得出来，但「距离」需要一个凸包工具，"
            "而通用维度的凸包/线性可行性判定与这一课要讲的东西无关。"
        )
    per_head = _checked_head_sets(head_contributions(forward, row))
    one_head = single_head_contributions(params, inputs, row, causal=causal)
    checked_target = _checked_target(target)
    multi_head = minkowski_vertices(per_head)
    return ReachabilityReport(
        heads=shape.heads,
        row=row,
        allowed=positions,
        head_dim=shape.head_dim,
        one_head_vertices=one_head,
        multi_head_vertices=multi_head,
        one_head_hull=convex_hull_2d(one_head),
        multi_head_hull=convex_hull_2d(multi_head),
        target=checked_target,
        one_head_distance=distance_to_convex_hull_2d(checked_target, one_head),
        multi_head_distance=distance_to_convex_hull_2d(checked_target, multi_head),
        witness=_first_combination(per_head) if witness is None else tuple(witness),
        notes=(
            "两种结构用的是**同一批参数**（形状与数值一字不差），"
            "唯一变化的只有 heads——因此这不是「两个模型的对比」，"
            "而是「同一批参数被组织成几种分布」的对比",
            "单头的可达集合是 conv{contrib_j}（一个凸包），"
            "多头的是 Σ_h conv{contrib^h_j}（闵可夫斯基和 = 网格点集的凸包）",
            "本报告只回答「可达集合里有没有这个点」（存在性），"
            "**不回答**「训练能不能找到那组参数」（优化问题）——两者完全不同",
            "结论的前提是 head_dim >= |A_i|：本函数在前提不成立时直接拒绝计算，"
            "因为偏大的可达集合会把「到不了」说成「到得了」",
            "缺省见证只是「每头各取第一个可见位置」——它是合法分布，"
            "但**只是存在性占位**，不保证兑现 target",
        ),
    )


def _checked_target(target: Vector) -> Vector:
    """校验目标点（二维、有限）."""
    if len(target) != 2:
        raise ParameterError(f"目标点必须是 2 维，收到 {len(target)} 维。")
    for value in target:
        if not math.isfinite(value):
            raise ParameterError(f"目标点必须由有限数组成，收到 {target}。")
    return (float(target[0]), float(target[1]))


def _checked_distributions(
    weights: tuple[Vector, ...] | list[Vector],
    *,
    heads: int,
    slots: int,
) -> tuple[Vector, ...]:
    """校验 heads 份分布：非负、和为 1、长度等于可见位置数."""
    if len(weights) != heads:
        raise ShapeError(f"给出 {len(weights)} 份分布，而头数是 {heads}。")
    checked: list[Vector] = []
    for head, distribution in enumerate(weights):
        values = tuple(float(value) for value in distribution)
        if len(values) != slots:
            raise ShapeError(
                f"第 {head} 份分布有 {len(values)} 项，而可见位置有 {slots} 个。"
            )
        if min(values) < 0.0:
            raise MultiHeadError(f"第 {head} 份分布出现了负权重：{values}。")
        total = math.fsum(values)
        if abs(total - 1.0) > 1e-9:
            raise MultiHeadError(
                f"第 {head} 份分布的和是 {total!r}：它必须是一份**合法的分布**。"
            )
        checked.append(values)
    return tuple(checked)


def _first_combination(sets: tuple[tuple[Vector, ...], ...]) -> tuple[Vector, ...]:
    """每一头各取第一个可见位置（**存在性占位**，不保证兑现目标）.

    它不是"解"，而是"一个合法的分布"：第 h 头把全部权重押在第一个可见位置上。
    报告里必须写清这一点——把占位当成解，会让"多头到得了"这句话失去它唯一的证据。
    """
    return tuple(
        tuple(1.0 if index == 0 else 0.0 for index in range(len(group))) for group in sets
    )


def realize_with_weights(
    forward: MultiHeadForward,
    row: int,
    weights: tuple[Vector, ...] | list[Vector],
) -> Vector:
    """把 heads 份分布代进前向，算出第 ``row`` 行的**输出向量**（宽度 ``d_out``）.

    ``weights[h][k]`` 是第 h 头押在 ``allowed_positions[row][k]`` 上的权重。
    计算路径恰好照着可达集合的推导走一遍：

    ```text
    第 h 头     Σ_k w^h_k · V^h_{j_k} = c_h        （该头的 context，head_value 维）
    贡献        W_o^h · c_h                        （d_out 维）
    输出        Σ_h W_o^h · c_h                    （**注意没有经过 softmax**）
    ```

    它是见证的核对方式：一个"多头到得了"的结论，最终必须能被这样验一遍——
    否则那只是一个几何断言，而不是一个能落到这一层上的事实。
    """
    positions = allowed_positions(forward, row)
    checked = _checked_distributions(
        weights, heads=forward.heads, slots=len(positions)
    )
    blocks = forward.shape.values_partition.split_columns(forward.params.w_output)
    head_value = forward.shape.head_value
    total = tuple(0.0 for _ in range(len(blocks[0])))
    for head in range(forward.heads):
        context = tuple(
            math.fsum(
                checked[head][slot] * forward.head_values[head][positions[slot]][dimension]
                for slot in range(len(positions))
            )
            for dimension in range(head_value)
        )
        contribution = matmul((context,), transpose(blocks[head]))[0]
        total = tuple(
            math.fsum((total[index], contribution[index])) for index in range(len(total))
        )
    return total


@dataclass(frozen=True)
class DesignedWitness:
    """手工设计的见证样本：**同一批参数，两种结构分离**（第三章的那个例子）.

    ```text
    inputs     I₄（四个 one-hot token，vocab = 4，长度 n = 4）
    d_in = d_v 4        d_k = 8        d_out = 2        heads = 2
    → head_dim 4（恰好等于可见位置数）   head_value 2
    W_v        四行分别是 (1,0,0,0.5)/(0,0,0,0)/(0,0,0,0)/(0,1,0,0.5)
               → V 的四行 = (1,0,0,0)、(0,0,0,1)、(0.5,0,0,0.5)、(0.5,0,0,0.5)
    W_o        [[1,0,0,0],[0,0,0,1]]  → 第 0 头取 value 第 0 维、第 1 头取第 3 维
    row        3（因果掩码下它能看到全部四个位置）
    target     (1, 1)
    ```

    三个手算的锚点（测试直接断言它们）：

    ```text
    单头贡献            (1,0)、(0,1)、(0.5,0.5)（后两个位置重合）
    单头凸包            线段 (0,1)—(1,0)（第三点是中点，共线）
    单头到目标距离      1/√2 = 0.7071067811865475
    兑现目标的权重      head 0 → (1,0,0,0)、head 1 → (0,1,0,0)（TV = 1.0）
    ```
    """

    params: AttentionParams
    inputs: Matrix
    heads: int
    row: int
    target: Vector
    expected_one_head_distance: float
    witness_weights: tuple[Vector, ...]

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（参数矩阵本身不进去——它们是 8×4 的）."""
        return {
            "heads": self.heads,
            "row": self.row,
            "target": list(self.target),
            "expected_one_head_distance": self.expected_one_head_distance,
            "witness_weights": [list(item) for item in self.witness_weights],
            "head_zero_pulls": list(WITNESS_HEAD_ZERO_PULLS),
            "head_one_pulls": list(WITNESS_HEAD_ONE_PULLS),
        }


def designed_witness() -> DesignedWitness:
    """构造第三章那个"单头到不了、双头到得了"的样本（**全部写死，可逐位复核**）.

    四个矩阵都刻意选成"能一眼读出"的形状：

    ```text
    W_q = W_k    每行只有一个 1，且**后天半段整体乘 0.5**
                 → 两头在这一组参数下给出不一样的分布（头间差异非零）
    W_v          只有两行非零：第 0 行是 head 0 的 pull、第 3 行是 head 1 的 pull
    W_o          两行分别"只读第 0 维"与"只读第 3 维"——于是两头的贡献沿两条轴
    ```

    两处必须写下来的说明：

    ```text
    ① W_v 有两行全零**不被拒绝**：投影矩阵出现全零行是初始化的自由
       （与 day075 的纪律一致——数据里的全零行才拒绝，参数里的不拒绝）
    ② W_q/W_k 是 0.5 这个数字只在"这一组具体参数"下有意义：
       可达集合只由 W_v 与 W_o 决定（它量化的是**所有可能的分布**），
       与 q/k 无关——因此换掉 q/k 不会动那个 1/√2 的距离
    ```
    """
    size = len(WITNESS_HEAD_ZERO_PULLS)
    query_or_key = tuple(
        tuple(
            (1.0 if index % size == column else 0.0) * (1.0 if index < size else 0.5)
            for column in range(size)
        )
        for index in range(2 * size)
    )
    value_weight = (
        tuple(WITNESS_HEAD_ZERO_PULLS),
        (0.0,) * size,
        (0.0,) * size,
        tuple(WITNESS_HEAD_ONE_PULLS),
    )
    output_weight = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    params = AttentionParams(
        w_query=query_or_key,
        w_key=query_or_key,
        w_value=value_weight,
        w_output=output_weight,
    )
    inputs = tuple(
        tuple(1.0 if index == column else 0.0 for column in range(size))
        for index in range(size)
    )
    return DesignedWitness(
        params=params,
        inputs=inputs,
        heads=2,
        row=size - 1,
        target=WITNESS_TARGET,
        expected_one_head_distance=WITNESS_ONE_HEAD_DISTANCE,
        witness_weights=WITNESS_WEIGHTS,
    )


def witness_report() -> ReachabilityReport:
    """直接对设计好的样本算一份可达集合报告（演示与测试共用同一条入口）."""
    witness = designed_witness()
    return reachability_report(
        witness.params,
        witness.inputs,
        witness.row,
        heads=witness.heads,
        target=witness.target,
        causal=True,
        witness=witness.witness_weights,
    )


def witness_parameters_are_shared() -> bool:
    """两种结构下的四个矩阵是否**逐位相同**（结构性对照的前提）.

    它是"同一批参数、只改 heads"这句话的可执行版本：若哪天有人为了"让对照更好看"
    而换了另一组参数，这条检查会红——而它红了以后，那个 0.7071 的距离
    就不再回答"多头结构多了什么"了。
    """
    witness = designed_witness()
    classic = multi_head_attention(witness.params, witness.inputs, heads=1, causal=True)
    separated = multi_head_attention(
        witness.params, witness.inputs, heads=witness.heads, causal=True
    )
    return (
        classic.params.w_query == separated.params.w_query
        and classic.params.w_key == separated.params.w_key
        and classic.params.w_value == separated.params.w_value
        and classic.params.w_output == separated.params.w_output
    )


def multi_head_shape_of(params: AttentionParams, heads: int) -> MultiHeadShape:
    """把一对（参数、头数）变成形状记录（报告里要能读出"这一层是什么形状"）."""
    return MultiHeadShape(params.shape, heads)


__all__ = [
    "HULL_TOLERANCE",
    "MAX_VERTEX_COMBINATIONS",
    "WITNESS_HEAD_ONE_PULLS",
    "WITNESS_HEAD_ZERO_PULLS",
    "WITNESS_ONE_HEAD_DISTANCE",
    "WITNESS_TARGET",
    "WITNESS_WEIGHTS",
    "DesignedWitness",
    "ReachabilityReport",
    "allowed_positions",
    "convex_hull_2d",
    "designed_witness",
    "distance_to_convex_hull_2d",
    "head_contributions",
    "minkowski_vertices",
    "multi_head_shape_of",
    "reachability_report",
    "realize_with_weights",
    "single_head_contributions",
    "witness_parameters_are_shared",
    "witness_report",
]
