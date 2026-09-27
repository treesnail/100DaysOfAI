"""``math_foundations`` 的形状与口径表：向量、矩阵、注意力报告（day073）.

```text
Vector            一串有限实数（tuple，不可变）
Matrix            若干行、每行等长的 Vector
VectorSummary     一个向量的画像：维度、模长、均值、极值、是不是单位向量
AttentionReport   一次注意力的账：权重矩阵 + 输出 + 每行的熵与峰值
```

## 为什么这一层要"口径表"

数学与工程最大的差别不是"难不难"，而是**同一个符号在不同地方指不同东西**：

```text
norm(x)       L2 模长？L1？无穷范数？          （这一课只做 L2，并写下来）
cos(a, b)     零向量怎么办？                    （三处实现三种答案，见 errors 的说明）
scale         1/√d_k 还是 1/d_k？              （原论文是 1/√d_k，并写清了理由）
entropy       单位是 nats 还是 bits？           （本包一律 nats：与 exp/ln 同底）
```

因此本模块把八张表写成**可读的常量**，并在导入期校验它们逐键对齐：

```text
LINALG_OPS          十个线性代数算子（名字 / 一句话解释 / 公式）
ATTENTION_VARIANTS  四种注意力变体（点积 / 缩放 / 因果 / 多头）
BRIDGE_TARGETS      与项目里既有实现对照的八项（含"对照的是哪个函数"）
DISTRIBUTION_CHECKS 分布的四条判据（非空 / 非负 / 和为 1 / 长度一致）
CALCULUS_METHODS    三种差分方法（前向 / 后向 / 中心，含截断误差阶数）      ← day074
OPTIMIZERS          三种优化器（SGD / 动量 / Adam，含那一行更新公式）        ← day074
SCHEDULES           四种学习率调度（常数 / 阶梯 / 余弦 / 热身+余弦）          ← day074
GRADIENT_TARGETS    六项梯度对照（解析式 vs 数值差分）                     ← day074
```

后四张表是 day074 加的，它们的键与 day073 的四张**各自独立**：
梯度对照与函数值对照是两件事（一个比"值"，一个比"变化率"），
合表会让"某一项到底比的是哪一个"变成一个需要读实现才能回答的问题。

少一个键**不会让任何测试变红**，只会让某一项在报告里缺少解释——
而"缺一行"与"这一项没问题"在读的时候长得一样（与 day071／day072 的封闭表同源）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.math_foundations.errors import (
    MathError,
    NumericError,
    ParameterError,
    ShapeError,
)

#: 向量（不可变元组；本包一律用 tuple，避免调用方原地改掉一个已被"记过账"的量）.
Vector = tuple[float, ...]

#: 矩阵（行优先的元组；每行等长由 :func:`validate_matrix` 保证）.
Matrix = tuple[tuple[float, ...], ...]

#: 浮点比较的缺省容差。**不作为配置项**：容差是"这次比较的判据"，
#: 它进的是函数参数——把它写进配置会让人以为"改了容差就等于改了结论"。
DEFAULT_TOLERANCE = 1e-9

#: "和必须为 1" 一类判据的缺省容差（比逐位比较松一档：求和会累积舍入误差）.
DEFAULT_SUM_TOLERANCE = 1e-9

#: 双精度浮点数的机器精度 ``eps = 2.22e-16``（``sys.float_info.epsilon``）.
#:
#: 它是 day074 反复出现的那个常数：``1 + eps`` 恰好是"下一个可表示的浮点数"。
#: 写死的理由与 :data:`DEFAULT_TOLERANCE` 相同——它是一个**数学常数**，
#: 换了它就等于换了"这一层的浮点语义"；而且它必须只有一处定义，
#: 否则"差分的最优步长"（``calculus``）与"差分的分辨率下限"（``gradcheck``）
#: 会各自代入不同的 eps，于是两处的结论对不上。
FLOAT_EPSILON = 2.220446049250313e-16

# --------------------------------------------------------------------------- #
# 形状：向量与矩阵
# --------------------------------------------------------------------------- #


def is_finite(value: float) -> bool:
    """这个数是不是有限实数（NaN 与 ±inf 都不算）."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(
        float(value)
    )


def validate_vector(values: Any, *, name: str = "vector") -> Vector:
    """把一串数折成 :data:`Vector`，并校验"非空 + 全部有限".

    ```text
    空向量         没有维度，任何"比较两个向量"的操作对它都没有定义
    非数值         字符串、None、布尔会一路混到公式里变成 TypeError（栈更深、更难读）
    NaN / inf      它们会污染整条链路：一个 NaN 进来，均值、模长、相似度全变 NaN，
                   而"全 NaN"在一张表里看起来像"这一列没数据"
    ```

    刻意**不**接受 ``list``（而不是"接受并转成 tuple"）：原地可变的对象
    会在"记过账之后被改"，而报告里那份账与后来的向量还是同一份引用。
    """
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise ShapeError(
            f"{name} 必须是一串数（可迭代），收到 {type(values).__name__}——"
            "字符串与单个数会被下游的 zip 悄悄按字符/迭代拆开。"
        )
    items = tuple(values)
    if not items:
        raise ShapeError(f"{name} 不能为空：空向量没有维度，任何逐元素操作都没有定义。")
    checked: list[float] = []
    for index, item in enumerate(items):
        if not is_finite(item):
            raise NumericError(
                f"{name} 的第 {index} 个分量不是有限实数（收到 {item!r}）："
                "NaN / inf 会污染整条链路——一个 NaN 进来，均值、模长与相似度全变 NaN，"
                "而那一列在表里看起来像'没数据'。"
            )
        checked.append(float(item))
    return tuple(checked)


def validate_matrix(rows: Any, *, name: str = "matrix") -> Matrix:
    """把一串行折成 :data:`Matrix`，并校验"非空 + 每行等长".

    ``zip`` 的静默截断是这一课最想拦下的一件事：
    一个 3×4 的矩阵与一个 3×3 的矩阵相乘不会报错，只会**少算一列**——
    而结果看起来是一组正常的数。
    """
    if isinstance(rows, (str, bytes)) or not hasattr(rows, "__iter__"):
        raise ShapeError(
            f"{name} 必须是若干行（可迭代的行的序列），收到 {type(rows).__name__}。"
        )
    items = list(rows)
    if not items:
        raise ShapeError(f"{name} 不能为空：空矩阵没有形状。")
    checked = [validate_vector(row, name=f"{name}[{index}]") for index, row in enumerate(items)]
    width = len(checked[0])
    for index, row in enumerate(checked):
        if len(row) != width:
            raise ShapeError(
                f"{name} 的第 {index} 行长度是 {len(row)}，第一行是 {width}："
                "每行必须等长，否则逐元素运算会被 zip 静默截断（少算一列而不报错），"
                "而结果看起来仍然是一组正常的数。"
            )
    return tuple(checked)


def matrix_shape(matrix: Matrix) -> tuple[int, int]:
    """矩阵形状 ``(行数, 列数)``（空矩阵给 ``(0, 0)``，由调用方决定是否允许）."""
    if not matrix:
        return (0, 0)
    return (len(matrix), len(matrix[0]))


def is_square(matrix: Matrix) -> bool:
    """是不是方阵（行列数相等）."""
    rows, columns = matrix_shape(matrix)
    return rows > 0 and rows == columns


def require_square(matrix: Matrix, *, name: str = "matrix") -> None:
    """要求方阵（幂迭代、特征值一类的操作需要它）."""
    if not is_square(matrix):
        raise ShapeError(
            f"{name} 必须是方阵，收到 {matrix_shape(matrix)}："
            "非方阵的'特征向量'没有定义，而按行列分别迭代会得到两个都说不清楚的量。"
        )


def require_same_dimension(left: Vector, right: Vector, *, name: str = "向量") -> None:
    """要求两个向量等长（点积与相似度的前置条件）."""
    if len(left) != len(right):
        raise ShapeError(
            f"两个{name}维度不一致：{len(left)} != {len(right)}——"
            "点积与余弦相似度只在同一维空间里有定义，"
            "而按较短的那个 zip 出来的结果会是一个'看起来正常'的数。"
        )


# --------------------------------------------------------------------------- #
# 数值比较
# --------------------------------------------------------------------------- #


def close(left: float, right: float, *, tolerance: float = DEFAULT_TOLERANCE) -> bool:
    """两个数是否在容差内相等（相对 + 绝对混合判据，容差必须为正）."""
    if tolerance <= 0:
        raise ParameterError(f"容差必须为正，收到 {tolerance}：零容差会让浮点比较永远失败。")
    if not is_finite(left) or not is_finite(right):
        return False
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def relative_error(approximate: float, reference: float) -> float:
    """相对误差 ``|a − r| / max(1, |r|)``（分母取 max(1, ·) 让 0 附近不爆炸）.

    分母为什么不直接取 ``|r|``：一个"精确值是 0、近似值是 1e-12"的结果
    在数学上是完美逼近，而 ``|a−r|/|r|`` 会给出 1（100% 误差）。
    取 ``max(1, |r|)`` 把 0 附近的判据变成绝对误差——这正是我们要的语义。
    """
    if not is_finite(approximate) or not is_finite(reference):
        raise NumericError(
            f"相对误差的两个入参都必须是有限实数，收到 {approximate!r} 与 {reference!r}。"
        )
    return abs(approximate - reference) / max(1.0, abs(reference))


def normalize_rows(weights: Matrix, *, tolerance: float = DEFAULT_SUM_TOLERANCE) -> Matrix:
    """把每一行缩放到和为 1（不改变全零行：它在概率上没有意义，由调用方判）.

    注意力权重的每一行是一个**条件分布**（"这一行在多少程度上看各个位置"），
    而"和不为 1"的行会让后续的加权求和变成一个缩放错误的组合。
    """
    result: list[Vector] = []
    for index, row in enumerate(weights):
        total = math.fsum(row)
        if abs(total) <= tolerance:
            result.append(tuple(row))
            continue
        result.append(tuple(value / total for value in row))
    return tuple(result)


def row_sums(matrix: Matrix) -> Vector:
    """每一行之和（对角线与边缘分布的检查都读它）."""
    return tuple(math.fsum(row) for row in matrix)


# --------------------------------------------------------------------------- #
# 闭合表 1：十个线性代数算子
# --------------------------------------------------------------------------- #

OP_DOT = "dot"
OP_NORM = "norm"
OP_NORMALIZE = "normalize"
OP_COSINE = "cosine"
OP_MATMUL = "matmul"
OP_TRANSPOSE = "transpose"
OP_PROJECTION = "projection"
OP_SOFTMAX = "softmax"
OP_POWER_ITERATION = "power_iteration"
OP_LOW_RANK = "low_rank"

#: 十个算子（顺序 = 文档与报告里的排列 = 从"一个向量"到"一个矩阵的近似"）.
LINALG_OPS: tuple[str, ...] = (
    OP_DOT,
    OP_NORM,
    OP_NORMALIZE,
    OP_COSINE,
    OP_MATMUL,
    OP_TRANSPOSE,
    OP_PROJECTION,
    OP_SOFTMAX,
    OP_POWER_ITERATION,
    OP_LOW_RANK,
)

#: 每个算子的一句话解释（进文档与演示脚本）.
LINALG_OP_DESCRIPTIONS: dict[str, str] = {
    OP_DOT: "点积：两个向量逐分量相乘再相加，衡量'方向上的重合程度'（不含长度信息）",
    OP_NORM: "L2 模长：向量的长度，√Σx²",
    OP_NORMALIZE: "归一化：把向量缩放到单位长度（零向量无方向，本包直接拒绝）",
    OP_COSINE: "余弦相似度：点积除以两个模长，落在 [-1, 1]（零向量与任一向量的相似度记 0.0）",
    OP_MATMUL: "矩阵乘法：(m×n)·(n×p) → (m×p)，两次线性组合的复合",
    OP_TRANSPOSE: "转置：行列互换，(m×n) → (n×m)",
    OP_PROJECTION: "投影：把 u 投到 v 的方向上，标量 = u·v̂",
    OP_SOFTMAX: "softmax：把实数打分变成概率分布（先减最大值再取指数，数值稳定）",
    OP_POWER_ITERATION: "幂迭代：反复乘矩阵再归一化，主特征向量会自己浮现出来",
    OP_LOW_RANK: "低秩近似：用前 r 个奇异向量重建矩阵（LoRA 的数学原型）",
}

#: 每个算子的公式（**口径可读**：报告里能读到"这一项算的到底是哪个式子"）.
LINALG_OP_FORMULAS: dict[str, str] = {
    OP_DOT: "a·b = Σ_i a_i b_i",
    OP_NORM: "‖a‖ = √(Σ_i a_i²)",
    OP_NORMALIZE: "â = a / ‖a‖（‖â‖ = 1）",
    OP_COSINE: "cos(a,b) = a·b / (‖a‖‖b‖)",
    OP_MATMUL: "(AB)_ij = Σ_k A_ik B_kj",
    OP_TRANSPOSE: "(Aᵀ)_ij = A_ji",
    OP_PROJECTION: "proj_v(u) = (u·v̂) v̂",
    OP_SOFTMAX: "softmax(z)_i = exp(z_i) / Σ_j exp(z_j)",
    OP_POWER_ITERATION: "v_{k+1} = A v_k / ‖A v_k‖",
    OP_LOW_RANK: "A_r = Σ_{i≤r} σ_i u_i v_iᵀ",
}

if not (
    set(LINALG_OPS) == set(LINALG_OP_DESCRIPTIONS) == set(LINALG_OP_FORMULAS)
):
    raise MathError(
        "线性代数算子的三张表不一致：LINALG_OPS / LINALG_OP_DESCRIPTIONS / "
        "LINALG_OP_FORMULAS 必须逐键对齐，否则某个算子在文档里只有名字、"
        "没有解释，也没有它可以被复核的公式。"
    )

# --------------------------------------------------------------------------- #
# 闭合表 2：四种注意力变体
# --------------------------------------------------------------------------- #

ATTENTION_DOT = "dot"
ATTENTION_SCALED = "scaled"
ATTENTION_CAUSAL = "causal"
ATTENTION_MULTI_HEAD = "multi_head"

#: 四种变体（顺序 = 逐步加一个约束的顺序）.
ATTENTION_VARIANTS: tuple[str, ...] = (
    ATTENTION_DOT,
    ATTENTION_SCALED,
    ATTENTION_CAUSAL,
    ATTENTION_MULTI_HEAD,
)

#: 每种变体的一句话解释.
ATTENTION_VARIANT_DESCRIPTIONS: dict[str, str] = {
    ATTENTION_DOT: "朴素点积注意力：权重 = softmax(QKᵀ)，不缩放",
    ATTENTION_SCALED: "缩放点积注意力：权重 = softmax(QKᵀ / √d_k)（原论文的默认形式）",
    ATTENTION_CAUSAL: "因果注意力：上三角被屏蔽（位置 i 看不到 > i 的位置）",
    ATTENTION_MULTI_HEAD: "多头注意力：把 d 维拆成 h 段各自算，再拼回来",
}

if set(ATTENTION_VARIANTS) != set(ATTENTION_VARIANT_DESCRIPTIONS):
    raise MathError(
        "注意力变体的两张表不一致：ATTENTION_VARIANTS / ATTENTION_VARIANT_DESCRIPTIONS "
        "必须逐键对齐，否则某种变体在报告里只有名字、没有解释。"
    )

# --------------------------------------------------------------------------- #
# 闭合表 3：与既有实现的对照项
# --------------------------------------------------------------------------- #

BRIDGE_COSINE = "cosine"
BRIDGE_NORMALIZE = "normalize"
BRIDGE_SOFTMAX = "softmax"
BRIDGE_LOG_SOFTMAX = "log_softmax"
BRIDGE_CROSS_ENTROPY = "cross_entropy"
BRIDGE_PERPLEXITY = "perplexity"
BRIDGE_SIGMOID = "sigmoid"
BRIDGE_LOG_SIGMOID = "log_sigmoid"

#: 八项对照（顺序 = 从"几何"到"概率"）.
BRIDGE_TARGETS: tuple[str, ...] = (
    BRIDGE_COSINE,
    BRIDGE_NORMALIZE,
    BRIDGE_SOFTMAX,
    BRIDGE_LOG_SOFTMAX,
    BRIDGE_CROSS_ENTROPY,
    BRIDGE_PERPLEXITY,
    BRIDGE_SIGMOID,
    BRIDGE_LOG_SIGMOID,
)

#: 每一项对照的说明（它问的是"这两处算的是不是同一个东西"）.
BRIDGE_TARGET_DESCRIPTIONS: dict[str, str] = {
    BRIDGE_COSINE: "检索的相似度口径（day064）与本包的余弦是否逐位一致，含零向量约定",
    BRIDGE_NORMALIZE: "写入侧的 L2 归一化（day041）与本包的归一化是否一致，含零向量约定",
    BRIDGE_SOFTMAX: "训练侧的 softmax（day050）与采样侧的温度 softmax（day043）是否一致",
    BRIDGE_LOG_SOFTMAX: "训练侧的 log-softmax 与本包的实现是否一致（含极小概率下溢）",
    BRIDGE_CROSS_ENTROPY: "单点交叉熵：本包与 day050 的实现是否一致",
    BRIDGE_PERPLEXITY: "困惑度 = exp(loss)：本包与 day050 的实现是否一致",
    BRIDGE_SIGMOID: "对齐侧的 sigmoid（day054）与本包的实现是否一致（含上溢分支）",
    BRIDGE_LOG_SIGMOID: "log σ(x) = −softplus(−x)：本包与 day054 的实现是否一致",
}

#: 每一项对照的**来源函数**（写清"对照的是哪个模块里的哪一个函数"）.
BRIDGE_TARGET_SOURCES: dict[str, str] = {
    BRIDGE_COSINE: "smart_research_agent.vectorstore.metrics.cosine_similarity",
    BRIDGE_NORMALIZE: "smart_research_agent.llm.embedding.l2_normalize",
    BRIDGE_SOFTMAX: "smart_research_agent.sft.loss.softmax + llm.sampling.softmax_with_temperature",
    BRIDGE_LOG_SOFTMAX: "smart_research_agent.sft.loss.log_softmax",
    BRIDGE_CROSS_ENTROPY: "smart_research_agent.sft.loss.cross_entropy",
    BRIDGE_PERPLEXITY: "smart_research_agent.sft.loss.perplexity",
    BRIDGE_SIGMOID: "smart_research_agent.alignment.objectives.sigmoid",
    BRIDGE_LOG_SIGMOID: "smart_research_agent.alignment.objectives.log_sigmoid",
}

if not (
    set(BRIDGE_TARGETS)
    == set(BRIDGE_TARGET_DESCRIPTIONS)
    == set(BRIDGE_TARGET_SOURCES)
):
    raise MathError(
        "对照项的三张表不一致：BRIDGE_TARGETS / BRIDGE_TARGET_DESCRIPTIONS / "
        "BRIDGE_TARGET_SOURCES 必须逐键对齐——少一个键的对照项会静默地不被执行，"
        "而'没对照'与'对照通过'在报告里长得一样。"
    )

# --------------------------------------------------------------------------- #
# 闭合表 4：分布的判据
# --------------------------------------------------------------------------- #

CHECK_NON_EMPTY = "non_empty"
CHECK_NON_NEGATIVE = "non_negative"
CHECK_SUMS_TO_ONE = "sums_to_one"
CHECK_LENGTH_MATCHES_LABELS = "length_matches_labels"

#: 四条判据（顺序 = 检查顺序：先"有没有"，再"是不是概率"，最后"对不对得上标签"）.
DISTRIBUTION_CHECKS: tuple[str, ...] = (
    CHECK_NON_EMPTY,
    CHECK_NON_NEGATIVE,
    CHECK_SUMS_TO_ONE,
    CHECK_LENGTH_MATCHES_LABELS,
)

#: 每条判据的一句话解释.
DISTRIBUTION_CHECK_DESCRIPTIONS: dict[str, str] = {
    CHECK_NON_EMPTY: "分布不能为空（空分布的熵、期望、采样都没有定义）",
    CHECK_NON_NEGATIVE: "每个概率必须 >= 0（负概率会让熵变成复数、让采样区间错位）",
    CHECK_SUMS_TO_ONE: "全部概率之和必须为 1（容差内）——它不是'差不多就行'，而是公理",
    CHECK_LENGTH_MATCHES_LABELS: "概率个数必须与标签个数一致（否则哪个概率属于哪个事件说不清）",
}

if set(DISTRIBUTION_CHECKS) != set(DISTRIBUTION_CHECK_DESCRIPTIONS):
    raise MathError(
        "分布判据的两张表不一致：DISTRIBUTION_CHECKS / DISTRIBUTION_CHECK_DESCRIPTIONS "
        "必须逐键对齐，否则某条判据在报告里只有名字、没有解释。"
    )


def check_distribution(
    probabilities: Vector,
    *,
    labels: tuple[str, ...] = (),
    tolerance: float = DEFAULT_SUM_TOLERANCE,
) -> Vector:
    """跑完四条判据，返回规范过的概率向量（**校验只有一处实现**）.

    四条判据的顺序就是 :data:`DISTRIBUTION_CHECKS` 的顺序，
    且"命中即抛"——先报最根本的那一条（空表谈不上"是否为负"）。
    """
    if not probabilities:
        raise NumericError(
            f"分布不能为空（{CHECK_NON_EMPTY}）：空分布的熵、期望与采样都没有定义。"
        )
    for index, value in enumerate(probabilities):
        if value < 0:
            raise NumericError(
                f"第 {index} 个概率是负数（{value}，{CHECK_NON_NEGATIVE}）："
                "负概率会让熵变成复数、让按累积概率采样的区间错位，"
                "而'一组和为 1 的数'看起来仍然正常。"
            )
    total = math.fsum(probabilities)
    if abs(total - 1.0) > tolerance:
        raise NumericError(
            f"概率之和是 {total!r}，不是 1（容差 {tolerance}，{CHECK_SUMS_TO_ONE}）："
            "归一化是公理而不是建议——差 1e-3 的分布会在几十步采样之后"
            "把概率质量全部推到最后一个事件上。"
        )
    if labels and len(labels) != len(probabilities):
        raise NumericError(
            f"概率个数 {len(probabilities)} 与标签个数 {len(labels)} 不一致"
            f"（{CHECK_LENGTH_MATCHES_LABELS}）：否则'哪个概率属于哪个事件'说不清，"
            "而 zip 会静默截断成较短的那一边。"
        )
    return probabilities


# --------------------------------------------------------------------------- #
# 闭合表 5：三种差分方法（day074）
# --------------------------------------------------------------------------- #

FORWARD = "forward"
BACKWARD = "backward"
CENTRAL = "central"

#: 三种差分方法（顺序 = 从"最少一次求值"到"多一次求值换精度"）.
CALCULUS_METHODS: tuple[str, ...] = (FORWARD, BACKWARD, CENTRAL)

#: 每种方法的一句话解释.
CALCULUS_METHOD_DESCRIPTIONS: dict[str, str] = {
    FORWARD: "前向差分：(f(x+h) − f(x)) / h，一次额外求值，截断误差 O(h)",
    BACKWARD: "后向差分：(f(x) − f(x−h)) / h，一次额外求值，截断误差 O(h)；"
    "在 f 只在左侧有定义时（如 log 在 0 处）唯一可用",
    CENTRAL: "中心差分：(f(x+h) − f(x−h)) / (2h)，两次额外求值，截断误差 O(h²)（本包缺省）",
}

#: 每种方法的公式（报告里能读到"这一项算的到底是哪个式子"）.
CALCULUS_METHOD_FORMULAS: dict[str, str] = {
    FORWARD: "f'(x) ≈ (f(x+h) − f(x)) / h",
    BACKWARD: "f'(x) ≈ (f(x) − f(x−h)) / h",
    CENTRAL: "f'(x) ≈ (f(x+h) − f(x−h)) / (2h)",
}

#: 每种方法的**截断误差阶数**（它是"h 每缩小 10 倍误差缩小多少倍"的答案）.
CALCULUS_METHOD_ORDERS: dict[str, str] = {
    FORWARD: "O(h)",
    BACKWARD: "O(h)",
    CENTRAL: "O(h²)",
}

#: 两种误差阶数的字面量（测试与报告里比对它们，而不是各写一遍字符串）.
CALCULUS_ORDER_FORWARD = "O(h)"
CALCULUS_ORDER_CENTRAL = "O(h²)"

if not (
    set(CALCULUS_METHODS)
    == set(CALCULUS_METHOD_DESCRIPTIONS)
    == set(CALCULUS_METHOD_FORMULAS)
    == set(CALCULUS_METHOD_ORDERS)
):
    raise MathError(
        "差分方法的三张表不一致：CALCULUS_METHODS / CALCULUS_METHOD_DESCRIPTIONS / "
        "CALCULUS_METHOD_FORMULAS / CALCULUS_METHOD_ORDERS 必须逐键对齐——"
        "少一个键的那一种方法会静默地没有公式可复核，"
        "而'没有公式'与'公式对得上'在读的时候长得一样。"
    )

if CALCULUS_METHOD_ORDERS[CENTRAL] != CALCULUS_ORDER_CENTRAL:  # pragma: no cover - 导入期不变式
    raise MathError("中心差分的阶数与常量 CALCULUS_ORDER_CENTRAL 不一致。")

# --------------------------------------------------------------------------- #
# 闭合表 6：三种优化器（day074）
# --------------------------------------------------------------------------- #

OPTIMIZER_SGD = "sgd"
OPTIMIZER_MOMENTUM = "momentum"
OPTIMIZER_ADAM = "adam"

#: 三种优化器（顺序 = 逐步加一个状态：无状态 → 一阶动量 → 一阶 + 二阶动量）.
OPTIMIZERS: tuple[str, ...] = (OPTIMIZER_SGD, OPTIMIZER_MOMENTUM, OPTIMIZER_ADAM)

#: 每种优化器的一句话解释.
OPTIMIZER_DESCRIPTIONS: dict[str, str] = {
    OPTIMIZER_SGD: "随机梯度下降：θ ← θ − lr·g（无状态，每个方向的步长只由当步梯度决定）",
    OPTIMIZER_MOMENTUM: "动量：v ← βv − lr·g，θ ← θ + v（把历史梯度累成速度，冲过平坦区与抖动）",
    OPTIMIZER_ADAM: "Adam：一阶动量 m 与二阶动量 v 各带偏差修正，θ ← θ − lr·m̂/(√v̂ + ε)"
    "（每个参数有各自的等效步长）",
}

#: 每种优化器的更新公式（**这就是可以被逐行读出来的算法**）.
OPTIMIZER_UPDATE_FORMULAS: dict[str, str] = {
    OPTIMIZER_SGD: "g = ∇f(θ);  θ ← θ − lr·g",
    OPTIMIZER_MOMENTUM: "g = ∇f(θ);  v ← β·v + g;  θ ← θ − lr·v",
    OPTIMIZER_ADAM: "m ← β₁m + (1−β₁)g;  v ← β₂v + (1−β₂)g²;  "
    "m̂ = m/(1−β₁ᵗ);  v̂ = v/(1−β₂ᵗ);  θ ← θ − lr·m̂/(√v̂ + ε)",
}

if not (
    set(OPTIMIZERS) == set(OPTIMIZER_DESCRIPTIONS) == set(OPTIMIZER_UPDATE_FORMULAS)
):
    raise MathError(
        "优化器的三张表不一致：OPTIMIZERS / OPTIMIZER_DESCRIPTIONS / "
        "OPTIMIZER_UPDATE_FORMULAS 必须逐键对齐，否则某种优化器在报告里"
        "只有名字、没有它实际执行的那一行更新。"
    )

# --------------------------------------------------------------------------- #
# 闭合表 7：四种学习率调度（day074）
# --------------------------------------------------------------------------- #

SCHEDULE_CONSTANT = "constant"
SCHEDULE_STEP_DECAY = "step_decay"
SCHEDULE_COSINE = "cosine"
SCHEDULE_WARMUP_COSINE = "warmup_cosine"

#: 四种调度（顺序 = 从"不调"到"先热身再退火"）.
SCHEDULES: tuple[str, ...] = (
    SCHEDULE_CONSTANT,
    SCHEDULE_STEP_DECAY,
    SCHEDULE_COSINE,
    SCHEDULE_WARMUP_COSINE,
)

#: 每种调度的一句话解释.
SCHEDULE_DESCRIPTIONS: dict[str, str] = {
    SCHEDULE_CONSTANT: "常数学习率：最朴素也最有用的基线——没有它，'调度有没有用'无法回答",
    SCHEDULE_STEP_DECAY: "阶梯衰减：每 drop_every 步乘一次 gamma（工程上最常见的粗调度）",
    SCHEDULE_COSINE: "余弦退火：从 base_lr 平滑降到 min_lr（后期小步走，收敛更稳）",
    SCHEDULE_WARMUP_COSINE: "线性热身 + 余弦退火：前期把 lr 从 0 拉起来（避免一开始就打乱随机初始化）",
}

#: 每种调度的公式（``t`` 从 1 开始计数，与"第几步"的日常说法一致）.
SCHEDULE_FORMULAS: dict[str, str] = {
    SCHEDULE_CONSTANT: "lr(t) = base_lr",
    SCHEDULE_STEP_DECAY: "lr(t) = base_lr · gamma^{⌊(t−1)/drop_every⌋}",
    SCHEDULE_COSINE: "lr(t) = min_lr + (base_lr − min_lr)·(1 + cos(π·(t−1)/(total_steps−1)))/2",
    SCHEDULE_WARMUP_COSINE: "t ≤ warmup: lr(t) = base_lr·t/warmup；"
    "之后按余弦从 base_lr 降到 min_lr",
}

if not (set(SCHEDULES) == set(SCHEDULE_DESCRIPTIONS) == set(SCHEDULE_FORMULAS)):
    raise MathError(
        "学习率调度的三张表不一致：SCHEDULES / SCHEDULE_DESCRIPTIONS / SCHEDULE_FORMULAS "
        "必须逐键对齐——少一个键的调度会静默地没有公式，"
        "而'没有公式'与'公式对得上'在读的时候长得一样。"
    )

# --------------------------------------------------------------------------- #
# 闭合表 8：梯度对照项（day074）
# --------------------------------------------------------------------------- #

GRADIENT_SOFTMAX = "softmax"
GRADIENT_CROSS_ENTROPY = "cross_entropy"
GRADIENT_SIGMOID = "sigmoid"
GRADIENT_LOG_SIGMOID = "log_sigmoid"
GRADIENT_PERPLEXITY = "perplexity"
GRADIENT_AUTOGRAD_CHAIN = "autograd_chain"

#: 六项梯度对照（顺序 = 从"一个矩阵的雅可比"到"一整条复合链"）.
GRADIENT_TARGETS: tuple[str, ...] = (
    GRADIENT_SOFTMAX,
    GRADIENT_CROSS_ENTROPY,
    GRADIENT_SIGMOID,
    GRADIENT_LOG_SIGMOID,
    GRADIENT_PERPLEXITY,
    GRADIENT_AUTOGRAD_CHAIN,
)

#: 每一项的解析梯度公式（**这一课的主角**：它必须与数值梯度对上）.
GRADIENT_TARGET_FORMULAS: dict[str, str] = {
    GRADIENT_SOFTMAX: "∂p_i/∂z_j = p_i(δ_ij − p_j)（softmax 的雅可比）",
    GRADIENT_CROSS_ENTROPY: "∂(−log p_y)/∂z = p − onehot(y)（day054 出现过的那一行）",
    GRADIENT_SIGMOID: "σ'(x) = σ(x)(1 − σ(x))",
    GRADIENT_LOG_SIGMOID: "(log σ)'(x) = σ(−x) = 1 − σ(x)",
    GRADIENT_PERPLEXITY: "d(exp(L))/dL = exp(L)",
    GRADIENT_AUTOGRAD_CHAIN: "复合函数的导数 = 各层局部导数之积（链式法则）",
}

#: 每一项的说明（对照的是"哪一处解析式"与"哪一处数值差分"）.
GRADIENT_TARGET_DESCRIPTIONS: dict[str, str] = {
    GRADIENT_SOFTMAX: "softmax 的雅可比解析式 vs 数值雅可比（它解释了为什么交叉熵的梯度那么简洁）",
    GRADIENT_CROSS_ENTROPY: "∂(−log p_y)/∂z = p − onehot(y) vs 对生产实现 sft.loss.cross_entropy 的数值差分",
    GRADIENT_SIGMOID: "σ(1−σ) vs 对生产实现 alignment.objectives.sigmoid 的数值差分",
    GRADIENT_LOG_SIGMOID: "σ(−x) vs 对生产实现 alignment.objectives.log_sigmoid 的数值差分",
    GRADIENT_PERPLEXITY: "d(exp L)/dL = exp L vs 对生产实现 sft.loss.perplexity 的数值差分",
    GRADIENT_AUTOGRAD_CHAIN: "autograd 反向传播算出的梯度 vs 对同一复合函数的数值差分（验证链式法则的自动化）",
}

#: 每一项对照的**来源**（写清"对照的是哪个模块里的哪一个东西"）.
GRADIENT_TARGET_SOURCES: dict[str, str] = {
    GRADIENT_SOFTMAX: "math_foundations.linalg.softmax（解析） vs math_foundations.calculus.jacobian（数值）",
    GRADIENT_CROSS_ENTROPY: "math_foundations.linalg.softmax 解析 vs smart_research_agent.sft.loss.cross_entropy 数值",
    GRADIENT_SIGMOID: "math_foundations.bridge 的 sigmoid 解析 vs smart_research_agent.alignment.objectives.sigmoid 数值",
    GRADIENT_LOG_SIGMOID: "math_foundations.bridge 的 log_sigmoid 解析 vs smart_research_agent.alignment.objectives.log_sigmoid 数值",
    GRADIENT_PERPLEXITY: "math_foundations.probability.perplexity_from_entropy 解析 vs smart_research_agent.sft.loss.perplexity 数值",
    GRADIENT_AUTOGRAD_CHAIN: "math_foundations.autograd.Scalar.backward 解析 vs math_foundations.calculus 的数值差分",
}

#: 梯度对照的两种结论（**比函数值对照少一种**，理由写在下面）.
GRADIENT_STATUSES: tuple[str, ...] = ("agrees", "differs")

if not (
    set(GRADIENT_TARGETS)
    == set(GRADIENT_TARGET_DESCRIPTIONS)
    == set(GRADIENT_TARGET_SOURCES)
    == set(GRADIENT_TARGET_FORMULAS)
):
    raise MathError(
        "梯度对照项的四张表不一致：GRADIENT_TARGETS / GRADIENT_TARGET_DESCRIPTIONS / "
        "GRADIENT_TARGET_SOURCES / GRADIENT_TARGET_FORMULAS 必须逐键对齐——"
        "少一个键的对照项会静默地不被执行，而'没对照'与'对照通过'在报告里长得一样。"
    )

# --------------------------------------------------------------------------- #
# 向量画像与注意力报告
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class VectorSummary:
    """一个向量的画像：维度、模长、均值、极值、是不是单位向量.

    均值与极值不是"顺手算的"：它们在排查时回答两个具体问题——
    "这个向量是不是被压成了一堆相近的数"（均值与极值挤在一起）
    与"有没有某个分量异常大"（max 远大于其余）。
    """

    dimension: int
    norm: float
    mean: float
    minimum: float
    maximum: float
    is_unit: bool

    @classmethod
    def of(cls, values: Vector, *, tolerance: float = 1e-6) -> VectorSummary:
        """按一个向量算出画像（``is_unit`` 用容差判定，不用精确相等）."""
        checked = validate_vector(values, name="vector")
        norm = math.sqrt(math.fsum(value * value for value in checked))
        return cls(
            dimension=len(checked),
            norm=norm,
            mean=math.fsum(checked) / len(checked),
            minimum=min(checked),
            maximum=max(checked),
            is_unit=close(norm, 1.0, tolerance=tolerance),
        )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "dimension": self.dimension,
            "norm": self.norm,
            "mean": self.mean,
            "min": self.minimum,
            "max": self.maximum,
            "is_unit": self.is_unit,
        }

    def summary_line(self) -> str:
        """一行说明：``d=3 | ‖x‖=1.000000 | 均值 0.0000 | 范围 [-1.0000, 1.0000] | 单位向量``."""
        unit = "单位向量" if self.is_unit else "非单位向量"
        return (
            f"d={self.dimension} | ‖x‖={self.norm:.6f} | 均值 {self.mean:.4f} | "
            f"范围 [{self.minimum:.4f}, {self.maximum:.4f}] | {unit}"
        )


@dataclass(frozen=True)
class AttentionReport:
    """一次注意力的账：权重矩阵 + 输出 + 每一行分布的熵与峰值.

    ```text
    weights      形状 (queries, keys)，**每一行是一个和为 1 的分布**
    output       形状 (queries, head_dim)，= weights · V
    entropies    每一行分布的熵（**nats**）：0 = 全押一个位置，ln(keys) = 均匀
    peak_weights 每一行的最大权重（"这个位置最看重谁，看重多少"）
    peak_indices 每一行的 argmax（"最看重的是哪一个位置"）
    ```

    两个派生量放在这里而不是让每个调用方各算一遍：**熵与峰值是同一份权重的
    两种读法**，两处实现迟早会在某一天给出不一致的值，而那种不一致看起来
    只是"某个位置有点怪"。
    """

    variant: str
    queries: int
    keys: int
    head_dim: int
    scale: float
    weights: Matrix
    output: Matrix
    entropies: Vector
    peak_weights: Vector
    peak_indices: tuple[int, ...]
    causal: bool
    temperature: float = 1.0
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.variant not in ATTENTION_VARIANTS:
            raise MathError(
                f"不认识的注意力变体 {self.variant!r}：可用取值 {list(ATTENTION_VARIANTS)}。"
            )
        rows, columns = matrix_shape(self.weights)
        if rows != self.queries or columns != self.keys:
            raise ShapeError(
                f"权重矩阵形状 {matrix_shape(self.weights)} 与声明的 "
                f"({self.queries}, {self.keys}) 不一致。"
            )
        for index, total in enumerate(row_sums(self.weights)):
            if not close(total, 1.0, tolerance=1e-6):
                raise NumericError(
                    f"权重第 {index} 行的和是 {total!r}，不是 1："
                    "注意力权重是一组条件分布（'这一行怎么看各个位置'），"
                    "和不为 1 会让后续加权求和变成一个缩放错误的组合。"
                )
        if not (
            len(self.entropies)
            == len(self.peak_weights)
            == len(self.peak_indices)
            == self.queries
        ):
            raise ShapeError(
                "entropies / peak_weights / peak_indices 的长度必须都等于 queries "
                f"（收到 {len(self.entropies)} / {len(self.peak_weights)} / "
                f"{len(self.peak_indices)}，queries={self.queries}）。"
            )
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def mean_entropy(self) -> float:
        """各行的平均熵（"这一层的注意力整体上有多集中"）."""
        if not self.entropies:
            return 0.0
        return math.fsum(self.entropies) / len(self.entropies)

    def max_entropy(self) -> float:
        """理论上界：``ln(keys)``（均匀分布时的熵，单位 nats）."""
        return math.log(self.keys) if self.keys > 0 else 0.0

    def focus_ratio(self) -> float:
        """集中度 ``1 − 平均熵 / ln(keys)``：0 = 完全均匀，1 = 每行全押一个位置.

        ``keys == 1`` 时返回 0.0（只有一列时"集中"这个词没有意义）。
        """
        ceiling = self.max_entropy()
        if ceiling <= 0:
            return 0.0
        return max(0.0, 1.0 - self.mean_entropy / ceiling)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（含派生量与范围说明）."""
        return {
            "variant": self.variant,
            "description": ATTENTION_VARIANT_DESCRIPTIONS[self.variant],
            "queries": self.queries,
            "keys": self.keys,
            "head_dim": self.head_dim,
            "scale": self.scale,
            "temperature": self.temperature,
            "causal": self.causal,
            "mean_entropy": self.mean_entropy,
            "max_entropy": self.max_entropy(),
            "focus_ratio": self.focus_ratio(),
            "weights": [list(row) for row in self.weights],
            "output": [list(row) for row in self.output],
            "entropies": list(self.entropies),
            "peak_weights": list(self.peak_weights),
            "peak_indices": list(self.peak_indices),
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``scaled | (2×3)·3 | 平均熵 0.7123 nats | 集中度 35.1%``."""
        return (
            f"{self.variant} | ({self.queries}×{self.keys})·{self.head_dim} | "
            f"平均熵 {self.mean_entropy:.4f} nats | 集中度 {self.focus_ratio():.1%} | "
            f"scale={self.scale:.6f}"
        )


__all__ = [
    "ATTENTION_CAUSAL",
    "ATTENTION_DOT",
    "ATTENTION_MULTI_HEAD",
    "ATTENTION_SCALED",
    "ATTENTION_VARIANTS",
    "ATTENTION_VARIANT_DESCRIPTIONS",
    "BACKWARD",
    "BRIDGE_COSINE",
    "BRIDGE_CROSS_ENTROPY",
    "BRIDGE_LOG_SIGMOID",
    "BRIDGE_LOG_SOFTMAX",
    "BRIDGE_NORMALIZE",
    "BRIDGE_PERPLEXITY",
    "BRIDGE_SIGMOID",
    "BRIDGE_SOFTMAX",
    "BRIDGE_TARGETS",
    "BRIDGE_TARGET_DESCRIPTIONS",
    "BRIDGE_TARGET_SOURCES",
    "CALCULUS_METHODS",
    "CALCULUS_METHOD_DESCRIPTIONS",
    "CALCULUS_METHOD_FORMULAS",
    "CALCULUS_METHOD_ORDERS",
    "CALCULUS_ORDER_CENTRAL",
    "CALCULUS_ORDER_FORWARD",
    "CENTRAL",
    "CHECK_LENGTH_MATCHES_LABELS",
    "CHECK_NON_EMPTY",
    "CHECK_NON_NEGATIVE",
    "CHECK_SUMS_TO_ONE",
    "DEFAULT_SUM_TOLERANCE",
    "DEFAULT_TOLERANCE",
    "DISTRIBUTION_CHECKS",
    "DISTRIBUTION_CHECK_DESCRIPTIONS",
    "FLOAT_EPSILON",
    "FORWARD",
    "GRADIENT_AUTOGRAD_CHAIN",
    "GRADIENT_CROSS_ENTROPY",
    "GRADIENT_LOG_SIGMOID",
    "GRADIENT_PERPLEXITY",
    "GRADIENT_SIGMOID",
    "GRADIENT_SOFTMAX",
    "GRADIENT_STATUSES",
    "GRADIENT_TARGETS",
    "GRADIENT_TARGET_DESCRIPTIONS",
    "GRADIENT_TARGET_FORMULAS",
    "GRADIENT_TARGET_SOURCES",
    "LINALG_OPS",
    "LINALG_OP_DESCRIPTIONS",
    "LINALG_OP_FORMULAS",
    "OP_COSINE",
    "OP_DOT",
    "OP_LOW_RANK",
    "OP_MATMUL",
    "OP_NORMALIZE",
    "OP_NORM",
    "OP_POWER_ITERATION",
    "OP_PROJECTION",
    "OP_SOFTMAX",
    "OP_TRANSPOSE",
    "OPTIMIZERS",
    "OPTIMIZER_ADAM",
    "OPTIMIZER_DESCRIPTIONS",
    "OPTIMIZER_MOMENTUM",
    "OPTIMIZER_SGD",
    "OPTIMIZER_UPDATE_FORMULAS",
    "SCHEDULES",
    "SCHEDULE_CONSTANT",
    "SCHEDULE_COSINE",
    "SCHEDULE_DESCRIPTIONS",
    "SCHEDULE_FORMULAS",
    "SCHEDULE_STEP_DECAY",
    "SCHEDULE_WARMUP_COSINE",
    "AttentionReport",
    "Matrix",
    "Vector",
    "VectorSummary",
    "check_distribution",
    "close",
    "is_finite",
    "is_square",
    "matrix_shape",
    "normalize_rows",
    "relative_error",
    "require_same_dimension",
    "require_square",
    "row_sums",
    "validate_matrix",
    "validate_vector",
]
