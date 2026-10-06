"""注意力：把"检索的相关性"写成矩阵乘法（day073 / Math-D1）.

这一课的落点。它的核心只有一行公式，而这一行里每一个符号都在前两半出现过：

```text
Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) · V
                     └────┬────┘  └─┬─┘   └┬┘
                     matmul    scaling    softmax（概率）· matmul（加权平均）
```

四步拆开看，每一步都是已经学过的东西：

```text
① Q Kᵀ            每一行是一个 query 对所有 key 的**打分**（点积 = 方向重合度）
② / √d_k          把打分的方差拉回 1（否则 d 一大，softmax 就饱和成 one-hot）
③ softmax         把打分变成**每一行和为 1 的分布**（row_normalize 的概率版）
④ · V             用这个分布对 value 做**加权平均**——输出是"按相关性混合出来的一段"
```

## 为什么第 ② 步是这一课最该被算清楚的一步

如果 ``q`` 与 ``k`` 的每个分量都是均值 0、方差 ``σ²`` 的独立随机数，
那么点积 ``q·k = Σ_{i} q_i k_i`` 的方差是 ``d_k σ⁴``——**随维度线性增长**：

```text
d_k = 4     打分标准差 ≈ 0.67    → softmax 还能给出有形状的分布
d_k = 64    打分标准差 ≈ 2.67    → 分布开始明显偏向最大值
d_k = 256   打分标准差 ≈ 5.33    → 分布几乎变成 one-hot（梯度消失）
```

除以 ``√d_k`` 之后，无论维度多高，打分标准差都回到 ``σ²``——
这就是"缩放点积注意力"里那个"缩放"的全部理由。
本模块用 :func:`sampled_dot_product_variance` 把这件事**量出来**（可复现的实验），
而不是只写在注释里。

## 与检索的类比（这一课最值钱的一句话）

```text
检索：cos(query, doc) 排序 → 取前 k 条 → 拼进提示词 → 交给模型（**离散的选择**）
注意力：softmax(q·k) 打分 → 得到一组权重 → 对 value 加权平均（**可微的混合**）
```

两者都在回答"哪些内容与当前问题相关"，差别在于**注意力是可微的**：
它不做"取前 k 条"这个不可导的取舍，而是把"想要什么"变成一个概率分布——
于是梯度可以沿着"权重该调大还是调小"一路回传。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.math_foundations.errors import NumericError, ParameterError
from smart_research_agent.math_foundations.linalg import (
    argmax,
    matmul,
    softmax,
    transpose,
)
from smart_research_agent.math_foundations.probability import (
    entropy,
    uniforms,
)
from smart_research_agent.math_foundations.types import (
    ATTENTION_CAUSAL,
    ATTENTION_DOT,
    ATTENTION_MULTI_HEAD,
    ATTENTION_SCALED,
    ATTENTION_VARIANT_DESCRIPTIONS,
    AttentionReport,
    Matrix,
    Vector,
    matrix_shape,
    validate_matrix,
)

#: 位置编码的下限频率基数（原论文用的是 10000）.
POSITIONAL_BASE = 10000.0


def scaling_factor(head_dim: int) -> float:
    """缩放系数 ``1/√d_k``（``head_dim < 1`` 抛 ``ParameterError``）."""
    if head_dim < 1:
        raise ParameterError(f"head_dim 必须 >= 1，收到 {head_dim}。")
    return 1.0 / math.sqrt(head_dim)


def attention_scores(
    queries: Matrix,
    keys: Matrix,
    *,
    scale: float | None = None,
) -> Matrix:
    """打分矩阵 ``Q Kᵀ / √d_k``（形状 ``(n_q, n_k)``）.

    ``queries`` 与 ``keys`` 的最后一维（每个头的维度 ``d_k``）必须相等：
    注意力打分的定义就是同一个空间里两个向量的点积——
    维数不同时"点积"根本没有定义（不是"少算几项"）。
    """
    checked_queries = validate_matrix(queries, name="queries")
    checked_keys = validate_matrix(keys, name="keys")
    head_dim = matrix_shape(checked_queries)[1]
    if matrix_shape(checked_keys)[1] != head_dim:
        raise NumericError(
            f"Q 与 K 的最后一维必须相同：{matrix_shape(checked_queries)} 与 "
            f"{matrix_shape(checked_keys)}——打分的定义是同一个空间里的点积，"
            "维度不同时它没有定义。"
        )
    resolved = scaling_factor(head_dim) if scale is None else float(scale)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ParameterError(f"缩放系数必须是正的有限数，收到 {resolved!r}。")
    scores = matmul(checked_queries, transpose(checked_keys))
    return tuple(tuple(value * resolved for value in row) for row in scores)


def causal_mask(size: int) -> tuple[tuple[bool, ...], ...]:
    """因果掩码：``mask[i][j]`` 为 ``True`` 表示"位置 i 允许看位置 j"（``j <= i``）.

    返回布尔矩阵而不是"被屏蔽的坐标列表"：掩码本身要能被打印出来看
    （一个形状不对的掩码会让整条链路悄悄多看到几个位置）。
    """
    if size < 1:
        raise ParameterError(f"掩码边长必须 >= 1，收到 {size}。")
    return tuple(tuple(column <= row for column in range(size)) for row in range(size))


def masked_softmax_rows(
    scores: Matrix,
    mask: tuple[tuple[bool, ...], ...],
) -> Matrix:
    """按行做"带掩码的 softmax"：被屏蔽的位置权重**恰好是 0.0**，且不计入分母.

    ## 为什么不把被屏蔽的位置写成 ``-inf``（这是本模块最想讲清的一处）

生产实现通常这么写（向量化友好）：

```python
scores = scores + (1.0 - mask) * float("-inf")   # 被屏蔽的位置变成 -inf
weights = softmax(scores)                        # exp(-inf) = 0，正好
```

本包**不这么做**，理由是两条具体的：

```text
① 非有限数不该进入概率层   softmax 会拒绝 -inf（见 linalg.softmax 的说明），
                          而"只在这一处放行"会让'非有限数一律拒绝'这条纪律
                          出现第一个例外——例外会变成第二个、第三个
② -inf 会被后续运算静默破坏  -inf · 0 = nan、-inf + inf = nan、
                          归一化/裁剪/温度缩放都会在 -inf 上出问题，
                          而那时错误出现在离掩码很远的地方
```

数学上两种做法**完全等价**：被屏蔽的位置在 softmax 之后权重就是 0
（``exp(-∞) = 0``），并且不参与分母。因此这里用**显式掩码**：
只在"允许看"的那些位置上做 softmax，再把 0 插回被屏蔽的位置。
好处是"每一行仍然是一个合法的概率分布"，而"合法"这个词在这一层是有定义的
（见 ``types.check_distribution`` 的四条判据）。
    """
    checked = validate_matrix(scores, name="scores")
    rows, columns = matrix_shape(checked)
    if len(mask) != rows or (rows and len(mask[0]) != columns):
        raise NumericError(
            f"掩码形状 {len(mask)}×{len(mask[0]) if mask else 0} 与打分矩阵 "
            f"{matrix_shape(checked)} 不一致。"
        )
    result: list[Vector] = []
    for row in range(rows):
        allowed = [checked[row][column] for column in range(columns) if mask[row][column]]
        if not allowed:
            raise NumericError(
                f"第 {row} 行没有任何允许的位置：这一行的注意力分布没有定义——"
                "因果掩码应该保证第 0 行至少能看到自己（mask[0][0] 必须为 True）。"
            )
        probabilities = iter(softmax(allowed))
        result.append(
            tuple(next(probabilities) if mask[row][column] else 0.0 for column in range(columns))
        )
    return tuple(result)


@dataclass(frozen=True)
class DotProductStudy:
    """一次"打分方差随维度增长"的实测结果（可复现的数值实验）.

    ```text
    measured_variance      实测的 Var(q·k)（在 trials 次抽样上统计）
    theoretical_variance   理论值 d · Var(q_i) · Var(k_i)（分量独立时）
    std                    实测标准差 = √measured_variance（"打分有多散"）
    ```

    三者一起给出来是为了让"公式"与"实测"互相印证：
    只给实测值，读的人无法判断它对不对；只给公式，读的人看不到它真的成立。
    """

    dimension: int
    trials: int
    component_bound: float
    measured_variance: float
    measured_mean: float
    theoretical_variance: float

    @property
    def std(self) -> float:
        """实测标准差（未缩放时打分的典型量级）."""
        return math.sqrt(self.measured_variance)

    @property
    def scaled_std(self) -> float:
        """除以 ``√d`` 之后的标准差（理论上是常数，与 d 无关）."""
        return self.std / math.sqrt(self.dimension)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "dimension": self.dimension,
            "trials": self.trials,
            "component_bound": self.component_bound,
            "measured_mean": self.measured_mean,
            "measured_variance": self.measured_variance,
            "theoretical_variance": self.theoretical_variance,
            "std": self.std,
            "scaled_std": self.scaled_std,
        }

    def summary_line(self) -> str:
        """一行说明：``d=64 | Var 7.0812（理论 7.1111）| 标准差 2.6610 | 缩放后 0.3326``."""
        return (
            f"d={self.dimension} | Var {self.measured_variance:.4f}"
            f"（理论 {self.theoretical_variance:.4f}）| 标准差 {self.std:.4f} | "
            f"缩放后 {self.scaled_std:.4f}"
        )


def sampled_dot_product_variance(
    dimension: int,
    *,
    trials: int = 1000,
    seed: int = 42,
    component_bound: float = 1.0,
) -> DotProductStudy:
    """量一次"随机向量的点积方差"（**这就是 √d_k 缩放的理由**）.

    抽样方式：每个分量独立取 ``[-bound, bound)`` 上的均匀数
    （因此每个分量的方差是 ``bound²/3``），然后算 ``q·k``。

    理论值 ``Var(q·k) = d · Var(q_i) · Var(k_i) = d · bound⁴/9``：
    两个独立分量的乘积的期望是 0、方差是两者方差之积，
    而 ``d`` 个独立项相加，方差直接相加——**这就是"随维度线性增长"的来源**。

    用的是 :func:`probability.uniforms` 那串确定性随机数，
    因此同一个 ``seed`` 永远得到同一组统计量（"这个实验做过"可以被复核）。
    """
    if dimension < 1:
        raise ParameterError(f"dimension 必须 >= 1，收到 {dimension}。")
    if trials < 1:
        raise ParameterError(f"trials 必须 >= 1，收到 {trials}。")
    if component_bound <= 0:
        raise ParameterError(f"component_bound 必须为正，收到 {component_bound}。")
    total = 2 * dimension * trials
    raw = uniforms(total, seed=seed)
    values = tuple((value * 2.0 - 1.0) * component_bound for value in raw)
    products: list[float] = []
    for trial in range(trials):
        base = trial * 2 * dimension
        left = values[base : base + dimension]
        right = values[base + dimension : base + 2 * dimension]
        products.append(math.fsum(x * y for x, y in zip(left, right)))
    count = len(products)
    mean = math.fsum(products) / count
    measured = math.fsum((value - mean) ** 2 for value in products) / count
    component_variance = component_bound**2 / 3.0
    return DotProductStudy(
        dimension=dimension,
        trials=trials,
        component_bound=component_bound,
        measured_variance=measured,
        measured_mean=mean,
        theoretical_variance=dimension * component_variance * component_variance,
    )


def _softmax_rows(scores: Matrix) -> Matrix:
    """按行做 softmax（**行是 query，列是 key**：每一行是一个条件分布）.

    这里复用 :func:`linalg.softmax` 的实现而不是自己写一遍：
    "先减最大值"这一步必须两处一致，否则一处在极端打分上给出 ``nan``、
    另一处给出正确值——而"两处算同一个东西"正是 day064 起反复出现的那类隐患。
    """
    result: list[Vector] = []
    for row in scores:
        result.append(softmax(row))
    return tuple(result)


def scaled_dot_product_attention(
    queries: Matrix,
    keys: Matrix,
    values: Matrix,
    *,
    causal: bool = False,
    scale: float | None = None,
    temperature: float = 1.0,
) -> AttentionReport:
    """缩放点积注意力（可选因果掩码），返回一份 :class:`AttentionReport`.

    形状约定（原论文一致）：

    ```text
    queries   (n_q, d_k)
    keys      (n_k, d_k)
    values    (n_k, d_v)
    output    (n_q, d_v)
    weights   (n_q, n_k)
    ```

    ``temperature`` 作用在 softmax 上（``< 1`` 更集中、``> 1`` 更均匀），
    它**不是** ``scale``：``scale`` 改的是打分的尺度（数学上等价于改温度），
    而温度是"我想让注意力多集中"的一个显式旋钮。
    两者同时给会叠乘——报告里两个数都会记下来，因此"这次到底调了什么"可读。
    """
    checked_queries = validate_matrix(queries, name="queries")
    checked_keys = validate_matrix(keys, name="keys")
    checked_values = validate_matrix(values, name="values")
    if matrix_shape(checked_keys)[0] != matrix_shape(checked_values)[0]:
        raise NumericError(
            f"K 与 V 的行数必须相同（一一对应）：{matrix_shape(checked_keys)} 与 "
            f"{matrix_shape(checked_values)}。"
        )
    scores = attention_scores(checked_queries, checked_keys, scale=scale)
    rows, columns = matrix_shape(scores)
    mask: tuple[tuple[bool, ...], ...] | None = None
    if causal:
        if rows != columns:
            raise NumericError(
                f"因果掩码要求打分矩阵是方阵（自己看自己）：收到 {matrix_shape(scores)}。"
            )
        mask = causal_mask(rows)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ParameterError(
            f"温度必须为正的有限数，收到 {temperature!r}："
            "T = 0 的贪心语义请用 argmax 表达。"
        )
    if temperature != 1.0:
        # 温度是"改打分的尺度"：等价于把 softmax 的输入整体除以 T。
        # 直接缩打分而不是在 softmax 里加参数，是为了让"温度"与"scale"
        # 走同一条路径——两者都是对打分的缩放，报告里也都记着各自的数值。
        scores = tuple(tuple(value / temperature for value in row) for row in scores)
    weights = masked_softmax_rows(scores, mask) if mask is not None else _softmax_rows(scores)
    output = matmul(weights, checked_values)
    entropies = tuple(entropy(row) for row in weights)
    peaks = tuple(max(row) for row in weights)
    indices = tuple(argmax(row) for row in weights)
    head_dim = matrix_shape(checked_queries)[1]
    explicit_dot_scale = scale is not None and abs(float(scale) - 1.0) <= 1e-12
    variant = ATTENTION_DOT if explicit_dot_scale else ATTENTION_SCALED
    if causal:
        variant = ATTENTION_CAUSAL
    notes: list[str] = []
    if causal:
        notes.append(
            "因果掩码：上三角（j > i）的权重**恰好是 0.0**——位置 i 看不到它之后的任何位置"
            "（本包用显式掩码而不是把打分写成 -inf，理由见 masked_softmax_rows）"
        )
    notes.append(f"缩放系数 {scaling_factor(head_dim):.6f}（1/√d_k，d_k={head_dim}）")
    if temperature != 1.0:
        notes.append(f"温度 {temperature}：作用在打分上（< 1 更集中、> 1 更均匀）")
    scale_value = scaling_factor(head_dim) if scale is None else float(scale)
    return AttentionReport(
        variant=variant,
        queries=rows,
        keys=columns,
        head_dim=matrix_shape(checked_values)[1],
        scale=scale_value,
        weights=weights,
        output=output,
        entropies=entropies,
        peak_weights=peaks,
        peak_indices=indices,
        causal=causal,
        temperature=temperature,
        notes=tuple(notes),
    )


def split_heads(matrix: Matrix, heads: int) -> tuple[Matrix, ...]:
    """把最后一维**均分**成 ``heads`` 段（每一段是一个头）.

    要求最后一维能被 ``heads`` 整除：不能整除时的"取整"会让最后一段变短，
    而不同长度的头无法拼回原形状（错误会出现在拼接那一刻，离根因很远）。
    """
    checked = validate_matrix(matrix, name="matrix")
    if heads < 1:
        raise ParameterError(f"heads 必须 >= 1，收到 {heads}。")
    width = matrix_shape(checked)[1]
    if width % heads != 0:
        raise NumericError(
            f"最后一维 {width} 不能被头数 {heads} 整除："
            "每一段（每个头）必须等长，否则拼不回来——"
            "而'拼不回来'这个错误会出现在很远的地方。"
        )
    head_dim = width // heads
    return tuple(
        tuple(tuple(row[head * head_dim : (head + 1) * head_dim]) for row in checked)
        for head in range(heads)
    )


def merge_heads(parts: tuple[Matrix, ...]) -> Matrix:
    """把若干个头拼回 ``(n_q, Σ head_dim)``（形状不一致时报错）."""
    if not parts:
        raise ParameterError("至少要有一个头。")
    checked = tuple(validate_matrix(part, name="head") for part in parts)
    rows = matrix_shape(checked[0])[0]
    for index, part in enumerate(checked):
        if matrix_shape(part)[0] != rows:
            raise NumericError(
                f"第 {index} 个头的行数 {matrix_shape(part)[0]} 与第一个头 {rows} 不一致。"
            )
    return tuple(
        tuple(value for part in checked for value in part[row]) for row in range(rows)
    )


@dataclass(frozen=True)
class MultiHeadReport:
    """多头注意力的账：每个头一份报告 + 拼回来的输出.

    "多头"存在的意义不是"更准"，而是**并行的多套子空间**：
    一个头可以关注"上一个 token"，另一个头可以关注"句子的主语"——
    它们各自在自己的低维子空间里算注意力，最后拼起来。
    """

    heads: int
    reports: tuple[AttentionReport, ...]
    output: Matrix
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.heads != len(self.reports):
            raise NumericError(
                f"声明的头数 {self.heads} 与报告个数 {len(self.reports)} 不一致。"
            )
        if not self.reports:
            raise ParameterError("多头报告至少需要一个头。")
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def variant(self) -> str:
        """这个报告的变体名（``multi_head``）——它自己也要能进报告的分类列."""
        return ATTENTION_MULTI_HEAD

    @property
    def mean_entropy(self) -> float:
        """各头各行的平均熵（"整体上有多集中"）."""
        return math.fsum(report.mean_entropy for report in self.reports) / len(self.reports)

    def focus_by_head(self) -> Vector:
        """每个头的集中度（用于看"是不是所有头都在看同一处"）."""
        return tuple(report.focus_ratio() for report in self.reports)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（含变体说明）."""
        return {
            "variant": self.variant,
            "description": ATTENTION_VARIANT_DESCRIPTIONS[self.variant],
            "heads": self.heads,
            "mean_entropy": self.mean_entropy,
            "focus_by_head": list(self.focus_by_head()),
            "output": [list(row) for row in self.output],
            "reports": [report.to_dict() for report in self.reports],
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``多头 2×2 | 平均熵 0.6931 | 各头集中度 [0.0%, 0.0%]``."""
        focus = ", ".join(f"{value:.1%}" for value in self.focus_by_head())
        return (
            f"多头 {len(self.reports)}×{self.reports[0].queries} | "
            f"平均熵 {self.mean_entropy:.4f} | 各头集中度 [{focus}]"
        )


def multi_head_attention(
    queries: Matrix,
    keys: Matrix,
    values: Matrix,
    *,
    heads: int,
    causal: bool = False,
) -> MultiHeadReport:
    """多头注意力：把 ``d`` 维拆成 ``heads`` 段各自算，再拼回输出.

    三个必须说清的点：

    ```text
    ① 每个头的 d_k = d / heads，因此缩放系数是 1/√(d/heads) —— **不是** 1/√d
       （拆开之后每个头自己就是一个完整的小注意力）
    ② 头与头之间**没有交互**（各自算、各自 softmax），交互发生在拼接之后的线性层
       （本课不含那一层——它只是一个矩阵乘法，见 OP_MATMUL）
    ③ 每个头的注意力分布都不一样，因此"平均熵"是有信息量的：
       全相等说明这个多头结构退化成了一头
    ```
    """
    checked_queries = validate_matrix(queries, name="queries")
    checked_keys = validate_matrix(keys, name="keys")
    checked_values = validate_matrix(values, name="values")
    head_queries = split_heads(checked_queries, heads)
    head_keys = split_heads(checked_keys, heads)
    head_values = split_heads(checked_values, heads)
    reports = tuple(
        scaled_dot_product_attention(q, k, v, causal=causal)
        for q, k, v in zip(head_queries, head_keys, head_values)
    )
    output = merge_heads(tuple(report.output for report in reports))
    return MultiHeadReport(
        heads=heads,
        reports=reports,
        output=output,
        notes=(
            f"每个头的 d_k = {matrix_shape(checked_queries)[1] // heads}，"
            f"缩放系数 {scaling_factor(matrix_shape(checked_queries)[1] // heads):.6f}",
            "头与头之间没有交互：交互发生在拼接之后的线性层",
        ),
    )


def positional_encoding(positions: int, dimension: int, *, base: float = POSITIONAL_BASE) -> Matrix:
    """正弦/余弦位置编码 ``PE(pos, 2i) = sin(pos / base^{2i/d})``、``PE(pos, 2i+1) = cos(...)``.

    为什么需要它：注意力本身是**置换不变**的（把 token 的顺序打乱，
    ``QKᵀ`` 只是行列跟着换，权重集合一模一样）。因此"顺序"必须由输入带进来，
    否则"猫追老鼠"与"老鼠追猫"在模型眼里是同一句话。

    为什么用不同频率的正弦波：每一维是一条不同波长的波，
    于是"位置差 k"在编码空间里对应一个**固定的旋转**——
    这让模型可以从"两个位置的编码之差"里读出相对距离，
    而不是死记每一个绝对位置（后者在训练长度之外完全失效）。
    """
    if positions < 1:
        raise ParameterError(f"positions 必须 >= 1，收到 {positions}。")
    if dimension < 2:
        raise ParameterError(f"dimension 必须 >= 2，收到 {dimension}。")
    if base <= 1:
        raise ParameterError(f"频率基数必须 > 1，收到 {base}。")
    rows: list[Vector] = []
    for position in range(positions):
        row: list[float] = []
        for index in range(dimension):
            frequency = position / (base ** (index / dimension))
            row.append(math.sin(frequency) if index % 2 == 0 else math.cos(frequency))
        rows.append(tuple(row))
    return tuple(rows)


__all__ = [
    "POSITIONAL_BASE",
    "DotProductStudy",
    "MultiHeadReport",
    "attention_scores",
    "causal_mask",
    "masked_softmax_rows",
    "merge_heads",
    "multi_head_attention",
    "positional_encoding",
    "sampled_dot_product_variance",
    "scaled_dot_product_attention",
    "scaling_factor",
    "split_heads",
]
