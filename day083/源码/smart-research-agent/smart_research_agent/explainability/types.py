"""``explainability`` 的类型与口径表：**把注意力变成可以打印、可以被断言的东西**（day083）.

昨天（day082）留下了三张权重表；今天把它们读出来、画出来、并给出可被断言的读数。

## 一、四件东西与它们各自的判据

```text
AttentionRecord   一层一头的权重 + 它用的掩码（**读出**的最小单位）
HeadProfile       一张权重表的性质：熵 / 天花板 / 归一化熵 / 峰值 / 支撑集 / 对角质量
Attribution       每个**输入位置**被分到多少注意力（列和——"谁在被看"）
Heatmap（文本）    权重 → 等级字符 → 一行一行打印（render.py 负责渲染与解析）
```

## 二、本课的第一条纪律：**熵要跟天花板比**

```text
一行的熵 ≤ ln(这一行能看到的位置数)        ← 这不是经验规律，是 Jensen 不等式
```

因此 `HeadProfile` 里同时留下三个数：

```text
entropy              这张表的平均每行熵
ceiling              平均每行的天花板 = mean(ln(allowed_count))
normalized_entropy   entropy / ceiling      ← **跨变体、跨层可比的读数**
```

没有第三个的话，两张表会被"掩码不同"这件事骗过去：day082 已经量过，
因果掩码把第 0 行的天花板压到 `ln 1 = 0`（全开是 `ln n`）。
于是"GPT 的注意力比 BERT 尖"这句话里，**有一部分是掩码造成的**——
而这个包的全部意义就是把它拆开。

## 三、第二条纪律：**等级是有损的，因此要能对回去**

```text
10 级（" .:-=+*#%@"）把 [0, 1] 切成 10 段
渲染 → 解析 ⇒ 应当**逐级**相同（不是"看起来差不多"）
```

`LEVELS` 与 `level_of` / `level_char` 的两处口径必须同源，
而"渲染 → 解析 → 逐级相同"是本课第 4 条性质（一条**可失败**的断言）。
它挡住的是那种"把 1 与 0 画得一样、把 0.09 与 0.11 画成不同等级"的实现错误——
那种错误只会让人看着图得出错的结论。

## 四、记录口径

```text
AttentionRecord   label（"层 1 · 头 0"）/ stream（self|cross）/ weights / mask /
                  tokens / source_tokens / notes
HeadProfile       label / entropy / ceiling / normalized_entropy / peak_weight /
                  peak_index / support / frobenius / diagonal_mass
Attribution       label / mass（列和，归一化）/ top_positions
```
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from smart_research_agent.explainability.errors import (
    AssemblyError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.math_foundations.types import (
    Matrix,
    Vector,
    matrix_shape,
    validate_matrix,
)
from smart_research_agent.transformer_core.types import check_weights_are_a_distribution

#: 等级字符（10 级，从"几乎没有"到"几乎全在这里"）.
LEVELS = " .:-=+*#%@"

#: 级数（``len(LEVELS)``）——渲染与解析共用它，因此只有一处口径.
LEVEL_COUNT = len(LEVELS)

#: 默认头数（`d = 6` 时 2 头，每头 3 维：必须能整除）.
DEFAULT_HEADS = 2

#: "这个权重算不算 0"的阈值（支撑集与稀疏度用它）.
DEFAULT_THRESHOLD = 1e-6

#: 注意力滚动的默认混合系数（``Â = α·A + (1−α)·I``）.
DEFAULT_ALPHA = 0.5

#: 两条流的名字.
STREAM_SELF = "self"
STREAM_CROSS = "cross"

STREAM_KINDS: tuple[str, ...] = (STREAM_SELF, STREAM_CROSS)

STREAM_DESCRIPTIONS: dict[str, str] = {
    STREAM_SELF: "自注意力：权重是**方阵**（第 i 个位置怎么看第 j 个位置）",
    STREAM_CROSS: "交叉注意力：权重是**长方形**（第 i 个解码位置怎么看第 j 个源位置）",
}

# ---------------------------------------------------------------------- 六条性质

PROPERTY_ROWS_ARE_DISTRIBUTIONS = "rows_are_distributions"
PROPERTY_MASKED_ENTRIES_ARE_EXACT_ZERO = "masked_entries_are_exact_zero"
PROPERTY_ENTROPY_WITHIN_CEILING = "entropy_within_ceiling"
PROPERTY_HEATMAP_ROUND_TRIP = "heatmap_round_trip"
PROPERTY_ROLLOUT_IS_STOCHASTIC = "rollout_is_stochastic"
PROPERTY_DETERMINISTIC = "deterministic"

EXPLAIN_PROPERTIES: tuple[str, ...] = (
    PROPERTY_ROWS_ARE_DISTRIBUTIONS,
    PROPERTY_MASKED_ENTRIES_ARE_EXACT_ZERO,
    PROPERTY_ENTROPY_WITHIN_CEILING,
    PROPERTY_HEATMAP_ROUND_TRIP,
    PROPERTY_ROLLOUT_IS_STOCHASTIC,
    PROPERTY_DETERMINISTIC,
)

PROPERTY_DESCRIPTIONS: dict[str, str] = {
    PROPERTY_ROWS_ARE_DISTRIBUTIONS: "每一行是一个条件分布（非负、和为 1、有限）",
    PROPERTY_MASKED_ENTRIES_ARE_EXACT_ZERO: "被掩码挡掉的位置**逐位**是 0.0（day082 的口径）",
    PROPERTY_ENTROPY_WITHIN_CEILING: "每一行的熵 ≤ ln(这一行能看到的位置数)——**Jensen 不等式**",
    PROPERTY_HEATMAP_ROUND_TRIP: "渲染成等级字符再解析回来，逐级相同（10 级是有损的，因此要能对回去）",
    PROPERTY_ROLLOUT_IS_STOCHASTIC: "滚动之后的表仍然是行随机的，且**因果零模式保持不变**",
    PROPERTY_DETERMINISTIC: "同一批记录跑两次得到逐位相同的结果（没有随机数）",
}

EXPLAIN_NOTES: tuple[str, ...] = (
    "本包不改任何算术：它读的是 day075~082 已经算好并验过的权重。",
    "熵必须跟天花板比：因果掩码把第 0 行的天花板压到 ln 1 = 0（day082 已量过）。",
    "等级渲染是有损的，因此 'render → parse 逐级相同' 是一条可失败的断言。",
)


def _checked_positive(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParameterError(f"{name} 必须是整数，收到 {value!r}。")
    if value < 1:
        raise ParameterError(f"{name} 必须 >= 1，收到 {value}。")
    return value


def row_entropy(row: Sequence[float]) -> float:
    """一行的 Shannon 熵 ``−Σ p·ln p``（**只对正项求和**：``0·ln 0`` 取 0）.

    用 ``math.fsum`` 而不是内建 ``sum``：跨表比较的两个数必须同口径
    （day073 起的一条纪律）。
    """
    values = tuple(float(value) for value in row)
    if not values:
        raise ParameterError("熵的输入不能是空行。")
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise NumericError(f"第 {index} 个权重不是有限数（{value!r}）。")
        if value < 0.0:
            raise NumericError(f"第 {index} 个权重是负数（{value}）：概率不能为负。")
    return -math.fsum(value * math.log(value) for value in values if value > 0.0)


def entropy_ceiling(allowed_count: int) -> float:
    """一行的熵天花板 ``ln(允许的位置数)``（``allowed_count >= 1``）."""
    return math.log(_checked_positive(allowed_count, name="allowed_count"))


def row_entropies(weights: Matrix) -> Vector:
    """逐行熵."""
    checked = validate_matrix(weights, name="weights")
    return tuple(row_entropy(row) for row in checked)


def ceiling_of_mask(mask: tuple[tuple[bool, ...], ...]) -> Vector:
    """逐行的熵天花板（由掩码的每一行允许的位置数给出）."""
    counts = allowed_counts(mask)
    return tuple(entropy_ceiling(count) for count in counts)


def allowed_counts(mask: tuple[tuple[bool, ...], ...]) -> tuple[int, ...]:
    """每一行允许的位置数（**只有一处实现**：``render`` / ``verify`` 都用它）."""
    if not mask:
        raise ParameterError("掩码不能为空。")
    width = len(mask[0])
    if any(len(row) != width for row in mask):
        raise ShapeError("掩码每行的宽度必须一致。")
    counts = tuple(sum(1 for value in row if value) for row in mask)
    for index, count in enumerate(counts):
        if count < 1:
            raise NumericError(
                f"掩码第 {index} 行没有任何允许的位置：那一行的分布没有定义"
                "（day075 会拒绝它）——因此它的熵天花板也无从谈起。"
            )
    return counts


def mean_of(values: Sequence[float]) -> float:
    """算术平均（空序列当场拒绝：一个"没有读数"的平均值不是 0）."""
    resolved = tuple(float(value) for value in values)
    if not resolved:
        raise ParameterError("平均值的输入不能为空。")
    return math.fsum(resolved) / len(resolved)


def full_mask(size: int) -> tuple[tuple[bool, ...], ...]:
    """全开掩码（**转发 day075 的口径**：每一行都能看到所有位置）."""
    from smart_research_agent.transformer_core.layers import full_mask as _core_full

    return _core_full(_checked_positive(size, name="size"))


def causal_mask(size: int) -> tuple[tuple[bool, ...], ...]:
    """因果掩码（**转发 day073 的口径**：``mask[i][j]`` 为真表示"允许看"且 ``j <= i``）."""
    from smart_research_agent.math_foundations.attention import causal_mask as _core_causal

    return tuple(row for row in _core_causal(_checked_positive(size, name="size")))


@dataclass(frozen=True)
class AttentionRecord:
    """一层一头的注意力账（**读出的最小单位**）.

    ```text
    label           人读的名字："层 1 · 头 0"
    stream          self（方阵）或 cross（**通常是**长方形，而 n_tgt == n_src 时是方阵）
    weights         (rows, columns)：每一行是一个条件分布
    mask            这一次前向用的掩码（**逐格与 weights 对齐**）
    tokens          行标签（可选）；source_tokens 给交叉那一侧用（可选）
    ```

    校验是"两层"的：形状（自注意力必须是方阵、掩码与权重对齐）与数值（每一行是分布）。
    数值那一层**转发** ``transformer_core.types.check_weights_are_a_distribution``——
    口径只有一处，因此"注意力是不是一个分布"这个问题的答案在全仓库是同一个。

    一处刻意的取舍：**两条流不是靠形状区分的**。``n_tgt == n_src`` 时自注意力与
    交叉注意力的形状完全一样，因此区分它们的只有 ``stream`` 这个字段；
    而 ``STREAM_SELF`` 是默认值 ⇒ **交叉那一侧必须显式说明**。
    本包不在这里拒绝"方阵的交叉记录"：那会把"等长翻译"这类完全正常的输入挡在门外，
    而真正的保护来自"字段是必填的"这件事。
    """

    label: str
    weights: Matrix
    mask: tuple[tuple[bool, ...], ...]
    stream: str = STREAM_SELF
    tokens: tuple[str, ...] = ()
    source_tokens: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ParameterError(f"label 必须是非空字符串，收到 {self.label!r}。")
        if self.stream not in STREAM_KINDS:
            raise ParameterError(
                f"未知的流名 {self.stream!r}：可选 {', '.join(STREAM_KINDS)}。"
            )
        checked = validate_matrix(self.weights, name="weights")
        object.__setattr__(self, "weights", checked)
        rows, columns = matrix_shape(checked)
        if len(self.mask) != rows or (self.mask and len(self.mask[0]) != columns):
            raise ShapeError(
                f"掩码形状 {len(self.mask)}×{len(self.mask[0]) if self.mask else 0} 与权重 "
                f"{rows}×{columns} 不一致：两张表必须逐格对齐，否则'哪一格被挡住'无从谈起。"
            )
        if self.stream == STREAM_SELF and rows != columns:
            raise AssemblyError(
                f"自注意力权重必须是方阵（自己看自己），收到 {rows}×{columns}——"
                "长方形的那一张是交叉注意力（Q 来自一路、K/V 来自另一路）。"
            )
        # 交叉那一侧**可以**是方阵（n_tgt == n_src 时形状完全一样），
        # 因此本包不靠形状区分两条流，而靠 `stream` 这个字段——
        # 而 `STREAM_SELF` 是默认值，于是交叉那一侧**必须显式说明**自己不是自注意力。
        allowed_counts(self.mask)
        check_weights_are_a_distribution(checked, tolerance=1e-9)
        object.__setattr__(self, "tokens", tuple(str(item) for item in self.tokens))
        object.__setattr__(self, "source_tokens", tuple(str(item) for item in self.source_tokens))
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))
        if self.tokens and len(self.tokens) != rows:
            raise ShapeError(
                f"行标签有 {len(self.tokens)} 个，权重有 {rows} 行。"
            )
        if self.source_tokens and len(self.source_tokens) != columns:
            raise ShapeError(
                f"列标签有 {len(self.source_tokens)} 个，权重有 {columns} 列。"
            )

    @property
    def rows(self) -> int:
        """行数（"谁在看"的那一侧）."""
        return len(self.weights)

    @property
    def columns(self) -> int:
        """列数（"被看"的那一侧）."""
        return len(self.weights[0])

    @property
    def allowed_counts(self) -> tuple[int, ...]:
        """每一行允许的位置数（熵天花板来自它）."""
        return allowed_counts(self.mask)

    @property
    def entropies(self) -> Vector:
        """逐行熵."""
        return row_entropies(self.weights)

    @property
    def ceiling(self) -> Vector:
        """逐行天花板."""
        return ceiling_of_mask(self.mask)

    @property
    def mean_entropy(self) -> float:
        """平均每行熵."""
        return mean_of(self.entropies)

    @property
    def mean_ceiling(self) -> float:
        """平均每行天花板."""
        return mean_of(self.ceiling)

    @property
    def normalized_entropy(self) -> float:
        """归一化熵 ``entropy / ceiling``（**跨层、跨变体可比的读数**）."""
        ceiling = self.mean_ceiling
        if ceiling == 0.0:
            return 0.0
        return self.mean_entropy / ceiling

    def weight_at(self, row: int, column: int) -> float:
        """取一个格子（越界当场拒绝：这张表是证据，索引错了就不该继续）."""
        if isinstance(row, bool) or not isinstance(row, int):
            raise ParameterError(f"行号必须是整数，收到 {row!r}。")
        if isinstance(column, bool) or not isinstance(column, int):
            raise ParameterError(f"列号必须是整数，收到 {column!r}。")
        if not 0 <= row < self.rows or not 0 <= column < self.columns:
            raise ParameterError(
                f"({row}, {column}) 越界：可选 0..{self.rows - 1} × 0..{self.columns - 1}。"
            )
        return self.weights[row][column]

    def masked_zeroes(self) -> int:
        """被掩码挡掉的格子数（**含权重恰好为 0 的那些**）."""
        return sum(
            1
            for row in range(self.rows)
            for column in range(self.columns)
            if not self.mask[row][column]
        )

    def summary_line(self) -> str:
        """一行说明（热力图的标题行）."""
        return (
            f"{self.label} | {self.rows}×{self.columns} | 流 {self.stream} | "
            f"熵 {self.mean_entropy:.4f}（天花板 {self.mean_ceiling:.4f}，"
            f"归一化 {self.normalized_entropy:.4f}）"
        )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（**矩阵本身也带上**：它是证据）."""
        return {
            "label": self.label,
            "stream": self.stream,
            "rows": self.rows,
            "columns": self.columns,
            "weights": [list(row) for row in self.weights],
            "mean_entropy": self.mean_entropy,
            "mean_ceiling": self.mean_ceiling,
            "normalized_entropy": self.normalized_entropy,
            "masked_zeroes": self.masked_zeroes(),
        }


@dataclass(frozen=True)
class HeadProfile:
    """一张权重表的性质（**进表进报告的那一份**）.

    ```text
    entropy / ceiling / normalized_entropy   三个数一起给（第 2 节）
    peak_weight / peak_index                 这一行/这一表最尖的格子在哪
    support                                  有多少个格子的权重 > 阈值（"它用了几个位置"）
    frobenius                                行随机矩阵的 Frobenius 范数 ∈ [1, √n]（越尖越大）
    diagonal_mass                            对角线上的质量之和（自看自的比例）
    """

    label: str
    entropy: float
    ceiling: float
    peak_weight: float
    peak_index: int
    support: int
    frobenius: float
    diagonal_mass: float
    rows: int
    columns: int
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        for name, value in (
            ("entropy", self.entropy),
            ("ceiling", self.ceiling),
            ("peak_weight", self.peak_weight),
            ("frobenius", self.frobenius),
            ("diagonal_mass", self.diagonal_mass),
        ):
            if not math.isfinite(value):
                raise NumericError(f"{name} 必须是有限数，收到 {value!r}。")
        if self.entropy < -1e-12:
            raise NumericError(f"熵不能是负数，收到 {self.entropy}。")
        if self.entropy > self.ceiling + 1e-12:
            raise NumericError(
                f"熵 {self.entropy} 超过了天花板 {self.ceiling}："
                "一行的熵不可能超过 ln(它能看到的位置数)——越界说明权重与掩码"
                "来自**两次不同的前向**。"
            )
        if self.support < 1:
            hint = f"（本次用的阈值：{self.notes[0]}）" if self.notes else ""
            raise NumericError(
                f"支撑集是 {self.support}：一张每一格都不超过阈值的表"
                f"{hint}不可能是 softmax 的输出（它的每一行和为 1）——"
                "如果这个阈值是调用方传的，请把它调小：本包不在构造期替你改阈值。"
            )
        _checked_positive(self.rows, name="rows")
        _checked_positive(self.columns, name="columns")
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def normalized_entropy(self) -> float:
        """归一化熵（天花板为 0 时取 0：那一行看不到任何别的位置）."""
        if self.ceiling == 0.0:
            return 0.0
        return self.entropy / self.ceiling

    @property
    def sparsity(self) -> float:
        """稀疏度 = 1 − 支撑集 / 总格子数（越大越"只用少数位置"）."""
        return 1.0 - self.support / (self.rows * self.columns)

    def summary_line(self) -> str:
        """一行说明（实验表里用的就是这一行）."""
        return (
            f"{self.label:<12} | 熵 {self.entropy:.4f} / 天花板 {self.ceiling:.4f} | "
            f"归一化 {self.normalized_entropy:.4f} | 峰值 {self.peak_weight:.4f} @ "
            f"{self.peak_index} | 支撑 {self.support}/{self.rows * self.columns} | "
            f"Frobenius {self.frobenius:.4f}"
        )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "label": self.label,
            "entropy": self.entropy,
            "ceiling": self.ceiling,
            "normalized_entropy": self.normalized_entropy,
            "peak_weight": self.peak_weight,
            "peak_index": self.peak_index,
            "support": self.support,
            "sparsity": self.sparsity,
            "frobenius": self.frobenius,
            "diagonal_mass": self.diagonal_mass,
            "rows": self.rows,
            "columns": self.columns,
        }


@dataclass(frozen=True)
class Attribution:
    """**谁在被看**：把权重按列加起来（每个输入位置分到的总注意力）.

    ```text
    mass            列和（**归一化到总和为 1**：一张 (n, n) 的表有 n 列，每列一个数）
    top_positions   质量最大的几个位置（按质量降序、并列时按下标升序）
    ```

    它与"每一行看哪里"（那些权重本身）是两件事：列和回答的是
    **"整张表里哪些位置重要"**，而它会把"某一个位置被很多人看"与
    "某一个位置被一个人看了全部"混在一起。因此 `Attribution` 只作为
    读数之一，而它的口径被写在 `summary_line` 里。
    """

    label: str
    mass: Vector
    top_positions: tuple[int, ...]
    columns: int

    def __post_init__(self) -> None:
        _checked_positive(self.columns, name="columns")
        if len(self.mass) != self.columns:
            raise ShapeError(
                f"列质量有 {len(self.mass)} 个数，权重有 {self.columns} 列。"
            )
        total = math.fsum(self.mass)
        if abs(total - 1.0) > 1e-9:
            raise NumericError(
                f"列质量的和是 {total!r}（应当归一化到 1）："
                "没有归一化的话'哪个位置重要'这件事会随表的大小变化。"
            )
        for index in self.top_positions:
            if not 0 <= index < self.columns:
                raise ParameterError(f"top_positions 里的 {index} 越界。")
        object.__setattr__(self, "mass", tuple(float(value) for value in self.mass))

    def mass_at(self, position: int) -> float:
        """某个位置分到的质量（越界当场拒绝）."""
        if isinstance(position, bool) or not isinstance(position, int):
            raise ParameterError(f"位置必须是整数，收到 {position!r}。")
        if not 0 <= position < self.columns:
            raise ParameterError(f"位置 {position} 越界：可选 0..{self.columns - 1}。")
        return self.mass[position]

    def summary_line(self) -> str:
        """一行说明."""
        top = ", ".join(f"{index}→{self.mass[index]:.4f}" for index in self.top_positions)
        return f"{self.label} | 列质量 top：{top} | 共 {self.columns} 列"

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "label": self.label,
            "mass": list(self.mass),
            "top_positions": list(self.top_positions),
            "columns": self.columns,
        }


__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_HEADS",
    "DEFAULT_THRESHOLD",
    "EXPLAIN_NOTES",
    "EXPLAIN_PROPERTIES",
    "LEVELS",
    "LEVEL_COUNT",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_DETERMINISTIC",
    "PROPERTY_ENTROPY_WITHIN_CEILING",
    "PROPERTY_HEATMAP_ROUND_TRIP",
    "PROPERTY_MASKED_ENTRIES_ARE_EXACT_ZERO",
    "PROPERTY_ROLLOUT_IS_STOCHASTIC",
    "PROPERTY_ROWS_ARE_DISTRIBUTIONS",
    "STREAM_CROSS",
    "STREAM_DESCRIPTIONS",
    "STREAM_KINDS",
    "STREAM_SELF",
    "AttentionRecord",
    "Attribution",
    "HeadProfile",
    "allowed_counts",
    "causal_mask",
    "ceiling_of_mask",
    "entropy_ceiling",
    "full_mask",
    "mean_of",
    "row_entropies",
    "row_entropy",
]
