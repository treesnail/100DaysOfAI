"""``features``：从 hidden states 到一条定长向量（day086 / M7-D10）.

特征抽取的最后一步不是模型，而是**一次池化**。三法各有各的毛病，
而其中一法的毛病**只在有填充的时候才出现**——这正是本模块要把它钉住的原因：

```text
last_token   取最后一个**真实**位置（因果模型的经典取法：它见过前面所有 token）
mean         对真实位置取平均 —— 分母是**真实长度**，不是总宽度
max          对真实位置逐维取最大（对离群值最敏感，因此最容易被填充污染）
```

## 一个只差一个分母的 bug

```python
mean_correct = sum(真实位置) / 真实长度     # ← 分母是真实长度
mean_buggy   = sum(全部位置) / 总宽度       # ← 分母是总宽度：填充越多，向量越"短"
```

两者在**一批里只有一条、且它正好是最长的那条**时**完全相等**——
因此"我本地测过没问题"与"它在线上是错的"可以同时成立。
本包把两法都实现出来（``ignore_mask=True`` 是故意错的那一版），
让"违反 mask 的代价"变成一个能被印出来的数，而不是一句规矩。

## 一个"看起来无害"的约定：填充只允许在右边

``last_token`` 取的是行末。若填充在左边，它会取到一个填充行——
结果是一组**长度为零的向量**，而它们在相似度上表现为"这条和谁都不像"。
本包因此只接受 ``padding_side="right"``（与真实库的默认值一致），
并在 :mod:`forward` 里把"掩码里有空洞"当场拒绝。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from smart_research_agent.hf_integration.errors import NumericError, ParameterError, ShapeError
from smart_research_agent.hf_integration.forward import HiddenBatch
from smart_research_agent.hf_integration.types import (
    POOLING_DESCRIPTIONS,
    POOLING_LAST_TOKEN,
    POOLING_MAX,
    POOLING_MEAN,
    POOLING_STRATEGIES,
    POOLING_TOLERANCE,
    PooledBatch,
)
from smart_research_agent.math_foundations.types import Matrix, Vector

#: 三法各自"取哪一行 / 取哪几行"的一句话（报告里读它）。
POOLING_RULES: dict[str, str] = {
    POOLING_LAST_TOKEN: "取行内**最后一个真实位置**：rows[i][-1]",
    POOLING_MEAN: "对行内所有位置取平均：fsum(rows[i]) / 真实长度",
    POOLING_MAX: "对行内逐维取最大：max over 位置（并列时取**最先**出现的那个）",
}


def _check_strategy(strategy: str) -> None:
    """未知策略当场拒绝（回退到 mean 的后果是——一份 last_token 的读数被印成均值）."""
    if strategy not in POOLING_STRATEGIES:
        raise ParameterError(
            f"未知的池化策略 {strategy!r}：本包只认 {list(POOLING_STRATEGIES)}。"
            "回退到均值的后果是——一次 last_token 的读数会被印成 mean，"
            "而两者在短句上**看起来很像**。"
        )


def pool_rows(rows: tuple[Matrix, ...] | list[Matrix], strategy: str) -> Matrix:
    """对**已经按真实长度切好**的若干行做池化（**无填充，因此没有分母问题**）.

    这是本包推荐的那条路：只要上游交出来的每一行都是"这条样本真实的那几行"，
    池化就与"这批里别人有多长"完全无关——这正是
    :data:`~smart_research_agent.hf_integration.types.PROPERTY_POOLING_RESPECTS_MASK`
    要断言的那件事。
    """
    _check_strategy(strategy)
    checked = tuple(tuple(row) for row in rows)
    if not checked:
        raise ShapeError("池化收到 0 行：没有样本就没有向量。")
    width = len(checked[0][0]) if checked[0] else 0
    if width == 0:
        raise ShapeError("第一行是空向量：零宽的向量上没有池化。")
    for index, row in enumerate(checked):
        if not row:
            raise NumericError(
                f"第 {index} 行没有位置可以池化：空行的 mean 是 0/0，"
                "而一个 nan 会顺着相似度排序污染整张表（它看起来像'这条和谁都不像'）。"
            )
        for value in (value for r in row for value in r):
            if not math.isfinite(value):
                raise NumericError(
                    "hidden state 里出现了非有限数：池化会把它平均进整条向量，"
                    "而下游只能看到一个'和谁都不像'的向量。"
                )
        if len(row[0]) != width:
            raise ShapeError(f"第 {index} 行的宽度与本组不一致（zip 会静默截断）。")
    if strategy == POOLING_LAST_TOKEN:
        return tuple(row[-1] for row in checked)
    if strategy == POOLING_MEAN:
        return tuple(
            tuple(math.fsum(row[position][column] for position in range(len(row))) / len(row)
                  for column in range(width))
            for row in checked
        )
    return tuple(
        tuple(max(row[position][column] for position in range(len(row))) for column in range(width))
        for row in checked
    )


def pool_masked(
    states: Matrix,
    mask: Vector,
    strategy: str,
    *,
    width: int,
    ignore_mask: bool = False,
) -> tuple[Matrix, tuple[str, ...]]:
    """对**带填充的张量**做池化：正确版与故意错版共用一条路径（``ignore_mask=True`` 是错版）.

    ``states`` 是 ``(batch × width, hidden)`` 的**摊平**形式，``mask`` 是一个长度
    ``batch × width`` 的 0/1 向量（**每个位置一个标志**，不是一张表）：
    切成 ``batch`` 行各 ``width`` 宽之后，真实位置就是那一段里标志为 1 的那些。

    返回值是``(向量组, 证据行)``——证据行里写明这一批**填了几格**，
    因为"填充的代价"必须带着数字，否则"我加了 mask"与"我没加"读起来一样。
    """
    _check_strategy(strategy)
    if width < 1:
        raise ParameterError(f"width 必须 >= 1，收到 {width}。")
    if len(states) % width != 0:
        raise ShapeError(
            f"摊平后的行数 {len(states)} 不能被 width={width} 整除："
            "切不对批次时每一行都会错位，而形状看起来是合法的（这正是 day085 "
            "第 3.2 节那条'形状合法、语义已变'的同族）。"
        )
    if len(mask) != len(states):
        raise ShapeError(
            f"掩码有 {len(mask)} 个位置、状态有 {len(states)} 行：两者必须逐行对齐。"
        )
    batch = len(states) // width
    vectors: list[Vector] = []
    evidence: list[str] = []
    for index in range(batch):
        start = index * width
        rows = states[start : start + width]
        flags = mask[start : start + width]
        real = [position for position, flag in enumerate(flags) if flag]
        if not real:
            raise NumericError(
                f"第 {index} 条样本的掩码全是 0：它没有真实位置，"
                "均值池化会得到 0/0（一个 nan），而下游只能看到'它和谁都不像'。"
            )
        if real != list(range(len(real))):  # pragma: no cover - 只有左填充会触发
            raise ShapeError(
                f"第 {index} 条的真实位置不是从 0 开始连续的一段："
                "本包只接受**右侧填充**（与真实库的 padding_side='right' 一致）——"
                "左填充时 last_token 会取到一个填充行。"
            )
        padded = width - len(real)
        if ignore_mask:
            rows_used = tuple(rows)
        else:
            rows_used = tuple(rows[position] for position in real)
        evidence.append(f"第 {index} 条：真实 {len(real)} 格、填充 {padded} 格")
        if len(rows_used) == 1:
            vectors.append(tuple(rows_used[0]))
            continue
        if strategy == POOLING_LAST_TOKEN:
            vectors.append(tuple(rows_used[-1]))
        elif strategy == POOLING_MEAN:
            width_hidden = len(rows_used[0])
            vectors.append(
                tuple(
                    math.fsum(rows_used[position][column] for position in range(len(rows_used)))
                    / len(rows_used)
                    for column in range(width_hidden)
                )
            )
        else:
            width_hidden = len(rows_used[0])
            vectors.append(
                tuple(
                    max(rows_used[position][column] for position in range(len(rows_used)))
                    for column in range(width_hidden)
                )
            )
    return tuple(vectors), tuple(evidence)


def pool_forward(
    batch: HiddenBatch,
    strategy: str,
    *,
    normalize_vectors: bool = False,
) -> PooledBatch:
    """把一次前向的结果池成一个 :class:`PooledBatch`（**推荐入口**）."""
    vectors = pool_rows(batch.rows, strategy)
    if normalize_vectors:
        vectors = l2_normalize(vectors)
    return PooledBatch(vectors=vectors, strategy=strategy, normalized=normalize_vectors)


def l2_normalize(vectors: Matrix) -> Matrix:
    """逐条做 L2 归一化（**零向量当场拒绝**，因为归一化会把它变成 0/0）."""
    out: list[Vector] = []
    for index, vector in enumerate(vectors):
        norm = math.sqrt(math.fsum(value * value for value in vector))
        if norm == 0.0:
            raise NumericError(
                f"第 {index} 条向量是全零：L2 归一化会得到 0/0。"
                "全零向量通常来自'取到了一个填充行'——"
                "例如 last_token 遇到左填充（那是本课唯一允许的填充方向之外的情况）。"
            )
        out.append(tuple(value / norm for value in vector))
    return tuple(out)


def cosine_similarity(left: Vector, right: Vector) -> float:
    """两条向量的余弦相似度（**先各归一再点积**，因此与长度无关）."""
    if len(left) != len(right):
        raise ShapeError(f"两条向量长度不同：{len(left)} 与 {len(right)}。")
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise NumericError("余弦相似度的两个操作数里出现了全零向量：分母是 0。")
    dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
    return dot / (left_norm * right_norm)


def max_gap(left: Matrix, right: Matrix) -> float:
    """两份同形矩阵之间的最大绝对差（两条路径之间的读数）."""
    if len(left) != len(right):
        raise ShapeError(f"两份矩阵行数不同：{len(left)} 与 {len(right)}。")
    worst = 0.0
    for row_left, row_right in zip(left, right, strict=True):
        if len(row_left) != len(row_right):
            raise ShapeError("两份矩阵的列数不同（zip 会静默截断）。")
        for a, b in zip(row_left, row_right, strict=True):
            worst = max(worst, abs(a - b))
    return worst


@dataclass(frozen=True)
class PoolingComparison:
    """一次"正确版 vs 违反 mask 版"的对照（study 与 verify 共用这一个记录）."""

    strategy: str
    correct: Matrix
    buggy: Matrix
    padding_waste: float
    evidence: tuple[str, ...] = ()

    @property
    def gap(self) -> float:
        """两版之间的最大绝对差（正确性问题的**读数**）."""
        return max_gap(self.correct, self.buggy)

    @property
    def identical(self) -> bool:
        """两版是否逐位相同（只有"这一批没有任何填充"时才应当相同）."""
        return self.correct == self.buggy

    def line(self) -> str:
        """一行可读的读数."""
        return (
            f"{self.strategy}: 填充占比 {self.padding_waste:.1%} | "
            f"违反 mask 的代价 {self.gap:.6e} | 两版相同：{self.identical}"
        )


def compare_pooling(
    states: Matrix,
    mask: Vector,
    strategy: str,
    *,
    width: int,
) -> PoolingComparison:
    """同一批数据上跑正确版与错版，把"代价"量出来.

    它存在的理由是 day082 的探针那条纪律：**一条规矩如果没有一个读数，
    它就无法被反驳**。"必须按 mask 池化"这句话的读数就是这两个版本的差。
    """
    correct, evidence = pool_masked(states, mask, strategy, width=width, ignore_mask=False)
    buggy, _ = pool_masked(states, mask, strategy, width=width, ignore_mask=True)
    real = sum(1 for flag in mask if flag)
    total = len(mask)
    waste = 0.0 if total == 0 else 1.0 - real / total
    return PoolingComparison(
        strategy=strategy,
        correct=correct,
        buggy=buggy,
        padding_waste=waste,
        evidence=evidence,
    )


def pooling_line(strategy: str) -> str:
    """一句话说明这个策略的规矩（报告里读它）."""
    _check_strategy(strategy)
    return f"{strategy}：{POOLING_RULES[strategy]} | {POOLING_DESCRIPTIONS[strategy]}"


def tolerance_of(strategy: str) -> float:
    """这个策略与"逐条跑"之间允许的差（三法共用同一个容差，因为它来自求和顺序）."""
    _check_strategy(strategy)
    return POOLING_TOLERANCE


__all__ = [
    "POOLING_RULES",
    "PoolingComparison",
    "compare_pooling",
    "cosine_similarity",
    "l2_normalize",
    "max_gap",
    "pool_forward",
    "pool_masked",
    "pool_rows",
    "pooling_line",
    "tolerance_of",
]
