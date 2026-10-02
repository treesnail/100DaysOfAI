"""``rollout.py``：把逐层权重**乘起来**，看信息走了几跳（day083）.

## 一、为什么"最后一层的热力图"会骗人

单看最后一层的权重时，"第 3 层没在看第 0 个位置"这件事看起来像"它不重要"。
而残差把每一层的输出**加上**了输入 ⇒ 第 3 层看第 1 层、第 1 层看第 0 层，
于是"第 0 个位置的信息"其实经过**两跳**就到了输出。滚动的做法就是把这件事写下来：

```text
Â_l = α·A_l + (1−α)·I_l            每个因子都会"顺手把输入带过去"
R   = Â_L · Â_{L−1} · … · Â_1       矩阵乘 ⇒ 路径被自动求和
```

那项 ``(1−α)·I`` 不是技巧，而是**残差**（day079 的 ``y = x + F(x)``）在注意力层面的样子：
它让"这一层什么都没做"也有一条直接的通路。``α = 0`` 时 ``R = I`` ——
所有质量都在对角线，信息一跳都没走。

## 二、三条可被断言的性质

```text
① R 仍然是行随机的         每个因子行随机 ⇒ 乘积行随机（凸组合 + 矩阵乘）
② 因果的零模式**逐位保持**  下三角矩阵的乘积还是下三角，而那些 0 是**精确的 0**
③ α = 0 时 R = I          一条把"这个公式在做什么"说清的判据
```

第 ② 条最值钱：它把 day082 的"因果性是逐位的 0"这件事**传到了滚动之后**——
于是"GPT 的滚动图不会有上三角"是一条可失败的断言，而不是"看图觉得没有"。

## 三、本课不回答什么

```text
它回答      "信息至少走几跳能到输出"（一个**上界**）
它不回答    "模型真的用这条路径了吗"——那要看权重本身与下游效果
```

这条边界与 day079 的"它回答的是传得到不到，不是训得好不好"同源。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from smart_research_agent.explainability.errors import (
    AssemblyError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.explainability.extract import aggregate_heads
from smart_research_agent.explainability.types import (
    DEFAULT_ALPHA,
    STREAM_SELF,
    AttentionRecord,
)
from smart_research_agent.math_foundations.linalg import matmul
from smart_research_agent.math_foundations.types import Matrix, matrix_shape

#: 滚动记录的标签前缀.
ROLLOUT_LABEL = "滚动"


def _checked_alpha(alpha: Any) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ParameterError(f"alpha 必须是数，收到 {alpha!r}。")
    resolved = float(alpha)
    if not math.isfinite(resolved):
        raise NumericError(f"alpha 必须是有限数，收到 {alpha!r}。")
    if not 0.0 <= resolved <= 1.0:
        raise ParameterError(
            f"alpha 必须落在 [0, 1]，收到 {alpha}：它是「这一层自己的注意力」与"
            "「直接把输入带过去」两件事的权重，和必须是 1。"
        )
    return resolved


def _layer_index(label: str) -> int:
    """从 ``"层 3 · 头 0"`` 里取出层号（不认识的命名当场拒绝）."""
    head = label.split(" ·")[0]
    parts = head.split(" ")
    if len(parts) != 2 or not parts[1].isdigit():
        raise AssemblyError(
            f"标签 {label!r} 里取不出层号：滚动按「层 i · ...」排序，"
            "因此它只接 head_records 那套命名（单流变体的自注意力）。"
        )
    return int(parts[1])


def aggregate_by_layer(
    records: Sequence[AttentionRecord],
) -> tuple[AttentionRecord, ...]:
    """按层分组、组内**逐格平均**、按层号升序返回（滚动的前置步骤）.

    只接``self``流：交叉注意力的权重是长方形，而"乘法"这件事要求两个因子同形——
    把它混进来会得到一个形状错误的乘积（或者更糟：n_tgt == n_src 时形状刚好对，而含义全变）。
    """
    resolved = tuple(records)
    if not resolved:
        raise ParameterError("aggregate_by_layer 需要至少一条记录。")
    groups: dict[int, list[AttentionRecord]] = {}
    for record in resolved:
        if record.stream != STREAM_SELF:
            raise AssemblyError(
                f"记录 {record.label!r} 的流是 {record.stream}：滚动只接自注意力"
                "（方阵）——交叉注意力的长方形权重与自注意力不能相乘。"
            )
        if record.rows != record.columns:
            raise ShapeError(
                f"记录 {record.label!r} 是 {record.rows}×{record.columns}："
                "自注意力必须是方阵。"
            )
        groups.setdefault(_layer_index(record.label), []).append(record)
    indices = sorted(groups)
    if indices != list(range(len(indices))):
        raise AssemblyError(
            f"层号不连续：{indices}——滚动要求层号从 0 开始且不跳号，"
            "否则「第 L 层乘第 L−1 层」这件事会少乘一层而看不出来。"
        )
    first = groups[indices[0]][0]
    merged: list[AttentionRecord] = []
    for index in indices:
        item = aggregate_heads(groups[index])
        if (item.rows, item.columns) != (first.rows, first.columns):
            raise ShapeError(
                f"层 {index} 的形状 ({item.rows}, {item.columns}) 与层 0 "
                f"({first.rows}, {first.columns}) 不一致：乘法只在同形状上成立。"
            )
        if item.mask != first.mask:
            raise AssemblyError(
                f"层 {index} 的掩码与层 0 不同：滚动的三个性质（行随机、"
                "因果零模式、α = 0 时为单位阵）都要求所有层用同一张掩码。"
            )
        merged.append(item)
    return tuple(merged)


def _blend(record: AttentionRecord, alpha: float) -> Matrix:
    """``Â = α·A + (1−α)·I``（那一项 ``(1−α)·I`` 就是残差的那条直通路）."""
    rows, columns = record.rows, record.columns
    return tuple(
        tuple(
            alpha * record.weights[row][column] + ((1.0 - alpha) if row == column else 0.0)
            for column in range(columns)
        )
        for row in range(rows)
    )


def rollout_weights(
    records: Sequence[AttentionRecord], *, alpha: float = DEFAULT_ALPHA
) -> Matrix:
    """``R = Â_L · … · Â_1``（**按层号升序**左乘：最底层先走）."""
    resolved = _checked_alpha(alpha)
    layers = aggregate_by_layer(records)
    current: Matrix | None = None
    for record in layers:
        blended = _blend(record, resolved)
        current = blended if current is None else matmul(blended, current)
    if current is None:  # pragma: no cover - aggregate_by_layer 已经保证非空
        raise NumericError("滚动至少需要一层。")
    return current


def rollout_record(
    records: Sequence[AttentionRecord], *, alpha: float = DEFAULT_ALPHA
) -> AttentionRecord:
    """滚动的结果作为一条 :class:`AttentionRecord`（标签里写明 α 与层数）."""
    resolved = _checked_alpha(alpha)
    layers = aggregate_by_layer(records)
    weights = rollout_weights(records, alpha=resolved)
    first = layers[0]
    return AttentionRecord(
        label=f"{ROLLOUT_LABEL}（α={resolved}，{len(layers)} 层）",
        weights=weights,
        mask=first.mask,
        stream=STREAM_SELF,
        tokens=first.tokens,
        notes=(
            "Â = α·A + (1−α)·I；R = Â_L ⋯ Â_1",
            "那条 (1−α)·I 就是残差：它让「这一层什么都没做」也有一条直通路",
        ),
    )


def rollout_focus(
    records: Sequence[AttentionRecord], *, alpha: float = DEFAULT_ALPHA
) -> tuple[float, float, float]:
    """``(最后一层的归一化熵, 滚动的归一化熵, 差)``.

    第三个数是"滚动把注意力变尖了多少"——
    **它不保证是正数**（一个已经很尖的模型滚动之后可能更散），
    因此本函数如实返回三个数，而判词留给调用方。
    """
    layers = aggregate_by_layer(records)
    rolled = rollout_record(records, alpha=alpha)
    last = layers[-1]
    return (
        last.normalized_entropy,
        rolled.normalized_entropy,
        last.normalized_entropy - rolled.normalized_entropy,
    )


def preserves_causal_zeroes(
    records: Sequence[AttentionRecord], *, alpha: float = DEFAULT_ALPHA
) -> bool:
    """滚动之后"上三角"是不是**逐位**为 0（第 ② 条性质）.

    ```text
    每个因子 Â 的 i < j 处 = α·0.0 + (1−α)·0.0 = 0.0（**精确**）
    下三角 × 下三角 = 下三角 ⇒ 乘积的上三角也是精确的 0.0
    ```

    **只对因果表有内容**：全开掩码的表在上三角本来就没有 0，
    因此这条判据在那种表上为 ``False``——报告里用的是它的"按掩码"版本
    （:func:`mask_respected_after_rollout`），因为那一版对两种掩码都说得通。
    """
    weights = rollout_weights(records, alpha=alpha)
    rows, columns = matrix_shape(weights)
    return all(
        weights[row][column] == 0.0
        for row in range(rows)
        for column in range(columns)
        if column > row
    )


def mask_respected_after_rollout(
    records: Sequence[AttentionRecord], *, alpha: float = DEFAULT_ALPHA
) -> tuple[int, float]:
    """滚动之后**被掩码挡掉的格子**是否仍然逐位为 0：``(格子数, 最大读数)``.

    它与 :func:`preserves_causal_zeroes` 的差别正是"口径"：

    ```text
    preserves_causal_zeroes      只看上三角（对因果表有内容，对全开表恒为假）
    mask_respected_after_rollout 看**掩码说的每一格**——因此对两种掩码都说得通
    ```

    第二个返回的读数必须是 ``0.0``：一个"看起来很小但仍非零"的读数说明
    某一层的被挡格子其实是脏的（而它会随着层数**被放大**）。
    """
    layers = aggregate_by_layer(records)
    weights = rollout_weights(records, alpha=alpha)
    mask = layers[0].mask
    rows, columns = matrix_shape(weights)
    blocked = 0
    worst = 0.0
    for row in range(rows):
        for column in range(columns):
            if not mask[row][column]:
                blocked += 1
                worst = max(worst, abs(weights[row][column]))
    return blocked, worst


def identity_when_alpha_zero(records: Sequence[AttentionRecord]) -> bool:
    """``α = 0`` 时 ``R`` 是不是单位阵（一条把公式说清的判据）."""
    weights = rollout_weights(records, alpha=0.0)
    rows, columns = matrix_shape(weights)
    return all(
        weights[row][column] == (1.0 if row == column else 0.0)
        for row in range(rows)
        for column in range(columns)
    )


def row_sums(weights: Matrix, *, tolerance: float = 1e-12) -> tuple[float, ...]:
    """逐行和（滚动之后它们应当**仍然是 1**，容差 1e-12）."""
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ParameterError(f"tolerance 必须是数，收到 {tolerance!r}。")
    resolved = float(tolerance)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ParameterError(f"tolerance 必须是有限的非负数，收到 {tolerance!r}。")
    rows, _columns = matrix_shape(weights)
    sums = tuple(math.fsum(weights[row]) for row in range(rows))
    for index, total in enumerate(sums):
        if abs(total - 1.0) > resolved:
            raise NumericError(
                f"第 {index} 行的和是 {total!r}（容差 {resolved}）："
                "滚动保住了行随机性吗？——一个因子算错就会让整个乘积漏质量。"
            )
    return sums


__all__ = [
    "ROLLOUT_LABEL",
    "aggregate_by_layer",
    "identity_when_alpha_zero",
    "mask_respected_after_rollout",
    "preserves_causal_zeroes",
    "rollout_focus",
    "rollout_record",
    "rollout_weights",
    "row_sums",
]
