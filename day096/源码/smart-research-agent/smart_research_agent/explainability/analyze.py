"""``analyze.py``：从权重表读出**可以被写进报告**的读数（day083）.

## 一、六个读数，各自回答一个问题

```text
profile_of       这张表有多尖？（熵 / 天花板 / 归一化熵 / 峰值 / 支撑 / Frobenius / 对角质量）
layer_profiles   每层"平均头"的读数（滚动与逐层对照都用它）
head_redundancy  同一层的几个头**在多大程度上看同一件事**？（逐格余弦）
offset_mass      质量落在"离对角线 k 格"上的比例（k = 0 是自看自）
attribution_of   **谁在被看**？（列和——与"每一行看哪里"是两件事）
dead_heads       哪些头"几乎没有偏好"？（归一化熵贴近 1）
```

## 二、"尖"这件事必须转成归一化熵

```text
熵本身不可比      因果掩码下第 0 行只能看到自己（天花板 ln 1 = 0）
归一化熵          熵 / 天花板 ∈ [0, 1]      ← 跨层、跨变体可比
```

## 三、余弦相似度里的一处细节

`head_redundancy` 用**逐格余弦**（把 (n, n) 展平成一个向量），
而两个头的行和都是 1 ⇒ 余弦恒为正、且"完全一样"时恰好是 1.0。
它挡住的是"多头退化成一头"这种**不会报错**的失败（每头的输出仍然合法）。

但它有一条已知的边界，本课把它写下来而不是藏起来：

```text
余弦度量的是"两张表的形状像不像"，不区分"看同一个位置"与"看同一片区域"
⇒ 因此它只作为**一条读数**，`dead_heads` 用的是另一条（归一化熵）
```

## 四、`offset_mass` 与"归纳头"

```text
offset_mass(record, k) = 平均每行在 (i, i−k) 这一条斜线上的质量
k = 0  自看自（对角线）
k = 1  看前一个位置（**归纳头**的第一特征：它常在这里出现）
k = 2  看前两个位置
```

day075 的 induction 任务与 day076 的"可达性"读数都指向这件事；
本课只负责把它量出来（**不解释它为什么值得学**——那是训练的目标函数决定的）。
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
    DEFAULT_THRESHOLD,
    STREAM_SELF,
    AttentionRecord,
    Attribution,
    HeadProfile,
    mean_of,
)
from smart_research_agent.math_foundations.types import matrix_shape

#: "这个头几乎没有偏好"的判据：归一化熵高于它就算"太均匀".
DEFAULT_DEAD_THRESHOLD = 0.95

#: 逐格余弦里对"全零表"的保护（行随机的表不可能全零，但减法之后的差表可能）.
_EPSILON = 1e-12


def _checked_record(record: Any) -> AttentionRecord:
    if not isinstance(record, AttentionRecord):
        raise ParameterError(
            f"record 必须是 AttentionRecord，收到 {type(record).__name__}。"
        )
    return record


def _checked_threshold(value: Any, *, name: str = "threshold") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterError(f"{name} 必须是数，收到 {value!r}。")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ParameterError(f"{name} 必须是有限的非负数，收到 {value!r}。")
    return resolved


def diagonal_mass(record: AttentionRecord) -> float:
    """对角质量：平均每行"看自己"的比例（方阵才有定义）."""
    checked = _checked_record(record)
    if checked.rows != checked.columns:
        raise AssemblyError(
            f"对角质量只对**方阵**有定义（自己看自己），收到 "
            f"{checked.rows}×{checked.columns}——长方形的那一张是交叉注意力。"
        )
    return mean_of(
        tuple(checked.weights[index][index] for index in range(checked.rows))
    )


def offset_mass(record: AttentionRecord, offset: int) -> float:
    """平均每行落在"离对角线 ``offset`` 格"上的质量（``offset = 0`` 是对角线）.

    ``offset >= n`` 时给出 ``0.0``（那一条斜线上没有格子）——
    这不是"错误"，而是一个合法的读数："这一层的注意力不可能看那么远"。
    """
    checked = _checked_record(record)
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise ParameterError(f"偏移必须是整数，收到 {offset!r}。")
    if offset < 0:
        raise ParameterError(
            f"偏移必须 >= 0，收到 {offset}：本包只量「向后看」的那几条斜线"
            "（向前看在因果掩码下恒为 0，量它没有信息量）。"
        )
    if offset >= checked.columns:
        return 0.0
    values = tuple(
        checked.weights[row][row - offset]
        for row in range(offset, checked.rows)
    )
    if not values:
        return 0.0
    return mean_of(values)


def attribution_of(record: AttentionRecord, *, top: int = 3) -> Attribution:
    """列和：**谁在被看**（归一化到总和为 1）."""
    checked = _checked_record(record)
    if isinstance(top, bool) or not isinstance(top, int) or top < 1:
        raise ParameterError(f"top 必须是 >= 1 的整数，收到 {top!r}。")
    if top > checked.columns:
        raise ParameterError(
            f"top = {top} 超过了列数 {checked.columns}：一份「最受关注的几个位置」"
            "不能比位置本身还多。"
        )
    total = math.fsum(
        checked.weights[row][column]
        for row in range(checked.rows)
        for column in range(checked.columns)
    )
    if total <= 0.0:  # pragma: no cover - 行随机的表不可能总和为 0
        raise NumericError("权重总和为 0：这张表不是一条分布。")
    mass = tuple(
        math.fsum(checked.weights[row][column] for row in range(checked.rows)) / total
        for column in range(checked.columns)
    )
    order = sorted(range(checked.columns), key=lambda index: (-mass[index], index))
    return Attribution(
        label=checked.label,
        mass=mass,
        top_positions=tuple(order[:top]),
        columns=checked.columns,
    )


def support_of(record: AttentionRecord, *, threshold: float = DEFAULT_THRESHOLD) -> int:
    """有多少个格子的权重 **大于** 阈值（"它用了几个位置"）."""
    checked = _checked_record(record)
    resolved = _checked_threshold(threshold)
    return sum(
        1
        for row in range(checked.rows)
        for column in range(checked.columns)
        if checked.weights[row][column] > resolved
    )


def frobenius_of(record: AttentionRecord) -> float:
    """逐格平方和的平方根（行随机表的取值范围是 ``[1, √n]``，越尖越大）."""
    checked = _checked_record(record)
    return math.sqrt(
        math.fsum(
            value * value for row in checked.weights for value in row
        )
    )


def peak_of(record: AttentionRecord) -> tuple[float, int]:
    """``(最大权重, 它的列号)``（并列时取最小的列号）."""
    checked = _checked_record(record)
    best = 0.0
    best_index = 0
    for row in range(checked.rows):
        for column in range(checked.columns):
            value = checked.weights[row][column]
            if value > best:
                best = value
                best_index = column
    return best, best_index


def profile_of(record: AttentionRecord, *, threshold: float = DEFAULT_THRESHOLD) -> HeadProfile:
    """一个 profile：把上面几个读数装进一条记录（**报告里的那一份**）.

    ``diagonal_mass`` 只在**自注意力**那一条流上有意义（"看自己"这件事对交叉注意力
    不存在：它的列是**源位置**，与行不是同一段序列）。因此交叉记录上它记作 ``0.0``，
    并在 notes 里说明——而不是算出一个"看起来正常"的数。
    """
    checked = _checked_record(record)
    resolved = _checked_threshold(threshold)
    peak_weight, peak_index = peak_of(checked)
    is_self = checked.stream == STREAM_SELF
    notes = [f"支撑阈值 {resolved}"]
    if not is_self:
        notes.append("交叉那一流没有「对角线」这件东西：对角质量记作 0.0")
    return HeadProfile(
        label=checked.label,
        entropy=checked.mean_entropy,
        ceiling=checked.mean_ceiling,
        peak_weight=peak_weight,
        peak_index=peak_index,
        support=support_of(checked, threshold=resolved),
        frobenius=frobenius_of(checked),
        diagonal_mass=diagonal_mass(checked) if is_self else 0.0,
        rows=checked.rows,
        columns=checked.columns,
        notes=tuple(notes),
    )


def profiles_of(
    records: Sequence[AttentionRecord], *, threshold: float = DEFAULT_THRESHOLD
) -> tuple[HeadProfile, ...]:
    """逐条 profile（空序列当场拒绝）."""
    resolved = tuple(records)
    if not resolved:
        raise ParameterError("profiles_of 需要至少一条记录。")
    return tuple(profile_of(record, threshold=threshold) for record in resolved)


def cosine_similarity(left: AttentionRecord, right: AttentionRecord) -> float:
    """两张同形状表的**逐格余弦**（完全一样时是 1.0，正交时是 0.0）."""
    first = _checked_record(left)
    second = _checked_record(right)
    if (first.rows, first.columns) != (second.rows, second.columns):
        raise ShapeError(
            f"两张表的形状不同：({first.rows}, {first.columns}) 与 "
            f"({second.rows}, {second.columns})——余弦只在同形状上有定义。"
        )
    flat_left = [value for row in first.weights for value in row]
    flat_right = [value for row in second.weights for value in row]
    dot = math.fsum(a * b for a, b in zip(flat_left, flat_right, strict=True))
    norm_left = math.sqrt(math.fsum(a * a for a in flat_left))
    norm_right = math.sqrt(math.fsum(b * b for b in flat_right))
    if norm_left <= _EPSILON or norm_right <= _EPSILON:
        raise NumericError(
            "有一张表的 Frobenius 范数接近 0：行随机的表不可能这样——"
            "它说明这张表没有归一化（或者根本不是一张权重表）。"
        )
    return dot / (norm_left * norm_right)


def head_redundancy(records: Sequence[AttentionRecord]) -> tuple[tuple[float, ...], ...]:
    """同一层几个头之间的余弦矩阵（对角线恒为 1）.

    入口要求"这一层的几张表形状与掩码都相同"——否则余弦比的是两件事
    （与 :func:`extract.aggregate_heads` 同一条纪律）。
    """
    resolved = tuple(records)
    if not resolved:
        raise ParameterError("head_redundancy 需要至少一条记录。")
    first = resolved[0]
    for index, record in enumerate(resolved):
        if (record.rows, record.columns) != (first.rows, first.columns):
            raise ShapeError(f"第 {index} 条记录的形状与第 1 条不同。")
        if record.mask != first.mask:
            raise AssemblyError(
                f"第 {index} 条记录的掩码与第 1 条不同：不同掩码的两张表之间比余弦，"
                "比的是'能看到多少'而不是'看了哪里'。"
            )
    return tuple(
        tuple(cosine_similarity(left, right) for right in resolved) for left in resolved
    )


def layer_profiles(
    records: Sequence[AttentionRecord], *, threshold: float = DEFAULT_THRESHOLD
) -> tuple[HeadProfile, ...]:
    """每层"平均头"的 profile（按标签里的"层 i"分组）.

    分组的判据是**标签前缀**（``"层 1 · ..."``）——因此它只接
    :func:`extract.head_records` 那套命名。别的命名（如 encoder_decoder 的
    "编码器 i · 头 0"）会被显式拒绝，而不是被悄悄丢掉几条。
    """
    resolved = tuple(records)
    if not resolved:
        raise ParameterError("layer_profiles 需要至少一条记录。")
    groups: dict[str, list[AttentionRecord]] = {}
    order: list[str] = []
    for record in resolved:
        if " ·" not in record.label:
            raise AssemblyError(
                f"记录 {record.label!r} 的标签里没有「 · 」分隔符："
                "本函数按「层 i · ...」分组，因此它只接 head_records 那套命名。"
            )
        prefix = record.label.split(" ·")[0]
        if prefix not in groups:
            groups[prefix] = []
            order.append(prefix)
        groups[prefix].append(record)
    profiles: list[HeadProfile] = []
    for prefix in order:
        merged = aggregate_heads(groups[prefix])
        profiles.append(profile_of(merged, threshold=threshold))
    return tuple(profiles)


def dead_heads(
    records: Sequence[AttentionRecord], *, threshold: float = DEFAULT_DEAD_THRESHOLD
) -> tuple[str, ...]:
    """归一化熵高于阈值的那些头（"几乎没有偏好"）.

    保留它们而不是丢掉：一个"太均匀"的头是一条**读数**，
    而"把它过滤掉"这件事会让报告里少一行——那条缺失本身没有记录。
    """
    resolved = _checked_threshold(threshold, name="threshold")
    if resolved > 1.0:
        raise ParameterError(
            f"阈值 {threshold} 超过 1：归一化熵最大就是 1（贴到天花板），"
            "因此大于 1 的阈值永远筛不出任何头，而报告会看起来「一切正常」。"
        )
    profiles = profiles_of(records)
    return tuple(profile.label for profile in profiles if profile.normalized_entropy >= resolved)


def focus_verdict(profile: HeadProfile, *, sharp: float = 0.7) -> str:
    """按归一化熵给一句判词（``尖`` / ``中等`` / ``接近均匀``）.

    阈值是可传的：**"多尖算尖"是调用方的判据**，不是本包的。
    """
    if not isinstance(profile, HeadProfile):
        raise ParameterError(f"profile 必须是 HeadProfile，收到 {type(profile).__name__}。")
    resolved = _checked_threshold(sharp, name="sharp")
    if resolved > 1.0:
        raise ParameterError(f"sharp 必须 <= 1，收到 {sharp}。")
    ratio = profile.normalized_entropy
    if ratio < resolved:
        return "尖"
    if ratio < (1.0 + resolved) / 2.0:
        return "中等"
    return "接近均匀"


def shape_of_record(record: AttentionRecord) -> tuple[int, int]:
    """一条记录的形状（转发 ``matrix_shape``，供报告模板调用）."""
    return matrix_shape(_checked_record(record).weights)


def entropy_gap(sharp: AttentionRecord, blunt: AttentionRecord) -> float:
    """两张表的归一化熵之差（正数表示前者更尖）.

    归一化之后才能比：两张表的掩码可能不同（例如"全开 vs 因果"），
    而它们的**天花板不同**——直接比熵会把掩码的差别算进去。
    """
    return blunt.normalized_entropy - sharp.normalized_entropy


def summarise(profiles: Sequence[HeadProfile]) -> str:
    """一行汇总：``n 个读数 | 平均归一化熵 x | 最尖 y（label）| 最散 z（label）``."""
    resolved = tuple(profiles)
    if not resolved:
        raise ParameterError("summarise 需要至少一个 profile。")
    for index, profile in enumerate(resolved):
        if not isinstance(profile, HeadProfile):
            raise ParameterError(
                f"第 {index} 个不是 HeadProfile，收到 {type(profile).__name__}："
                "汇总要读 label 与 normalized_entropy，一个别的东西会让报错"
                "发生在离调用点很远的地方。"
            )
    ratios = tuple(profile.normalized_entropy for profile in resolved)
    sharpest = min(resolved, key=lambda profile: profile.normalized_entropy)
    bluntest = max(resolved, key=lambda profile: profile.normalized_entropy)
    return (
        f"{len(resolved)} 个读数 | 平均归一化熵 {mean_of(ratios):.4f} | "
        f"最尖 {sharpest.normalized_entropy:.4f}（{sharpest.label}）| "
        f"最散 {bluntest.normalized_entropy:.4f}（{bluntest.label}）"
    )


__all__ = [
    "DEFAULT_DEAD_THRESHOLD",
    "attribution_of",
    "cosine_similarity",
    "dead_heads",
    "diagonal_mass",
    "entropy_gap",
    "focus_verdict",
    "frobenius_of",
    "head_redundancy",
    "layer_profiles",
    "offset_mass",
    "peak_of",
    "profile_of",
    "profiles_of",
    "shape_of_record",
    "summarise",
    "support_of",
]
