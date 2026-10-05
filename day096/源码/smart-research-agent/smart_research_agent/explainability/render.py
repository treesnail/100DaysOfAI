"""``render.py``：把权重画成**文本热力图**，并且能画回去（day083）.

## 一、为什么用文本而不是图

```text
① 它能进仓库     一张 PNG 进 diff 没有意义；而一段文本可以被 review、被逐行比
② 它能被断言     渲染 → 解析 ⇒ 逐级相同（第 4 条性质），因此"图对不对"是一条测试
③ 它零依赖       不需要 matplotlib，任何终端都能看
```

## 二、10 级是有损的，因此"能对回去"是一条纪律

```text
等级   0    1    2    3    4    5    6    7    8    9
字符   空格   .    :    -    =    +    *    #    %    @
区间   [0,1/9) [1/9,2/9) …                     [8/9, 1]
```

第 ``k`` 级的区间是 ``[k/9, (k+1)/9)``，而 ``level_of`` 用**向下取整**。
两处口径（``level_of`` 与 ``LEVELS`` 的下标）必须同源——
测试里有一条"渲染 → 解析 → 逐级相同"把它们钉在一起。

## 三、一处会骗人的写法（本课刻意避开）

```text
坏写法    cell = "#" if value > 0.5 else " "
           ⇒ 0.49 与 0.01 画得一样，而 0.51 与 0.99 也画得一样
好写法    10 级 + 明确的区间 + 图例
```

热力图的价值全在"看得出差别"，因此本包把图例**印在每一段输出里**，
而不是只在文档里写一次。

## 四、两个读数也画出来

```text
profile_bar     一个 profile 的归一化熵 → 一条 width 字符的条
profile_block   一个 profile 的全部读数（熵 / 天花板 / 峰值 / 支撑 / Frobenius / 对角质量）
```
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from smart_research_agent.explainability.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.explainability.types import (
    LEVEL_COUNT,
    LEVELS,
    AttentionRecord,
    HeadProfile,
)

#: 图例（**每一段输出都会印它**：读者要能自己判断那个字符代表多少）.
LEGEND = "等级（10 级，从 0 到 1）：'" + LEVELS + "'（第 0 级是空格）"


def level_of(value: float, *, tolerance: float = 0.0) -> int:
    """把一个权重变成 0..9 的等级（``int(value · 9)``，向下取整）.

    两条校验：

    ```text
    ① 必须是有限数（非有限数画不出来）
    ② 必须落在 [0, 1]（**一个权重不能大于 1**：它是一条分布里的一项）
    ```

    第 ② 条不是形式主义：`1.0000000001` 这种数只可能来自"两张表被加在一起"
    或者"没归一化"，而它会让热力图最右边那一列全是 `@`——看起来完全正常。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterError(f"权重必须是数，收到 {value!r}。")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise NumericError(f"权重必须是有限数，收到 {value!r}。")
    if resolved < -tolerance or resolved > 1.0 + tolerance:
        raise NumericError(
            f"权重 {value!r} 落在 [0, 1] 之外：一个权重是分布里的一项，"
            "它不可能大于 1——越界只可能来自'两张表加在了一起'或'没有归一化'。"
        )
    if resolved <= tolerance:
        return 0
    level = int(resolved * (LEVEL_COUNT - 1))
    return min(LEVEL_COUNT - 1, max(0, level))


def level_char(value: float, *, tolerance: float = 0.0) -> str:
    """一个权重对应的字符."""
    return LEVELS[level_of(value, tolerance=tolerance)]


def char_level(char: str) -> int:
    """一个字符的等级（不认识的字符当场拒绝：解析侧不能"猜一个最接近的"）."""
    if not isinstance(char, str) or len(char) != 1:
        raise ParameterError(f"等级必须是一个字符，收到 {char!r}。")
    if char not in LEVELS:
        raise ParameterError(
            f"未知的等级字符 {char!r}：可选 {LEGEND}——"
            "解析侧不猜，因为「猜错一个字符」会让整张图偏一格而看不出来。"
        )
    return LEVELS.index(char)


def heatmap_block(
    record: AttentionRecord,
    *,
    indent: str = "",
    column_labels: bool = True,
    row_numbers: bool = True,
) -> str:
    """一条记录画成一段文本热力图（标题 + 列表头 + 逐行 + 图例）.

    行格式是**机器可解析**的：``{行号:>3} |{cells}| {峰值说明}``，
    其中 ``cells`` 恰好 ``columns`` 个字符。解析侧只认"恰好两个 ``|``"的行，
    因此标题与图例不会被误读。
    """
    if not isinstance(record, AttentionRecord):
        raise ParameterError(
            f"record 必须是 AttentionRecord，收到 {type(record).__name__}。"
        )
    lines: list[str] = [f"{indent}{record.summary_line()}"]
    if column_labels:
        header = "".join(f"{column:>4}" for column in range(record.columns))
        lines.append(f"{indent}    {header}")
    for row in range(record.rows):
        cells = "".join(level_char(record.weights[row][column]) for column in range(record.columns))
        peak = max(range(record.columns), key=lambda column: record.weights[row][column])
        prefix = f"{row:>3} " if row_numbers else ""
        lines.append(
            f"{indent}{prefix}|{cells}| 峰值 {record.weights[row][peak]:.4f} @ {peak} "
            f"（第 {row} 行）"
        )
    lines.append(f"{indent}{LEGEND}")
    return "\n".join(lines)


def render_records(
    records: Sequence[AttentionRecord],
    *,
    indent: str = "",
    column_labels: bool = True,
) -> str:
    """多条记录拼成一段输出（记录之间空一行）."""
    resolved = tuple(records)
    if not resolved:
        raise ParameterError("渲染需要至少一条记录。")
    return "\n\n".join(
        heatmap_block(record, indent=indent, column_labels=column_labels)
        for record in resolved
    )


def parse_heatmap(text: str) -> tuple[tuple[int, ...], ...]:
    """把一段热力图解析回**等级矩阵**（渲染的逆运算）.

    只认"行首是数字、且整行恰好两个 ``|``"的行——这条判据让标题（三个 ``|``）
    与图例（没有 ``|``）都不会被误读。解析结果与 ``level_of`` 的输入
    **逐级相同**（不是"看起来差不多"）。
    """
    if not isinstance(text, str):
        raise ParameterError(f"text 必须是字符串，收到 {type(text).__name__}。")
    rows: list[tuple[int, ...]] = []
    width: int | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.count("|") != 2:
            continue
        prefix, rest = line.split("|", 1)
        cells, _tail = rest.split("|", 1)
        if not prefix.strip().isdigit():
            continue
        if width is None:
            width = len(cells)
            if width == 0:
                raise ShapeError("热力图的一行是空的。")
        elif len(cells) != width:
            raise ShapeError(
                f"热力图的每行宽度不一致：{width} 与 {len(cells)}——"
                "宽度不一致说明这段文本不是一张表。"
            )
        rows.append(tuple(char_level(char) for char in cells))
    if not rows:
        raise ShapeError(
            "这段文本里没有可解析的热力图行："
            "解析侧只认'行首是数字、且整行恰好两个 |'的行。"
        )
    return tuple(rows)


def normalized_levels(record: AttentionRecord) -> tuple[tuple[int, ...], ...]:
    """一条记录的等级矩阵（**不经过文本**：给"渲染 → 解析"的对照用）."""
    return tuple(
        tuple(level_of(record.weights[row][column]) for column in range(record.columns))
        for row in range(record.rows)
    )


def round_trip(record: AttentionRecord) -> bool:
    """渲染 → 解析是否**逐级**相同（第 4 条性质）."""
    parsed = parse_heatmap(heatmap_block(record))
    return parsed == normalized_levels(record)


def profile_bar(profile: HeadProfile, *, width: int = 20, indent: str = "") -> str:
    """一个 profile 的归一化熵画成一条条（``#`` 是"用到"的部分）."""
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ParameterError(f"宽度必须是 >= 1 的整数，收到 {width!r}。")
    if not isinstance(profile, HeadProfile):
        raise ParameterError(f"profile 必须是 HeadProfile，收到 {type(profile).__name__}。")
    ratio = min(1.0, max(0.0, profile.normalized_entropy))
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"{indent}{bar} {profile.normalized_entropy:.4f}"


def profile_block(profile: HeadProfile, *, indent: str = "", width: int = 20) -> str:
    """一个 profile 的全部读数（熵 / 天花板 / 峰值 / 支撑 / Frobenius / 对角质量）."""
    if not isinstance(profile, HeadProfile):
        raise ParameterError(f"profile 必须是 HeadProfile，收到 {type(profile).__name__}。")
    return "\n".join(
        [
            f"{indent}{profile.label}",
            f"{indent}{profile_bar(profile, width=width)}  ← 归一化熵（1.0 = 贴到天花板）",
            f"{indent}熵 {profile.entropy:.6f} | 天花板 {profile.ceiling:.6f} | "
            f"峰值 {profile.peak_weight:.6f} @ {profile.peak_index}",
            f"{indent}支撑 {profile.support}/{profile.rows * profile.columns} | "
            f"稀疏度 {profile.sparsity:.4f} | Frobenius {profile.frobenius:.6f} | "
            f"对角质量 {profile.diagonal_mass:.6f}",
        ]
    )


def render_profiles(
    profiles: Sequence[HeadProfile], *, indent: str = "  ", width: int = 20
) -> str:
    """多个 profile 各一小段（中间空一行）."""
    resolved = tuple(profiles)
    if not resolved:
        raise ParameterError("渲染需要至少一个 profile。")
    return "\n\n".join(profile_block(profile, indent=indent, width=width) for profile in resolved)


def legend_line() -> str:
    """图例（供报告模板调用）."""
    return LEGEND


def explanation_of(value: float) -> str:
    """一个权重与它的等级（一行说明：``0.1765 → 等级 1（'.'）``）."""
    level = level_of(value)
    return f"{value:.4f} → 等级 {level}（{LEVELS[level]!r}）"


__all__ = [
    "LEGEND",
    "char_level",
    "explanation_of",
    "heatmap_block",
    "legend_line",
    "level_char",
    "level_of",
    "normalized_levels",
    "parse_heatmap",
    "profile_bar",
    "profile_block",
    "render_profiles",
    "render_records",
    "round_trip",
]
