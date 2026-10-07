"""``visualize``：把训练日志画成**文本**（day095 / M8-D6）.

```text
sparkline(values)        一行"迷你曲线"：每个读数一个档位字符
loss_curve(values)       一张 ASCII 折线图（高度 × 宽度都给得出）
bar_chart(pairs)         一张水平条形图（"哪个配置更好"一眼可见）
summary_block(title, rows)   一张两列表（标签右对齐）
curve_summary(values)    一行读数：步数 / 首末 / 最好的一步与最坏的一步
```

## 为什么值得单列一个模块

训练日志是这一课唯一"给人看"的产物。而一份**只能读数字**的日志有三个具体问题：

```text
① 几十个损失值排成一列，看不出"哪一段在下降、哪一段在平台期"
② "最好的一步"要靠肉眼在一串小数里找，而它正是早停与回滚的判据
③ 两组配置对比时，读者要在两列数字之间来回数——条形图把这件事变成"看长度"
```

因此本模块的三张图都遵守同一条纪律：**它们只做映射，不做判断**。
"哪一步最好"由 ``argmin`` 决定（一个可被断言的整数），图只是把这个整数画出来——
第 ⑦ 条性质断言的正是这件事：**图的极值位置必须等于 ``argmin`` / ``argmax``**。

## 一条必须写下来的边界

文本图会**丢精度**（8 个档位、几十列宽）。因此它的定位是"日志的第一眼"，
而不是"读数的唯一来源"：任何一个要写进结论的数都必须回到 ``study`` 的表里取。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from smart_research_agent.regularization.normalization import as_vector
from smart_research_agent.regularization.errors import ParameterError, ShapeError

#: 八个档位（从最低到最高）。用方块字符是因为它们在等宽字体里高度单调。
GLYPHS = "▁▂▃▄▅▆▇█"

#: sparkline 的缺省宽度上限（超过它就按等间隔抽样——一行不该无限长）。
DEFAULT_SPARK_WIDTH = 72

#: 折线图的缺省尺寸。
DEFAULT_CURVE_HEIGHT = 6
DEFAULT_CURVE_WIDTH = 48

#: 条形图的缺省最长条宽。
DEFAULT_BAR_WIDTH = 24


def _checked_height(value: object, *, name: str, minimum: int = 1) -> int:
    """正整数护栏（高度 / 宽度都走它）."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParameterError(f"{name} 必须是整数，收到 {value!r}。")
    if value < minimum:
        raise ParameterError(f"{name} 必须 >= {minimum}，收到 {value}。")
    return value


def _resample(values: tuple[float, ...], width: int) -> tuple[float, ...]:
    """等间隔抽样到 ``width`` 个点（宽度 >= 长度时原样返回）."""
    if width >= len(values):
        return values
    if width == 1:
        return (values[0],)
    step = (len(values) - 1) / (width - 1)
    return tuple(values[round(index * step)] for index in range(width))


def _levels(values: tuple[float, ...], *, levels: int) -> tuple[int, ...]:
    """把一串数映射到 ``0 … levels-1`` 的档位（全部相等时取中间档）."""
    lowest, highest = min(values), max(values)
    if highest == lowest:
        middle = (levels - 1) // 2
        return tuple(middle for _ in values)
    span = highest - lowest
    return tuple(
        min(levels - 1, max(0, int((value - lowest) / span * (levels - 1) + 1e-12)))
        for value in values
    )


def sparkline(values: Sequence[float], *, width: int = DEFAULT_SPARK_WIDTH) -> str:
    """一行迷你曲线：最低值映到最低档、最高值映到最高档.

    返回的字符数等于**抽样之后**的长度（默认不超过 ``DEFAULT_SPARK_WIDTH``）。
    ``max == min`` 时全部取中间档——"一条平线"与"一条从 0 到 1 的线"不该长得一样。
    """
    checked = as_vector(values, name="values")
    resolved_width = _checked_height(width, name="width")
    sampled = _resample(checked, resolved_width)
    levels = _levels(sampled, levels=len(GLYPHS))
    return "".join(GLYPHS[level] for level in levels)


def loss_curve(
    values: Sequence[float],
    *,
    height: int = DEFAULT_CURVE_HEIGHT,
    width: int = DEFAULT_CURVE_WIDTH,
) -> tuple[str, ...]:
    """一张 ASCII 折线图（每一行是一条水平带，``*`` 标出曲线经过的位置）.

    最上面一行的左端标注最大值、最下面一行标注最小值——
    **两个数都在图上**，因此这张图可以被"读"而不只是被"看"。
    """
    checked = as_vector(values, name="values")
    resolved_height = _checked_height(height, name="height", minimum=3)
    resolved_width = _checked_height(width, name="width", minimum=2)
    sampled = _resample(checked, resolved_width)
    levels = _levels(sampled, levels=resolved_height)
    lowest, highest = min(checked), max(checked)
    lines: list[str] = []
    for row in range(resolved_height - 1, -1, -1):
        bar = "".join("*" if level == row else " " for level in levels)
        if row == resolved_height - 1:
            label = f"{highest:>12.6f} "
        elif row == 0:
            label = f"{lowest:>12.6f} "
        else:
            label = " " * 13
        lines.append(label + "|" + bar)
    lines.append(" " * 13 + "+" + "-" * len(sampled))
    lines.append(" " * 13 + f"  第 1 步 → 第 {len(checked)} 步（抽样成 {len(sampled)} 列）")
    return tuple(lines)


def bar_chart(
    pairs: Sequence[tuple[str, float]], *, width: int = DEFAULT_BAR_WIDTH
) -> tuple[str, ...]:
    """水平条形图：每行的条长与它的数值成正比（最大的那条占满 ``width``）."""
    if not pairs:
        raise ShapeError("空条形图没有意义：至少要有一项。")
    checked: list[tuple[str, float]] = []
    for index, item in enumerate(pairs):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ShapeError(f"第 {index} 项必须是 (标签, 数值)，收到 {item!r}。")
        label, value = item
        if not isinstance(label, str) or not label:
            raise ShapeError(f"第 {index} 项的标签必须是非空字符串，收到 {label!r}。")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ParameterError(f"第 {index} 项的数值必须是非负有限数，收到 {value!r}。")
        checked.append((label, number))
    resolved_width = _checked_height(width, name="width", minimum=1)
    largest = max(value for _label, value in checked)
    label_width = max(len(label) for label, _value in checked)
    lines: list[str] = []
    for label, value in checked:
        filled = 0 if largest == 0.0 else round(value / largest * resolved_width)
        lines.append(f"{label:>{label_width}} |{'█' * filled}{'·' * (resolved_width - filled)}| {value:.6f}")
    return tuple(lines)


def summary_block(title: str, rows: Sequence[tuple[str, str]]) -> tuple[str, ...]:
    """一张两列表（标签右对齐），首行是标题."""
    if not isinstance(title, str) or not title:
        raise ParameterError("标题必须是非空字符串。")
    prepared: list[tuple[str, str]] = []
    for index, item in enumerate(rows):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ShapeError(f"第 {index} 行必须是 (标签, 值)，收到 {item!r}。")
        prepared.append((str(item[0]), str(item[1])))
    width = max((len(label) for label, _value in prepared), default=0)
    lines = [title]
    for label, value in prepared:
        lines.append(f"  {label:>{width}} : {value}")
    return tuple(lines)


def curve_summary(values: Sequence[float]) -> str:
    """一行读数：步数 / 首末 / 最好的一步与最坏的一步（**都带下标**）."""
    checked = as_vector(values, name="values")
    best_index = min(range(len(checked)), key=lambda index: checked[index])
    worst_index = max(range(len(checked)), key=lambda index: checked[index])
    return (
        f"{len(checked)} 步 | 首 {checked[0]:.6f} → 末 {checked[-1]:.6f} | "
        f"最好第 {best_index + 1} 步（{checked[best_index]:.6f}）| "
        f"最坏第 {worst_index + 1} 步（{checked[worst_index]:.6f}）"
    )


__all__ = [
    "DEFAULT_BAR_WIDTH",
    "DEFAULT_CURVE_HEIGHT",
    "DEFAULT_CURVE_WIDTH",
    "DEFAULT_SPARK_WIDTH",
    "GLYPHS",
    "bar_chart",
    "curve_summary",
    "loss_curve",
    "sparkline",
    "summary_block",
]
