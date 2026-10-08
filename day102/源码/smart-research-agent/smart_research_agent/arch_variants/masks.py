"""``masks.py``：三张掩码、一次组合，以及“这张掩码是因果的吗”（day082）.

掩码是本课唯一的“新东西”，而它的实现**一行都没有重写**：

```text
全开掩码      transformer_core.layers.full_mask（day075）  转发
因果掩码      math_foundations.attention.causal_mask（day073）  转发
填充掩码      **新**：这一张是“数据”的性质，不是“结构”的性质
组合          逐位与（**新**：两张掩码同时生效时的唯一自然含义）
```

## 一、为什么填充掩码是新的，而它长这样

```text
掩码[i][j] 为 True 表示“位置 i 允许看位置 j”
填充位（pad）是**输入数据**的一部分：它没有内容，因此谁也不该把注意力分给它
```

于是第一版会写成 ``allowed[i][j] = not pads[j]``，而它会**当场失败**：
day075 的 ``masked_softmax_rows`` 明确拒绝“某一行的允许位置为空”，
因为那一行的注意力分布**没有定义**（softmax 的分母是 0）。

而一行全 False 在真实数据里**一定会出现**：`tokens = 4`、`pads = [T, T, T, T]`
（整条序列都是填充）时，每一行都会被拒绝。修法是加一条**规则**，而不是加一个特判：

```text
allowed[i][j] = (not pads[j]) or (i == j)
```

它同时给三件事：

```text
① 每一行至少有一个允许的位置（对角线**永远**允许）  ⇒ 那一行的分布总有定义
② 一个填充行只看得见它自己                        ⇒ 它的输出与别的行无关
③ 非填充行看不到任何填充位                        ⇒ 这是填充掩码的本意
```

第 ② 条有一个真实后果，本课把它写进文档而不是含糊过去：
**填充行的输出是“它自己那一点噪声”，没有任何意义**——因此它必须在池化/损失里被丢掉
（day083 的“可解释性”里也会看到同一件事：那一行的注意力是一根对角线）。

## 二、组合 = 逐位与

```text
因果 ∧ 填充     位置 i 只允许看 j <= i 且（j 非填充 或 j == i）
```

没有第三条规则可选：两张掩码各自说“这些位置不许看”，
同时生效时的含义**只能是交集**——写成并集的话，多看到的位置谁也不会报错。

## 三、``mask_is_causal`` —— 判据不读 ``causal`` 这个参数

```text
本函数只看掩码本身：allowed(i, j) ⇒ j <= i，且 mask[0][0] 为 True
```

这条定义的方式很重要：它让“因果性”成为**表的性质**而不是**调用的性质**——
于是它可以被用在三处：显式掩码（decoder 那一流）、组合掩码（因果 ∧ 填充）、
以及 ``probe`` 里那张**实测**依赖表（那里输入的是浮点读数，不是布尔表，
因此 ``probe`` 用的是它自己的阈值版判据）。
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from smart_research_agent.arch_variants.errors import (
    AssemblyError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.math_foundations.attention import causal_mask as _causal_mask
from smart_research_agent.math_foundations.types import Matrix
from smart_research_agent.transformer_core.layers import full_mask as _full_mask

#: 两种结构性的掩码名（填充是数据的性质，因此不在这里）.
MASK_FULL = "full"
MASK_CAUSAL = "causal"

MASK_KINDS: tuple[str, ...] = (MASK_FULL, MASK_CAUSAL)

MASK_DESCRIPTIONS: dict[str, str] = {
    MASK_FULL: "全开：每一行都允许看所有位置（**双向**——编码器的选择）",
    MASK_CAUSAL: "因果：第 i 行只允许看 j <= i（**只看过去**——解码器那一流的选择）",
}

Mask = tuple[tuple[bool, ...], ...]


def _checked_size(size: Any) -> int:
    if isinstance(size, bool) or not isinstance(size, int):
        raise ParameterError(f"掩码边长必须是整数，收到 {size!r}。")
    if size < 1:
        raise ParameterError(f"掩码边长必须 >= 1，收到 {size}。")
    return size


def mask_of(kind: str, size: int) -> Mask:
    """按名字取一张结构性掩码（``full`` / ``causal``）——**转发 day073/075**."""
    if not isinstance(kind, str):
        raise ParameterError(f"掩码名必须是字符串，收到 {type(kind).__name__}。")
    if kind not in MASK_KINDS:
        raise ParameterError(
            f"未知的掩码名 {kind!r}：可选 {', '.join(MASK_KINDS)}——"
            "本包不接受'拼错就用全开'，因为那会让'漏掉了因果'这件事彻底看不见。"
        )
    checked = _checked_size(size)
    if kind == MASK_FULL:
        return _full_mask(checked)
    return tuple(tuple(row) for row in _causal_mask(checked))


def padding_mask(pads: Sequence[bool]) -> Mask:
    """填充掩码：``allowed[i][j] = (not pads[j]) or (i == j)``.

    加 ``or (i == j)`` 的理由写在模块说明里（每一行必须至少有一个允许的位置，
    而一行全 False 在“整条序列都是填充”时一定会出现）。它给三件事，**三条都要说清**：

    ```text
    ① 每一行至少有一个允许的位置（对角线永远允许）  ⇒ 那一行的分布总有定义
    ② 非填充行看不到任何填充位                      ⇒ 这是填充掩码的本意
    ③ 填充行仍然看得到**非填充**的位置与它自己        ⇒ 它的输出没有意义
    ```

    第 ③ 条是本包刻意**不**处理的：把填充行与其他人彻底隔离需要“按行按列同时判”，
    而它换来的只是“噪声更小”——真正该做的是在池化与损失里把那一行**丢掉**
    （day083 的可解释性里会再看到同一件事：那一行的注意力是一根对角线）。
    极端情形（整条序列都是填充）下第 ③ 条会自动退化成“只有对角线”。
    """
    resolved = tuple(pads)
    if not resolved:
        raise ParameterError("填充掩码至少要有一个位置（长度为 0 的序列没有定义）。")
    for index, value in enumerate(resolved):
        if not isinstance(value, bool):
            raise ParameterError(
                f"第 {index} 个填充标记必须是 bool，收到 {value!r}："
                "填充是一张**表**，而不是一个可以随便填的数。"
            )
    size = len(resolved)
    return tuple(
        tuple((not resolved[column]) or (row == column) for column in range(size))
        for row in range(size)
    )


def combine_masks(*masks: Mask) -> Mask:
    """两张（或更多）掩码的**交集**：同时生效时唯一自然的含义.

    形状必须一致：一张 ``(3, 5)`` 的掩码配一段 4 个 token 的序列时，
    交集这件事根本没有定义（它不会报错，只会让某几行悄悄少看一个位置）。
    """
    if not masks:
        raise ParameterError("组合掩码至少要给一张掩码。")
    first = _checked_mask(masks[0])
    rows, columns = len(first), len(first[0])
    for index, item in enumerate(masks[1:], start=1):
        checked = _checked_mask(item)
        if len(checked) != rows or len(checked[0]) != columns:
            raise ShapeError(
                f"第 {index} 张掩码的形状 ({len(checked)}, {len(checked[0])}) "
                f"与第 1 张 ({rows}, {columns}) 不一致：交集只在同形状上成立。"
            )
        first = tuple(
            tuple(
                first[row][column] and checked[row][column] for column in range(columns)
            )
            for row in range(rows)
        )
    # 交集也可能**把某一行清空**（两张各自有允许位置的掩码，交集却可以什么都没有）——
    # 这一条必须在这里挡住：`masked_softmax_rows` 会在更远的地方抛错，
    # 而那时报错信息与“哪两张掩码组合出的问题”已经隔了很远。
    return _checked_mask(first)


def _checked_mask(mask: Any) -> Mask:
    """把一张掩码还原成规范形状（每一行至少一个允许的位置）."""
    if not mask:
        raise ParameterError("掩码不能为空。")
    try:
        rows = len(mask)
        widths = {len(row) for row in mask}
    except TypeError as error:  # pragma: no cover - 只有传入非序列时触发
        raise ShapeError(f"掩码必须是可迭代的行序列，收到 {type(mask).__name__}。") from error
    if len(widths) != 1:
        raise ShapeError(f"掩码每行的宽度不一致：{sorted(widths)}。")
    width = widths.pop()
    if width == 0:
        raise ShapeError("掩码的每一行都不能为空。")
    checked: list[tuple[bool, ...]] = []
    for index in range(rows):
        row = tuple(bool(value) for value in mask[index])
        if not any(row):
            raise AssemblyError(
                f"掩码第 {index} 行没有任何允许的位置：这一行的注意力分布没有定义——"
                "填充掩码应当用 `or (i == j)` 保住对角线（见 padding_mask）。"
            )
        checked.append(row)
    return tuple(checked)


def mask_allowed_counts(mask: Mask) -> tuple[int, ...]:
    """每一行允许的位置数（因果掩码下应当是 ``1, 2, ..., n``）."""
    checked = _checked_mask(mask)
    return tuple(sum(1 for value in row if value) for row in checked)


def mask_allowed_pairs(mask: Mask) -> int:
    """允许的位置对总数（``full`` 是 ``n²``，``causal`` 是 ``n(n+1)/2``）."""
    return sum(mask_allowed_counts(mask))


def mask_to_floats(mask: Mask) -> Matrix:
    """把掩码变成 0/1 矩阵（**给对比用**：实测依赖表要和它逐格比）."""
    checked = _checked_mask(mask)
    return tuple(tuple(1.0 if value else 0.0 for value in row) for row in checked)


def mask_is_shape_only(mask: Mask, rows: int, columns: int) -> bool:
    """掩码的形状是否是 ``(rows, columns)``（**长方形也允许**：交叉注意力就没有掩码）."""
    checked = _checked_mask(mask)
    return len(checked) == rows and len(checked[0]) == columns


def mask_fits_sequence(mask: Mask, length: int) -> Mask:
    """检查掩码边长与序列长度一致，并原样返回（**只有一处实现**）."""
    checked = _checked_mask(mask)
    if len(checked) != length:
        raise ShapeError(
            f"掩码边长 {len(checked)} 与序列长度 {length} 不一致："
            "一张对不上的掩码不会报错，它只会让某几行悄悄多看或少看一个位置。"
        )
    return checked


def mask_is_causal(mask: Mask) -> bool:
    """这张表是不是因果的：``allowed(i, j) ⇒ j <= i``，且 ``mask[0][0]`` 为 True.

    **判据不读 ``causal`` 这个参数**（模块说明第三节）：它让“因果”成为表的性质，
    于是同一张判据能用在显式掩码、组合掩码与实测依赖表上。
    """
    checked = _checked_mask(mask)
    rows, columns = len(checked), len(checked[0])
    if rows != columns:
        raise ShapeError(
            f"因果性是**同一段序列内部**的性质，因此它只对 (n, n) 的方阵有定义；"
            f"收到 ({rows}, {columns})——一张长方形的表要么是交叉注意力（它不该有掩码），"
            "要么是把两路的长度搞混了。"
        )
    if not checked[0][0]:
        return False
    return all(
        not (checked[row][column] and column > row)
        for row in range(rows)
        for column in range(columns)
    )


def mask_leaks(mask: Mask) -> tuple[tuple[int, int], ...]:
    """**允许看未来**的那些格子（因果掩码下应当是空元组）.

    这张表是“因果”最直接的证据：它把“第 i 行多看了哪几个位置”变成一串坐标，
    而坐标可以被印出来、被计数、被比较——比“写了 causal=True”强得多。
    """
    checked = _checked_mask(mask)
    return tuple(
        (row, column)
        for row, values in enumerate(checked)
        for column, value in enumerate(values)
        if value and column > row
    )


def mask_summary(mask: Mask) -> str:
    """一行说明：``4×4 | 允许 10/16 | 因果 | 泄漏格子 0 个``."""
    checked = _checked_mask(mask)
    rows, columns = len(checked), len(checked[0])
    allowed = mask_allowed_pairs(checked)
    causal = (
        "因果"
        if rows == columns and mask_is_causal(checked)
        else ("非方阵（无因果可言）" if rows != columns else "非因果")
    )
    leaks = len(mask_leaks(checked))
    return (
        f"{rows}×{columns} | 允许 {allowed}/{rows * columns} | {causal} | 泄漏格子 {leaks} 个"
    )


def mask_entropy_ceiling(mask: Mask) -> float:
    """这一张掩码允许的最大注意力熵（``ln(每行允许的位置数)`` 的平均）.

    它是“能看到的范围”的一个读数：全开掩码下它是 ``ln n``，
    因果掩码下它是 ``(1/n)·Σ ln(i+1)``——**熵的上限被掩码压低了**，
    这也是 day083（可解释性）里“熵要跟天花板比”这句话的来源。
    """
    counts = mask_allowed_counts(mask)
    return sum(math.log(count) for count in counts) / len(counts)


def dependency_from_floats(values: Matrix, *, tolerance: float = 0.0) -> Mask:
    """把一张**实数**依赖表（如 ``probe`` 的实测读数）按阈值变成掩码.

    ``tolerance = 0.0`` 是本课的默认：被掩码挡掉的格子**逐位为 0.0**，
    因此“是否依赖”这个问题在默认口径下是精确的（不需要容差）。
    一旦把阈值调到 1e-12 以上，那些“看起来像 0”的格子就会被当成依赖——
    这条接口存在的意义正是让读者可以**亲手把阈值调大**，看这张表怎么退化。
    """
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ParameterError(f"tolerance 必须是数，收到 {tolerance!r}。")
    resolved = float(tolerance)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ParameterError(f"tolerance 必须是有限的非负数，收到 {tolerance!r}。")
    if not values:
        raise ShapeError("依赖表不能为空。")
    width = len(values[0])
    if any(len(row) != width for row in values):
        raise ShapeError("依赖表的每行宽度必须一致。")
    return tuple(tuple(bool(value > resolved) for value in row) for row in values)


def shape_of_mask(mask: Mask) -> tuple[int, int]:
    """掩码的形状（**不调用 matrix_shape**：掩码是布尔表，不是数值矩阵）."""
    checked = _checked_mask(mask)
    return (len(checked), len(checked[0]))


__all__ = [
    "MASK_CAUSAL",
    "MASK_DESCRIPTIONS",
    "MASK_FULL",
    "MASK_KINDS",
    "Mask",
    "combine_masks",
    "dependency_from_floats",
    "mask_allowed_counts",
    "mask_allowed_pairs",
    "mask_entropy_ceiling",
    "mask_fits_sequence",
    "mask_is_causal",
    "mask_is_shape_only",
    "mask_leaks",
    "mask_of",
    "mask_summary",
    "mask_to_floats",
    "padding_mask",
    "shape_of_mask",
]
