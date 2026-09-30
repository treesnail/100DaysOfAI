"""``positional_encoding`` 的造表、注入与前向/反向（day078 / M7-D3）.

## 这一课的全部算术只有一行

```text
injected[i] = inputs[i] + table[positions[i]]
```

一行加法就是“位置编码”这件事在实现上的全部动作。**值钱的部分不在这一行**，
而在它前后的三件事：

```text
① 造表      正弦表由公式给出（无参数、可外推）；可学习表由参数给出（定长、越界必须拒绝）
② 它的两条性质  「行范数恒定」与「内积只依赖位移」——只在**对齐频率**时成立
③ 反向的两步  dx 逐位传回（恒等）；dTable 按位置**累加**（漏掉不会报错的那一处）
```

## 两条性质为什么值钱（而不是“数学上的漂亮”）

day075 给本课留了一句话：**无掩码时注意力是置换等变的——它不知道顺序**。
把顺序交给模型的最省事做法就是“给第 i 行加一个只依赖 i 的向量”。
但这件事有一个前提：**“第 i 行加的那个向量”必须让“位移”在向量空间里有意义**。
对齐频率的正弦表恰好给出这一点：

```text
PE(p + δ) = R_δ · PE(p)          R_δ 是分块正交旋转（每一对频率一个 2×2 旋转块）
⇒ <PE(p), PE(q)> = Σ_i cos((p − q)/f_i)       只依赖 p − q
```

于是“相对距离”变成一件**算得出来**的事，而不是“希望模型自己学会”。
可学习表**不保证**这一点：它的行是自由参数，位移律要靠训练去逼近（或者根本不成立）。

## 一处必须说清的差别：day073 那一份的 ``cos`` 用了隔壁的频率

day073 的 ``math_foundations.attention.positional_encoding`` 是

```python
frequency = position / (base ** (index / dimension))
row.append(math.sin(frequency) if index % 2 == 0 else math.cos(frequency))
```

也就是说：第 ``2i`` 维用 ``base^(2i/d)``、第 ``2i+1`` 维用 ``base^((2i+1)/d)``——
**``sin`` 与 ``cos`` 不在同一个频率上**。这一份“错位”有两个立刻可测的后果：

```text
① sin² + cos² = 1 不再成立 → 每一行的范数不再是 √(d/2)（甚至随位置变化）
② PE(p + δ) 不再是 PE(p) 的旋转 → 内积不再只依赖 p − q
```

本课因此把两种配对都实现出来（``pairing=aligned / staggered``），
并让性质检查**同时钉住两边**：对齐那一种两条都成立，错位那一种两条都不成立。
**“看起来像正弦表”与“是正弦表”的差别，在这里第一次变成两条可断言的不等式。**
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from smart_research_agent.math_foundations.attention import POSITIONAL_BASE as DAY073_BASE
from smart_research_agent.math_foundations.probability import uniforms
from smart_research_agent.math_foundations.types import (
    Matrix,
    Vector,
    matrix_shape,
    validate_matrix,
)
from smart_research_agent.positional_encoding.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.positional_encoding.types import (
    ENCODING_LEARNABLE,
    ENCODING_SINUSOIDAL,
    INIT_RANDOM,
    INIT_SCALE,
    INIT_SINUSOIDAL,
    INITIALIZERS,
    PAIRING_ALIGNED,
    PAIRING_STAGGERED,
    PAIRINGS,
    EncodingTable,
    PositionalForward,
    PositionalGradients,
    PositionalShape,
    add_matrices,
    default_positions,
    gather_rows,
    scatter_add_rows,
    validate_positions,
)
from smart_research_agent.transformer_core.layers import (
    attention_backward,
    masked_mean_squared_error,
    masked_mse_gradient,
    mean_squared_error,
    mse_gradient,
    row_argmax_hits,
    self_attention,
)
from smart_research_agent.transformer_core.types import AttentionParams

#: 位置编码的频率基数（**与 day073 同值**，导入期把两者钉成相等）.
POSITIONAL_BASE = 10000.0

if POSITIONAL_BASE != DAY073_BASE:  # pragma: no cover - 只在有人改常量时触发
    raise NumericError(
        f"本课的频率基数 {POSITIONAL_BASE} 与 day073 的 {DAY073_BASE} 不一致："
        "同一个公式在两处必须逐位相同，否则'两天的结果对不上'会变成一个查不出来的问题。"
    )


def _checked_position_count(value: Any, *, name: str) -> int:
    """校验“位置的个数”（正整数）."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParameterError(f"{name} 必须是整数，收到 {value!r}。")
    if value < 1:
        raise ParameterError(f"{name} 必须 >= 1，收到 {value}。")
    return value


def _checked_dimension(dimension: Any) -> int:
    """校验维度（正整数）."""
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise ParameterError(f"dimension 必须是整数，收到 {dimension!r}。")
    if dimension < 2:
        raise ParameterError(
            f"dimension 必须 >= 2，收到 {dimension}：位置编码的每一维都与一个频率配对，"
            "维数 1 时没有可配对的频率（正弦表至少要一对）。"
        )
    return dimension


def _checked_even_dimension(dimension: Any, *, what: str) -> int:
    """校验维度是**偶数**（正弦表的两条性质都以它为前提）."""
    checked = _checked_dimension(dimension)
    if checked % 2 != 0:
        raise ShapeError(
            f"{what}要求维数是偶数，收到 {checked}：奇数维时最后一维只有一个 "
            "sin、没有配对的 cos，于是 'sin² + cos² = 1' 在那一维上不成立——"
            "每一行的范数会随位置变化，而'位移是旋转'这条性质也不再成立。"
        )
    return checked


def _checked_base(base: Any) -> float:
    """校验频率基数是正的有限数且 > 1."""
    if isinstance(base, bool) or not isinstance(base, (int, float)):
        raise ParameterError(f"频率基数必须是数，收到 {base!r}。")
    resolved = float(base)
    if not math.isfinite(resolved) or resolved <= 1.0:
        raise ParameterError(
            f"频率基数必须 > 1，收到 {resolved!r}：base = 1 时所有频率都是 1，"
            "每一维都在同一个尺度上振荡——位置信息会退化成'每一维都一样'。"
        )
    return resolved


def _checked_pairing(pairing: Any) -> str:
    """校验配对方式."""
    if pairing not in PAIRINGS:
        raise ParameterError(
            f"未知的频率配对方式 {pairing!r}：可选 {', '.join(PAIRINGS)}——"
            "本包不为未知配对挑一个默认值。"
        )
    return str(pairing)


# ---------------------------------------------------------------------- 造表


def frequency_of(pair: int, dimension: int, *, base: float = POSITIONAL_BASE) -> float:
    """第 ``pair`` 对频率的**周期** ``f_i = base^(2i/d)``.

    ``i = 0`` 时 ``f = 1``（波长 2π），``i`` 每加一，频率按 ``base^(2/d)`` 变快一点。
    ``d = 4``、``base = 10000`` 时两个周期恰好是 ``1`` 与 ``100``——本课所有手算样本
    都用这一组数（``√10000 = 100``，因此不需要计算器也能写出第二对的频率）。
    """
    checked_dimension = _checked_dimension(dimension)
    checked_base = _checked_base(base)
    if isinstance(pair, bool) or not isinstance(pair, int):
        raise ParameterError(f"pair 必须是整数，收到 {pair!r}。")
    if pair < 0 or pair >= checked_dimension // 2:
        raise ParameterError(
            f"频率对下标 {pair} 超出 [0, {checked_dimension // 2})："
            f"维数 {checked_dimension} 一共有 {checked_dimension // 2} 对频率。"
        )
    return checked_base ** (2.0 * pair / checked_dimension)


def sinusoidal_row(
    position: int,
    dimension: int,
    *,
    base: float = POSITIONAL_BASE,
    pairing: str = PAIRING_ALIGNED,
) -> Vector:
    """位置 ``position`` 的那一行：``aligned`` 用同一个频率配 ``sin``/``cos``."""
    if isinstance(position, bool) or not isinstance(position, int):
        raise ParameterError(f"position 必须是整数，收到 {position!r}。")
    if position < 0:
        raise ParameterError(f"position 必须 >= 0，收到 {position}（位置从 0 开始编号）。")
    checked_dimension = _checked_even_dimension(dimension, what="正弦表")
    checked_base = _checked_base(base)
    checked_pairing = _checked_pairing(pairing)
    row: list[float] = []
    if checked_pairing == PAIRING_ALIGNED:
        for pair in range(checked_dimension // 2):
            period = frequency_of(pair, checked_dimension, base=checked_base)
            angle = position / period
            row.append(math.sin(angle))
            row.append(math.cos(angle))
    else:
        # 错位：第 j 维用 base^(j/d)。偶数维取 sin、奇数维取 cos——
        # 于是第 2i 维与第 2i+1 维**不在同一个频率上**（day073 的写法）。
        for index in range(checked_dimension):
            angle = position / (checked_base ** (index / checked_dimension))
            row.append(math.sin(angle) if index % 2 == 0 else math.cos(angle))
    return tuple(row)


def sinusoidal_table(
    positions: int,
    dimension: int,
    *,
    base: float = POSITIONAL_BASE,
    pairing: str = PAIRING_ALIGNED,
) -> EncodingTable:
    """正弦/余弦位置编码表（**没有任何参数**，表长之外仍然有定义）.

    ```text
    aligned     PE(p, 2i) = sin(p / base^(2i/d))      PE(p, 2i+1) = cos(p / base^(2i/d))
    staggered   PE(p, j)  = sin / cos(p / base^(j/d))（j 为偶数取 sin）——day073 的写法
    ```

    两条性质（见 :mod:`smart_research_agent.positional_encoding.verify`）：

    ```text
    行范数恒定   ‖PE(p)‖ = √(d/2)（只有 aligned 成立：每一对贡献 sin²+cos² = 1）
    位移律       <PE(p), PE(q)> = Σ_i cos((p−q)/f_i)（只有 aligned 成立）
    ```
    """
    checked_positions = _checked_position_count(positions, name="positions")
    checked_dimension = _checked_even_dimension(dimension, what="正弦表")
    checked_base = _checked_base(base)
    checked_pairing = _checked_pairing(pairing)
    rows = tuple(
        sinusoidal_row(
            index, checked_dimension, base=checked_base, pairing=checked_pairing
        )
        for index in range(checked_positions)
    )
    periods = tuple(
        frequency_of(pair, checked_dimension, base=checked_base)
        for pair in range(checked_dimension // 2)
    )
    shown = "、".join(f"{value:g}" for value in periods[:4])
    if len(periods) > 4:
        shown += "、…"
    return EncodingTable(
        kind=ENCODING_SINUSOIDAL,
        table=rows,
        base=checked_base,
        pairing=checked_pairing,
        notes=(
            f"公式生成：{checked_positions} 行 × {checked_dimension} 维，base={checked_base:g}，"
            f"{checked_pairing}",
            f"频率对的周期 f_i = base^(2i/d)：{shown}",
            "表长之外仍然有定义——本表的 extends_to() 对任意非负位置都返回 True",
        ),
    )


def staggered_table(
    positions: int,
    dimension: int,
    *,
    base: float = POSITIONAL_BASE,
) -> EncodingTable:
    """**错位频率**的正弦表（day073 的写法），专门用来让两条性质同时失效.

    它不是“错误实现”，而是“另一种实现”——本课把它保留下来，因为它让
    “行范数恒定”与“位移律”这两条判据**有了被反向检验的机会**：
    一个只在正例上测过的性质，无法区分“实现对了”与“判据太松”。
    """
    return sinusoidal_table(positions, dimension, base=base, pairing=PAIRING_STAGGERED)


def _random_table(positions: int, dimension: int, *, scale: float, seed: int) -> Matrix:
    """确定性随机表（``[-scale, scale)``，用 day073 那串 LCG 随机数）."""
    if not math.isfinite(scale) or scale <= 0:
        raise ParameterError(f"scale 必须是正的有限数，收到 {scale!r}。")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ParameterError(f"seed 必须是整数，收到 {seed!r}。")
    raw = uniforms(positions * dimension, seed=seed)
    rows: list[Vector] = []
    cursor = 0
    for _ in range(positions):
        row = tuple((value * 2.0 - 1.0) * scale for value in raw[cursor : cursor + dimension])
        rows.append(row)
        cursor += dimension
    return tuple(rows)


def learnable_table(
    positions: int,
    dimension: int,
    *,
    initializer: str = INIT_SINUSOIDAL,
    base: float = POSITIONAL_BASE,
    scale: float = INIT_SCALE,
    seed: int = 7,
    values: Matrix | None = None,
) -> EncodingTable:
    """可学习的位置表：**每一行都是一个参数**.

    两种初始化（``initializer``）：

    ```text
    sinusoidal   从正弦表出发（实践中的默认做法）——训练从一个“已经知道怎么按位置读”的点开始
    random       从 [-scale, scale) 出发——位置知识必须全部从数据里学出来
    ```

    为什么把“从正弦表出发”作为缺省：它把两件事分开了——
    **“正弦表能不能被训练改进”**（一个优化问题）与
    **“位置知识能不能从零学出来”**（一个更难的优化问题）。
    本课的对照实验正是靠这个旋钮把两者分开的。

    ``values`` 非空时直接用它（形状必须对得上）——数值差分与梯度校验需要“换一张表”，
    而那时**不能**重新初始化（那就变成另一个点了）。
    """
    checked_positions = _checked_position_count(positions, name="positions")
    checked_dimension = _checked_dimension(dimension)
    if values is not None:
        checked_values = validate_matrix(values, name="values")
        if matrix_shape(checked_values) != (checked_positions, checked_dimension):
            raise ShapeError(
                f"给的表形状 {matrix_shape(checked_values)} 与要求的 "
                f"({checked_positions}, {checked_dimension}) 不一致。"
            )
        origin = "外部给定"
        table = checked_values
    elif initializer == INIT_SINUSOIDAL:
        table = sinusoidal_table(
            checked_positions, checked_dimension, base=base, pairing=PAIRING_ALIGNED
        ).table
        origin = f"从对齐正弦表初始化（base={_checked_base(base):g}）"
    elif initializer == INIT_RANDOM:
        table = _random_table(
            checked_positions, checked_dimension, scale=scale, seed=seed
        )
        origin = f"随机初始化（[-{scale}, {scale})，seed={seed}）"
    else:
        raise ParameterError(
            f"未知的初始化方式 {initializer!r}：可选 {', '.join(INITIALIZERS)}。"
        )
    return EncodingTable(
        kind=ENCODING_LEARNABLE,
        table=table,
        notes=(
            f"{origin}：{checked_positions} 行 × {checked_dimension} 维，"
            f"共 {checked_positions * checked_dimension} 个参数",
            "表长是一条硬边界：越界的位置必须拒绝（本表的 extends_to() 只覆盖表内下标）",
        ),
    )


def zero_table(positions: int, dimension: int) -> EncodingTable:
    """全零表：**注入退化成恒等映射**（最容易手算、也是“位置信息为零”的消融）.

    它被登记为 ``learnable``：全零不是“公式的结果”，而是一组恰好取 0 的参数。
    这一区分有用——“位置信息为零”与“没有位置编码”在数值上一样，
    但前者的梯度表**非零**（表会开始学），后者根本没有表可学。
    """
    checked_positions = _checked_position_count(positions, name="positions")
    checked_dimension = _checked_dimension(dimension)
    rows = tuple(tuple(0.0 for _ in range(checked_dimension)) for _ in range(checked_positions))
    return EncodingTable(
        kind=ENCODING_LEARNABLE,
        table=rows,
        notes=(
            "全零表：injected == inputs（注入是恒等映射）",
            "它的梯度表**非零**——表会开始学；而'不加位置编码'根本没有表可学",
        ),
    )


# ---------------------------------------------------------------------- 性质的两条闭式


def closed_form_offset_inner(
    dimension: int,
    offset: int,
    *,
    base: float = POSITIONAL_BASE,
) -> float:
    """位移律的**闭式**：``<PE(p), PE(p+δ)> = Σ_{i=0}^{d/2−1} cos(δ / base^(2i/d))``.

    推导只有一步——每一对频率的贡献是

    ```text
    sin(a)sin(b) + cos(a)cos(b) = cos(a − b)        a = p/f_i，b = q/f_i
    ⇒ 那一对贡献 cos((p − q)/f_i)
    ```

    因此和只依赖 ``δ = p − q``。它要求**偶数维**（奇数维最后一维没有配对）与
    **对齐频率**（错位时 ``a`` 与 ``b`` 的分母不同，那一步推不下去）。
    """
    checked_dimension = _checked_even_dimension(dimension, what="位移律的闭式")
    checked_base = _checked_base(base)
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise ParameterError(f"offset 必须是整数，收到 {offset!r}。")
    if offset < 0:
        raise ParameterError(
            f"offset 必须 >= 0，收到 {offset}：内积是对称的（<PE(p), PE(q)> = "
            "<PE(q), PE(p)>），因此只需要非负位移这一半。"
        )
    return math.fsum(
        math.cos(offset / frequency_of(pair, checked_dimension, base=checked_base))
        for pair in range(checked_dimension // 2)
    )


def rotation_factor(
    dimension: int,
    offset: int,
    *,
    base: float = POSITIONAL_BASE,
) -> Matrix:
    """把“位移 δ”写成一个**分块正交旋转矩阵** ``R_δ``：``PE(p + δ) = R_δ · PE(p)``.

    每一对频率一个 2×2 块：

    ```text
    [ cos(ωδ)  sin(ωδ) ]   [ sin(ωp) ]   [ sin(ω(p+δ)) ]
    [ −sin(ωδ) cos(ωδ) ] · [ cos(ωp) ] = [ cos(ω(p+δ)) ]        ω = 1/f_i
    ```

    这一条是位移律的**理由**而不是“又一条观察”：``R_δ`` 是正交的，
    因此它保持范数（行范数恒定）与内积（位移律）。
    两条性质于是有了同一个来源，而错位频率的版本**两条都塌**——
    因为那里根本没有一个统一的 ``ω`` 可以让上面那一行成立。
    """
    checked_dimension = _checked_even_dimension(dimension, what="旋转因子")
    checked_base = _checked_base(base)
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise ParameterError(f"offset 必须是整数，收到 {offset!r}。")
    zeros = [0.0] * checked_dimension
    rows: list[list[float]] = [list(zeros) for _ in range(checked_dimension)]
    for pair in range(checked_dimension // 2):
        period = frequency_of(pair, checked_dimension, base=checked_base)
        angle = offset / period
        cosine = math.cos(angle)
        sine = math.sin(angle)
        first = 2 * pair
        second = 2 * pair + 1
        rows[first][first] = cosine
        rows[first][second] = sine
        rows[second][first] = -sine
        rows[second][second] = cosine
    return tuple(tuple(row) for row in rows)


def rotate(table: EncodingTable, offset: int) -> Matrix:
    """把整张对齐正弦表按 ``offset`` 旋转（``PE(p + δ) = R_δ·PE(p)`` 的左边）.

    只有对齐频率的正弦表能这么算——可学习表没有 ``R_δ``（它的行是自由参数），
    错位表也没有（见 :func:`rotation_factor` 的说明）。
    """
    if not table.pairing_is_aligned:
        raise ParameterError(
            f"只有对齐频率的正弦表有旋转因子，这一张是 "
            f"{table.kind}/{table.pairing}：可学习表的行是自由参数、"
            "错位表的 sin 与 cos 不在同一频率上——两者都不存在一个统一的 ω。"
        )
    if table.base is None:  # pragma: no cover - 对齐正弦表必然带 base
        raise NumericError("对齐正弦表缺少频率基数。")
    _checked_dimension(table.dimension)
    if isinstance(offset, bool) or not isinstance(offset, int):
        raise ParameterError(f"offset 必须是整数，收到 {offset!r}。")
    if offset < 0:
        raise ParameterError(f"offset 必须 >= 0，收到 {offset}。")
    factor = rotation_factor(table.dimension, offset, base=table.base)
    return tuple(
        tuple(
            math.fsum(factor[row][column] * row_values[column] for column in range(len(row_values)))
            for row in range(len(factor))
        )
        for row_values in table.table
    )


# ---------------------------------------------------------------------- 注入与前向


def inject(
    inputs: Matrix,
    table: EncodingTable,
    *,
    positions: tuple[int, ...] | None = None,
) -> tuple[Matrix, Matrix]:
    """注入：返回 ``(取出的表行, inputs + 取出的表行)``.

    两个返回值都给出，而不是只给后者——“模型看到了什么”与“位置那部分是哪些数”
    必须同时可见，否则“位置编码到底贡献了多少”就只能靠反推。
    """
    checked_inputs = validate_matrix(inputs, name="inputs")
    rows, columns = matrix_shape(checked_inputs)
    resolved = (
        default_positions(rows)
        if positions is None
        else validate_positions(
            positions, expected=rows, table_positions=table.positions
        )
    )
    if columns != table.dimension:
        raise ShapeError(
            f"输入的列数 {columns} 与位置表的宽度 {table.dimension} 不一致："
            "注入是逐位相加，两边的行宽必须逐行对齐。"
        )
    gathered = gather_rows(table.table, resolved)
    return gathered, add_matrices(checked_inputs, gathered)


def _checked_supervised(rows: Sequence[int] | None, tokens: int) -> tuple[int, ...]:
    """校验监督行：非空、整数、不重复、在范围内（**空集抛 ``ParameterError``**）."""
    if rows is None:
        return tuple(range(tokens))
    resolved: list[int] = []
    seen: set[int] = set()
    for index, value in enumerate(rows):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ShapeError(f"supervised[{index}] 必须是整数，收到 {value!r}。")
        if value < 0 or value >= tokens:
            raise ShapeError(
                f"supervised[{index}] = {value} 超出 [0, {tokens})：监督行必须指向真实的行。"
            )
        if value in seen:
            raise ShapeError(
                f"supervised 里出现了重复的行 {value}：重复行会被损失算两次，"
                "于是那一行的权重悄悄变成两倍——而这在报告里只是一个'损失更小'。"
            )
        seen.add(value)
        resolved.append(value)
    if not resolved:
        raise ParameterError(
            "supervised 不能为空：既然给了 target，就至少要有一行参与损失——"
            "一行都不监督时'这次前向的损失'没有定义。"
        )
    return tuple(resolved)


def positional_forward(
    params: AttentionParams,
    table: EncodingTable,
    inputs: Matrix,
    *,
    positions: tuple[int, ...] | None = None,
    target: Matrix | None = None,
    supervised: Sequence[int] | None = None,
    causal: bool = False,
) -> PositionalForward:
    """注入位置 + 走 day075 那一层：六个阶段的完整前向.

    day075 那一层**一行都没有改**（本课也不改）：位置编码插在它**之前**，
    而“注意力原本不知道顺序”这件事因此变成了“它现在知道了”。

    ``causal=True`` 时本层的性质检查会**跳过**“位置编码打破了置换等变性”那一条——
    因为因果掩码本身就已经让那一层不等变了，“位置编码有没有起作用”在那里
    无法被单独验证（“跳过”与“通过”必须分开，这是 day075/076 同一条纪律）。
    """
    if not isinstance(params, AttentionParams):
        raise ParameterError(f"params 必须是 AttentionParams，收到 {type(params).__name__}。")
    if not isinstance(table, EncodingTable):
        raise ParameterError(f"table 必须是 EncodingTable，收到 {type(table).__name__}。")
    checked_inputs = validate_matrix(inputs, name="inputs")
    rows, columns = matrix_shape(checked_inputs)
    resolved_positions = (
        default_positions(rows)
        if positions is None
        else validate_positions(
            positions, expected=rows, table_positions=table.positions
        )
    )
    shape = PositionalShape(
        inputs=columns,
        positions=rows,
        dimension=table.dimension,
        outputs=params.shape.outputs,
    )
    if params.shape.inputs != table.dimension:
        raise ShapeError(
            f"注意力层的输入宽度 {params.shape.inputs} 与位置表宽度 "
            f"{table.dimension} 不一致：注入之后的每一行都要能被那层的四个投影吃掉。"
        )
    gathered, injected = inject(checked_inputs, table, positions=resolved_positions)
    attention = self_attention(params, injected, causal=causal)
    selected: tuple[int, ...] = ()
    loss: float | None = None
    if target is not None:
        checked_target = validate_matrix(target, name="target")
        if matrix_shape(checked_target) != matrix_shape(attention.output):
            raise ShapeError(
                f"目标形状 {matrix_shape(checked_target)} 与输出形状 "
                f"{matrix_shape(attention.output)} 不一致。"
            )
        selected = _checked_supervised(supervised, rows)
        if len(selected) == rows:
            loss = mean_squared_error(attention.output, checked_target)
        else:
            loss = masked_mean_squared_error(attention.output, checked_target, selected)
        target = checked_target
    notes = (
        f"注入：{table.kind} 表，位置 ({', '.join(str(value) for value in resolved_positions[:6])}"
        + ("…" if len(resolved_positions) > 6 else "")
        + ")",
        "注意力那一层是 day075 的实现，本课一行未改——位置编码只发生在它**之前**",
    )
    if causal:
        notes += ("因果掩码已启用：'位置编码打破了置换等变性'这一条性质将**跳过**",)
    return PositionalForward(
        shape=shape,
        table=table,
        positions=resolved_positions,
        inputs=checked_inputs,
        gathered=gathered,
        injected=injected,
        attention=attention,
        target=target,
        supervised=selected,
        loss=loss,
        notes=notes,
    )


# ---------------------------------------------------------------------- 反向


def loss_gradient(forward: PositionalForward) -> Matrix:
    """损失对**输出**的梯度（与 :meth:`PositionalForward.loss` 同口径）.

    两处必须同口径：day075 栽过一次（数值侧算全行 MSE、解析侧算监督行 MSE，
    差 15% 而两边都对）。这里把它做成**一个函数**：谁要损失梯度都得走它，
    于是“两边算的是不是同一个函数”这个问题在结构上被消掉了。
    """
    if forward.target is None:
        raise ParameterError(
            "这一份前向没有 target：没有目标就没有损失，也就没有损失梯度。"
        )
    output = forward.attention.output
    if not forward.supervised:
        return mse_gradient(output, forward.target)
    return masked_mse_gradient(output, forward.target, forward.supervised)


def positional_backward(
    forward: PositionalForward,
    grad_output: Matrix,
) -> PositionalGradients:
    """六步反向：**前四步与 day075 逐字相同**，只多了最后两步.

    ```text
    ⑥ dW_o = dOutᵀ·attention_out            与 day075 完全一致（位置编码不进这一项）
    ⑤ dV   = weightsᵀ·dInjected ；dW_v = dVᵀ·injected
    ④ dScores ← softmax 反向（逐行）；被掩码位置再显式置 0
    ③ dRaw = dScores·scale ；dQ = dRaw·K ；dK = dRawᵀ·Q
    ② dW_q = dQᵀ·injected ；dW_k = dKᵀ·injected ；dInjected = 三条链之和
    ① dx = dInjected（**逐位**）；dTable[p] = Σ_{i: positions[i]=p} dInjected[i]（**累加**）
    ```

    第 ① 步是这一课的全部：加法注入的偏导数恰好是 ``1``（不是“接近 1”、
    也不是“某个缩放系数”），因此 ``dx`` 与传给它的 ``dInjected`` **逐位相等**——
    而表那一份必须按位置**累加**。
    """
    checked_grad = validate_matrix(grad_output, name="grad_output")
    if matrix_shape(checked_grad) != matrix_shape(forward.attention.output):
        raise ShapeError(
            f"grad_output 的形状 {matrix_shape(checked_grad)} 与输出的 "
            f"{matrix_shape(forward.attention.output)} 不一致。"
        )
    inner = attention_backward(forward.attention, checked_grad)
    grad_table = scatter_add_rows(
        inner.grad_inputs,
        forward.positions,
        table_positions=forward.table.positions,
    )
    return PositionalGradients(
        grad_table=grad_table,
        grad_inputs=inner.grad_inputs,
        grad_w_query=inner.grad_w_query,
        grad_w_key=inner.grad_w_key,
        grad_w_value=inner.grad_w_value,
        grad_w_output=inner.grad_w_output,
    )


def positional_loss_from_flat(
    params: AttentionParams,
    template: EncodingTable,
    inputs: Matrix,
    *,
    positions: tuple[int, ...] | None = None,
    target: Matrix,
    supervised: Sequence[int] | None = None,
    causal: bool = False,
) -> Any:
    """返回一个 ``θ ↦ loss`` 的目标函数（**数值梯度与优化器共用这一条口径**）.

    ``θ`` 的形状是 ``flatten_parameters(params, template)`` 的那个向量，
    顺序 ``(w_query, w_key, w_value, w_output, table)``。
    两条路径共用它是刻意的：day076 的教训是“数值侧忘了传某个口径”——
    共用构造函数让“忘了传”在结构上不可能发生。
    """
    from smart_research_agent.positional_encoding.types import flatten_parameters

    _flat, shapes = flatten_parameters(params, template)

    def objective(theta: Vector) -> float:
        """给定压平后的参数，返回这一次前向的损失."""
        resolved_params, resolved_table = unflatten_flat(theta, shapes, template=template)
        forward = positional_forward(
            resolved_params,
            resolved_table,
            inputs,
            positions=positions,
            target=target,
            supervised=supervised,
            causal=causal,
        )
        if forward.loss is None:  # pragma: no cover - target 已给定，损失必然存在
            raise NumericError("损失没有被算出来：这一份前向的账不完整。")
        return forward.loss

    del _flat
    return objective


def unflatten_flat(
    theta: Vector,
    shapes: tuple[tuple[int, int], ...],
    *,
    template: EncodingTable,
) -> tuple[AttentionParams, EncodingTable]:
    """把压平的参数还原成“四个投影 + 一张表”（供 :func:`positional_loss_from_flat` 用）."""
    from smart_research_agent.positional_encoding.types import unflatten_parameters

    return unflatten_parameters(theta, shapes, template=template)


def batch_loss(
    params: AttentionParams,
    table: EncodingTable,
    sequences: Sequence[tuple[Matrix, Matrix]],
    *,
    causal: bool = False,
) -> float:
    """一批序列上的平均损失（**每条序列一份前向**）.

    为什么是“一批序列”而不是“一条长序列”：位置表是**按位置共享**的，
    而“共享”这件事只有在同一批里出现**重复位置**时才看得见。
    day076 的同参数量对照与本课的“位置表梯度按位置累加”都靠这一点。
    """
    if not sequences:
        raise ParameterError("至少要给一条序列：空批的损失没有定义。")
    total = 0.0
    for index, (inputs, target) in enumerate(sequences):
        forward = positional_forward(
            params, table, inputs, target=target, causal=causal
        )
        if forward.loss is None:  # pragma: no cover - target 已给定
            raise NumericError(f"第 {index} 条序列没有算出损失。")
        total += forward.loss
    return total / len(sequences)


def batch_accuracy(
    params: AttentionParams,
    table: EncodingTable,
    sequences: Sequence[tuple[Matrix, Matrix]],
    *,
    causal: bool = False,
) -> float:
    """一批序列上的平均命中率（监督行的 argmax 是否一致）."""
    if not sequences:
        raise ParameterError("至少要给一条序列：空批的命中率没有定义。")
    total = 0.0
    for inputs, target in sequences:
        forward = positional_forward(
            params, table, inputs, target=target, causal=causal
        )
        total += row_argmax_hits(
            forward.attention.output, target, list(range(forward.tokens))
        )
    return total / len(sequences)


def batch_mean_injection_ratio(
    params: AttentionParams,
    table: EncodingTable,
    sequences: Sequence[tuple[Matrix, Matrix]],
    *,
    causal: bool = False,
) -> float:
    """一批序列上“位置/内容”比的平均值（一个“位置占多少”的读数）."""
    if not sequences:
        raise ParameterError("至少要给一条序列：空批的读数没有定义。")
    ratios: list[float] = []
    for inputs, _target in sequences:
        forward = positional_forward(params, table, inputs, causal=causal)
        ratios.extend(
            value for value in forward.injection_ratios if math.isfinite(value)
        )
    if not ratios:
        raise NumericError("所有序列的所有输入行都是全零：这个读数没有定义。")
    return math.fsum(ratios) / len(ratios)


__all__ = [
    "POSITIONAL_BASE",
    "batch_accuracy",
    "batch_loss",
    "batch_mean_injection_ratio",
    "closed_form_offset_inner",
    "frequency_of",
    "inject",
    "learnable_table",
    "loss_gradient",
    "positional_backward",
    "positional_forward",
    "positional_loss_from_flat",
    "rotate",
    "rotation_factor",
    "sinusoidal_row",
    "sinusoidal_table",
    "staggered_table",
    "unflatten_flat",
    "zero_table",
]
