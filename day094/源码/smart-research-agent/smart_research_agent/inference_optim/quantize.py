"""``quantize``：把 32 位压成 8 位 / 4 位（day087 / M7-D11）.

量化只有两件事，而它们**必须分开说**：

```text
① 定标     找一个 scale（对称）或 scale + zero（非对称），把浮点映射到整数格
② 取整     q = round(x / scale) —— **误差只发生在这一步**
```

打包（int4 两个数挤进一个字节）**不是**第三件事：它是无损的位运算，
一位都不丢。把"打包"和"取整"混在一起说，就会出现"4 位量化精度是 8 位的一半"
这种话——而真正决定精度的是**格数**（8 位 256 格、4 位 16 格）。

## 一条可推导的上界（本课最值钱的一条）

对称量化里 `scale = max|x| / (2^(b-1) − 1)`，而 `round` 的误差不超过 `0.5` 格，
因此

```text
|x − x̂| = |x − round(x/scale)·scale| ≤ 0.5 · scale = scale/2
```

它不是经验，是推导。它有**两条前提**，而两条都要当场检查：

```text
前提 ①  每个元素都落在 [-max_abs, max_abs] 内   ⇒ 出现 inf/nan 时前提就破了
前提 ②  scale ≠ 0                             ⇒ 整块同值时 scale = 0（本包当场拒绝）
```

## 另一条要写下来的：粒度比位数更值钱

```text
per_tensor   整块共用一个 scale ⇒ **一个离群值抬高所有人的台阶**
per_channel  每一行一个 scale   ⇒ 离群值只抬高它自己那一行（代价是每行存一个 scale）
```

本课把两种粒度都实现出来，因为"收益"这件事在这里是可以被量出来的：
同一份权重、同一个位数、只换粒度，误差通常会差出一个量级。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from smart_research_agent.inference_optim.errors import (
    NumericError,
    QuantError,
    ShapeError,
)
from smart_research_agent.inference_optim.types import (
    BITS_PER_BYTE,
    GRANULARITY_CHANNEL,
    GRANULARITY_TENSOR,
    INT4_PER_BYTE,
    LEVEL_INT4,
    SCHEME_ABSMAX,
    SCHEME_ZERO_POINT,
    ErrorStats,
    QuantSpec,
    QuantizedMatrix,
    flatten,
)
from smart_research_agent.math_foundations.types import Matrix, Vector

#: 量化误差上界的容差（浮点求和的有理误差量级；它是"上界"不是"近似"）。
BOUND_TOLERANCE = 1e-12


def logits_of(card: object, weights: object, row: Vector) -> Vector:
    """把一行 hidden state 投回词表空间（**缓存路径的最后一步**）.

    它与 day086 的 ``forward.logits`` 的算术逐字相同：取末位那一行、与 ``head``
    做点积。放在本模块里的原因是"哪一步做什么"这一层的分工——
    ``cache`` 负责"哪一行"，``quantize`` 负责"这一行怎么变成一个向量"。
    """
    from smart_research_agent.hf_integration.types import ModelCard
    from smart_research_agent.hf_integration.forward import ModelWeights
    from smart_research_agent.transformer_core.types import project

    if not isinstance(card, ModelCard) or not isinstance(weights, ModelWeights):
        raise ShapeError(  # pragma: no cover - 类型守卫
            "logits_of 需要（卡片, 权重, 一行 hidden）：收到 "
            f"{type(card).__name__} / {type(weights).__name__}。"
        )
    if len(row) != card.hidden:
        raise ShapeError(
            f"这一行的宽度 {len(row)} 与 hidden {card.hidden} 不一致："
            "宽度不对时点积会静默截断（少乘几列），而结果看起来是一串正常的数。"
        )
    return project((tuple(row),), weights.head)[0]


# ---------------------------------------------------------------------------
# 定标
# ---------------------------------------------------------------------------


def _max_abs(values: Vector) -> float:
    return max((abs(value) for value in values), default=0.0)


def _row_scales(values: Vector, rows: int, columns: int, *, scheme: str, width: int) -> tuple[float, ...]:
    """per-channel 的每一行一个 scale（对称与非对称各一条式子）."""
    scales: list[float] = []
    for index in range(rows):
        chunk = values[index * columns : (index + 1) * columns]
        scales.append(_scale_of(chunk, scheme=scheme, width=width))
    return tuple(scales)


def _scale_of(values: Vector, *, scheme: str, width: int) -> float:
    """一块元素上的 scale（**两条式子只在这里写一遍**）."""
    if scheme == SCHEME_ABSMAX:
        top = _max_abs(values)
        denominator = float(2 ** (width - 1) - 1)
    else:
        if not values:
            raise QuantError("空块没有动态范围可以定标。")
        top = max(values) - min(values)
        denominator = float(2**width - 1)
    if top == 0.0:
        raise QuantError(
            "这一块的所有元素都是同一个值（动态范围为 0）⇒ scale = 0。"
            "反量化会把整块还原成 0，而'误差很大'与'这一层本来就没东西'读起来不一样——"
            "因此本包在定标这一步就拒绝，而不是让它以'效果变差'的形式出现在下游。"
        )
    if not math.isfinite(top):
        raise NumericError(
            f"这一块里有非有限数（最大/最小是 {top!r}）："
            "量化误差的上界 ``scale/2`` 依赖'每个元素都落在量程内'这条前提，"
            "而 inf 会让前提直接失效。"
        )
    return top / denominator


def _quantize_values(
    values: Vector,
    *,
    spec: QuantSpec,
    scales: tuple[float, ...],
    columns: int,
    zeros: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    """把元素逐个映射到整数格（**取整只在这一处发生**）.

    非对称那一支的写法是"先平移再定标"：``q = round(x/scale) + zero``，
    然后**把格号夹在 ``[0, 2^b-1]`` 内**。夹哪一个很重要：
    夹的是"格号"（``xˆ = (q - zero)·scale`` 里的那个 ``q``），而不是"格号减去零点"——
    后者在区间不跨 0 的时候会整体错位一个零点（形状全对、数值全偏）。
    """
    top = spec.max_quantized
    floor = -top - 1 if spec.scheme == SCHEME_ABSMAX else 0
    out: list[int] = []
    for index, value in enumerate(values):
        scale = scales[0] if len(scales) == 1 else scales[index // columns]
        zero = 0 if zeros is None else (zeros[0] if len(zeros) == 1 else zeros[index // columns])
        quantized = int(round(value / scale)) + zero
        clamped = 0
        if quantized < floor:
            clamped = floor
        elif quantized > top:
            clamped = top
        else:
            clamped = quantized
        out.append(clamped)
    return tuple(out)


def quantize_matrix(matrix: Matrix, spec: QuantSpec) -> QuantizedMatrix:
    """把一块矩阵量化成整数格（**两种方案 × 两种粒度都在这一条路径上**）.

    per-channel 的"通道"就是**行**：这与 :func:`types.flatten` 的逐行顺序一致，
    因此粒度这件事只改"用几个 scale"，不改数据的排布。

    > 名字里带 ``_matrix`` 是刻意的：包入口再导出一个叫 ``quantize`` 的函数，
    > 会把子模块 ``inference_optim.quantize`` **同名遮掉**——
    > 于是 ``from ...inference_optim import quantize`` 拿到的是函数而不是模块，
    > 而 ``quantize.measure(...)`` 会以一个很难看懂的方式失败。
    """
    rows = len(matrix)
    if rows == 0:
        raise ShapeError("空矩阵没有东西可以量化。")
    columns = len(matrix[0])
    for index, row in enumerate(matrix):
        if len(row) != columns:
            raise ShapeError(f"第 {index} 行的列数与本组不一致（zip 会静默截断）。")
    values = flatten(matrix)
    for value in values:
        if not math.isfinite(value):
            raise NumericError(
                f"矩阵里有非有限数 {value!r}：量化会把它变成一个巨大而有限的整数，"
                "于是'这一块坏了'会以'这一块误差很大'的形式出现。"
            )
    if spec.granularity == GRANULARITY_TENSOR:
        scales = (_scale_of(values, scheme=spec.scheme, width=spec.bits),)
    elif spec.granularity == GRANULARITY_CHANNEL:
        scales = _row_scales(values, rows, columns, scheme=spec.scheme, width=spec.bits)
    else:  # pragma: no cover - QuantSpec 已经拦过
        raise QuantError(f"未知粒度 {spec.granularity!r}。")
    zeros: tuple[int, ...] = ()
    if spec.scheme == SCHEME_ZERO_POINT:
        # ``zero`` 是"0 对应哪个格号"，它**可以是负的**（当整个区间都在 0 的同一侧时）——
        # 那不需要修正，因为 0 本来就不在这个动态范围里。修正它反而会把整块平移。
        zeros = tuple(
            int(round(-_min_of(values, index, columns, len(scales)) / scale))
            for index, scale in enumerate(scales)
        )
    quantized = _quantize_values(
        values, spec=spec, scales=scales, columns=columns, zeros=zeros or None
    )
    return QuantizedMatrix(
        values=quantized,
        rows=rows,
        columns=columns,
        scale=scales[0] if len(scales) == 1 else max(scales),
        zero=zeros[0] if zeros else 0,
        scheme=spec.scheme,
        granularity=spec.granularity,
        scales=scales,
        zeros=zeros,
        bits=spec.bits,
    )


def _min_of(values: Vector, index: int, columns: int, scale_count: int) -> float:
    """某一"通道"上的最小元素（通道数 = 1 时就是整块）."""
    if scale_count == 1:
        return min(values)
    return min(values[index * columns : (index + 1) * columns])


def dequantize(quantized: QuantizedMatrix) -> Matrix:
    """把整数格还原成浮点（**就是"格号 × 格宽"**，非对称再减回 zero）."""
    if quantized.count != quantized.rows * quantized.columns:
        raise ShapeError(
            f"整数值有 {quantized.count} 个，而形状是 {quantized.rows}×{quantized.columns}："
            "两者对不上时'按行还原'会把某一行接到别人的 scale 上。"
        )
    scales = quantized.scales or (
        (quantized.scale,) if quantized.granularity == GRANULARITY_TENSOR else ()
    )
    if not scales:
        raise QuantError(
            "per_channel 量化缺了逐行的表（``scales`` 为空）："
            "没有它只能整块共用一个 scale，而那正是另一种粒度。"
        )
    if quantized.granularity == GRANULARITY_CHANNEL and len(scales) != quantized.rows:
        raise QuantError(
            f"per_channel 的 scale 有 {len(scales)} 个、行数 {quantized.rows}："
            "数量不匹配时某些行会用到别人的格宽（形状完全合法）。"
        )
    rows: list[tuple[float, ...]] = []
    for row_index in range(quantized.rows):
        start = row_index * quantized.columns
        chunk = quantized.values[start : start + quantized.columns]
        scale = scales[0] if len(scales) == 1 else scales[row_index]
        zero = 0
        if quantized.zeros:
            zero = quantized.zeros[0] if len(quantized.zeros) == 1 else quantized.zeros[row_index]
        restored = tuple((value - zero) * scale for value in chunk)
        for value in restored:
            if not math.isfinite(value):
                raise NumericError(
                    f"反量化后出现了非有限值 {value!r}："
                    "它通常来自一次 scale 与格号的错配（数值全坏、形状全对）。"
                )
        rows.append(restored)
    return tuple(rows)


# ---------------------------------------------------------------------------
# 打包与解包（**无损的位运算**）
# ---------------------------------------------------------------------------


def pack_int4(values: Vector | tuple[int, ...], *, signed: bool = True) -> tuple[int, ...]:
    """把一串 4 位数两个一组装进字节（**顺序固定：先低 4 位、再高 4 位**）.

    负数的处理是这一族最容易错的地方：4 位里没有符号位，
    因此有符号值要先补码到无符号（``v & 0xF``），解包时再按同一个约定还原。
    奇数个元素时最后半个字节补 0——**它不是"多出来的一个数"**，
    而是"那个字节的高 4 位没有意义"，因此解包时要传回元素个数。
    """
    nibbles = tuple(_to_nibble(value, signed=signed) for value in values)
    out: list[int] = []
    for index in range(0, len(nibbles), INT4_PER_BYTE):
        low = nibbles[index]
        high = nibbles[index + 1] if index + 1 < len(nibbles) else 0
        out.append(low | (high << 4))
    return tuple(out)


def unpack_int4(
    packed: tuple[int, ...] | list[int],
    count: int,
    *,
    signed: bool = True,
) -> tuple[int, ...]:
    """把字节解回 4 位数（``count`` 决定要几个——它必须与打包时一致）."""
    if count < 0:
        raise QuantError(f"要解的元素个数不能为负，收到 {count}。")
    needed = math.ceil(count / INT4_PER_BYTE)
    if len(packed) != needed:
        raise QuantError(
            f"打包数据有 {len(packed)} 个字节、而 {count} 个元素需要 {needed} 个："
            "长度不对时解包会多出或少掉半个字节，而结果看起来是一串合法的 4 位数。"
        )
    nibbles: list[int] = []
    for byte in packed:
        if not 0 <= byte < 2**BITS_PER_BYTE:
            raise QuantError(f"打包字节必须在 [0, 255] 内，收到 {byte}。")
        nibbles.append(byte & 0xF)
        nibbles.append((byte >> 4) & 0xF)
    return tuple(_from_nibble(value, signed=signed) for value in nibbles[:count])


def _to_nibble(value: int, *, signed: bool) -> int:
    """把一个 4 位整数写成一个无符号的 4 位码（有符号时补码）."""
    if signed:
        if not -2**3 <= value <= 2**3 - 1:
            raise QuantError(
                f"有符号 4 位只能表示 [-8, 7]，收到 {value}："
                "越界的值补码之后会串到相邻的那个数上（形状与长度全对）。"
            )
        return value & 0xF
    if not 0 <= value <= 2**4 - 1:
        raise QuantError(f"无符号 4 位只能表示 [0, 15]，收到 {value}。")
    return value


def _from_nibble(code: int, *, signed: bool) -> int:
    """把无符号 4 位码还原成整数（有符号时判最高位）."""
    if not signed:
        return code
    return code - 2**4 if code >= 2**3 else code


def packed_length(count: int) -> int:
    """``count`` 个 4 位数要几个字节（**向上取整**，因此奇数个会多半个字节）."""
    if count < 0:
        raise QuantError(f"元素个数不能为负，收到 {count}。")
    return math.ceil(count / INT4_PER_BYTE)


# ---------------------------------------------------------------------------
# 误差与上界
# ---------------------------------------------------------------------------


def error_stats(
    original: Matrix,
    restored: Matrix,
    *,
    level: str,
    scheme: str,
    granularity: str,
) -> ErrorStats:
    """四个误差读数：最大、均值、SNR（dB）、元素个数.

    SNR 用**信号功率 / 噪声功率**的对数（``10·log10``）——
    它与"误差有多大"是同一件事的两种写法，但它在跨层比较时更稳
    （因为不同层的权重量级可能差几个数量级）。
    """
    if len(original) != len(restored):
        raise ShapeError(f"行数不同：{len(original)} 与 {len(restored)}。")
    signal = 0.0
    noise = 0.0
    worst = 0.0
    count = 0
    for row_a, row_b in zip(original, restored, strict=True):
        if len(row_a) != len(row_b):
            raise ShapeError("列数不同（zip 会静默截断）。")
        for a, b in zip(row_a, row_b, strict=True):
            signal += a * a
            noise += (a - b) * (a - b)
            worst = max(worst, abs(a - b))
            count += 1
    mean = 0.0 if count == 0 else math.sqrt(noise / count)
    if noise == 0.0:
        snr = math.inf
    elif signal == 0.0:
        snr = -math.inf
    else:
        snr = 10.0 * math.log10(signal / noise)
    return ErrorStats(
        level=level,
        scheme=scheme,
        granularity=granularity,
        max_abs_error=worst,
        mean_abs_error=mean,
        snr_db=snr,
        elements=count,
    )


def error_bound(scale: float) -> float:
    """误差上界 ``scale/2``（**一条式子，一处实现**）."""
    if not math.isfinite(scale) or scale < 0:
        raise NumericError(f"scale 必须是有限非负数，收到 {scale!r}。")
    return scale / 2.0


def measure(matrix: Matrix, spec: QuantSpec) -> tuple[QuantizedMatrix, ErrorStats]:
    """量化 → 反量化 → 统计（**三步一条路径**，因此读数不可能对不上**）."""
    quantized = quantize_matrix(matrix, spec)
    restored = dequantize(quantized)
    return quantized, error_stats(
        matrix,
        restored,
        level=spec.level,
        scheme=spec.scheme,
        granularity=spec.granularity,
    )


def bound_line(matrix: Matrix, spec: QuantSpec) -> str:
    """一行读数：误差上下界与实测最大值（**上界必须被满足**）."""
    quantized, stats = measure(matrix, spec)
    bound = error_bound(quantized.scale)
    return (
        f"{spec.line()} | 上界 scale/2 = {bound:.3e} | 实测最大误差 "
        f"{stats.max_abs_error:.3e} | 满足 {stats.max_abs_error <= bound + BOUND_TOLERANCE}"
    )


def round_trip_line(values: tuple[int, ...], *, signed: bool = True) -> str:
    """一行读数：``int4`` 打包与解包的往返（**逐位**）."""
    packed = pack_int4(values, signed=signed)
    restored = unpack_int4(packed, len(values), signed=signed)
    return (
        f"{len(values)} 个 4 位数 → {len(packed)} 字节（每字节 {INT4_PER_BYTE} 个）"
        f" | 逐位还原 {restored == tuple(values)} | 前 4 个字节 {list(packed[:4])}"
    )


@dataclass(frozen=True)
class LayerError:
    """一层的误差读数（与层号绑在一起，因此"哪一层更敏感"可读）."""

    layer: int
    target: str
    stats: ErrorStats

    def line(self) -> str:
        """一行说明：``层 0 的 w_query：最大误差 1.9e-03（SNR 52.4 dB）``."""
        return f"层 {self.layer} 的 {self.target}：{self.stats.line()}"


__all__ = [
    "BOUND_TOLERANCE",
    "LayerError",
    "bound_line",
    "dequantize",
    "error_bound",
    "error_stats",
    "logits_of",
    "measure",
    "pack_int4",
    "packed_length",
    "quantize_matrix",
    "round_trip_line",
    "unpack_int4",
]
