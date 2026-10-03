"""六个激活函数的**前向**实现（day089 / M8-D1）.

本模块只做一件事：把一个数（或一行 logits）放进一个激活里，取回一个数（或一行分布）。
**没有反向**——反向传播是 day090 的主题。

## 一、三个数值稳定点（每一个都对着一个可观察的坏结果）

```text
sigmoid    按符号分支：   x >= 0 ⇒ 1/(1+e^{-x})；  x < 0 ⇒ e^x/(1+e^x)
           为什么：单一式子 1/(1+e^{-x}) 在 x = -1000 时要算 e^{1000}——
           在 CPython 里 math.exp(1000) 直接抛 OverflowError（numpy / torch 里则是 inf，
           随后 1/inf = 0.0 或 inf/inf = nan）；按符号分支后两端都只算 e^{≤0}，不会上溢。

softmax    先减最大值：   e^{z_i - max} / Σ_j e^{z_j - max}
           为什么：这是**恒等变形**（分子分母同乘 e^{-max}），把 exp 的自变量压到 ≤ 0；
           不做它，z = 1000 会去算 e^{1000}——CPython 抛 OverflowError，
           numpy / torch 得到 inf、再 inf/inf = nan。求和用 ``math.fsum``，
           与 day073 的 ``math_foundations.linalg.softmax`` 逐位对齐。

gelu       采用**精确 erf 式**：0.5x(1 + erf(x/√2))
           为什么：它与项目里已存在的实现（``encoder_decoder.layers.activate`` 的 gelu、
           ``hf_source.blocks.gelu_exact``）是**同一个式子**，因此跨天对账是逐位的；
           tanh 近似（``gelu_new``）另列为 :func:`gelu_tanh`，不冒充精确式。
```

## 二、值域的检查用两套网格（这不是重复，是两种不同的问题）

```text
FINITE_GRID  含 ±1000  问"在这种极端点上，还**有限**吗"——挑数值稳定
RANGE_GRID   有界区间  问"值域被**遵守**吗"——sigmoid(-1000) 会饱和到 0.0，
                       那是**定义域代价**而不是 bug，因此严格的开区间断言只在有界网格上做
```

## 三、一个量只写一遍

``softmax`` 只在本模块实现一次；``losses.cross_entropy`` 会用
``log_softmax``（另一条稳定路径），但**不会**再写一个 softmax。
"""

from __future__ import annotations

import math
from typing import NamedTuple

from smart_research_agent.math_foundations.types import Vector
from smart_research_agent.neural_basics.errors import ActivationError, NumericError
from smart_research_agent.neural_basics.types import (
    ACT_GELU,
    ACT_LEAKY_RELU,
    ACT_RELU,
    ACT_SIGMOID,
    ACT_SOFTMAX,
    ACT_TANH,
    ACTIVATIONS,
    GELU_TANH_CUBIC,
    LEAKY_SLOPE,
)

#: ``gelu_new`` 里 ``tanh`` 的内系数 ``√(2/π)``（与 day085 的 ``hf_source.blocks`` 同源）.
GELU_TANH_INNER = math.sqrt(2.0 / math.pi)

#: 有限性网格（**写死**）：含 ``x = ±1000``，用来逼出 exp 上溢。
FINITE_GRID: tuple[float, ...] = (-1000.0, -100.0, -10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 10.0, 100.0, 1000.0)

#: 值域网格（**写死且有个有限的上界**）：严格的开区间断言只在这里做。
RANGE_GRID: tuple[float, ...] = (-10.0, -5.0, -1.0, -0.5, 0.0, 0.5, 1.0, 5.0, 10.0)


class ActivationRange(NamedTuple):
    """一个激活的值域（**含端点开闭**——``sigmoid`` 的开区间与 ``relu`` 的闭区间不是一回事）."""

    lower: float
    upper: float
    lower_open: bool = False
    upper_open: bool = False

    def contains(self, value: float) -> bool:
        """是否落在值域内（``±inf`` 边界视为无界；``nan`` 一律拒绝）."""
        if math.isnan(value):
            return False
        if math.isinf(self.lower):
            lower_ok = True
        elif self.lower_open:
            lower_ok = value > self.lower
        else:
            lower_ok = value >= self.lower
        if math.isinf(self.upper):
            upper_ok = True
        elif self.upper_open:
            upper_ok = value < self.upper
        else:
            upper_ok = value <= self.upper
        return lower_ok and upper_ok

    def line(self) -> str:
        """一行说明：``[0, +inf)`` 或 ``(0, 1)``."""
        left = "(" if self.lower_open else "["
        right = ")" if self.upper_open else "]"
        low = "-inf" if self.lower == -math.inf else f"{self.lower:g}"
        high = "+inf" if self.upper == math.inf else f"{self.upper:g}"
        return f"{left}{low}, {high}{right}"


#: 六个激活的值域表（**只对可严格断言的四个写开区间**；gelu 的下界是近似值，故写无界）.
ACTIVATION_RANGES: dict[str, ActivationRange] = {
    ACT_RELU: ActivationRange(0.0, math.inf, lower_open=False),
    ACT_LEAKY_RELU: ActivationRange(-math.inf, math.inf),
    ACT_SIGMOID: ActivationRange(0.0, 1.0, lower_open=True, upper_open=True),
    ACT_TANH: ActivationRange(-1.0, 1.0, lower_open=True, upper_open=True),
    ACT_GELU: ActivationRange(-math.inf, math.inf),
    ACT_SOFTMAX: ActivationRange(0.0, 1.0, lower_open=True, upper_open=True),
}


def activation_range(name: str) -> ActivationRange:
    """按名字取值域（未知名字抛 :class:`ActivationError`）."""
    if name not in ACTIVATION_RANGES:
        raise ActivationError(f"未知的激活 {name!r}：可选 {list(ACTIVATIONS)}。")
    return ACTIVATION_RANGES[name]


# --------------------------------------------------------------------------------------
# 1. 标量激活（逐元素）
# --------------------------------------------------------------------------------------


def relu(value: float) -> float:
    """``max(0, x)``——负半轴恒为 0，因此它是一个**分段线性**算子."""
    return value if value > 0.0 else 0.0


def leaky_relu(value: float, slope: float = LEAKY_SLOPE) -> float:
    """``x if x > 0 else slope·x``——给负半轴留一条小坡度（默认 0.01）."""
    if not math.isfinite(slope) or slope < 0.0:
        raise NumericError(f"leaky_relu 的斜率必须是非负有限数，收到 {slope!r}。")
    return value if value > 0.0 else slope * value


def sigmoid(value: float) -> float:
    """``1/(1+e^{-x})``——**按符号分支**两条式子，两端都不上溢.

    ```text
    x >= 0 ⇒ 1 / (1 + e^{-x})      （e^{-x} ≤ 1，不会上溢）
    x <  0 ⇒ e^x / (1 + e^x)       （e^x  < 1，不会上溢）
    ```
    """
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def tanh(value: float) -> float:
    """``(e^x - e^{-x})/(e^x + e^{-x})``——标准库实现已经是稳定式."""
    return math.tanh(value)


def gelu(value: float) -> float:
    """**精确** GELU：``0.5x(1 + erf(x/√2))``（与 ``encoder_decoder`` / ``hf_source`` 同一个式子）."""
    return 0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0)))


def gelu_tanh(value: float) -> float:
    """``gelu_new`` 的 tanh 近似：``0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))``.

    它**不是** :func:`gelu` 的等价写法：``x = 0`` 处两式的函数值同为 0、导数同为 0.5，
    而两侧各有 O(1e-4) 量级的偏差（day085 把它量过）。本课的对账用精确式。
    """
    inner = GELU_TANH_INNER * (value + GELU_TANH_CUBIC * value**3)
    return 0.5 * value * (1.0 + math.tanh(inner))


#: 五个**逐元素**激活的调度表（``softmax`` 不是逐元素，单独处理）.
_ELEMENTWISE: dict[str, object] = {
    ACT_RELU: relu,
    ACT_LEAKY_RELU: leaky_relu,
    ACT_SIGMOID: sigmoid,
    ACT_TANH: tanh,
    ACT_GELU: gelu,
}


# --------------------------------------------------------------------------------------
# 2. 行激活：softmax（唯一的非逐元素激活）
# --------------------------------------------------------------------------------------


def softmax(vector: Vector) -> Vector:
    """数值稳定的 softmax（**先减最大值**，求和用 ``math.fsum``）.

    ```text
    softmax(z)_i = e^{z_i - max z} / Σ_j e^{z_j - max z}
    ```

    减去最大值不改变结果（分子分母同乘 ``e^{-max}``），但把 ``exp`` 的自变量压到
    ``≤ 0``，从而不会上溢。求和用 ``math.fsum``：**刻意**与 day073 的
    ``math_foundations.linalg.softmax`` 用同一种求和顺序，于是跨天对账可以逐位比较。
    空向量抛 :class:`ActivationError`——"对空 logits 求分布"没有定义。
    """
    values = tuple(float(value) for value in vector)
    if not values:
        raise ActivationError("softmax 的输入不能为空：对空 logits 求分布没有定义。")
    if any(not math.isfinite(value) for value in values):
        raise ActivationError("softmax 的输入必须是有限数：出现 nan / inf 时分布没有定义。")
    peak = max(values)
    exponentials = [math.exp(value - peak) for value in values]
    total = math.fsum(exponentials)
    return tuple(value / total for value in exponentials)


# --------------------------------------------------------------------------------------
# 3. 统一入口
# --------------------------------------------------------------------------------------


def activate(name: str, vector: Vector) -> Vector:
    """按名字把一个向量过一遍激活（未知名字**当场拒绝**，绝不回退到 relu）.

    逐元素激活对向量里每个数各算一次；``softmax`` 把整个向量当作一行 logits。
    返回类型统一是 ``Vector``（``softmax`` 的"一行和为 1"因此可以被直接断言）。
    """
    if name not in ACTIVATIONS:
        raise ActivationError(
            f"未知的激活 {name!r}：可选 {list(ACTIVATIONS)}。"
            "回退到 relu 的后果是——一份 sigmoid 的读数会被印成 relu 的，"
            "而'我确实选了 sigmoid'与'它被偷偷改成了 relu'在读表时长得一样。"
        )
    values = tuple(float(value) for value in vector)
    if not values:
        raise ActivationError(f"激活 {name!r} 的输入不能为空。")
    if any(not math.isfinite(value) for value in values):
        raise ActivationError(f"激活 {name!r} 的输入必须是有限数（收到 nan / inf）。")
    if name == ACT_SOFTMAX:
        return softmax(values)
    function = _ELEMENTWISE[name]
    return tuple(function(value) for value in values)  # type: ignore[operator]


def activation_at(name: str, value: float) -> float:
    """标量便捷入口：把一个数过一遍逐元素激活（``softmax`` 需要一整行，故在此拒绝）."""
    if name == ACT_SOFTMAX:
        raise ActivationError(
            "softmax 不是逐元素激活：它需要一整行 logits，请用 activate(softmax, 行)。"
            "把一个标量过 softmax 会静默得到 1.0，而它与'这一行只有一项'读起来一样。"
        )
    return activate(name, (value,))[0]


__all__ = [
    "ACTIVATION_RANGES",
    "ActivationRange",
    "FINITE_GRID",
    "GELU_TANH_INNER",
    "RANGE_GRID",
    "activate",
    "activation_at",
    "activation_range",
    "gelu",
    "gelu_tanh",
    "leaky_relu",
    "relu",
    "sigmoid",
    "softmax",
    "tanh",
]
