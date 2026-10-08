"""一层全连接与它的初始化（day089 / M8-D1）.

## 一、``Dense`` 层的口径：与 ``torch.nn.Linear`` 逐一对应

```text
权重形状    weight  (out_features, in_features)   ← 与 nn.Linear 一致
偏置形状    bias    (out_features,)               ← nn.Linear(bias=True) 的默认
前向        y = x · Wᵀ + b                         ← 对每一行 x 都成立
参数量      in_features × out_features + out_features
```

``x`` 的一行长度必须等于 ``in_features``；输出一行长度等于 ``out_features``。
本模块**只做前向**：没有 ``backward``，也没有 ``torch.autograd``。

## 二、初始化是确定性的：LCG + 可注入种子

```text
state ← (1103515245 · state + 12345) mod 2^31        （glibc 的 LCG 常数）
u     ← state / 2^31                                  （落在 [0, 1)）
```

种子注入到每一条 ``DenseSpec``（``seed`` 字段），因此"同一份 spec 得到同一组权重"
是一条可复现的事实：两次 ``dense_forward`` 必须**逐位相同**。正态初始化用 Box-Muller
从同一串 LCG 造两个独立均匀数，再取一对正态数。

## 三、四种初始化（外加可选的 He）

```text
zeros     W = 0、b = 0                          对称性不破 ⇒ 每个神经元学到的东西完全相同
uniform   U(−1/√fan_in, 1/√fan_in)              尺度只看 fan_in（与 nn.Linear 的默认同阶）
normal    N(0, 1/√fan_in)                       Box-Muller（可复现）
xavier    U(−a, a)，a = √(6/(fan_in+fan_out))   分母为 0 时抛 InitializationError
he        N(0, √(2/fan_in))                     为 ReLU 一族设计的下游尺度
```

## 四、仿射合成：多层塌缩成一层的**唯一**实现处

:func:`compose_affine` 把一串 ``(W, b)`` 折成一个 ``(W, b)``：

```text
x·W₁ᵀ + b₁  再  ·W₂ᵀ + b₂   =   x·(W₂·W₁)ᵀ + (b₁·W₂ᵀ + b₂)
```

它没有用到任何非线性——这正是第 4 条性质要断言的那件事。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from smart_research_agent.math_foundations.types import Matrix, Vector
from smart_research_agent.neural_basics.activations import activate
from smart_research_agent.neural_basics.errors import (
    InitializationError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.neural_basics.types import (
    INITIALIZATIONS,
    INIT_HE,
    INIT_NORMAL,
    INIT_UNIFORM,
    INIT_XAVIER,
    INIT_ZEROS,
    DenseSpec,
)

#: glibc 的线性同余常数（写进常量：换一个常数，所有读数都会变）。
LCG_MODULUS = 2**31
LCG_MULTIPLIER = 1103515245
LCG_INCREMENT = 12345


def lcg_stream(seed: int, count: int) -> tuple[float, ...]:
    """一串确定性均匀数（``[0, 1)``，长度 ``count``；``count < 0`` 抛 ``ParameterError``）."""
    if count < 0:
        raise ParameterError(f"lcg_stream 的长度必须非负，收到 {count}。")
    state = seed % LCG_MODULUS
    values: list[float] = []
    for _ in range(count):
        state = (LCG_MULTIPLIER * state + LCG_INCREMENT) % LCG_MODULUS
        values.append(state / LCG_MODULUS)
    return tuple(values)


def _normal_pair(first: float, second: float) -> tuple[float, float]:
    """Box-Muller：两个均匀数 → 两个标准正态数（``first`` 被夹到正数以避免 log(0)）."""
    positive = first if first > 0.0 else 1e-12
    radius = math.sqrt(-2.0 * math.log(positive))
    angle = 2.0 * math.pi * second
    return radius * math.cos(angle), radius * math.sin(angle)


def xavier_limit(fan_in: int, fan_out: int) -> float:
    """Xavier 的均匀半宽 ``√(6/(fan_in + fan_out))``（分母为 0 抛 ``InitializationError``）."""
    fan_sum = fan_in + fan_out
    if fan_sum <= 0:
        raise InitializationError(
            f"xavier 的分母 fan_in + fan_out = {fan_sum}，尺度没有定义："
            "先决定这一层的宽度，再谈初始化。"
        )
    return math.sqrt(6.0 / fan_sum)


def he_scale(fan_in: int) -> float:
    """He 正态标准差 ``√(2/fan_in)``（``fan_in <= 0`` 抛 ``InitializationError``）."""
    if fan_in <= 0:
        raise InitializationError(f"he 的 fan_in = {fan_in}，标准差没有定义。")
    return math.sqrt(2.0 / fan_in)


def initialize(
    init: str, out_features: int, in_features: int, *, seed: int = 0
) -> tuple[Matrix, Vector]:
    """按名字造一组 ``(weight, bias)``（**全部确定性**）.

    ``weight`` 形状 ``(out_features, in_features)``、``bias`` 形状 ``(out_features,)``。
    未知初始化名抛 ``ParameterError``；尺度非法（分母为 0）抛 ``InitializationError``。
    """
    if init not in INITIALIZATIONS:
        raise ParameterError(f"未知的初始化 {init!r}：可选 {list(INITIALIZATIONS)}。")
    if in_features < 1 or out_features < 1:
        raise InitializationError(
            f"初始化的两侧宽度都必须为正，收到 fan_in={in_features}、fan_out={out_features}。"
        )
    if init == INIT_ZEROS:
        weight = tuple(tuple(0.0 for _ in range(in_features)) for _ in range(out_features))
        return weight, tuple(0.0 for _ in range(out_features))

    total = out_features * in_features
    uniform = lcg_stream(seed, total)
    bias_uniform = lcg_stream(seed + 1, out_features)

    if init == INIT_UNIFORM:
        limit = 1.0 / math.sqrt(in_features)
        weight = tuple(
            tuple(uniform[row * in_features + column] * 2.0 * limit - limit for column in range(in_features))
            for row in range(out_features)
        )
        bias = tuple(value * 2.0 * limit - limit for value in bias_uniform)
        return weight, bias

    if init == INIT_XAVIER:
        limit = xavier_limit(in_features, out_features)
        weight = tuple(
            tuple(uniform[row * in_features + column] * 2.0 * limit - limit for column in range(in_features))
            for row in range(out_features)
        )
        bias = tuple(value * 2.0 * limit - limit for value in bias_uniform)
        return weight, bias

    if init == INIT_NORMAL:
        sigma = 1.0 / math.sqrt(in_features)
    else:  # INIT_HE
        sigma = he_scale(in_features)

    normal: list[float] = []
    index = 0
    while len(normal) < total:
        first = uniform[index] if index < total else 0.5
        second = uniform[index + 1] if index + 1 < total else 0.5
        pair = _normal_pair(first, second)
        normal.extend(pair)
        index += 2
    weight = tuple(
        tuple(normal[row * in_features + column] * sigma for column in range(in_features))
        for row in range(out_features)
    )
    normal_bias: list[float] = []
    index = 0
    while len(normal_bias) < out_features:
        first = bias_uniform[index] if index < out_features else 0.5
        second = bias_uniform[index + 1] if index + 1 < out_features else 0.5
        normal_bias.extend(_normal_pair(first, second))
        index += 2
    return weight, tuple(value * sigma for value in normal_bias[:out_features])


def _validate_matrix(rows: object, *, name: str) -> Matrix:
    """矩阵护栏：非空、行等长、元素有限."""
    if not isinstance(rows, (tuple, list)) or not rows:
        raise ShapeError(f"{name} 必须是非空的二维序列（行 × 列）。")
    checked: list[tuple[float, ...]] = []
    width: int | None = None
    for index, row in enumerate(rows):
        if not isinstance(row, (tuple, list)) or not row:
            raise ShapeError(f"{name} 的第 {index} 行不是非空的序列。")
        values = tuple(float(value) for value in row)
        if any(not math.isfinite(value) for value in values):
            raise ShapeError(f"{name} 的第 {index} 行含有非有限数（nan / inf）。")
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise ShapeError(f"{name} 的第 {index} 行宽度 {len(values)} 与首行 {width} 不一致。")
        checked.append(values)
    return tuple(checked)


def dense_linear(weight: Matrix, bias: Vector, inputs: Matrix) -> Matrix:
    """``y = x · Wᵀ + b``（**逐行**；累加用 ``math.fsum``，与项目里的 FFN 口径一致）.

    ``weight`` 形状 ``(out_features, in_features)``、``inputs`` 形状 ``(rows, in_features)``。
    宽度不齐抛 :class:`ShapeError`。
    """
    checked_weight = _validate_matrix(weight, name="weight")
    checked_inputs = _validate_matrix(inputs, name="inputs")
    out_features = len(checked_weight)
    in_features = len(checked_weight[0])
    if len(bias) != out_features:
        raise ShapeError(f"bias 长度 {len(bias)} 与 weight 的行数 {out_features} 不一致。")
    if len(checked_inputs[0]) != in_features:
        raise ShapeError(
            f"输入的列数 {len(checked_inputs[0])} 与 weight 的列数 {in_features} 不一致："
            "前向要求 x 与 W 共享同一个 in_features。"
        )
    return tuple(
        tuple(
            math.fsum(unit * value for unit, value in zip(weight_row, row, strict=True))
            + float(bias[index])
            for index, weight_row in enumerate(checked_weight)
        )
        for row in checked_inputs
    )


@dataclass(frozen=True)
class Dense:
    """一层全连接：权重 + 偏置 + 可选激活（``None`` 表示**恒等**）."""

    weight: Matrix
    bias: Vector
    activation: str | None = None

    def __post_init__(self) -> None:
        checked = _validate_matrix(self.weight, name="Dense.weight")
        if len(self.bias) != len(checked):
            raise ShapeError(
                f"Dense.bias 长度 {len(self.bias)} 与 weight 的行数 {len(checked)} 不一致。"
            )
        if self.activation is not None:
            # 借 activate 的护栏把未知激活名挡在构造期（用一行长度 1 的探针即可）
            activate(self.activation, (0.0,))
        object.__setattr__(self, "weight", checked)
        object.__setattr__(self, "bias", tuple(float(value) for value in self.bias))

    @property
    def in_features(self) -> int:
        """输入宽度."""
        return len(self.weight[0])

    @property
    def out_features(self) -> int:
        """输出宽度."""
        return len(self.weight)

    @property
    def parameter_count(self) -> int:
        """参数量 = ``in_features × out_features + out_features``."""
        return self.out_features * self.in_features + self.out_features

    @classmethod
    def from_spec(cls, spec: DenseSpec) -> Dense:
        """按 spec 造一层（权重由 LCG 生成，因此**可复现**）."""
        weight, bias = initialize(spec.init, spec.out_features, spec.in_features, seed=spec.seed)
        return cls(weight=weight, bias=bias, activation=spec.activation)

    def linear(self, inputs: Matrix) -> Matrix:
        """只做 ``x·Wᵀ + b``（不激活）."""
        return dense_linear(self.weight, self.bias, inputs)

    def forward(self, inputs: Matrix) -> Matrix:
        """``x·Wᵀ + b``，有激活时逐行再过一个激活（``softmax`` 按行处理）."""
        linear = self.linear(inputs)
        if self.activation is None:
            return linear
        return tuple(activate(self.activation, row) for row in linear)

    def to_dict(self) -> dict[str, object]:
        """摊平成一行字段（**不含**权重本体）."""
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "activation": self.activation,
            "parameter_count": self.parameter_count,
        }

    def line(self) -> str:
        """一行说明：``dense(3→4) | 激活 relu | 参数 16``."""
        act = self.activation if self.activation is not None else "identity"
        return (
            f"dense({self.in_features}→{self.out_features}) | 激活 {act} | "
            f"参数 {self.parameter_count}"
        )


def dense_forward(spec: DenseSpec, inputs: Matrix) -> Matrix:
    """按 spec 造一层并前向（同一份 spec 两次调用必须**逐位相同**）."""
    return Dense.from_spec(spec).forward(inputs)


def layer_accounting(spec: DenseSpec) -> dict[str, object]:
    """一层"账"：权重个数、偏置个数、总数、以及权重形状字符串."""
    return {
        "in_features": spec.in_features,
        "out_features": spec.out_features,
        "weight_shape": f"({spec.out_features}, {spec.in_features})",
        "weights": spec.weight_count,
        "biases": spec.out_features,
        "total": spec.parameter_count,
    }


def affine_forward(weight: Matrix, bias: Vector, inputs: Matrix) -> Matrix:
    """单个仿射映射 ``x·Wᵀ + b``（多层塌缩之后与它逐点比较）."""
    return dense_linear(weight, bias, inputs)


def _matmul(left: Matrix, right: Matrix) -> Matrix:
    """矩阵乘 ``A·B``（内维必须一致，否则抛 :class:`ShapeError`）."""
    rows = len(left)
    inner = len(left[0])
    if len(right) != inner:
        raise ShapeError(f"矩阵乘的内维不一致：{inner} != {len(right)}。")
    columns = len(right[0])
    return tuple(
        tuple(
            math.fsum(left[row][k] * right[k][column] for k in range(inner))
            for column in range(columns)
        )
        for row in range(rows)
    )


def compose_affine(layers: tuple[tuple[Matrix, Vector], ...]) -> tuple[Matrix, Vector]:
    """把一串仿射映射折成**一个**仿射映射（多层塌缩的唯一实现处）.

    输入是一串 ``(W_i, b_i)``，每个都表示 ``y = x·W_iᵀ + b_i``。
    合成结果是 ``(W, b)``，满足对所有 ``x`` 都有：

    ```text
    x·W_nᵀ + b_n  ∘  …  ∘  x·W_1ᵀ + b_1   ==   x·Wᵀ + b
    ```

    空串抛 :class:`ParameterError`（"零层网络"没有定义）。
    """
    if not layers:
        raise ParameterError("compose_affine 至少需要一层：零层网络的仿射没有定义。")
    weight, bias = layers[0]
    current_w = _validate_matrix(weight, name="weight")
    current_b = tuple(float(value) for value in bias)
    if len(current_b) != len(current_w):
        raise ShapeError("第一层的 bias 长度与 weight 行数不一致。")
    for index, (next_w, next_b) in enumerate(layers[1:], start=2):
        checked_w = _validate_matrix(next_w, name=f"weight#{index}")
        checked_b = tuple(float(value) for value in next_b)
        if len(checked_b) != len(checked_w):
            raise ShapeError(f"第 {index} 层的 bias 长度与 weight 行数不一致。")
        if len(checked_w[0]) != len(current_w):
            raise ShapeError(
                f"第 {index} 层的输入宽度 {len(checked_w[0])} 与上一层的输出宽度 "
                f"{len(current_w)} 不一致。"
            )
        # b ← W_next · b_cur + b_next ；W ← W_next · W_cur
        current_b = tuple(
            math.fsum(checked_w[row][k] * current_b[k] for k in range(len(current_b)))
            + checked_b[row]
            for row in range(len(checked_w))
        )
        current_w = _matmul(checked_w, current_w)
    return current_w, current_b


__all__ = [
    "LCG_INCREMENT",
    "LCG_MODULUS",
    "LCG_MULTIPLIER",
    "Dense",
    "affine_forward",
    "compose_affine",
    "dense_forward",
    "dense_linear",
    "he_scale",
    "initialize",
    "layer_accounting",
    "lcg_stream",
    "xavier_limit",
]
