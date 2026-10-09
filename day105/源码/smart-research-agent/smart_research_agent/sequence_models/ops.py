"""``ops``：循环网络用到的**算子层**（day094 / M8-D5）.

这一层只回答"一个算子把一个向量（或一串向量）变成什么"，不含参数、不含训练：

```text
as_vector / as_matrix / as_sequence   三样护栏（非空、等长、有限）
matvec        W·x          仿射的一半
affine        W·x + b      一个循环单元每一步真正算的那件事
matvec_t      Wᵀ·g         反向的一半（把梯度投影回输入空间）
outer         g ⊗ x        dW 的每一时刻那一片（**要累加**）
vec_add / vec_hadamard / vec_scale    逐分量三件套
gate_blocks   把 4H 的打分切成 i / f / o / g 四个块
parameter_count  两种单元的参数量公式（**与 T 无关**）
```

## 一个必须写下来的约定：行是输出、列是输入

```text
W 的形状 (H, D)：H 行 = H 个输出单元，D 列 = D 个输入分量
W·x 的第 i 个分量 = Σ_j W[i][j]·x[j]        ← 第 i 行与 x 的点积
```

本包与 ``torch.nn.Linear``（``weight`` 形状 ``(out, in)``）以及 day089 的
``Dense``（``(out_features, in_features)``）**同口径**。转置写反了形状常常仍然合法
（方阵），它只会算错——第 ① 条性质用一个非方阵把这件事钉住。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from smart_research_agent.math_foundations.types import Matrix, Vector
from smart_research_agent.sequence_models.errors import (
    NumericError,
    ParameterError,
    ShapeError,
    TimeStepError,
)
from smart_research_agent.sequence_models.types import CELL_LSTM, CELL_RNN, CELL_TYPES


def checked_positive_int(value: object, *, name: str, minimum: int = 1) -> int:
    """校验一个 ``>= minimum`` 的整数（输入维 / 隐层宽 / 步数都走这里）."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParameterError(f"{name} 必须是整数，收到 {type(value).__name__}（{value!r}）。")
    if value < minimum:
        raise ParameterError(
            f"{name} 必须 >= {minimum}，收到 {value}："
            "非正的宽度会让'我有多少个可学的量'这件事失去定义，而它不会报错，只会算出一堆空向量。"
        )
    return value


def as_vector(values: Sequence[float], *, name: str) -> Vector:
    """把一串数收敛成 ``Vector``（非空、元素有限）."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise ShapeError(f"{name} 必须是非空的向量（一串数）。")
    checked: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ShapeError(f"{name}[{index}] 必须是实数，收到 {value!r}。")
        number = float(value)
        if not math.isfinite(number):
            raise NumericError(f"{name}[{index}] 是非有限数（{value!r}）。")
        checked.append(number)
    return tuple(checked)


def as_matrix(rows: Sequence[Sequence[float]], *, name: str) -> Matrix:
    """把二维序列收敛成 ``Matrix``（非空、行等长、元素有限）."""
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ShapeError(f"{name} 必须是非空的二维序列（行 × 列）。")
    checked: list[tuple[float, ...]] = []
    width: int | None = None
    for index, row in enumerate(rows):
        values = as_vector(row, name=f"{name}[{index}]")
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise ShapeError(f"{name} 的第 {index} 行宽度 {len(values)} 与首行 {width} 不一致。")
        checked.append(values)
    return tuple(checked)


def as_sequence(sequence: Sequence[Sequence[float]], *, name: str) -> tuple[Vector, ...]:
    """收敛成一串**等长**的向量（``T >= 1``，每步宽度一致）.

    "每步宽度一致"是循环网络的前提：同一组权重要在每个时刻乘同一个 ``x_t``，
    宽度不一样的输入会让第 t 步的乘法根本没有定义。空序列抛 :class:`TimeStepError`
    ——"零步的序列"与"一步全零的序列"是两件不同的事，本包不接受前者。
    """
    if isinstance(sequence, (str, bytes)) or not isinstance(sequence, Sequence) or not sequence:
        raise TimeStepError(
            f"{name} 必须是非空的序列（至少一步）："
            "零步的序列没有状态可返回，而它与'一步全零'在读报告时长得一样。"
        )
    checked: list[Vector] = []
    width: int | None = None
    for index, step in enumerate(sequence):
        values = as_vector(step, name=f"{name}[{index}]")
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise ShapeError(
                f"{name} 的第 {index} 步宽度 {len(values)} 与首步 {width} 不一致："
                "同一组权重要在每一步乘同一个形状的输入。"
            )
        checked.append(values)
    return tuple(checked)


def zeros(size: int) -> Vector:
    """长度为 ``size`` 的全零向量（``h₀`` 的缺省值）."""
    return (0.0,) * checked_positive_int(size, name="size")


def vec_add(left: Vector, right: Vector) -> Vector:
    """逐分量相加（长度不一致抛 :class:`ShapeError`）."""
    if len(left) != len(right):
        raise ShapeError(f"逐分量相加要求等长：{len(left)} != {len(right)}。")
    return tuple(a + b for a, b in zip(left, right, strict=True))


def vec_scale(vector: Vector, factor: float) -> Vector:
    """逐分量乘一个标量（``factor`` 非有限抛 :class:`NumericError`）."""
    if not math.isfinite(factor):
        raise NumericError(f"缩放因子必须有限，收到 {factor!r}。")
    return tuple(value * factor for value in vector)


def vec_hadamard(left: Vector, right: Vector) -> Vector:
    """逐分量相乘（LSTM 的三处 ``⊙`` 都走它；长度不一致抛 :class:`ShapeError`）."""
    if len(left) != len(right):
        raise ShapeError(f"逐分量相乘要求等长：{len(left)} != {len(right)}。")
    return tuple(a * b for a, b in zip(left, right, strict=True))


def add_triple(first: Vector, second: Vector, third: Vector) -> Vector:
    """三个等长向量逐分量相加（``z = W_x·x + W_h·h + b``；循环前向的每一行都是它）."""
    if not (len(first) == len(second) == len(third)):
        raise ShapeError(
            f"三项相加要求等长：{len(first)} / {len(second)} / {len(third)} 不一致。"
        )
    return tuple(a + b + c for a, b, c in zip(first, second, third, strict=True))


def matvec(weight: Matrix, vector: Vector) -> Vector:
    """``W·x``：``weight`` 形状 ``(H, D)``、``vector`` 长度 ``D``（内维不一致抛 ``ShapeError``）.

    累加用 ``math.fsum``——与 day089 的 ``dense_linear``、day093 的 ``conv2d``
    同一口径，于是"同一组权重在 Python 侧算出来的数"可以逐位比较。
    """
    checked_weight = as_matrix(weight, name="weight")
    checked_vector = as_vector(vector, name="vector")
    columns = len(checked_weight[0])
    if len(checked_vector) != columns:
        raise ShapeError(
            f"W·x 的内维不一致：W 有 {columns} 列，而 x 有 {len(checked_vector)} 个分量。"
        )
    return tuple(
        math.fsum(unit * value for unit, value in zip(row, checked_vector, strict=True))
        for row in checked_weight
    )


def matvec_transpose(weight: Matrix, gradient: Vector) -> Vector:
    """``Wᵀ·g``：把回传梯度投影回输入空间（``weight`` 形状 ``(H, D)``、``gradient`` 长度 ``H``）.

    这是 ``matvec`` 的转置，也是循环反向里 ``dx`` 与 ``dh_{t−1}`` 的唯一来源。
    把 ``matvec`` 与 ``matvec_transpose`` 写反，形状常常仍然合法（方阵时必然合法），
    只有数值差分能把它们区分开。
    """
    checked_weight = as_matrix(weight, name="weight")
    checked_gradient = as_vector(gradient, name="gradient")
    rows = len(checked_weight)
    if len(checked_gradient) != rows:
        raise ShapeError(
            f"Wᵀ·g 的内维不一致：W 有 {rows} 行，而 g 有 {len(checked_gradient)} 个分量。"
        )
    columns = len(checked_weight[0])
    return tuple(
        math.fsum(checked_weight[row][column] * checked_gradient[row] for row in range(rows))
        for column in range(columns)
    )


def outer(left: Vector, right: Vector) -> Matrix:
    """外积 ``left ⊗ right``：形状 ``(len(left), len(right))``.

    ``dW[i][j] = dz[i]·x[j]`` 就是这一件事。注意它是**每一步各一片**，
    累加发生在 ``gradients`` 里（``+=`` 是 BPTT 的纪律）。
    """
    checked_left = as_vector(left, name="left")
    checked_right = as_vector(right, name="right")
    return tuple(tuple(a * b for b in checked_right) for a in checked_left)


def affine(vector: Vector, weight: Matrix, bias: Vector) -> Vector:
    """``W·x + b``：一个循环单元每一步真正算的那件事（门把 ``W_x`` 与 ``W_h`` 拼在一起）."""
    projected = matvec(weight, vector)
    if len(projected) != len(bias):
        raise ShapeError(
            f"bias 长度 {len(bias)} 与 W 的行数 {len(projected)} 不一致：逐分量相加要求等长。"
        )
    return vec_add(projected, as_vector(bias, name="bias"))


def gate_blocks(vector: Vector, hidden_size: int) -> tuple[Vector, Vector, Vector, Vector]:
    """把 ``4H`` 的打分切成 ``(i, f, o, g)`` 四块（顺序 = :data:`types.GATES`）.

    **切块的顺序就是权重行块的顺序**：``i`` 是前 ``H`` 行、``f`` 是接着 ``H`` 行……
    顺序写错不会报错（四个块都还是长度 ``H`` 的向量），只会让"输入门"做"遗忘门"的事。
    """
    checked = as_vector(vector, name="vector")
    checked_hidden = checked_positive_int(hidden_size, name="hidden_size")
    expected = 4 * checked_hidden
    if len(checked) != expected:
        raise ShapeError(
            f"门的打分长度 {len(checked)} 与 4×hidden_size = {expected} 不一致。"
        )
    return (
        checked[0:checked_hidden],
        checked[checked_hidden : 2 * checked_hidden],
        checked[2 * checked_hidden : 3 * checked_hidden],
        checked[3 * checked_hidden : 4 * checked_hidden],
    )


def parameter_count(cell: str, input_size: int, hidden_size: int) -> int:
    """两种单元的参数量（**不含 T**——这就是"时间轴上的权重共享"）.

    ```text
    rnn    H·D + H·H + H          （W_x、W_h、偏置）
    lstm   4·(H·D + H·H + H)      （四个门各要一套）
    ```
    """
    if cell not in CELL_TYPES:
        raise ParameterError(f"未知的循环单元 {cell!r}：可用取值 {list(CELL_TYPES)}。")
    checked_input = checked_positive_int(input_size, name="input_size")
    checked_hidden = checked_positive_int(hidden_size, name="hidden_size")
    per_gate = checked_hidden * checked_input + checked_hidden * checked_hidden + checked_hidden
    return per_gate if cell == CELL_RNN else 4 * per_gate


def sequence_norms(sequence: Sequence[Vector]) -> Vector:
    """每一步隐状态的 L2 范数（"信息在这条链上还剩多少"的最直接读数）."""
    return tuple(
        math.sqrt(math.fsum(value * value for value in step))
        for step in as_sequence(sequence, name="sequence")
    )


def as_finite_vector(values: Sequence[float], *, name: str) -> Vector:
    """与 :func:`as_vector` 同义但**允许为空**（给"梯度整体范数"一类的读数用）."""
    checked: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ShapeError(f"{name}[{index}] 必须是实数，收到 {value!r}。")
        number = float(value)
        if not math.isfinite(number):
            raise NumericError(f"{name}[{index}] 是非有限数（{value!r}）：先查那一步的学习率。")
        checked.append(number)
    return tuple(checked)


__all__ = [
    "add_triple",
    "affine",
    "as_finite_vector",
    "as_matrix",
    "as_sequence",
    "as_vector",
    "checked_positive_int",
    "gate_blocks",
    "matvec",
    "matvec_transpose",
    "outer",
    "parameter_count",
    "sequence_norms",
    "vec_add",
    "vec_hadamard",
    "vec_scale",
    "zeros",
]
