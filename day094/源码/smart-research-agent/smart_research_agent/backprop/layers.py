"""一层全连接的**反向**，以及参数的压平与还原（day090 / M8-D2）.

day089 的 ``neural_basics.layers`` 写下了 ``y = x·Wᵀ + b`` 这一行前向；今天写下它的三块梯度。

## 一、三块梯度（**这是本课最该背下来的三条式子**）

```text
grad_weight  (out, in)  = dYᵀ·X            回传梯度的每一列 · 输入
grad_bias    (out,)     = Σ_i dY[i]        偏置被所有行共用 ⇒ 按**行**求和
grad_inputs  (rows, in) = dY·W             按 W 的**列**收，而不是按行
```

第三条是这一课最容易把下标搞反的地方：

```text
正确   dx_c = Σ_r dY_r · W[r][c]    遍历的是 W 的**列**（r 是输出维）
写错   dx_c = Σ_r dY_r · W[c][r]    形状仍然对（都是 (rows, in)），数值全错
```

"形状仍然对"是它危险的地方：一个把 W 与 Wᵀ 用反的实现不会报任何错，
只会让训练变慢或发散——而它看起来像"学习率没调好"。

## 二、累加方式与 ``encoder_decoder`` 逐一相同（**这不是随便挑的**）

```text
grad_weight / grad_bias   用**普通累加**（与 encoder_decoder.layers.feed_forward_backward 相同）
grad_inputs               用 math.fsum（同样与它相同）
```

两条都对齐，才能让第 7 条性质（与项目里真实的前馈反向**逐位**一致）成立。
换一个求和顺序，两个实现会在最后一位上分家，而"最后一位不同"与"实现写错了"
在读一个标量时**看不出来**。这不是"随手挑了一个更精确的求和"——是为了那条对账。

## 三、参数的压平：一层一层摊开，形状表必须一起留着

训练时参数要交给 day074 的优化器，而优化器的更新公式是**逐分量**的一元运算
（``θ ← θ − lr·g``）——与参数是不是矩阵无关。因此：

```text
压平   (W, b) 的每一层按**行优先**摊平，前后拼起来；同时留下每一层的形状
还原   按形状表切回去；长度对不上时**当场报错**
```

长度对不上时报错而不是"按顺序切"：切错位置会把数值填进某个形状正确的矩阵里，
而结果是"若干形状正确、数值错位"的参数——它比一次崩溃难查得多。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from smart_research_agent.backprop.errors import NumericError, ShapeError
from smart_research_agent.backprop.types import DenseGradients, LayerGradients
from smart_research_agent.math_foundations.types import Matrix, Vector

#: 一层全连接的参数（权重 + 偏置）.
ParameterPair = tuple[Matrix, Vector]

#: 一层的形状表：``(out_features, in_features, out_features)``——权重形状 + 偏置长度.
LayerShape = tuple[int, int, int]


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
            raise NumericError(
                f"{name} 的第 {index} 行含有非有限数（nan / inf）："
                "非有限数会一路污染到整张梯度表，而根因在更早的一次除法或 log 上。"
            )
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise ShapeError(f"{name} 的第 {index} 行宽度 {len(values)} 与首行 {width} 不一致。")
        checked.append(values)
    return tuple(checked)


def _norm(values: Sequence[float]) -> float:
    """一串数的欧氏范数 ``√Σv²``（用 ``math.fsum`` 累加，与项目其余部分同口径）."""
    return math.sqrt(math.fsum(float(value) * float(value) for value in values))


def dense_backward(weight: Matrix, bias: Vector, inputs: Matrix, grad_output: Matrix) -> DenseGradients:
    """一层全连接的反向（三块：``dW``、``db``、``dx``）.

    ``weight`` 形状 ``(out, in)``、``inputs`` 形状 ``(rows, in)``、``grad_output`` 形状
    ``(rows, out)``。累加方式与 ``encoder_decoder.layers.feed_forward_backward`` **逐字相同**，
    因此两条实现可以被逐位对账（第 7 条性质）。

    形状不齐抛 :class:`ShapeError`；出现非有限数抛 :class:`NumericError`。
    """
    checked_weight = _validate_matrix(weight, name="weight")
    checked_inputs = _validate_matrix(inputs, name="inputs")
    checked_grad = _validate_matrix(grad_output, name="grad_output")
    out_features = len(checked_weight)
    in_features = len(checked_weight[0])
    rows = len(checked_inputs)
    if len(checked_grad) != rows:
        raise ShapeError(
            f"回传梯度有 {len(checked_grad)} 行而输入有 {rows} 行："
            "反向的每一行对应前向的每一行，两者必须一一对应。"
        )
    if len(checked_grad[0]) != out_features:
        raise ShapeError(
            f"回传梯度的列数 {len(checked_grad[0])} 与权重的行数 {out_features} 不一致："
            "这一层的输出宽度必须与回传梯度对上。"
        )
    if len(bias) != out_features:
        raise ShapeError(f"bias 长度 {len(bias)} 与 weight 的行数 {out_features} 不一致。")
    if len(checked_inputs[0]) != in_features:
        raise ShapeError(
            f"输入的列数 {len(checked_inputs[0])} 与权重的列数 {in_features} 不一致。"
        )

    # 两块的累加顺序刻意与 encoder_decoder 相同：按行 → 按输出维 → 按输入维。
    grad_weight: list[list[float]] = [[0.0] * in_features for _ in range(out_features)]
    grad_bias: list[float] = [0.0] * out_features
    for row_grad, input_row in zip(checked_grad, checked_inputs, strict=True):
        for index in range(out_features):
            grad_bias[index] += row_grad[index]
            for column in range(in_features):
                grad_weight[index][column] += row_grad[index] * input_row[column]

    # dx 按 W 的**列**收（每一项遍历输出维），并用 fsum——与生产实现同一口径。
    grad_inputs = tuple(
        tuple(
            math.fsum(
                row_grad[index] * checked_weight[index][column] for index in range(out_features)
            )
            for column in range(in_features)
        )
        for row_grad in checked_grad
    )
    return DenseGradients(
        grad_weight=tuple(tuple(row) for row in grad_weight),
        grad_bias=tuple(grad_bias),
        grad_inputs=grad_inputs,
    )


def affine_backward(weight: Matrix, bias: Vector, inputs: Matrix, grad_output: Matrix) -> DenseGradients:
    """仿射映射（无激活的那一半）的反向——就是 :func:`dense_backward`.

    单独给出这个名字，是因为"带激活的一层"的反向是
    ``activation_backward ∘ affine_backward`` 的复合，而这一课希望那个复合
    在代码里看得见：先过激活的反向，再过仿射的反向。
    """
    return dense_backward(weight, bias, inputs, grad_output)


def gradient_summary(grads: DenseGradients, index: int) -> LayerGradients:
    """把一层的三块梯度收成一行账（三个范数 + 权重形状）."""
    rows, columns = grads.weight_shape
    return LayerGradients(
        index=index,
        weight_shape=(rows, columns),
        weight_norm=_norm(value for row in grads.grad_weight for value in row),
        bias_norm=_norm(grads.grad_bias),
        input_norm=_norm(value for row in grads.grad_inputs for value in row),
    )


# --------------------------------------------------------------------------------------
# 参数的压平与还原
# --------------------------------------------------------------------------------------


def flatten_parameters(params: Sequence[ParameterPair]) -> tuple[Vector, tuple[LayerShape, ...]]:
    """把一层一层的 ``(W, b)`` 按行优先压平成一串数，返回 ``(向量, 形状表)``.

    空参数表抛 :class:`ShapeError`——一个没有参数的"模型"没有可优化的东西，
    而它在下游会表现为"每一步的损失完全不变"（因为确实什么都没改）。
    """
    if not params:
        raise ShapeError("至少要有一层参数：空参数表没有可优化的东西。")
    shapes: list[LayerShape] = []
    flat: list[float] = []
    for index, (weight, bias) in enumerate(params):
        checked_weight = _validate_matrix(weight, name=f"weight[{index}]")
        out_features = len(checked_weight)
        in_features = len(checked_weight[0])
        if len(bias) != out_features:
            raise ShapeError(
                f"第 {index} 层的 bias 长度 {len(bias)} 与 weight 行数 {out_features} 不一致。"
            )
        shapes.append((out_features, in_features, out_features))
        for row in checked_weight:
            flat.extend(row)
        flat.extend(float(value) for value in bias)
    return tuple(flat), tuple(shapes)


def unflatten_parameters(flat: Sequence[float], shapes: Sequence[LayerShape]) -> tuple[ParameterPair, ...]:
    """把一串数按形状表还原成一层一层的 ``(W, b)``（长度对不上时**当场报错**）."""
    values = tuple(float(value) for value in flat)
    if any(not math.isfinite(value) for value in values):
        raise NumericError(
            "压平的参数里出现非有限数：一步更新把参数推到 inf，"
            "而下一次前向会给出 nan 损失——根因是这一步的学习率。"
        )
    if not shapes:
        raise ShapeError("形状表不能为空：没有形状就没有可还原的参数。")
    required = sum(out * inn + out for out, inn, _ in shapes)
    if len(values) != required:
        raise ShapeError(
            f"压平的向量有 {len(values)} 个数，而形状表需要 {required} 个："
            "长度对不上时'按顺序切'会把切错的位置静默地填进某一层，"
            "而结果是若干'形状正确、数值错位'的参数。"
        )
    params: list[ParameterPair] = []
    cursor = 0
    for out_features, in_features, _bias_len in shapes:
        weight_rows: list[tuple[float, ...]] = []
        for _ in range(out_features):
            weight_rows.append(values[cursor : cursor + in_features])
            cursor += in_features
        bias = values[cursor : cursor + out_features]
        cursor += out_features
        params.append((tuple(weight_rows), bias))
    return tuple(params)


def parameter_count(shapes: Sequence[LayerShape]) -> int:
    """形状表描述的参数总数（用于报告：它必须与压平后的长度相等）."""
    return sum(out * inn + out for out, inn, _ in shapes)


__all__ = [
    "LayerShape",
    "ParameterPair",
    "affine_backward",
    "dense_backward",
    "flatten_parameters",
    "gradient_summary",
    "parameter_count",
    "unflatten_parameters",
]
