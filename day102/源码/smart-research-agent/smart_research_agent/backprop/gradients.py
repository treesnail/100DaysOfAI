"""六个激活的**局部导数**与两个损失的**梯度**（day090 / M8-D2）.

day089 逐个写下了六个激活的**前向**；今天逐个写下它们的**导数**，并补上两个损失的梯度。
这一模块的位置与 day089 的 ``activations`` / ``losses`` 一一对应：

```text
neural_basics.activations.relu          ↔  backprop.gradients.relu_derivative
neural_basics.activations.softmax       ↔  backprop.gradients.softmax_jacobian
neural_basics.losses.mse                ↔  backprop.gradients.mse_grad
neural_basics.losses.cross_entropy      ↔  backprop.gradients.cross_entropy_grad
```

## 一、三个"用输出就能算"的导数，与一个"必须知道输入"的导数

```text
sigmoid    σ'(x) = σ(x)(1 − σ(x))     用**前向输出** ⇒ 不必再算一次 exp
tanh       tanh'(x) = 1 − tanh²(x)    同理
relu       relu'(x) = 1 if x > 0 else 0   只看**输入**的符号
gelu       gelu'(x) = 0.5(1 + erf(x/√2)) + x·e^{−x²/2}/√(2π)   ← **三项都要输入 x**
```

最后一条是这一课最容易漏的工程细节：逐元素反向的签名里必须带上激活**前**的值。
生产实现（``encoder_decoder.layers.activation_backward``）接受的正是 ``pre_activation``，
而不是激活之后的输出——因为 gelu 的导数里有 ``x·e^{−x²/2}`` 这一项，输出 ``y`` 里没有这个信息。

## 二、softmax：局部导数是一张矩阵，但**从不真的建出来**

```text
J[i][j] = p_i(δ_ij − p_j)                      显式：O(n²) 个元素
(Jᵀv)_i = p_i(v_i − ⟨p, v⟩)                     反向真正要算的东西：O(n)
```

:func:`softmax_jacobian` 把那张矩阵**显式**建出来——它的唯一用途是**被对照**：
与 day074 的数值雅可比比一次（第 3 条性质），并证明 :func:`softmax_jacobian_vector_product`
与它逐点一致（第 4 条性质）。**生产路径只走后者。**

## 三、交叉熵的梯度：一条"把雅可比消掉"的恒等式

```text
∂(−log p_k)/∂z_j = (∂/∂z_j)(−log p_k) = p_j − δ_jk
```

写成一行就是 ``p − onehot(k)``。它**刻意**没有经过我们刚写下的 ``softmax_jacobian``——
因为两者相乘之后中间那一大堆项全部抵消。这带来一个真实的风险：

```text
softmax 的雅可比写错了   ⇒  交叉熵的梯度**仍然是** p − onehot（因为正确的实现也不直接用它）
```

因此这一课把 softmax 雅可比**单独**验一次（第 3 条性质），而不是只验交叉熵。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from smart_research_agent.backprop.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.backprop.types import (
    ELEMENTWISE_ACTIVATIONS,
    LOSS_CROSS_ENTROPY,
    LOSS_MSE,
)
from smart_research_agent.math_foundations.types import Matrix, Vector
from smart_research_agent.neural_basics.activations import (
    ACTIVATIONS,
    LEAKY_SLOPE,
    sigmoid,
    softmax,
)
from smart_research_agent.neural_basics.types import GELU_TANH_CUBIC

#: ``√2``（gelu 的精确式与它的导数共用这一个常量）.
SQRT_TWO = math.sqrt(2.0)

#: ``√(2π)``（gelu 导数里那一项的系数）.
SQRT_TWO_PI = math.sqrt(2.0 * math.pi)

#: ``√(2/π)``（``gelu_new`` 的 tanh 近似里那个内系数，与 day085 / day089 同源）.
SQRT_TWO_OVER_PI = math.sqrt(2.0 / math.pi)

#: sigmoid 导数的上界（``σ(1−σ)`` 在 σ = 0.5 处取到 0.25）——写进常量让"它有多小"可断言.
SIGMOID_DERIVATIVE_CEILING = 0.25

# --------------------------------------------------------------------------------------
# 1. 五个逐元素激活的导数
# --------------------------------------------------------------------------------------


def relu_derivative(value: float) -> float:
    """``relu'(x) = 1 if x > 0 else 0``——在 ``x = 0`` 处取**次梯度 0**（约定，不是事实）.

    取 0 而不是 1 的理由很具体：一个**恒为负**的输入若在 0 处得到梯度，
    它会一直"被学习"，与"不激活就不学习"的直觉相反。取 0 之后，"关掉的神经元"
    在反向里梯度**恰好**是 0（第 2 条性质把它写成一条可断言的结论）。
    """
    return 1.0 if value > 0.0 else 0.0


def leaky_relu_derivative(value: float, slope: float = LEAKY_SLOPE) -> float:
    """``leaky_relu'(x) = 1 if x > 0 else slope``——负半轴那条坡度也出现在导数里."""
    if not math.isfinite(slope) or slope < 0.0:
        raise NumericError(f"leaky_relu 的斜率必须是非负有限数，收到 {slope!r}。")
    return 1.0 if value > 0.0 else float(slope)


def sigmoid_derivative(value: float) -> float:
    """``σ'(x) = σ(x)(1 − σ(x))``——值域 ``(0, 1)``，因此导数上界是 ``0.25``.

    它是唯一一个"导数上界恒小于 1"的逐元素激活：一连串 sigmoid 相乘会**指数衰减**
    （这正是"深层 sigmoid 网络梯度消失"的最小解释，与 day089 第 3 条笔记同源）。
    """
    output = sigmoid(value)
    return output * (1.0 - output)


def sigmoid_derivative_from_output(output: float) -> float:
    """``y(1 − y)``（直接用前向输出算）——省下一次 ``exp``，也让实现更难写错."""
    if not math.isfinite(output):
        raise NumericError(f"sigmoid 的前向输出必须是有限数，收到 {output!r}。")
    return output * (1.0 - output)


def tanh_derivative(value: float) -> float:
    """``tanh'(x) = 1 − tanh²(x)``——值域 ``(−1, 1)``，因此导数上界是 ``1``."""
    output = math.tanh(value)
    return 1.0 - output * output


def tanh_derivative_from_output(output: float) -> float:
    """``1 − y²``（直接用前向输出算）."""
    if not math.isfinite(output):
        raise NumericError(f"tanh 的前向输出必须是有限数，收到 {output!r}。")
    return 1.0 - output * output


def gelu_derivative(value: float) -> float:
    """精确 GELU 的导数 ``0.5(1 + erf(x/√2)) + x·e^{−x²/2}/√(2π)``.

    在 ``x = 0`` 处恰好是 ``0.5``（第一项 0.5、第二项 0），与 ``gelu_tanh`` 的导数
    在 0 处同值——两条式子在那里相切，两侧才分开（day085 量过那个偏差）。
    表达式与 ``encoder_decoder.layers.activation_backward`` **逐字相同**，
    因此逐位对账成立。
    """
    return 0.5 * (1.0 + math.erf(value / SQRT_TWO)) + value * math.exp(
        -0.5 * value * value
    ) / SQRT_TWO_PI


def gelu_tanh_derivative(value: float) -> float:
    """``gelu_new`` 的 tanh 近似的导数（``0.5(1+t) + 0.5x(1−t²)·√(2/π)(1+3ax²)``）.

    其中 ``t = tanh(√(2/π)(x + a x³))``、``a = 0.044715``。
    它**不是** :func:`gelu_derivative` 的等价写法：两式在 0 处同值同导，
    两侧各有 O(1e-4) 量级的偏差。
    """
    inner = SQRT_TWO_OVER_PI * (value + GELU_TANH_CUBIC * value**3)
    activation = math.tanh(inner)
    inner_derivative = SQRT_TWO_OVER_PI * (1.0 + 3.0 * GELU_TANH_CUBIC * value * value)
    return 0.5 * (1.0 + activation) + 0.5 * value * (1.0 - activation * activation) * inner_derivative


#: 五个逐元素激活的导数调度表（``softmax`` 不是逐元素，单独处理）.
_ELEMENTWISE_DERIVATIVES: dict[str, object] = {
    "relu": relu_derivative,
    "leaky_relu": leaky_relu_derivative,
    "sigmoid": sigmoid_derivative,
    "tanh": tanh_derivative,
    "gelu": gelu_derivative,
}

if set(_ELEMENTWISE_DERIVATIVES) != set(ELEMENTWISE_ACTIVATIONS):  # pragma: no cover - 导入期不变式
    raise ParameterError(
        "导数调度表与 ELEMENTWISE_ACTIVATIONS 不一致：少一个的那条激活会在反向里静默地没有梯度。"
    )


def activation_derivative(name: str, value: float) -> float:
    """按名字取一个**逐元素**激活在 ``value`` 处的导数（未知名字/``softmax`` 当场拒绝）.

    ``softmax`` 被显式拒绝，而不是"顺手返回 1"：它的局部导数是一张矩阵，
    对它取"一个数对一个数的导数"没有定义。
    """
    if name not in ACTIVATIONS:
        raise ParameterError(
            f"未知的激活 {name!r}：可选 {list(ACTIVATIONS)}。"
            "回退到恒等的后果是——一份 sigmoid 的梯度会被印成恒等的梯度，"
            "而'我确实选了 sigmoid'与'它被偷偷改成了恒等'在读表时长得一样。"
        )
    if name not in _ELEMENTWISE_DERIVATIVES:
        raise ParameterError(
            f"{name!r} 不是逐元素激活：它需要一整行（请用 softmax_jacobian 或 "
            "softmax_jacobian_vector_product）。对它取逐元素导数没有定义。"
        )
    function = _ELEMENTWISE_DERIVATIVES[name]
    return float(function(float(value)))  # type: ignore[operator]


# --------------------------------------------------------------------------------------
# 2. softmax：显式雅可比（只为被对照）与 JVP（生产路径）
# --------------------------------------------------------------------------------------


def softmax_jacobian(probabilities: Sequence[float]) -> Matrix:
    """softmax 的完整雅可比 ``J[i][j] = p_i(δ_ij − p_j)``.

    **入参是 softmax 的输出（概率），不是打分。** 这一条必须写下来，因为它不会报错：
    把 logits 传进来时，函数照样返回一张"看起来很像雅可比"的矩阵
    （对角线上是 ``z_i(1 − z_i)`` 这样的数），而它描述的是**另一个**函数。
    本课真的踩过一次——见 day090 教程里"同族坑的第三次记录"。

    **它的唯一用途是被对照**（第 3、4 条性质）：生产反向走
    :func:`softmax_jacobian_vector_product`，不建这张矩阵。
    空输入抛 :class:`ShapeError`——"一个不存在的一行的雅可比"没有定义。
    """
    values = tuple(float(value) for value in probabilities)
    if not values:
        raise ShapeError("softmax_jacobian 的输入不能为空：空分布没有雅可比。")
    return tuple(
        tuple(
            values[row] * ((1.0 if row == column else 0.0) - values[column])
            for column in range(len(values))
        )
        for row in range(len(values))
    )


def softmax_jacobian_vector_product(probabilities: Sequence[float], vector: Sequence[float]) -> Vector:
    """``(Jᵀv)_i = p_i(v_i − ⟨p, v⟩)``——**反向真正要算的东西**（不建 ``n × n`` 的矩阵）.

    这条恒等式的推导只有三行：

    ```text
    (Jᵀv)_i = Σ_j J[j][i]·v_j
            = Σ_j p_j(δ_ji − p_i)·v_j
            = p_i·v_i − p_i·Σ_j p_j·v_j
            = p_i(v_i − ⟨p, v⟩)
    ```

    它与"显式建 J 再乘"**逐点一致**（第 4 条性质），但中间量从 ``n²`` 降到 ``n``。
    生产实现里 softmax 的反向就是这一行。
    """
    values = tuple(float(value) for value in probabilities)
    tangents = tuple(float(value) for value in vector)
    if not values or not tangents:
        raise ShapeError("softmax 的 JVP 不接受空输入。")
    if len(values) != len(tangents):
        raise ShapeError(
            f"概率有 {len(values)} 项而回传梯度有 {len(tangents)} 项："
            "两者必须一一对应（每一维各有一个偏导数）。"
        )
    inner = math.fsum(p * v for p, v in zip(values, tangents, strict=True))
    return tuple(p * (v - inner) for p, v in zip(values, tangents, strict=True))


# --------------------------------------------------------------------------------------
# 3. 逐元素反向与逐行反向
# --------------------------------------------------------------------------------------


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


def elementwise_backward(name: str, pre_activation: Matrix, grad_output: Matrix) -> Matrix:
    """逐元素激活的反向：**逐元素**把局部导数乘上回传梯度.

    ``pre_activation`` 是激活**前**的值（第 3 章那条工程细节：gelu 的导数需要它）。
    ``relu`` 走一条专门的分支：``x <= 0`` 时**直接返回字面量 0.0**，
    而不是 ``grad * 0.0``——后者在 ``grad`` 是 ``inf`` 时会得到 ``nan``，
    而"关掉的神经元"应当无条件地得到 0。
    """
    if name not in _ELEMENTWISE_DERIVATIVES:
        raise ParameterError(
            f"elementwise_backward 只支持逐元素激活，收到 {name!r}："
            "softmax 请用 softmax_backward_rows。"
        )
    checked_pre = _validate_matrix(pre_activation, name="pre_activation")
    checked_grad = _validate_matrix(grad_output, name="grad_output")
    if len(checked_pre) != len(checked_grad) or len(checked_pre[0]) != len(checked_grad[0]):
        raise ShapeError(
            f"激活前值 {len(checked_pre)}×{len(checked_pre[0])} 与回传梯度 "
            f"{len(checked_grad)}×{len(checked_grad[0])} 的形状必须一致（激活是逐元素的）。"
        )
    if name == "relu":
        return tuple(
            tuple(grad if value > 0.0 else 0.0 for value, grad in zip(row, grad_row, strict=True))
            for row, grad_row in zip(checked_pre, checked_grad, strict=True)
        )
    function = _ELEMENTWISE_DERIVATIVES[name]
    return tuple(
        tuple(
            grad * float(function(value)) for value, grad in zip(row, grad_row, strict=True)
        )
        for row, grad_row in zip(checked_pre, checked_grad, strict=True)
    )


def softmax_backward_rows(probabilities: Matrix, grad_output: Matrix) -> Matrix:
    """逐行 softmax 的反向：每一行各自做一次 :func:`softmax_jacobian_vector_product`.

    ``probabilities`` 是 softmax 的**前向输出**（每行和为 1）——softmax 的反向只需要输出，
    不需要输入（这也是它与 gelu 的又一处不同）。
    """
    checked_prob = _validate_matrix(probabilities, name="probabilities")
    checked_grad = _validate_matrix(grad_output, name="grad_output")
    if len(checked_prob) != len(checked_grad) or len(checked_prob[0]) != len(checked_grad[0]):
        raise ShapeError(
            f"概率 {len(checked_prob)}×{len(checked_prob[0])} 与回传梯度 "
            f"{len(checked_grad)}×{len(checked_grad[0])} 的形状必须一致。"
        )
    return tuple(
        softmax_jacobian_vector_product(prob, grad)
        for prob, grad in zip(checked_prob, checked_grad, strict=True)
    )


# --------------------------------------------------------------------------------------
# 4. 两个损失的梯度
# --------------------------------------------------------------------------------------


def mse_grad(predictions: Matrix, targets: Matrix) -> Matrix:
    """``∂mean((p − t)²)/∂p = 2(p − t)/N``（``N`` = **全部元素**的个数）.

    分母是元素总数（不是行数、也不是"每行的元素数"）——因为它对的是一个"均值"。
    少一个 2 会让梯度整体减半，而它表现为"学习率看起来需要加倍"。
    """
    if len(predictions) != len(targets) or not predictions:
        raise ShapeError(
            f"mse_grad 需要两份**同形非空**的张量：收到 {len(predictions)} 行与 {len(targets)} 行。"
        )
    differences: list[float] = []
    for index, (pred_row, target_row) in enumerate(zip(predictions, targets, strict=True)):
        if len(pred_row) != len(target_row):
            raise ShapeError(
                f"mse_grad 的第 {index} 行宽度 {len(pred_row)} 与目标 {len(target_row)} 不一致。"
            )
        for pred, target in zip(pred_row, target_row, strict=True):
            if not (math.isfinite(pred) and math.isfinite(target)):
                raise NumericError(f"mse_grad 的第 {index} 行出现非有限数（nan / inf）。")
            differences.append(float(pred) - float(target))
    total = len(differences)
    return tuple(
        tuple(2.0 * differences[row * len(predictions[0]) + column] / total for column in range(len(predictions[0])))
        for row in range(len(predictions))
    )


def _checked_target(logits: Sequence[float], target: int) -> int:
    """标签护栏：必须是 ``[0, len)`` 内的整数（越界抛 ``GradientError`` 的邻居：ShapeError）."""
    if isinstance(target, bool) or not isinstance(target, int):
        raise ParameterError(f"标签必须是整数，收到 {target!r}。")
    if not 0 <= target < len(logits):
        raise ShapeError(
            f"标签 {target} 落在 [0, {len(logits)}) 之外：越界标签若被静默兜成 0，"
            "梯度会指向一个错误的方向，而报告里看不出是标签错了。"
        )
    return target


def cross_entropy_grad(logits: Sequence[float], target: int) -> Vector:
    """单点交叉熵的梯度 ``∂(−log softmax(z)[k])/∂z = p − onehot(k)``.

    这一条**刻意**不经过 :func:`softmax_jacobian`——两者相乘之后中间项全部抵消。
    读者因此必须知道：**交叉熵的梯度对不对，说明不了 softmax 的雅可比对不对**。
    """
    values = tuple(float(value) for value in logits)
    if not values:
        raise ShapeError("cross_entropy_grad 的 logits 不能为空。")
    if any(not math.isfinite(value) for value in values):
        raise NumericError("cross_entropy_grad 的 logits 必须是有限数（收到 nan / inf）。")
    checked = _checked_target(values, target)
    probabilities = softmax(values)
    return tuple(
        probability - (1.0 if index == checked else 0.0)
        for index, probability in enumerate(probabilities)
    )


def cross_entropy_grad_rows(logits_rows: Matrix, targets: Sequence[int]) -> Matrix:
    """逐行交叉熵的梯度（对**行取平均**）：``dZ_i = (p_i − onehot(k_i))/rows``.

    除以行数是因为损失是一个**批均值**。忘记这一个分母会让梯度大 ``rows`` 倍，
    而它看起来只是"这一批的学习率需要除以 batch size"。
    """
    if len(logits_rows) != len(targets):
        raise ShapeError(
            f"打分有 {len(logits_rows)} 行而标签有 {len(targets)} 个：两者必须一一对应。"
        )
    if not logits_rows:
        raise ShapeError("cross_entropy_grad_rows 的样本为空：对空批求梯度没有定义。")
    rows = len(logits_rows)
    return tuple(
        tuple(value / rows for value in cross_entropy_grad(logits, target))
        for logits, target in zip(logits_rows, targets, strict=True)
    )


#: 两个损失的梯度调度表（键与 :data:`types.LOSSES` 逐键对齐；值为"损失名 → 公式"）.
LOSS_GRADIENT_NAMES: tuple[str, ...] = (LOSS_MSE, LOSS_CROSS_ENTROPY)

__all__ = [
    "LOSS_GRADIENT_NAMES",
    "SIGMOID_DERIVATIVE_CEILING",
    "SQRT_TWO",
    "SQRT_TWO_OVER_PI",
    "SQRT_TWO_PI",
    "activation_derivative",
    "cross_entropy_grad",
    "cross_entropy_grad_rows",
    "elementwise_backward",
    "gelu_derivative",
    "gelu_tanh_derivative",
    "leaky_relu_derivative",
    "mse_grad",
    "relu_derivative",
    "sigmoid_derivative",
    "sigmoid_derivative_from_output",
    "softmax_backward_rows",
    "softmax_jacobian",
    "softmax_jacobian_vector_product",
    "tanh_derivative",
    "tanh_derivative_from_output",
]
