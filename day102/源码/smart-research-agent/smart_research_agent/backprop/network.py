"""一个网络的**反向**：逐层回传，以及与前馈生产实现的接缝（day090 / M8-D2）.

day089 的 ``neural_basics.network`` 写下了 MLP 的前向与两条结构事实。今天在它上面
补一件事：**把前向的中间量缓存下来，然后从损失往回走一遍**。

```text
前向（带缓存）   x → Dense₁ → act₁ → … → Dense_n →（可选 act_n）→ ŷ
损失            L = mse(ŷ, t)  或  L = cross_entropy(ŷ, targets)
反向            dL/dŷ → 逐层：激活的反向 → 仿射的反向（dW、db、dx）→ 交给上一层
```

## 一、为什么要缓存，缓存了什么

```text
Dense 的反向需要        这一层的**输入**（算 dW = dYᵀ·X）
逐元素激活的反向需要     激活**前**的值（gelu 的导数里有 x·e^{−x²/2}）
softmax 的反向需要       激活**后**的值（σ 的输出就是它自己的概率）
```

三样东西各不相同——这就是"缓存"不是"把前向重跑一遍"的原因：重跑一遍在数学上一样，
但**它会让反向依赖前向的参数**，而参数在反向过程中正是我们要算的东西。
把中间量留下来之后，反向只读缓存，不碰参数。

## 二、接口与项目里真实前馈的接缝

:func:`ffn_forward` / :func:`ffn_backward` 接受任何"有 ``w_in / b_in / w_out / b_out``
四个属性"的对象（鸭式），因此可以直接把 ``encoder_decoder.types.FFNWeights`` 传进来，
与 ``encoder_decoder.layers.feed_forward_backward`` **逐位**对账（第 7 条性质）。
累加方式与它逐字相同——这不是"随手挑了一个更精确的求和"，是为了那条对账。

## 三、一次反向的三个"看见"（这一课最想留下的直觉）

```text
看见一   relu 的反向是一个开关：关掉的神经元**恰好**拿到 0（不是"接近 0"）
看见二   softmax 的反向不需要雅可比矩阵：一行一次 JVP，O(n) 而不是 O(n²)
看见三   一层的 dW 是"回传梯度的每一列 · 输入"——它把"这一层看到了什么"与"损失要什么"乘在一起
```
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from smart_research_agent.backprop import layers as layer_ops
from smart_research_agent.backprop.errors import BackwardError, ParameterError, ShapeError
from smart_research_agent.backprop.gradients import (
    cross_entropy_grad_rows,
    elementwise_backward,
    softmax_backward_rows,
)
from smart_research_agent.backprop.layers import ParameterPair
from smart_research_agent.backprop.types import (
    LOSS_CROSS_ENTROPY,
    LOSS_MSE,
    DenseGradients,
    FFNGradients,
    LayerGradients,
)
from smart_research_agent.math_foundations.types import Matrix, Vector
from smart_research_agent.neural_basics.layers import Dense, dense_linear, initialize
from smart_research_agent.neural_basics.losses import cross_entropy as cross_entropy_loss
from smart_research_agent.neural_basics.losses import mse as mse_loss
from smart_research_agent.neural_basics.types import ACT_RELU, ACTIVATIONS, MLPSpec

#: 前馈的反向只支持这两个激活（与 ``encoder_decoder`` 的两个激活一一对应）.
FFN_ACTIVATIONS: tuple[str, ...] = (ACT_RELU, "gelu")


@dataclass(frozen=True)
class ForwardCache:
    """一次 MLP 前向留下的中间量（反向只读它，不碰参数）.

    ```text
    inputs          每一层的**输入**（第 0 项是整个网络的输入）
    pre_activations 每一层激活**前**的值（无激活时为 None）
    outputs         每一层激活**后**的值（softmax 的反向读它）
    ```

    三个长度都等于层数；把它们写下来是因为反向对它们的需求**各不相同**
    （见模块说明第一节）。
    """

    layers: tuple[Dense, ...]
    inputs: tuple[Matrix, ...]
    pre_activations: tuple[Matrix | None, ...]
    outputs: tuple[Matrix, ...]

    @property
    def depth(self) -> int:
        """层数."""
        return len(self.layers)

    @property
    def output(self) -> Matrix:
        """整个网络的输出（最后一层激活之后）."""
        return self.outputs[-1]

    def __post_init__(self) -> None:
        if self.depth == 0:
            raise BackwardError("ForwardCache 至少要有一层：零层的前向没有缓存。")
        for label, series in (
            ("inputs", self.inputs),
            ("pre_activations", self.pre_activations),
            ("outputs", self.outputs),
        ):
            if len(series) != self.depth:
                raise BackwardError(
                    f"缓存 {label} 有 {len(series)} 项而层数是 {self.depth}："
                    "长度对不上的缓存会让反向读错某一层的中间量，而它的形状仍然是对的。"
                )


@dataclass(frozen=True)
class BackwardTrace:
    """一次 MLP 反向的全部产物：逐层的三块梯度 + 对输入的梯度.

    ``layer_gradients`` 的顺序是**从前到后**（第 1 层在最前），
    与 :meth:`layer_summaries` 印出来的表顺序一致——报告与数据用同一个方向，
    免得读的人在脑子里翻一次。
    """

    layer_gradients: tuple[DenseGradients, ...]
    grad_inputs: Matrix

    def __post_init__(self) -> None:
        if not self.layer_gradients:
            raise BackwardError("BackwardTrace 至少要有一层的梯度。")

    def layer_summaries(self) -> tuple[LayerGradients, ...]:
        """逐层的三个范数（报告里的"一层反向表"读的就是它）."""
        return tuple(
            layer_ops.gradient_summary(grads, index)
            for index, grads in enumerate(self.layer_gradients, start=1)
        )

    def parameter_gradients(self) -> tuple[ParameterPair, ...]:
        """把逐层梯度收成一串 ``(dW, db)``（顺序 = 层序，供优化器使用）."""
        return tuple((grads.grad_weight, grads.grad_bias) for grads in self.layer_gradients)

    def flat_parameter_gradients(self) -> Vector:
        """按 :func:`backprop.layers.flatten_parameters` 的口径压平全部参数梯度."""
        flat: list[float] = []
        for grads in self.layer_gradients:
            for row in grads.grad_weight:
                flat.extend(row)
            flat.extend(grads.grad_bias)
        return tuple(flat)

    def flatten(self) -> tuple[Vector, tuple[layer_ops.LayerShape, ...]]:
        """压平梯度并给出形状表（形状表与参数共用，因此两者可以相减）."""
        return layer_ops.flatten_parameters(self.parameter_gradients())


# --------------------------------------------------------------------------------------
# 前向（带缓存）
# --------------------------------------------------------------------------------------


def build_chain(spec: MLPSpec) -> tuple[Dense, ...]:
    """按 spec 造一串 :class:`~smart_research_agent.neural_basics.layers.Dense`（权重由 LCG 生成）."""
    return tuple(Dense.from_spec(layer) for layer in spec.layers)


def chain_from_parameters(
    params: Sequence[ParameterPair], activations: Sequence[str | None]
) -> tuple[Dense, ...]:
    """用**显式的** ``(W, b)`` 造一串层（训练中每一步参数都在变，因此必须能外部注入）."""
    if len(params) != len(activations):
        raise ShapeError(
            f"参数有 {len(params)} 组而激活有 {len(activations)} 个：两者必须一一对应。"
        )
    if not params:
        raise ShapeError("至少要有一层：零层网络没有前向。")
    chain: list[Dense] = []
    for index, ((weight, bias), activation) in enumerate(zip(params, activations, strict=True)):
        if activation is not None and activation not in ACTIVATIONS:
            raise ParameterError(f"第 {index + 1} 层未知的激活 {activation!r}：可选 {list(ACTIVATIONS)}。")
        chain.append(Dense(weight=weight, bias=bias, activation=activation))
    return tuple(chain)


def chain_forward(chain: Sequence[Dense], inputs: Matrix) -> Matrix:
    """逐层前向（与 ``neural_basics.network.mlp_forward`` **逐位相同**——都走 ``Dense.forward``）."""
    if not chain:
        raise ShapeError("chain_forward 的层链不能为空。")
    if not inputs:
        raise ShapeError("chain_forward 的输入不能为空：至少要有 1 行样本。")
    current: Matrix = inputs
    for dense in chain:
        current = dense.forward(current)
    return current


def chain_forward_with_cache(chain: Sequence[Dense], inputs: Matrix) -> ForwardCache:
    """逐层前向并留下中间量（反向只读它）."""
    if not chain:
        raise ShapeError("chain_forward_with_cache 的层链不能为空。")
    if not inputs:
        raise ShapeError("chain_forward_with_cache 的输入不能为空。")
    layer_inputs: list[Matrix] = []
    pre_activations: list[Matrix | None] = []
    outputs: list[Matrix] = []
    current: Matrix = inputs
    for dense in chain:
        pre = dense.linear(current)
        activated = dense.forward(current)
        layer_inputs.append(current)
        pre_activations.append(None if dense.activation is None else pre)
        outputs.append(activated)
        current = activated
    return ForwardCache(
        layers=tuple(chain),
        inputs=tuple(layer_inputs),
        pre_activations=tuple(pre_activations),
        outputs=tuple(outputs),
    )


def mlp_forward_with_cache(spec: MLPSpec, inputs: Matrix) -> ForwardCache:
    """按 spec 前向并留下缓存（``spec`` 的权重由 LCG 生成，因此可复现）."""
    return chain_forward_with_cache(build_chain(spec), inputs)


# --------------------------------------------------------------------------------------
# 反向
# --------------------------------------------------------------------------------------


def activation_backward(
    activation: str | None, pre_activation: Matrix | None, output: Matrix, grad_output: Matrix
) -> Matrix:
    """一层的**激活**反向（逐元素激活读输入、``softmax`` 读输出）.

    这一个函数把"哪些激活的反向需要什么"这件事收在一处：
    逐元素需要激活**前**的值，``softmax`` 需要激活**后**的值。
    把 ``softmax`` 也当成逐元素的后果是——它会被当成"每个数各自一个导数"，
    而它其实是一整行一起变（形状对、数值错）。
    """
    if activation is None:
        return grad_output
    if activation == "softmax":
        return softmax_backward_rows(output, grad_output)
    if pre_activation is None:
        raise BackwardError(
            f"激活 {activation!r} 的反向需要激活**前**的值，而缓存里没有它："
            "这一次反向用的不是同一次前向的缓存。"
        )
    return elementwise_backward(activation, pre_activation, grad_output)


def mlp_backward(cache: ForwardCache, grad_output: Matrix) -> BackwardTrace:
    """逐层回传（**从最后一层走到第一层**），返回每一层的三块梯度与对输入的梯度.

    ```text
    对第 i 层：  激活的反向 → 仿射的反向（dW、db、dx）→ 把 dx 交给第 i−1 层
    ```
    """
    if len(grad_output) != len(cache.output) or len(grad_output[0]) != len(cache.output[0]):
        raise ShapeError(
            f"回传梯度 {len(grad_output)}×{len(grad_output[0])} 与网络输出 "
            f"{len(cache.output)}×{len(cache.output[0])} 形状不一致。"
        )
    current: Matrix = grad_output
    reversed_gradients: list[DenseGradients] = []
    for index in range(cache.depth - 1, -1, -1):
        dense = cache.layers[index]
        through_activation = activation_backward(
            dense.activation, cache.pre_activations[index], cache.outputs[index], current
        )
        grads = layer_ops.dense_backward(
            dense.weight, dense.bias, cache.inputs[index], through_activation
        )
        reversed_gradients.append(grads)
        current = grads.grad_inputs
    reversed_gradients.reverse()
    return BackwardTrace(layer_gradients=tuple(reversed_gradients), grad_inputs=current)


def mlp_loss_gradients(cache: ForwardCache, targets: Matrix) -> BackwardTrace:
    """以 **MSE** 为损失做一次完整反向（``dL/dŷ = 2(ŷ − t)/N``）."""
    from smart_research_agent.backprop.gradients import mse_grad

    return mlp_backward(cache, mse_grad(cache.output, targets))


def softmax_loss_gradients(cache: ForwardCache, targets: Sequence[int]) -> BackwardTrace:
    """以**交叉熵**为损失做一次完整反向（``dL/dz = (p − onehot)/rows``）."""
    return mlp_backward(cache, cross_entropy_grad_rows(cache.output, targets))


# --------------------------------------------------------------------------------------
# 前馈：与 encoder_decoder 的接缝
# --------------------------------------------------------------------------------------


def _ffn_parts(weights: object) -> tuple[Matrix, Vector, Matrix, Vector]:
    """从鸭式对象里取出四块前馈参数（缺字段抛 ParameterError，而不是 AttributeError）."""
    resolved = {}
    for name in ("w_in", "b_in", "w_out", "b_out"):
        value = getattr(weights, name, None)
        if value is None:
            raise ParameterError(
                "前馈的 weights 必须有 w_in / b_in / w_out / b_out 四个属性"
                "（本包的 FFNGradients.dict 与 encoder_decoder.FFNWeights 都满足）。"
            )
        resolved[name] = value
    return resolved["w_in"], resolved["b_in"], resolved["w_out"], resolved["b_out"]


def ffn_forward(
    inputs: Matrix, weights: object, *, activation: str = ACT_RELU
) -> tuple[Matrix, Matrix, Matrix]:
    """前馈的前向，返回 ``(输出, 激活前的值, 激活后的值)`` 三样.

    与 ``encoder_decoder.layers.feed_forward`` 的算术**逐字相同**（``math.fsum`` 累加 +
    ``+ bias``），因此第 7 条性质能逐位成立。
    """
    if activation not in FFN_ACTIVATIONS:
        raise ParameterError(
            f"前馈的激活只能是 {list(FFN_ACTIVATIONS)}，收到 {activation!r}："
            "它与 encoder_decoder 的两个激活一一对应。"
        )
    w_in, b_in, w_out, b_out = _ffn_parts(weights)
    if not inputs:
        raise ShapeError("ffn_forward 的输入不能为空。")
    if len(w_in[0]) != len(inputs[0]):
        raise ShapeError(
            f"W_in 的列数 {len(w_in[0])} 与输入列数 {len(inputs[0])} 不一致。"
        )
    pre_activation = dense_linear(w_in, b_in, inputs)
    from smart_research_agent.neural_basics.activations import activate

    hidden = tuple(activate(activation, row) for row in pre_activation)
    output = dense_linear(w_out, b_out, hidden)
    return output, pre_activation, hidden


def ffn_backward(
    inputs: Matrix, weights: object, *, activation: str = ACT_RELU, grad_output: Matrix
) -> FFNGradients:
    """前馈的反向（五块：``dW_in``、``db_in``、``dW_out``、``db_out``、``dx``）.

    五块全部是 :func:`backprop.layers.dense_backward` 与
    :func:`backprop.gradients.elementwise_backward` 的复合——这一层没有一条新式子。
    它与 ``encoder_decoder.layers.feed_forward_backward`` **逐位**一致（第 7 条性质）。
    """
    output, pre_activation, hidden = ffn_forward(inputs, weights, activation=activation)
    if len(grad_output) != len(output) or len(grad_output[0]) != len(output[0]):
        raise ShapeError(
            f"回传梯度 {len(grad_output)}×{len(grad_output[0])} 与前馈输出 "
            f"{len(output)}×{len(output[0])} 形状不一致。"
        )
    w_in, b_in, w_out, b_out = _ffn_parts(weights)
    out_grads = layer_ops.dense_backward(w_out, b_out, hidden, grad_output)
    grad_pre = elementwise_backward(activation, pre_activation, out_grads.grad_inputs)
    in_grads = layer_ops.dense_backward(w_in, b_in, inputs, grad_pre)
    return FFNGradients(
        grad_w_in=in_grads.grad_weight,
        grad_b_in=in_grads.grad_bias,
        grad_w_out=out_grads.grad_weight,
        grad_b_out=out_grads.grad_bias,
        grad_inputs=in_grads.grad_inputs,
    )


# --------------------------------------------------------------------------------------
# 参数与数值差分的接缝
# --------------------------------------------------------------------------------------


def spec_shapes(spec: MLPSpec) -> tuple[layer_ops.LayerShape, ...]:
    """一个 MLP 的形状表（与 :func:`backprop.layers.flatten_parameters` 同口径）."""
    return tuple(
        (layer.out_features, layer.in_features, layer.out_features) for layer in spec.layers
    )


def build_parameters(spec: MLPSpec) -> tuple[ParameterPair, ...]:
    """按 spec 造一串 ``(W, b)``（权重由 LCG 生成，因此**可复现**）."""
    return tuple(
        initialize(layer.init, layer.out_features, layer.in_features, seed=layer.seed)
        for layer in spec.layers
    )


def parameter_objective(
    spec: MLPSpec,
    inputs: Matrix,
    targets: Matrix,
    *,
    loss: str = LOSS_MSE,
) -> Callable[[Vector], float]:
    """把"压平的参数 → 一个标量损失"包成一个普通函数（数值差分用）.

    它存在的唯一理由，是让 :func:`math_foundations.calculus.gradient` 能作用在
    "整个 MLP 的参数"上——那一条路径**不依赖本包的任何推导**，
    因此它是这一课的"校准尺"。
    """
    if loss not in (LOSS_MSE, LOSS_CROSS_ENTROPY):
        raise ParameterError(f"未知的损失 {loss!r}：可选 {[LOSS_MSE, LOSS_CROSS_ENTROPY]}。")
    shapes = spec_shapes(spec)
    activations = tuple(layer.activation for layer in spec.layers)

    def objective(flat: Vector) -> float:
        params = layer_ops.unflatten_parameters(flat, shapes)
        chain = chain_from_parameters(params, activations)
        output = chain_forward(chain, inputs)
        if loss == LOSS_MSE:
            return mse_loss(output, targets)
        indexes = tuple(int(float(value)) for row in targets for value in row)
        return cross_entropy_loss(output[0], indexes[0])

    return objective


def gradient_norm(values: Sequence[float]) -> float:
    """一串梯度的欧氏范数（报告里的读数之一）."""
    return math.sqrt(math.fsum(float(value) * float(value) for value in values))


__all__ = [
    "FFN_ACTIVATIONS",
    "BackwardTrace",
    "ForwardCache",
    "activation_backward",
    "build_chain",
    "build_parameters",
    "chain_forward",
    "chain_forward_with_cache",
    "chain_from_parameters",
    "ffn_backward",
    "ffn_forward",
    "gradient_norm",
    "mlp_backward",
    "mlp_forward_with_cache",
    "mlp_loss_gradients",
    "parameter_objective",
    "softmax_loss_gradients",
    "spec_shapes",
]
