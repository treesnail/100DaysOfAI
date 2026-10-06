"""一个网络的前向，以及两条"结构事实"（day089 / M8-D1）.

## 一、MLP 前向：逐层 ``Dense → 激活``

```text
x → Dense_1 → act_1 → Dense_2 → act_2 → … → Dense_n →（可选 act_n）
```

末端是否带激活由 ``DenseSpec.activation`` 决定；``None`` 表示**恒等**（不激活）。
本模块只做前向——没有反向。

## 二、两条结构事实（都是可断言的性质，不是比喻）

```text
事实一  恒等激活的多层网络**塌缩成一个仿射映射**
        W = W_n·…·W_1，b = W_n·…·W_2·b_1 + … + b_n
        ⇒ mlp_forward(恒等 spec, x) 与 affine_forward(合成结果, x) 逐点相等（容差 1e-12）

事实二  前馈块 = Dense → 激活 → Dense
        ⇒ :func:`ffn_block` 接受项目里真实的 ``FFNWeights``（鸭式），
          与 ``encoder_decoder.layers.feed_forward`` 落在**同一个值**上
```

事实一是"为什么需要非线性"的**反证**：没有激活，深度不增加表达力。
事实二是"神经网络基础"与 M7（Transformer 底层）之间的**接缝**：
Transformer 的前馈块并不是新东西，它就是一个两层网络加一个激活。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from smart_research_agent.math_foundations.types import Matrix, Vector
from smart_research_agent.neural_basics.activations import activate
from smart_research_agent.neural_basics.errors import ParameterError, ShapeError
from smart_research_agent.neural_basics.layers import (
    Dense,
    affine_forward,
    compose_affine,
    dense_linear,
    initialize,
)
from smart_research_agent.neural_basics.types import (
    ACT_RELU,
    DenseSpec,
    ForwardTrace,
    MLPSpec,
    NeuronSpec,
)

#: 前馈的默认扩张比（Transformer 里 ``d_ff = 4·d``；写进常量让"4 倍"可断言）.
FFN_RATIO = 4


def neuron_forward(spec: NeuronSpec, inputs: Vector) -> float:
    """**一个神经元**的前向：``act(Σ w_i·x_i + b)``（输出是一个标量）."""
    if len(inputs) != spec.inputs:
        raise ShapeError(
            f"神经元的输入长度 {len(inputs)} 与 spec.inputs={spec.inputs} 不一致。"
        )
    layer = Dense.from_spec(
        DenseSpec(
            in_features=spec.inputs,
            out_features=1,
            activation=spec.activation,
            init=spec.init,
            seed=spec.seed,
        )
    )
    return layer.forward((tuple(inputs),))[0][0]


def mlp_forward(spec: MLPSpec, inputs: Matrix) -> Matrix:
    """逐层前向（每层 ``Dense → 激活``；``activation=None`` 表示恒等）."""
    if len(inputs) == 0:
        raise ShapeError("mlp_forward 的输入不能为空：至少要有 1 行样本。")
    if len(inputs[0]) != spec.input_width:
        raise ShapeError(
            f"输入的列数 {len(inputs[0])} 与网络的 input_width={spec.input_width} 不一致。"
        )
    current: Matrix = tuple(tuple(float(value) for value in row) for row in inputs)
    for index, layer_spec in enumerate(spec.layers):
        current = Dense.from_spec(layer_spec).forward(current)
        if any(not math.isfinite(value) for row in current for value in row):  # pragma: no cover
            raise ShapeError(
                f"第 {index + 1} 层输出了非有限值：前向在这一层失稳（通常是数值稳定没做对）。"
            )
    return current


def forward_trace(spec: MLPSpec) -> ForwardTrace:
    """逐层的宽度账（**不跑数据**，只把接口写下来）."""
    layer_widths = tuple((layer.in_features, layer.out_features) for layer in spec.layers)
    return ForwardTrace(input_width=spec.input_width, layer_widths=layer_widths)


def collapse_identity_mlp(spec: MLPSpec) -> tuple[Matrix, Vector]:
    """把**全恒等激活**的多层网络塌缩成一个仿射映射 ``(W, b)``.

    只要有任何一层带了激活（不是 ``None``），就抛 :class:`ParameterError`——
    因为"塌缩"这件事**只在仿射映射上成立**，一个 ReLU 会让它立刻失效。
    """
    for index, layer in enumerate(spec.layers):
        if layer.activation is not None:
            raise ParameterError(
                f"第 {index + 1} 层带了激活 {layer.activation!r}：多层网络只有"
                "**全恒等**时才塌缩成一个仿射映射。带上非线性之后，这条等式不再成立。"
            )
    pairs = tuple(
        (initialize(layer.init, layer.out_features, layer.in_features, seed=layer.seed))
        for layer in spec.layers
    )
    return compose_affine(pairs)


def identity_spec(spec: MLPSpec) -> MLPSpec:
    """把一份 spec 的所有激活换成恒等（用来构造塌缩实验）."""
    return MLPSpec(
        layers=tuple(
            DenseSpec(
                in_features=layer.in_features,
                out_features=layer.out_features,
                activation=None,
                init=layer.init,
                seed=layer.seed,
            )
            for layer in spec.layers
        )
    )


@dataclass(frozen=True)
class FFNParams:
    """前馈的四块参数（**与 ``encoder_decoder.FFNWeights`` 字段名逐一相同**，因此可互相传递）.

    ```text
    w_in   (ffn, hidden)    b_in   (ffn,)
    w_out  (hidden, ffn)    b_out  (hidden,)
    ```
    """

    w_in: Matrix
    b_in: Vector
    w_out: Matrix
    b_out: Vector

    def __post_init__(self) -> None:
        hidden = len(self.w_in[0]) if self.w_in else 0
        ffn = len(self.w_in)
        if len(self.w_out) != hidden:
            raise ShapeError(
                f"w_out 的行数 {len(self.w_out)} 与 w_in 的列数 {hidden} 不一致："
                "前馈是 (hidden → ffn → hidden) 的一个往返。"
            )
        if len(self.b_in) != ffn:
            raise ShapeError(f"b_in 长度 {len(self.b_in)} 与 ffn={ffn} 不一致。")
        if len(self.b_out) != hidden:
            raise ShapeError(f"b_out 长度 {len(self.b_out)} 与 hidden={hidden} 不一致。")


def build_ffn_params(hidden: int, ffn: int | None = None, *, seed: int = 0) -> FFNParams:
    """按 LCG + Xavier 造一份前馈参数（``ffn`` 默认 ``4·hidden``）."""
    if hidden < 1:
        raise ParameterError(f"hidden 必须为正整数，收到 {hidden}。")
    resolved_ffn = FFN_RATIO * hidden if ffn is None else ffn
    if resolved_ffn < 1:
        raise ParameterError(f"ffn 必须为正整数，收到 {resolved_ffn}。")
    w_in, b_in = initialize("xavier", resolved_ffn, hidden, seed=seed)
    w_out, b_out = initialize("xavier", hidden, resolved_ffn, seed=seed + 1)
    return FFNParams(w_in=w_in, b_in=b_in, w_out=w_out, b_out=b_out)


def ffn_block(inputs: Matrix, weights: object, *, activation: str = ACT_RELU) -> Matrix:
    """``Dense → 激活 → Dense``（Transformer 前馈块的最小复刻）.

    ``weights`` 只需**有** ``w_in / b_in / w_out / b_out`` 四个属性（鸭式），
    因此可以直接把项目里真实的 ``encoder_decoder.types.FFNWeights`` 传进来对账。
    """
    w_in = getattr(weights, "w_in", None)
    b_in = getattr(weights, "b_in", None)
    w_out = getattr(weights, "w_out", None)
    b_out = getattr(weights, "b_out", None)
    if w_in is None or b_in is None or w_out is None or b_out is None:
        raise ParameterError(
            "ffn_block 的 weights 必须有 w_in / b_in / w_out / b_out 四个属性"
            "（本包的 FFNParams 与 encoder_decoder.FFNWeights 都满足）。"
        )
    hidden = dense_linear(tuple(w_in), tuple(b_in), inputs)
    activated = tuple(activate(activation, row) for row in hidden)
    return dense_linear(tuple(w_out), tuple(b_out), activated)


def ffn_ratio(hidden: int, ffn: int) -> float:
    """``ffn / hidden``（Transformer 的默认取值是 4.0）."""
    if hidden < 1:
        raise ParameterError(f"hidden 必须为正整数，收到 {hidden}。")
    return ffn / hidden


__all__ = [
    "FFN_RATIO",
    "FFNParams",
    "affine_forward",
    "build_ffn_params",
    "collapse_identity_mlp",
    "ffn_block",
    "ffn_ratio",
    "forward_trace",
    "identity_spec",
    "mlp_forward",
    "neuron_forward",
]
