"""``transformer_stack`` 的链式前向与链式反向（day080 / M7-D5）.

## 今天新增的全部代码

```text
前向   一个 for：current = encoder_block(current)        五阶段（进入/块/记账/传递/出口）
反向   一个 for：grad    = encoder_block_backward(...)    从最后一层倒着走回第 0 层
记账   每一层记一行（‖x‖ / ‖y‖ / ‖b1‖ / ‖b2‖ / 最大绝对值）
```

三段都只有几行。**今天真正新增的判断只有两处**，而它们都藏在“看起来平凡”的地方：

```text
① 传递的是什么        把 y_i **原样**交给第 i+1 层，而不是重算一遍
                      （重算一遍在数值上会差 1e-16 量级，而它会让“逐位一致”失效）
② 反向从哪里接        第 i 层拿到的 grad_output 必须是第 i+1 层的 grad_inputs，
                      而**不是**“第 i+1 层某一支路的梯度” —— 这个错法形状全对
```

## 一个必须写下来的边界：链式反向**不是**把每一层单独反传一遍

```text
对的     grad = dLoss/dy_N
         for i in reversed(range(N)):
             grads[i] = block_backward(block_i, grad)     ← 用上一轮的结果
             grad     = grads[i].grad_inputs              ← 换上手
错的     for i in reversed(range(N)):
             grads[i] = block_backward(block_i, dLoss/dy_N)  ← 每一层都从同一点起步
```

错的那一版**形状全对、也真的会跑**，只是第 0 层收到的梯度只反映了最后一层的影响。
本课用一条数值校验把它钉住：把 ``stack_backward`` 的 ``grad_inputs``
与整条链损失的**数值梯度**逐项比（第 6 章）。

## 为什么“逐层读数”要用 ``math.fsum``

一层读数是一个平方和；链长了以后普通求和会累积出 ``1e-13`` 量级的偏差。
``math.fsum`` 是精确求和，代价是一次额外的遍历——而今天的读数是要拿去
**跨层比较**的（“第几层开始塌”），同一条链上两个数的口径必须完全一致。
"""

from __future__ import annotations

import math
from typing import Sequence

from smart_research_agent.encoder_decoder.depth import make_block_parameters
from smart_research_agent.encoder_decoder.layers import (
    block_attention,
    encoder_block,
    encoder_block_backward,
)
from smart_research_agent.encoder_decoder.types import (
    ACTIVATION_RELU,
    NORM_PRE,
    BlockGradients,
    BlockParameters,
    FFNWeights,
    _checked_activation,
    _checked_placement,
)
from smart_research_agent.math_foundations.types import (
    Matrix,
    matrix_shape,
    validate_matrix,
)
from smart_research_agent.transformer_core.layers import mean_squared_error, mse_gradient
from smart_research_agent.transformer_core.train import default_parameters
from smart_research_agent.transformer_core.types import AttentionParams
from smart_research_agent.transformer_stack.errors import AssemblyError, ParameterError
from smart_research_agent.transformer_stack.types import (
    DEFAULT_INIT_SCALE,
    LayerCensus,
    StackForward,
    StackGradients,
    StackParameters,
    StackShape,
    StackedLayer,
    _checked_layer_index,
    _checked_layers,
    _checked_scale,
    frobenius,
    max_absolute,
)

#: 逐层参数的种子步长（与 day079 的深度实验同值）：第 i 层用 ``seed + i*13``.
LAYER_SEED_STRIDE = 13
#: 逐层注意力的种子步长（day079 用 17）：两份参数各自走各自的步长.
ATTENTION_SEED_STRIDE = 17
#: 默认的层数（四层：够看出“逐层单调”，又不至于让演示跑太久）.
DEFAULT_LAYERS = 4
#: 默认的隐藏维与序列长度（与 day073~079 的样本同值，便于跨天对照）.
DEFAULT_HIDDEN = 6
#: 默认的前馈倍数（原论文的取值）.
DEFAULT_FFN_RATIO = 4
#: 默认的序列长度.
DEFAULT_TOKENS = 4


def make_shape(
    *,
    layers: int = DEFAULT_LAYERS,
    hidden: int = DEFAULT_HIDDEN,
    tokens: int = DEFAULT_TOKENS,
    ffn_ratio: int = DEFAULT_FFN_RATIO,
) -> StackShape:
    """按“层数 + 隐藏维 + 序列长度 + 前馈倍数”造一个形状（``ffn = d × ratio``）."""
    if isinstance(ffn_ratio, bool) or not isinstance(ffn_ratio, int) or ffn_ratio < 1:
        raise ParameterError(f"ffn_ratio 必须是 >= 1 的整数，收到 {ffn_ratio!r}。")
    return StackShape(
        hidden=hidden,
        ffn=hidden * ffn_ratio,
        tokens=tokens,
        layers=layers,
    )


def make_stack_parameters(
    shape: StackShape,
    *,
    seed: int = 7,
    scale: float = DEFAULT_INIT_SCALE,
) -> StackParameters:
    """一摞块各自的参数：**第 i 层用第 i 份种子**（避免“层与层完全一样”）.

    两份参数各走各的步长（块 13、注意力 17），因此“第 i 层的块参数”与
    “第 i 层的注意力参数”不会因为种子巧合而彼此相关——而这一点在深度实验里是必要的：
    一旦两层完全一样，``‖dx‖`` 的逐层曲线就会变成一条“周期 1”的序列。
    """
    resolved_scale = _checked_scale(scale)
    layers = _checked_layers(shape.layers)
    blocks = tuple(
        make_block_parameters(
            shape.block_shape, seed=seed + index * LAYER_SEED_STRIDE, scale=resolved_scale
        )
        for index in range(layers)
    )
    attentions = tuple(
        default_parameters(
            shape.hidden, seed=seed + index * ATTENTION_SEED_STRIDE, scale=resolved_scale
        )
        for index in range(layers)
    )
    return StackParameters(blocks=blocks, attentions=attentions)


def stack_forward(
    params: StackParameters,
    inputs: Matrix,
    *,
    placement: str = NORM_PRE,
    use_residual: bool = True,
    activation: str = ACTIVATION_RELU,
) -> StackForward:
    """前向穿过整条链，并把**每一层**的账与读数都留下来（五阶段）.

    ``current`` 就是“链上此刻的那份张量”：它被交给下一层的 ``encoder_block``，
    因此第 i+1 层的 ``inputs`` 与第 i 层的 ``output`` **逐位相等**——
    这条性质由 :class:`StackForward` 的构造检查逐层核对（用 ``==``）。

    > 注意本包只承诺**逐位相等**，不承诺“同一个对象”：记录在构造时会过一遍
    > ``validate_matrix``，而它返回的是一份**重建**出来的等值元组。
    > 把 ``is`` 当成判据会在这个细节上亮红——而那不是 bug。
    """
    resolved_placement = _checked_placement(placement)
    resolved_activation = _checked_activation(activation)
    checked = validate_matrix(inputs, name="inputs")
    if matrix_shape(checked)[1] != params.blocks[0].hidden:
        raise ParameterError(
            f"输入宽度 {matrix_shape(checked)[1]} 与块的隐藏维 "
            f"{params.blocks[0].hidden} 不一致。"
        )
    layers: list[StackedLayer] = []
    censuses: list[LayerCensus] = []
    current = checked
    for index, (block_params, attention_params) in enumerate(
        zip(params.blocks, params.attentions, strict=True)
    ):
        attention = block_attention(
            block_params, current, attention_params, placement=resolved_placement
        )
        forward = encoder_block(
            block_params,
            current,
            attention,
            placement=resolved_placement,
            use_residual=use_residual,
            activation=resolved_activation,
        )
        layers.append(
            StackedLayer(
                index=index, block=forward, attention=attention, params=block_params
            )
        )
        censuses.append(
            LayerCensus(
                index=index,
                input_norm=frobenius(current),
                output_norm=frobenius(forward.output),
                branch1_norm=frobenius(forward.branch1),
                branch2_norm=frobenius(forward.branch2),
                max_abs_input=max_absolute(current),
                use_residual=use_residual,
            )
        )
        current = forward.output
    return StackForward(
        shape=params.shape_for(matrix_shape(checked)[0]),
        placement=resolved_placement,
        use_residual=use_residual,
        activation=resolved_activation,
        inputs=checked,
        output=current,
        layers=tuple(layers),
        censuses=tuple(censuses),
        notes=(
            "逐层读数里的范数用 math.fsum 求和：跨层比较的两个数必须同口径",
            "直通占比 = ‖x_i‖ / (‖x_i‖ + ‖b1‖ + ‖b2‖)：残差关时它恰好是 0",
        ),
    )


def stack_loss(
    params: StackParameters,
    inputs: Matrix,
    target: Matrix,
    *,
    placement: str = NORM_PRE,
    use_residual: bool = True,
    activation: str = ACTIVATION_RELU,
) -> float:
    """整条链上的 MSE（与 ``encoder_block`` 用**同一个**损失函数）."""
    forward = stack_forward(
        params,
        inputs,
        placement=placement,
        use_residual=use_residual,
        activation=activation,
    )
    return mean_squared_error(forward.output, target)


def stack_backward(
    forward: StackForward,
    params: StackParameters,
    grad_output: Matrix,
    *,
    activation: str = ACTIVATION_RELU,
) -> StackGradients:
    """反向穿过整条链：**从最后一层倒着走**，每一环的输入梯度就是上一环的输出梯度.

    这一趟是今天唯一的新代码，而它只有三行要紧的：

    ```text
    layer_grads = encoder_block_backward(账, 参数, grad, activation=...)
    grads[index] = layer_grads
    grad = layer_grads.grad_inputs        ← 换上手（day079 的深度实验在这里也用了一行）
    ```

    第二行与第三行的**顺序**要看清：先存账（按层号），再换手——
    而“换手”这件事一旦漏掉，就退化成“每一层都从同一个点起步”的那种错法。
    """
    resolved_activation = _checked_activation(activation)
    if forward.depth != params.layers:
        raise AssemblyError(
            f"账里有 {forward.depth} 层，而参数有 {params.layers} 份："
            "反向必须与产生这份账的那次前向用同一摞参数。"
        )
    checked_grad = validate_matrix(grad_output, name="grad_output")
    grads: list[BlockGradients | None] = [None] * forward.depth
    current = checked_grad
    for index in reversed(range(forward.depth)):
        layer = forward.layer_at(index)
        layer_grads = encoder_block_backward(
            layer.block, layer.params, current, activation=resolved_activation
        )
        grads[index] = layer_grads
        current = layer_grads.grad_inputs
    resolved: list[BlockGradients] = []
    for index, item in enumerate(grads):
        if item is None:
            raise ParameterError(f"第 {index} 层的梯度没有被算到：反向的循环没有走完。")
        resolved.append(item)
    return StackGradients(
        shape=forward.shape,
        grad_inputs=resolved[0].grad_inputs,
        layers=tuple(resolved),
        notes=(
            "layers 与 StackForward.layers 同序（下标即层号），而**计算**顺序是倒着的",
            "grad_inputs 恒等于第 0 层的输入梯度——这一条由 StackGradients 的构造检查保证",
        ),
    )


def stack_loss_gradient(
    params: StackParameters,
    inputs: Matrix,
    target: Matrix,
    *,
    placement: str = NORM_PRE,
    use_residual: bool = True,
    activation: str = ACTIVATION_RELU,
) -> tuple[StackForward, StackGradients]:
    """前向 + 反向一条龙（读数与梯度都拿到），供演示与校验使用."""
    forward = stack_forward(
        params,
        inputs,
        placement=placement,
        use_residual=use_residual,
        activation=activation,
    )
    grads = stack_backward(
        forward,
        params,
        mse_gradient(forward.output, target),
        activation=activation,
    )
    return forward, grads


def stack_gradient_profile(
    params: StackParameters,
    inputs: Matrix,
    target: Matrix,
    *,
    placement: str = NORM_PRE,
    use_residual: bool = True,
    activation: str = ACTIVATION_RELU,
) -> tuple[float, ...]:
    """逐层入口梯度的范数（按层号顺序，**含第 0 层**）.

    今天把 day079 只看过的那一个数变成了一个序列——而“梯度在第几层开始塌”
    只有在一个序列上才有意义。
    """
    _forward, grads = stack_loss_gradient(
        params,
        inputs,
        target,
        placement=placement,
        use_residual=use_residual,
        activation=activation,
    )
    return grads.norms()


def zero_branches(params: BlockParameters) -> BlockParameters:
    """把**一个块的两个分支**都置零（前馈的两个输出层参数归零）.

    只把 ``w_out`` 置零**不够**：残差会把 ``inputs + 注意力输出`` 传下去，
    而注意力那一支并不在这一层的参数里（它来自 day075 的那份账）。
    因此“块退化成恒等”这条性质必须在 ``w_out`` 归零**并且**注意力输出为零
    的配置下才成立——本函数只负责前者，后者由调用方把注意力参数也置零。
    """
    if not isinstance(params, BlockParameters):
        raise ParameterError(f"params 必须是 BlockParameters，收到 {type(params).__name__}。")
    zero_row = tuple(0.0 for _ in range(matrix_shape(params.ffn_w_out)[1]))
    zero_out = tuple(zero_row for _ in range(matrix_shape(params.ffn_w_out)[0]))
    return params.with_ffn(
        FFNWeights(
            w_in=params.ffn_w_in,
            b_in=params.ffn_b_in,
            w_out=zero_out,
            b_out=tuple(0.0 for _ in params.ffn_b_out),
        )
    )


def zero_attention(params: AttentionParams) -> AttentionParams:
    """把注意力的四个投影全部置零（于是 ``self_attention`` 的输出恒为零矩阵）."""
    def blank(matrix: Matrix) -> Matrix:
        rows, columns = matrix_shape(matrix)
        return tuple(tuple(0.0 for _ in range(columns)) for _ in range(rows))

    return AttentionParams(
        w_query=blank(params.w_query),
        w_key=blank(params.w_key),
        w_value=blank(params.w_value),
        w_output=blank(params.w_output),
    )


def flatten_layer_parameter_norms(params: StackParameters) -> tuple[float, ...]:
    """逐层**参数**的 Frobenius 范数（一张便宜的“每层有多少东西可学”的表）."""
    norms: list[float] = []
    for block in params.blocks:
        total = math.fsum(
            value * value for matrix in block.matrices() for row in matrix for value in row
        )
        norms.append(math.sqrt(total))
    return tuple(norms)


def layer_census_of(forward: StackForward, index: int) -> LayerCensus:
    """取第 ``index`` 层的读数（转发 ``StackForward.census_at``，越界当场报错）."""
    return forward.census_at(_checked_layer_index(index, forward.depth))


def sequence_slice(values: Sequence[float], index: int) -> float:
    """按层号取一个序列里的读数（供演示脚本按层打印时使用）."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise ParameterError(f"下标必须是整数，收到 {index!r}。")
    if not 0 <= index < len(values):
        raise ParameterError(f"下标必须落在 [0, {len(values)})，收到 {index}。")
    return float(values[index])


__all__ = [
    "ATTENTION_SEED_STRIDE",
    "DEFAULT_FFN_RATIO",
    "DEFAULT_HIDDEN",
    "DEFAULT_LAYERS",
    "DEFAULT_TOKENS",
    "LAYER_SEED_STRIDE",
    "flatten_layer_parameter_norms",
    "layer_census_of",
    "make_shape",
    "make_stack_parameters",
    "sequence_slice",
    "stack_backward",
    "stack_forward",
    "stack_gradient_profile",
    "stack_loss",
    "stack_loss_gradient",
    "zero_attention",
    "zero_branches",
]
