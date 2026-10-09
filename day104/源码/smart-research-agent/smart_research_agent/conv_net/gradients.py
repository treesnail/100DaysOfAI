"""``gradients``：卷积与池化的**反向**（day093 / M8-D4）.

```text
conv2d_backward      单通道：d_kernel[u,v] = Σ grad_out[i,j]·x_pad[i·s+u·d, j·s+v·d]
                               d_image       = 把 grad_out 按核散布回每个输入位置（再去掉补边）
conv_block_backward  多通道：d_bias、d_kernels[o][i]，以及把所有 o 的贡献在 d_image[i] 上**累加**
pool2d_backward      max 把梯度给到窗口的最大值那一格；avg 把梯度均分给窗口内每一格
```

## 三块梯度各自的形状口诀

```text
d_kernel   形状与 kernel **相同**：它是"每个核元素被用到的次数"的加权和
d_bias     形状与 bias 相同（标量）：它是"这个输出通道被用到的总强度"
d_image    形状与 image **相同**：它是"每个输入位置对损失的贡献"（所有输出位置累加）
```

这三块最容易互相写串的地方是 **d_kernel 与 d_image 的索引**：前者把 `grad_out` 当权重、
把输入当被求和项；后者反过来。第 ⑤ 条性质用数值差分把两者一起钉住。

## 激活的反向导数从哪来

卷积块里的激活不是新东西：``relu`` / ``gelu`` / ``sigmoid`` / ``tanh`` 的导数
**只有一份**，写在 day090 的 ``backprop.gradients``。本模块直接调用
:func:`backprop.gradients.elementwise_backward`，不重写任何一条导数公式。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from smart_research_agent.backprop import gradients as backprop_gradients
from smart_research_agent.conv_net.errors import (
    BackwardError,
    ParameterError,
    ShapeError,
    WindowError,
)
from smart_research_agent.conv_net.layers import ConvParams, ConvSpec
from smart_research_agent.conv_net.ops import (
    as_matrix,
    conv2d,
    output_size,
    pad2d,
    pool2d,
    resolve_padding,
)
from smart_research_agent.conv_net.types import POOL_AVG, POOL_MAX, POOL_MODES
from smart_research_agent.math_foundations.types import Matrix, Vector


@dataclass(frozen=True)
class ConvGradients:
    """一层卷积的三块梯度：``kernel_grads[o][i]``、``bias_grad``、``input_grads``."""

    kernel_grads: tuple[tuple[Matrix, ...], ...]
    bias_grad: Vector
    input_grads: tuple[Matrix, ...]

    def kernel_norm(self) -> float:
        """所有核梯度的整体范数（"这一层的权重被推动了多少"）."""
        return math.sqrt(
            math.fsum(
                value * value
                for per_output in self.kernel_grads
                for kernel in per_output
                for row in kernel
                for value in row
            )
        )

    def input_norm(self) -> float:
        """输入梯度的整体范数（"传回去多少"）."""
        return math.sqrt(
            math.fsum(value * value for channel in self.input_grads for row in channel for value in row)
        )


def conv2d_backward(
    image: Matrix,
    kernel: Matrix,
    grad_output: Matrix,
    *,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> tuple[Matrix, Matrix]:
    """单通道卷积的反向，返回 ``(d_kernel, d_image)``（输入是**未补边**的原图）.

    实现方式是最朴素的"按定义累加"：不用 im2col、不做向量化——这一课要的是
    **每一个索引都对得上**，而不是速度。
    """
    checked_image = as_matrix(image, name="image")
    checked_kernel = as_matrix(kernel, name="kernel")
    checked_grad = as_matrix(grad_output, name="grad_output")
    height, width = len(checked_image), len(checked_image[0])
    kernel_h, kernel_w = len(checked_kernel), len(checked_kernel[0])
    expected_h = output_size(height, kernel_h, stride=stride, padding=padding, dilation=dilation)
    expected_w = output_size(width, kernel_w, stride=stride, padding=padding, dilation=dilation)
    if (len(checked_grad), len(checked_grad[0])) != (expected_h, expected_w):
        raise BackwardError(
            f"grad_output 的形状 {(len(checked_grad), len(checked_grad[0]))} 与按参数推出的 "
            f"{(expected_h, expected_w)} 不一致：反向的梯度必须与**同一次**前向的输出配对。"
        )
    source = pad2d(checked_image, padding) if padding else checked_image
    d_kernel = [[0.0] * kernel_w for _ in range(kernel_h)]
    d_padded = [[0.0] * (width + 2 * padding) for _ in range(height + 2 * padding)]
    for i in range(expected_h):
        for j in range(expected_w):
            weight = checked_grad[i][j]
            for u in range(kernel_h):
                for v in range(kernel_w):
                    d_kernel[u][v] += weight * source[i * stride + u * dilation][j * stride + v * dilation]
                    d_padded[i * stride + u * dilation][j * stride + v * dilation] += weight * checked_kernel[u][v]
    d_image = (
        tuple(tuple(d_padded[row][column] for column in range(padding, padding + width)) for row in range(padding, padding + height))
        if padding
        else tuple(tuple(row) for row in d_padded)
    )
    return tuple(tuple(row) for row in d_kernel), d_image


def conv_block_backward(
    spec: ConvSpec,
    params: ConvParams,
    image_channels: tuple[Matrix, ...],
    grad_outputs: tuple[Matrix, ...],
    *,
    pre_activations: tuple[Matrix, ...] | None = None,
) -> ConvGradients:
    """一层多通道卷积的反向（含激活的反向导数与通道维的累加）.

    ``pre_activations`` 是前向时**激活之前**的特征图；当激活非恒等而它缺失时抛
    :class:`BackwardError`——因为 ``relu`` 的导数必须知道"这一格原来是正是负"。
    """
    if len(grad_outputs) != params.out_channels:
        raise BackwardError(
            f"grad_outputs 有 {len(grad_outputs)} 个通道，而这一层输出 {params.out_channels} 个："
            "反向的通道数必须与前向的输出通道数一致。"
        )
    if spec.activation is not None:
        if pre_activations is None:
            raise BackwardError(
                f"激活 {spec.activation!r} 的反向需要 pre_activations（激活之前的值）："
                "relu 的导数必须知道这一格原来是正是负，缺了就重跑一次前向。"
            )
        if len(pre_activations) != params.out_channels:
            raise BackwardError("pre_activations 的通道数与输出通道数不一致。")
        d_pre = tuple(
            backprop_gradients.elementwise_backward(
                spec.activation, as_matrix(pre, name=f"pre[{i}]"), as_matrix(grad, name=f"grad[{i}]")
            )
            for i, (pre, grad) in enumerate(zip(pre_activations, grad_outputs))
        )
    else:
        d_pre = tuple(as_matrix(grad, name=f"grad[{i}]") for i, grad in enumerate(grad_outputs))

    checked_channels = tuple(as_matrix(channel, name=f"image[{i}]") for i, channel in enumerate(image_channels))
    if len(checked_channels) != params.in_channels:
        raise ShapeError(
            f"输入有 {len(checked_channels)} 个通道，而这一层有 {params.in_channels} 个输入核。"
        )
    pad_h = resolve_padding(
        len(checked_channels[0]), params.kernel_size[0], stride=spec.stride, padding=spec.padding, dilation=spec.dilation
    )
    kernel_grads: list[list[Matrix]] = [[None] * params.in_channels for _ in range(params.out_channels)]  # type: ignore[list-item]
    bias_grad: list[float] = []
    input_grads: list[Matrix] = [
        tuple(tuple(0.0 for _ in row) for row in channel) for channel in checked_channels
    ]
    for out_index in range(params.out_channels):
        bias_grad.append(math.fsum(value for row in d_pre[out_index] for value in row))
        for in_index in range(params.in_channels):
            d_kernel, d_image = conv2d_backward(
                checked_channels[in_index],
                params.kernels[out_index][in_index],
                d_pre[out_index],
                stride=spec.stride,
                padding=pad_h,
                dilation=spec.dilation,
            )
            kernel_grads[out_index][in_index] = d_kernel
            input_grads[in_index] = tuple(
                tuple(a + b for a, b in zip(row_a, row_b))
                for row_a, row_b in zip(input_grads[in_index], d_image)
            )
    return ConvGradients(
        kernel_grads=tuple(tuple(per_input) for per_input in kernel_grads),  # type: ignore[arg-type]
        bias_grad=tuple(bias_grad),
        input_grads=tuple(input_grads),
    )


def pool2d_backward(
    image: Matrix,
    grad_output: Matrix,
    *,
    mode: str = POOL_MAX,
    window: int = 2,
    stride: int | None = None,
) -> Matrix:
    """池化的反向：``max`` 把梯度给到**窗口最大值那一格**，``avg`` 均分给窗口内每一格.

    这里从 ``image`` **重新算一次前向**来确定 max 的位置，并在形状对不上时抛
    :class:`BackwardError`——而不是相信调用方传进来的下标。
    """
    checked_image = as_matrix(image, name="image")
    checked_grad = as_matrix(grad_output, name="grad_output")
    if mode not in POOL_MODES:
        raise ParameterError(f"未知的池化模式 {mode!r}：可用取值 {list(POOL_MODES)}。")
    forward = pool2d(checked_image, mode=mode, window=window, stride=stride)
    if (len(forward), len(forward[0])) != (len(checked_grad), len(checked_grad[0])):
        raise BackwardError(
            f"grad_output 的形状 {(len(checked_grad), len(checked_grad[0]))} 与池化输出的形状 "
            f"{(len(forward), len(forward[0]))} 不一致：反向的梯度必须与同一次前向配对。"
        )
    step = window if stride is None else stride
    height, width = len(checked_image), len(checked_image[0])
    d_image = [[0.0] * width for _ in range(height)]
    for i in range(len(forward)):
        for j in range(len(forward[0])):
            weight = checked_grad[i][j]
            cells = [
                (i * step + u, j * step + v) for u in range(window) for v in range(window)
            ]
            if mode == POOL_MAX:
                best = None
                for row, column in cells:
                    value = checked_image[row][column]
                    if best is None or value > best[0]:
                        best = (value, row, column)
                assert best is not None
                d_image[best[1]][best[2]] += weight
            else:
                share = weight / len(cells)
                for row, column in cells:
                    d_image[row][column] += share
    return tuple(tuple(row) for row in d_image)


def avg_pool_backward(image: Matrix, grad_output: Matrix, window: int = 2, stride: int | None = None) -> Matrix:
    """平均池化的反向（等价于 ``pool2d_backward(mode="avg")``）."""
    return pool2d_backward(image, grad_output, mode=POOL_AVG, window=window, stride=stride)


def max_pool_backward(image: Matrix, grad_output: Matrix, window: int = 2, stride: int | None = None) -> Matrix:
    """最大池化的反向（等价于 ``pool2d_backward(mode="max")``）."""
    return pool2d_backward(image, grad_output, mode=POOL_MAX, window=window, stride=stride)


def conv_forward_with_cache(
    spec: ConvSpec, params: ConvParams, image_channels: tuple[Matrix, ...]
) -> tuple[tuple[Matrix, ...], tuple[Matrix, ...]]:
    """前向并返回 ``(激活后的输出, 激活之前的中间量)``——反向需要后者.

    它在数值上与 ``layers.conv_block_forward`` 完全一致（只是多留了一份中间量），
    因此"前向只写一遍"这件事仍然成立：激活之前的中间量由同一段滑窗逻辑产出。
    """
    checked_channels = tuple(as_matrix(channel, name=f"image[{i}]") for i, channel in enumerate(image_channels))
    if len(checked_channels) != params.in_channels:
        raise ShapeError(
            f"输入有 {len(checked_channels)} 个通道，而这一层有 {params.in_channels} 个输入核。"
        )
    pre_activations: list[Matrix] = []
    for out_index, kernels in enumerate(params.kernels):
        accumulated = None
        for channel, kernel in zip(checked_channels, kernels):
            partial = conv2d(
                channel, kernel, stride=spec.stride, padding=spec.padding, dilation=spec.dilation
            )
            if accumulated is None:
                accumulated = partial
            else:
                accumulated = tuple(
                    tuple(a + b for a, b in zip(row_a, row_b))
                    for row_a, row_b in zip(accumulated, partial)
                )
        assert accumulated is not None
        pre_activations.append(
            tuple(tuple(value + params.bias[out_index] for value in row) for row in accumulated)
        )
    from smart_research_agent.conv_net.layers import activate_map

    outputs = tuple(activate_map(spec.activation, pre) for pre in pre_activations)
    return outputs, tuple(pre_activations)


__all__ = [
    "ConvGradients",
    "avg_pool_backward",
    "conv2d_backward",
    "conv_block_backward",
    "conv_forward_with_cache",
    "max_pool_backward",
    "pool2d_backward",
]
