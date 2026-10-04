"""``layers``：一层卷积 / 一层池化的**可复现实现**（day093 / M8-D4）.

```text
ConvSpec        一层卷积的画像：通道数 / 核 / 步长 / 填充 / 膨胀 / 激活 / 种子
initialize_conv 用 day089 的 LCG 造一组核（**同一份 spec 两次调用逐位相同**）
ConvParams      参数本体：kernels[out][in] 是一张 k×k 的核，bias 长度为 out_channels
conv_block_forward  多通道卷积：先在**通道维**加权求和，再加偏置、过激活
pool_block_forward  逐通道池化（降采样）
```

## 多通道卷积到底在做什么

```text
对每一个输出通道 c_out：
    y = Σ_{c_in} conv2d(x[c_in], K[c_out][c_in]) + b[c_out]
    y = activation(y)
```

"通道维上的加权求和"就是一次**全连接式的混合**：`C_in` 个通道 → 1 个通道。
这也是为什么 ``conv2d`` 只做单通道——多通道只是把单通道的结果按不同的核加起来。

## 初始化沿用 day089 的 LCG

卷积核与全连接权重一样是"要被学的东西"，因此初始化必须**可复现**：
``initialize_conv`` 复用 ``neural_basics.layers.lcg_stream``，尺度取
``U(−1/√fan_in, 1/√fan_in)``，其中 ``fan_in = C_in·k_h·k_w``——
每一张核的 fan_in 与"它与输入相连的元素个数"一致。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from smart_research_agent.conv_net.errors import (
    ParameterError,
    ShapeError,
    WindowError,
)
from smart_research_agent.conv_net.ops import (
    as_matrix,
    checked_positive_int,
    conv2d,
    pool2d,
)
from smart_research_agent.conv_net.types import (
    CONV_ACTIVATIONS,
    CONV_PARAM_FORMULA,
    PADDING_MODES,
    PADDING_VALID,
    POOL_MAX,
)
from smart_research_agent.math_foundations.types import Matrix, Vector
from smart_research_agent.neural_basics.activations import activate
from smart_research_agent.neural_basics.layers import lcg_stream


@dataclass(frozen=True)
class ConvSpec:
    """一层二维卷积的画像（**激活用 ``None`` 表示恒等**，与 day089 一致）."""

    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: str | int = PADDING_VALID
    dilation: int = 1
    activation: str | None = "relu"
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("in_channels", "out_channels", "kernel_size"):
            object.__setattr__(self, name, checked_positive_int(getattr(self, name), name=name))
        object.__setattr__(self, "stride", checked_positive_int(self.stride, name="stride"))
        object.__setattr__(self, "dilation", checked_positive_int(self.dilation, name="dilation"))
        if isinstance(self.padding, int):
            if self.padding < 0:
                raise ParameterError(f"整数 padding 必须非负，收到 {self.padding}。")
        elif self.padding not in PADDING_MODES:
            raise ParameterError(
                f"未知的填充模式 {self.padding!r}：可用取值 {list(PADDING_MODES)}。"
            )
        if self.activation is not None and self.activation not in CONV_ACTIVATIONS:
            raise ParameterError(
                f"未知的卷积激活 {self.activation!r}：可用取值 {list(CONV_ACTIVATIONS)}"
                "（恒等请传 None）。"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ParameterError(f"seed 必须是非负整数，收到 {self.seed!r}。")

    @property
    def fan_in(self) -> int:
        """每张核相连的输入元素个数 ``C_in·k_h·k_w``."""
        return self.in_channels * self.kernel_size * self.kernel_size

    @property
    def parameter_count(self) -> int:
        """参数量 ``C_out·(C_in·k²) + C_out``."""
        return self.out_channels * self.fan_in + self.out_channels

    def line(self) -> str:
        """一行说明：``conv(1→2, k=3, s=1, pad=valid, act=relu) | 参数 20``."""
        act = self.activation if self.activation is not None else "identity"
        return (
            f"conv({self.in_channels}→{self.out_channels}, k={self.kernel_size}, "
            f"s={self.stride}, pad={self.padding}, act={act}) | 参数 {self.parameter_count}"
        )


@dataclass(frozen=True)
class ConvParams:
    """一层卷积的参数：``kernels[out][in]`` 是一张 ``k×k`` 核，``bias`` 长度为 out_channels.

    形状约定与 ``torch.nn.Conv2d`` 的 ``weight (C_out, C_in, k_h, k_w)`` 一致，
    只是本仓库把最后两维放进了 ``Matrix``：``kernels[o][i]`` 就是 ``weight[o, i]``。
    """

    kernels: tuple[tuple[Matrix, ...], ...]
    bias: Vector

    def __post_init__(self) -> None:
        if not self.kernels or not self.kernels[0]:
            raise ShapeError("ConvParams 至少要有一个输出通道、每个输出通道至少要有一个输入核。")
        checked: list[tuple[Matrix, ...]] = []
        for out_index, kernels in enumerate(self.kernels):
            checked_kernels = tuple(as_matrix(kernel, name=f"kernels[{out_index}][{i}]") for i, kernel in enumerate(kernels))
            shape = (len(checked_kernels[0]), len(checked_kernels[0][0]))
            for i, kernel in enumerate(checked_kernels):
                if (len(kernel), len(kernel[0])) != shape:
                    raise ShapeError(
                        f"第 {out_index} 个输出通道里的核形状不齐：核 {i} 是 "
                        f"{len(kernel)}×{len(kernel[0])}，而首核是 {shape[0]}×{shape[1]}。"
                    )
            checked.append(checked_kernels)
        in_channels = len(checked[0])
        for out_index, kernels in enumerate(checked):
            if len(kernels) != in_channels:
                raise ShapeError(
                    f"第 {out_index} 个输出通道有 {len(kernels)} 张核，而首通道有 {in_channels} 张："
                    "每个输出通道都必须对全部输入通道各有一张核。"
                )
        if len(self.bias) != len(checked):
            raise ShapeError(
                f"bias 长度 {len(self.bias)} 与输出通道数 {len(checked)} 不一致。"
            )
        object.__setattr__(self, "kernels", tuple(checked))
        object.__setattr__(self, "bias", tuple(float(value) for value in self.bias))

    @property
    def out_channels(self) -> int:
        """输出通道数."""
        return len(self.kernels)

    @property
    def in_channels(self) -> int:
        """输入通道数."""
        return len(self.kernels[0])

    @property
    def kernel_size(self) -> tuple[int, int]:
        """核的空间尺寸 ``(k_h, k_w)``."""
        return len(self.kernels[0][0]), len(self.kernels[0][0][0])

    @property
    def parameter_count(self) -> int:
        """参数量 = 所有核的元素个数 + 偏置个数."""
        total = sum(len(kernel) * len(kernel[0]) for kernels in self.kernels for kernel in kernels)
        return total + len(self.bias)


def initialize_conv(spec: ConvSpec) -> ConvParams:
    """按 spec 造一组核（LCG + 可注入种子 ⇒ **逐位可复现**）."""
    if not isinstance(spec, ConvSpec):
        raise ParameterError(f"initialize_conv 需要一个 ConvSpec，收到 {type(spec).__name__}。")
    total = spec.out_channels * spec.in_channels * spec.kernel_size * spec.kernel_size
    limit = 1.0 / math.sqrt(spec.fan_in)
    uniform = lcg_stream(spec.seed, total)
    bias_uniform = lcg_stream(spec.seed + 1, spec.out_channels)
    kernels: list[tuple[Matrix, ...]] = []
    cursor = 0
    for _ in range(spec.out_channels):
        per_input: list[Matrix] = []
        for _ in range(spec.in_channels):
            rows: list[tuple[float, ...]] = []
            for _ in range(spec.kernel_size):
                row = tuple(
                    uniform[cursor + column] * 2.0 * limit - limit
                    for column in range(spec.kernel_size)
                )
                cursor += spec.kernel_size
                rows.append(row)
            per_input.append(tuple(rows))
        kernels.append(tuple(per_input))
    bias = tuple(value * 2.0 * limit - limit for value in bias_uniform)
    return ConvParams(kernels=tuple(kernels), bias=bias)


def activate_map(name: str | None, image: Matrix) -> Matrix:
    """对特征图逐行过激活（``None`` 表示恒等，**不改变空间结构**）."""
    if name is None:
        return image
    return tuple(activate(name, row) for row in image)


def conv_block_forward(params: ConvParams, image_channels: tuple[Matrix, ...], spec: ConvSpec) -> tuple[Matrix, ...]:
    """多通道卷积：对每个输出通道做"通道加权求和 → 加偏置 → 过激活".

    ``image_channels`` 的长度必须等于 ``params.in_channels``（否则抛 :class:`ShapeError`）。
    """
    checked_channels = tuple(as_matrix(channel, name=f"image[{i}]") for i, channel in enumerate(image_channels))
    if len(checked_channels) != params.in_channels:
        raise ShapeError(
            f"输入有 {len(checked_channels)} 个通道，而这一层有 {params.in_channels} 个输入核："
            "多通道卷积要求每一个输入通道都对全部输出通道有贡献。"
        )
    outputs: list[Matrix] = []
    for out_index, kernels in enumerate(params.kernels):
        accumulated: Matrix | None = None
        for channel, kernel in zip(checked_channels, kernels):
            partial = conv2d(
                channel,
                kernel,
                stride=spec.stride,
                padding=spec.padding,
                dilation=spec.dilation,
            )
            if accumulated is None:
                accumulated = partial
            else:
                if len(partial) != len(accumulated) or len(partial[0]) != len(accumulated[0]):
                    raise ShapeError(  # pragma: no cover - 相同参数下不会发生
                        "同一输出通道内各输入通道产生的特征图尺寸不一致。"
                    )
                accumulated = tuple(
                    tuple(a + b for a, b in zip(row_a, row_b))
                    for row_a, row_b in zip(accumulated, partial)
                )
        assert accumulated is not None  # in_channels >= 1 保证非空
        with_bias = tuple(
            tuple(value + params.bias[out_index] for value in row) for row in accumulated
        )
        outputs.append(activate_map(spec.activation, with_bias))
    return tuple(outputs)


def pool_block_forward(
    image_channels: tuple[Matrix, ...],
    *,
    mode: str = POOL_MAX,
    window: int = 2,
    stride: int | None = None,
) -> tuple[Matrix, ...]:
    """逐通道池化（降采样）."""
    if not image_channels:
        raise WindowError("池化的输入至少要有一个通道。")
    return tuple(pool2d(channel, mode=mode, window=window, stride=stride) for channel in image_channels)


def conv_layer_forward(spec: ConvSpec, image_channels: tuple[Matrix, ...]) -> tuple[Matrix, ...]:
    """按 spec 造一层并前向（同一份 spec 两次调用必须**逐位相同**）."""
    return conv_block_forward(initialize_conv(spec), image_channels, spec)


def parameter_count_line(spec: ConvSpec) -> str:
    """一层卷积的账（公式与数一起印出来）."""
    return f"{spec.line()}（{CONV_PARAM_FORMULA}）"


__all__ = [
    "ConvParams",
    "ConvSpec",
    "activate_map",
    "conv_block_forward",
    "conv_layer_forward",
    "initialize_conv",
    "parameter_count_line",
    "pool_block_forward",
]
