"""``network``：一个小 CNN 的前向、反向与参数压平（day093 / M8-D4）.

网络结构（**写死一条链**，为了可读与可对账）：

```text
输入 1×6×6  →  conv(1→2, k=3, same, relu)  →  maxpool(2)  →  flatten(2×3×3=18)
            →  dense(18→2)  →  logits  →  softmax + 交叉熵
```

## 三个"接缝"都在这里被钉住

```text
① 卷积与池化      conv 的输出（含 pre_activation 缓存）直接喂给池化
② 池化与全连接    池化后的每张特征图按行优先压平、通道序拼接 ⇒ dense 的输入
③ 反向            dense 的 d_flat 拆回各通道 ⇒ 逐通道池化反向 ⇒ 卷积反向
```

## 参数压平：沿用 day090 的契约

``flatten_params`` / ``unflatten_params`` 的顺序是**写死**的：
``conv.kernels[o][i]`` → ``conv.bias`` → ``head_weight`` → ``head_bias``。
还原时用**参数本体当形状表**（template）：长度对不上当场抛 :class:`ShapeError`，
而不是"按顺序切"——后者会给出若干"形状正确、数值错位"的参数。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from smart_research_agent.conv_net.errors import (
    BackwardError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.conv_net.gradients import (
    ConvGradients,
    conv_block_backward,
    conv_forward_with_cache,
    pool2d_backward,
)
from smart_research_agent.conv_net.layers import ConvParams, ConvSpec, initialize_conv
from smart_research_agent.conv_net.ops import (
    as_matrix,
    feature_size,
    flatten_map,
    pool2d,
)
from smart_research_agent.conv_net.types import POOL_MAX
from smart_research_agent.math_foundations.types import Matrix, Vector
from smart_research_agent.neural_basics.layers import initialize
from smart_research_agent.neural_basics.losses import cross_entropy


@dataclass(frozen=True)
class CNNParams:
    """一个小 CNN 的全部参数：一层卷积 + 一个全连接头."""

    conv: ConvParams
    head_weight: Matrix
    head_bias: Vector

    def __post_init__(self) -> None:
        checked = as_matrix(self.head_weight, name="head_weight")
        if len(self.head_bias) != len(checked):
            raise ShapeError(
                f"head_bias 长度 {len(self.head_bias)} 与 head_weight 的行数 {len(checked)} 不一致。"
            )
        object.__setattr__(self, "head_weight", checked)
        object.__setattr__(self, "head_bias", tuple(float(value) for value in self.head_bias))

    @property
    def classes(self) -> int:
        """类别数（全连接头的输出宽度）."""
        return len(self.head_weight)

    @property
    def flatten_features(self) -> int:
        """全连接头的输入宽度（压平后的特征维）."""
        return len(self.head_weight[0])

    @property
    def parameter_count(self) -> int:
        """全部参数量 = 卷积层 + 全连接头."""
        return self.conv.parameter_count + self.classes * self.flatten_features + self.classes


@dataclass(frozen=True)
class CNNCache:
    """一次前向留下的三样中间量（反向需要的**全部**东西）."""

    pre_activations: tuple[Matrix, ...]
    conv_outputs: tuple[Matrix, ...]
    pooled: tuple[Matrix, ...]
    flat: Vector


@dataclass(frozen=True)
class CNNGradients:
    """一次反向的三块梯度（卷积层 + 全连接头的两块）."""

    conv: ConvGradients
    head_weight: Matrix
    head_bias: Vector

    def total_norm(self) -> float:
        """全部梯度的整体范数（"这一步被推动了多大"）."""
        head = math.sqrt(
            math.fsum(value * value for row in self.head_weight for value in row)
            + math.fsum(value * value for value in self.head_bias)
        )
        return math.sqrt(self.conv.kernel_norm() ** 2 + head**2)


def pooled_size(image_size: tuple[int, int], spec: ConvSpec, pool_window: int) -> tuple[int, int]:
    """卷积 + 池化之后的空间尺寸（先卷积再池化，两次都用同一个尺寸公式）."""
    height, width = image_size
    conv_h = feature_size(height, spec.kernel_size, stride=spec.stride, padding=spec.padding, dilation=spec.dilation)
    conv_w = feature_size(width, spec.kernel_size, stride=spec.stride, padding=spec.padding, dilation=spec.dilation)
    pool_h = feature_size(conv_h, pool_window, stride=pool_window)
    pool_w = feature_size(conv_w, pool_window, stride=pool_window)
    return pool_h, pool_w


def build_cnn(
    spec: ConvSpec,
    image_size: tuple[int, int],
    *,
    classes: int = 2,
    pool_window: int = 2,
    seed: int = 100,
    init: str = "xavier",
) -> CNNParams:
    """按 spec 造一个小 CNN（卷积核与全连接头都由 LCG 生成 ⇒ **逐位可复现**）."""
    if classes < 2:
        raise ParameterError(f"classes 必须 >= 2，收到 {classes}。")
    conv = initialize_conv(spec)
    height, width = pooled_size(image_size, spec, pool_window)
    in_features = spec.out_channels * height * width
    head_weight, head_bias = initialize(init, classes, in_features, seed=seed)
    return CNNParams(conv=conv, head_weight=head_weight, head_bias=head_bias)


def cnn_forward_cached(
    params: CNNParams,
    spec: ConvSpec,
    image_channels: tuple[Matrix, ...],
    *,
    pool_window: int = 2,
) -> tuple[Vector, CNNCache]:
    """前向并返回 ``(logits, cache)``——cache 是反向需要的全部中间量."""
    conv_outputs, pre_activations = conv_forward_with_cache(spec, params.conv, image_channels)
    pooled = tuple(pool2d(channel, mode=POOL_MAX, window=pool_window) for channel in conv_outputs)
    flat: list[float] = []
    for channel in pooled:
        flat.extend(flatten_map(channel))
    if len(flat) != params.flatten_features:
        raise ShapeError(
            f"压平后的特征维 {len(flat)} 与全连接头期望的 {params.flatten_features} 不一致："
            "网络结构与参数不是同一份（先决定结构，再造参数）。"
        )
    logits = tuple(
        math.fsum(weight * value for weight, value in zip(row, flat)) + params.head_bias[c]
        for c, row in enumerate(params.head_weight)
    )
    cache = CNNCache(
        pre_activations=pre_activations,
        conv_outputs=conv_outputs,
        pooled=pooled,
        flat=tuple(flat),
    )
    return logits, cache


def cnn_forward(
    params: CNNParams,
    spec: ConvSpec,
    image_channels: tuple[Matrix, ...],
    *,
    pool_window: int = 2,
) -> Vector:
    """只要 logits 的前向（``cnn_forward_cached`` 的薄包装）."""
    logits, _cache = cnn_forward_cached(params, spec, image_channels, pool_window=pool_window)
    return logits


def cnn_backward(
    params: CNNParams,
    spec: ConvSpec,
    image_channels: tuple[Matrix, ...],
    grad_logits: Vector,
    *,
    pool_window: int = 2,
    cache: CNNCache | None = None,
) -> CNNGradients:
    """完整反向：``logits → 全连接头 → 池化 → 卷积``（每一步都复用已有的反向实现）.

    ``cache`` 缺省时**重跑一次前向**取得中间量（保真优先于省一次前向）。
    """
    checked_grad = tuple(float(value) for value in grad_logits)
    if len(checked_grad) != params.classes:
        raise BackwardError(
            f"grad_logits 长度 {len(checked_grad)} 与类别数 {params.classes} 不一致。"
        )
    if cache is None:
        _logits, cache = cnn_forward_cached(params, spec, image_channels, pool_window=pool_window)
    # ③a 全连接头：dW = g ⊗ flat；dflat = Wᵀ·g；dB = g
    head_weight_grad = tuple(
        tuple(gradient * value for value in cache.flat) for gradient in checked_grad
    )
    d_flat = tuple(
        math.fsum(params.head_weight[c][f] * checked_grad[c] for c in range(params.classes))
        for f in range(params.flatten_features)
    )
    head_bias_grad = checked_grad
    # ③b 拆回各通道，逐通道池化反向
    pooled_shapes = [(len(channel), len(channel[0])) for channel in cache.pooled]
    d_channels: list[Matrix] = []
    cursor = 0
    for channel, (rows, columns) in zip(cache.conv_outputs, pooled_shapes):
        size = rows * columns
        grad_pooled = tuple(
            tuple(d_flat[cursor + r * columns + c] for c in range(columns)) for r in range(rows)
        )
        cursor += size
        d_channels.append(
            pool2d_backward(channel, grad_pooled, mode=POOL_MAX, window=pool_window)
        )
    # ③c 卷积反向（带激活导数）
    conv_grads = conv_block_backward(
        spec,
        params.conv,
        tuple(as_matrix(channel, name=f"image[{i}]") for i, channel in enumerate(image_channels)),
        tuple(d_channels),
        pre_activations=cache.pre_activations,
    )
    return CNNGradients(
        conv=conv_grads,
        head_weight=head_weight_grad,
        head_bias=head_bias_grad,
    )


def predict(params: CNNParams, spec: ConvSpec, image_channels: tuple[Matrix, ...], *, pool_window: int = 2) -> int:
    """取 logits 的最大值下标（**预测类别**）."""
    logits = cnn_forward(params, spec, image_channels, pool_window=pool_window)
    return max(range(len(logits)), key=lambda index: logits[index])


def loss_and_grad(
    params: CNNParams, spec: ConvSpec, image_channels: tuple[Matrix, ...], label: int, *, pool_window: int = 2
) -> tuple[float, CNNGradients]:
    """一次"前向 → 交叉熵 → 反向"，返回 ``(损失, 梯度)``.

    损失复用 day089 的 :func:`neural_basics.losses.cross_entropy`（`-log p[标签]`）；
    它的梯度就是 ``p − onehot``（day090 第 7 章写下的那条），这里直接写出来。
    """
    logits, cache = cnn_forward_cached(params, spec, image_channels, pool_window=pool_window)
    loss = cross_entropy(logits, label)
    probabilities = _softmax(logits)
    grad_logits = tuple(
        probability - (1.0 if index == label else 0.0)
        for index, probability in enumerate(probabilities)
    )
    grads = cnn_backward(params, spec, image_channels, grad_logits, pool_window=pool_window, cache=cache)
    return loss, grads


def _softmax(logits: Vector) -> Vector:
    """数值稳定的 softmax（减最大值；与本仓库其余实现同口径）."""
    if not logits:
        raise BackwardError("softmax 的输入不能为空。")
    largest = max(logits)
    exps = [math.exp(value - largest) for value in logits]
    total = math.fsum(exps)
    return tuple(value / total for value in exps)


def flatten_params(params: CNNParams) -> Vector:
    """按写死顺序压平全部参数：``conv.kernels → conv.bias → head_weight → head_bias``."""
    flat: list[float] = []
    for per_output in params.conv.kernels:
        for kernel in per_output:
            for row in kernel:
                flat.extend(row)
    flat.extend(params.conv.bias)
    for row in params.head_weight:
        flat.extend(row)
    flat.extend(params.head_bias)
    return tuple(flat)


def unflatten_params(flat: Vector, template: CNNParams) -> CNNParams:
    """按模板（**参数本体当形状表**）把一串数还原成 :class:`CNNParams`."""
    values = tuple(float(value) for value in flat)
    if any(not math.isfinite(value) for value in values):
        raise ShapeError("压平的参数里出现非有限数（nan / inf）：先查那一步的学习率。")
    required = template.parameter_count
    if len(values) != required:
        raise ShapeError(
            f"压平的向量有 {len(values)} 个数，而模板需要 {required} 个："
            "长度对不上时'按顺序切'会静默地填错位置，给出形状正确、数值错位的参数。"
        )
    cursor = 0
    kernels: list[tuple[Matrix, ...]] = []
    for per_output in template.conv.kernels:
        per_input: list[Matrix] = []
        for kernel in per_output:
            rows: list[tuple[float, ...]] = []
            for _ in kernel:
                row = values[cursor : cursor + len(kernel[0])]
                cursor += len(kernel[0])
                rows.append(tuple(row))
            per_input.append(tuple(rows))
        kernels.append(tuple(per_input))
    bias = values[cursor : cursor + template.conv.out_channels]
    cursor += template.conv.out_channels
    head_weight: list[tuple[float, ...]] = []
    for row in template.head_weight:
        head_weight.append(tuple(values[cursor : cursor + len(row)]))
        cursor += len(row)
    head_bias = values[cursor : cursor + template.classes]
    cursor += template.classes
    return CNNParams(
        conv=ConvParams(kernels=tuple(kernels), bias=bias),
        head_weight=tuple(head_weight),
        head_bias=head_bias,
    )


def flatten_gradients(grads: CNNGradients) -> Vector:
    """按与 :func:`flatten_params` **相同**的顺序压平梯度（两者必须能逐位相加）."""
    flat: list[float] = []
    for per_output in grads.conv.kernel_grads:
        for kernel in per_output:
            for row in kernel:
                flat.extend(row)
    flat.extend(grads.conv.bias_grad)
    for row in grads.head_weight:
        flat.extend(row)
    flat.extend(grads.head_bias)
    return tuple(flat)


__all__ = [
    "CNNCache",
    "CNNGradients",
    "CNNParams",
    "build_cnn",
    "cnn_backward",
    "cnn_forward",
    "cnn_forward_cached",
    "flatten_gradients",
    "flatten_params",
    "loss_and_grad",
    "pooled_size",
    "predict",
    "unflatten_params",
]
