"""``ops``：卷积与池化的**算子层**（day093 / M8-D4）.

这一层只回答"一个算子把一张图变成什么"，不含参数、不含训练：

```text
output_size       ⌊(H + 2p − d·(k−1) − 1)/s⌋ + 1     ——先算尺寸，再谈别的
resolve_padding   valid / same（步长 1、奇数核时补到同尺寸）
pad2d             常量补零（对称，两侧各 pad）
conv2d            单通道滑窗点积（**不做核翻转**：这是互相关，不是数学卷积）
max_pool2d/avg    窗口降采样
receptive_field   逐层递推：r ← r + (k−1)·∏s
```

## 一个必须写下来的约定：互相关 vs 卷积

深度学习里的"卷积"实际是**互相关**（cross-correlation）：输出
``y[i,j] = Σ_{u,v} x[i+u, j+v]·K[u,v]``，**不把核翻转 180°**。
PyTorch 的 ``nn.Conv2d`` 也是互相关。本包沿用这个约定，并在 ``conv2d`` 的文档串里写明——
否则"卷积核要不要翻转"会成为一个永远说不清的分歧。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from smart_research_agent.conv_net.errors import (
    NumericError,
    ParameterError,
    ShapeError,
    WindowError,
)
from smart_research_agent.conv_net.types import (
    PADDING_MODES,
    PADDING_SAME,
    PADDING_VALID,
    POOL_AVG,
    POOL_MAX,
    POOL_MODES,
)
from smart_research_agent.math_foundations.types import Matrix, Vector


def checked_positive_int(value: object, *, name: str, minimum: int = 1) -> int:
    """校验一个 ``>= minimum`` 的整数（核尺寸 / 步长 / 膨胀 / 窗口都走这里）."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ParameterError(f"{name} 必须是整数，收到 {type(value).__name__}（{value!r}）。")
    if value < minimum:
        raise ParameterError(
            f"{name} 必须 >= {minimum}，收到 {value}："
            "非正尺寸会让窗口失去定义，而它不会报错，只会算出一张空的特征图。"
        )
    return value


def as_matrix(rows: Sequence[Sequence[float]], *, name: str) -> Matrix:
    """把二维序列收敛成 ``Matrix``（非空、行等长、元素有限）."""
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
        raise ShapeError(f"{name} 必须是非空的二维序列（行 × 列）。")
    checked: list[tuple[float, ...]] = []
    width: int | None = None
    for index, row in enumerate(rows):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or not row:
            raise ShapeError(f"{name} 的第 {index} 行不是非空的序列。")
        values: list[float] = []
        for column, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ShapeError(f"{name}[{index}][{column}] 必须是实数，收到 {value!r}。")
            number = float(value)
            if not math.isfinite(number):
                raise NumericError(f"{name}[{index}][{column}] 是非有限数（{value!r}）。")
            values.append(number)
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise ShapeError(f"{name} 的第 {index} 行宽度 {len(values)} 与首行 {width} 不一致。")
        checked.append(tuple(values))
    return tuple(checked)


def resolve_padding(
    size: int,
    kernel: int,
    *,
    stride: int = 1,
    padding: int | str = PADDING_VALID,
    dilation: int = 1,
) -> int:
    """把 ``padding`` 解析成具体的补边数（支持 ``valid`` / ``same`` / 整数）.

    ``same`` 只在**步长 1** 且有解时成立：补边数 ``= (dilation·(k−1))//2``，
    并要求解析出的输出尺寸恰好等于输入。偶数核配步长 1 会得到 ``size − 1``，
    因此当场抛 :class:`WindowError`，而不是默默给出一个"少一列"的特征图。
    """
    checked_size = checked_positive_int(size, name="size")
    checked_kernel = checked_positive_int(kernel, name="kernel")
    checked_stride = checked_positive_int(stride, name="stride")
    checked_dilation = checked_positive_int(dilation, name="dilation")
    if isinstance(padding, bool):
        raise ParameterError(f"padding 不能是布尔值（{padding!r}）。")
    if isinstance(padding, int):
        if padding < 0:
            raise ParameterError(f"整数 padding 必须非负，收到 {padding}。")
        return padding
    if padding not in PADDING_MODES:
        raise ParameterError(f"未知的填充模式 {padding!r}：可用取值 {list(PADDING_MODES)}。")
    if padding == PADDING_VALID:
        return 0
    if checked_stride != 1:
        raise WindowError(
            f"same 填充要求 stride = 1，收到 stride = {checked_stride}："
            "步长大于 1 时'同尺寸'与'步长'是两个互相冲突的要求。"
        )
    pad = (checked_dilation * (checked_kernel - 1)) // 2
    produced = output_size(
        checked_size, checked_kernel, stride=1, padding=pad, dilation=checked_dilation
    )
    if produced != checked_size:
        raise WindowError(
            f"same 填充对 size={checked_size}、kernel={checked_kernel}、dilation={checked_dilation} "
            f"无解：解析出 {pad} 的补边，输出却是 {produced}（期望 {checked_size}）。"
            "same 只在奇数核时能精确成立。"
        )
    return pad


def output_size(
    size: int,
    kernel: int,
    *,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """输出尺寸 ``⌊(H + 2p − d·(k−1) − 1)/s⌋ + 1``（结果 < 1 抛 ``WindowError``）."""
    checked_size = checked_positive_int(size, name="size")
    checked_kernel = checked_positive_int(kernel, name="kernel")
    checked_stride = checked_positive_int(stride, name="stride")
    checked_dilation = checked_positive_int(dilation, name="dilation")
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
        raise ParameterError(f"output_size 的 padding 必须是非负整数，收到 {padding!r}。")
    effective = checked_dilation * (checked_kernel - 1) + 1
    numerator = checked_size + 2 * padding - effective
    if numerator < 0:
        raise WindowError(
            f"输入 {checked_size} 装不下有效核宽 {effective}（补边 {padding}）："
            "窗口在输入上放不下，输出尺寸没有定义。"
        )
    return numerator // checked_stride + 1


def feature_size(
    size: int,
    kernel: int,
    *,
    stride: int = 1,
    padding: int | str = PADDING_VALID,
    dilation: int = 1,
) -> int:
    """先解析填充（``valid`` / ``same`` / 整数）再套尺寸公式——**调用方不用自己解析**.

    ``output_size`` 只接受整数补边（它的职责是"算一次公式"），
    而实际调用点拿到的往往是 ``spec.padding``（可能是 ``"same"``）。
    两者的接缝放在这里一处，而不是散落在每个调用点。
    """
    pad = resolve_padding(size, kernel, stride=stride, padding=padding, dilation=dilation)
    return output_size(size, kernel, stride=stride, padding=pad, dilation=dilation)


def pad2d(image: Sequence[Sequence[float]], padding: int) -> Matrix:
    """常量补零（**对称**：上下左右各补 ``padding`` 行 / 列）."""
    checked = as_matrix(image, name="image")
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
        raise ParameterError(f"padding 必须是非负整数，收到 {padding!r}。")
    if padding == 0:
        return checked
    height, width = len(checked), len(checked[0])
    new_width = width + 2 * padding
    blank = (0.0,) * new_width
    padded: list[tuple[float, ...]] = []
    for _ in range(padding):
        padded.append(blank)
    for row in checked:
        padded.append((0.0,) * padding + row + (0.0,) * padding)
    for _ in range(padding):
        padded.append(blank)
    return tuple(padded)


def conv2d(
    image: Sequence[Sequence[float]],
    kernel: Sequence[Sequence[float]],
    *,
    stride: int = 1,
    padding: int | str = PADDING_VALID,
    dilation: int = 1,
) -> Matrix:
    """单通道二维卷积（**互相关约定**：核不翻转）.

    ```text
    y[i,j] = Σ_{u,v} x[i·s + u·d, j·s + v·d] · K[u,v]
    ```

    ``padding`` 可以是 ``valid`` / ``same`` / 非负整数。当输入已被外部补过边时，
    传整数 0 即可（``pad2d`` 先补、``conv2d`` 后卷，两步分开写更可读）。
    """
    checked_image = as_matrix(image, name="image")
    checked_kernel = as_matrix(kernel, name="kernel")
    height, width = len(checked_image), len(checked_image[0])
    kernel_h, kernel_w = len(checked_kernel), len(checked_kernel[0])
    pad = resolve_padding(
        height, kernel_h, stride=stride, padding=padding, dilation=dilation
    )
    # 宽方向独立解析一次，保证偶数核 + same 在任一维不成立时都当场拒绝
    pad_w = resolve_padding(
        width, kernel_w, stride=stride, padding=padding, dilation=dilation
    )
    source = pad2d(checked_image, pad) if pad else checked_image
    if pad_w != pad:  # pragma: no cover - 只有在非方形输入 + same 的极端组合下才可能
        raise WindowError("宽方向的补边数与高方向不一致：same 要求输入与核在每一维都能对齐。")
    out_h = output_size(height, kernel_h, stride=stride, padding=pad, dilation=dilation)
    out_w = output_size(width, kernel_w, stride=stride, padding=pad_w, dilation=dilation)
    result: list[tuple[float, ...]] = []
    for row in range(out_h):
        values: list[float] = []
        for column in range(out_w):
            total = 0.0
            for u in range(kernel_h):
                for v in range(kernel_w):
                    total += source[row * stride + u * dilation][column * stride + v * dilation] * checked_kernel[u][v]
            values.append(total)
        result.append(tuple(values))
    return tuple(result)


def pool2d(
    image: Sequence[Sequence[float]],
    *,
    mode: str = POOL_MAX,
    window: int = 2,
    stride: int | None = None,
) -> Matrix:
    """池化（``max`` / ``avg``）：窗口与步长必须能**整除地**盖满输入.

    ``stride`` 缺省等于 ``window``（不重叠池化）。窗口盖不满时抛 :class:`WindowError`，
    而不是"丢掉最后一块不平整的部分"——后者会让实际下采样率与你以为的不同。
    """
    checked = as_matrix(image, name="image")
    if mode not in POOL_MODES:
        raise ParameterError(f"未知的池化模式 {mode!r}：可用取值 {list(POOL_MODES)}。")
    checked_window = checked_positive_int(window, name="window")
    checked_stride = checked_window if stride is None else checked_positive_int(stride, name="stride")
    height, width = len(checked), len(checked[0])
    if height % checked_stride != 0 or width % checked_stride != 0:
        raise WindowError(
            f"输入 {height}×{width} 不能被步长 {checked_stride} 整除："
            "本包要求窗口整除地盖满输入，不做'丢掉最后一块'的兜底。"
        )
    if (height - checked_window) % checked_stride != 0 or (width - checked_window) % checked_stride != 0:
        raise WindowError(
            f"窗口 {checked_window} 与步长 {checked_stride} 无法整齐地覆盖 {height}×{width} 的输入。"
        )
    out_h = (height - checked_window) // checked_stride + 1
    out_w = (width - checked_window) // checked_stride + 1
    result: list[tuple[float, ...]] = []
    for row in range(out_h):
        values: list[float] = []
        for column in range(out_w):
            window_values: list[float] = []
            for u in range(checked_window):
                for v in range(checked_window):
                    window_values.append(
                        checked[row * checked_stride + u][column * checked_stride + v]
                    )
            if mode == POOL_MAX:
                values.append(max(window_values))
            else:
                values.append(math.fsum(window_values) / len(window_values))
        result.append(tuple(values))
    return tuple(result)


def max_pool2d(image: Sequence[Sequence[float]], window: int = 2, stride: int | None = None) -> Matrix:
    """最大池化（``pool2d(mode="max")`` 的便捷入口）."""
    return pool2d(image, mode=POOL_MAX, window=window, stride=stride)


def avg_pool2d(image: Sequence[Sequence[float]], window: int = 2, stride: int | None = None) -> Matrix:
    """平均池化（``pool2d(mode="avg")`` 的便捷入口）."""
    return pool2d(image, mode=POOL_AVG, window=window, stride=stride)


def receptive_field(layers: Sequence[tuple[int, int]]) -> int:
    """感受野递推：``r₀ = 1；r ← r + (k − 1)·∏s``（``layers`` 是``(核, 步长)``序列）.

    空序列的感受野是 1（"什么都不做"时一个输出只看一个输入像素）。
    """
    if not layers:
        return 1
    field = 1
    product = 1
    for index, (kernel, stride) in enumerate(layers):
        checked_kernel = checked_positive_int(kernel, name=f"layers[{index}].kernel")
        checked_stride = checked_positive_int(stride, name=f"layers[{index}].stride")
        field += (checked_kernel - 1) * product
        product *= checked_stride
    return field


def flatten_map(image: Sequence[Sequence[float]]) -> Vector:
    """把特征图按行优先压平成一串数（**全连接头的输入**）."""
    checked = as_matrix(image, name="image")
    return tuple(value for row in checked for value in row)


def feature_stats(image: Sequence[Sequence[float]]) -> dict[str, float]:
    """特征图的一行画像：形状 / 均值 / 最大值 / 非零比例（"这一层学到了什么"的可读读数）."""
    checked = as_matrix(image, name="image")
    flat = [value for row in checked for value in row]
    total = len(flat)
    return {
        "height": float(len(checked)),
        "width": float(len(checked[0])),
        "mean": math.fsum(flat) / total,
        "max": max(flat),
        "nonzero_ratio": sum(1 for value in flat if value != 0.0) / total,
    }


__all__ = [
    "as_matrix",
    "avg_pool2d",
    "checked_positive_int",
    "conv2d",
    "feature_size",
    "feature_stats",
    "flatten_map",
    "max_pool2d",
    "output_size",
    "pad2d",
    "pool2d",
    "receptive_field",
    "resolve_padding",
]
