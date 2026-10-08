"""``verify``：七条性质与三类判据（day093 / M8-D4）.

```text
相等（逐位）  ① conv2d 与手写滑窗点积逐位一致
              ③ same 填充（奇数核、步长 1）让输出与输入同尺寸
              ④ 尺寸公式给出的数与实际特征图形状一致
              ⑥ 池化的每个输出都真的是那个窗口的极值 / 均值
              ⑦ 感受野的递推公式与逐层区间传播的结果一致
上界          ② 卷积是线性算子（偏差 <= 1e-12）
              ⑤ 卷积解析梯度与数值差分一致（相对误差 <= 1e-9）
```

判据类型与 day090 / day092 同源：**相等**（逐位 / 整数）与**不超过上界**（误差）。
本课没有"下界"判据——因为这一课没有"两个东西必须不同"的断言，
它的每一条都在问"这个对不对"。

## 第 ⑤ 条为什么要用数值差分

``conv2d_backward`` 里最容易写串的是 **d_kernel 与 d_image 的索引**：
两者互为"转置"关系，写反了仍然会得到形状正确的梯度。唯一能区分它俩的，
是一把**独立于这段推导**的尺子——day074 的数值差分。这一条同时钉住两块梯度。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from smart_research_agent.conv_net.errors import ConvError
from smart_research_agent.conv_net.gradients import conv2d_backward
from smart_research_agent.conv_net.ops import conv2d, output_size, pool2d, receptive_field
from smart_research_agent.conv_net.types import (
    CONV_PROPERTIES,
    PADDING_SAME,
    POOL_AVG,
    POOL_MAX,
    PROPERTY_CONV_BACKWARD_MATCHES_NUMERICAL,
    PROPERTY_CONV_IS_LINEAR,
    PROPERTY_CONV_MATCHES_SLIDING_DOT,
    PROPERTY_OUTPUT_SIZE_FORMULA_MATCHES_SHAPE,
    PROPERTY_POOL_RETURNS_WINDOW_EXTREME,
    PROPERTY_RECEPTIVE_FIELD_MATCHES_DIRECT,
    PROPERTY_SAME_PADDING_PRESERVES_SIZE,
)
from smart_research_agent.math_foundations.calculus import gradient

#: 第 ① 条用的写死输入：4×4 的图与 3×3 的核.
SLIDING_IMAGE: tuple[tuple[float, ...], ...] = (
    (1.0, 2.0, 0.0, -1.0),
    (0.0, 1.0, 2.0, 1.0),
    (-1.0, 0.0, 1.0, 2.0),
    (2.0, -1.0, 0.0, 1.0),
)
SLIDING_KERNEL: tuple[tuple[float, ...], ...] = (
    (1.0, 0.0, -1.0),
    (0.0, 1.0, 0.0),
    (-1.0, 0.0, 1.0),
)

#: 第 ③ 条用的奇数核（same 填充要求奇数核）.
SAME_KERNEL_SIZE = 3

#: 第 ⑤ 条用的样本（4×4 图、3×3 核、写死的回传梯度）.
GRAD_IMAGE: tuple[tuple[float, ...], ...] = (
    (0.5, -1.0, 0.25, 2.0),
    (1.0, 0.0, -0.5, 0.75),
    (-0.25, 1.5, 0.0, -1.0),
    (0.3, -0.2, 0.6, 0.4),
)
GRAD_KERNEL: tuple[tuple[float, ...], ...] = (
    (0.2, -0.4, 0.6),
    (-0.1, 0.3, 0.5),
    (0.7, -0.2, 0.1),
)
GRAD_OUTPUT: tuple[tuple[float, ...], ...] = (
    (1.0, -0.5),
    (0.25, 2.0),
)
GRAD_TOLERANCE = 1e-9

#: 第 ⑦ 条用的层序列（核, 步长）——公式与区间传播两条路径.
RECEPTIVE_LAYERS: tuple[tuple[int, int], ...] = ((3, 1), (3, 1), (2, 2), (3, 1))

#: 第 ⑥ 条用的池化输入.
POOL_IMAGE: tuple[tuple[float, ...], ...] = (
    (1.0, 3.0, 2.0, 0.0),
    (4.0, 2.0, 1.0, 5.0),
    (0.5, 1.5, 3.5, 2.5),
    (2.0, 0.0, 1.0, 4.0),
)


@dataclass(frozen=True)
class Check:
    """一次性质校验的读数与判据（相等 / 上界）."""

    reading: float
    upper_bound: float | None = None
    left: str = "-"
    right: str = "-"

    def passed(self) -> bool:
        """读数是否落在判据内."""
        if self.upper_bound is not None and self.reading > self.upper_bound:
            return False
        return True

    def bound_text(self) -> str:
        """判据的一行文本."""
        if self.upper_bound is None:
            return "== 逐位"
        if self.upper_bound == 0.0:
            return "== 0"
        return f"<= {self.upper_bound:.1e}"


@dataclass(frozen=True)
class PropertyOutcome:
    """一条性质的结论：是否通过 + 现场读数 + 两个来源."""

    name: str
    passed: bool
    check: Check

    def line(self) -> str:
        """``通过 conv_matches_sliding_dot | 读数 0.000e+00 == 逐位 | a vs b``."""
        mark = "通过" if self.passed else "失败"
        return (
            f"{mark} {self.name:<42} | 读数 {self.check.reading:.3e} "
            f"{self.check.bound_text()} | {self.check.left} vs {self.check.right}"
        )


@dataclass
class PropertyReport:
    """七条性质的汇总报告."""

    outcomes: list[PropertyOutcome] = field(default_factory=list)

    @property
    def passed(self) -> int:
        """通过的条数."""
        return sum(1 for outcome in self.outcomes if outcome.passed)

    @property
    def total(self) -> int:
        """总条数."""
        return len(self.outcomes)

    def all_passed(self) -> bool:
        """是否全部通过."""
        return self.passed == self.total

    def lines(self) -> tuple[str, ...]:
        """逐行印出（演示脚本与教程引用的是同一批读数）."""
        return tuple(outcome.line() for outcome in self.outcomes)


# --------------------------------------------------------------------------- #
# 单条性质
# --------------------------------------------------------------------------- #


def _manual_sliding_dot(image, kernel, stride: int = 1) -> tuple[tuple[float, ...], ...]:
    """手写滑窗点积（**独立于 ops.conv2d** 的第二条实现，用来对账）."""
    height, width = len(image), len(image[0])
    kernel_h, kernel_w = len(kernel), len(kernel[0])
    out_h = (height - kernel_h) // stride + 1
    out_w = (width - kernel_w) // stride + 1
    result: list[tuple[float, ...]] = []
    for i in range(out_h):
        row: list[float] = []
        for j in range(out_w):
            total = 0.0
            for u in range(kernel_h):
                for v in range(kernel_w):
                    total += image[i * stride + u][j * stride + v] * kernel[u][v]
            row.append(total)
        result.append(tuple(row))
    return tuple(result)


def check_conv_matches_sliding_dot() -> PropertyOutcome:
    """① conv2d 的输出与手写滑窗点积逐位一致（读数 = 最大绝对差）."""
    produced = conv2d(SLIDING_IMAGE, SLIDING_KERNEL)
    manual = _manual_sliding_dot(SLIDING_IMAGE, SLIDING_KERNEL)
    gap = max(
        (abs(a - b) for row_a, row_b in zip(produced, manual) for a, b in zip(row_a, row_b)),
        default=0.0,
    )
    check = Check(reading=gap, upper_bound=0.0, left="ops.conv2d", right="手写滑窗点积")
    return PropertyOutcome(PROPERTY_CONV_MATCHES_SLIDING_DOT, check.passed(), check)


def check_conv_is_linear() -> PropertyOutcome:
    """② 卷积是线性算子：conv(αx + βy) = α·conv(x) + β·conv(y)."""
    alpha, beta = 1.5, -0.75
    other_image = tuple(tuple(-value for value in row) for row in SLIDING_IMAGE)
    combined = tuple(
        tuple(alpha * a + beta * b for a, b in zip(row_a, row_b))
        for row_a, row_b in zip(SLIDING_IMAGE, other_image)
    )
    left = conv2d(combined, SLIDING_KERNEL)
    right_parts = (conv2d(SLIDING_IMAGE, SLIDING_KERNEL), conv2d(other_image, SLIDING_KERNEL))
    gap = 0.0
    for i, row in enumerate(left):
        for j, value in enumerate(row):
            expected = alpha * right_parts[0][i][j] + beta * right_parts[1][i][j]
            gap = max(gap, abs(value - expected))
    check = Check(
        reading=gap, upper_bound=1e-12, left="conv(αx+βy)", right="α·conv(x)+β·conv(y)"
    )
    return PropertyOutcome(PROPERTY_CONV_IS_LINEAR, check.passed(), check)


def check_same_padding_preserves_size() -> PropertyOutcome:
    """③ same 填充（奇数核、步长 1）让输出与输入同尺寸（读数 = 尺寸差）."""
    produced = conv2d(SLIDING_IMAGE, SLIDING_KERNEL, padding=PADDING_SAME)
    gap = abs(len(produced) - len(SLIDING_IMAGE)) + abs(len(produced[0]) - len(SLIDING_IMAGE[0]))
    check = Check(
        reading=float(gap),
        upper_bound=0.0,
        left="conv2d(padding=same)",
        right=f"输入 {len(SLIDING_IMAGE)}×{len(SLIDING_IMAGE[0])}",
    )
    return PropertyOutcome(PROPERTY_SAME_PADDING_PRESERVES_SIZE, check.passed(), check)


def check_output_size_formula_matches_shape() -> PropertyOutcome:
    """④ 尺寸公式给出的数与实际特征图形状一致（对几组参数各验一次）."""
    worst = 0.0
    for kernel, stride, padding, dilation in ((3, 1, 0, 1), (3, 2, 1, 1), (2, 2, 0, 1), (3, 1, 1, 2)):
        image = tuple(tuple(float(i * 4 + j) for j in range(6)) for i in range(6))
        produced = conv2d(image, tuple(tuple(1.0 for _ in range(kernel)) for _ in range(kernel)),
                          stride=stride, padding=padding, dilation=dilation)
        expected_h = output_size(6, kernel, stride=stride, padding=padding, dilation=dilation)
        expected_w = expected_h
        worst = max(worst, abs(len(produced) - expected_h), abs(len(produced[0]) - expected_w))
    check = Check(reading=float(worst), upper_bound=0.0, left="实际形状", right="输出尺寸公式")
    return PropertyOutcome(PROPERTY_OUTPUT_SIZE_FORMULA_MATCHES_SHAPE, check.passed(), check)


def check_conv_backward_matches_numerical() -> PropertyOutcome:
    """⑤ 卷积解析梯度（dK / dX）与 day074 的数值差分一致（读数 = 最大绝对差）."""
    kernel_h, kernel_w = len(GRAD_KERNEL), len(GRAD_KERNEL[0])
    image_h, image_w = len(GRAD_IMAGE), len(GRAD_IMAGE[0])
    kernel_size = kernel_h * kernel_w

    def objective(flat: tuple[float, ...]) -> float:
        kernel = tuple(
            tuple(flat[r * kernel_w + c] for c in range(kernel_w)) for r in range(kernel_h)
        )
        image = tuple(
            tuple(flat[kernel_size + r * image_w + c] for c in range(image_w))
            for r in range(image_h)
        )
        output = conv2d(image, kernel)
        return math.fsum(
            output[i][j] * GRAD_OUTPUT[i][j]
            for i in range(len(output))
            for j in range(len(output[0]))
        )

    flat_start = tuple(value for row in GRAD_KERNEL for value in row) + tuple(
        value for row in GRAD_IMAGE for value in row
    )
    numeric = gradient(objective, flat_start)
    analytic_kernel, analytic_image = conv2d_backward(GRAD_IMAGE, GRAD_KERNEL, GRAD_OUTPUT)
    analytic = tuple(value for row in analytic_kernel for value in row) + tuple(
        value for row in analytic_image for value in row
    )
    worst = max((abs(a - b) for a, b in zip(analytic, numeric)), default=0.0)
    check = Check(
        reading=worst,
        upper_bound=GRAD_TOLERANCE,
        left="conv2d_backward（解析）",
        right="math_foundations.calculus.gradient（数值）",
    )
    return PropertyOutcome(PROPERTY_CONV_BACKWARD_MATCHES_NUMERICAL, check.passed(), check)


def check_pool_returns_window_extreme() -> PropertyOutcome:
    """⑥ 池化的每个输出都真的是那个窗口的极值 / 均值（读数 = 最大偏差）."""
    window = 2
    pooled_max = pool2d(POOL_IMAGE, mode=POOL_MAX, window=window)
    pooled_avg = pool2d(POOL_IMAGE, mode=POOL_AVG, window=window)
    worst = 0.0
    for i in range(len(pooled_max)):
        for j in range(len(pooled_max[0])):
            cells = [
                POOL_IMAGE[i * window + u][j * window + v]
                for u in range(window)
                for v in range(window)
            ]
            worst = max(worst, abs(pooled_max[i][j] - max(cells)))
            worst = max(worst, abs(pooled_avg[i][j] - math.fsum(cells) / len(cells)))
    check = Check(reading=worst, upper_bound=1e-12, left="pool2d 输出", right="窗口极值 / 均值")
    return PropertyOutcome(PROPERTY_POOL_RETURNS_WINDOW_EXTREME, check.passed(), check)


def direct_receptive_field(layers: tuple[tuple[int, int], ...]) -> int:
    """**区间传播**：从输出的一格往回推它覆盖了输入的多少格（独立于递推公式）.

    ```text
    区间初值 [0, 0]（输出的一格只看输入的 0 号位置）
    每往回一层： new_start = start·s + 0, new_end = end·s + (k−1)
    感受野 = new_end − new_start + 1
    ```
    """
    start, end = 0, 0
    for kernel, stride in reversed(layers):
        start, end = start * stride, end * stride + (kernel - 1)
    return end - start + 1


def check_receptive_field_matches_direct() -> PropertyOutcome:
    """⑦ 感受野的递推公式与逐层区间传播一致（读数 = 整数差）."""
    formula = receptive_field(RECEPTIVE_LAYERS)
    direct = direct_receptive_field(RECEPTIVE_LAYERS)
    check = Check(
        reading=float(abs(formula - direct)),
        upper_bound=0.0,
        left=f"ops.receptive_field = {formula}",
        right=f"区间传播 = {direct}",
    )
    return PropertyOutcome(PROPERTY_RECEPTIVE_FIELD_MATCHES_DIRECT, check.passed(), check)


#: 七条性质的名字 -> 检查函数（键顺序 = :data:`types.CONV_PROPERTIES`）.
CHECKS = {
    PROPERTY_CONV_MATCHES_SLIDING_DOT: check_conv_matches_sliding_dot,
    PROPERTY_CONV_IS_LINEAR: check_conv_is_linear,
    PROPERTY_SAME_PADDING_PRESERVES_SIZE: check_same_padding_preserves_size,
    PROPERTY_OUTPUT_SIZE_FORMULA_MATCHES_SHAPE: check_output_size_formula_matches_shape,
    PROPERTY_CONV_BACKWARD_MATCHES_NUMERICAL: check_conv_backward_matches_numerical,
    PROPERTY_POOL_RETURNS_WINDOW_EXTREME: check_pool_returns_window_extreme,
    PROPERTY_RECEPTIVE_FIELD_MATCHES_DIRECT: check_receptive_field_matches_direct,
}

if set(CHECKS) != set(CONV_PROPERTIES):  # pragma: no cover - 导入期不变式
    raise ConvError(
        "性质名单与检查函数表不一致：少一条的性质会静默地不在报告里出现，"
        "而'少一条'与'它通过了'在读报告时长得一样。"
    )


def check_all() -> PropertyReport:
    """跑完七条性质，返回汇总报告（顺序与 :data:`types.CONV_PROPERTIES` 一致）."""
    report = PropertyReport()
    for name in CONV_PROPERTIES:
        report.outcomes.append(CHECKS[name]())
    return report


__all__ = [
    "CHECKS",
    "GRAD_IMAGE",
    "GRAD_KERNEL",
    "GRAD_OUTPUT",
    "GRAD_TOLERANCE",
    "POOL_IMAGE",
    "RECEPTIVE_LAYERS",
    "SAME_KERNEL_SIZE",
    "SLIDING_IMAGE",
    "SLIDING_KERNEL",
    "Check",
    "PropertyOutcome",
    "PropertyReport",
    "check_all",
    "check_conv_backward_matches_numerical",
    "check_conv_is_linear",
    "check_conv_matches_sliding_dot",
    "check_output_size_formula_matches_shape",
    "check_pool_returns_window_extreme",
    "check_receptive_field_matches_direct",
    "check_same_padding_preserves_size",
    "direct_receptive_field",
]
