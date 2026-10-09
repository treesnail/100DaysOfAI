"""``study``：五张表（day090 / M8-D2）.

每一天最后都留下几张表，因为**一个数只有被印出来才可能被反驳**。
今天五张表各回答一个"这一次反向算对了吗"的问题：

```text
导数表      六个激活的导数：公式 / 现场读数（softmax 那一行印的是"雅可比每行和为 0"）
一层反向表  一层 (3→4)：dW / db / dx 三块的形状与范数
网络反向表  MLP 3→5→2：逐层的三个范数（梯度爆炸最早的可读信号）
校验表      七条性质各自的读数与上界（解析 vs 数值，跨包 vs 跨包）
性质表      七条性质：是否通过 / 现场读数 / 跨包对象
```

## 一条纪律：每一行都要带**两个数**

与前面各天的表同源——一行只写"通过"的表是没法反驳的。
因此每张表的每一行都至少有两列：**读数**与**参照**（公式 / 形状 / 期望 / 跨包对象）。

## softmax 那一行印什么

``softmax`` 的"导数"不是一个数，而是一张矩阵。它的对角线 ``p_i(1−p_i)`` 看着像
"每个概率自己的导数"，但那只说对了一半。这一课选择印**每一行之和**：

```text
Σ_j J[i][j] = Σ_j p_i(δ_ij − p_j) = p_i(1 − Σ_j p_j) = p_i·0 = 0
```

"每一行和为 0"是 softmax 雅可比最可读、也最容易验证的一条性质——
它同时说明"概率整体平移不改变结果"（概率之和恒为 1，因此它的导数必须正交于全 1 方向）。
把这一条印出来，比把 n² 个元素印出来有用得多。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from smart_research_agent.backprop import gradients, layers as layer_ops, network, verify
from smart_research_agent.backprop.types import (
    ACTIVATION_DERIVATIVE_FORMULAS,
    BACKPROP_NOTES,
    BACKPROP_NOTES_ORDER,
    BACKPROP_PROPERTIES,
    ELEMENTWISE_ACTIVATIONS,
    LayerGradients,
)
from smart_research_agent.math_foundations.types import Matrix
from smart_research_agent.neural_basics.activations import softmax
from smart_research_agent.neural_basics.layers import dense_linear, initialize
from smart_research_agent.neural_basics.types import ACTIVATIONS, DenseSpec

#: 导数表的读数点（五个逐元素激活各自取 ``x = 1.0``；``softmax`` 取一整行）.
DERIVATIVE_SAMPLE_X = 1.0

#: 导数表里 softmax 那一行用的打分.
SOFTMAX_SAMPLE_ROW: tuple[float, ...] = (1.0, 2.0, 3.0)

#: 一层反向表的样本层：``3 → 4``、relu、xavier、种子固定 ⇒ 每个读数都可重跑.
DENSE_SAMPLE = DenseSpec(3, 4, activation="relu", init="xavier", seed=41)

#: 一层反向表的输入与回传梯度（**写死**）.
DENSE_INPUTS: Matrix = ((0.5, -1.0, 2.0), (1.0, 0.25, -0.5))
DENSE_GRAD_OUTPUT: Matrix = ((0.5, -1.0, 0.25, 2.0), (-0.5, 0.75, 1.0, -0.25))


@dataclass(frozen=True)
class DerivativeRow:
    """导数表的一行：名字 + 公式 + 现场读数."""

    name: str
    formula: str
    reading: str

    def line(self) -> str:
        """``relu        = +1.000000 | d/dx = 1 if x > 0 else 0（...）``."""
        return f"{self.name:<11} {self.reading} | {self.formula}"


@dataclass(frozen=True)
class CheckRow:
    """校验表的一行：读数 + 判据 + 两个来源."""

    name: str
    reading: float
    bound: float | None
    left: str
    right: str

    def line(self) -> str:
        """``通过 activation_derivatives... | 读数 1.2e-12 ≤ 1e-06 | ...``."""
        mark = "上界" if self.bound is not None else "相等"
        bound_text = f"≤ {self.bound:.1e}" if self.bound is not None else ""
        return (
            f"{mark} {self.name:<36} | 读数 {self.reading:.3e} {bound_text} | "
            f"{self.left} vs {self.right}"
        )


@dataclass(frozen=True)
class PropertyRow:
    """性质表的一行：是否通过 + 现场读数 + 跨包对象."""

    name: str
    passed: bool
    reading: float
    cross_check: str

    def line(self) -> str:
        """``通过 activation_derivatives_match_numerical | 读数 1.2e-12 | ...``."""
        mark = "通过" if self.passed else "失败"
        return f"{mark} {self.name:<48} | 读数 {self.reading:.3e} | {self.cross_check}"


def derivative_rows() -> tuple[DerivativeRow, ...]:
    """导数表：六个激活逐行（读数来自 ``gradients``）."""
    rows: list[DerivativeRow] = []
    for name in ACTIVATIONS:
        formula = ACTIVATION_DERIVATIVE_FORMULAS[name]
        if name == "softmax":
            distribution = softmax(SOFTMAX_SAMPLE_ROW)
            jacobian = gradients.softmax_jacobian(distribution)
            row_sums = tuple(math.fsum(row) for row in jacobian)
            reading = (
                f"每行和 max|Σ J| = {max(abs(value) for value in row_sums):.3e}"
                f"（{len(jacobian)}×{len(jacobian)}）"
            )
        else:
            value = gradients.activation_derivative(name, DERIVATIVE_SAMPLE_X)
            reading = f"d/dx({DERIVATIVE_SAMPLE_X:g}) = {value:+.6f}"
        rows.append(DerivativeRow(name=name, formula=formula, reading=reading))
    return tuple(rows)


def dense_layer_gradients() -> LayerGradients:
    """一层反向表的读数：按 :data:`DENSE_SAMPLE` 造一层并做一次反向."""
    weight, bias = initialize(
        DENSE_SAMPLE.init, DENSE_SAMPLE.out_features, DENSE_SAMPLE.in_features, seed=DENSE_SAMPLE.seed
    )
    pre_activation = dense_linear(weight, bias, DENSE_INPUTS)
    hidden = gradients.elementwise_backward(DENSE_SAMPLE.activation, pre_activation, DENSE_GRAD_OUTPUT)
    grads = layer_ops.dense_backward(weight, bias, DENSE_INPUTS, hidden)
    return layer_ops.gradient_summary(grads, index=1)


def network_layer_rows() -> tuple[LayerGradients, ...]:
    """网络反向表：MLP ``3 → 5 → 2`` 的逐层三个范数（读数来自 ``network``）."""
    cache = network.mlp_forward_with_cache(verify.MLP_SPEC, verify.MLP_INPUTS)
    trace = network.mlp_loss_gradients(cache, verify.MLP_TARGETS)
    return trace.layer_summaries()


def check_rows() -> tuple[CheckRow, ...]:
    """校验表：七条性质各自的读数与判据（读数来自 :func:`verify.check_all`）."""
    report = verify.check_all()
    rows: list[CheckRow] = []
    for outcome in report.outcomes:
        check = outcome.cross_check
        reading = check.reading if check is not None else 0.0
        bound = check.upper_bound if check is not None else None
        left = check.left if check is not None else "-"
        right = check.right if check is not None else "-"
        rows.append(CheckRow(name=outcome.name, reading=reading, bound=bound, left=left, right=right))
    return tuple(rows)


def property_rows() -> tuple[PropertyRow, ...]:
    """性质表：七条性质逐行（是否通过 / 读数 / 跨包对象）."""
    report = verify.check_all()
    rows: list[PropertyRow] = []
    for outcome in report.outcomes:
        check = outcome.cross_check
        reading = check.reading if check is not None else 0.0
        cross = f"{check.left} vs {check.right}" if check is not None else "-"
        rows.append(
            PropertyRow(
                name=outcome.name,
                passed=outcome.passed,
                reading=reading,
                cross_check=cross,
            )
        )
    return tuple(rows)


def note_lines(limit: int | None = None) -> tuple[str, ...]:
    """十条笔记逐行印出（顺序即写入顺序）."""
    keys = BACKPROP_NOTES_ORDER if limit is None else BACKPROP_NOTES_ORDER[:limit]
    return tuple(f"{index:>2}. {BACKPROP_NOTES[key]}" for index, key in enumerate(keys, start=1))


def study_lines() -> tuple[str, ...]:
    """一次跑完五张表（演示脚本与教程引用的是同一批读数）."""
    lines: list[str] = []
    lines.append("== 1. 导数表（六个激活的公式与现场读数）")
    for row in derivative_rows():
        lines.append("  " + row.line())
    lines.append(f"  逐元素激活 {len(ELEMENTWISE_ACTIVATIONS)} 个；softmax 是唯一的逐行激活")
    lines.append("== 2. 一层反向表（dense 3→4 + relu：三块梯度的形状与范数）")
    lines.append("  " + dense_layer_gradients().line())
    lines.append("== 3. 网络反向表（MLP 3→5→2：逐层的三个范数）")
    for row in network_layer_rows():
        lines.append("  " + row.line())
    lines.append("== 4. 校验表（七条性质：读数与判据）")
    for row in check_rows():
        lines.append("  " + row.line())
    lines.append("== 5. 性质表（七条性质：是否通过 / 读数 / 跨包对象）")
    for row in property_rows():
        lines.append("  " + row.line())
    return tuple(lines)


#: 本课的性质名单（供报告核对：表里的行数必须等于它）.
PROPERTY_NAMES = BACKPROP_PROPERTIES

__all__ = [
    "DERIVATIVE_SAMPLE_X",
    "DENSE_GRAD_OUTPUT",
    "DENSE_INPUTS",
    "DENSE_SAMPLE",
    "PROPERTY_NAMES",
    "SOFTMAX_SAMPLE_ROW",
    "CheckRow",
    "DerivativeRow",
    "PropertyRow",
    "check_rows",
    "dense_layer_gradients",
    "derivative_rows",
    "network_layer_rows",
    "note_lines",
    "property_rows",
    "study_lines",
]
