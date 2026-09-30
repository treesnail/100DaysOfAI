"""``training_optim`` 的 Dropout：一个掩码、两个相（day081 / M7-D6）.

## 它只做一件事

```text
训练相    每个元素以概率 p 被置零，活下来的元素**乘上 1/(1−p)**
推理相    恒等（原样返回）
```

那个 ``1/(1−p)`` 是"inverted dropout"的全部内容，也是它比原版更好的地方：

```text
原版（2012）      训练时只置零；**推理时**把整层乘 (1−p)
inverted（现在）  训练时置零并放大；推理时什么也不做
```

两种写法在期望上等价（``E[输出]`` 都等于输入）。差别在**代码**：

```text
原版      推理路径必须知道 p        ⇒ 每个调用点都要多一个参数
inverted  推理路径与"没有 dropout 的模型"**逐位相同**  ⇒ 没有任何一处需要特判
```

本包选 inverted，并把那条等式写成一条可验证的性质（第 3 节）：
**``E[掩码 × 缩放] = 1``**。

## 掩码是确定性的

掩码用 day073 的那串 LCG（``uniforms(count, seed=...)``）生成，
因此"同一颗种子 ⇒ 同一个掩码"。这不是为了让 dropout 可预测，
而是为了让**整个实验可复现**：一份训练报告如果没有它，
"这次的损失曲线"就只能被相信，不能被复核。

## 一条必须写下来的边界：掩码被固定之后，梯度是**可以**逐项校验的

```text
通常的说法    "dropout 是随机的，所以它的梯度没法用数值差分校验"
准确的说法    掩码一旦**被固定**（记在账里），整个算子就是"逐元素乘以一个常数"——
             一个线性的、对角的东西，因此它的梯度完全能用中心差分逐项核对
```

本包把掩码记在 ``DropoutReport`` 里，正是为了让这件事成立（见 :func:`dropout_backward`）。
"""

from __future__ import annotations

import math
from typing import Any

from smart_research_agent.math_foundations.probability import uniforms
from smart_research_agent.math_foundations.types import (
    Matrix,
    matrix_shape,
    validate_matrix,
)
from smart_research_agent.training_optim.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)

#: 两个相的名字（与 day081 的 ``PHASES`` 逐键对齐）.
PHASE_TRAIN = "train"
PHASE_EVAL = "eval"

PHASES: tuple[str, ...] = (PHASE_TRAIN, PHASE_EVAL)

PHASE_DESCRIPTIONS: dict[str, str] = {
    PHASE_TRAIN: "训练相：置零 + 放大 1/(1−p)（掩码由种子决定，因此可复现）",
    PHASE_EVAL: "推理相：恒等（inverted dropout 把缩放放在训练侧，推理路径因此不需要知道 p）",
}

#: 默认的丢弃概率（**0 是基线**：没有它就说不清 dropout 有没有用）.
DEFAULT_DROPOUT_RATE = 0.0


def _checked_rate(rate: Any) -> float:
    """丢弃概率：必须落在 ``[0, 1)``.

    ``1.0`` 被拒的理由是可算的：那个 ``1/(1−p)`` 会变成除零，
    而"整层全丢"的模型没有任何可学的东西。
    """
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise ParameterError(f"rate 必须是数，收到 {rate!r}。")
    value = float(rate)
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ParameterError(
            f"rate 必须落在 [0, 1)，收到 {value!r}："
            "等于 1 时 1/(1−p) 会除零（整层全丢的模型没有可学的东西），"
            "大于 1 则连'概率'都不是。"
        )
    return value


def _checked_phase(phase: Any) -> str:
    """相的名字必须是 ``train`` / ``eval`` 之一（**不给它挑一个默认值**）."""
    if phase not in PHASES:
        raise ParameterError(f"不认识的相 {phase!r}：可用取值 {list(PHASES)}。")
    return str(phase)


def expected_keep(rate: float) -> float:
    """期望的保留比例 ``1 − p``（与 :func:`kept_fraction` 的实测值对照用）."""
    return 1.0 - _checked_rate(rate)


def dropout_mask(rows: int, columns: int, *, rate: float, seed: int) -> Matrix:
    """确定性的 0/1 掩码：``u >= rate`` 的元素活下来（``u`` 来自种子固定的 LCG）."""
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ParameterError(f"掩码行数必须是 >= 1 的整数，收到 {rows!r}。")
    if isinstance(columns, bool) or not isinstance(columns, int) or columns < 1:
        raise ParameterError(f"掩码列数必须是 >= 1 的整数，收到 {columns!r}。")
    resolved = _checked_rate(rate)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ParameterError(f"seed 必须是整数，收到 {seed!r}。")
    raw = uniforms(rows * columns, seed=seed)
    kept = tuple(1.0 if value >= resolved else 0.0 for value in raw)
    return tuple(
        kept[index * columns : (index + 1) * columns] for index in range(rows)
    )


def kept_fraction(mask: Matrix) -> float:
    """掩码里活下来的比例（一个可读的实测读数）."""
    checked = validate_matrix(mask, name="mask")
    total = sum(len(row) for row in checked)
    if total == 0:
        raise NumericError("空掩码没有'保留比例'这回事。")
    kept = sum(1 for row in checked for value in row if value != 0.0)
    return kept / total


def dropout_forward(
    inputs: Matrix,
    *,
    rate: float,
    seed: int,
    phase: str = PHASE_TRAIN,
) -> tuple[Matrix, Matrix, float]:
    """``(输出, 掩码, 缩放)``.

    **三个返回值都给出**，而不是只给输出：反向要用掩码与缩放，
    而"这一层到底丢了哪些单元"必须可读——只给输出的话，
    "训练与推理的差别有多大"就只能靠猜。
    """
    checked = validate_matrix(inputs, name="inputs")
    resolved_rate = _checked_rate(rate)
    resolved_phase = _checked_phase(phase)
    rows, columns = matrix_shape(checked)
    if resolved_phase == PHASE_EVAL or resolved_rate == 0.0:
        ones = tuple(tuple(1.0 for _ in range(columns)) for _ in range(rows))
        return checked, ones, 1.0
    scale = 1.0 / (1.0 - resolved_rate)
    mask = dropout_mask(rows, columns, rate=resolved_rate, seed=seed)
    return (
        tuple(
            tuple(value * flag * scale for value, flag in zip(row, flags, strict=True))
            for row, flags in zip(checked, mask, strict=True)
        ),
        mask,
        scale,
    )


def dropout_backward(grad_output: Matrix, mask: Matrix, scale: float) -> Matrix:
    """Dropout 的反向：``dInput = dOutput × 掩码 × 缩放``.

    它就是本课最该被看懂的一行：

```text
前向    y = x ⊙ m · s        （⊙ 逐元素、m 是 0/1、s 是标量）
反向    dx = dy ⊙ m · s
```

    而它的**正确性可以被数值差分逐项核对**——条件是掩码固定。
    也就是说 ``m`` 不是一个随机变量，而是一个已知的常数矩阵；
    于是这个算子是一个对角的线性映射，中心差分会给出与它逐项对上的答案。
    """
    checked_grad = validate_matrix(grad_output, name="grad_output")
    checked_mask = validate_matrix(mask, name="mask")
    if matrix_shape(checked_grad) != matrix_shape(checked_mask):
        raise ShapeError(
            f"梯度 {matrix_shape(checked_grad)} 与掩码 {matrix_shape(checked_mask)} "
            "不同形：掩码是逐元素的，两者的形状必须逐位对齐。"
        )
    if not isinstance(scale, (int, float)) or not math.isfinite(float(scale)):
        raise NumericError(f"scale 必须是有限数，收到 {scale!r}。")
    factor = float(scale)
    return tuple(
        tuple(slope * flag * factor for slope, flag in zip(row, flags, strict=True))
        for row, flags in zip(checked_grad, checked_mask, strict=True)
    )


def scale_of(rate: float, *, phase: str = PHASE_TRAIN) -> float:
    """给定概率与相，返回放大系数（训练相 ``1/(1−p)``、推理相 ``1.0``）."""
    resolved_phase = _checked_phase(phase)
    resolved_rate = _checked_rate(rate)
    if resolved_phase == PHASE_EVAL or resolved_rate == 0.0:
        return 1.0
    return 1.0 / (1.0 - resolved_rate)


__all__ = [
    "DEFAULT_DROPOUT_RATE",
    "PHASES",
    "PHASE_DESCRIPTIONS",
    "PHASE_EVAL",
    "PHASE_TRAIN",
    "dropout_backward",
    "dropout_forward",
    "dropout_mask",
    "expected_keep",
    "kept_fraction",
    "scale_of",
]
