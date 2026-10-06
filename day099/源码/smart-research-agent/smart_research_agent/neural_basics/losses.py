"""三个损失函数与它们的**稳定路径**（day089 / M8-D1）.

## 一、三个损失

```text
mse             mean((pred − target)²)          预测等于目标时**恰好**为 0.0
mae             mean(|pred − target|)           对离群点比 MSE 温和
cross_entropy   −log softmax(logits)[target]    从 logits 出发，标签越界当场拒绝
```

## 二、交叉熵的两条路径（一条是稳定路径，一条用来对照）

```text
路径 A（稳定）    −log_softmax(logits)[target]
路径 B（直白）    −log(softmax(logits)[target])
```

两条路径在正常区间上一致；在"某个概率极小"时会分家：路径 B 的
``softmax`` 先下溢成 ``0.0``、再取 ``log`` 得到 ``-inf``，而路径 A 的
``log_softmax`` 在同样的输入下仍给出**有限值**。

## 三、为什么 ``log_softmax`` 沿用 ``sft.loss`` 的求和方式

本模块的 ``log_softmax`` 用**内置 ``sum``**，与生产实现 ``sft.loss.log_softmax`` 逐字对齐——
因此"本包 vs 项目里已有的交叉熵"这条对账可以**逐位**比较。
``activations.softmax`` 则用 ``math.fsum``、对齐 day073 的
``math_foundations.linalg.softmax``。两种求和各自服务一个对账对象，这不是重复实现。

## 四、一个量只写一遍

``softmax`` 只在 :mod:`activations` 里实现一次；本模块从那里 import 它，
只新增 ``log_softmax``（另一条稳定路径）。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from smart_research_agent.math_foundations.types import Matrix, Vector
from smart_research_agent.neural_basics.activations import softmax
from smart_research_agent.neural_basics.errors import LossError, NumericError

#: 困惑度的溢出保护：``exp(700)`` 已接近 float 上限，超过它困惑度已无信息量.
PERPLEXITY_CEILING = 700.0


def _flatten(predictions: Matrix, targets: Matrix, *, name: str) -> tuple[float, ...]:
    """把两份张量摊平并做形状 / 有限性护栏（形状不一致或为空抛 ``LossError``）."""
    if len(predictions) != len(targets):
        raise LossError(f"{name}：预测有 {len(predictions)} 行、目标有 {len(targets)} 行，数量不一致。")
    if not predictions:
        raise LossError(f"{name}：样本为空——对空样本求损失没有定义，不能静默返回 0.0。")
    flat: list[float] = []
    for index, (pred_row, target_row) in enumerate(zip(predictions, targets, strict=True)):
        if len(pred_row) != len(target_row):
            raise LossError(f"{name}：第 {index} 行宽度 {len(pred_row)} 与目标 {len(target_row)} 不一致。")
        for pred_value, target_value in zip(pred_row, target_row, strict=True):
            pred_float = float(pred_value)
            target_float = float(target_value)
            if not math.isfinite(pred_float) or not math.isfinite(target_float):
                raise NumericError(f"{name}：第 {index} 行出现非有限数（nan / inf）。")
            flat.append(pred_float - target_float)
    return tuple(flat)


def mse(predictions: Matrix, targets: Matrix) -> float:
    """均方误差 ``mean((pred − target)²)``（预测等于目标时**恰好**是 0.0）."""
    differences = _flatten(predictions, targets, name="mse")
    return math.fsum(value * value for value in differences) / len(differences)


def mae(predictions: Matrix, targets: Matrix) -> float:
    """平均绝对误差 ``mean(|pred − target|)``."""
    differences = _flatten(predictions, targets, name="mae")
    return math.fsum(abs(value) for value in differences) / len(differences)


def log_softmax(logits: Sequence[float]) -> Vector:
    """数值稳定的 log-softmax：``z − max − log Σ e^{z − max}``（求和方式对齐 ``sft.loss``）.

    直接算 ``log(softmax(z))`` 在概率极小时会先下溢成 ``0.0``、再取对数得到 ``-inf``；
    本表达式在同样的输入下仍给出有限值。空输入抛 :class:`LossError`。
    """
    values = tuple(float(value) for value in logits)
    if not values:
        raise LossError("log_softmax 的输入不能为空：对空 logits 求对数没有定义。")
    if any(not math.isfinite(value) for value in values):
        raise NumericError("log_softmax 的输入必须是有限数（收到 nan / inf）。")
    largest = max(values)
    shifted = [value - largest for value in values]
    log_total = math.log(sum(math.exp(value) for value in shifted))
    return tuple(value - log_total for value in shifted)


def _checked_target(logits: Sequence[float], target: int) -> int:
    """标签护栏：必须是 ``[0, len)`` 内的整数（越界抛 ``LossError``）."""
    if isinstance(target, bool) or not isinstance(target, int):
        raise LossError(f"标签必须是整数，收到 {target!r}。")
    if not 0 <= target < len(logits):
        raise LossError(
            f"标签 {target} 落在 [0, {len(logits)}) 之外：越界标签若被静默兜成 0，"
            "模型只是'没有在学'，而报告里看不出是标签错了。"
        )
    return target


def cross_entropy(logits: Sequence[float], target: int) -> float:
    """单点交叉熵 ``−log softmax(logits)[target]``（**稳定路径**）."""
    values = tuple(float(value) for value in logits)
    if not values:
        raise LossError("cross_entropy 的 logits 不能为空。")
    checked = _checked_target(values, target)
    return -log_softmax(values)[checked]


def cross_entropy_via_probability(logits: Sequence[float], target: int) -> float:
    """单点交叉熵的**直白路径** ``−log(softmax(logits)[target])``.

    它与 :func:`cross_entropy` 在正常区间上一致，用来做"两条路径"的对照。
    概率下溢成 0 时抛 :class:`LossError`——"概率为 0 取 log"没有定义，
    不能返回 ``inf`` 让它混进平均里。
    """
    values = tuple(float(value) for value in logits)
    if not values:
        raise LossError("cross_entropy_via_probability 的 logits 不能为空。")
    checked = _checked_target(values, target)
    probability = softmax(values)[checked]
    if probability <= 0.0:
        raise LossError(
            f"概率为 0 取 log 没有定义（logits={values}、target={checked}）："
            "这一条必须显式失败，而不是把 -inf 混进平均值里。"
        )
    return -math.log(probability)


def accuracy(logits_seq: Sequence[Sequence[float]], targets: Sequence[int]) -> float:
    """下一 token 命中率：``argmax(logits) == target`` 的比例（空样本抛 ``LossError``）."""
    if len(logits_seq) != len(targets):
        raise LossError(f"logits 与标签数量不一致：{len(logits_seq)} != {len(targets)}。")
    if not logits_seq:
        raise LossError("accuracy：样本为空，命中率没有定义。")
    hits = 0
    for logits, target in zip(logits_seq, targets, strict=True):
        values = tuple(float(value) for value in logits)
        checked = _checked_target(values, target)
        best = 0
        for index in range(1, len(values)):
            if values[index] > values[best]:
                best = index
        hits += int(best == checked)
    return hits / len(logits_seq)


def perplexity(loss: float) -> float:
    """困惑度 ``exp(loss)``——"模型在每一步平均在多少个候选之间犹豫"（负损失抛 ``LossError``）."""
    if not math.isfinite(loss) or loss < 0.0:
        raise LossError(f"loss 必须是非负有限数，收到 {loss!r}。")
    if loss > PERPLEXITY_CEILING:  # pragma: no cover - 防御式分支：exp(700) 已接近 float 上限
        return math.inf
    return math.exp(loss)


__all__ = [
    "PERPLEXITY_CEILING",
    "accuracy",
    "cross_entropy",
    "cross_entropy_via_probability",
    "log_softmax",
    "mae",
    "mse",
    "perplexity",
]
