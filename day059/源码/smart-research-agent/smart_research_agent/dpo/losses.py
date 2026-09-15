"""三种可以对同一份偏好数据使用的损失函数（M5-D7）.

day054 只讲一条公式（DPO 的 sigmoid loss），因为那一课要讲清"偏好怎么变成
目标函数"。day055 要真的开一次训练，就必须回答一个工程问题：
**``DPOConfig(loss_type=...)`` 到底允许填什么，填不同的值会改变什么**。

本模块把 TRL ``DPOConfig.loss_type`` 里最常用的三档做成**可手算核对**的实现：

===================================  ==================================================  ==================
``loss_type``                        损失（``m`` = 策略对数比之差 − 参考对数比之差）            起点值（``m = 0``）
===================================  ==================================================  ==================
``sigmoid``                          ``−log σ(β·m)``                                     ``ln 2 ≈ 0.693147``
``hinge``                            ``max(0, 1 − β·m)``                                 ``1``
``ipo``                              ``(m − 1/(2β))²``                                   ``1/(4β²)``
===================================  ==================================================  ==================

**最后那一列是本课最值得记住的一行**：day054 的"起点自检 = ``ln 2``"只对
``sigmoid`` 成立。``β = 0.1`` 时 ``hinge`` 的起点是 ``1``、``ipo`` 的起点是
``25``——如果把"未训练时 loss 应当约等于 0.6931"写成训练脚本里的断言，
换成 ``ipo`` 之后它会立刻失败，而失败信息看起来像"参考模型接错了"。
公式变了，自检的期望值也必须跟着变，这就是 ``zero_margin_loss`` 存在的理由。

三个函数都做同一件事的两种表示：``*_loss`` 给损失、``*_gradient`` 给
``∂L/∂m``（训练器要用它把梯度缩放到 chosen / rejected 两侧）。两者必须
自洽，测试里用**中心差分**核对：``(L(m+h) − L(m−h)) / 2h`` 与解析梯度
在 ``h = 1e-6`` 上逐位相符。
"""

from __future__ import annotations

import math
from typing import Any

from smart_research_agent.alignment.objectives import log_sigmoid, sigmoid
from smart_research_agent.dpo.errors import DPOError

#: 允许的损失函数名（与 TRL ``DPOConfig.loss_type`` 的最常用三档同名）.
#:
#: 刻意 **不** 把 TRL 支持的全部取值（``kto_pair`` / ``bco_pair`` /
#: ``apo_zero`` / ``discopop`` …）都列进来：那些方法各自需要额外的数据或
#: 超参（KTO 需要"只标好/坏"的单侧数据、BCO 要成对奖励），把它们并列在
#: 同一张表里会给人"随便换一个就行"的错觉。**本课程只讲能被同一份
#: (prompt, chosen, rejected) 数据直接驱动的三种**。
LOSS_TYPES: tuple[str, ...] = ("sigmoid", "hinge", "ipo")

#: 默认损失函数（TRL 的默认值也是它，理由见 ``loss_table`` 的说明）.
DEFAULT_LOSS_TYPE = "sigmoid"

#: 起点自检时允许的浮点容差（与 day054 的 ``ZERO_MARGIN_TOLERANCE`` 同量级）.
GRADIENT_CHECK_STEP = 1e-6


def _check_beta(beta: float) -> None:
    """β 必须为正：它是 KL 约束的强度，取 0 会让 DPO 退化成"只看偏好"."""
    if beta <= 0:
        raise DPOError(f"beta 必须为正数（KL 约束强度），收到 {beta}")


def check_loss_type(loss_type: str) -> str:
    """校验损失函数名并原样返回（便于写成表达式）."""
    if loss_type not in LOSS_TYPES:
        raise DPOError(
            f"未知的损失函数 {loss_type!r}，可选：{', '.join(LOSS_TYPES)}"
        )
    return loss_type


def sigmoid_dpo_loss(margin: float, *, beta: float) -> float:
    """``−log σ(β·m)``：DPO 论文（Rafailov et al., 2023）的原始损失.

    ``m`` 是"策略的对数比之差减参考模型的对数比之差"。写成对数比的形式
    是为了让**隐式奖励**（day054 的 ``implicit_reward``）在数学上等价于
    "在奖励模型下做 Bradley-Terry 二分类"，于是不需要真的训一个奖励模型。
    """
    _check_beta(beta)
    return -log_sigmoid(beta * margin)


def sigmoid_dpo_gradient(margin: float, *, beta: float) -> float:
    """``∂L/∂m = −β·σ(−β·m)``.

    符号与"loss 随 margin 增大而减小"一致：margin 已经很大时梯度趋于 0，
    这就是 sigmoid loss 的"**一旦分对就不再使劲**"——它也是过优化曲线的
    数学来源（训练 margin 涨到一定程度后，留出收益不再增加）。
    """
    _check_beta(beta)
    return -beta * sigmoid(-beta * margin)


def hinge_dpo_loss(margin: float, *, beta: float) -> float:
    """``max(0, 1 − β·m)``：来自 SLiC-HF / RSO 的合页损失.

    **这里 β 的语义变了**：在 sigmoid 里 β 是"温度"，越大越激进；在 hinge
    里 ``1/β`` 是**间隔阈值**——margin 超过 ``1/β`` 就完全不再产生梯度。
    换损失函数时超参的语义会跟着换，这是"换个 loss_type 试试"最容易被
    忽略的代价（``loss_table`` 把两列并列，就是为了让这件事看得见）。
    """
    _check_beta(beta)
    return max(0.0, 1.0 - beta * margin)


def hinge_dpo_gradient(margin: float, *, beta: float) -> float:
    """``∂L/∂m = −β``（margin 不足时）或 ``0``（已越过间隔）.

    间断点在 ``β·m = 1``：真实实现（TRL 用 ``torch.relu``）在不可导点取
    次梯度 0，本实现保持一致——若取 ``−β``，测试里的中心差分会在该点
    两侧给出不同的结果，掩盖真正的符号错误。
    """
    _check_beta(beta)
    if beta * margin < 1.0:
        return -beta
    return 0.0


def ipo_dpo_loss(margin: float, *, beta: float) -> float:
    """``(m − 1/(2β))²``：IPO（Azar et al., 2023）把对齐写成回归问题.

    关键差别在于**它有一个非零的目标 margin**：``m* = 1/(2β)``。sigmoid
    损失把 margin 推得越正越好（"赢就够了，赢多少都要"），而 IPO 认为
    这会让 logits 无界增长、在留出集上过拟合；它改成"把 margin 拉到
    ``1/(2β)`` 就停"——**这是"过优化"在损失函数层面的解法**，也是
    day054 ``optimization_summary`` 那三条告警在算法侧的对照。
    """
    _check_beta(beta)
    return (margin - 1.0 / (2.0 * beta)) ** 2


def ipo_dpo_gradient(margin: float, *, beta: float) -> float:
    """``∂L/∂m = 2·(m − 1/(2β))``：在目标 margin 处过零，两侧反向."""
    _check_beta(beta)
    return 2.0 * (margin - 1.0 / (2.0 * beta))


def loss_value(loss_type: str, margin: float, *, beta: float) -> float:
    """按名字取损失值（未知名字抛 ``DPOError``）."""
    check_loss_type(loss_type)
    if loss_type == "sigmoid":
        return sigmoid_dpo_loss(margin, beta=beta)
    if loss_type == "hinge":
        return hinge_dpo_loss(margin, beta=beta)
    return ipo_dpo_loss(margin, beta=beta)


def loss_gradient(loss_type: str, margin: float, *, beta: float) -> float:
    """按名字取 ``∂L/∂m``（训练器的梯度缩放系数就是它的相反数）."""
    check_loss_type(loss_type)
    if loss_type == "sigmoid":
        return sigmoid_dpo_gradient(margin, beta=beta)
    if loss_type == "hinge":
        return hinge_dpo_gradient(margin, beta=beta)
    return ipo_dpo_gradient(margin, beta=beta)


def zero_margin_loss(loss_type: str, *, beta: float) -> float:
    """``m = 0``（策略与参考模型逐位相同）时的损失值——**起点自检的期望值**."""
    return loss_value(loss_type, 0.0, beta=beta)


def zero_margin_losses(*, beta: float) -> dict[str, float]:
    """三种损失在 ``m = 0`` 时的取值（进 API 响应与文档表格）."""
    _check_beta(beta)
    return {name: zero_margin_loss(name, beta=beta) for name in LOSS_TYPES}


def numeric_gradient(
    loss_type: str, margin: float, *, beta: float, step: float = GRADIENT_CHECK_STEP
) -> float:
    """中心差分求 ``∂L/∂m``，用于**核对解析梯度**（不参与训练）.

    把它放进生产代码而不是只放进测试，是因为它同时是文档的一部分：
    ``scripts/dpo_demo.py`` 会把它打出来，让"梯度确实是我写下的那个式子"
    这件事在学员眼前发生一次，而不是只在一句注释里被断言。
    """
    if step <= 0:
        raise DPOError(f"差分步长必须为正数，收到 {step}")
    forward = loss_value(loss_type, margin + step, beta=beta)
    backward = loss_value(loss_type, margin - step, beta=beta)
    return (forward - backward) / (2.0 * step)


def loss_table(*, beta: float = 0.1) -> list[dict[str, Any]]:
    """三种损失的对照表（名字 / 公式 / 梯度 / 起点值 / 适用情形）.

    表格里的每个数字都由本模块的函数算出来，**不是抄进文档的常量**——
    文档与实现一旦分家，最先过期的一定是文档。
    """
    rows: list[dict[str, Any]] = []
    for name in LOSS_TYPES:
        rows.append(
            {
                "loss_type": name,
                "formula": _FORMULAS[name],
                "gradient": _GRADIENTS[name],
                "zero_margin_loss": round(zero_margin_loss(name, beta=beta), 6),
                "beta_role": _BETA_ROLES[name],
                "note": _NOTES[name],
            }
        )
    return rows


_FORMULAS: dict[str, str] = {
    "sigmoid": "L = -log σ(β·m)",
    "hinge": "L = max(0, 1 - β·m)",
    "ipo": "L = (m - 1/(2β))^2",
}

_GRADIENTS: dict[str, str] = {
    "sigmoid": "dL/dm = -β·σ(-β·m)",
    "hinge": "dL/dm = -β（β·m < 1）否则 0",
    "ipo": "dL/dm = 2·(m - 1/(2β))",
}

_BETA_ROLES: dict[str, str] = {
    "sigmoid": "温度：β 越大越激进，隐式奖励被放大 β 倍",
    "hinge": "间隔的倒数：margin 超过 1/β 后完全不再产生梯度",
    "ipo": "目标 margin 的倒数：把 margin 拉到 1/(2β) 就停",
}

_NOTES: dict[str, str] = {
    "sigmoid": "默认档；起点自检值 ln2，是唯一能与 day054 的 ZERO_MARGIN_LOSS 直接对照的一档",
    "hinge": "对噪声标注更宽容（越界样本不再贡献梯度），但间隔内外的梯度不连续",
    "ipo": "抑制过优化；起点值 1/(4β²) 不是 ln2，换档必须同时换自检期望值",
}


def describe_loss(loss_type: str, *, beta: float = 0.1) -> dict[str, Any]:
    """单个损失函数的说明（供 API 与调试使用）."""
    return next(row for row in loss_table(beta=beta) if row["loss_type"] == loss_type)


def expected_zero_margin_tolerance(expected: float) -> float:
    """起点自检的容差：与期望值同量级的浮点噪声，但不放过真实的接线错误.

    取 ``max(1e-9, |expected|·1e-9)``：``ipo`` 在 ``β = 0.01`` 时的期望值是
    ``2500``，此时绝对容差 ``1e-9`` 会被浮点噪声直接击穿。
    """
    if not math.isfinite(expected):
        raise DPOError(f"期望值必须是有限数，收到 {expected}")
    return max(1e-9, abs(expected) * 1e-9)


__all__ = [
    "DEFAULT_LOSS_TYPE",
    "GRADIENT_CHECK_STEP",
    "LOSS_TYPES",
    "check_loss_type",
    "describe_loss",
    "expected_zero_margin_tolerance",
    "hinge_dpo_gradient",
    "hinge_dpo_loss",
    "ipo_dpo_gradient",
    "ipo_dpo_loss",
    "loss_gradient",
    "loss_table",
    "loss_value",
    "numeric_gradient",
    "sigmoid_dpo_gradient",
    "sigmoid_dpo_loss",
    "zero_margin_loss",
    "zero_margin_losses",
]
