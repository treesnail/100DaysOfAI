"""``techniques``：四个旋钮的**转发层**（day095 / M8-D6）.

这一模块刻意写得薄：它不重新推导任何公式，只做三件事——

```text
dropout           把一个行向量塞进 day081 的 ``training_optim.dropout``
learning_rate_at  把名字与参数交给 day074 的 ``math_foundations.optim.make_schedule``
early stop        转发 day081 的 ``training_optim.controls.EarlyStopping``
RegularizationConfig  把"这一次训练开了哪些旋钮"收成一份可读的配置
```

## 为什么是转发而不是重写

day074 已经推过、测过四种调度；day081 已经写过 dropout 与前向/反向、以及一个
带 ``min_delta`` 的早停。本课重写一遍的代价不是"多写 100 行"，而是
**两处口径会慢慢分开**：某一天有人在一边改了公式，另一边还是旧的，
于是"余弦退火"在两个地方指两条曲线。

因此本课只在 `normalization.py` 里写了**一件新的数学**（BatchNorm），
其余全是转发——而 :data:`types.TECHNIQUE_SOURCES` 把"谁实现了它"写成了一张可核对的表。

## 一个刻意的收窄：dropout 只作用于**向量**

day081 的 dropout 作用在 ``Matrix`` 上（一层激活之后的整张特征图）。
本课需要的是"对最后一步的隐状态做 dropout"——它是一行 ``H`` 个数。
于是 :func:`dropout_vector` 把它包成 ``1 × H`` 再调用 day081：
**形状换了，数学没变**。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from smart_research_agent.math_foundations.optim import make_schedule
from smart_research_agent.math_foundations.types import SCHEDULES, Vector
from smart_research_agent.regularization.errors import (
    NumericError,
    ParameterError,
    PhaseError,
)
from smart_research_agent.regularization.normalization import as_vector
from smart_research_agent.regularization.types import PHASES, TECHNIQUES
from smart_research_agent.training_optim.controls import EarlyStopping
from smart_research_agent.training_optim.dropout import (
    PHASE_TRAIN,
    dropout_backward,
    dropout_forward,
)
from smart_research_agent.training_optim.types import EarlyStopReport, TrainingConfig  # noqa: F401


def _checked_rate(rate: Any) -> float:
    """丢弃概率必须落在 ``[0, 1)``（与 day081 同一条判据）."""
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise ParameterError(f"rate 必须是数，收到 {rate!r}。")
    value = float(rate)
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ParameterError(
            f"rate 必须落在 [0, 1)，收到 {value!r}："
            "等于 1 时 1/(1−p) 会除零（整层全丢的模型没有可学的东西）。"
        )
    return value


def _checked_phase(phase: Any) -> str:
    """相的名字必须是 ``train`` / ``eval`` 之一（**不给它挑一个默认值**）."""
    if phase not in PHASES:
        raise PhaseError(f"不认识的相 {phase!r}：可用取值 {list(PHASES)}。")
    return str(phase)


def dropout_vector(
    vector: Vector, *, rate: float, seed: int, phase: str = PHASE_TRAIN
) -> tuple[Vector, tuple[tuple[float, ...], ...], float]:
    """对**一行**隐状态做 dropout，返回 ``(输出, 掩码, 缩放)``.

    实现只有一行：把它包成 ``1 × H`` 交给 day081 的 :func:`dropout_forward`。
    三个返回值都留下，因此反向不需要重算掩码（"掩码固定之后梯度可校验"）。
    """
    checked = as_vector(vector, name="vector")
    resolved_phase = _checked_phase(phase)
    resolved_rate = _checked_rate(rate)
    output, mask, scale = dropout_forward(
        (checked,), rate=resolved_rate, seed=seed, phase=resolved_phase
    )
    return output[0], mask, scale


def dropout_vector_backward(
    gradient: Vector, mask: tuple[tuple[float, ...], ...], scale: float
) -> Vector:
    """对**一行**隐状态做 dropout 的反向（转发 day081 的 :func:`dropout_backward`）."""
    checked = as_vector(gradient, name="gradient")
    return dropout_backward((checked,), mask, scale)[0]


def dropout_vector_backward_matrix(
    gradient: tuple[tuple[float, ...], ...],
    mask: tuple[tuple[float, ...], ...],
    scale: float,
) -> tuple[tuple[float, ...], ...]:
    """对**一整批**做 dropout 的反向（转发 day081 的 :func:`dropout_backward`）.

    批版本与单行版本是同一个函数——只是这里不做那层 ``1 × H`` 的包装。
    """
    return dropout_backward(gradient, mask, scale)


def learning_rate_at(
    step: int, *, base_lr: float, schedule: str, **schedule_kwargs: Any
) -> float:
    """第 ``step`` 步的学习率（**转发 day074 的调度**，步号 1-based）.

    学习率允许是 **0**：``warmup_cosine`` 的 ``min_lr`` 默认 0，
    因此退火的最后一步拿到 0 是它故意的端点（那一步不做更新）。
    本包只拒绝负数与非有限数。
    """
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ParameterError(f"step 必须是 >= 1 的整数，收到 {step!r}。")
    if isinstance(base_lr, bool) or not isinstance(base_lr, (int, float)):
        raise ParameterError(f"base_lr 必须是数，收到 {base_lr!r}。")
    if schedule not in SCHEDULES:
        raise ParameterError(
            f"不认识的学习率调度 {schedule!r}：可用取值 {list(SCHEDULES)}"
            "——这一层先把名字挡住，否则报出来的会是 day074 那一族的异常。"
        )
    try:
        curve = make_schedule(schedule, base_lr=float(base_lr), **schedule_kwargs)
        value = float(curve(step))
    except TypeError as error:
        raise ParameterError(
            f"调度 {schedule!r} 的参数对不上：{error}——"
            "本包不替调用方补默认参数（那样'这一次到底用了什么'就不可读了）。"
        ) from error
    if not math.isfinite(value) or value < 0.0:  # pragma: no cover - day074 已在调度层拒绝负数
        raise NumericError(
            f"第 {step} 步的学习率是 {value!r}：调度必须给出**非负的有限数**——"
            "负数会把下降变成上升，而它不会报错，只会让损失一路变大。"
            "（day074 的四种调度都已经拒绝负数，这是第二道皮带。）"
        )
    return value


def early_stopping_watch(
    losses: tuple[float, ...], *, patience: int, min_delta: float = 0.0
) -> EarlyStopReport:
    """把一串损失喂给 day081 的 :class:`EarlyStopping`，返回它的报告.

    **整段一次喂完**（而不是在训练回路里逐步喂）是刻意的：
    这样"早停在第几步触发"这件事与训练回路完全解耦，可以被单独断言
    （见 ``verify.check_early_stop`` 与第 ⑦ 条性质的邻居）。
    """
    if not losses:
        raise ParameterError("空损失曲线没有早停可言。")
    watcher = EarlyStopping(patience, min_delta=min_delta)
    for index, loss in enumerate(losses, start=1):
        watcher.update(index, loss)
    return watcher.report()


@dataclass(frozen=True)
class RegularizationConfig:
    """"这一次训练开了哪些旋钮"——一份可读的配置（**它进报告，不进服务级配置**）."""

    norm: bool = True
    bn_momentum: float = 0.1
    dropout_rate: float = 0.2
    dropout_seed: int = 0
    schedule: str = "cosine"
    base_lr: float = 0.05
    min_lr: float = 0.0
    patience: int = 0
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.bn_momentum, bool) or not isinstance(self.bn_momentum, (int, float)):
            raise ParameterError(f"bn_momentum 必须是数，收到 {self.bn_momentum!r}。")
        momentum = float(self.bn_momentum)
        if not math.isfinite(momentum) or not 0.0 <= momentum <= 1.0:
            raise ParameterError(
                f"bn_momentum 必须落在 [0, 1]，收到 {self.bn_momentum!r}："
                "它决定 running 统计量跟得有多快（1 = 只看最近一批）。"
            )
        object.__setattr__(self, "bn_momentum", momentum)
        object.__setattr__(self, "dropout_rate", _checked_rate(self.dropout_rate))
        if isinstance(self.dropout_seed, bool) or not isinstance(self.dropout_seed, int):
            raise ParameterError(f"dropout_seed 必须是整数，收到 {self.dropout_seed!r}。")
        if isinstance(self.patience, bool) or not isinstance(self.patience, int) or self.patience < 0:
            raise ParameterError(f"patience 必须是非负整数，收到 {self.patience!r}。")
        if not isinstance(self.base_lr, (int, float)) or not math.isfinite(float(self.base_lr)):
            raise ParameterError(f"base_lr 必须是有限数，收到 {self.base_lr!r}。")
        if not isinstance(self.min_lr, (int, float)) or not math.isfinite(float(self.min_lr)):
            raise ParameterError(f"min_lr 必须是有限数，收到 {self.min_lr!r}。")

    def enabled(self) -> tuple[str, ...]:
        """这一份配置真正打开的旋钮（顺序与 :data:`types.TECHNIQUES` 一致）."""
        active: list[str] = []
        for technique in TECHNIQUES:
            if technique == "batchnorm" and self.norm:
                active.append(technique)
            elif technique == "dropout" and self.dropout_rate > 0.0:
                active.append(technique)
            elif technique == "lr_decay" and self.schedule != "constant":
                active.append(technique)
            elif technique == "early_stop" and self.patience > 0:
                active.append(technique)
        return tuple(active)

    def schedule_kwargs(self, total_steps: int) -> dict[str, Any]:
        """按调度名给出 day074 需要的那几个参数（**多传一个就会 TypeError，因此要精确**）.

        四种调度的参数本来就不一样（``base_lr`` 是它们**共有**的那一个，
        常数调度也照样要它），因此这份映射是显式的——
        而不是"把所有参数都传进去、让它自己忽略"。
        """
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 1:
            raise ParameterError(f"total_steps 必须是 >= 1 的整数，收到 {total_steps!r}。")
        shared: dict[str, Any] = {"base_lr": self.base_lr}
        if self.schedule == "constant":
            return shared
        if self.schedule == "cosine":
            return {**shared, "total_steps": total_steps, "min_lr": self.min_lr}
        if self.schedule == "warmup_cosine":
            return {
                **shared,
                "warmup_steps": max(1, total_steps // 10),
                "total_steps": total_steps,
                "min_lr": self.min_lr,
            }
        if self.schedule == "step_decay":
            return {**shared, "drop_every": max(1, total_steps // 4), "gamma": 0.5}
        raise ParameterError(
            f"未知的调度 {self.schedule!r}：可用取值 constant / step_decay / cosine / warmup_cosine。"
        )

    def line(self) -> str:
        """一行说明：``norm=on | dropout=0.2 | cosine(lr 0.05→0) | patience=0``."""
        norm = "on" if self.norm else "off"
        patience = "off" if self.patience == 0 else str(self.patience)
        return (
            f"norm={norm} | dropout={self.dropout_rate:g} | "
            f"{self.schedule}(lr {self.base_lr:g}→{self.min_lr:g}) | patience={patience}"
        )


__all__ = [
    "EarlyStopping",
    "EarlyStopReport",
    "RegularizationConfig",
    "dropout_vector",
    "dropout_vector_backward",
    "dropout_vector_backward_matrix",
    "early_stopping_watch",
    "learning_rate_at",
]
