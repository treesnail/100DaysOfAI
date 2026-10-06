"""``rules``：三个更新规则的**纯函数**，外加两种衰减与裁剪的接线（day092 / M8-D3）.

这一模块刻意写成**纯函数**（状态进、状态出），原因与 day074 的取向一致：
参数是一串不可变的数，因此"这一步的结果"与"上一步的结果"是两个不同的对象，
可以直接进历史记录、被对比、被断言。

```text
共享规则（与 day074 逐位一致，**本课不重写**）
  sgd / momentum / adam  ——  由 optimizer.py 直接复用 math_foundations.optim 的类

新增规则（day074 刻意留白）
  nesterov_step     v ← βv + g；方向 = g + β·v（**迈步后**再取一次）
  rmsprop_step      s ← ρs + (1−ρ)g²；θ ← θ − lr·g/(√s + ε)
  两种衰减           l2_gradient（进梯度） / apply_decoupled_decay（缩参数）

接线
  clip_gradients    整体范数裁剪，**复用** day074 的 clip_by_global_norm（只加薄薄一层校验）
```

## Nesterov 与普通动量差在哪一行

```text
普通动量    方向 = v            （当前速度）
Nesterov    方向 = g + β·v      （当前梯度 + 一点未来的速度）
```

PyTorch 的 SGD 在 ``nesterov=True`` 时用的是后者，而 ``buf`` 的更新本身与普通动量
**逐字相同**——差别只在"用哪一项当方向"。因此两件事必须分开写：
"速度怎么更新"只有一处，"用哪一项当方向"由开关决定。

## 为什么衰减要写成两个函数而不是一个参数

```text
l2_gradient            把 λθ 加进**梯度**     → 它会被 Adam 的分母 √v̂ 除一下
apply_decoupled_decay  把参数乘 **(1−lr·λ)**  → 与分母无关
```

这两者在数学上不是"同一个东西的两种写法"：耦合版的实际衰减强度随梯度尺度变化，
解耦版才是"人人等量衰减"。把它们写成两个函数，是让这件事在代码里无处可藏。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from smart_research_agent.math_foundations.optim import (
    clip_by_global_norm as math_clip_by_global_norm,
)
from smart_research_agent.math_foundations.optim import global_norm as math_global_norm
from smart_research_agent.math_foundations.types import is_finite, validate_vector
from smart_research_agent.optimizers.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.optimizers.types import (
    DECAY_DECOUPLED,
    DECAY_L2,
    DECAY_MODES,
    DECAY_NONE,
)

#: 优化器参数的缺省值（与 day074 的 AdamOptimizer 同源）.
DEFAULT_MOMENTUM = 0.9
DEFAULT_ALPHA = 0.99
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.999
DEFAULT_EPSILON = 1e-8


def as_vector(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    """把一串数收敛成 ``tuple[float, ...]``，非有限数当场拒绝.

    本包不接受"读取时再顺手转一下"：一个 ``nan`` 混进参数后，
    所有比较（相等或上界）都会**静默为假**，而报告里只会显示"这条性质不通过"——
    根因却在更早的一次除法上。因此在入口就把它挡住。
    """
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ShapeError(f"{name} 必须是一串数（list / tuple），收到 {type(values).__name__}。")
    checked: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ShapeError(f"{name}[{index}] 必须是实数，收到 {type(value).__name__}（{value!r}）。")
        number = float(value)
        if not is_finite(number):
            raise NumericError(f"{name}[{index}] 是非有限数（{value!r}）：在 nan / inf 上更新参数没有意义。")
        checked.append(number)
    return tuple(checked)


def checked_positive(value: float, *, name: str) -> float:
    """校验一个必须为正的有限数（学习率、ε、裁剪阈值都走这里）."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterError(f"{name} 必须是实数，收到 {type(value).__name__}（{value!r}）。")
    number = float(value)
    if not is_finite(number) or number <= 0:
        raise ParameterError(
            f"{name} 必须是正的有限数，收到 {value!r}："
            "负的学习率会把梯度下降变成梯度上升，而它不会报错——"
            "只会让损失一路变大，看起来像'数据有问题'。"
        )
    return number


def checked_non_negative(value: float, *, name: str) -> float:
    """校验一个**非负**的有限数（学习率曲线在末段可以取到 0）.

    为什么要与 :func:`checked_positive` 分开：余弦退火的最后一步恰好落在 ``min_lr``，
    而 ``min_lr = 0`` 是一个完全合法的终点（"这一步不再更新"）。若用"必须为正"
    去卡它，常见的 ``warmup_cosine`` 会在**最后一步**报错——而前面每一步都对。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterError(f"{name} 必须是实数，收到 {type(value).__name__}（{value!r}）。")
    number = float(value)
    if not is_finite(number) or number < 0:
        raise ParameterError(
            f"{name} 必须是非负的有限数，收到 {value!r}："
            "负的学习率会把梯度下降变成梯度上升；0 是允许的（末段不再更新）。"
        )
    return number


def checked_fraction(value: float, *, name: str) -> float:
    """校验一个落在 ``[0, 1)`` 的系数（动量 β、衰减率 ρ）."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterError(f"{name} 必须是实数，收到 {type(value).__name__}（{value!r}）。")
    number = float(value)
    if not (0.0 <= number < 1.0):
        raise ParameterError(
            f"{name} 必须落在 [0, 1)，收到 {value!r}："
            "等于 1 时对应的滑动平均不衰减（历史梯度全部留在里面），"
            "而它不会报错，只会让参数一路冲出去。"
        )
    return number


def checked_pair(params: Sequence[float], grads: Sequence[float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """校验"参数与梯度同形"（它们必须一一对应）."""
    checked_params = as_vector(params, name="params")
    checked_grads = as_vector(grads, name="grads")
    if len(checked_params) != len(checked_grads):
        raise ShapeError(
            f"参数 {len(checked_params)} 维而梯度 {len(checked_grads)} 维："
            "梯度是'损失对每一个参数的偏导数'，两者必须一一对应——"
            "长度不同说明在某一处压平/展开的环节丢了东西。"
        )
    return checked_params, checked_grads


def checked_decay_mode(mode: str) -> str:
    """校验衰减模式的取值."""
    if mode not in DECAY_MODES:
        raise ParameterError(f"不认识的衰减模式 {mode!r}：可用取值 {list(DECAY_MODES)}。")
    return mode


# --------------------------------------------------------------------------- #
# 两种衰减
# --------------------------------------------------------------------------- #


def l2_gradient(grads: Sequence[float], params: Sequence[float], weight_decay: float) -> tuple[float, ...]:
    """耦合式 L2：把 ``λθ`` 加进梯度（``g ← g + λ·θ``）.

    这就是 day074 ``SGD(lr, weight_decay=λ)`` 在幕后做的事。它的问题在于：
    当 ``g`` 后面还要被 Adam 的 ``√v̂`` 除一下时，**衰减项也一起被除了**——
    于是梯度大的参数衰减得更弱。要"人人等量衰减"，就得用解耦版。
    """
    checked_grads = as_vector(grads, name="grads")
    checked_params = as_vector(params, name="params")
    if len(checked_grads) != len(checked_params):
        raise ShapeError(f"梯度 {len(checked_grads)} 维而参数 {len(checked_params)} 维：两者必须一一对应。")
    if weight_decay == 0:
        return checked_grads
    if not is_finite(weight_decay) or weight_decay < 0:
        raise ParameterError(f"weight_decay 必须是非负的有限数，收到 {weight_decay!r}。")
    return tuple(value + weight_decay * param for value, param in zip(checked_grads, checked_params))


def apply_decoupled_decay(params: Sequence[float], learning_rate: float, weight_decay: float) -> tuple[float, ...]:
    """解耦权重衰减：``θ ← θ·(1 − lr·λ)``（AdamW 的那一步）.

    它与 :func:`l2_gradient` 的关键差别是**位置**：这里乘在参数上、不进梯度，
    因此不会被任何自适应分母除。``lr·λ`` 通常远小于 1，所以这一步是"轻轻缩一下"。
    """
    checked_params = as_vector(params, name="params")
    if weight_decay == 0:
        return checked_params
    rate = checked_positive(learning_rate, name="learning_rate")
    if not is_finite(weight_decay) or weight_decay < 0:
        raise ParameterError(f"weight_decay 必须是非负的有限数，收到 {weight_decay!r}。")
    factor = 1.0 - rate * weight_decay
    return tuple(value * factor for value in checked_params)


# --------------------------------------------------------------------------- #
# 两个新增更新规则
# --------------------------------------------------------------------------- #


def nesterov_step(
    params: Sequence[float],
    grads: Sequence[float],
    velocity: Sequence[float],
    *,
    learning_rate: float,
    momentum: float = DEFAULT_MOMENTUM,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """走一步 Nesterov 动量，返回 ``(新参数, 新速度)``.

    ```text
    v ← β·v + g            速度的更新与普通动量**逐字相同**
    方向 = g + β·v         这是唯一的差别：用"再看一眼未来"的那一项当方向
    θ ← θ − lr·方向
    ```

    速度的长度必须与参数一致（否则抛 :class:`StateError` 的兄弟 ``ShapeError``）——
    它最可能来自"换了任务却忘了 ``reset()``"。
    """
    checked_params, checked_grads = checked_pair(params, grads)
    checked_velocity = as_vector(velocity, name="velocity")
    if len(checked_velocity) != len(checked_params):
        raise ShapeError(
            f"速度 {len(checked_velocity)} 维而参数 {len(checked_params)} 维："
            "动量是一个与参数同形的向量，长度不符说明它是上一段任务留下的状态。"
        )
    rate = checked_positive(learning_rate, name="learning_rate")
    beta = checked_fraction(momentum, name="momentum")
    new_velocity = tuple(beta * previous + slope for previous, slope in zip(checked_velocity, checked_grads))
    direction = tuple(slope + beta * speed for slope, speed in zip(checked_grads, new_velocity))
    updated = tuple(value - rate * step for value, step in zip(checked_params, direction))
    return updated, new_velocity


def rmsprop_step(
    params: Sequence[float],
    grads: Sequence[float],
    square_avg: Sequence[float],
    *,
    learning_rate: float,
    alpha: float = DEFAULT_ALPHA,
    epsilon: float = DEFAULT_EPSILON,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """走一步 RMSProp，返回 ``(新参数, 新二阶动量)``.

    ```text
    s ← ρ·s + (1−ρ)·g²          只保留二阶动量（Adam 的一半）
    θ ← θ − lr·g/(√s + ε)       分母把每个参数的尺度归一
    ```

    ``s`` 的稳态是 ``E[g²]``，因此 ``g/√s`` 的尺度与 ``g`` 的大小无关——
    这正是"梯度恒为 100 与恒为 0.01 的参数走得一样快"的来源。
    """
    checked_params, checked_grads = checked_pair(params, grads)
    checked_square = as_vector(square_avg, name="square_avg")
    if len(checked_square) != len(checked_params):
        raise ShapeError(
            f"二阶动量 {len(checked_square)} 维而参数 {len(checked_params)} 维：两者必须一一对应。"
        )
    rate = checked_positive(learning_rate, name="learning_rate")
    rho = checked_fraction(alpha, name="alpha")
    eps = checked_positive(epsilon, name="epsilon")
    new_square = tuple(
        rho * previous + (1.0 - rho) * slope * slope
        for previous, slope in zip(checked_square, checked_grads)
    )
    updated = tuple(
        value - rate * slope / (math.sqrt(square) + eps)
        for value, slope, square in zip(checked_params, checked_grads, new_square)
    )
    return updated, new_square


# --------------------------------------------------------------------------- #
# 裁剪（复用 day074）
# --------------------------------------------------------------------------- #


def clip_gradients(grads: Sequence[float], max_norm: float) -> tuple[tuple[float, ...], float]:
    """整体范数裁剪，返回 ``(裁剪后的梯度, 缩放系数)``——复用 day074 的实现.

    本函数只做两件事：① 用本包的失败族校验入参；② 把 day074 的
    :func:`math_foundations.optim.clip_by_global_norm` 的结论原样透传。
    **裁剪的数学只有一份**，写在 day074；本课把它接进训练期更新，
    并在 :mod:`optimizers.verify` 里断言"接线没有改动它"。
    """
    checked = as_vector(grads, name="grads")
    limit = checked_positive(max_norm, name="max_norm")
    clipped, factor = math_clip_by_global_norm(checked, limit)
    return tuple(clipped), factor


def global_norm(grads: Sequence[float]) -> float:
    """梯度的整体范数 ``‖g‖``（计算只有一份：复用 day074 的 ``global_norm``）."""
    checked = as_vector(grads, name="grads")
    return math_global_norm(checked)


def cosine_between(left: Sequence[float], right: Sequence[float]) -> float:
    """:func:`裁剪前后方向的余弦``——用来断言"只改长度、不改方向".

    两个非零向量夹角的余弦 = 点积 /（模长之积）。裁剪真的保住方向时它恰好是 1.0
    （浮点下是 1 减去一个极小量）；写成逐分量裁剪（``clip_by_value``）时，
    它会在"分量尺度悬殊"的输入上明显小于 1。
    """
    first = as_vector(left, name="left")
    second = as_vector(right, name="right")
    if len(first) != len(second):
        raise ShapeError(f"两个向量长度不一致（{len(first)} vs {len(second)}），无法比较方向。")
    numerator = math.fsum(a * b for a, b in zip(first, second))
    norm_first = math.sqrt(math.fsum(a * a for a in first))
    norm_second = math.sqrt(math.fsum(b * b for b in second))
    if norm_first == 0.0 or norm_second == 0.0:
        raise NumericError("零向量没有方向，无法计算夹角。")
    return numerator / (norm_first * norm_second)


#: 本课新增的两个更新规则（供优化器工厂分派；共享的三个由 day074 提供）.
NEW_RULES = ("nesterov", "rmsprop")

#: 三种衰减的常量集中在这里，避免调用方散落字符串字面量.
DECAY_CHOICES = (DECAY_NONE, DECAY_L2, DECAY_DECOUPLED)

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_BETA1",
    "DEFAULT_BETA2",
    "DEFAULT_EPSILON",
    "DEFAULT_MOMENTUM",
    "DECAY_CHOICES",
    "NEW_RULES",
    "apply_decoupled_decay",
    "as_vector",
    "checked_decay_mode",
    "checked_fraction",
    "checked_non_negative",
    "checked_pair",
    "checked_positive",
    "clip_gradients",
    "cosine_between",
    "global_norm",
    "l2_gradient",
    "nesterov_step",
    "rmsprop_step",
]
