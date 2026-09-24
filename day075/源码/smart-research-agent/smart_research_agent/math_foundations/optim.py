"""优化：从"往哪走"到"走多远"（day074 / Math-D2）.

``calculus`` 回答了"往哪走"（梯度），``autograd`` 回答了"怎么自动算出来"。
这一模块回答最后一个问题：**一次走多远**。

```text
θ_{t+1} = θ_t − lr · g_t
             └┬─┘   └┬┘
           学习率   梯度（往哪走）
```

它看着只有一行，但这一行里有三个各自独立的问题，本模块分别给出实现：

```text
① 方向不只是梯度        动量把历史梯度累成"速度"，冲过平坦区、压住抖动
② 步长不该人人相同      Adam 用二阶动量给每个参数各自的等效步长
③ 步长不该一成不变      学习率调度：前期热身、后期退火（见 SCHEDULES）
```

## 为什么这一课要把"调度"当成主角之一

一个常数学习率在真实训练里几乎总是错的，而对的原因每次都不同：

```text
前期 lr 太大       随机初始化时梯度方向信息量很低，一大步就把参数推到一个坏区域
                    → 热身（warmup）：从 0 线性拉到 base_lr
后期 lr 太大       已经接近谷底，大步会来回横跳（loss 曲线开始震荡）
                    → 退火（cosine/step decay）：把步长收小
```

这两种需求方向相反，因此"一个调度同时满足两者"的唯一形式是**先升后降**——
``warmup_cosine`` 就是它的最小实现。而 **常数调度不是被淘汰的旧方案，而是基线**：
没有它，"调度有没有用"这个问题无法回答（两条曲线必须能对比）。

## Adam 的偏差修正：为什么第一步不是"几乎没有走"

Adam 维护两个滑动平均，它们都从 0 出发：

```text
m₀ = 0     →  m₁ = (1−β₁)g₁ = 0.1g₁        只有真值的 1/10
v₀ = 0     →  v₁ = (1−β₂)g₁² = 0.001g₁²    只有真值的 1/1000
```

不做修正时第一步的更新量是 ``lr·0.1g / (√0.001·|g|) ≈ lr·3.16``——
**比设定值大 3 倍**，而它看起来只是"前期有点抖"。
除以 ``1−β₁ᵗ`` 与 ``1−β₂ᵗ`` 之后，第一步的更新量恰好是 ``lr·sign(g)``：
这是一个可以被逐位断言的结论（见 ``tests/test_math_optim.py`` 里的那一条）。

## 这一层的边界

```text
只做向量参数     参数是一串数（矩阵请先用 flatten_matrices 压平）
不做参数分组      所有参数共享一个学习率（分层学习率是工程问题，不是数学问题）
不做梯度累积/裁剪的"策略层"   只提供 clip_by_global_norm 这个原语
```

一句话：这一层实现的是**三个更新公式本身**，而不是一个训练框架。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.math_foundations.calculus import DEFAULT_STEP, gradient
from smart_research_agent.math_foundations.errors import (
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.math_foundations.types import (
    CENTRAL,
    OPTIMIZER_ADAM,
    OPTIMIZER_DESCRIPTIONS,
    OPTIMIZER_MOMENTUM,
    OPTIMIZER_SGD,
    OPTIMIZERS,
    SCHEDULE_CONSTANT,
    SCHEDULE_COSINE,
    SCHEDULE_STEP_DECAY,
    SCHEDULE_WARMUP_COSINE,
    SCHEDULES,
    Matrix,
    Vector,
    is_finite,
    validate_matrix,
    validate_vector,
)

#: 一条学习率曲线：给"第几步"（从 1 开始）返回这一步的学习率.
Schedule = Callable[[int], float]

#: 目标函数：一串参数进、一个损失出.
Objective = Callable[[Vector], float]

#: Adam 的缺省数值稳定项（原论文 1e-8）.
DEFAULT_EPSILON = 1e-8


def _checked_positive(value: float, *, name: str) -> float:
    """校验一个必须为正的有限数（学习率、动量系数、限幅阈值都走这里）."""
    if not is_finite(value) or value <= 0:
        raise ParameterError(
            f"{name}必须是正的有限数，收到 {value!r}："
            "负的学习率会把梯度下降变成梯度上升，而它不会报错——"
            "只会让 loss 一路变大，看起来像'数据有问题'。"
        )
    return float(value)


def _checked_step_number(step: int, *, name: str = "step") -> int:
    """校验步号：必须是 >= 1 的整数（**调度从第 1 步开始，与日常说法一致**）."""
    if isinstance(step, bool) or not isinstance(step, int):
        raise ParameterError(f"{name} 必须是整数，收到 {type(step).__name__}（{step!r}）。")
    if step < 1:
        raise ParameterError(
            f"{name} 必须 >= 1，收到 {step}："
            "本包一律用 1-based 计步（第 1 步就是第一次更新）——"
            "0-based 与 1-based 混用时，热身少一步、退火早一步，"
            "而两条曲线都'看起来正常'。"
        )
    return step


# --------------------------------------------------------------------------- #
# 学习率调度
# --------------------------------------------------------------------------- #


def constant_schedule(step: int, *, base_lr: float) -> float:
    """常数学习率（**基线**：没有它就说不清"调度有没有用"）."""
    _checked_step_number(step)
    return _checked_positive(base_lr, name="base_lr")


def step_decay_schedule(
    step: int,
    *,
    base_lr: float,
    drop_every: int,
    gamma: float = 0.1,
) -> float:
    """阶梯衰减 ``lr = base_lr · gamma^{⌊(t−1)/drop_every⌋}``.

    第 1~drop_every 步是 ``base_lr``，第 drop_every+1 步起乘一次 ``gamma``。
    ``⌊(t−1)/drop_every⌋`` 里的 ``−1`` 就是"从第 1 步开始数"的体现：
    写成 ``⌊t/drop_every⌋`` 会让第 drop_every 步就提前掉档，
    而"提前一步"在任何曲线上都看不出来。
    """
    current = _checked_step_number(step)
    base = _checked_positive(base_lr, name="base_lr")
    if isinstance(drop_every, bool) or not isinstance(drop_every, int) or drop_every < 1:
        raise ParameterError(f"drop_every 必须是 >= 1 的整数，收到 {drop_every!r}。")
    if not (0.0 < gamma < 1.0):
        raise ParameterError(
            f"gamma 必须落在 (0, 1)，收到 {gamma!r}："
            "gamma >= 1 是'越走越大'，gamma <= 0 会让学习率变号（等于梯度上升）。"
        )
    return base * gamma ** ((current - 1) // drop_every)


def cosine_schedule(
    step: int,
    *,
    base_lr: float,
    total_steps: int,
    min_lr: float = 0.0,
) -> float:
    """余弦退火 ``lr = min_lr + (base_lr−min_lr)·(1+cos(π·(t−1)/(T−1)))/2``.

    两端都取到端点：第 1 步是 ``base_lr``、第 ``T`` 步是 ``min_lr``。
    分母用 ``T − 1`` 而不是 ``T`` 就是为了这个——用 ``T`` 时第 T 步
    才走到余弦的 ``(T−1)/T`` 处，永远差一点点到不了 ``min_lr``，
    而"最后一步的学习率比预期大 0.1%"不会被任何人注意到。
    """
    current = _checked_step_number(step)
    base = _checked_positive(base_lr, name="base_lr")
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 2:
        raise ParameterError(
            f"total_steps 必须是 >= 2 的整数，收到 {total_steps!r}："
            "只有一步的余弦退火没有'退'的过程（分母 T−1 也会变成 0）。"
        )
    if not is_finite(min_lr) or min_lr < 0:
        raise ParameterError(f"min_lr 必须是非负的有限数，收到 {min_lr!r}。")
    if min_lr > base:
        raise ParameterError(
            f"min_lr（{min_lr}）不能大于 base_lr（{base}）："
            "退火的方向是'越来越小'，反过来是升温。"
        )
    # 步号超出总步数时不外推：调度是"计划"，跑到计划之外说明计划该改了
    clipped = min(current, total_steps)
    progress = (clipped - 1) / (total_steps - 1)
    return min_lr + (base - min_lr) * (1.0 + math.cos(math.pi * progress)) / 2.0


def warmup_cosine_schedule(
    step: int,
    *,
    base_lr: float,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 0.0,
) -> float:
    """线性热身 + 余弦退火（**先升后降**，同时解决前期与后期两个问题）.

    ```text
    t <= warmup           lr(t) = base_lr · t / warmup      （从 base_lr/warmup 爬到 base_lr）
    t >  warmup           从 base_lr 余弦降到 min_lr
    ```

    热身的斜率是 ``base_lr / warmup``：第 1 步就拿到 ``base_lr/warmup``
    （**不是 0**）——"第 1 步 lr = 0"会让第一次更新完全不发生，
    而它在日志里表现为"第 1 步的 loss 与初始 loss 一模一样"，
    看起来像"梯度算错了"。
    """
    current = _checked_step_number(step)
    base = _checked_positive(base_lr, name="base_lr")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 1:
        raise ParameterError(f"warmup_steps 必须是 >= 1 的整数，收到 {warmup_steps!r}。")
    if warmup_steps >= total_steps:
        raise ParameterError(
            f"warmup_steps（{warmup_steps}）必须小于 total_steps（{total_steps}）："
            "热身占满全程时没有退火段，那它就不是'热身+退火'。"
        )
    if current <= warmup_steps:
        return base * current / warmup_steps
    return cosine_schedule(
        current - warmup_steps + 1,
        base_lr=base,
        total_steps=total_steps - warmup_steps + 1,
        min_lr=min_lr,
    )


#: 调度工厂表（键与 :data:`types.SCHEDULES` 逐键对齐；值为"造一条一条曲线"的工厂）.
SCHEDULE_FACTORIES: dict[str, Callable[..., Schedule]] = {
    SCHEDULE_CONSTANT: lambda **params: lambda step: constant_schedule(step, **params),
    SCHEDULE_STEP_DECAY: lambda **params: lambda step: step_decay_schedule(step, **params),
    SCHEDULE_COSINE: lambda **params: lambda step: cosine_schedule(step, **params),
    SCHEDULE_WARMUP_COSINE: lambda **params: lambda step: warmup_cosine_schedule(step, **params),
}

if set(SCHEDULE_FACTORIES) != set(SCHEDULES):  # pragma: no cover - 导入期不变式
    raise ParameterError(
        "调度工厂表与 SCHEDULES 不一致：少一项的那个调度会静默地造不出来，"
        "而'造不出来'与'没人用它'在报告里长得一样。"
    )


def make_schedule(name: str, **params: Any) -> Schedule:
    """按名字造一条学习率曲线（参数由各调度自己的签名决定）.

    为什么用工厂而不是"统一签名的调度函数"：四种调度的参数本来就不一样
    （只有常数调度不需要任何额外参数）。硬凑一个
    ``schedule(step, base_lr, drop_every, gamma, warmup_steps, total_steps, min_lr)``
    会让"这个调度到底用到了哪几个参数"变得不可读。
    """
    if name not in SCHEDULE_FACTORIES:
        raise ParameterError(
            f"不认识的学习率调度 {name!r}：可用取值 {list(SCHEDULES)}"
            f"（{SCHEDULE_CONSTANT} 是基线）。"
        )
    return SCHEDULE_FACTORIES[name](**params)


# --------------------------------------------------------------------------- #
# 梯度裁剪
# --------------------------------------------------------------------------- #


def global_norm(grads: Vector) -> float:
    """梯度的整体范数 ``‖g‖``（按分量平方和开根，**所有参数共享一个尺度**）.

    用整体范数而不是逐参数裁剪，是因为"这一次更新有多大"是**参数向量整体**的性质：
    逐参数把每个分量都限制在 ±1 之内，会在一个方向上推出一个很长的步，
    而整体范数约束的是那一步的实际长度。
    """
    return math.sqrt(math.fsum(value * value for value in validate_vector(grads, name="grads")))


def clip_by_global_norm(grads: Vector, max_norm: float) -> tuple[Vector, float]:
    """按整体范数裁剪，返回 ``(裁剪后的梯度, 缩放系数)``.

    ```text
    ‖g‖ <= max_norm    原样返回（缩放系数 1.0）
    ‖g‖ >  max_norm    整体乘 max_norm/‖g‖（**方向不变，只改长度**）
    ```

    返回缩放系数而不是只返回梯度，是因为"这一次裁剪有没有生效"必须可读：
    只给梯度的话，一串"看起来正常"的数既可能是没裁剪，也可能已经被缩了 50 倍。
    """
    checked = validate_vector(grads, name="grads")
    limit = _checked_positive(max_norm, name="max_norm")
    length = math.sqrt(math.fsum(value * value for value in checked))
    if length <= limit:
        return checked, 1.0
    scale_factor = limit / length
    return tuple(value * scale_factor for value in checked), scale_factor


def clip_by_value(grads: Vector, limit: float) -> Vector:
    """逐分量裁剪到 ``[-limit, limit]``（**会改变方向**，因此要写清它做了什么）.

    它与整体范数裁剪不是一回事：一个梯度 ``(100, 0)`` 逐分量裁到 ``(1, 0)`` 之后
    方向不变；但 ``(100, 100)`` 裁到 ``(1, 1)`` 之后方向也恰好不变（同比缩放）。
    真正会出现方向差异的是 ``(100, 1)`` → ``(1, 1)``：方向从"几乎水平"变成 45°。
    因此这一条只适合"个别分量爆炸"的场景。
    """
    checked = validate_vector(grads, name="grads")
    limit_value = _checked_positive(limit, name="limit")
    return tuple(max(-limit_value, min(limit_value, value)) for value in checked)


# --------------------------------------------------------------------------- #
# 三个优化器
# --------------------------------------------------------------------------- #


class Optimizer:
    """优化器基类：持有学习率与步数，``step(params, grads) -> 新参数``.

    函数式返回新向量（而不是原地改传入的列表）是与 ``math_foundations``
    其余部分一致的取向：参数是 ``tuple``，**不可变**，
    因此"这一步的参数"与"上一步的参数"是两个不同的对象，可以直接进历史记录。
    """

    name: str = "optimizer"

    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = _checked_positive(learning_rate, name="learning_rate")
        self.step_count = 0

    def step(self, params: Vector, grads: Vector) -> Vector:  # pragma: no cover - 抽象
        """按各自的更新公式走一步（子类实现）."""
        raise NotImplementedError

    def reset(self) -> None:
        """清空优化器状态（**参数不动，动量/二阶动量归零**）.

        为什么需要它：换任务或换学习率曲线时，"上一段的动量"没有任何意义，
        而它会被悄悄带进新任务的前几步——表现为"换了任务之后前几步莫名地大"。
        """
        self.step_count = 0

    def describe(self) -> str:
        """一行说明（进报告：报告里要能读出"这个优化器是什么"）."""
        return f"{self.name}（{OPTIMIZER_DESCRIPTIONS[self.name]}），lr={self.learning_rate:g}"

    def state(self) -> dict[str, Any]:
        """可 json.dumps 的状态（只含"跨步保留的东西"）."""
        return {"name": self.name, "learning_rate": self.learning_rate, "step_count": self.step_count}


def _checked_pair(params: Vector, grads: Vector) -> tuple[Vector, Vector]:
    """校验"参数与梯度同形"（它们必须一一对应）."""
    checked_params = validate_vector(params, name="params")
    checked_grads = validate_vector(grads, name="grads")
    if len(checked_params) != len(checked_grads):
        raise ShapeError(
            f"参数 {len(checked_params)} 维而梯度 {len(checked_grads)} 维："
            "梯度是'损失对每一个参数的偏导数'，两者必须一一对应——"
            "长度不同说明在某一处压平/展开的环节丢了东西。"
        )
    return checked_params, checked_grads


class SGDOptimizer(Optimizer):
    """随机梯度下降 ``θ ← θ − lr·g``（**无状态**，也是最诚实的基线）.

    它的"缺点"（在峡谷形曲面上来回横跳）恰好是另两个优化器存在的理由，
    因此把它实现清楚是理解后两者的前提。
    """

    name = OPTIMIZER_SGD

    def step(self, params: Vector, grads: Vector) -> Vector:
        """按 ``θ − lr·g`` 走一步."""
        checked_params, checked_grads = _checked_pair(params, grads)
        self.step_count += 1
        return tuple(
            value - self.learning_rate * slope
            for value, slope in zip(checked_params, checked_grads)
        )


class MomentumOptimizer(Optimizer):
    """动量法 ``v ← β·v + g;  θ ← θ − lr·v``.

    注意这里**没有** ``(1 − β)`` 的归一化：本实现用的是"累加型"写法
    （Polyak 原始形式），因此 ``v`` 的稳态幅度是 ``g/(1−β)``——
    动量系数 0.9 时速度是梯度的 10 倍。
    两种写法（累加型与 EMA 型）在数学上可以通过"把 lr 除以 (1−β)"互相换算，
    但**不能混着用**：混用的后果是等效学习率差一个 ``1/(1−β)`` 的因子，
    而它表现为"换了优化器之后需要重新调 lr"，看起来只是"优化器风格不同"。
    """

    name = OPTIMIZER_MOMENTUM

    def __init__(self, learning_rate: float, momentum: float = 0.9) -> None:
        super().__init__(learning_rate)
        if not (0.0 <= momentum < 1.0):
            raise ParameterError(
                f"momentum 必须落在 [0, 1)，收到 {momentum!r}："
                "等于 1 时速度永不衰减（历史梯度全部留在里面），"
                "而它不会报错，只会让参数一路冲出去。"
            )
        self.momentum = float(momentum)
        self.velocity: Vector = ()

    def step(self, params: Vector, grads: Vector) -> Vector:
        """按动量的更新公式走一步."""
        checked_params, checked_grads = _checked_pair(params, grads)
        if not self.velocity:
            self.velocity = tuple(0.0 for _ in checked_grads)
        self.step_count += 1
        self.velocity = tuple(
            self.momentum * previous + slope
            for previous, slope in zip(self.velocity, checked_grads)
        )
        return tuple(
            value - self.learning_rate * speed
            for value, speed in zip(checked_params, self.velocity)
        )

    def reset(self) -> None:
        """清空步数与速度（**速度不清零就等于带着上一段的惯性进新任务**）."""
        super().reset()
        self.velocity = ()

    def state(self) -> dict[str, Any]:
        """可 json.dumps 的状态."""
        return {**super().state(), "momentum": self.momentum, "velocity": list(self.velocity)}


class AdamOptimizer(Optimizer):
    """Adam ``θ ← θ − lr·m̂/(√v̂ + ε)``（一阶 + 二阶动量，各带偏差修正）.

    ```text
    m ← β₁m + (1−β₁)g          一阶动量：梯度的滑动平均（方向）
    v ← β₂v + (1−β₂)g²         二阶动量：梯度平方的滑动平均（尺度）
    m̂ = m/(1−β₁ᵗ)             偏差修正：否则第一步只有真值的 1−β₁
    v̂ = v/(1−β₂ᵗ)
    θ ← θ − lr·m̂/(√v̂ + ε)    每个参数的等效步长 ≈ lr（与梯度大小无关）
    ```

    "每个参数的等效步长 ≈ lr"是 Adam 最反直觉也最有用的性质：
    梯度恒为 100 的参数与梯度恒为 0.01 的参数**走得一样快**。
    这既解决了不同尺度参数的统一学习率问题，也带来一个副作用——
    在凸问题上它不保证收敛（后期会在最优点附近来回抖），因此才有了
    "后期把 lr 退火"这件事（见 :func:`cosine_schedule`）。
    """

    name = OPTIMIZER_ADAM

    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = DEFAULT_EPSILON,
    ) -> None:
        super().__init__(learning_rate)
        for label, value in (("beta1", beta1), ("beta2", beta2)):
            if not (0.0 <= value < 1.0):
                raise ParameterError(
                    f"{label} 必须落在 [0, 1)，收到 {value!r}："
                    "等于 1 时对应的滑动平均不衰减（一阶动量永不忘记历史梯度）。"
                )
        if not is_finite(epsilon) or epsilon <= 0:
            raise ParameterError(
                f"epsilon 必须为正的有限数，收到 {epsilon!r}："
                "它是分母的保护项，为 0 时梯度为 0 的参数会得到 0/0。"
            )
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.first_moment: Vector = ()
        self.second_moment: Vector = ()

    def step(self, params: Vector, grads: Vector) -> Vector:
        """按 Adam 的更新公式走一步（含偏差修正）."""
        checked_params, checked_grads = _checked_pair(params, grads)
        if not self.first_moment:
            self.first_moment = tuple(0.0 for _ in checked_grads)
            self.second_moment = tuple(0.0 for _ in checked_grads)
        self.step_count += 1
        self.first_moment = tuple(
            self.beta1 * previous + (1.0 - self.beta1) * slope
            for previous, slope in zip(self.first_moment, checked_grads)
        )
        self.second_moment = tuple(
            self.beta2 * previous + (1.0 - self.beta2) * slope * slope
            for previous, slope in zip(self.second_moment, checked_grads)
        )
        correction1 = 1.0 - self.beta1**self.step_count
        correction2 = 1.0 - self.beta2**self.step_count
        updated: list[float] = []
        for value, mean, square in zip(checked_params, self.first_moment, self.second_moment):
            adjusted_mean = mean / correction1
            adjusted_square = square / correction2
            updated.append(
                value - self.learning_rate * adjusted_mean / (math.sqrt(adjusted_square) + self.epsilon)
            )
        return tuple(updated)

    def reset(self) -> None:
        """清空步数与两个动量（**偏差修正依赖 step_count，必须一起清**）."""
        super().reset()
        self.first_moment = ()
        self.second_moment = ()

    def state(self) -> dict[str, Any]:
        """可 json.dumps 的状态."""
        return {
            **super().state(),
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "first_moment": list(self.first_moment),
            "second_moment": list(self.second_moment),
        }


#: 优化器类表（键与 :data:`types.OPTIMIZERS` 逐键对齐）.
OPTIMIZER_CLASSES: dict[str, type[Optimizer]] = {
    OPTIMIZER_SGD: SGDOptimizer,
    OPTIMIZER_MOMENTUM: MomentumOptimizer,
    OPTIMIZER_ADAM: AdamOptimizer,
}

if set(OPTIMIZER_CLASSES) != set(OPTIMIZERS):  # pragma: no cover - 导入期不变期
    raise ParameterError(
        "优化器类表与 OPTIMIZERS 不一致：少一项的那个优化器会静默地不可用，"
        "而'不可用'与'没人用'在报告里长得一样。"
    )


def make_optimizer(name: str, learning_rate: float, **params: Any) -> Optimizer:
    """按名字造优化器（参数由各优化器自己的签名决定）."""
    if name not in OPTIMIZER_CLASSES:
        raise ParameterError(
            f"不认识的优化器 {name!r}：可用取值 {list(OPTIMIZERS)}。"
        )
    return OPTIMIZER_CLASSES[name](learning_rate, **params)


# --------------------------------------------------------------------------- #
# 训练回路
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainingTrace:
    """一次优化的全过程：每一刻的损失、学习率与参数.

    ```text
    losses          长度 steps + 1（含**初始损失**，否则"下降了没有"无法判断）
    learning_rates  长度 steps（每一步实际用的 lr，调度生效的证据）
    params          长度 steps + 1（含初始参数，用来画轨迹）
    ```

    三个长度刻意不一致并写在这里：``losses`` 比 ``learning_rates`` 多一个，
    因为"第 0 步"是一个真实的观测（还没迈步时的损失），
    而它没有对应的学习率。把两者都写成 ``steps`` 会让报告里少一行初始损失，
    于是"下降了多少"变成"相对第一次更新之后的损失"——一个没有意义的参照点。
    """

    losses: tuple[float, ...]
    learning_rates: tuple[float, ...]
    params: tuple[Vector, ...]
    converged: bool = False
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if len(self.losses) != len(self.params):
            raise NumericError(
                f"损失有 {len(self.losses)} 个而参数有 {len(self.params)} 个："
                "两者必须一一对应（初始点 + 每一步）。"
            )
        if len(self.learning_rates) != len(self.losses) - 1:
            raise NumericError(
                f"学习率有 {len(self.learning_rates)} 个而损失有 {len(self.losses)} 个："
                "学习率比损失**少一个**（初始损失没有对应的学习率）。"
            )
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def steps(self) -> int:
        """实际走过的步数."""
        return len(self.learning_rates)

    @property
    def initial_loss(self) -> float:
        """初始损失（起点，作为"下降了多少"的参照）."""
        return self.losses[0]

    @property
    def final_loss(self) -> float:
        """最终损失."""
        return self.losses[-1]

    @property
    def best_loss(self) -> float:
        """过程中的最小损失（它可能**不等于**最终损失：后期可能在最优点附近来回抖）."""
        return min(self.losses)

    @property
    def best_index(self) -> int:
        """最小损失出现在第几步（0 = 初始点）."""
        return min(range(len(self.losses)), key=lambda index: self.losses[index])

    @property
    def improvement(self) -> float:
        """相对下降比例 ``(L₀ − L_T) / L₀``（初始损失为 0 时记 0.0）."""
        if self.initial_loss == 0.0:
            return 0.0
        return (self.initial_loss - self.final_loss) / abs(self.initial_loss)

    def monotone(self) -> bool:
        """损失是否**一路不升**（更严格的判断：每一步都比上一步小）."""
        return all(
            following <= previous for previous, following in zip(self.losses, self.losses[1:])
        )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "steps": self.steps,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "best_index": self.best_index,
            "improvement": self.improvement,
            "monotone": self.monotone(),
            "converged": self.converged,
            "learning_rates": list(self.learning_rates),
            "losses": list(self.losses),
            "params": [list(item) for item in self.params],
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``20 步 | 损失 4.520000 → 0.000123（↓100.0%）| 单调 是``."""
        return (
            f"{self.steps} 步 | 损失 {self.initial_loss:.6f} → {self.final_loss:.6f}"
            f"（↓{self.improvement:.1%}）| 单调 {'是' if self.monotone() else '否'}"
        )

    def loss_line(self, index: int) -> str:
        """第 ``index`` 步的一行读数（含该步的学习率）."""
        if index < 0 or index >= len(self.losses):
            raise ParameterError(f"步号 {index} 超出范围 [0, {len(self.losses) - 1}]。")
        if index == 0:
            return f"step {index:>4} | lr      — | loss {self.losses[index]:.8f}"
        rate = self.learning_rates[index - 1]
        return f"step {index:>4} | lr {rate:.6f} | loss {self.losses[index]:.8f}"


def minimize(
    objective: Objective,
    initial: Vector,
    *,
    optimizer: Optimizer,
    steps: int,
    step_size: float = DEFAULT_STEP,
    method: str = CENTRAL,
    schedule: Schedule | None = None,
    tolerance: float = 1e-12,
) -> TrainingTrace:
    """跑 ``steps`` 步梯度下降，返回全过程 :class:`TrainingTrace`.

    梯度由 :func:`calculus.gradient` 用数值差分给出——**这是慢但不会错的选择**：
    它让"优化器对不对"这件事与"梯度算得对不对"这件事解耦：
    两者一起出问题时，先怀疑这一层；这一层被数值梯度钉住之后，
    再换解析梯度（:mod:`autograd` 或手写反向）就只可能错在"那一次替换"上。

    ``schedule`` 非空时每一步都会把 ``optimizer.learning_rate`` 更新成
    ``schedule(step)``（``step`` 从 1 开始）——注意是**每一步之前**更新，
    这样报告里的 ``learning_rates[i]`` 就是第 ``i+1`` 步真正用的那个值。
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ParameterError(f"steps 必须是 >= 1 的整数，收到 {steps!r}。")
    if not is_finite(tolerance) or tolerance < 0:
        raise ParameterError(f"tolerance 必须是非负的有限数，收到 {tolerance!r}。")
    params = validate_vector(initial, name="initial")
    losses: list[float] = []
    rates: list[float] = []
    history: list[Vector] = [params]
    current_loss = _checked_loss(objective, params)
    losses.append(current_loss)
    for index in range(1, steps + 1):
        if schedule is not None:
            optimizer.learning_rate = _checked_positive(schedule(index), name=f"第 {index} 步的学习率")
        grads = gradient(objective, params, step=step_size, method=method)
        rates.append(optimizer.learning_rate)
        params = optimizer.step(params, grads)
        current_loss = _checked_loss(objective, params)
        losses.append(current_loss)
        history.append(params)
        if current_loss <= tolerance:
            break
    converged = losses[-1] <= tolerance
    return TrainingTrace(
        losses=tuple(losses),
        learning_rates=tuple(rates),
        params=tuple(history),
        converged=converged,
        notes=(
            f"梯度由数值差分给出（{method} 方法，h={step_size:g}）——"
            "慢但独立于任何推导，因此它验证的是优化器本身",
            "初始损失记在 losses[0]：没有它，'下降了多少'就没有参照点",
            f"收敛判据：损失 <= {tolerance:g}（{'已满足' if converged else '未满足'}）",
        ),
    )


def _checked_loss(objective: Objective, params: Vector) -> float:
    """调用一次目标函数并校验返回值（非有限数当场拒绝）."""
    value = objective(params)
    if not is_finite(value):
        raise NumericError(
            f"目标函数返回了非有限数（{value!r}）："
            "在 inf/nan 上做梯度下降会得到 nan 参数，"
            "而 nan 会一路传到'报告里的均值也是 nan'，根因却在更早的一次除法或 log 上。"
        )
    return float(value)


# --------------------------------------------------------------------------- #
# 形状工具：给"参数是矩阵"的层用
# --------------------------------------------------------------------------- #


def flatten_matrices(
    matrices: Sequence[Matrix],
) -> tuple[Vector, tuple[tuple[int, int], ...]]:
    """把若干矩阵按行优先压平成一串数，返回 ``(向量, 各矩阵形状)``.

    为什么参数要压平：优化器的更新公式是**逐分量**的一元运算
    （``θ ← θ − lr·g``），与参数是不是矩阵无关。压平之后
    ``AdamOptimizer`` 不需要知道它优化的是一个矩阵，
    而 day075 的注意力层只要把梯度也压平就能复用同一个优化器。
    """
    if not matrices:
        raise ShapeError("至少要有一个矩阵：空参数列表没有形状。")
    shapes: list[tuple[int, int]] = []
    flat: list[float] = []
    for index, matrix in enumerate(matrices):
        checked = validate_matrix(matrix, name=f"matrix[{index}]")
        shapes.append((len(checked), len(checked[0])))
        for row in checked:
            flat.extend(row)
    return tuple(flat), tuple(shapes)


def unflatten_matrices(
    flat: Vector,
    shapes: Sequence[tuple[int, int]],
) -> tuple[Matrix, ...]:
    """把一串数按给定形状还原成若干矩阵（长度对不上时当场报错）."""
    checked = validate_vector(flat, name="flat")
    required = sum(rows * columns for rows, columns in shapes)
    if len(checked) != required:
        raise ShapeError(
            f"压平的向量有 {len(checked)} 个数，而形状表需要 {required} 个："
            "长度对不上时'按顺序切'会把切错的位置静默地填进某个矩阵，"
            "而结果是若干'形状正确、数值错位'的参数。"
        )
    matrices: list[Matrix] = []
    cursor = 0
    for rows, columns in shapes:
        rows_data = []
        for _ in range(rows):
            rows_data.append(checked[cursor : cursor + columns])
            cursor += columns
        matrices.append(tuple(rows_data))
    return tuple(matrices)


__all__ = [
    "DEFAULT_EPSILON",
    "OPTIMIZER_CLASSES",
    "SCHEDULE_FACTORIES",
    "AdamOptimizer",
    "MomentumOptimizer",
    "Objective",
    "Optimizer",
    "SGDOptimizer",
    "Schedule",
    "TrainingTrace",
    "clip_by_global_norm",
    "clip_by_value",
    "constant_schedule",
    "cosine_schedule",
    "flatten_matrices",
    "global_norm",
    "make_optimizer",
    "make_schedule",
    "minimize",
    "step_decay_schedule",
    "unflatten_matrices",
    "warmup_cosine_schedule",
]
