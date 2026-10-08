"""``optimizer``：把"衰减 + 裁剪 + 调度 + 更新规则"组装成**一次真正的更新**（day092 / M8-D3）.

day074 给了三个更新公式与两个裁剪原语，day090 给了梯度。这一模块把它们接成
一个**训练期优化器**：它接收 ``(参数, 梯度)``，返回新的参数，并在内部按固定顺序
做四件事：

```text
① 取学习率    步号自增；有调度就按 lr(t) 取值（**每一步之前**取，报告里的 lr 才是真用上的）
② 裁剪        可选：先按整体范数裁剪梯度（记录缩放系数与裁剪前范数）
③ 更新规则    sgd / momentum / adam 直接复用 day074；nesterov / rmsprop 用本包的纯函数
④ 衰减        耦合（λθ 进梯度，在③之前）或解耦（θ·(1−lr·λ)，在③之后）
```

## 顺序为什么不能乱

```text
衰减在裁剪之后 / 之前   λθ 是"参数的一部分"，它不该被梯度裁剪的阈值管到
                        （若先加 λθ 再裁剪，衰减会被裁剪系数一起缩掉）
解耦衰减必须在更新之后  它是"更新完再缩一下参数"，不是"把衰减当成梯度"
```

两者都是**不会报错、只会算错**的顺序问题：结果"看起来正常"，只是那个衰减系数
不再等于你设的那个值。因此本模块把顺序写死在 ``step`` 里，不给调用方自己拼的机会。

## 与 day074 的接缝

共享的三个规则**一字不改**地交给 ``math_foundations.optim`` 的类；本模块只负责
"什么时候调用它、调用前后各做什么"。因此"同一条公式写了两遍"这件事不会发生。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from smart_research_agent.math_foundations.optim import (
    make_optimizer as math_make_optimizer,
)
from smart_research_agent.math_foundations.optim import (
    make_schedule as math_make_schedule,
)
from smart_research_agent.optimizers.errors import (
    ParameterError,
    ShapeError,
    StateError,
    StepError,
)
from smart_research_agent.optimizers.rules import (
    DEFAULT_ALPHA,
    DEFAULT_BETA1,
    DEFAULT_BETA2,
    DEFAULT_EPSILON,
    DEFAULT_MOMENTUM,
    apply_decoupled_decay,
    as_vector,
    checked_decay_mode,
    checked_fraction,
    checked_non_negative,
    checked_pair,
    checked_positive,
    clip_gradients,
    global_norm,
    l2_gradient,
    nesterov_step,
    rmsprop_step,
)
from smart_research_agent.optimizers.types import (
    DECAY_DECOUPLED,
    DECAY_L2,
    DECAY_NONE,
    OPTIMIZER_ADAM,
    OPTIMIZER_ADAMW,
    OPTIMIZER_DESCRIPTIONS,
    OPTIMIZER_MOMENTUM,
    OPTIMIZER_NESTEROV,
    OPTIMIZER_RMSPROP,
    OPTIMIZER_SGD,
    TRAIN_OPTIMIZERS,
)

#: 一条学习率曲线：给"第几步"（从 1 开始）返回这一步的学习率.
Schedule = Callable[[int], float]

#: 共享 day074 的三个规则（其余三个由本包实现）.
SHARED_RULES: tuple[str, ...] = (OPTIMIZER_SGD, OPTIMIZER_MOMENTUM, OPTIMIZER_ADAM)

#: 规则名 -> 它实际复用/实现的**核心**规则（AdamW 的核心是 Adam）.
CORE_RULE_FOR: dict[str, str] = {
    OPTIMIZER_SGD: OPTIMIZER_SGD,
    OPTIMIZER_MOMENTUM: OPTIMIZER_MOMENTUM,
    OPTIMIZER_NESTEROV: OPTIMIZER_NESTEROV,
    OPTIMIZER_RMSPROP: OPTIMIZER_RMSPROP,
    OPTIMIZER_ADAM: OPTIMIZER_ADAM,
    OPTIMIZER_ADAMW: OPTIMIZER_ADAM,
}


class TrainingOptimizer:
    """训练期优化器：持有学习率/衰减/裁剪/调度与跨步状态，``step(params, grads)`` 走一步.

    它与 day074 的 ``Optimizer`` **接口兼容**（都有 ``learning_rate`` 属性与
    ``step(params, grads)`` 方法），因此可以直接喂给 day074 的 ``minimize``——
    这正是 :mod:`optimizers.compare` 能把六个变体放在同一条流水线上比较的原因。
    """

    def __init__(
        self,
        name: str,
        learning_rate: float,
        *,
        momentum: float = DEFAULT_MOMENTUM,
        alpha: float = DEFAULT_ALPHA,
        beta1: float = DEFAULT_BETA1,
        beta2: float = DEFAULT_BETA2,
        epsilon: float = DEFAULT_EPSILON,
        weight_decay: float = 0.0,
        decay_mode: str | None = None,
        max_grad_norm: float | None = None,
        schedule: Schedule | None = None,
    ) -> None:
        if name not in TRAIN_OPTIMIZERS:
            raise ParameterError(
                f"不认识的优化器 {name!r}：可用取值 {list(TRAIN_OPTIMIZERS)}。"
            )
        self.name = name
        self.learning_rate = checked_positive(learning_rate, name="learning_rate")
        self.momentum = checked_fraction(momentum, name="momentum")
        self.alpha = checked_fraction(alpha, name="alpha")
        self.beta1 = checked_fraction(beta1, name="beta1")
        self.beta2 = checked_fraction(beta2, name="beta2")
        self.epsilon = checked_positive(epsilon, name="epsilon")
        if not math.isfinite(weight_decay) or weight_decay < 0.0:
            raise ParameterError(f"weight_decay 必须是非负的有限数，收到 {weight_decay!r}。")
        self.weight_decay = float(weight_decay)
        # AdamW 的默认模式是解耦（这是它存在的理由）；其余默认不衰减
        default_mode = DECAY_DECOUPLED if name == OPTIMIZER_ADAMW else DECAY_NONE
        self.decay_mode = checked_decay_mode(decay_mode or default_mode)
        if max_grad_norm is not None:
            max_grad_norm = checked_positive(max_grad_norm, name="max_grad_norm")
        self.max_grad_norm = max_grad_norm
        self.schedule = schedule
        self.step_count = 0
        self.last_grad_norm = 0.0
        self.last_clip_factor = 1.0
        # 跨步状态：共享规则交给 day074 的对象持有；新增规则由本对象持有
        self._base = self._build_base()
        self.velocity: tuple[float, ...] = ()
        self.square_avg: tuple[float, ...] = ()

    # ------------------------------------------------------------------ 构造
    def _build_base(self):
        """按名字造一个 day074 的优化器（只对共享的三个规则；其余返回 None）."""
        core = CORE_RULE_FOR[self.name]
        if core not in SHARED_RULES:
            return None
        params: dict[str, Any] = {}
        if core == OPTIMIZER_MOMENTUM:
            params["momentum"] = self.momentum
        elif core == OPTIMIZER_ADAM:
            params["beta1"] = self.beta1
            params["beta2"] = self.beta2
            params["epsilon"] = self.epsilon
        return math_make_optimizer(core, self.learning_rate, **params)

    # ------------------------------------------------------------------ 状态
    @property
    def core_rule(self) -> str:
        """它实际执行的核心规则（AdamW 报 ``adam``）."""
        return CORE_RULE_FOR[self.name]

    def set_schedule(self, name: str, **params: Any) -> None:
        """按名字装一条学习率曲线（工厂复用 day074 的 ``make_schedule``）."""
        self.schedule = math_make_schedule(name, **params)

    def reset(self) -> None:
        """清空步数与全部跨步状态（**参数不动**）.

        为什么必须清：换任务或换学习率曲线时，"上一段的动量"没有任何意义，
        而它会被悄悄带进新任务的前几步——表现为"换了任务之后前几步莫名地大"。
        """
        self.step_count = 0
        self.last_grad_norm = 0.0
        self.last_clip_factor = 1.0
        self.velocity = ()
        self.square_avg = ()
        if self._base is not None:
            self._base.reset()

    def state(self) -> dict[str, Any]:
        """可 ``json.dumps`` 的状态（只含"跨步保留的东西"）."""
        payload: dict[str, Any] = {
            "name": self.name,
            "core_rule": self.core_rule,
            "learning_rate": self.learning_rate,
            "step_count": self.step_count,
            "weight_decay": self.weight_decay,
            "decay_mode": self.decay_mode,
            "max_grad_norm": self.max_grad_norm,
            "last_grad_norm": self.last_grad_norm,
            "last_clip_factor": self.last_clip_factor,
        }
        if self.velocity:
            payload["velocity"] = list(self.velocity)
        if self.square_avg:
            payload["square_avg"] = list(self.square_avg)
        if self._base is not None:
            payload["base"] = self._base.state()
        return payload

    def describe(self) -> str:
        """一行说明（进报告：报告里要能读出"这个优化器是什么"）."""
        clip = "none" if self.max_grad_norm is None else f"{self.max_grad_norm:g}"
        return (
            f"{self.name}（{OPTIMIZER_DESCRIPTIONS[self.name]}），lr={self.learning_rate:g}"
            f"，decay={self.decay_mode}:{self.weight_decay:g}，clip={clip}"
        )

    # ------------------------------------------------------------------ 更新
    def step(self, params, grads) -> tuple[float, ...]:
        """按固定顺序走一步：取 lr → 裁剪 → 更新规则 → 衰减."""
        checked_params, checked_grads = checked_pair(params, grads)
        self.step_count += 1
        # ① 学习率：调度在**这一步之前**取值，报告里的 lr 才是真用上的那个
        #    （用非负校验：余弦退火的最后一步合法地取到 min_lr=0）
        if self.schedule is not None:
            self.learning_rate = checked_non_negative(
                self.schedule(self.step_count), name=f"第 {self.step_count} 步的学习率"
            )
        # ② 裁剪（可选）：先记录裁剪前的范数，再替换梯度
        if self.max_grad_norm is not None:
            self.last_grad_norm = global_norm(checked_grads)
            checked_grads, self.last_clip_factor = clip_gradients(checked_grads, self.max_grad_norm)
        else:
            self.last_grad_norm = global_norm(checked_grads)
            self.last_clip_factor = 1.0
        # ③ 耦合衰减：λθ 进梯度（在更新规则之前）
        effective = checked_grads
        if self.decay_mode == DECAY_L2:
            effective = l2_gradient(checked_grads, checked_params, self.weight_decay)
        updated = self._apply_rule(checked_params, effective)
        # ④ 解耦衰减：先更新、再缩参数（AdamW 的那一步）
        if self.decay_mode == DECAY_DECOUPLED:
            updated = apply_decoupled_decay(updated, self.learning_rate, self.weight_decay)
        return self._checked_result(updated)

    def _apply_rule(self, params: tuple[float, ...], grads: tuple[float, ...]) -> tuple[float, ...]:
        """把更新规则作用到 ``(params, grads)`` 上（共享规则交给 day074）."""
        core = self.core_rule
        if self._base is not None:
            self._base.learning_rate = self.learning_rate
            return self._base.step(params, grads)
        if core == OPTIMIZER_NESTEROV:
            if not self.velocity:
                self.velocity = tuple(0.0 for _ in params)
            self._check_state(self.velocity, params, name="速度")
            updated, self.velocity = nesterov_step(
                params,
                grads,
                self.velocity,
                learning_rate=self.learning_rate,
                momentum=self.momentum,
            )
            return updated
        if core == OPTIMIZER_RMSPROP:
            if not self.square_avg:
                self.square_avg = tuple(0.0 for _ in params)
            self._check_state(self.square_avg, params, name="二阶动量")
            updated, self.square_avg = rmsprop_step(
                params,
                grads,
                self.square_avg,
                learning_rate=self.learning_rate,
                alpha=self.alpha,
                epsilon=self.epsilon,
            )
            return updated
        raise ParameterError(  # pragma: no cover - 六个规则都有实现，这里只是兜底
            f"优化器 {self.name!r} 的核心规则 {core!r} 没有实现。"
        )

    def _check_state(self, state: tuple[float, ...], params: tuple[float, ...], *, name: str) -> None:
        if len(state) != len(params):
            raise StateError(
                f"{name} {len(state)} 维而参数 {len(params)} 维："
                "跨步状态必须与参数同形——长度不符最可能是换了任务却忘了 reset()。"
            )

    @staticmethod
    def _checked_result(params: tuple[float, ...]) -> tuple[float, ...]:
        """一步走完后的结果校验：参数必须仍然有限（否则抛 StepError）."""
        for index, value in enumerate(params):
            if not math.isfinite(value):
                raise StepError(
                    f"第 {index} 个参数在这一步之后变成了非有限数（{value!r}）："
                    "梯度合法不代表这一步走完还合法——先调小学习率或打开整体范数裁剪。"
                )
        return params


def make_train_optimizer(name: str, learning_rate: float, **params: Any) -> TrainingOptimizer:
    """按名字造一个训练期优化器（参数由各优化器自己的语义决定）."""
    if name not in TRAIN_OPTIMIZERS:
        raise ParameterError(
            f"不认识的优化器 {name!r}：可用取值 {list(TRAIN_OPTIMIZERS)}"
            f"（{OPTIMIZER_SGD} 是最诚实的基线）。"
        )
    return TrainingOptimizer(name, learning_rate, **params)


#: 三个共享规则 + 三个新规则，一共六个（与 :data:`types.TRAIN_OPTIMIZERS` 逐键一致）.
SUPPORTED_OPTIMIZERS: tuple[str, ...] = TRAIN_OPTIMIZERS

if set(SUPPORTED_OPTIMIZERS) != set(CORE_RULE_FOR):  # pragma: no cover - 导入期不变式
    raise ParameterError(
        "优化器名单与核心规则表不一致：少一个键的那个优化器会静默地没有更新规则，"
        "而'没有规则'与'没被用到'在报告里长得一样。"
    )


def as_shape_check(params, grads) -> None:
    """一个显式的形状检查入口（给调用方在进入训练回路前自查用）."""
    if len(as_vector(params, name="params")) != len(as_vector(grads, name="grads")):
        raise ShapeError("参数与梯度的长度不一致。")


__all__ = [
    "CORE_RULE_FOR",
    "SHARED_RULES",
    "SUPPORTED_OPTIMIZERS",
    "Schedule",
    "TrainingOptimizer",
    "as_shape_check",
    "make_train_optimizer",
]
