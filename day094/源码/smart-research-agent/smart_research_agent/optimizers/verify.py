"""``verify``：七条性质与两类判据（day092 / M8-D3）.

判据只有两类，与 day090 逐字相同：**相等**（逐位 / 整数 / 计数）与**不超过上界**
（相对误差），本课再加一类**不低于下界**（用来断言"两个东西**确实不同**"）。

```text
① shared_rules_match_day074                        相等：共享的三个规则逐步与 day074 逐位一致
② nesterov_is_not_momentum                         下界：lookahead 真的改变了轨迹
③ decoupled_decay_differs_under_varying_gradients  下界：AdamW 与 Adam+L2 不是同一个东西
④ rmsprop_equalizes_scale                          上界：两个相差 1e4 的梯度给出相同的等效步长
⑤ clip_preserves_direction                         上界：裁剪只改长度、不改方向
⑥ clipping_matches_day074                          相等：裁剪的数学只有一份（day074）
⑦ schedule_matches_day074                          相等：热身的计步口径只有一份（day074）
```

## 为什么"不同"也要一条判据

前六天所有判据都是"**≤ 某个上界**"或"**逐位相等**"——它们回答"这个对不对"。
但 day092 有一半的内容是"**两个规则不是一回事**"（Nesterov vs 动量、AdamW vs Adam+L2）。
如果只写"对上界"，那么把 Nesterov 写成普通动量、把解耦写成耦合，**报告会全绿**：
因为两种写法都能算出"看起来正常"的参数。于是本课给 ``Check`` 加一个
``lower_bound``：**读数必须不小于它**——专门用来钉住"它们确实不同"。
把两类混成一个判据，就会出现"两个规则写成了一个，报告却通过"这种事。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from smart_research_agent.math_foundations.optim import (
    clip_by_global_norm as math_clip_by_global_norm,
)
from smart_research_agent.math_foundations.optim import minimize
from smart_research_agent.math_foundations.optim import (
    warmup_cosine_schedule as math_warmup_cosine_schedule,
)
from smart_research_agent.math_foundations.optim import make_optimizer as math_make_optimizer
from smart_research_agent.optimizers.compare import anisotropic_valley
from smart_research_agent.optimizers.optimizer import TrainingOptimizer
from smart_research_agent.optimizers.rules import (
    clip_gradients,
    cosine_between,
    rmsprop_step,
)
from smart_research_agent.optimizers.types import (
    OPTIMIZER_ADAM,
    OPTIMIZER_ADAMW,
    OPTIMIZER_MOMENTUM,
    OPTIMIZER_NESTEROV,
    OPTIMIZER_PROPERTIES,
    OPTIMIZER_RMSPROP,
    OPTIMIZER_SGD,
    PROPERTY_CLIPPING_MATCHES_DAY074,
    PROPERTY_CLIP_PRESERVES_DIRECTION,
    PROPERTY_DECOUPLED_DECAY_DIFFERS_UNDER_VARYING_GRADIENTS,
    PROPERTY_NESTEROV_IS_NOT_MOMENTUM,
    PROPERTY_RMSPROP_EQUALIZES_SCALE,
    PROPERTY_SCHEDULE_MATCHES_DAY074,
    PROPERTY_SHARED_RULES_MATCH_DAY074,
)

#: 共享规则对账用的写死输入（参数与一串梯度）——读数因此可重跑.
SHARED_SAMPLE_PARAMS: tuple[float, ...] = (1.0, -2.0, 3.0)
SHARED_SAMPLE_GRADS: tuple[tuple[float, ...], ...] = (
    (0.5, -1.0, 2.0),
    (-0.25, 0.75, -0.5),
    (1.0, 0.5, -1.5),
    (0.1, -0.2, 0.3),
    (-0.6, 0.4, 0.9),
)

#: 病态峡谷上比较两个规则用的起点与步数.
COMPARE_START: tuple[float, ...] = (3.0, 2.0)
COMPARE_STEPS = 25

#: RMSProp 尺度归一用的两个梯度（相差 1e4）.
RMS_SCALE_BIG = 100.0
RMS_SCALE_SMALL = 0.01
RMS_EQUILIBRIUM_STEPS = 200

#: 裁剪性质用的写死梯度与阈值.
CLIP_SAMPLE_GRADS: tuple[float, ...] = (100.0, 1.0, -50.0)
CLIP_MAX_NORM = 1.0

#: 调度对账用的参数.
SCHEDULE_BASE_LR = 0.1
SCHEDULE_WARMUP_STEPS = 5
SCHEDULE_TOTAL_STEPS = 20
SCHEDULE_MIN_LR = 0.0


@dataclass(frozen=True)
class Check:
    """一次性质校验的读数与判据（相等 / 上界 / 下界）."""

    reading: float
    upper_bound: float | None = None
    lower_bound: float | None = None
    left: str = "-"
    right: str = "-"

    def passed(self) -> bool:
        """读数是否落在判据内（上界 <= / 下界 >=；两者都给时两个都要满足）."""
        if self.upper_bound is not None and self.reading > self.upper_bound:
            return False
        if self.lower_bound is not None and self.reading < self.lower_bound:
            return False
        return True

    def relation(self) -> str:
        """人类可读的判据（``==`` / ``<=`` / ``>=`` / ``∈``）."""
        if self.upper_bound is not None and self.lower_bound is not None:
            return "∈"
        if self.upper_bound is not None:
            return "<="
        if self.lower_bound is not None:
            return ">="
        return "=="

    def bound_text(self) -> str:
        """判据的一行文本（``<= 1.0e-12`` 之类）."""
        relation = self.relation()
        if relation == "∈":
            return f"∈ [{self.lower_bound:.1e}, {self.upper_bound:.1e}]"
        if relation == "<=":
            return f"<= {self.upper_bound:.1e}"
        if relation == ">=":
            return f">= {self.lower_bound:.1e}"
        return "== 逐位"


@dataclass(frozen=True)
class PropertyOutcome:
    """一条性质的结论：是否通过 + 现场读数 + 两个来源."""

    name: str
    passed: bool
    check: Check

    def line(self) -> str:
        """``通过 shared_rules_match_day074 | 读数 0.000e+00 == 逐位 | a vs b``."""
        mark = "通过" if self.passed else "失败"
        return (
            f"{mark} {self.name:<48} | 读数 {self.check.reading:.3e} "
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
# 单条性质的实现
# --------------------------------------------------------------------------- #


def _run_ours(name: str, *, steps: int, objective, start, **kwargs) -> tuple[float, ...]:
    """用本包的 ``TrainingOptimizer`` 跑一条确定性轨迹，返回**最终参数**.

    只跑本包这一侧：跨天对账由 :func:`check_shared_rules_match_day074` 与
    :func:`check_clipping_matches_day074` 单独负责（那两条才去调 day074 的对象）。
    """
    optimizer = TrainingOptimizer(name, 0.05, **kwargs)
    trace = minimize(objective, start, optimizer=optimizer, steps=steps)
    return trace.params[-1]


def _max_abs_gap(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """两个同形向量的最大绝对差."""
    return max((abs(a - b) for a, b in zip(left, right)), default=0.0)


def check_shared_rules_match_day074() -> PropertyOutcome:
    """① 共享的三个规则逐步与 day074 逐位一致（读数 = 最大绝对差，判据 == 0）."""
    readings: list[float] = []
    for name in (OPTIMIZER_SGD, OPTIMIZER_MOMENTUM, OPTIMIZER_ADAM):
        ours = TrainingOptimizer(name, 0.05)
        theirs = math_make_optimizer(
            name,
            0.05,
            **(
                {"momentum": 0.9}
                if name == OPTIMIZER_MOMENTUM
                else (
                    {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}
                    if name == OPTIMIZER_ADAM
                    else {}
                )
            ),
        )
        params_ours: tuple[float, ...] = SHARED_SAMPLE_PARAMS
        params_theirs: tuple[float, ...] = SHARED_SAMPLE_PARAMS
        for grads in SHARED_SAMPLE_GRADS:
            params_ours = ours.step(params_ours, grads)
            params_theirs = theirs.step(params_theirs, grads)
        readings.append(_max_abs_gap(params_ours, params_theirs))
    check = Check(
        reading=max(readings, default=0.0),
        upper_bound=0.0,
        left="optimizers.TrainingOptimizer",
        right="math_foundations.optim",
    )
    return PropertyOutcome(PROPERTY_SHARED_RULES_MATCH_DAY074, check.passed(), check)


def check_nesterov_is_not_momentum() -> PropertyOutcome:
    """② Nesterov 与普通动量在同一轨迹上给出不同的一步（读数 = 最终参数最大差）."""
    momentum_params = _run_ours(OPTIMIZER_MOMENTUM, steps=COMPARE_STEPS, objective=anisotropic_valley, start=COMPARE_START)
    nesterov_params = _run_ours(OPTIMIZER_NESTEROV, steps=COMPARE_STEPS, objective=anisotropic_valley, start=COMPARE_START)
    check = Check(
        reading=_max_abs_gap(momentum_params, nesterov_params),
        lower_bound=1e-6,
        left="momentum",
        right="nesterov",
    )
    return PropertyOutcome(PROPERTY_NESTEROV_IS_NOT_MOMENTUM, check.passed(), check)


def check_decoupled_decay_differs_under_varying_gradients() -> PropertyOutcome:
    """③ 梯度尺度变化时，AdamW（解耦）与 Adam+L2（耦合）的轨迹显著不同."""
    adamw = TrainingOptimizer(OPTIMIZER_ADAMW, 0.05, weight_decay=0.1)
    adam_l2 = TrainingOptimizer(OPTIMIZER_ADAM, 0.05, weight_decay=0.1, decay_mode="l2")
    adamw_trace = minimize(anisotropic_valley, COMPARE_START, optimizer=adamw, steps=COMPARE_STEPS)
    adam_l2_trace = minimize(anisotropic_valley, COMPARE_START, optimizer=adam_l2, steps=COMPARE_STEPS)
    check = Check(
        reading=_max_abs_gap(adamw_trace.params[-1], adam_l2_trace.params[-1]),
        lower_bound=1e-9,
        left="adamw(decoupled)",
        right="adam(coupled L2)",
    )
    return PropertyOutcome(PROPERTY_DECOUPLED_DECAY_DIFFERS_UNDER_VARYING_GRADIENTS, check.passed(), check)


def check_rmsprop_equalizes_scale() -> PropertyOutcome:
    """④ 两个相差 1e4 的常量梯度，在 RMSProp 下得到几乎相同的每步位移.

    做法：各自跑 ``RMS_STEPS`` 步常量梯度让二阶动量进入稳态，再取**最后一步的
    位移绝对值**比较。SGD 会差 1e4 倍，而 RMSProp 只差一个 ε 造成的极小量。
    """
    def steady_step(gradient: float) -> float:
        params = (0.0,)
        square = (0.0,)
        displacement = 0.0
        for _ in range(RMS_EQUILIBRIUM_STEPS):
            grads = (gradient,)
            previous = params[0]
            params, square = rmsprop_step(
                params, grads, square, learning_rate=0.05, alpha=0.99, epsilon=1e-8
            )
            displacement = abs(params[0] - previous)
            # 位移只在 x 轴上累计会走到负无穷：这里每一步把参数拉回 0 只保留"步长"
            params = (0.0,)
        return displacement

    reading = abs(steady_step(RMS_SCALE_BIG) - steady_step(RMS_SCALE_SMALL))
    check = Check(
        reading=reading,
        upper_bound=1e-5,
        left=f"g={RMS_SCALE_BIG:g}",
        right=f"g={RMS_SCALE_SMALL:g}",
    )
    return PropertyOutcome(PROPERTY_RMSPROP_EQUALIZES_SCALE, check.passed(), check)


def check_clip_preserves_direction() -> PropertyOutcome:
    """⑤ 整体范数裁剪只改长度、不改方向（读数 = 1 − 余弦，判据 <= 1e-12）."""
    clipped, _factor = clip_gradients(CLIP_SAMPLE_GRADS, CLIP_MAX_NORM)
    angle = 1.0 - cosine_between(CLIP_SAMPLE_GRADS, clipped)
    check = Check(
        reading=angle,
        upper_bound=1e-12,
        left="clip_by_global_norm",
        right="原始方向",
    )
    return PropertyOutcome(PROPERTY_CLIP_PRESERVES_DIRECTION, check.passed(), check)


def check_clipping_matches_day074() -> PropertyOutcome:
    """⑥ 本课的裁剪与 day074 的 ``clip_by_global_norm`` 逐位一致（读数 = 最大差 + 系数差）."""
    ours, our_factor = clip_gradients(CLIP_SAMPLE_GRADS, CLIP_MAX_NORM)
    theirs, their_factor = math_clip_by_global_norm(CLIP_SAMPLE_GRADS, CLIP_MAX_NORM)
    reading = max(_max_abs_gap(ours, theirs), abs(our_factor - their_factor))
    check = Check(
        reading=reading,
        upper_bound=0.0,
        left="optimizers.rules.clip_gradients",
        right="math_foundations.optim.clip_by_global_norm",
    )
    return PropertyOutcome(PROPERTY_CLIPPING_MATCHES_DAY074, check.passed(), check)


def check_schedule_matches_day074() -> PropertyOutcome:
    """⑦ 本课接线出的热身+余弦与 day074 的 ``warmup_cosine_schedule`` 逐点一致."""
    optimizer = TrainingOptimizer(OPTIMIZER_SGD, SCHEDULE_BASE_LR)
    optimizer.set_schedule(
        "warmup_cosine",
        base_lr=SCHEDULE_BASE_LR,
        warmup_steps=SCHEDULE_WARMUP_STEPS,
        total_steps=SCHEDULE_TOTAL_STEPS,
        min_lr=SCHEDULE_MIN_LR,
    )
    readings: list[float] = []
    for step in range(1, SCHEDULE_TOTAL_STEPS + 1):
        ours = optimizer.schedule(step)
        theirs = math_warmup_cosine_schedule(
            step,
            base_lr=SCHEDULE_BASE_LR,
            warmup_steps=SCHEDULE_WARMUP_STEPS,
            total_steps=SCHEDULE_TOTAL_STEPS,
            min_lr=SCHEDULE_MIN_LR,
        )
        readings.append(abs(ours - theirs))
    check = Check(
        reading=max(readings, default=0.0),
        upper_bound=0.0,
        left="optimizers.TrainingOptimizer.schedule",
        right="math_foundations.optim.warmup_cosine_schedule",
    )
    return PropertyOutcome(PROPERTY_SCHEDULE_MATCHES_DAY074, check.passed(), check)


#: 七条性质的名字 -> 检查函数（键顺序 = :data:`types.OPTIMIZER_PROPERTIES`）.
CHECKS = {
    PROPERTY_SHARED_RULES_MATCH_DAY074: check_shared_rules_match_day074,
    PROPERTY_NESTEROV_IS_NOT_MOMENTUM: check_nesterov_is_not_momentum,
    PROPERTY_DECOUPLED_DECAY_DIFFERS_UNDER_VARYING_GRADIENTS: (
        check_decoupled_decay_differs_under_varying_gradients
    ),
    PROPERTY_RMSPROP_EQUALIZES_SCALE: check_rmsprop_equalizes_scale,
    PROPERTY_CLIP_PRESERVES_DIRECTION: check_clip_preserves_direction,
    PROPERTY_CLIPPING_MATCHES_DAY074: check_clipping_matches_day074,
    PROPERTY_SCHEDULE_MATCHES_DAY074: check_schedule_matches_day074,
}

if set(CHECKS) != set(OPTIMIZER_PROPERTIES):  # pragma: no cover - 导入期不变式
    from smart_research_agent.optimizers.errors import OptimizerError

    raise OptimizerError(
        "性质名单与检查函数表不一致：少一条的性质会静默地不在报告里出现，"
        "而'少一条'与'它通过了'在读报告时长得一样。"
    )


def check_all() -> PropertyReport:
    """跑完七条性质，返回汇总报告（顺序与 :data:`types.OPTIMIZER_PROPERTIES` 一致）."""
    report = PropertyReport()
    for name in OPTIMIZER_PROPERTIES:
        report.outcomes.append(CHECKS[name]())
    return report


__all__ = [
    "CHECKS",
    "CLIP_MAX_NORM",
    "CLIP_SAMPLE_GRADS",
    "COMPARE_START",
    "COMPARE_STEPS",
    "RMS_EQUILIBRIUM_STEPS",
    "RMS_SCALE_BIG",
    "RMS_SCALE_SMALL",
    "SCHEDULE_BASE_LR",
    "SCHEDULE_MIN_LR",
    "SCHEDULE_TOTAL_STEPS",
    "SCHEDULE_WARMUP_STEPS",
    "SHARED_SAMPLE_GRADS",
    "SHARED_SAMPLE_PARAMS",
    "Check",
    "PropertyOutcome",
    "PropertyReport",
    "check_all",
    "check_clip_preserves_direction",
    "check_clipping_matches_day074",
    "check_decoupled_decay_differs_under_varying_gradients",
    "check_nesterov_is_not_momentum",
    "check_rmsprop_equalizes_scale",
    "check_schedule_matches_day074",
    "check_shared_rules_match_day074",
]
