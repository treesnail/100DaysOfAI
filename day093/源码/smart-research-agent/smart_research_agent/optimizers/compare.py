"""``compare``：六个优化器在**同一条流水线**上的收敛对比（day092 / M8-D3）.

比较是这一课的落点：六个更新规则各有各的公式，但"哪个更好"不能靠读公式回答，
只能靠**在同一个目标函数上跑同一批步数**，把读数摆在一起。

```text
三个目标函数    良态碗（各向同性） / 病态峡谷（一个方向陡 24 倍） / Rosenbrock（香蕉形）
同一条流水线    复用 day074 的 minimize：梯度由数值差分给出（**独立于任何推导**）
三列读数        到容差用了多少步 / 最终损失 / 损失是否一路不升
```

## 为什么梯度要用 day074 的数值差分

```text
数值差分    慢，但独立于任何推导 ⇒ 它验证的是"优化器本身"
解析梯度    快，但与推导绑定     ⇒ 优化器写错时会与梯度错误混在一起，查不清是谁
```

day090 已经用"手写反向 vs 数值差分 vs 自动微分"三条路径把梯度钉住；到了这一课，
gradient 是一个**已经可信的输入**，因此比较里统一用同一把尺子（数值差分），
把变量收敛到"只剩优化器"。

## 一个必须写下来的边界

本模块的 ``CompareRow`` 是**单条轨迹的读数**，不是统计意义上的"哪个优化器更好"：
随机初始化、批大小、数据分布都没被扫。它证明的是"这几个更新规则**在这三个
已写死的确定性目标上**表现如何"，而不是一条普遍结论。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from smart_research_agent.math_foundations.optim import minimize
from smart_research_agent.math_foundations.types import Vector
from smart_research_agent.optimizers.errors import ParameterError
from smart_research_agent.optimizers.optimizer import TrainingOptimizer, make_train_optimizer

#: 数值差分步长（与 day074 的缺省一致）。
DEFAULT_STEP_SIZE = 1e-5

#: 收敛判据：损失 <= 该值即视为"到容差"。
DEFAULT_TOLERANCE = 1e-6


def quadratic_bowl(theta: Sequence[float]) -> float:
    """良态碗：``f(θ) = Σ θᵢ²``（各向同性，最容易被任何一阶方法解掉）."""
    return sum(value * value for value in theta)


def anisotropic_valley(theta: Sequence[float]) -> float:
    """病态峡谷：``f(θ) = 0.5·θ₀² + 12·θ₁²``（一个方向陡 24 倍）.

    病态是动量与 Adam 存在的**第一条理由**：SGD 在陡方向上反复横跳，
    而动量把横跳平均掉、Adam 把两个方向的尺度归一。
    """
    first = theta[0] if len(theta) > 0 else 0.0
    second = theta[1] if len(theta) > 1 else 0.0
    return 0.5 * first * first + 12.0 * second * second


def rosenbrock(theta: Sequence[float]) -> float:
    """Rosenbrock 函数 ``f = (1−θ₀)² + 100(θ₁−θ₀²)²``（香蕉形窄谷，一阶方法的天敌）."""
    first = theta[0] if len(theta) > 0 else 0.0
    second = theta[1] if len(theta) > 1 else 0.0
    return (1.0 - first) ** 2 + 100.0 * (second - first * first) ** 2


#: 三个目标函数（名字 -> 函数 / 起点 / 步数 / 该目标上的起点学习率）.
#:
#: 每个目标给**自己的**学习率，原因就是这一课要讲的那件事：一个在良态碗上刚好的
#: 学习率，在 Rosenbrock 这种窄谷上会让轨迹**发散**（读数会跑到 1e+50 上去）。
#: 把 lr 写进目标、而不是全表共用一个，是为了让对比"比的是优化器，不是比谁先炸"。
OBJECTIVES: dict[str, tuple] = {
    "quadratic_bowl": (quadratic_bowl, (3.0, -2.0), 40, 0.05),
    "anisotropic_valley": (anisotropic_valley, (3.0, 2.0), 40, 0.05),
    "rosenbrock": (rosenbrock, (-1.2, 1.0), 60, 0.002),
}

#: 对比用的六行（顺序与 :data:`types.TRAIN_OPTIMIZERS` 一致）.
DEFAULT_COMPARE_NAMES: tuple[str, ...] = (
    "sgd",
    "momentum",
    "nesterov",
    "rmsprop",
    "adam",
    "adamw",
)


@dataclass(frozen=True)
class CompareRow:
    """一次对比运行的一行读数（**两个数起步**：到容差的步数 + 最终损失）."""

    name: str
    objective: str
    steps: int
    initial_loss: float
    final_loss: float
    best_loss: float
    improvement: float
    monotone: bool
    steps_to_tolerance: int | None
    converged: bool
    diverged: bool = False

    def line(self) -> str:
        """一行读数；**发散**时只报"损失涨了多少倍"，不再印那个荒谬的百分比.

        ``sgd        | 40 步 | 损失 1.300000e+01 → 3.2e-01（↓97.5%）| 到容差  12 | 单调 是``
        ``sgd        | 60 步 | 损失 2.420000e+01 → 6.7e+57（发散）    | 到容差   — | 单调 否``
        """
        reached = "—" if self.steps_to_tolerance is None else str(self.steps_to_tolerance)
        if self.diverged:
            ratio = self.final_loss / self.initial_loss if self.initial_loss else float("inf")
            trend = f"{ratio:.1e}× 初始（发散）"
        else:
            trend = f"↓{self.improvement:.1%}"
        return (
            f"{self.name:<10} | {self.steps:>3} 步 | 损失 {self.initial_loss:.6e} → "
            f"{self.final_loss:.3e}（{trend}）| 到容差 {reached:>3} | "
            f"单调 {'是' if self.monotone else '否'}"
        )


def run_optimizer(
    optimizer: TrainingOptimizer,
    objective,
    initial: Vector,
    *,
    objective_name: str = "custom",
    steps: int = 40,
    step_size: float = DEFAULT_STEP_SIZE,
    tolerance: float = DEFAULT_TOLERANCE,
) -> CompareRow:
    """在一条确定性轨迹上跑一个优化器，返回 :class:`CompareRow`.

    梯度由 day074 的 ``minimize`` **数值差分**给出——本函数只负责把
    ``TrainingTrace`` 折成一行可比的读数。
    """
    trace = minimize(
        objective,
        initial,
        optimizer=optimizer,
        steps=steps,
        step_size=step_size,
        tolerance=tolerance,
    )
    steps_to_tolerance: int | None = None
    for index, loss in enumerate(trace.losses):
        if loss <= tolerance:
            steps_to_tolerance = index
            break
    diverged = (not math.isfinite(trace.final_loss)) or trace.final_loss > trace.initial_loss
    return CompareRow(
        name=optimizer.name,
        objective=objective_name,
        steps=trace.steps,
        initial_loss=trace.initial_loss,
        final_loss=trace.final_loss,
        best_loss=trace.best_loss,
        improvement=trace.improvement,
        monotone=trace.monotone(),
        steps_to_tolerance=steps_to_tolerance,
        converged=trace.converged,
        diverged=diverged,
    )


def compare_optimizers(
    names: Sequence[str] = DEFAULT_COMPARE_NAMES,
    objective_name: str = "quadratic_bowl",
    *,
    learning_rate: float | None = None,
    steps: int | None = None,
    step_size: float = DEFAULT_STEP_SIZE,
    tolerance: float = DEFAULT_TOLERANCE,
    **optimizer_kwargs,
) -> tuple[CompareRow, ...]:
    """在同一目标函数上跑一批优化器，返回逐行读数.

    ``learning_rate`` 为 ``None`` 时用**该目标自己的起点学习率**
    （见 :data:`OBJECTIVES`）；显式给出时全表共用它。
    ``optimizer_kwargs`` 会原样透传给每个优化器（例如 ``momentum=0.9``、
    ``weight_decay=0.01``）——使得"同一规则不同超参"也可以放进同一张表。
    """
    if objective_name not in OBJECTIVES:
        raise ParameterError(
            f"不认识的目标函数 {objective_name!r}：可用取值 {list(OBJECTIVES)}。"
        )
    function, initial, default_steps, default_lr = OBJECTIVES[objective_name]
    total_steps = default_steps if steps is None else steps
    rate = default_lr if learning_rate is None else learning_rate
    rows: list[CompareRow] = []
    for name in names:
        optimizer = make_train_optimizer(name, rate, **optimizer_kwargs)
        rows.append(
            run_optimizer(
                optimizer,
                function,
                initial,
                objective_name=objective_name,
                steps=total_steps,
                step_size=step_size,
                tolerance=tolerance,
            )
        )
    return tuple(rows)


def best_row(rows: Sequence[CompareRow]) -> CompareRow:
    """按"最终损失最小"挑出最好的一行（并列时取名字序第一，保证确定性）."""
    if not rows:
        raise ParameterError("空的对比结果没有最好的一行。")
    return min(rows, key=lambda row: (row.final_loss, row.name))


__all__ = [
    "DEFAULT_COMPARE_NAMES",
    "DEFAULT_STEP_SIZE",
    "DEFAULT_TOLERANCE",
    "OBJECTIVES",
    "CompareRow",
    "anisotropic_valley",
    "best_row",
    "compare_optimizers",
    "quadratic_bowl",
    "rosenbrock",
    "run_optimizer",
]
