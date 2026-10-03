"""``study``：六张表（day092 / M8-D3）.

每一天最后都留下几张表，因为**一个数只有被印出来才可能被反驳**。
今天六张表各回答一个"这一次更新算对了吗"的问题：

```text
① 规则表     六个规则：公式 / 状态变量 / 复用还是新增
② 一步表     六个规则从同一组写死输入各走一步：读数（首分量 + 范数）
③ 裁剪表     整体范数裁剪：裁剪前范数 / 缩放系数 / 裁剪后范数 / 方向余弦
④ 调度表     先升后降：热身与退火各取几个点，印出实际学习率
⑤ 对比表     三个目标函数上六个优化器的到容差步数与最终损失
⑥ 性质表     七条性质是否通过 / 现场读数 / 跨包对象
```

## 一条纪律：每一行都要带**两个数**

与前面各天的表同源——一行只写"通过"的表是没法反驳的。
因此每张表的每一行都至少有两列：**读数**与**参照**（公式 / 形状 / 期望 / 跨包对象）。

## 为什么"一步表"要印范数而不是全部参数

一步之后参数向量有若干个分量，全印出来没人读。**首分量 + 范数**是两个正交的读数：
首分量告诉你"这一步往哪个方向偏"，范数告诉你"这一步到底走了多远"
（学习率与更新规则的乘积都在它里面）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from smart_research_agent.optimizers import compare as compare_mod
from smart_research_agent.optimizers import verify
from smart_research_agent.optimizers.optimizer import TrainingOptimizer
from smart_research_agent.optimizers.rules import clip_gradients, cosine_between
from smart_research_agent.optimizers.types import (
    OPTIMIZER_FORMULAS,
    OPTIMIZER_PROPERTIES,
    OPTIMIZER_STATE_KEYS,
    TRAIN_OPTIMIZERS,
    UPDATE_RULE_FAMILIES,
)

#: 一步表用的写死输入（参数、梯度、学习率）.
STEP_SAMPLE_PARAMS: tuple[float, ...] = (1.0, -2.0, 3.0)
STEP_SAMPLE_GRADS: tuple[float, ...] = (0.5, -1.0, 2.0)
STEP_SAMPLE_LR = 0.1

#: 裁剪表用的写死梯度与阈值（与 :mod:`verify` 同源）.
CLIP_SAMPLE_GRADS = verify.CLIP_SAMPLE_GRADS
CLIP_MAX_NORM = verify.CLIP_MAX_NORM

#: 调度表要看的那几步（热身段两个点 + 退火段三个点）.
SCHEDULE_STEPS: tuple[int, ...] = (1, 2, 5, 6, 13, 20)


def _norm(values: tuple[float, ...]) -> float:
    """向量的模长（范数表读数的口径）."""
    return math.sqrt(math.fsum(value * value for value in values))


@dataclass(frozen=True)
class RuleRow:
    """规则表的一行：名字 + 公式 + 状态 + 归属."""

    name: str
    formula: str
    state: str
    family: str

    def line(self) -> str:
        """``sgd        | shared | 状态 无                    | θ ← θ − lr·g``."""
        return f"{self.name:<10} | {self.family:<6} | 状态 {self.state:<22} | {self.formula}"


@dataclass(frozen=True)
class StepRow:
    """一步表的一行：首分量变化 + 步长范数."""

    name: str
    first_after: float
    step_norm: float

    def line(self) -> str:
        """``sgd        | θ₀ 1.000000 → 0.950000 | ‖一步‖ = 5.0e-02``."""
        return (
            f"{self.name:<10} | θ₀ {STEP_SAMPLE_PARAMS[0]:.6f} → {self.first_after:.6f} | "
            f"‖一步‖ = {self.step_norm:.3e}"
        )


@dataclass(frozen=True)
class ClipRow:
    """裁剪表的一行：范数 / 系数 / 方向余弦."""

    norm_before: float
    factor: float
    norm_after: float
    cosine: float

    def line(self) -> str:
        """``‖g‖ 1.119e+02 > 1 | 系数 8.9e-03 | ‖g'‖ 1.000e+00 | cos 1.000000``."""
        return (
            f"‖g‖ {self.norm_before:.3e} → ‖g'‖ {self.norm_after:.3e} | "
            f"系数 {self.factor:.3e} | cos {self.cosine:.6f}"
        )


@dataclass(frozen=True)
class ScheduleRow:
    """调度表的一行：步号 + 学习率."""

    step: int
    learning_rate: float

    def line(self) -> str:
        """``step  5 | lr 0.100000``."""
        return f"step {self.step:>2} | lr {self.learning_rate:.6f}"


@dataclass(frozen=True)
class PropertyRow:
    """性质表的一行：是否通过 + 现场读数 + 跨包对象."""

    name: str
    passed: bool
    reading: float
    cross_check: str

    def line(self) -> str:
        """``通过 shared_rules_match_day074 | 读数 0.000e+00 | a vs b``."""
        mark = "通过" if self.passed else "失败"
        return f"{mark} {self.name:<48} | 读数 {self.reading:.3e} | {self.cross_check}"


def rule_rows() -> tuple[RuleRow, ...]:
    """规则表：六个规则逐行（读数来自 ``types``）."""
    rows: list[RuleRow] = []
    for name in TRAIN_OPTIMIZERS:
        keys = OPTIMIZER_STATE_KEYS[name]
        state = "无" if not keys else " / ".join(keys)
        rows.append(
            RuleRow(
                name=name,
                formula=OPTIMIZER_FORMULAS[name],
                state=state,
                family=UPDATE_RULE_FAMILIES[name],
            )
        )
    return tuple(rows)


def one_step_rows() -> tuple[StepRow, ...]:
    """一步表：六个规则各走一步（读数来自 ``TrainingOptimizer.step``）."""
    rows: list[StepRow] = []
    for name in TRAIN_OPTIMIZERS:
        optimizer = TrainingOptimizer(name, STEP_SAMPLE_LR)
        updated = optimizer.step(STEP_SAMPLE_PARAMS, STEP_SAMPLE_GRADS)
        step = tuple(after - before for after, before in zip(updated, STEP_SAMPLE_PARAMS))
        rows.append(StepRow(name=name, first_after=updated[0], step_norm=_norm(step)))
    return tuple(rows)


def clip_row() -> ClipRow:
    """裁剪表：整体范数裁剪的一行读数（读数来自 ``rules.clip_gradients``）."""
    clipped, factor = clip_gradients(CLIP_SAMPLE_GRADS, CLIP_MAX_NORM)
    return ClipRow(
        norm_before=_norm(CLIP_SAMPLE_GRADS),
        factor=factor,
        norm_after=_norm(clipped),
        cosine=cosine_between(CLIP_SAMPLE_GRADS, clipped),
    )


def schedule_rows() -> tuple[ScheduleRow, ...]:
    """调度表：热身+余弦的几个点（读数来自接在优化器上的调度）."""
    optimizer = TrainingOptimizer("sgd", verify.SCHEDULE_BASE_LR)
    optimizer.set_schedule(
        "warmup_cosine",
        base_lr=verify.SCHEDULE_BASE_LR,
        warmup_steps=verify.SCHEDULE_WARMUP_STEPS,
        total_steps=verify.SCHEDULE_TOTAL_STEPS,
        min_lr=verify.SCHEDULE_MIN_LR,
    )
    assert optimizer.schedule is not None  # set_schedule 之后一定非空
    return tuple(ScheduleRow(step=step, learning_rate=optimizer.schedule(step)) for step in SCHEDULE_STEPS)


def compare_rows(objective_name: str) -> tuple[compare_mod.CompareRow, ...]:
    """对比表：六个优化器在给定目标函数上的读数（读数来自 ``compare``）."""
    return compare_mod.compare_optimizers(objective_name=objective_name)


def property_rows() -> tuple[PropertyRow, ...]:
    """性质表：七条性质逐行（读数来自 :func:`verify.check_all`）."""
    report = verify.check_all()
    rows: list[PropertyRow] = []
    for outcome in report.outcomes:
        check = outcome.check
        rows.append(
            PropertyRow(
                name=outcome.name,
                passed=outcome.passed,
                reading=check.reading,
                cross_check=f"{check.left} vs {check.right}",
            )
        )
    return tuple(rows)


def study_lines() -> tuple[str, ...]:
    """一次跑完六张表（演示脚本与教程引用的是同一批读数）."""
    lines: list[str] = []
    lines.append("== 1. 规则表（六个更新规则：公式 / 状态 / 归属）")
    for row in rule_rows():
        lines.append("  " + row.line())
    lines.append("== 2. 一步表（同一组写死输入各走一步）")
    for row in one_step_rows():
        lines.append("  " + row.line())
    lines.append("== 3. 裁剪表（整体范数裁剪：只改长度不改方向）")
    lines.append("  " + clip_row().line())
    lines.append("== 4. 调度表（热身 + 余弦：先升后降）")
    for row in schedule_rows():
        lines.append("  " + row.line())
    lines.append("== 5. 对比表（三个目标函数上六个优化器）")
    for objective_name in compare_mod.OBJECTIVES:
        lines.append(f"  -- {objective_name} --")
        for row in compare_rows(objective_name):
            lines.append("    " + row.line())
    lines.append("== 6. 性质表（七条性质：是否通过 / 读数 / 跨包对象）")
    for row in property_rows():
        lines.append("  " + row.line())
    return tuple(lines)


#: 本课的性质名单（供报告核对：表里的行数必须等于它）.
PROPERTY_NAMES = OPTIMIZER_PROPERTIES

__all__ = [
    "CLIP_MAX_NORM",
    "CLIP_SAMPLE_GRADS",
    "PROPERTY_NAMES",
    "SCHEDULE_STEPS",
    "STEP_SAMPLE_GRADS",
    "STEP_SAMPLE_LR",
    "STEP_SAMPLE_PARAMS",
    "ClipRow",
    "PropertyRow",
    "RuleRow",
    "ScheduleRow",
    "StepRow",
    "clip_row",
    "compare_rows",
    "one_step_rows",
    "property_rows",
    "rule_rows",
    "schedule_rows",
    "study_lines",
]
