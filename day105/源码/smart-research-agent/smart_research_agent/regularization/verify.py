"""``verify``：七条性质与三类判据（day095 / M8-D6）.

```text
相等（逐位 / 整数）  ① batch_norm_forward 与手写逐特征标准化逐位一致
                     ② 把批转置之后，BatchNorm 与 day079 的 LayerNorm 给出同一个矩阵
                     ⑥ dropout 的推理相与 rate=0 的训练相逐位相同
                     ⑦ sparkline 的长度与极值位置都与序列一致
上界                 ④ 训练相反向（三项）与数值差分一致（含整条链）
                     ⑤ 推理相反向（一项）与数值差分一致（含整条链）
下界                 ③ 同一个 batch 上，训练相与推理相的输出**必须不同**
```

判据三类与 day092 / day094 同源：**相等**、**不超过上界**、**不超过下界**。

## 为什么第 ③ 条必须是下界

因为"两相"这件事的可证伪形式是"它们**不一样**"：

```text
把 BatchNorm 的两相写成同一件事（都用本批统计量）
  ⇒ ① 仍然通过（训练相本来就那样）
  ⇒ ② 仍然通过（LayerNorm 等价性只依赖训练相）
  ⇒ ④ 仍然通过（训练相的反向照样对）
  ⇒ ⑤ **会红**（推理相的反向被写成了三项版本）
  ⇒ ③ 也会红（两相给出了同一个输出）
⇒ 两条下界/路径判据一起把"两相被写成一相"钉死
```

## 第 ② 条的容差为什么是 1e-12 而不是"逐位"

两条路的求和顺序不同（本包用 ``math.fsum``，day079 的 LayerNorm 用它自己的写法），
因此末位可能差一个 ULP。实测 **0.000e+00**，但判据写成"不超过 1e-12"——
这是一条**上界**判据，它承认浮点求值顺序的自由，只要求两条路给出同一个数。

## 第 ④ / ⑤ 条的容差为什么是 1e-7（比 day093 / day094 松两档）

这里的被检验对象是**一条四级的链**（BPTT → BN → dropout → dense）在**一批样本**上
的梯度，梯度量级 ~14；中心差分的绝对误差随量级放大，
实测**相对误差约 3.6e-10**（与 day093 / day094 的 1e-10 同量级）。
把绝对容差写成 1e-9 会让这条判据在"量级变大"时无辜变红——
而"容差低于分辨率"是 day074 已经写下来的一种失败。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from smart_research_agent.math_foundations.calculus import gradient
from smart_research_agent.regularization import network as reg_network
from smart_research_agent.regularization.errors import RegularizationError
from smart_research_agent.regularization.normalization import (
    batch_norm_backward,
    batch_norm_forward,
    RunningStatistics,
    transpose,
)
from smart_research_agent.regularization.techniques import dropout_vector
from smart_research_agent.regularization.types import (
    PHASE_EVAL,
    PHASE_TRAIN,
    PROPERTY_BATCHNORM_BACKWARD_MATCHES_NUMERICAL,
    PROPERTY_BATCHNORM_EQUALS_TRANSPOSED_LAYERNORM,
    PROPERTY_BATCHNORM_MATCHES_MANUAL,
    PROPERTY_DESCRIPTIONS,
    PROPERTY_DROPOUT_EVAL_MATCHES_RATE_ZERO,
    PROPERTY_EVAL_BACKWARD_MATCHES_NUMERICAL,
    PROPERTY_SPARKLINE_MATCHES_EXTREMES,
    PROPERTY_TWO_PHASES_DIFFER,
    REGULARIZATION_PROPERTIES,
)
from smart_research_agent.regularization.visualize import GLYPHS, sparkline
from smart_research_agent.sequence_models.train import make_sign_dataset

#: 第 ①②③④⑤ 条用的写死样本（4 条样本 × 3 个特征）.
SAMPLE_BATCH: tuple[tuple[float, ...], ...] = (
    (0.5, -1.0, 0.25),
    (1.0, 0.0, -0.5),
    (-0.25, 0.75, 0.5),
    (0.75, -0.5, 1.0),
)
SAMPLE_GAMMA: tuple[float, ...] = (1.3, 0.7, -1.1)
SAMPLE_BETA: tuple[float, ...] = (0.2, -0.4, 0.9)
SAMPLE_SEED_GRAD: tuple[tuple[float, ...], ...] = (
    (1.0, -0.5, 0.25),
    (0.25, 2.0, -1.0),
    (-1.0, 0.5, 0.75),
    (0.5, 1.0, -0.25),
)

#: 第 ② 条用的容差（上界：两条路的求和顺序不同，末位可以差一个 ULP）.
LAYERNORM_TOLERANCE = 1e-12

#: 第 ③ 条用的下界（"两相必须不同"）。
PHASE_GAP_LOWER_BOUND = 0.5

#: 第 ④ ⑤ 条用的容差（**四级链 + 一批样本**，见模块文档串）。
GRAD_TOLERANCE = 1e-7

#: 第 ④ ⑤ 条用的网络与批（小一点，好让数值差分便宜）。
CHAIN_SEED = 21
CHAIN_HIDDEN = 3
CHAIN_BATCH_SIZE = 6
CHAIN_DATASET_LENGTH = 3


@dataclass(frozen=True)
class Check:
    """一次性质校验的读数与判据（三类：相等 / 上界 / 下界）."""

    reading: float
    upper_bound: float | None = None
    lower_bound: float | None = None
    left: str = "-"
    right: str = "-"

    def passed(self) -> bool:
        """读数是否落在判据内（上界与下界同时生效）."""
        if self.upper_bound is not None and self.reading > self.upper_bound:
            return False
        return not (self.lower_bound is not None and self.reading < self.lower_bound)

    def bound_text(self) -> str:
        """判据的一行文本."""
        if self.lower_bound is not None and self.upper_bound is not None:
            return f"∈ [{self.lower_bound:.1e}, {self.upper_bound:.1e}]"
        if self.lower_bound is not None:
            return f">= {self.lower_bound:.1e}"
        if self.upper_bound is None:
            return "== 逐位"
        if self.upper_bound == 0.0:
            return "== 0"
        return f"<= {self.upper_bound:.1e}"


@dataclass(frozen=True)
class PropertyOutcome:
    """一条性质的结论：是否通过 + 现场读数 + 两个来源."""

    name: str
    passed: bool
    check: Check

    def line(self) -> str:
        """``通过 batchnorm_matches_manual_standardization | 读数 0.000e+00 == 0 | a vs b``."""
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
# 单条性质
# --------------------------------------------------------------------------- #


def _manual_standardize(
    batch: tuple[tuple[float, ...], ...], *, gamma, beta
) -> tuple[tuple[float, ...], ...]:
    """**独立于 normalization.batch_norm_forward** 的第二条实现：显式逐特征标准化.

    它刻意不走 ``batch_statistics`` / ``as_matrix``：一条性质要成立，
    "两条路径独立"这件事必须是真的。共享的只有"求和用 ``math.fsum``"这条数值约定
    与 ``ε = 1e-5`` 这个常量。
    """
    rows = len(batch)
    columns = len(batch[0])
    means = [
        math.fsum(batch[row][column] for row in range(rows)) / rows for column in range(columns)
    ]
    variances = [
        math.fsum((batch[row][column] - means[column]) ** 2 for row in range(rows)) / rows
        for column in range(columns)
    ]
    sigmas = [math.sqrt(value + 1e-5) for value in variances]
    return tuple(
        tuple(
            (batch[row][column] - means[column]) / sigmas[column] * gamma[column] + beta[column]
            for column in range(columns)
        )
        for row in range(rows)
    )


def check_batchnorm_matches_manual() -> PropertyOutcome:
    """① ``batch_norm_forward`` 与手写的逐特征标准化逐位一致（读数 = 最大绝对差）."""
    produced, _cache, _running = batch_norm_forward(
        SAMPLE_BATCH, gamma=SAMPLE_GAMMA, beta=SAMPLE_BETA
    )
    manual = _manual_standardize(SAMPLE_BATCH, gamma=SAMPLE_GAMMA, beta=SAMPLE_BETA)
    gap = max(
        abs(a - b) for row_a, row_b in zip(produced, manual, strict=True) for a, b in zip(row_a, row_b, strict=True)
    )
    check = Check(reading=gap, upper_bound=0.0, left="normalization.batch_norm_forward", right="手写逐特征标准化")
    return PropertyOutcome(PROPERTY_BATCHNORM_MATCHES_MANUAL, check.passed(), check)


def check_batchnorm_equals_transposed_layernorm() -> PropertyOutcome:
    """② 把批转置之后，BatchNorm 与 day079 的 LayerNorm 给出同一个矩阵.

    注意 γ/β 用的是"什么都不改"的那一组（``γ = 1``、``β = 0``）：
    **LayerNorm 的 γ 是逐列的、BatchNorm 的 γ 是逐特征的**，
    两套仿射参数不在同一条轴上——因此这条等式只在"没有仿射"时精确成立，
    而"仿射参数在哪条轴上"本身就是这一课要讲清的一件事。
    """
    from smart_research_agent.encoder_decoder.layers import layer_norm

    produced, _cache, _running = batch_norm_forward(SAMPLE_BATCH)
    rotated, _reference_cache = layer_norm(transpose(SAMPLE_BATCH))
    back = transpose(rotated)
    gap = max(
        abs(a - b)
        for row_a, row_b in zip(produced, back, strict=True)
        for a, b in zip(row_a, row_b, strict=True)
    )
    check = Check(
        reading=gap,
        upper_bound=LAYERNORM_TOLERANCE,
        left="normalization.batch_norm_forward",
        right="transpose(encoder_decoder.layer_norm(transpose(X)))",
    )
    return PropertyOutcome(PROPERTY_BATCHNORM_EQUALS_TRANSPOSED_LAYERNORM, check.passed(), check)


def check_two_phases_differ() -> PropertyOutcome:
    """③ 同一个 batch 上训练相与推理相的输出**必须不同**（读数 = 最大差，下界 0.5）."""
    training, _cache, running = batch_norm_forward(
        SAMPLE_BATCH, gamma=SAMPLE_GAMMA, beta=SAMPLE_BETA, phase=PHASE_TRAIN
    )
    inference, _inference_cache, _same = batch_norm_forward(
        SAMPLE_BATCH,
        gamma=SAMPLE_GAMMA,
        beta=SAMPLE_BETA,
        phase=PHASE_EVAL,
        running=running,
    )
    gap = max(
        abs(a - b)
        for row_a, row_b in zip(training, inference, strict=True)
        for a, b in zip(row_a, row_b, strict=True)
    )
    check = Check(
        reading=gap,
        lower_bound=PHASE_GAP_LOWER_BOUND,
        left="训练相输出",
        right="推理相输出（同一批、同一组 γ/β）",
    )
    return PropertyOutcome(PROPERTY_TWO_PHASES_DIFFER, check.passed(), check)


def _chain_gradient_check(phase: str) -> PropertyOutcome:
    """④ / ⑤ 共用的实现：整条链在给定相下的解析梯度 vs 数值差分."""
    dataset = make_sign_dataset(CHAIN_DATASET_LENGTH)
    batch = dataset[:CHAIN_BATCH_SIZE]
    params = reg_network.build_regularized(
        "lstm",
        input_size=1,
        hidden_size=CHAIN_HIDDEN,
        classes=2,
        seed=CHAIN_SEED,
    )
    rows = len(batch)
    columns = params.classes
    seed_grad = tuple(
        tuple(1.0 if (row + column) % 2 else -0.5 for column in range(columns))
        for row in range(rows)
    )
    use_norm = True
    running = RunningStatistics(
        mean=(0.1, -0.2, 0.3), variance=(0.8, 1.2, 0.5), batches=3
    )
    _logits, cache = reg_network.regularized_forward_cached(
        params,
        batch,
        phase=phase,
        running=running,
        dropout_rate=0.3,
        dropout_seed=5,
        use_norm=use_norm,
    )
    analytic = reg_network.flatten_gradients(
        reg_network.regularized_backward(params, cache, seed_grad)
    )

    def objective(flat: tuple[float, ...]) -> float:
        probe = reg_network.unflatten_params(flat, template=params)
        out, _probe_cache = reg_network.regularized_forward_cached(
            probe,
            batch,
            phase=phase,
            running=running,
            dropout_rate=0.3,
            dropout_seed=5,
            use_norm=use_norm,
        )
        return math.fsum(
            out[row][column] * seed_grad[row][column]
            for row in range(len(out))
            for column in range(len(out[0]))
        )

    numeric = gradient(objective, reg_network.flatten_params(params))
    worst = max(abs(a - b) for a, b in zip(analytic, numeric, strict=True))
    norm = math.sqrt(math.fsum(value * value for value in analytic))
    label = "训练相" if phase == PHASE_TRAIN else "推理相"
    check = Check(
        reading=worst,
        upper_bound=GRAD_TOLERANCE,
        left=f"整条链的解析梯度（{label}）",
        right=f"calculus.gradient（数值；‖∇‖ ≈ {norm:.3f}）",
    )
    return PropertyOutcome(
        PROPERTY_BATCHNORM_BACKWARD_MATCHES_NUMERICAL
        if phase == PHASE_TRAIN
        else PROPERTY_EVAL_BACKWARD_MATCHES_NUMERICAL,
        check.passed(),
        check,
    )


def check_batchnorm_backward_matches_numerical() -> PropertyOutcome:
    """④ 训练相（三项公式）的解析梯度与数值差分一致."""
    return _chain_gradient_check(PHASE_TRAIN)


def check_eval_backward_matches_numerical() -> PropertyOutcome:
    """⑤ 推理相（一项公式）的解析梯度与数值差分一致."""
    return _chain_gradient_check(PHASE_EVAL)


def check_dropout_eval_matches_rate_zero() -> PropertyOutcome:
    """⑥ dropout 的推理相与 ``rate=0`` 的训练相逐位相同（转发 day081 的那条不变量）."""
    sample = (1.0, -2.0, 0.5, 3.0)
    evaluation, _mask, scale = dropout_vector(sample, rate=0.5, seed=7, phase=PHASE_EVAL)
    training_zero, _zero_mask, _zero_scale = dropout_vector(
        sample, rate=0.0, seed=7, phase=PHASE_TRAIN
    )
    gap = max(abs(a - b) for a, b in zip(evaluation, training_zero, strict=True))
    extra = abs(scale - 1.0)
    check = Check(
        reading=gap + extra,
        upper_bound=0.0,
        left="dropout(rate=0.5, eval)",
        right="dropout(rate=0.0, train)",
    )
    return PropertyOutcome(PROPERTY_DROPOUT_EVAL_MATCHES_RATE_ZERO, check.passed(), check)


def check_sparkline_matches_extremes() -> PropertyOutcome:
    """⑦ sparkline 的长度与极值位置都与序列一致（读数 = 不一致的项数，整数）."""
    values = (0.9, 0.7, 0.5, 0.3, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95, 0.05, 0.55)
    drawn = sparkline(values, width=len(values))
    lowest_index = min(range(len(values)), key=lambda index: values[index])
    highest_index = max(range(len(values)), key=lambda index: values[index])
    mismatches = abs(len(drawn) - len(values))
    mismatches += 0 if drawn[lowest_index] == GLYPHS[0] else 1
    mismatches += 0 if drawn[highest_index] == GLYPHS[-1] else 1
    check = Check(
        reading=float(mismatches),
        upper_bound=0.0,
        left=f"sparkline（{len(drawn)} 个字符）",
        right=f"argmin = {lowest_index} / argmax = {highest_index}",
    )
    return PropertyOutcome(PROPERTY_SPARKLINE_MATCHES_EXTREMES, check.passed(), check)


#: 七条性质的名字 -> 检查函数（键顺序 = :data:`types.REGULARIZATION_PROPERTIES`）.
CHECKS = {
    PROPERTY_BATCHNORM_MATCHES_MANUAL: check_batchnorm_matches_manual,
    PROPERTY_BATCHNORM_EQUALS_TRANSPOSED_LAYERNORM: check_batchnorm_equals_transposed_layernorm,
    PROPERTY_TWO_PHASES_DIFFER: check_two_phases_differ,
    PROPERTY_BATCHNORM_BACKWARD_MATCHES_NUMERICAL: check_batchnorm_backward_matches_numerical,
    PROPERTY_EVAL_BACKWARD_MATCHES_NUMERICAL: check_eval_backward_matches_numerical,
    PROPERTY_DROPOUT_EVAL_MATCHES_RATE_ZERO: check_dropout_eval_matches_rate_zero,
    PROPERTY_SPARKLINE_MATCHES_EXTREMES: check_sparkline_matches_extremes,
}

if set(CHECKS) != set(REGULARIZATION_PROPERTIES):  # pragma: no cover - 导入期不变式
    raise RegularizationError(
        "性质名单与检查函数表不一致：少一条的性质会静默地不在报告里出现，"
        "而'少一条'与'它通过了'在读报告时长得一样。"
    )

if set(PROPERTY_DESCRIPTIONS) != set(REGULARIZATION_PROPERTIES):  # pragma: no cover
    raise RegularizationError("性质说明表与名单不一致。")


def check_all() -> PropertyReport:
    """跑完七条性质，返回汇总报告（顺序与 :data:`types.REGULARIZATION_PROPERTIES` 一致）."""
    report = PropertyReport()
    for name in REGULARIZATION_PROPERTIES:
        report.outcomes.append(CHECKS[name]())
    return report


__all__ = [
    "CHAIN_BATCH_SIZE",
    "CHAIN_HIDDEN",
    "CHAIN_SEED",
    "CHECKS",
    "GRAD_TOLERANCE",
    "LAYERNORM_TOLERANCE",
    "PHASE_GAP_LOWER_BOUND",
    "SAMPLE_BATCH",
    "SAMPLE_BETA",
    "SAMPLE_GAMMA",
    "SAMPLE_SEED_GRAD",
    "Check",
    "PropertyOutcome",
    "PropertyReport",
    "check_all",
    "check_batchnorm_backward_matches_numerical",
    "check_batchnorm_equals_transposed_layernorm",
    "check_batchnorm_matches_manual",
    "check_dropout_eval_matches_rate_zero",
    "check_eval_backward_matches_numerical",
    "check_sparkline_matches_extremes",
    "check_two_phases_differ",
]
