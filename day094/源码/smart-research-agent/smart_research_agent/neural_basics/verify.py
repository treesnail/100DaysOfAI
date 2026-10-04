"""七条性质与两类判据（day089 / M8-D1）.

本模块是"从神经元到 FFN 这条链搭对了吗"的判据所在。七条性质分四类：

```text
有限性      六个激活在写死的网格（含 ±1000）上都有限
值域        激活的值域被遵守（relu≥0、sigmoid∈(0,1)、tanh∈(-1,1)、softmax 行和为 1）
跨天对账①   softmax 与 day073 的 math_foundations 逐位一致
结构事实    恒等激活的多层网络塌缩成一个仿射映射（逐点，容差 1e-12）
损失         mse 在预测等于目标时恰为 0.0，且等于逐元素平方差的均值
跨天对账②   交叉熵两条路径一致、CE ≥ 0，且与 sft.loss.cross_entropy 逐位一致
跨天对账③   ffn_block 与 encoder_decoder.layers.feed_forward 逐点一致
```

## 判据分两类（与 day087 / day088 同源）

```text
相等（== / 容差 / 逐位）   读数与期望一致（有限性、值域、逐位对账、整数计数）
上界（<=）                 读数不超过某个界（塌缩偏差 <= 1e-12、两条路径偏差 <= 1e-12）
```

因此 :class:`CrossCheck` 带 ``upper_bound`` 字段：有它时判据是"≤"，没有时才是"=="。
把两类混成一个判据，就会出现"偏差恰好是 0（因为输入全是 0）被当成通过"这种事。

## 三条跨天对账，分别对的是哪个真实函数

```text
softmax      activations.softmax                  ↔ math_foundations.linalg.softmax（day073）
交叉熵       losses.cross_entropy                 ↔ sft.loss.cross_entropy（生产实现，day050 起）
前馈块       network.ffn_block                    ↔ encoder_decoder.layers.feed_forward（day079）
附加         activations.gelu                     ↔ hf_source.blocks.gelu_exact（day085）
```

三条都调用**别的包**来算读数，因此它们能抓住"本包自己写错了一个数"——
自证是不成立的，跨包对账才是。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from smart_research_agent.neural_basics import activations, layers, losses, network
from smart_research_agent.neural_basics.errors import NeuralError, NumericError
from smart_research_agent.neural_basics.types import (
    ACTIVATIONS,
    ACT_GELU,
    ACT_RELU,
    NEURAL_PROPERTIES,
    PROPERTY_ACTIVATIONS_ARE_FINITE,
    PROPERTY_CROSS_ENTROPY_PATHS_AGREE,
    PROPERTY_FFN_MATCHES_TRANSFORMER_STACK,
    PROPERTY_IDENTITY_STACK_COLLAPSES,
    PROPERTY_MSE_IS_ZERO_AT_PERFECT,
    PROPERTY_RANGES_ARE_RESPECTED,
    PROPERTY_SOFTMAX_ROWS_ARE_DISTRIBUTIONS,
    DenseSpec,
    MLPSpec,
)

#: 恒等塌缩的容差（多层合成的浮点误差上界）.
IDENTITY_TOLERANCE = 1e-12

#: 交叉熵两条路径的容差（A 走 ``log_softmax`` 的 ``sum``、B 走 ``softmax`` 的 ``fsum``）.
PATH_TOLERANCE = 1e-12

#: softmax 行和与 1 的容差.
ROW_TOLERANCE = 1e-12

#: softmax 跨天对账用的行（**写死**）.
SOFTMAX_CASES: tuple[tuple[float, ...], ...] = (
    (1.0, 2.0, 3.0),
    (-5.0, 0.0, 5.0),
    (1000.0, 1001.0),
    (0.0,),
)

#: 交叉熵的用例（**写死**）：logits 与 target.
CE_CASES: tuple[tuple[tuple[float, ...], int], ...] = (
    ((1.0, 2.0, 3.0), 0),
    ((3.0, 2.0, 1.0), 2),
    ((0.5,), 0),
    ((10.0, -10.0, 0.0), 1),
)

#: 前馈对账的输入（**写死**）：两行、每行 4 个特征.
FFN_INPUTS: tuple[tuple[float, ...], ...] = (
    (0.5, -1.0, 2.0, 0.25),
    (1.5, 0.0, -0.5, 3.0),
)

#: 前馈对账的隐藏维与种子（``d_ff = 4·hidden = 16``）.
FFN_HIDDEN = 4
FFN_SEED = 7

#: mse 的对账样本（**写死**）：预测 / 目标（逐元素平方差的均值 = 0.625）.
MSE_PRED: tuple[tuple[float, ...], ...] = ((1.0, 2.0), (3.0, 4.0))
MSE_TARGET: tuple[tuple[float, ...], ...] = ((1.5, 1.5), (4.0, 3.0))

#: 恒等塌缩实验的网络：``3 → 4 → 5 → 2``（**三层、全恒等**）.
IDENTITY_SPEC = MLPSpec(
    layers=(
        DenseSpec(3, 4, activation=None, init="xavier", seed=11),
        DenseSpec(4, 5, activation=None, init="xavier", seed=12),
        DenseSpec(5, 2, activation=None, init="xavier", seed=13),
    )
)

#: 恒等塌缩实验的输入（**写死**）：两行、每行 3 个特征.
IDENTITY_INPUTS: tuple[tuple[float, ...], ...] = ((0.3, -0.7, 1.1), (2.0, 0.5, -1.5))


@dataclass(frozen=True)
class CrossCheck:
    """两个来源之间的一次对账：来源、读数、判据（**含方向与上界**）.

    与 day087 / day088 的同类逐字同源：有 ``upper_bound`` 的时候判据是"实测 <= 上界"，
    没有的时候是"两个数相等"。``exact=True`` 要求逐位/整数相等，否则走容差。
    """

    name: str
    left: str
    right: str
    reading: float
    expected: float
    exact: bool = True
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.reading) or not math.isfinite(self.expected):
            raise NumericError(
                f"对账读数必须有限：reading={self.reading!r}、expected={self.expected!r}。"
            )
        if self.upper_bound is not None and (
            not math.isfinite(self.upper_bound) or self.upper_bound < 0
        ):
            raise NumericError(f"上界必须是有限非负数，收到 {self.upper_bound!r}。")

    @property
    def passed(self) -> bool:
        """相等（逐位 / 整数 / 容差）或"不超过上界"两种判据."""
        if self.upper_bound is not None:
            return float(self.reading) <= float(self.upper_bound) + 1e-12
        if self.exact:
            return self.reading == self.expected
        return abs(float(self.reading) - float(self.expected)) <= 1e-12

    def line(self) -> str:
        """一行可读的读数（带上两个来源与那个上界）."""
        verdict = "满足" if self.passed else "不满足"
        if self.upper_bound is not None:
            return (
                f"[{verdict}] {self.name}: 读数 {self.reading:.6e} ≤ 上界 "
                f"{self.upper_bound:.6e}（{self.left} vs {self.right}）"
            )
        return (
            f"[{verdict}] {self.name}: {self.left} vs {self.right} | "
            f"读数 {self.reading} / 期望 {self.expected}"
        )


@dataclass(frozen=True)
class PropertyOutcome:
    """一条性质的结论：适用性 + 结果 + 证据行."""

    name: str
    applicable: bool
    passed: bool
    evidence: tuple[str, ...] = field(default_factory=tuple)
    cross_check: CrossCheck | None = None

    def __post_init__(self) -> None:
        if not self.applicable and self.passed:
            raise NumericError(
                f"性质 {self.name!r} 标了'不适用'却又标了'通过'：两者必须分开——"
                "否则'把这条检查删掉'与'它通过了'在报告里长得一模一样。"
            )

    def line(self) -> str:
        """一行可读的结论（不适用也要印出来）."""
        if not self.applicable:
            return f"[不适用] {self.name} | {'；'.join(self.evidence)}"
        verdict = "通过" if self.passed else "失败"
        detail = "；".join(self.evidence)
        suffix = f" | {detail}" if detail else ""
        return f"[{verdict}] {self.name}{suffix}"


@dataclass(frozen=True)
class PropertyReport:
    """一组性质的报告（``ok`` 要求**所有适用**的都通过）."""

    outcomes: tuple[PropertyOutcome, ...]

    @property
    def applicable(self) -> tuple[PropertyOutcome, ...]:
        """适用（``applicable=True``）的那些性质."""
        return tuple(outcome for outcome in self.outcomes if outcome.applicable)

    @property
    def ok(self) -> bool:
        """是否全部通过——不适用不算通过、也不算失败."""
        return all(outcome.passed for outcome in self.applicable)

    def require_ok(self) -> None:
        """不通过时抛 :class:`errors.NeuralError`."""
        if self.ok:
            return
        failures = [outcome.line() for outcome in self.applicable if not outcome.passed]
        raise NeuralError("性质检查未全部通过：" + "；".join(failures))

    def lines(self) -> tuple[str, ...]:
        """逐行文本（**先印不适用**，让"这一次没查它"一眼可见）."""
        return tuple(
            outcome.line()
            for outcome in sorted(self.outcomes, key=lambda outcome: outcome.applicable)
        )

    def to_dict(self) -> dict[str, object]:
        """摊平成可 JSON 化的字段."""
        return {
            "ok": self.ok,
            "counts": {"total": len(self.outcomes), "applicable": len(self.applicable)},
            "lines": list(self.lines()),
        }


def check_activations_are_finite() -> PropertyOutcome:
    """六个激活在 :data:`activations.FINITE_GRID`（含 ``±1000``）上都有限."""
    non_finite = 0
    details: list[str] = []
    for name in ACTIVATIONS:
        output = activations.activate(name, activations.FINITE_GRID)
        bad = sum(int(not math.isfinite(value)) for value in output)
        non_finite += bad
        details.append(f"{name}={bad}")
    check = CrossCheck(
        name="网格上的非有限个数",
        left="六个激活 × FINITE_GRID",
        right="全部有限",
        reading=float(non_finite),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_ACTIVATIONS_ARE_FINITE,
        applicable=True,
        passed=non_finite == 0,
        evidence=(
            f"网格 {len(activations.FINITE_GRID)} 点（含 ±1000）、非有限个数 {non_finite}",
            "逐激活非有限个数：" + "、".join(details),
        ),
        cross_check=check,
    )


def check_ranges_are_respected() -> PropertyOutcome:
    """激活的值域被遵守；softmax 每一行和为 1（在 :data:`activations.RANGE_GRID` 上）."""
    violations = 0
    details: list[str] = []
    for name in ACTIVATIONS:
        output = activations.activate(name, activations.RANGE_GRID)
        bound = activations.activation_range(name)
        bad = sum(int(not bound.contains(value)) for value in output)
        violations += bad
        details.append(f"{name}({bound.line()})={bad}")
    distribution = activations.softmax(activations.RANGE_GRID)
    row_deviation = abs(math.fsum(distribution) - 1.0)
    if row_deviation > ROW_TOLERANCE:
        violations += 1
    check = CrossCheck(
        name="值域越界个数",
        left="六个激活 × RANGE_GRID",
        right="各自的值域",
        reading=float(violations),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_RANGES_ARE_RESPECTED,
        applicable=True,
        passed=violations == 0,
        evidence=(
            f"越界个数 {violations}；softmax 行和偏差 {row_deviation:.3e}（容差 {ROW_TOLERANCE:.0e}）",
            "逐激活越界：" + "、".join(details),
        ),
        cross_check=check,
    )


def check_softmax_rows_are_distributions() -> PropertyOutcome:
    """**跨天对账**：与 ``math_foundations.linalg.softmax`` 在同一输入上**逐位**一致."""
    from smart_research_agent.math_foundations.linalg import softmax as reference_softmax

    mismatches = 0
    worst = 0.0
    for logits in SOFTMAX_CASES:
        mine = activations.softmax(logits)
        theirs = tuple(reference_softmax(list(logits)))
        for left, right in zip(mine, theirs, strict=True):
            mismatches += int(left != right)
            worst = max(worst, abs(left - right))
    check = CrossCheck(
        name="softmax 逐位不一致的分量个数",
        left="activations.softmax",
        right="math_foundations.linalg.softmax",
        reading=float(mismatches),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_SOFTMAX_ROWS_ARE_DISTRIBUTIONS,
        applicable=True,
        passed=mismatches == 0,
        evidence=(
            f"{len(SOFTMAX_CASES)} 行、逐位不一致 {mismatches} 个、最大偏差 {worst:.3e}",
            "两处都先减最大值、都用 math.fsum",
        ),
        cross_check=check,
    )


def check_identity_stack_collapses() -> PropertyOutcome:
    """恒等激活的多层网络 == 合成后的单层仿射映射（逐点，容差 1e-12）."""
    spec = IDENTITY_SPEC
    multi = network.mlp_forward(spec, IDENTITY_INPUTS)
    weight, bias = network.collapse_identity_mlp(spec)
    single = layers.affine_forward(weight, bias, IDENTITY_INPUTS)
    worst = 0.0
    for multi_row, single_row in zip(multi, single, strict=True):
        for left, right in zip(multi_row, single_row, strict=True):
            worst = max(worst, abs(left - right))
    check = CrossCheck(
        name="多层输出与合成单层的最大偏差",
        left=f"mlp_forward({len(spec.layers)} 层恒等)",
        right="affine_forward(compose_affine(...))",
        reading=worst,
        expected=0.0,
        exact=False,
        upper_bound=IDENTITY_TOLERANCE,
    )
    return PropertyOutcome(
        name=PROPERTY_IDENTITY_STACK_COLLAPSES,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"宽度链 {'→'.join(str(width) for width in spec.widths)}、最大偏差 {worst:.3e}",
            f"判据：偏差 <= {IDENTITY_TOLERANCE:.0e}",
        ),
        cross_check=check,
    )


def check_mse_is_zero_at_perfect() -> PropertyOutcome:
    """``mse`` 在预测等于目标时恰为 0.0，且等于逐元素平方差的均值（与手算对照）."""
    perfect = losses.mse(MSE_PRED, MSE_PRED)
    measured = losses.mse(MSE_PRED, MSE_TARGET)
    hand = math.fsum(
        (pred - target) ** 2
        for pred_row, target_row in zip(MSE_PRED, MSE_TARGET, strict=True)
        for pred, target in zip(pred_row, target_row, strict=True)
    ) / sum(len(row) for row in MSE_PRED)
    deviation = abs(measured - hand)
    check = CrossCheck(
        name="预测等于目标时的 mse",
        left="mse(P, P)",
        right="0.0",
        reading=perfect,
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_MSE_IS_ZERO_AT_PERFECT,
        applicable=True,
        passed=(perfect == 0.0) and (deviation <= 1e-12),
        evidence=(
            f"mse(P, P) = {perfect}（恰好 0）",
            f"mse(P, T) = {measured:.6e}，手算 {hand:.6e}，偏差 {deviation:.3e}",
        ),
        cross_check=check,
    )


def check_cross_entropy_paths_agree() -> PropertyOutcome:
    """**跨天对账**：CE 两条路径一致且 CE ≥ 0，并与 ``sft.loss.cross_entropy`` 逐位一致."""
    from smart_research_agent.sft.loss import cross_entropy as reference_cross_entropy

    worst_path = 0.0
    negatives = 0
    mismatches = 0
    for logits, target in CE_CASES:
        stable = losses.cross_entropy(logits, target)
        naive = losses.cross_entropy_via_probability(logits, target)
        worst_path = max(worst_path, abs(stable - naive))
        negatives += int(stable < 0.0)
        mismatches += int(stable != reference_cross_entropy(list(logits), target))
    check = CrossCheck(
        name="CE 与 sft.loss 逐位不一致的个数",
        left="losses.cross_entropy",
        right="sft.loss.cross_entropy",
        reading=float(mismatches),
        expected=0.0,
        exact=True,
    )
    passed = (mismatches == 0) and (negatives == 0) and (worst_path <= PATH_TOLERANCE)
    return PropertyOutcome(
        name=PROPERTY_CROSS_ENTROPY_PATHS_AGREE,
        applicable=True,
        passed=passed,
        evidence=(
            f"{len(CE_CASES)} 个用例：与 sft.loss 逐位不一致 {mismatches} 个、CE<0 的 {negatives} 个",
            f"两条路径最大偏差 {worst_path:.3e}（容差 {PATH_TOLERANCE:.0e}：一条走 sum、一条走 fsum）",
        ),
        cross_check=check,
    )


def check_ffn_matches_transformer_stack() -> PropertyOutcome:
    """**跨天对账**：``network.ffn_block`` 与 ``encoder_decoder.layers.feed_forward`` 逐点一致.

    顺带把 ``activations.gelu`` 与 ``hf_source.blocks.gelu_exact`` 也在同一网格上比一遍。
    """
    from smart_research_agent.encoder_decoder.layers import feed_forward
    from smart_research_agent.encoder_decoder.types import (
        ACTIVATION_GELU,
        ACTIVATION_RELU,
        FFNWeights,
    )
    from smart_research_agent.hf_source.blocks import gelu_exact

    params = network.build_ffn_params(hidden=FFN_HIDDEN, seed=FFN_SEED)
    real_weights = FFNWeights(
        w_in=params.w_in, b_in=params.b_in, w_out=params.w_out, b_out=params.b_out
    )
    mismatches = 0
    for activation, day079_name in ((ACT_RELU, ACTIVATION_RELU), (ACT_GELU, ACTIVATION_GELU)):
        mine = network.ffn_block(FFN_INPUTS, real_weights, activation=activation)
        theirs, _cache = feed_forward(FFN_INPUTS, real_weights, activation=day079_name)
        for mine_row, their_row in zip(mine, theirs, strict=True):
            for left, right in zip(mine_row, their_row, strict=True):
                mismatches += int(left != right)
    gelu_mismatches = sum(
        int(activations.gelu(value) != gelu_exact(value)) for value in activations.FINITE_GRID
    )
    check = CrossCheck(
        name="FFN 逐位不一致的元素个数",
        left="network.ffn_block",
        right="encoder_decoder.layers.feed_forward",
        reading=float(mismatches),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=PROPERTY_FFN_MATCHES_TRANSFORMER_STACK,
        applicable=True,
        passed=(mismatches == 0) and (gelu_mismatches == 0),
        evidence=(
            f"hidden={FFN_HIDDEN}、d_ff={network.FFN_RATIO * FFN_HIDDEN}、2 种激活、逐位不一致 {mismatches} 个",
            f"gelu vs hf_source.blocks.gelu_exact：{len(activations.FINITE_GRID)} 点上不一致 {gelu_mismatches} 个",
        ),
        cross_check=check,
    )


def check_all() -> PropertyReport:
    """一次跑完七条性质（顺序与 :data:`types.NEURAL_PROPERTIES` 一致）."""
    outcomes = (
        check_activations_are_finite(),
        check_ranges_are_respected(),
        check_softmax_rows_are_distributions(),
        check_identity_stack_collapses(),
        check_mse_is_zero_at_perfect(),
        check_cross_entropy_paths_agree(),
        check_ffn_matches_transformer_stack(),
    )
    names = {outcome.name for outcome in outcomes}
    extra = names - set(NEURAL_PROPERTIES)
    missing = set(NEURAL_PROPERTIES) - names
    if extra or missing:  # pragma: no cover - 只在有人改性质名单时触发
        raise NeuralError(
            "性质名单与 types.NEURAL_PROPERTIES 不一致："
            f"多 {sorted(extra)}、缺 {sorted(missing)}。"
            "名单对不上时，报告里那七行会安静地少一行或多一行。"
        )
    return PropertyReport(outcomes=outcomes)


__all__ = [
    "CE_CASES",
    "FFN_HIDDEN",
    "FFN_INPUTS",
    "FFN_SEED",
    "IDENTITY_INPUTS",
    "IDENTITY_SPEC",
    "IDENTITY_TOLERANCE",
    "MSE_PRED",
    "MSE_TARGET",
    "PATH_TOLERANCE",
    "ROW_TOLERANCE",
    "SOFTMAX_CASES",
    "CrossCheck",
    "PropertyOutcome",
    "PropertyReport",
    "check_activations_are_finite",
    "check_all",
    "check_cross_entropy_paths_agree",
    "check_ffn_matches_transformer_stack",
    "check_identity_stack_collapses",
    "check_mse_is_zero_at_perfect",
    "check_ranges_are_respected",
    "check_softmax_rows_are_distributions",
]
