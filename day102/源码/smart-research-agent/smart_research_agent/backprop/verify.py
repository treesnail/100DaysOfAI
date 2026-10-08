"""七条性质与两类判据（day090 / M8-D2）.

本模块是"这一次反向算对了吗"的判据所在。七条性质分四类：

```text
逐点导数    五个逐元素激活的解析导数 vs 数值差分
一个约定    relu 在 x = 0 处取次梯度 0（"关掉的神经元恰好拿到 0"）
跨天对账①   softmax 雅可比 vs day074 的数值雅可比（并读 day074 的结论）
恒等式      (Jᵀv)_i = p_i(v_i − ⟨p,v⟩) 与显式矩阵乘一致
跨天对账②   交叉熵梯度 = p − onehot，vs day074 的数值差分（并读它的结论）
整条网络    整个 MLP 的解析梯度（对权重/偏置/输入）vs 数值差分
跨天对账③   ffn_backward 与 encoder_decoder.layers.feed_forward_backward **逐位**一致
```

## 判据分两类（与 day087~089 同源）

```text
相等（== / 逐位 / 整数计数）   读数与期望一致
上界（<=）                    读数不超过某个界（相对误差、最大偏差）
```

因此 :class:`GradientCheck` 带 ``upper_bound`` 字段：有它时判据是"≤"，没有时才是"=="。
把两类混成一个判据，就会出现"偏差恰好是 0（因为输入全是 0）被当成通过"这种事。

## 为什么第 3、5 条要"两条腿走路"

第 3、5 条各做两件事：

```text
一、本包自己的解析式 vs 数值差分        （自证：证明我们这份实现没抄错）
二、读 day074 的 gradcheck 的结论       （跨包：证明它与项目里**生产口径**一致）
```

只有第一条时，"本包与 day074 一起错"这种可能性无法排除；加上第二条，
两份**独立写下**的解析式被同一把尺子量了两次——那时"错"才需要一个巧合。

## softmax 的雅可比为什么必须单独验一次

```text
∂(−log p_k)/∂z = p − onehot(k)      ← 第 5 条的读数
```

这一条**完全不经过** :func:`backprop.gradients.softmax_jacobian`：两者相乘之后
中间项全部抵消。于是"雅可比写错了"这件事在交叉熵的梯度上**看不见**——
它必须被第 3 条单独钉住。这不是"多验一遍"，而是"只有这一遍能发现它"。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from smart_research_agent.backprop import gradients, layers as layer_ops, network, types
from smart_research_agent.backprop.errors import (
    GradientError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.math_foundations.types import FLOAT_EPSILON, Matrix, Vector
from smart_research_agent.neural_basics.activations import activate
from smart_research_agent.neural_basics.types import DenseSpec, MLPSpec

#: 数值差分的默认步长（与 day074 的 ``calculus.DEFAULT_STEP`` 同值：中心差分的最优量级）.
DIFFERENCE_STEP = 1e-6

#: 逐点导数的对照容差（**相对**判据：见 :func:`max_scaled_gap`）.
DERIVATIVE_TOLERANCE = 1e-6

#: 恒等式（第 4 条）的容差：两侧是同一点上的两条代数路径，因此可以卡得很紧.
IDENTITY_TOLERANCE = 1e-12

#: 整条网络（第 6 条）的容差：数值差分自身的分辨率约 1e-9，取 1e-5 留五个数量级余量.
NETWORK_TOLERANCE = 1e-5

#: 逐点导数的对照点（**写死**）：刻意为避开 relu / leaky_relu 在 0 处的拐点.
#:
#: 0 处两条激活都不可导（左右导数不同），数值差分在那里会给出"两侧平均"——
#: 那不是"实现错了"，而是"这个点不适合做双向差分"。它应当被**从样本里换掉**，
#: 而不是被记成一条"记录的差异"（与 day074 gradcheck 的取舍同源）。
DERIVATIVE_POINTS: tuple[float, ...] = (-2.0, -0.5, 0.25, 1.0, 3.0)

#: 恒等式（第 4 条）的对照行（**写死**）：``(概率分布, 回传梯度 v)``.
#:
#: 第一个元素刻意是**真正的分布**（和为 1）而不是打分：恒等式
#: ``(Jᵀv)_i = p_i(v_i − ⟨p,v⟩)`` 对任何 ``p`` 都成立，但"它是不是一个分布"
#: 决定了这条读数有没有意义——用打分去算，得到的是一个**另一个**恒等式的读数。
JVP_CASES: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...] = (
    ((0.2, 0.3, 0.5), (0.5, -1.0, 2.0)),
    ((1.0,), (3.0,)),
    ((0.1, 0.2, 0.3, 0.4), (1.0, 1.0, 1.0, 1.0)),
)

#: 第 5 条用的 （打分, 正确下标） 用例（**写死**；长度覆盖 1 / 2 / 3 / 4 四档）.
CE_CASES: tuple[tuple[tuple[float, ...], int], ...] = (
    ((1.0, 2.0, 3.0), 0),
    ((0.2,), 0),
    ((-1.0, 0.0, 1.0), 2),
    ((1.0, 2.0, 3.0, 4.0), 1),
)

#: 第 6 条用的网络：``3 → 5 → 2``（隐藏层 relu、输出层恒等，权重由 LCG 生成）.
MLP_SPEC = MLPSpec(
    layers=(
        DenseSpec(3, 5, activation="relu", init="xavier", seed=31),
        DenseSpec(5, 2, activation=None, init="xavier", seed=32),
    )
)

#: 第 6 条用的输入与目标（**写死**）.
MLP_INPUTS: Matrix = ((0.4, -0.9, 1.2), (1.5, 0.3, -0.6))
MLP_TARGETS: Matrix = ((0.25, -0.75), (1.0, 0.5))

#: 第 7 条用的前馈输入 / 隐藏维 / 种子（与 day089 的对账同源，便于互相印证）.
FFN_INPUTS: Matrix = ((0.5, -1.0, 2.0, 0.25), (1.5, 0.0, -0.5, 3.0))
FFN_HIDDEN = 4
FFN_SEED = 7

#: 第 7 条用的回传梯度（写死：一个 2×4 的矩阵，元素刻意不全同）.
FFN_GRAD_OUTPUT: Matrix = ((0.5, -1.0, 2.0, 0.25), (-0.75, 0.5, 1.0, -0.5))


def difference_resolution(*, magnitude: float, step: float = DIFFERENCE_STEP) -> float:
    """中心差分在"函数值量级为 ``magnitude``"时的**分辨率下限** ``eps·|f|/(2h)``.

    与 day074 ``gradcheck.difference_resolution`` **同一口径**（同一公式、同一取向），
    在这里再写一遍是为了让本模块的容差选择可以被单独验算：

    ```text
    |f| ≈ 1      h = 1e-6   →  1.1e-10       容差 1e-6 是它的 9000 倍，够松
    |f| ≈ 30     h = 1e-6   →  3.3e-09       仍在容差之内
    ```

    低于这个下限的容差只会产出假警报，而那比漏报更坏——它会让人去改一份正确的代码。
    """
    if not math.isfinite(magnitude):
        raise NumericError(f"量级必须是有限实数，收到 {magnitude!r}。")
    if not math.isfinite(step) or step <= 0:
        raise ParameterError(f"步长必须是正的有限数，收到 {step!r}。")
    return FLOAT_EPSILON * abs(magnitude) / (2.0 * step)


def max_scaled_gap(left: Vector, right: Vector) -> float:
    """最大**相对**逐点误差 ``|l − r| / max(1, |r|)``（对照实际使用的判据）.

    为什么判据要缩放：一个"导数是 30"的量，绝对误差 1e-6 是精确到 7 位；
    而一个"导数是 1"的量，同样的绝对误差只精确到 6 位。
    数值差分的误差下限正比于**函数值的量级**，因此用绝对阈值去卡不同量级的量，
    等于对大数更苛刻——那会产出一条"不一致"，而它其实是"测量手段的极限"。
    """
    if len(left) != len(right):
        return math.inf
    worst = 0.0
    for first, second in zip(left, right, strict=True):
        if not (math.isfinite(first) and math.isfinite(second)):
            return math.inf
        worst = max(worst, abs(first - second) / max(1.0, abs(second)))
    return worst


def numerical_gradient(function: Callable[[Vector], float], point: Vector, *, step: float = DIFFERENCE_STEP) -> Vector:
    """中心差分梯度（**本模块唯一调用尺子的地方**；它不依赖本包的任何推导）."""
    from smart_research_agent.math_foundations.calculus import gradient

    return gradient(function, point, step=step)


@dataclass(frozen=True)
class GradientCheck:
    """两个来源之间的一次对账：来源、读数、判据（**含方向与上界**）.

    与 day087~089 的同类逐字同源：有 ``upper_bound`` 的时候判据是"实测 <= 上界"，
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
            return float(self.reading) <= float(self.upper_bound) + IDENTITY_TOLERANCE
        if self.exact:
            return self.reading == self.expected
        return abs(float(self.reading) - float(self.expected)) <= IDENTITY_TOLERANCE

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
    cross_check: GradientCheck | None = None

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
        """不通过时抛 :class:`backprop.errors.GradientError`（**第一次有人真的抛它**）."""
        if self.ok:
            return
        failures = [outcome.line() for outcome in self.applicable if not outcome.passed]
        raise GradientError("性质检查未全部通过：" + "；".join(failures))

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


# --------------------------------------------------------------------------------------
# 第 1 ~ 2 条：逐点导数与那个约定
# --------------------------------------------------------------------------------------

def evaluate_derivatives_vs_numerical(*, step: float = DIFFERENCE_STEP) -> PropertyOutcome:
    """五个逐元素激活的解析导数 vs 中心差分（**尺子不依赖任何推导**）."""
    analytic: list[float] = []
    numeric: list[float] = []
    for name in types.ELEMENTWISE_ACTIVATIONS:
        for point in DERIVATIVE_POINTS:
            analytic.append(gradients.activation_derivative(name, point))
            numeric.append(
                numerical_gradient(lambda vector, n=name: activate(n, vector)[0], (point,), step=step)[0]
            )
    worst = max_scaled_gap(tuple(analytic), tuple(numeric))
    check = GradientCheck(
        name="逐点导数的最大相对误差",
        left="解析导数（gradients）",
        right="中心差分（day074 calculus）",
        reading=worst,
        expected=0.0,
        exact=False,
        upper_bound=DERIVATIVE_TOLERANCE,
    )
    return PropertyOutcome(
        name=types.PROPERTY_ACTIVATION_DERIVATIVES_MATCH_NUMERICAL,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"{len(types.ELEMENTWISE_ACTIVATIONS)} 个激活 × {len(DERIVATIVE_POINTS)} 个点"
            f" = {len(analytic)} 个读数",
            f"最大相对误差 {worst:.3e}（容差 {DERIVATIVE_TOLERANCE:.0e}）",
        ),
        cross_check=check,
    )


def check_relu_subgradient_is_zero() -> PropertyOutcome:
    """``relu`` 在 ``x = 0`` 处取 0：反向里一个恒为负的输入**恰好**拿到 0（不是"接近 0"）.

    三处一起断言，缺一处就会漏掉一整类实现：

    ```text
    x = 0    用 >= 而不是 > 的实现会在这里把梯度放过去（差一个 grad）
    x < 0    恒负的输入必须得到字面量 0.0
    x > 0    必须原样通过（否则"开关"变成了"衰减器"）
    ```
    """
    origin = gradients.activation_derivative("relu", 0.0)
    backward = gradients.elementwise_backward("relu", ((0.0, -1.0, 2.0),), ((5.0, 5.0, 5.0),))
    origin_exactly_zero = backward[0][0] == 0.0
    negative_exactly_zero = backward[0][1] == 0.0
    positive_passthrough = backward[0][2] == 5.0
    check = GradientCheck(
        name="relu 在 0 处的次梯度",
        left="activation_derivative('relu', 0.0)",
        right="0.0",
        reading=origin,
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=types.PROPERTY_RELU_SUBGRADIENT_IS_ZERO,
        applicable=True,
        passed=(
            origin == 0.0
            and origin_exactly_zero
            and negative_exactly_zero
            and positive_passthrough
            and gradients.activation_derivative("relu", -1e6) == 0.0
        ),
        evidence=(
            f"导数在 0 处 = {origin}（恰好 0）；回传里 x=0 处拿到 {backward[0][0]}、"
            f"x<0 处拿到 {backward[0][1]}",
            f"正输入原样通过：5.0 → {backward[0][2]}",
        ),
        cross_check=check,
    )


# --------------------------------------------------------------------------------------
# 第 3 ~ 4 条：softmax 的雅可比与那条恒等式
# --------------------------------------------------------------------------------------


def check_softmax_jacobian_matches_numerical(*, step: float = DIFFERENCE_STEP) -> PropertyOutcome:
    """**跨天对账**：本包的 softmax 雅可比 vs day074 的数值雅可比，并读 day074 的结论."""
    from smart_research_agent.math_foundations.calculus import jacobian
    from smart_research_agent.math_foundations.gradcheck import (
        check_softmax_jacobian as day074_softmax_check,
    )

    analytic: list[float] = []
    numeric: list[float] = []
    for logits, _vector in JVP_CASES:
        probabilities = tuple(activate("softmax", logits))
        observed = jacobian(lambda point: activate("softmax", point), logits, step=step)
        analytic.extend(value for row in gradients.softmax_jacobian(probabilities) for value in row)
        numeric.extend(value for row in observed for value in row)
    worst = max_scaled_gap(tuple(analytic), tuple(numeric))
    day074 = day074_softmax_check(step=step)
    check = GradientCheck(
        name="雅可比的最大相对误差",
        left="gradients.softmax_jacobian",
        right=f"day074 jacobian（{len(analytic)} 个元素）",
        reading=worst,
        expected=0.0,
        exact=False,
        upper_bound=DERIVATIVE_TOLERANCE,
    )
    return PropertyOutcome(
        name=types.PROPERTY_SOFTMAX_JACOBIAN_MATCHES_NUMERICAL,
        applicable=True,
        passed=check.passed and day074.ok,
        evidence=(
            f"{len(JVP_CASES)} 行、{len(analytic)} 个元素，最大相对误差 {worst:.3e}",
            f"day074 gradcheck.check_softmax_jacobian 的结论 {day074.status}"
            f"（最大误差 {day074.max_absolute_error:.2e}）",
        ),
        cross_check=check,
    )


def check_softmax_jvp_avoids_matrix() -> PropertyOutcome:
    """恒等式 ``(Jᵀv)_i = p_i(v_i − ⟨p,v⟩)`` 与**显式** ``Jᵀ·v`` 一致（偏差 <= 1e-12）."""
    worst = 0.0
    for probabilities, vector in JVP_CASES:
        jacobian_matrix = gradients.softmax_jacobian(probabilities)
        explicit = tuple(
            math.fsum(jacobian_matrix[row][column] * vector[row] for row in range(len(vector)))
            for column in range(len(vector))
        )
        implicit = gradients.softmax_jacobian_vector_product(probabilities, vector)
        for left, right in zip(implicit, explicit, strict=True):
            worst = max(worst, abs(left - right))
    check = GradientCheck(
        name="恒等式与显式矩阵乘的最大偏差",
        left="p⊙(v − ⟨p,v⟩)",
        right=f"Jᵀ·v（显式 {len(JVP_CASES[0][0])}×{len(JVP_CASES[0][0])}）",
        reading=worst,
        expected=0.0,
        exact=False,
        upper_bound=IDENTITY_TOLERANCE,
    )
    return PropertyOutcome(
        name=types.PROPERTY_SOFTMAX_JVP_AVOIDS_MATRIX,
        applicable=True,
        passed=check.passed,
        evidence=(
            f"{len(JVP_CASES)} 行、最大偏差 {worst:.3e}（容差 {IDENTITY_TOLERANCE:.0e}）",
            "中间量从 O(n²) 降到 O(n)：生产路径只走这一条",
        ),
        cross_check=check,
    )


# --------------------------------------------------------------------------------------
# 第 5 条：交叉熵的梯度
# --------------------------------------------------------------------------------------


def check_cross_entropy_gradient_is_p_minus_onehot(*, step: float = DIFFERENCE_STEP) -> PropertyOutcome:
    """**跨天对账**：CE 的梯度 = ``p − onehot``，vs day074 的数值差分，并读它的结论."""
    from smart_research_agent.math_foundations.gradcheck import (
        check_cross_entropy_gradient as day074_ce_check,
    )
    from smart_research_agent.sft.loss import cross_entropy as production_cross_entropy

    analytic: list[float] = []
    numeric: list[float] = []
    for logits, target in CE_CASES:
        analytic.extend(gradients.cross_entropy_grad(logits, target))
        numeric.extend(
            numerical_gradient(
                lambda point, index=target: production_cross_entropy(list(point), index),
                logits,
                step=step,
            )
        )
    worst = max_scaled_gap(tuple(analytic), tuple(numeric))
    day074 = day074_ce_check(step=step)
    check = GradientCheck(
        name="p − onehot 的最大相对误差",
        left=f"gradients.cross_entropy_grad（{len(analytic)} 个偏导数）",
        right="对 sft.loss.cross_entropy 的数值差分",
        reading=worst,
        expected=0.0,
        exact=False,
        upper_bound=DERIVATIVE_TOLERANCE,
    )
    return PropertyOutcome(
        name=types.PROPERTY_CROSS_ENTROPY_GRADIENT_IS_P_MINUS_ONEHOT,
        applicable=True,
        passed=check.passed and day074.ok,
        evidence=(
            f"{len(CE_CASES)} 组（长度 1~4）、{len(analytic)} 个偏导数，最大相对误差 {worst:.3e}",
            f"day074 gradcheck.check_cross_entropy_gradient 的结论 {day074.status}"
            f"（最大误差 {day074.max_absolute_error:.2e}）",
        ),
        cross_check=check,
    )


# --------------------------------------------------------------------------------------
# 第 6 ~ 7 条：整条网络与跨包逐位
# --------------------------------------------------------------------------------------


def check_mlp_backward_matches_numerical(*, step: float = DIFFERENCE_STEP) -> PropertyOutcome:
    """整条 MLP 的解析梯度（参数）vs 数值差分（**尺子**在 day074 的 ``calculus`` 里）."""
    cache = network.mlp_forward_with_cache(MLP_SPEC, MLP_INPUTS)
    trace = network.mlp_loss_gradients(cache, MLP_TARGETS)
    analytic = trace.flat_parameter_gradients()
    objective = network.parameter_objective(MLP_SPEC, MLP_INPUTS, MLP_TARGETS)
    params, _ = layer_ops.flatten_parameters(network.build_parameters(MLP_SPEC))
    numeric = numerical_gradient(objective, params, step=step)
    worst = max_scaled_gap(analytic, numeric)
    # 顺带对账一次"两条前向路径给出同一个输出"（同一组权重、同一批输入）
    from smart_research_agent.neural_basics.network import mlp_forward

    chain = network.build_chain(MLP_SPEC)
    ours = network.chain_forward(chain, MLP_INPUTS)
    theirs = mlp_forward(MLP_SPEC, MLP_INPUTS)
    forward_mismatches = sum(
        1
        for row_a, row_b in zip(ours, theirs, strict=True)
        for a, b in zip(row_a, row_b, strict=True)
        if a != b
    )
    check = GradientCheck(
        name="整条 MLP 参数梯度的最大相对误差",
        left=f"mlp_loss_gradients（{len(analytic)} 个偏导数）",
        right="对 parameter_objective 的中心差分",
        reading=worst,
        expected=0.0,
        exact=False,
        upper_bound=NETWORK_TOLERANCE,
    )
    return PropertyOutcome(
        name=types.PROPERTY_MLP_BACKWARD_MATCHES_NUMERICAL,
        applicable=True,
        passed=check.passed and forward_mismatches == 0,
        evidence=(
            f"宽度链 {'→'.join(str(width) for width in MLP_SPEC.widths)}、"
            f"{len(analytic)} 个偏导数、最大相对误差 {worst:.3e}",
            f"chain_forward vs day089 mlp_forward 逐位不一致 {forward_mismatches} 个"
            f"（{len(MLP_INPUTS)} 行 × {MLP_SPEC.output_width} 列）",
        ),
        cross_check=check,
    )


def check_ffn_backward_matches_encoder_decoder() -> PropertyOutcome:
    """**跨天对账**：``ffn_backward`` vs ``encoder_decoder.layers.feed_forward_backward`` **逐位**."""
    from smart_research_agent.encoder_decoder.layers import feed_forward, feed_forward_backward
    from smart_research_agent.encoder_decoder.types import (
        ACTIVATION_GELU,
        ACTIVATION_RELU,
        FFNWeights,
    )
    from smart_research_agent.neural_basics.network import build_ffn_params

    params = build_ffn_params(hidden=FFN_HIDDEN, seed=FFN_SEED)
    real = FFNWeights(w_in=params.w_in, b_in=params.b_in, w_out=params.w_out, b_out=params.b_out)
    mismatches = 0
    fields = ("grad_w_in", "grad_b_in", "grad_w_out", "grad_b_out", "grad_inputs")
    for ours_name, day079_name in (("relu", ACTIVATION_RELU), ("gelu", ACTIVATION_GELU)):
        ours = network.ffn_backward(FFN_INPUTS, real, activation=ours_name, grad_output=FFN_GRAD_OUTPUT)
        _output, cache = feed_forward(FFN_INPUTS, real, activation=day079_name)
        theirs = feed_forward_backward(cache, FFN_GRAD_OUTPUT)
        for field_name in fields:
            ours_values = _flatten_field(getattr(ours, field_name))
            their_values = _flatten_field(getattr(theirs, field_name))
            for left, right in zip(ours_values, their_values, strict=True):
                mismatches += int(left != right)
    check = GradientCheck(
        name="五块梯度的逐位不一致个数",
        left="backprop.network.ffn_backward",
        right="encoder_decoder.layers.feed_forward_backward",
        reading=float(mismatches),
        expected=0.0,
        exact=True,
    )
    return PropertyOutcome(
        name=types.PROPERTY_FFN_BACKWARD_MATCHES_ENCODER_DECODER,
        applicable=True,
        passed=mismatches == 0,
        evidence=(
            f"hidden={FFN_HIDDEN}、d_ff={FFN_HIDDEN * 4}、2 种激活、5 块梯度，"
            f"逐位不一致 {mismatches} 个",
            "四条前提：同一组权重、同样的 (d_ff, d) 形状约定、同样的 fsum 累加、同样的 relu/gelu 导数",
        ),
        cross_check=check,
    )


def _flatten_field(value: object) -> tuple[float, ...]:
    """把一个可能是矩阵或向量的字段压平成一串数（对照两侧必须用同一个顺序）."""
    items = tuple(value)  # type: ignore[call-overload]
    if items and isinstance(items[0], (tuple, list)):
        return tuple(float(item) for row in items for item in row)  # type: ignore[union-attr]
    return tuple(float(item) for item in items)


# --------------------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------------------


def check_all(*, step: float = DIFFERENCE_STEP) -> PropertyReport:
    """一次跑完七条性质（顺序与 :data:`types.BACKPROP_PROPERTIES` 一致）."""
    outcomes = (
        evaluate_derivatives_vs_numerical(step=step),
        check_relu_subgradient_is_zero(),
        check_softmax_jacobian_matches_numerical(step=step),
        check_softmax_jvp_avoids_matrix(),
        check_cross_entropy_gradient_is_p_minus_onehot(step=step),
        check_mlp_backward_matches_numerical(step=step),
        check_ffn_backward_matches_encoder_decoder(),
    )
    names = {outcome.name for outcome in outcomes}
    extra = names - set(types.BACKPROP_PROPERTIES)
    missing = set(types.BACKPROP_PROPERTIES) - names
    if extra or missing:  # pragma: no cover - 只在有人改性质名单时触发
        raise ShapeError(
            "性质名单与 types.BACKPROP_PROPERTIES 不一致："
            f"多 {sorted(extra)}、缺 {sorted(missing)}。"
            "名单对不上时，报告里那七行会安静地少一行或多一行。"
        )
    return PropertyReport(outcomes=outcomes)


__all__ = [
    "CE_CASES",
    "DERIVATIVE_POINTS",
    "DERIVATIVE_TOLERANCE",
    "DIFFERENCE_STEP",
    "FFN_GRAD_OUTPUT",
    "FFN_HIDDEN",
    "FFN_INPUTS",
    "FFN_SEED",
    "IDENTITY_TOLERANCE",
    "JVP_CASES",
    "MLP_INPUTS",
    "MLP_SPEC",
    "MLP_TARGETS",
    "NETWORK_TOLERANCE",
    "GradientCheck",
    "PropertyOutcome",
    "PropertyReport",
    "check_all",
    "check_cross_entropy_gradient_is_p_minus_onehot",
    "check_ffn_backward_matches_encoder_decoder",
    "check_mlp_backward_matches_numerical",
    "check_relu_subgradient_is_zero",
    "check_softmax_jacobian_matches_numerical",
    "check_softmax_jvp_avoids_matrix",
    "difference_resolution",
    "evaluate_derivatives_vs_numerical",
    "max_scaled_gap",
    "numerical_gradient",
]
