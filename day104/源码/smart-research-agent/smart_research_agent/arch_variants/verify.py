"""``verify.py``：六条性质与一次**整条变体**的梯度校验（day082）.

## 一、六条性质里，四条是“实测 vs 声明”

```text
mask_zeroes_are_exact            权重表里被掩码挡掉的格子**逐位是 0.0**（不留容差）
decoder_reads_only_past          实测依赖表与**因果掩码**逐格一致（且逐位精确）
encoder_reads_every_position     实测依赖表与**全开掩码**逐格一致（每一行都看得到全部）
cross_spans_all_sources          交叉那一条路**一格 0 都没有**（长方形、无掩码）
stack_preserves_shape            变体保形（(n, d) → (n, d)）
deterministic                    两次前向的输出与权重逐位相同
```

其中第 2、3 条在 ``decoder_only`` / ``encoder_only`` 上各有一条**不适用**：
一个变体没有解码器，就谈不上“解码器只读过去”。
本包因此给每条读数加了 ``applicable``：

```text
不适用 ≠ 通过        applicable=False 时 passed 也是 False，而报告只对**适用**的那些下结论
```

这条区分是被真实需求逼出来的：如果“不适用”直接记成“通过”，
那么把 ``encoder_only`` 的检查全删掉、与“它全都通过”在报告里长得一模一样。

## 二、梯度校验为什么放在**整条变体**上

day075 验过注意力的七步、day079 验过块的九块参数。今天新出现的只有**接线**：

```text
① 掩码有没有被穿进反向（前向因果、反向把未来的梯度也算上 ⇒ 前向完全正确）
② 两条流的梯度有没有接上（dEncoder 要累加，之后还要穿过整条编码器链）
```

第 ② 条是 ``encoder_decoder`` 独有的：``VariantGradients.grad_source_inputs``
必须穿过编码器链才能回到源序列；漏掉那一段时形状**完全合法**。
因此本节的判据是：**解析梯度 vs 中心差分**，两条流各比一次。

一个刻意的选择：梯度校验默认用 ``gelu``（精确 erf 版本，处处光滑），
因为 ReLU 在 ``x = 0`` 处不可导，中心差分会踩到折点——
那是**判据自身的噪声**，而不是实现的错。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.arch_variants.errors import (
    AssemblyError,
    GradientError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.arch_variants.masks import (
    MASK_CAUSAL,
    MASK_FULL,
    Mask,
    mask_of,
)
from smart_research_agent.arch_variants.probe import (
    DEFAULT_PERTURBATION,
    STREAM_MAIN,
    STREAM_SOURCE,
    DependencyReport,
    cross_dependency,
    dependency_matrix,
    encoder_dependency,
)
from smart_research_agent.arch_variants.stacks import (
    VariantGradients,
    VariantParameters,
    loss_gradient,
    variant_backward,
    variant_forward,
    variant_loss,
)
from smart_research_agent.arch_variants.types import (
    ARCH_PROPERTIES,
    PROPERTY_CROSS_SPANS_ALL_SOURCES,
    PROPERTY_DECODER_READS_ONLY_PAST,
    PROPERTY_DESCRIPTIONS,
    PROPERTY_DETERMINISTIC,
    PROPERTY_ENCODER_READS_EVERY_POSITION,
    PROPERTY_MASK_ZEROES_ARE_EXACT,
    PROPERTY_STACK_PRESERVES_SHAPE,
    VARIANT_DECODER_ONLY,
    VARIANT_ENCODER_DECODER,
    VARIANT_ENCODER_ONLY,
    VariantForward,
    max_absolute,
    relative_matrix_error,
)
from smart_research_agent.encoder_decoder.types import ACTIVATION_GELU, NORM_PRE
from smart_research_agent.math_foundations.types import (
    Matrix,
    matrix_shape,
    validate_matrix,
)

#: 梯度容差（与 day075/079 同值——**三处不许走散**）.
GRADIENT_TOLERANCE = 1e-6

#: 中心差分的步长（day074 的 ``best_step_for_central`` 量出来的量级）.
NUMERICAL_STEP = 1e-6


@dataclass(frozen=True)
class PropertyOutcome:
    """一条性质的读数（``applicable=False`` 表示这条性质对这个变体不适用）."""

    name: str
    passed: bool
    evidence: str
    detail: str = ""
    applicable: bool = True

    def __post_init__(self) -> None:
        if self.name not in ARCH_PROPERTIES:
            raise ParameterError(
                f"未知的性质名 {self.name!r}：可选 {', '.join(ARCH_PROPERTIES)}。"
            )
        if not self.applicable and self.passed:
            raise NumericError(
                f"性质 {self.name} 标了“不适用”却又标了“通过”："
                "两者必须分开——否则'删掉了这条检查'与'它通过了'在报告里长得一样。"
            )
        object.__setattr__(self, "evidence", str(self.evidence))
        object.__setattr__(self, "detail", str(self.detail))

    @property
    def description(self) -> str:
        """这一条在说什么."""
        return PROPERTY_DESCRIPTIONS[self.name]

    @property
    def state(self) -> str:
        """三态：``通过`` / ``未通过`` / ``不适用``."""
        if not self.applicable:
            return "不适用"
        return "通过" if self.passed else "未通过"

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "name": self.name,
            "state": self.state,
            "applicable": self.applicable,
            "evidence": self.evidence,
            "detail": self.detail,
        }

    def summary_line(self) -> str:
        """一行说明：``[通过] mask_zeroes_are_exact | 掩码外 18 个权重，最大读数 0.000000e+00``."""
        return f"[{self.state}] {self.name} | {self.evidence}"


@dataclass(frozen=True)
class PropertyReport:
    """六条性质的报告（**只对适用的那些下结论**）."""

    variant: str
    outcomes: tuple[PropertyOutcome, ...]
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        resolved = tuple(self.outcomes)
        if not resolved:
            raise ParameterError("性质报告不能为空。")
        names = {item.name for item in resolved}
        if names != set(ARCH_PROPERTIES):
            raise NumericError(
                f"性质报告的名单不完整：缺 {sorted(set(ARCH_PROPERTIES) - names)}——"
                "缺一条与'它不适用'在报告里必须长得不一样。"
            )
        object.__setattr__(self, "outcomes", resolved)
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def ok(self) -> bool:
        """**适用的**那些全部通过."""
        return all(item.passed for item in self.outcomes if item.applicable)

    @property
    def applicable_outcomes(self) -> tuple[PropertyOutcome, ...]:
        """适用的那几条."""
        return tuple(item for item in self.outcomes if item.applicable)

    @property
    def skipped(self) -> tuple[str, ...]:
        """不适用的那几条的名字."""
        return tuple(item.name for item in self.outcomes if not item.applicable)

    def outcome_of(self, name: str) -> PropertyOutcome:
        """按名字取一条读数（名字不认识时当场拒绝）."""
        for item in self.outcomes:
            if item.name == name:
                return item
        raise ParameterError(
            f"报告里没有性质 {name!r}：可选 {', '.join(item.name for item in self.outcomes)}。"
        )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "variant": self.variant,
            "ok": self.ok,
            "skipped": list(self.skipped),
            "outcomes": [item.to_dict() for item in self.outcomes],
        }

    def summary_lines(self) -> tuple[str, ...]:
        """逐行说明（实验脚本直接印它）."""
        return tuple(item.summary_line() for item in self.outcomes)


# ---------------------------------------------------------------------- 六条性质


def check_weights_are_zero_where_masked(
    forward: VariantForward,
) -> PropertyOutcome:
    """第 1 条：被掩码挡掉的位置，权重**逐位**是 0.0.

    三条路各查一次，而它们用的掩码不同：

    ```text
    编码器（或唯一）流   用 ``forward.source_mask``（若有）否则 ``forward.mask``
    解码器自注意力       用**因果掩码**（day079 的 ``decoder_block`` 强制的那个）
    交叉注意力           **不查**：它没有掩码（第 4 条查的是它“一格 0 都没有”）
    ```
    """
    effective: Mask = (
        forward.source_mask if forward.source_mask is not None else forward.mask
    )
    outside = 0
    worst = 0.0
    for weights in forward.block_weights:
        rows, columns = matrix_shape(weights)
        if (rows, columns) != (len(effective), len(effective[0])):
            raise ShapeError(
                f"权重表 {rows}×{columns} 与掩码 "
                f"{len(effective)}×{len(effective[0])} 形状不一致。"
            )
        for row in range(rows):
            for column in range(columns):
                if not effective[row][column]:
                    outside += 1
                    worst = max(worst, abs(weights[row][column]))
    if forward.decoder_self_weights:
        causal = mask_of(MASK_CAUSAL, forward.shape.tokens)
        for weights in forward.decoder_self_weights:
            for row in range(forward.shape.tokens):
                for column in range(forward.shape.tokens):
                    if not causal[row][column]:
                        outside += 1
                        worst = max(worst, abs(weights[row][column]))
    return PropertyOutcome(
        name=PROPERTY_MASK_ZEROES_ARE_EXACT,
        passed=worst == 0.0,
        evidence=(
            f"掩码外共 {outside} 个权重，最大读数 {worst:.6e}"
            f"（{'逐位为 0' if worst == 0.0 else '**不是逐位为 0**'}）"
        ),
        detail="判据用 `==`：被掩码的打分从来没有被读过，因此它逐位为 0 而不是'接近 0'",
    )


def _matches_expected(report: DependencyReport, expected: Mask) -> bool:
    """实测表与**期望掩码**逐格比（期望由本函数内部构造，**不读 ``report.mask``**）.

    这一点是本课最容易被写松的地方：如果拿 ``report.mask`` 当期望，
    那么“实测与它自己前向用的那张掩码一致”永远成立——
    而“这张掩码选对了没有”根本没有被检查。
    """
    if (len(expected), len(expected[0])) != (report.rows, report.columns):
        raise ShapeError(
            f"期望掩码 {len(expected)}×{len(expected[0])} 与实测表 "
            f"{report.rows}×{report.columns} 形状不一致。"
        )
    measured = report.measured_mask
    return all(
        measured[row][column] == expected[row][column]
        for row in range(report.rows)
        for column in range(report.columns)
    )


def check_reads_only_past(
    params: VariantParameters,
    report: DependencyReport,
) -> PropertyOutcome:
    """第 2 条：解码器第 i 行只依赖 ``j <= i``（**实测**，不是读 ``causal=True``）."""
    if params.variant == VARIANT_ENCODER_ONLY:
        return PropertyOutcome(
            name=PROPERTY_DECODER_READS_ONLY_PAST,
            passed=False,
            applicable=False,
            evidence=f"{params.variant} 没有解码器那一流：这条性质对它不适用",
        )
    expected = mask_of(MASK_CAUSAL, params.shape.tokens)
    matches = _matches_expected(report, expected)
    return PropertyOutcome(
        name=PROPERTY_DECODER_READS_ONLY_PAST,
        passed=matches and report.exact,
        evidence=(
            f"实测可达 {report.reach_profile()}（因果掩码应为 "
            f"{tuple(range(1, params.shape.tokens + 1))}）| 挡掉的格子最大读数 "
            f"{report.blocked_gap:.6e}"
        ),
        detail="判据落在**扰动实测**上：掩码传错、漏传、或反向没穿掩码时它都会亮红",
    )


def check_reads_every_position(
    params: VariantParameters,
    report: DependencyReport,
) -> PropertyOutcome:
    """第 3 条：**双向**那一条流每一行都依赖所有位置（实测覆盖全表）.

    期望掩码是**全开**的那张（由本函数构造）——因此“把因果掩码给编码器”
    这种错法会在这里亮红，而不是因为“与它自己一致”而蒙混过去。
    """
    if params.variant == VARIANT_DECODER_ONLY:
        return PropertyOutcome(
            name=PROPERTY_ENCODER_READS_EVERY_POSITION,
            passed=False,
            applicable=False,
            evidence="decoder_only 只有一条因果流：这条性质对它不适用",
        )
    if report.rows != report.columns:
        raise AssemblyError(
            f"编码器那一侧的自注意力权重表应当是方阵，实测表是 "
            f"{report.rows}×{report.columns}——长方形的那一张是交叉注意力。"
        )
    expected = mask_of(MASK_FULL, report.columns)
    matches = _matches_expected(report, expected)
    return PropertyOutcome(
        name=PROPERTY_ENCODER_READS_EVERY_POSITION,
        passed=matches and report.exact and report.allowed_floor > 0.0,
        evidence=(
            f"实测可达 {report.reach_profile()}（应当全是 {report.columns}）| 最小的允许读数 "
            f"{report.allowed_floor:.6e}"
        ),
        detail=(
            "`allowed_floor > 0` 是这条的一半：只看'挡掉的格子是 0'时，"
            "一张把一切都挡住的掩码也会通过"
        ),
    )


def check_cross_spans_all_sources(
    params: VariantParameters,
    report: DependencyReport | None,
) -> PropertyOutcome:
    """第 4 条：交叉注意力每一行覆盖**全部**源位置（长方形、一格 0 都没有）."""
    if params.variant != VARIANT_ENCODER_DECODER:
        return PropertyOutcome(
            name=PROPERTY_CROSS_SPANS_ALL_SOURCES,
            passed=False,
            applicable=False,
            evidence=f"{params.variant} 没有交叉注意力：这条性质对它不适用",
        )
    if report is None:  # pragma: no cover - 调用方按变体给表
        raise AssemblyError("encoder_decoder 变体必须给出交叉依赖表。")
    sources = params.shape.source_length
    expected: Mask = tuple(
        tuple(True for _ in range(report.columns)) for _ in range(report.rows)
    )
    matches = _matches_expected(report, expected)
    return PropertyOutcome(
        name=PROPERTY_CROSS_SPANS_ALL_SOURCES,
        passed=matches and report.exact and report.allowed_floor > 0.0,
        evidence=(
            f"权重表 {report.rows}×{report.columns}（**长方形**）| 每一行的可达源位置 "
            f"{report.reach_profile()}（应当全是 {sources}）| 最小的允许读数 "
            f"{report.allowed_floor:.6e}"
        ),
        detail=(
            "它与第 2 条在**同一个解码器**里同时成立：自注意力因果、交叉注意力不因果——"
            "这正是 day079 '交叉注意力不能被赋因果掩码'那句拒绝的正面读法"
        ),
    )


def check_shape_preserved(forward: VariantForward) -> PropertyOutcome:
    """第 5 条：变体保形（不保形就没法堆叠）."""
    source_shape = matrix_shape(forward.inputs)
    output_shape = matrix_shape(forward.output)
    return PropertyOutcome(
        name=PROPERTY_STACK_PRESERVES_SHAPE,
        passed=source_shape == output_shape,
        evidence=f"输入 {source_shape} → 输出 {output_shape}",
        detail="主流保形与编码器输出形状是两件事：编码器输出必须等于 (源长度, 隐维)",
    )


def _bitwise_equal(left: Matrix, right: Matrix) -> bool:
    """两个矩阵**逐位**相等（``==``，不留容差）."""
    if matrix_shape(left) != matrix_shape(right):
        return False
    return all(
        a == b
        for left_row, right_row in zip(left, right, strict=True)
        for a, b in zip(left_row, right_row, strict=True)
    )


def check_deterministic(
    params: VariantParameters,
    inputs: Matrix,
    **kwargs: Any,
) -> PropertyOutcome:
    """第 6 条：两次前向逐位相同（没有随机数，也没有隐藏的全局状态）."""
    first = variant_forward(params, inputs, **kwargs)
    second = variant_forward(params, inputs, **kwargs)
    same_output = _bitwise_equal(first.output, second.output)
    same_weights = all(
        _bitwise_equal(left, right)
        for left, right in zip(first.block_weights, second.block_weights, strict=True)
    )
    same_cross = all(
        _bitwise_equal(left, right)
        for left, right in zip(first.cross_weights, second.cross_weights, strict=True)
    )
    return PropertyOutcome(
        name=PROPERTY_DETERMINISTIC,
        passed=same_output and same_weights and same_cross,
        evidence=(
            f"两次前向的输出相同：{same_output}；权重相同：{same_weights}；"
            f"交叉权重相同：{same_cross}（{len(first.block_weights)} 层逐位比较）"
        ),
        detail="同一条判据让'换一颗种子重跑一遍'这件事有确定的结果",
    )


def check_properties(
    params: VariantParameters,
    inputs: Matrix,
    *,
    source: Matrix | None = None,
    pads: Any = None,
    perturbation: float = DEFAULT_PERTURBATION,
    activation: str = ACTIVATION_GELU,
) -> PropertyReport:
    """跑完六条性质（第 2、3、4 条各自可能需要一张实测依赖表）."""
    if not isinstance(params, VariantParameters):
        raise ParameterError(f"params 必须是 VariantParameters，收到 {type(params).__name__}。")
    if source is not None and params.variant != VARIANT_ENCODER_DECODER:
        raise AssemblyError(
            f"{params.variant} 变体只有一条流，不该收到 source："
            "本包不做静默忽略——多给一路时那个乘法根本没有定义。"
        )
    forward = variant_forward(
        params, inputs, source=source, pads=pads, activation=activation
    )
    main_report = dependency_matrix(
        params,
        inputs,
        source=source,
        pads=pads,
        perturbation=perturbation,
        activation=activation,
    )
    encoder_report = encoder_dependency(
        params,
        inputs,
        source=source,
        perturbation=perturbation,
        activation=activation,
    )
    cross_report = (
        cross_dependency(
            params,
            inputs,
            source if source is not None else forward.source,
            perturbation=perturbation,
            activation=activation,
        )
        if params.variant == VARIANT_ENCODER_DECODER
        else None
    )
    outcomes = (
        check_weights_are_zero_where_masked(forward),
        check_reads_only_past(params, main_report),
        check_reads_every_position(params, encoder_report),
        check_cross_spans_all_sources(params, cross_report),
        check_shape_preserved(forward),
        check_deterministic(params, inputs, source=source, pads=pads, activation=activation),
    )
    notes = [
        f"变体 {params.variant} | {params.shape.summary_line()}",
        f"实测依赖表（主流）：{main_report.summary_line()}",
        f"实测依赖表（编码器流）：{encoder_report.summary_line()}",
        (
            f"实测依赖表（交叉那一路）：{cross_report.summary_line()}"
            if cross_report is not None
            else "交叉那一路：不适用"
        ),
        f"梯度校验用的激活是 {activation}（光滑；ReLU 在 0 处不可导会踩折点）",
    ]
    return PropertyReport(variant=params.variant, outcomes=outcomes, notes=tuple(notes))


# ---------------------------------------------------------------------- 梯度校验


@dataclass(frozen=True)
class GradientOutcome:
    """一条流上的梯度读数：解析 vs 中心差分."""

    stream: str
    analytic_norm: float
    numerical_norm: float
    max_gap: float
    relative_gap: float
    samples: int
    passed: bool

    def summary_line(self) -> str:
        """一行说明."""
        return (
            f"{self.stream:<6} | 解析范数 {self.analytic_norm:.6f} | 数值范数 "
            f"{self.numerical_norm:.6f} | 最大差 {self.max_gap:.3e} | 相对差 "
            f"{self.relative_gap:.3e} | {self.samples} 个分量 | "
            f"{'通过' if self.passed else '未通过'}"
        )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "stream": self.stream,
            "analytic_norm": self.analytic_norm,
            "numerical_norm": self.numerical_norm,
            "max_gap": self.max_gap,
            "relative_gap": self.relative_gap,
            "samples": self.samples,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class GradientReport:
    """整条变体的梯度报告（单流一条、两条流两条）."""

    variant: str
    tolerance: float
    step: float
    outcomes: tuple[GradientOutcome, ...]
    analytic: VariantGradients
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.outcomes:
            raise ParameterError("梯度报告不能为空。")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ParameterError(f"容差必须是正有限数，收到 {self.tolerance!r}。")
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def ok(self) -> bool:
        """全部通过（不通过时当场抛 :class:`GradientError` 的是 ``require_ok``）."""
        return all(item.passed for item in self.outcomes)

    def outcome_of(self, stream: str) -> GradientOutcome:
        """按流名取读数."""
        for item in self.outcomes:
            if item.stream == stream:
                return item
        raise ParameterError(
            f"报告里没有流 {stream!r}：可选 {', '.join(item.stream for item in self.outcomes)}。"
        )

    def require_ok(self) -> None:
        """断言全绿——**不过就抛** :class:`GradientError`（它属于控制流，不进报告）."""
        if not self.ok:
            failing = ", ".join(item.stream for item in self.outcomes if not item.passed)
            raise GradientError(
                f"梯度校验未通过（{failing}）：解析与数值的差超过容差 {self.tolerance}——"
                "而这通常意味着掩码没有被穿进反向（前向完全正确的那种失败）。"
            )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "variant": self.variant,
            "tolerance": self.tolerance,
            "step": self.step,
            "ok": self.ok,
            "outcomes": [item.to_dict() for item in self.outcomes],
        }

    def summary_lines(self) -> tuple[str, ...]:
        """逐行说明."""
        return tuple(item.summary_line() for item in self.outcomes)


def _loss_on_stream(
    params: VariantParameters,
    inputs: Matrix,
    target: Matrix,
    *,
    stream: str,
    source: Matrix | None,
    source_seed: int,
    activation: str,
    placement: str,
    use_residual: bool,
) -> Any:
    """把损失写成一个**以某一条流的输入为自变量**的一元函数（中心差分要它）."""

    def loss_of(matrix: Matrix) -> float:
        if stream == STREAM_SOURCE:
            return variant_loss(
                params,
                inputs,
                target,
                source=matrix,
                activation=activation,
                placement=placement,
                use_residual=use_residual,
            )
        return variant_loss(
            params,
            matrix,
            target,
            source=source,
            source_seed=source_seed,
            activation=activation,
            placement=placement,
            use_residual=use_residual,
        )

    return loss_of


def numerical_stream_gradient(
    params: VariantParameters,
    inputs: Matrix,
    target: Matrix,
    *,
    stream: str = STREAM_MAIN,
    source: Matrix | None = None,
    step: float = NUMERICAL_STEP,
    source_seed: int = 13,
    activation: str = ACTIVATION_GELU,
    placement: str = NORM_PRE,
    use_residual: bool = True,
) -> Matrix:
    """中心差分：逐个分量算 ``(L(+h) − L(−h)) / 2h``.

    逐分量而不是逐行：这样做出来的矩阵与解析梯度**逐格可比**，
    而“只对某一行错了”这种偏差不会被行内求和掩盖掉。
    """
    if isinstance(step, bool) or not isinstance(step, (int, float)):
        raise ParameterError(f"步长必须是数，收到 {step!r}。")
    resolved_step = float(step)
    if not math.isfinite(resolved_step) or resolved_step <= 0.0:
        raise ParameterError(f"步长必须是正的有限数，收到 {step!r}。")
    if stream == STREAM_SOURCE:
        if params.variant != VARIANT_ENCODER_DECODER:
            raise AssemblyError(
                f"{params.variant} 变体没有第二路：源序列的梯度对它没有定义。"
            )
        if source is None:
            raise AssemblyError("要算源序列的梯度就必须给出源序列。")
        base = validate_matrix(source, name="source")
    elif stream == STREAM_MAIN:
        base = validate_matrix(inputs, name="inputs")
    else:
        raise ParameterError(f"未知的流 {stream!r}：可选 {STREAM_MAIN}, {STREAM_SOURCE}。")
    loss_of = _loss_on_stream(
        params,
        inputs,
        target,
        stream=stream,
        source=source,
        source_seed=source_seed,
        activation=activation,
        placement=placement,
        use_residual=use_residual,
    )
    rows, columns = matrix_shape(base)
    out: list[tuple[float, ...]] = []
    for row in range(rows):
        cells: list[float] = []
        for column in range(columns):
            plus = _shift_component(base, row, column, resolved_step)
            minus = _shift_component(base, row, column, -resolved_step)
            cells.append((loss_of(plus) - loss_of(minus)) / (2.0 * resolved_step))
        out.append(tuple(cells))
    return tuple(out)


def _shift_component(matrix: Matrix, row: int, column: int, delta: float) -> Matrix:
    """把 ``(row, column)`` 那一个分量加 ``delta``（其余分量逐位不变）."""
    return tuple(
        tuple(
            value + delta if (r == row and c == column) else value
            for c, value in enumerate(values)
        )
        for r, values in enumerate(matrix)
    )


def check_variant_gradients(
    params: VariantParameters,
    inputs: Matrix,
    target: Matrix,
    *,
    source: Matrix | None = None,
    tolerance: float = GRADIENT_TOLERANCE,
    step: float = NUMERICAL_STEP,
    source_seed: int = 13,
    activation: str = ACTIVATION_GELU,
    placement: str = NORM_PRE,
    use_residual: bool = True,
) -> GradientReport:
    """整条变体的梯度校验：主流一次、两条流时源流再来一次."""
    if not isinstance(params, VariantParameters):
        raise ParameterError(f"params 必须是 VariantParameters，收到 {type(params).__name__}。")
    forward = variant_forward(
        params,
        inputs,
        source=source,
        source_seed=source_seed,
        activation=activation,
        placement=placement,
        use_residual=use_residual,
    )
    analytic = variant_backward(
        forward, params, loss_gradient(forward.output, target), activation=activation
    )
    streams = [STREAM_MAIN]
    if params.variant == VARIANT_ENCODER_DECODER:
        streams.append(STREAM_SOURCE)
    outcomes: list[GradientOutcome] = []
    for stream in streams:
        reference = numerical_stream_gradient(
            params,
            inputs,
            target,
            stream=stream,
            source=source if source is not None else forward.source,
            step=step,
            source_seed=source_seed,
            activation=activation,
            placement=placement,
            use_residual=use_residual,
        )
        if stream == STREAM_MAIN:
            analytic_matrix = analytic.grad_inputs
        else:
            if analytic.grad_source_inputs is None:  # pragma: no cover - 变体保证过
                raise GradientError("encoder_decoder 变体应当给出源序列的梯度。")
            analytic_matrix = analytic.grad_source_inputs
        gap = max(
            (
                abs(a - b)
                for left, right in zip(analytic_matrix, reference, strict=True)
                for a, b in zip(left, right, strict=True)
            ),
            default=0.0,
        )
        relative = relative_matrix_error(analytic_matrix, reference)
        outcomes.append(
            GradientOutcome(
                stream=stream,
                analytic_norm=max_absolute(analytic_matrix),
                numerical_norm=max_absolute(reference),
                max_gap=gap,
                relative_gap=relative,
                samples=matrix_shape(reference)[0] * matrix_shape(reference)[1],
                passed=relative <= tolerance,
            )
        )
    return GradientReport(
        variant=params.variant,
        tolerance=tolerance,
        step=step,
        outcomes=tuple(outcomes),
        analytic=analytic,
        notes=(
            f"中心差分：{len(streams)} 条流各逐分量算一次，步长 {step}",
            "两条流都要比：dEncoder 要累加、之后还要穿过整条编码器链才回到源序列",
            f"激活用 {activation}：ReLU 在 0 处不可导，中心差分会踩到折点",
        ),
    )


def check_all(
    params: VariantParameters,
    inputs: Matrix,
    target: Matrix,
    *,
    source: Matrix | None = None,
    pads: Any = None,
    **kwargs: Any,
) -> tuple[PropertyReport, GradientReport]:
    """六条性质 + 梯度校验一次跑完（**实验脚本的入口**）."""
    properties = check_properties(params, inputs, source=source, pads=pads, **kwargs)
    gradients = check_variant_gradients(
        params, inputs, target, source=source, **kwargs
    )
    return properties, gradients


__all__ = [
    "GRADIENT_TOLERANCE",
    "NUMERICAL_STEP",
    "GradientOutcome",
    "GradientReport",
    "PropertyOutcome",
    "PropertyReport",
    "check_all",
    "check_cross_spans_all_sources",
    "check_deterministic",
    "check_properties",
    "check_reads_every_position",
    "check_reads_only_past",
    "check_shape_preserved",
    "check_variant_gradients",
    "check_weights_are_zero_where_masked",
    "numerical_stream_gradient",
]
