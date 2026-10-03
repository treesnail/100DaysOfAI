"""``transformer_stack`` 的**深度实验**：逐层梯度衰减因子与深度无关（day080 / M7-D5）.

## 问题

day079 问的是“**堆到最底层**还剩多少梯度”，答案是一个数（最底层的范数）。
今天链已经被建起来了，于是可以问一个更细的问题：

```text
每经过一层，‖∂loss/∂x‖ 平均乘上多少倍？  而这个“每层倍数”随**总深度**变不变？
```

这个问题之所以值得单独量，是因为**一个与深度无关的每层因子**意味着
“梯度按几何级数衰减，而指数的底与 N 无关”——于是“能不能训 100 层”
变成一个可以外推的问题；而如果每层因子本身随深度变化，外推就不成立。

## 两个变体（一次只改一个旋钮）

```text
residual   y = x + F(LN(x))     现代实现的主流（残差开）
bare       y = F(LN(x))         **把那一项 +x 关掉**（同一条代码路径）
```

## 三个读数，以及它们**各自**回答什么

```text
overall_ratio     ‖dx_0‖ / ‖dy_N‖        整条链把 loss 梯度放大/缩小了多少倍
mean_step_ratio   ‖dx_{i+1}‖ / ‖dx_i‖ 的几何平均     “每层乘上多少”的单一数字
output_drift      ‖y − x‖ / ‖x‖          前向把输入搬离了多远（一个前向读数）
```

两个梯度读数**不是同一个量**，而它们之间有一条精确的恒等式：

```text
mean_step_ratio ^ (N − 1)  ==  ‖dx_{N−1}‖ / ‖dx_0‖      ← 链**内部**首尾之比
overall_ratio              ==  ‖dx_0‖   / ‖dy_N‖        ← 链的入口与损失之间
```

样本上（4 层）实测：``mean_step_ratio = 0.894712``，``0.894712³ = 0.716225``
而 ``‖dx_3‖/‖dx_0‖ = 0.716225``——逐位对上（第 10 章）。这条恒等式很重要，
因为第一版文档把 ``overall_ratio`` 也写成了“各层 step 相乘”，
而那只对 ``‖dx_{N−1}‖/‖dx_0‖`` 成立：``dx`` 序列有 N 个数、只有 N−1 个比值，
而 ``dy_N`` **不在那个序列里**。

``mean_step_ratio`` 用**几何平均**而不是算术平均：比值是相乘的东西，
取 N−1 次方正好回到链内首尾之比。算术平均会把 ``3.0`` 与 ``0.33``
平均成 ``1.66``——那是一个没有任何含义的数。

## 这一课**不回答**什么

```text
它回答      “每层的梯度倍数是多少、它与深度有没有关系”（一个**数值**问题）
它不回答    “这摞块能训多深”——那要看损失曲线、学习率与数据（day081 接手）
```

而且 ``overall_ratio`` 大于 1 **不一定是好事**：那同样可能是“梯度在放大”。
这一条读数只说明**那条路没有被掐断**。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from smart_research_agent.encoder_decoder.types import (
    ACTIVATION_RELU,
    NORM_PRE,
)
from smart_research_agent.math_foundations.types import Matrix, matrix_shape, validate_matrix
from smart_research_agent.transformer_core.layers import mse_gradient
from smart_research_agent.transformer_stack.errors import NumericError, ParameterError
from smart_research_agent.transformer_stack.layers import (
    DEFAULT_FFN_RATIO,
    DEFAULT_HIDDEN,
    DEFAULT_TOKENS,
    make_shape,
    make_stack_parameters,
    stack_loss_gradient,
)
from smart_research_agent.transformer_stack.types import (
    DEFAULT_INIT_SCALE,
    StackGradients,
    StackShape,
    _checked_layers,
    _checked_scale,
    frobenius,
)
from smart_research_agent.transformer_stack.verify import relative_gradient_change

#: 残差开的变体（现代实现的主流）.
VARIANT_RESIDUAL = "residual"
#: 把 ``+x`` 关掉的变体（同一条代码路径，只改一个布尔量）.
VARIANT_BARE = "bare"

STUDY_VARIANTS: tuple[str, ...] = (VARIANT_RESIDUAL, VARIANT_BARE)

VARIANT_DESCRIPTIONS: dict[str, str] = {
    VARIANT_RESIDUAL: "y = x + F(LN(x))：残差**开**（现代实现的主流）",
    VARIANT_BARE: "y = F(LN(x))：把那一项 +x **关掉**（同一条代码路径）",
}

#: 默认的深度序列（与 day079 的深度实验同序列，便于跨天对照）.
DEFAULT_STUDY_DEPTHS: tuple[int, ...] = (1, 2, 3, 4, 6, 8)


@dataclass(frozen=True)
class StackRow:
    """一个变体在一个深度上的一行读数."""

    variant: str
    layers: int
    input_gradient_norm: float
    output_gradient_norm: float
    overall_ratio: float
    mean_step_ratio: float
    output_drift: float
    parameter_count: int

    def __post_init__(self) -> None:
        if self.variant not in STUDY_VARIANTS:
            raise ParameterError(f"未知的变体名 {self.variant!r}。")
        for name in (
            "input_gradient_norm",
            "output_gradient_norm",
            "overall_ratio",
            "mean_step_ratio",
            "output_drift",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise NumericError(f"{name} 必须是有限数，收到 {value!r}。")
        if self.layers < 1:
            raise ParameterError(f"层数必须 >= 1，收到 {self.layers}。")
        if self.parameter_count < 1:
            raise ParameterError(f"参数量必须 >= 1，收到 {self.parameter_count}。")

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "variant": self.variant,
            "layers": self.layers,
            "input_gradient_norm": self.input_gradient_norm,
            "output_gradient_norm": self.output_gradient_norm,
            "overall_ratio": self.overall_ratio,
            "mean_step_ratio": self.mean_step_ratio,
            "output_drift": self.output_drift,
            "parameter_count": self.parameter_count,
        }

    def summary_line(self) -> str:
        """一行说明."""
        return (
            f"{self.variant:<9} {self.layers} 层 | 整体比 {self.overall_ratio:.6f} | "
            f"每层倍数 {self.mean_step_ratio:.6f} | 输出漂移 {self.output_drift:.6f} | "
            f"参数 {self.parameter_count}"
        )


@dataclass(frozen=True)
class StackStudy:
    """两个变体在一条深度序列上的对照（**这一课的值钱结论落在这里**）."""

    hidden: int
    ffn: int
    tokens: int
    depths: tuple[int, ...]
    rows: tuple[StackRow, ...]
    seed: int
    scale: float
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        resolved = tuple(self.rows)
        if not resolved:
            raise ParameterError("深度实验至少要有一行读数。")
        if {row.variant for row in resolved} != set(STUDY_VARIANTS):
            raise NumericError(
                "两个变体必须齐备：缺少的对照会让'残差有用'这句话失去参照物。"
            )
        if sorted(set(self.depths)) != list(self.depths) or not self.depths:
            raise ParameterError(f"深度序列必须严格递增且非空，收到 {self.depths}。")
        object.__setattr__(self, "rows", resolved)
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def deepest(self) -> int:
        """最深的那一层."""
        return max(row.layers for row in self.rows)

    def ratios(self, variant: str) -> tuple[float, ...]:
        """某个变体在各深度上的“整体比”（按下标顺序）."""
        if variant not in STUDY_VARIANTS:
            raise ParameterError(f"未知的变体名 {variant!r}：可选 {', '.join(STUDY_VARIANTS)}。")
        return tuple(row.overall_ratio for row in self.rows if row.variant == variant)

    def step_ratios(self, variant: str) -> tuple[float, ...]:
        """某个变体在各深度上的“每层倍数”（按下标顺序）."""
        if variant not in STUDY_VARIANTS:
            raise ParameterError(f"未知的变体名 {variant!r}：可选 {', '.join(STUDY_VARIANTS)}。")
        return tuple(row.mean_step_ratio for row in self.rows if row.variant == variant)

    @property
    def step_ratio_spread(self) -> float:
        """残差开时“每层倍数”在整条深度序列上的**极差**（与深度无关的判据）.

        它的判据是“最深的那个每层倍数与最浅的那个相差不到一倍”
        （``max / min < 2``）——一个与深度无关的因子应该落在这条线内。

        深度 1 那一行没有相邻对，于是 :func:`geometric_mean` 按定义返回 ``1.0``
        （乘法的单位元）。它作为一个**真实读数**被算进极差里：
        把它悄悄排除掉会让“与深度无关”这句话的检验少一档。
        """
        values = self.step_ratios(VARIANT_RESIDUAL)
        low = min(values)
        high = max(values)
        if low <= 0.0:
            return math.inf
        return high / low

    @property
    def verdict_ok(self) -> bool:
        """判决：最深那一层上，**关掉残差**的变体衰减得明显更快.

        判据是“快一个数量级以上”（``bare`` 的整体比 < ``residual`` 的 1/10），
        而且残差开时的“每层倍数”在整条深度序列上几乎不变（``max/min < 2``）。
        """
        bare = self.ratios(VARIANT_BARE)[-1]
        residual = self.ratios(VARIANT_RESIDUAL)[-1]
        return bare < residual / 10.0 and self.step_ratio_spread < 2.0

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "hidden": self.hidden,
            "ffn": self.ffn,
            "tokens": self.tokens,
            "depths": list(self.depths),
            "seed": self.seed,
            "scale": self.scale,
            "verdict_ok": self.verdict_ok,
            "step_ratio_spread": self.step_ratio_spread,
            "rows": [row.to_dict() for row in self.rows],
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明."""
        parts = []
        for variant in STUDY_VARIANTS:
            parts.append(f"{variant} {self.ratios(variant)[-1]:.2e}")
        return (
            f"堆叠实验：{self.depths[0]}~{self.deepest} 层 | 最深一层整体比："
            + "、".join(parts)
            + f" | 每层倍数极差 {self.step_ratio_spread:.4f}"
        )

    def table_lines(self) -> tuple[str, ...]:
        """把两个变体排成一张表（每一行一个变体 × 深度）."""
        header = (
            f"  {'变体':<9} | {'层数':>4} | {'整体比':>12} | {'每层倍数':>11} | "
            f"{'输出漂移':>10} | {'参数量':>7}"
        )
        lines = [header, "  " + "-" * 70]
        for row in self.rows:
            lines.append(
                f"  {row.variant:<9} | {row.layers:>4} | {row.overall_ratio:>12.6f} | "
                f"{row.mean_step_ratio:>11.6f} | {row.output_drift:>10.6f} | "
                f"{row.parameter_count:>7}"
            )
        return tuple(lines)


def geometric_mean(values: Sequence[float]) -> float:
    """几何平均（空序列返回 ``1.0``：**乘法的单位元**，而不是 0）.

    比值的“平均”必须用几何平均，因为比值是**相乘**的：
    N−1 个相邻比值的乘积恰好是链内首尾之比 ``‖dx_{N−1}‖/‖dx_0‖``，
    因此把它们的几何平均取 ``N−1`` 次方就回到那个比值。用算术平均会把
    ``3.0`` 与 ``0.33`` 平均成 ``1.66``——一个没有含义的数。
    """
    resolved = tuple(float(item) for item in values)
    if not resolved:
        return 1.0
    for value in resolved:
        if not math.isfinite(value) or value < 0.0:
            raise NumericError(f"几何平均的每一项都必须是非负有限数，收到 {value!r}。")
    total = math.fsum(math.log(value) for value in resolved if value > 0.0)
    zeros = sum(1 for value in resolved if value == 0.0)
    if zeros:
        return 0.0
    return math.exp(total / len(resolved))


def _row_for(
    variant: str,
    layers: int,
    *,
    hidden: int,
    ffn_ratio: int,
    tokens: int,
    seed: int,
    scale: float,
    activation: str,
) -> StackRow:
    """跑一次“某个深度 + 某个变体”，返回一行读数."""
    shape = make_shape(
        layers=layers, hidden=hidden, tokens=tokens, ffn_ratio=ffn_ratio
    )
    params = make_stack_parameters(shape, seed=seed, scale=scale)
    inputs = _sample_inputs(tokens, hidden)
    target = _sample_target(tokens, hidden)
    use_residual = variant == VARIANT_RESIDUAL
    forward, grads = stack_loss_gradient(
        params,
        inputs,
        target,
        placement=NORM_PRE,
        use_residual=use_residual,
        activation=activation,
    )
    output_norm = frobenius(mse_gradient(forward.output, target))
    input_norm = frobenius(grads.grad_inputs)
    ratios = relative_gradient_change(grads.norms())
    drift = _output_drift(forward.output, inputs)
    return StackRow(
        variant=variant,
        layers=layers,
        input_gradient_norm=input_norm,
        output_gradient_norm=output_norm,
        overall_ratio=input_norm / output_norm if output_norm > 0 else 0.0,
        mean_step_ratio=geometric_mean(ratios),
        output_drift=drift,
        parameter_count=shape.total_parameter_count,
    )


def stack_study(
    *,
    depths: Sequence[int] = DEFAULT_STUDY_DEPTHS,
    hidden: int = DEFAULT_HIDDEN,
    ffn_ratio: int = DEFAULT_FFN_RATIO,
    tokens: int = DEFAULT_TOKENS,
    seed: int = 7,
    scale: float = DEFAULT_INIT_SCALE,
    activation: str = ACTIVATION_RELU,
) -> StackStudy:
    """跑一遍堆叠实验：两个变体 × 一条深度序列.

    两个变体共用**同一颗种子**与**同一个目标**，因此表里的差别只来自
    残差那一个开关；而同一个变体在不同深度上的差别只来自**层数**——
    第 i 层的参数在所有深度上都相同（种子是 ``seed + i*13``）。
    """
    resolved_depths = tuple(depths)
    if not resolved_depths:
        raise ParameterError("深度序列不能为空。")
    for layers in resolved_depths:
        _checked_layers(layers)
    if sorted(set(resolved_depths)) != list(resolved_depths):
        raise ParameterError(f"深度序列必须严格递增且不重复，收到 {resolved_depths}。")
    resolved_scale = _checked_scale(scale)
    rows: list[StackRow] = []
    for variant in STUDY_VARIANTS:
        for layers in resolved_depths:
            rows.append(
                _row_for(
                    variant,
                    layers,
                    hidden=hidden,
                    ffn_ratio=ffn_ratio,
                    tokens=tokens,
                    seed=seed,
                    scale=resolved_scale,
                    activation=activation,
                )
            )
    return StackStudy(
        hidden=hidden,
        ffn=hidden * ffn_ratio,
        tokens=tokens,
        depths=resolved_depths,
        rows=tuple(rows),
        seed=seed,
        scale=resolved_scale,
        notes=(
            "两个变体共用同一颗种子与同一个目标，只差残差那一个开关",
            "每层倍数取几何平均：比值相乘，几何平均的 N 次方正好回到整体比",
            "整体比大于 1 也可能是放大而不是好事——这一条读数只说明那条路没被掐断",
        ),
    )


def _sample_inputs(tokens: int, hidden: int) -> Matrix:
    """样本输入：确定性的小矩阵（与演示脚本、测试共用同一组数）."""
    if tokens < 1 or hidden < 1:
        raise ParameterError(f"样本形状必须为正，收到 ({tokens}, {hidden})。")
    return tuple(
        tuple(0.1 * (row + 1) - 0.05 * (column + 1) for column in range(hidden))
        for row in range(tokens)
    )


def _sample_target(tokens: int, hidden: int) -> Matrix:
    """样本目标：确定性的小矩阵（**固定**，两个变体共用同一个）."""
    if tokens < 1 or hidden < 1:
        raise ParameterError(f"样本形状必须为正，收到 ({tokens}, {hidden})。")
    return tuple(
        tuple(0.2 - 0.03 * (row * hidden + column) for column in range(hidden))
        for row in range(tokens)
    )


def _output_drift(output: Matrix, inputs: Matrix) -> float:
    """``‖y − x‖ / ‖x‖``（前向把输入搬离了多远；``‖x‖ = 0`` 时返回 ``0.0``）."""
    checked_output = validate_matrix(output, name="output")
    checked_inputs = validate_matrix(inputs, name="inputs")
    if matrix_shape(checked_output) != matrix_shape(checked_inputs):
        raise ParameterError(
            f"输出 {matrix_shape(checked_output)} 与输入 {matrix_shape(checked_inputs)} "
            "不同形：漂移只在同形时才有定义。"
        )
    delta = tuple(
        tuple(left - right for left, right in zip(row_out, row_in, strict=True))
        for row_out, row_in in zip(checked_output, checked_inputs, strict=True)
    )
    reference = frobenius(checked_inputs)
    if reference == 0.0:
        return 0.0
    return frobenius(delta) / reference


def gradient_norms_of(params_norms: StackGradients) -> tuple[float, ...]:
    """逐层梯度的范数（转发 ``StackGradients.norms``，供演示脚本使用）."""
    if not isinstance(params_norms, StackGradients):
        raise ParameterError(
            f"grads 必须是 StackGradients，收到 {type(params_norms).__name__}。"
        )
    return params_norms.norms()


def shape_of_study(study: StackStudy, layers: int) -> StackShape:
    """实验里某个深度对应的形状（演示脚本按深度重放时用）."""
    return make_shape(
        layers=layers, hidden=study.hidden, tokens=study.tokens, ffn_ratio=study.ffn // study.hidden
    )


__all__ = [
    "DEFAULT_STUDY_DEPTHS",
    "STUDY_VARIANTS",
    "VARIANT_BARE",
    "VARIANT_DESCRIPTIONS",
    "VARIANT_RESIDUAL",
    "StackRow",
    "StackStudy",
    "geometric_mean",
    "gradient_norms_of",
    "shape_of_study",
    "stack_study",
]
