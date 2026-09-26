"""``encoder_decoder`` 的梯度校验与性质检查（day079 / M7-D4）.

```text
九项块级校验      新增的八块 + 输入   norm1_gamma … norm2_beta → inputs
六项交叉校验      四个投影 + **两路输入**
两项解码器校验    dDecoder / dEncoder（三段子层已被逐项验过，这里只查接线）
```

## 一处口径必须写下来：注意力那一层**在块内部**被造出来

```text
pre-LN 时   注意力作用在 LN(x) 上 ⇒ γ₁/β₁ 会改变注意力的输出
```

因此“把一份算好的注意力账从外面传进来”是**不行**的：数值侧扰动 γ₁ 时那一份账不会变，
而解析侧（用注意力自己的 ``grad_inputs``）却算进了那条链——两边算的不是同一个函数。
本模块因此统一走 :func:`layers.block_attention`：**同一个构造函数**，
由它按摆放位置决定注意力作用在什么上面。

第一版踩到这个坑时，表现是“``norm1_gamma`` 那一项差 1e-1 量级”——
**最像 bug 的量级**，而根因是“块没有拥有它的第一个子层”。

## 八条性质里最值钱的一条（第 8 条）

```text
交叉注意力**不能**加因果掩码 —— 而加错了在 n_tgt == n_src 时**不报错**
```

本模块不只断言“它会拒绝”，还**量出加错之后的偏差**。这个读数是这一条性质的全部说服力。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from smart_research_agent.encoder_decoder.errors import (
    AssemblyError,
    GradientError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.encoder_decoder.layers import (
    block_attention,
    cross_attention,
    cross_attention_backward,
    decoder_block,
    decoder_block_backward,
    encoder_block,
    encoder_block_backward,
    feed_forward,
    layer_norm,
)
from smart_research_agent.encoder_decoder.types import (
    ACTIVATION_RELU,
    BLOCK_GRADIENT_FORMULAS,
    BLOCK_GRADIENT_TARGETS,
    CROSS_GRADIENT_FORMULAS,
    CROSS_GRADIENT_TARGETS,
    ENCODER_DECODER_PROPERTIES,
    NORM_PRE,
    PROPERTY_CROSS_NOT_CAUSAL,
    PROPERTY_DESCRIPTIONS,
    PROPERTY_FFN_POSITION_WISE,
    PROPERTY_NORM_ROW_INDEPENDENT,
    PROPERTY_RESIDUAL_IDENTITY,
    PROPERTY_RESIDUAL_UNIT_PATH,
    PROPERTY_ROWS_STANDARDIZED,
    PROPERTY_SCALE_EQUIVARIANT,
    PROPERTY_SHIFT_INVARIANT,
    BlockGradients,
    BlockParameters,
    CrossGradients,
    CrossParameters,
    FFNWeights,
    NormCache,
    _checked_activation,
    _checked_placement,
)
from smart_research_agent.math_foundations.calculus import DEFAULT_STEP, gradient
from smart_research_agent.math_foundations.optim import flatten_matrices, unflatten_matrices
from smart_research_agent.math_foundations.types import (
    Matrix,
    Vector,
    matrix_shape,
    validate_matrix,
)
from smart_research_agent.transformer_core.layers import (
    masked_softmax_rows,
    mean_squared_error,
    mse_gradient,
    self_attention,
)
from smart_research_agent.transformer_core.types import (
    AttentionForward,
    AttentionParams,
    relative_matrix_error,
)
from smart_research_agent.transformer_core.verify import GRADIENT_TOLERANCE as CORE_TOLERANCE

#: 梯度校验的容差（**与 day075~078 同值**，导入期把五者钉成相等）.
GRADIENT_TOLERANCE = 1e-6

if GRADIENT_TOLERANCE != CORE_TOLERANCE:  # pragma: no cover - 只在有人改常量时触发
    raise NumericError(
        f"本课的梯度容差 {GRADIENT_TOLERANCE} 与 day075 的 {CORE_TOLERANCE} 不一致。"
    )

#: 报告的种类（决定取哪一张公式表）.
KIND_BLOCK = "block"
KIND_CROSS = "cross"
KIND_DECODER = "decoder"

KINDS: tuple[str, ...] = (KIND_BLOCK, KIND_CROSS, KIND_DECODER)

#: 每一类报告的名单（**名单是构造参数**：缺项检查只有一处实现）.
_TARGETS: dict[str, tuple[str, ...]] = {
    KIND_BLOCK: BLOCK_GRADIENT_TARGETS,
    KIND_CROSS: CROSS_GRADIENT_TARGETS,
    KIND_DECODER: ("decoder_inputs", "encoder_outputs"),
}


def permute_rows(matrix: Matrix, permutation: Sequence[int]) -> Matrix:
    """按置换把行重排（``permutation[i]`` 是新第 i 行的来源）."""
    checked = validate_matrix(matrix, name="matrix")
    rows = matrix_shape(checked)[0]
    if len(permutation) != rows or sorted(permutation) != list(range(rows)):
        raise ParameterError(f"置换 {tuple(permutation)} 不是 0..{rows - 1} 的一个排列。")
    return tuple(checked[index] for index in permutation)


def reverse_permutation(rows: int) -> tuple[int, ...]:
    """倒序置换（“最不像恒等”的那个）."""
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ParameterError(f"rows 必须是 >= 1 的整数，收到 {rows!r}。")
    return tuple(reversed(range(rows)))


def _scaled_error(left: float, right: float) -> float:
    """逐点相对误差（分母 ``max(1, |左|, |右|)``，与 day075~078 同口径）."""
    if not (math.isfinite(left) and math.isfinite(right)):
        return math.inf
    return abs(left - right) / max(1.0, abs(left), abs(right))


def _checked_tolerance(tolerance: Any) -> float:
    """校验容差（正的有限数）."""
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise ParameterError(f"tolerance 必须是数，收到 {tolerance!r}。")
    resolved = float(tolerance)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ParameterError(f"tolerance 必须是正的有限数，收到 {tolerance!r}。")
    return resolved


def _compare(target: str, analytic: Matrix, numeric: Matrix) -> tuple[float, float, int]:
    """比较两块梯度，返回 ``(最大绝对误差, 最大相对误差, 逐点数)``."""
    left = validate_matrix(analytic, name="analytic")
    right = validate_matrix(numeric, name="numeric")
    if matrix_shape(left) != matrix_shape(right):
        raise ShapeError(
            f"{target}：解析梯度 {matrix_shape(left)} 与数值梯度 {matrix_shape(right)} 不一致。"
        )
    worst_absolute = 0.0
    worst_scaled = 0.0
    points = 0
    for left_row, right_row in zip(left, right, strict=True):
        for a, b in zip(left_row, right_row, strict=True):
            worst_absolute = max(worst_absolute, abs(a - b))
            worst_scaled = max(worst_scaled, _scaled_error(a, b))
            points += 1
    return worst_absolute, worst_scaled, points


@dataclass(frozen=True)
class GradientOutcome:
    """一项梯度校验的读数（**通用形状**：名字由名单给出）."""

    target: str
    max_absolute_error: float
    max_scaled_error: float
    tolerance: float
    compared_points: int
    message: str = ""

    def __post_init__(self) -> None:
        for name in ("max_absolute_error", "max_scaled_error"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or math.isnan(value) or value < 0:
                raise NumericError(f"{name} 必须是非负的数，收到 {value!r}。")
        _checked_tolerance(self.tolerance)
        if not isinstance(self.compared_points, int) or self.compared_points < 0:
            raise NumericError(f"compared_points 必须是非负整数，收到 {self.compared_points!r}。")
        object.__setattr__(self, "message", str(self.message))

    @property
    def passed(self) -> bool:
        """这一项是否通过."""
        return self.max_scaled_error <= self.tolerance

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "target": self.target,
            "passed": self.passed,
            "max_absolute_error": self.max_absolute_error,
            "max_scaled_error": self.max_scaled_error,
            "tolerance": self.tolerance,
            "compared_points": self.compared_points,
            "message": self.message,
        }

    def summary_line(self) -> str:
        """一行说明：``[ok] norm1_gamma 最大相对误差 3.1e-11（比较 6 个逐点，容差 1e-06）``."""
        mark = "ok" if self.passed else "!!"
        return (
            f"[{mark}] {self.target:<14} 最大相对误差 {self.max_scaled_error:.2e}"
            f"（绝对 {self.max_absolute_error:.2e}，比较 {self.compared_points} 个逐点，"
            f"容差 {self.tolerance:g}）"
        )


@dataclass(frozen=True)
class GradientReport:
    """一份按名单校验的梯度报告（**它属于控制流**：不通过就该停下来）."""

    kind: str
    outcomes: tuple[GradientOutcome, ...]
    tolerance: float = GRADIENT_TOLERANCE
    step: float = DEFAULT_STEP
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ParameterError(f"未知的报告种类 {self.kind!r}：可选 {', '.join(KINDS)}。")
        resolved = tuple(self.outcomes)
        if not resolved:
            raise ParameterError("梯度报告不能为空。")
        expected = set(_TARGETS[self.kind])
        names = {item.target for item in resolved}
        if names != expected:
            raise NumericError(
                f"{self.kind} 报告的校验项与名单不一致：缺 {sorted(expected - names)}、"
                f"多 {sorted(names - expected)}——少一项的报告在'全绿'时看起来与完整报告一样。"
            )
        _checked_tolerance(self.tolerance)
        object.__setattr__(self, "outcomes", resolved)
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def targets(self) -> tuple[str, ...]:
        """这一份报告的名单."""
        return _TARGETS[self.kind]

    @property
    def formulas(self) -> dict[str, str]:
        """这一份报告的公式表（解码器那一份只查接线，因此没有逐项公式）."""
        if self.kind == KIND_BLOCK:
            return dict(BLOCK_GRADIENT_FORMULAS)
        if self.kind == KIND_CROSS:
            return dict(CROSS_GRADIENT_FORMULAS)
        return {}

    @property
    def ok(self) -> bool:
        """全部通过."""
        return all(item.passed for item in self.outcomes)

    @property
    def failures(self) -> tuple[GradientOutcome, ...]:
        """没通过的那些项."""
        return tuple(item for item in self.outcomes if not item.passed)

    @property
    def worst_scaled_error(self) -> float:
        """最大的相对误差."""
        return max(item.max_scaled_error for item in self.outcomes)

    def raise_if_failed(self) -> None:
        """有任意一项不通过就抛 :class:`GradientError`."""
        if not self.ok:
            detail = "；".join(
                f"{item.target}={item.max_scaled_error:.2e}" for item in self.failures
            )
            raise GradientError(
                f"{self.kind} 梯度校验未通过（容差 {self.tolerance:g}）：{detail}——"
                "先确认两边算的是**同一个函数**（损失口径 / placement / 残差开关 / 因果开关），"
                "再怀疑推导。"
            )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "kind": self.kind,
            "ok": self.ok,
            "tolerance": self.tolerance,
            "step": self.step,
            "worst_scaled_error": self.worst_scaled_error,
            "outcomes": [item.to_dict() for item in self.outcomes],
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``block 梯度校验 9 项：通过 9、失败 0 | 最大相对误差 3.4e-11``."""
        passed = sum(1 for item in self.outcomes if item.passed)
        return (
            f"{self.kind} 梯度校验 {len(self.outcomes)} 项：通过 {passed}、失败 "
            f"{len(self.outcomes) - passed} | 最大相对误差 {self.worst_scaled_error:.2e}"
            f"（容差 {self.tolerance:g}）"
        )


def _report(
    kind: str,
    analytic: dict[str, Matrix],
    numeric: dict[str, Matrix],
    *,
    tolerance: float,
    step: float,
    notes: tuple[str, ...],
) -> GradientReport:
    """按名单逐项对照并组装报告（**两侧的名单与顺序都来自 ``_TARGETS``**）."""
    outcomes: list[GradientOutcome] = []
    for name in _TARGETS[kind]:
        worst_absolute, worst_scaled, points = _compare(
            name, analytic[name], numeric[name]
        )
        message = ""
        if worst_scaled > tolerance:
            message = (
                f"{name} 的解析梯度与数值差分差 {worst_scaled:.2e}（容差 {tolerance:g}）："
                "先确认两边算的是同一个函数，再怀疑推导。"
            )
        outcomes.append(
            GradientOutcome(
                target=name,
                max_absolute_error=worst_absolute,
                max_scaled_error=worst_scaled,
                tolerance=tolerance,
                compared_points=points,
                message=message,
            )
        )
    return GradientReport(
        kind=kind, outcomes=tuple(outcomes), tolerance=tolerance, step=step, notes=notes
    )


# ---------------------------------------------------------------------- 块级


def block_loss(
    params: BlockParameters,
    inputs: Matrix,
    attention_params: AttentionParams,
    target: Matrix,
    *,
    placement: str = NORM_PRE,
    use_residual: bool = True,
    activation: str = ACTIVATION_RELU,
    causal: bool = False,
) -> float:
    """块级目标函数：``θ ↦ mean_squared_error(block(x), target)``（**数值与解析共用**）.

    ``attention_params`` 在这里被用来**造**注意力（见 :func:`layers.block_attention`）——
    这一点是这一课与前几天最大的差别：块的第一子层也在“被检查的函数”里面。
    """
    resolved_placement = _checked_placement(placement)
    attention = block_attention(
        params, inputs, attention_params, placement=resolved_placement, causal=causal
    )
    forward = encoder_block(
        params,
        inputs,
        attention,
        placement=resolved_placement,
        use_residual=use_residual,
        activation=activation,
    )
    return mean_squared_error(forward.output, target)


def numerical_block_gradients(
    params: BlockParameters,
    inputs: Matrix,
    attention_params: AttentionParams,
    target: Matrix,
    *,
    placement: str = NORM_PRE,
    use_residual: bool = True,
    activation: str = ACTIVATION_RELU,
    causal: bool = False,
    step: float = DEFAULT_STEP,
) -> BlockGradients:
    """用中心差分算块级九块梯度（八块参数一趟 + 输入一趟）."""
    resolved_placement = _checked_placement(placement)
    resolved_activation = _checked_activation(activation)
    flat, shapes = params.flatten()

    def objective(theta: Vector) -> float:
        """给定压平后的参数，返回块级损失（**每一步都重新造注意力**）."""
        return block_loss(
            BlockParameters.unflatten(theta, shapes),
            inputs,
            attention_params,
            target,
            placement=resolved_placement,
            use_residual=use_residual,
            activation=resolved_activation,
            causal=causal,
        )

    numeric_flat = gradient(objective, flat, step=step)
    parts = unflatten_matrices(numeric_flat, shapes)
    checked_inputs = validate_matrix(inputs, name="inputs")
    flat_inputs, input_shapes = flatten_matrices((checked_inputs,))

    def input_objective(theta: Vector) -> float:
        """给定压平后的输入，返回块级损失."""
        rebuilt = unflatten_matrices(theta, input_shapes)[0]
        return block_loss(
            params,
            rebuilt,
            attention_params,
            target,
            placement=resolved_placement,
            use_residual=use_residual,
            activation=resolved_activation,
            causal=causal,
        )

    grad_inputs = unflatten_matrices(
        gradient(input_objective, flat_inputs, step=step), input_shapes
    )[0]
    return BlockGradients(
        grad_norm1_gamma=parts[0][0],
        grad_norm1_beta=parts[1][0],
        grad_ffn_w_in=parts[2],
        grad_ffn_b_in=parts[3][0],
        grad_ffn_w_out=parts[4],
        grad_ffn_b_out=parts[5][0],
        grad_norm2_gamma=parts[6][0],
        grad_norm2_beta=parts[7][0],
        grad_inputs=grad_inputs,
    )


def check_block_gradients(
    params: BlockParameters,
    inputs: Matrix,
    attention_params: AttentionParams,
    target: Matrix,
    *,
    placement: str = NORM_PRE,
    use_residual: bool = True,
    activation: str = ACTIVATION_RELU,
    causal: bool = False,
    step: float = DEFAULT_STEP,
    tolerance: float = GRADIENT_TOLERANCE,
) -> GradientReport:
    """九项逐点对照（解析 vs 数值）.

    **四处口径必须两边一致**（前三条是 day076 的教训，第四条是本课新增的）：

    ```text
    损失口径      两侧都用 mean_squared_error（同一个函数）
    placement     两侧都是同一个 pre/post
    use_residual  两侧都是同一个残差开关
    causal        两侧都是同一个因果开关——**而它会影响注意力的输出**
    ```

    它们全部由**同一组参数**传给两侧，而且两侧都走同一个
    :func:`layers.block_attention` 与 :func:`layers.encoder_block`，
    因此“忘了传”在结构上不可能发生。
    """
    resolved_tolerance = _checked_tolerance(tolerance)
    resolved_placement = _checked_placement(placement)
    attention = block_attention(
        params, inputs, attention_params, placement=resolved_placement, causal=causal
    )
    forward = encoder_block(
        params,
        inputs,
        attention,
        placement=resolved_placement,
        use_residual=use_residual,
        activation=activation,
    )
    analytic = encoder_block_backward(
        forward, params, mse_gradient(forward.output, target), activation=activation
    )
    numeric = numerical_block_gradients(
        params,
        inputs,
        attention_params,
        target,
        placement=resolved_placement,
        use_residual=use_residual,
        activation=activation,
        causal=causal,
        step=step,
    )
    return _report(
        KIND_BLOCK,
        analytic.as_dict(),
        numeric.as_dict(),
        tolerance=resolved_tolerance,
        step=step,
        notes=(
            f"placement={placement}、残差 {'开' if use_residual else '关'}、"
            f"因果 {'开' if causal else '关'}、激活 {activation}——两侧完全一致",
            "两个偏置与两个 LN 的 γ/β 都是**按行相加**的，因此它们的比较点数等于隐藏维",
        ),
    )


# ---------------------------------------------------------------------- 交叉注意力


def cross_loss(
    params: CrossParameters,
    target_inputs: Matrix,
    source_inputs: Matrix,
    target: Matrix,
) -> float:
    """交叉注意力的目标函数（数值与解析共用）."""
    forward = cross_attention(params, target_inputs, source_inputs)
    return mean_squared_error(forward.output, target)


def numerical_cross_gradients(
    params: CrossParameters,
    target_inputs: Matrix,
    source_inputs: Matrix,
    target: Matrix,
    *,
    step: float = DEFAULT_STEP,
) -> CrossGradients:
    """用中心差分算交叉注意力的六块（四个投影 + 两路输入）."""
    flat, shapes = flatten_matrices(params.matrices())

    def objective(theta: Vector) -> float:
        """给定压平后的四个投影，返回损失."""
        parts = unflatten_matrices(theta, shapes)
        return cross_loss(
            CrossParameters(
                w_query=parts[0], w_key=parts[1], w_value=parts[2], w_output=parts[3]
            ),
            target_inputs,
            source_inputs,
            target,
        )

    parts = unflatten_matrices(gradient(objective, flat, step=step), shapes)
    target_shape = (matrix_shape(target_inputs),)
    source_shape = (matrix_shape(source_inputs),)
    grad_target = unflatten_matrices(
        gradient(
            lambda theta: cross_loss(
                params, unflatten_matrices(theta, target_shape)[0], source_inputs, target
            ),
            flatten_matrices((target_inputs,))[0],
            step=step,
        ),
        target_shape,
    )[0]
    grad_source = unflatten_matrices(
        gradient(
            lambda theta: cross_loss(
                params, target_inputs, unflatten_matrices(theta, source_shape)[0], target
            ),
            flatten_matrices((source_inputs,))[0],
            step=step,
        ),
        source_shape,
    )[0]
    return CrossGradients(
        grad_w_query=parts[0],
        grad_w_key=parts[1],
        grad_w_value=parts[2],
        grad_w_output=parts[3],
        grad_target_inputs=grad_target,
        grad_source_inputs=grad_source,
    )


def check_cross_gradients(
    params: CrossParameters,
    target_inputs: Matrix,
    source_inputs: Matrix,
    target: Matrix,
    *,
    step: float = DEFAULT_STEP,
    tolerance: float = GRADIENT_TOLERANCE,
) -> GradientReport:
    """六项逐点对照：四个投影 + **两路各自的输入**.

    最后两项是本课唯一“两路”的地方，而它们各自有一条要注意的地方：

    ```text
    target_inputs   只有一条链（dQ·W_q）
    source_inputs   **两条链之和**（dK·W_k + dV·W_v）——少一条不报错，只会让梯度偏小
    ```
    """
    resolved_tolerance = _checked_tolerance(tolerance)
    forward = cross_attention(params, target_inputs, source_inputs)
    analytic = cross_attention_backward(forward, mse_gradient(forward.output, target))
    numeric = numerical_cross_gradients(
        params, target_inputs, source_inputs, target, step=step
    )
    return _report(
        KIND_CROSS,
        analytic.as_dict(),
        numeric.as_dict(),
        tolerance=resolved_tolerance,
        step=step,
        notes=(
            "交叉注意力**没有掩码**：K/V 是另一路的全部位置",
            "source 那一侧拿到的是两条链之和，因此它的比较点数比 target 多",
        ),
    )


def check_decoder_input_gradients(
    self_attention_forward: Any,
    cross_params: CrossParameters,
    gammas: tuple[Vector, Vector, Vector],
    betas: tuple[Vector, Vector, Vector],
    ffn: FFNWeights,
    decoder_inputs: Matrix,
    encoder_outputs: Matrix,
    target: Matrix,
    *,
    use_residual: bool = True,
    activation: str = ACTIVATION_RELU,
    step: float = DEFAULT_STEP,
    tolerance: float = GRADIENT_TOLERANCE,
) -> GradientReport:
    """解码器块的**两路输入**梯度校验（其余八块走的是同一套 LN/FFN 反向）.

    解码器块的**新东西只有接线**：三段子层的前向/反向都在编码器块与交叉注意力里
    逐项验过（9 项 + 6 项），因此这一份报告只查接线：

    ```text
    decoder_inputs   要走通**三条**残差路（三个子层各一条）
    encoder_outputs  要出现在交叉注意力 source 那一路的终点——**编码器会收到梯度**
    ```

    一处必须写下来的边界（**本函数唯一的纪律**）：解码器块的自注意力是**从外面传进来的**，
    因此 :func:`layers.decoder_block` 自己**不是**一个关于 ``decoder_inputs`` 的纯函数——
    pre-LN 的契约是“注意力作用在 ``LN(x)`` 上”，而一份**算好的**注意力账不会随 ``x`` 变。
    这一点与编码器块刚好相反：那里由 :func:`layers.block_attention` 保证第一个子层在
    “被检查的函数”里面。

    本函数因此**按同一个契约把注意力重建在** ``LN(decoder_inputs)`` **上**
    （重建只用到传入 forward 的 ``params`` 与 ``causal``）。不重建的后果是
    "数值侧扰动的是一份**冻结**的注意力，而解析侧把 LN1 那条链也算进去了"——
    两边算的不是同一个函数，实测差 ``1e0`` 量级（**最像 bug 的那个读数**）。
    重建之后，本函数仍然**只查两路输入**，不查三个 LN 的 γ/β。
    """
    if not isinstance(self_attention_forward, AttentionForward):
        raise ParameterError(
            f"self_attention_forward 必须是 AttentionForward，收到 "
            f"{type(self_attention_forward).__name__}。"
        )
    resolved_tolerance = _checked_tolerance(tolerance)
    attention_params = self_attention_forward.params
    attention_causal = bool(self_attention_forward.causal)

    def build_attention(dec_in: Matrix) -> AttentionForward:
        """按 pre-LN 契约重建因果自注意力 ``Attn(LN(dec_in))``（与 block_attention 同一纪律）."""
        normed, _cache = layer_norm(dec_in, gamma=gammas[0], beta=betas[0])
        return self_attention(attention_params, normed, causal=attention_causal)

    def forward_call(dec_in: Matrix, enc_out: Matrix) -> tuple[Any, ...]:
        """给定两路输入，跑一遍解码器块的前向（第一个子层是 ``dec_in`` 的函数）."""
        return decoder_block(
            build_attention(dec_in),
            cross_params,
            gammas,
            betas,
            ffn,
            dec_in,
            enc_out,
            use_residual=use_residual,
            activation=activation,
        )

    output, cross_forward, caches, ffn_cache = forward_call(
        decoder_inputs, encoder_outputs
    )
    analytic = decoder_block_backward(
        build_attention(decoder_inputs),
        cross_forward,
        caches,
        ffn_cache,
        mse_gradient(output, target),
        use_residual=use_residual,
    )
    target_shape = (matrix_shape(decoder_inputs),)
    encoder_shape = (matrix_shape(encoder_outputs),)

    def redo(dec_in: Matrix, enc_out: Matrix) -> float:
        """给定两路输入，返回解码器块的损失."""
        return mean_squared_error(forward_call(dec_in, enc_out)[0], target)

    numeric_decoder = unflatten_matrices(
        gradient(
            lambda theta: redo(
                unflatten_matrices(theta, target_shape)[0], encoder_outputs
            ),
            flatten_matrices((decoder_inputs,))[0],
            step=step,
        ),
        target_shape,
    )[0]
    numeric_encoder = unflatten_matrices(
        gradient(
            lambda theta: redo(
                decoder_inputs, unflatten_matrices(theta, encoder_shape)[0]
            ),
            flatten_matrices((encoder_outputs,))[0],
            step=step,
        ),
        encoder_shape,
    )[0]
    analytic_blocks = {
        "decoder_inputs": analytic.grad_decoder_inputs,
        "encoder_outputs": analytic.grad_encoder_outputs,
    }
    numeric_blocks = {
        "decoder_inputs": numeric_decoder,
        "encoder_outputs": numeric_encoder,
    }
    return _report(
        KIND_DECODER,
        analytic_blocks,
        numeric_blocks,
        tolerance=resolved_tolerance,
        step=step,
        notes=(
            "解码器块的新东西只有**接线**：三段子层的前向/反向都已被逐项验过",
            "encoder_outputs 那一项的梯度来自交叉注意力的 source 那一路——**编码器会收到梯度**",
            "自注意力按 pre-LN 契约重建在 LN(decoder_inputs) 上，"
            "因此第一个子层也在“被检查的函数”里面（只用到传入 forward 的 params/causal）",
        ),
    )


# ---------------------------------------------------------------------- 性质


@dataclass(frozen=True)
class PropertyOutcome:
    """一条性质的读数."""

    name: str
    passed: bool
    evidence: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.name not in ENCODER_DECODER_PROPERTIES:
            raise ParameterError(
                f"未知的性质名 {self.name!r}：可选 {', '.join(ENCODER_DECODER_PROPERTIES)}。"
            )
        object.__setattr__(self, "evidence", str(self.evidence))
        object.__setattr__(self, "detail", str(self.detail))

    @property
    def description(self) -> str:
        """这一条在说什么."""
        return PROPERTY_DESCRIPTIONS[self.name]

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "name": self.name,
            "passed": self.passed,
            "evidence": self.evidence,
            "detail": self.detail,
            "description": self.description,
        }

    def summary_line(self) -> str:
        """一行说明：``[ok] norm_rows_are_standardized 4 行，均值最大偏离 0.00e+00``."""
        mark = "ok" if self.passed else "!!"
        return f"[{mark}] {self.name} {self.evidence}"


@dataclass(frozen=True)
class PropertyReport:
    """八条性质的报告."""

    outcomes: tuple[PropertyOutcome, ...]
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        resolved = tuple(self.outcomes)
        if not resolved:
            raise ParameterError("性质报告不能为空。")
        names = {item.name for item in resolved}
        if names != set(ENCODER_DECODER_PROPERTIES):
            raise NumericError(
                f"性质报告的名单不完整：缺 {sorted(set(ENCODER_DECODER_PROPERTIES) - names)}。"
            )
        object.__setattr__(self, "outcomes", resolved)
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def ok(self) -> bool:
        """全部通过."""
        return all(item.passed for item in self.outcomes)

    @property
    def failures(self) -> tuple[PropertyOutcome, ...]:
        """没通过的那些条."""
        return tuple(item for item in self.outcomes if not item.passed)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "ok": self.ok,
            "outcomes": [item.to_dict() for item in self.outcomes],
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``性质检查 8 项：通过 8、失败 0``."""
        passed = sum(1 for item in self.outcomes if item.passed)
        return (
            f"性质检查 {len(self.outcomes)} 项：通过 {passed}、失败 "
            f"{len(self.outcomes) - passed}"
        )


def _row_statistics(matrix: Matrix) -> tuple[Vector, Vector]:
    """每一行的均值与**有偏**方差."""
    checked = validate_matrix(matrix, name="matrix")
    columns = matrix_shape(checked)[1]
    means: list[float] = []
    variances: list[float] = []
    for row in checked:
        mean = math.fsum(row) / columns
        means.append(mean)
        variances.append(math.fsum((value - mean) ** 2 for value in row) / columns)
    return tuple(means), tuple(variances)


def check_rows_are_standardized(
    cache: NormCache, *, tolerance: float = 1e-12
) -> PropertyOutcome:
    """标准化之后每一行的均值**恰好**为 0；方差是 ``σ²/(σ² + eps)`` 而不是恰好 1.

    后一句话是这一条判据的全部内容，而它有一个具体的后果：

    ```text
    eps → 0      x̂ 的方差恰好是 1（教科书里的那一句）
    eps > 0      x̂ 的方差是 σ²/(σ² + eps) < 1，偏差 ≈ eps/σ²
    ```

    本课**不**把那个偏差说成“误差”：它是 ``eps`` 的定义。因此判据对每一行**分别**算
    期望值 ``σ²_i/(σ²_i + eps)`` 再比——而不是拿“方差应当为 1”去比，
    后者的失败量与 ``eps`` 成正比，看起来像 bug 而其实是数学。
    """
    resolved = _checked_tolerance(tolerance)
    means, variances = _row_statistics(cache.normalized)
    expected = tuple(
        value / (value + cache.epsilon) if value + cache.epsilon > 0 else 0.0
        for value in cache.variance
    )
    worst_mean = max(abs(value) for value in means)
    worst_variance = max(
        abs(value - target) for value, target in zip(variances, expected, strict=True)
    )
    naive = max(abs(value - 1.0) for value in variances)
    return PropertyOutcome(
        name=PROPERTY_ROWS_STANDARDIZED,
        passed=worst_mean <= resolved and worst_variance <= resolved,
        evidence=(
            f"检查 {len(means)} 行：均值最大偏离 {worst_mean:.2e}；"
            f"方差与 σ²/(σ²+eps) 的最大偏离 {worst_variance:.2e}"
            f"（而“方差离 1 多远”是 {naive:.2e}，它由 eps={cache.epsilon:g} 决定）"
        ),
        detail="均值那一项与 eps 无关（恰好为 0）；方差那一项是 eps 的直接后果，不是误差",
    )


def check_shift_invariance(
    inputs: Matrix,
    *,
    shift: float = 3.5,
    gamma: Vector | None = None,
    beta: Vector | None = None,
    epsilon: float = 1e-12,
    tolerance: float = 1e-9,
) -> PropertyOutcome:
    """``LN(x + c·1) == LN(x)``：**平移不是信息**（每一行加同一个常数）.

    这是 ``eps`` 无关的**精确**恒等式：``(x + c) − (μ + c) = x − μ`` 逐位成立，
    而 ``eps`` 在分子分母里是同一个数。本函数默认用 ``eps = 1e-12`` 量它。
    """
    resolved = _checked_tolerance(tolerance)
    checked = validate_matrix(inputs, name="inputs")
    shifted = tuple(tuple(value + shift for value in row) for row in checked)
    left, _ = layer_norm(checked, gamma=gamma, beta=beta, epsilon=epsilon)
    right, _ = layer_norm(shifted, gamma=gamma, beta=beta, epsilon=epsilon)
    worst = max(
        abs(a - b)
        for left_row, right_row in zip(left, right, strict=True)
        for a, b in zip(left_row, right_row, strict=True)
    )
    return PropertyOutcome(
        name=PROPERTY_SHIFT_INVARIANT,
        passed=worst <= resolved,
        evidence=(
            f"加常数 {shift} 前后的最大偏差 {worst:.2e}"
            f"（eps={epsilon:g}，容差 {resolved:g}）"
        ),
        detail="均值被减掉了两次（一次在前向、一次在标准化里），因此常数完全消失",
    )


def check_scale_equivariance(
    inputs: Matrix,
    *,
    factor: float = 2.5,
    gamma: Vector | None = None,
    beta: Vector | None = None,
    epsilon: float = 1e-12,
    tolerance: float = 1e-9,
) -> PropertyOutcome:
    """``LN(λx) == LN(x)``（``λ > 0``）：**尺度也不是信息**（``eps → 0`` 时严格成立）.

    与上一条的差别值得说清：平移不依赖 ``eps``，而缩放只在 ``eps = 0`` 时精确——
    ``eps > 0`` 时分母是 ``√(λ²σ² + eps)``，与分子的 ``λ`` 不再恰好抵消，
    残差是 ``O(eps/σ²)`` 量级。本函数用 ``eps = 1e-12`` 量它，并把 ``eps`` 印在证据里。
    """
    resolved = _checked_tolerance(tolerance)
    if not math.isfinite(factor) or factor <= 0:
        raise ParameterError(f"factor 必须是正的有限数，收到 {factor!r}。")
    checked = validate_matrix(inputs, name="inputs")
    scaled = tuple(tuple(value * factor for value in row) for row in checked)
    left, _ = layer_norm(checked, gamma=gamma, beta=beta, epsilon=epsilon)
    right, _ = layer_norm(scaled, gamma=gamma, beta=beta, epsilon=epsilon)
    worst = max(
        abs(a - b)
        for left_row, right_row in zip(left, right, strict=True)
        for a, b in zip(left_row, right_row, strict=True)
    )
    return PropertyOutcome(
        name=PROPERTY_SCALE_EQUIVARIANT,
        passed=worst <= resolved,
        evidence=(
            f"乘正数 {factor} 前后的最大偏差 {worst:.2e}"
            f"（eps={epsilon:g}，容差 {resolved:g}）"
        ),
        detail="分母里的 σ 按同一个倍数变大，因此 λ 约掉了——而 eps 让它只约到 O(eps/σ²)",
    )


def check_norm_is_row_independent(
    inputs: Matrix,
    *,
    gamma: Vector | None = None,
    beta: Vector | None = None,
) -> PropertyOutcome:
    """换行序 → 输出**逐位**跟着换（每一行只依赖它自己）.

    这是 LayerNorm 与 **BatchNorm** 的分水岭：后者的统计量跨样本，
    因此换行序会**改变每一行的输出**。这一条用 ``==`` 断言。
    """
    checked = validate_matrix(inputs, name="inputs")
    rows = matrix_shape(checked)[0]
    permutation = reverse_permutation(rows)
    permuted = permute_rows(checked, permutation)
    base, _ = layer_norm(checked, gamma=gamma, beta=beta)
    other, _ = layer_norm(permuted, gamma=gamma, beta=beta)
    expected = permute_rows(base, permutation)
    return PropertyOutcome(
        name=PROPERTY_NORM_ROW_INDEPENDENT,
        passed=other == expected,
        evidence=(
            f"置换 {permutation} 下 LN(πx) 与 πLN(x) 是否**逐位**相等：{other == expected}"
        ),
        detail="判据是 == 而不是容差：逐行算子不产生任何求和顺序的变化",
    )


def check_feed_forward_is_position_wise(
    inputs: Matrix,
    weights: FFNWeights,
    *,
    activation: str = ACTIVATION_RELU,
) -> PropertyOutcome:
    """前馈是逐位置的：``FFN(πx) == πFFN(x)``（**逐位**）."""
    checked = validate_matrix(inputs, name="inputs")
    rows = matrix_shape(checked)[0]
    permutation = reverse_permutation(rows)
    permuted = permute_rows(checked, permutation)
    base, _ = feed_forward(checked, weights, activation=activation)
    other, _ = feed_forward(permuted, weights, activation=activation)
    expected = permute_rows(base, permutation)
    return PropertyOutcome(
        name=PROPERTY_FFN_POSITION_WISE,
        passed=other == expected,
        evidence=(
            f"置换 {permutation} 下 FFN(πx) 与 πFFN(x) 是否**逐位**相等：{other == expected}"
        ),
        detail="它与上一条合起来说明：打破置换等变性的只有位置编码与掩码",
    )


def _zero_attention_params(hidden: int) -> AttentionParams:
    """四个投影都是零矩阵的注意力参数（于是它的输出恰好是全零）."""
    zeros = tuple(tuple(0.0 for _ in range(hidden)) for _ in range(hidden))
    return AttentionParams(w_query=zeros, w_key=zeros, w_value=zeros, w_output=zeros)


def _zero_ffn(weights: FFNWeights) -> FFNWeights:
    """前馈的四块全部置零（于是它的输出恰好是全零）."""
    return FFNWeights(
        w_in=tuple(tuple(0.0 for _ in row) for row in weights.w_in),
        b_in=tuple(0.0 for _ in weights.b_in),
        w_out=tuple(tuple(0.0 for _ in row) for row in weights.w_out),
        b_out=tuple(0.0 for _ in weights.b_out),
    )


def check_residual_identity(
    params: BlockParameters,
    inputs: Matrix,
    *,
    placement: str = NORM_PRE,
) -> PropertyOutcome:
    """**两个分支都为 0** 时块退化成恒等映射（``output == inputs`` **逐位**）.

    “分支为零”要**两支**都为零，这一点本身就是一个值得记下的坑：

    ```text
    只把前馈的输出层置零   → residual2 = residual1 = inputs + 注意力输出 ≠ inputs
    两个分支都置零         → output = inputs（**逐位**）
    ```

    注意力那一支的零由“四个投影都是零矩阵”构造；前馈那一支的零同理。
    此时残差那一条 ``+1`` 的路是唯一还活着的东西——**输出与输入逐位相等**。

    反向里 ``dx`` **不会**恰好等于 ``dy``：两个 LN 的链仍然在贡献，
    因此本函数把两者的范数比也印出来（它回答“残差那条路占了多少”）。
    """
    checked = validate_matrix(inputs, name="inputs")
    zero_params = params.with_ffn(_zero_ffn(params.ffn))
    resolved_placement = _checked_placement(placement)
    zero_attention_params = _zero_attention_params(params.hidden)
    attention = block_attention(
        zero_params,
        checked,
        zero_attention_params,
        placement=resolved_placement,
    )
    forward = encoder_block(
        zero_params,
        checked,
        attention,
        placement=resolved_placement,
        use_residual=True,
    )
    identity = forward.output == checked
    if not identity:
        return PropertyOutcome(
            name=PROPERTY_RESIDUAL_IDENTITY,
            passed=False,
            evidence="两个分支都为 0 时块的输出与输入**不**逐位相等——残差那条路没有生效",
            detail="这一条要求**两个**分支都为零，只零一个是不够的",
        )
    grad_out = tuple(tuple(1.0 for _ in row) for row in forward.output)
    grads = encoder_block_backward(forward, zero_params, grad_out)
    dy_norm = math.sqrt(math.fsum(value * value for row in grad_out for value in row))
    dx_norm = math.sqrt(
        math.fsum(value * value for row in grads.grad_inputs for value in row)
    )
    ratio = dx_norm / dy_norm if dy_norm > 0 else (0.0 if dx_norm == 0 else math.inf)
    return PropertyOutcome(
        name=PROPERTY_RESIDUAL_IDENTITY,
        passed=identity,
        evidence=(
            f"{resolved_placement}-LN：两个分支都为 0 时 output == inputs（{identity}）；"
            f"上游取全 1 时 ‖dx‖/‖dy‖ = {ratio:.6f}（两条 LN 链在贡献，因此它不等于 1）"
        ),
        detail="判据用 ==（恒等映射没有求和顺序问题），而那个不等于 1 的比值是 LN 的贡献",
    )


def check_residual_unit_path(
    params: BlockParameters,
    inputs: Matrix,
    attention_params: AttentionParams,
    *,
    placement: str = NORM_PRE,
) -> PropertyOutcome:
    """反向里存在一条**不随分支缩放**的 +1 路：把分支整体缩小，``dx`` 不随之趋零.

    做法：把前馈的输出层整体乘上一个越来越小的系数 ``s``，然后比较 ``dx`` 的量级。

    ```text
    带残差   dx ≈ dy + O(s)       → s 再小也有一条 dy 在那里
    不带残差 dx = O(s)            → 分支被压小时梯度一起被压小
    ```
    """
    checked = validate_matrix(inputs, name="inputs")
    resolved_placement = _checked_placement(placement)
    scales = (1.0, 0.25, 0.0625)
    lines: list[str] = []
    with_residual: list[float] = []
    without_residual: list[float] = []
    for scale in scales:
        scaled_ffn = FFNWeights(
            w_in=params.ffn_w_in,
            b_in=params.ffn_b_in,
            w_out=tuple(tuple(value * scale for value in row) for row in params.ffn_w_out),
            b_out=tuple(value * scale for value in params.ffn_b_out),
        )
        local = params.with_ffn(scaled_ffn)
        for use_residual, bucket in ((True, with_residual), (False, without_residual)):
            attention = block_attention(
                local, checked, attention_params, placement=resolved_placement
            )
            forward = encoder_block(
                local,
                checked,
                attention,
                placement=resolved_placement,
                use_residual=use_residual,
            )
            grad_out = mse_gradient(forward.output, checked)
            grads = encoder_block_backward(forward, local, grad_out)
            norm = math.sqrt(
                math.fsum(value * value for row in grads.grad_inputs for value in row)
            )
            bucket.append(norm)
        lines.append(
            f"s={scale:<7g} 有残差 {with_residual[-1]:.6f} | 无残差 {without_residual[-1]:.6f}"
        )
    with_ratio = with_residual[-1] / with_residual[0]
    without_ratio = without_residual[-1] / without_residual[0]
    gap = with_ratio / without_ratio if without_ratio > 0 else math.inf
    return PropertyOutcome(
        name=PROPERTY_RESIDUAL_UNIT_PATH,
        passed=with_ratio > 3.0 * without_ratio,
        evidence=(
            f"把前馈输出层压到 1/16：有残差 ‖dx‖ 变为原来的 {with_ratio:.2e}、"
            f"无残差变为 {without_ratio:.2e}——两者相差 {gap:.1f} 倍"
        ),
        detail="有残差时 dx 有一个 dy 给出的下界；无残差时它随分支一起被压小",
    )


def causal_mask_damage(forward: Any) -> tuple[float, str]:
    """**故意加错**：把因果掩码乘到交叉注意力的权重上，量出它造成的偏差.

    返回 ``(最大相对偏差, 说明)``。它是第 8 条性质的说服力所在：

    ```text
    n_tgt == n_src   掩码形状刚好合适 → **不报错**，只是每一行能看到的源位置少了一半
    n_tgt >  n_src   掩码形状对不上   → 会报形状错误（那还算运气好）
    ```
    """
    weights = forward.weights
    targets, sources = matrix_shape(weights)
    if targets != sources:
        return 0.0, (
            f"n_tgt={targets} != n_src={sources}：因果掩码的形状对不上，"
            "因此它**会**以形状错误的形式暴露出来（这一次不算静默错误）"
        )
    mask = tuple(tuple(column <= row for column in range(sources)) for row in range(targets))
    masked = masked_softmax_rows(forward.scores, mask)
    worst = relative_matrix_error(masked, weights)
    return worst, (
        f"n_tgt == n_src == {targets}：掩码形状刚好合适 → **不报错**，"
        f"而权重被改掉了 {worst:.2e}（每一行能看到的源位置少了一半）"
    )


def check_cross_attention_is_not_causal(
    params: CrossParameters,
    target_inputs: Matrix,
    source_inputs: Matrix,
) -> PropertyOutcome:
    """交叉注意力**不能**加因果掩码：本包直接拒绝，而加错的代价可以量出来."""
    refused = False
    try:
        cross_attention(params, target_inputs, source_inputs, causal=True)
    except AssemblyError:
        refused = True
    forward = cross_attention(params, target_inputs, source_inputs)
    damage, note = causal_mask_damage(forward)
    return PropertyOutcome(
        name=PROPERTY_CROSS_NOT_CAUSAL,
        passed=refused,
        evidence=f"causal=True 是否被拒绝：{refused}；{note}",
        detail=(
            "权重形状是 (n_tgt, n_src)，它是**长方形**的——"
            "一份方阵掩码只可能在 n_tgt == n_src 时“恰好”合适"
        ),
    )


def check_properties(
    params: BlockParameters,
    inputs: Matrix,
    attention_params: AttentionParams,
    cross_params: CrossParameters,
    *,
    placement: str = NORM_PRE,
) -> PropertyReport:
    """跑完八条性质（第 4、5 条是逐位断言；第 8 条量“加错的代价”）."""
    _norm1, cache = layer_norm(
        inputs, gamma=params.norm1_gamma, beta=params.norm1_beta
    )
    source_width = matrix_shape(cross_params.w_key)[1]
    source = tuple(
        tuple(0.5 * (index + column + 1) for column in range(source_width))
        for index in range(matrix_shape(inputs)[0])
    )
    outcomes = (
        check_rows_are_standardized(cache),
        check_shift_invariance(inputs, gamma=params.norm1_gamma, beta=params.norm1_beta),
        check_scale_equivariance(inputs, gamma=params.norm1_gamma, beta=params.norm1_beta),
        check_norm_is_row_independent(
            inputs, gamma=params.norm1_gamma, beta=params.norm1_beta
        ),
        check_feed_forward_is_position_wise(inputs, params.ffn),
        check_residual_identity(params, inputs, placement=placement),
        check_residual_unit_path(params, inputs, attention_params, placement=placement),
        check_cross_attention_is_not_causal(cross_params, inputs, source),
    )
    return PropertyReport(
        outcomes=outcomes,
        notes=(
            "前四条关于 LN、第五条关于前馈、第六七条关于残差、第八条关于交叉注意力",
            "第 4、5 两条**逐位**相等；第 6、7 条是“残差值钱”的两个角度",
        ),
    )
