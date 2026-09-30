"""``transformer_stack`` 的校验：两处梯度 + 六条性质（day080 / M7-D5）.

## 今天要钉的两件事

```text
① 链式反向接对了没有       整条链的输入梯度 vs 数值差分（逐项，容差 1e-6）
② 链上的每一环都接到了正确的上游
                        第 k 层的**八块参数**梯度 vs 数值差分
                        —— 这是今天唯一能抓住"每层都用同一份 grad_output"那种错误的检查
```

为什么②是必要的：那种错法在**最后一层**上是完全正确的（最后一层的
``grad_output`` 恰好就是 ``dLoss/dy_N``），因此只看最后一层会全绿；
而第 0 层的参数梯度会整体小一截。为了让检查**只针对链**而不重复 day079 的九项，
本课在第 k=0 层上查那八块，其余层由“逐层读数”与第①条覆盖。

## 六条性质里有两条要用 ``==``

```text
保形              逐层比较形状 ⇒ 可以逐位（形状是整数，没有容差一说）
与手写循环一致     两条路径做的是**同一串浮点运算** ⇒ 必须逐位相等
确定性            同一组输入的两次调用 ⇒ 必须逐位相等
分支全零即恒等     输出与输入**逐位**比较（`x + 0 = x` 在 IEEE 里是精确的）
参数量与构造对齐   整数比较
组装脚本带着形状    从脚本的 ``CONFIG`` 里逐键读回来比
```
“与手写循环逐位一致”这一条是本课最便宜也最值钱的一条：它把
“``stack_forward`` 只是 ``encoder_block`` 外面套了一个 ``for``”这句话
变成了一条**可以失败的断言**——一旦有人把中间结果存成 ``float32``、
或者多过一次 ``LN``，它立刻亮红。
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from smart_research_agent.encoder_decoder.layers import block_attention, encoder_block
from smart_research_agent.encoder_decoder.types import (
    ACTIVATION_RELU,
    BLOCK_GRADIENT_TARGETS,
    GRAD_INPUTS,
    NORM_PRE,
    BlockParameters,
    _checked_activation,
    _checked_placement,
)
from smart_research_agent.math_foundations.calculus import DEFAULT_STEP, gradient
from smart_research_agent.math_foundations.optim import (
    flatten_matrices,
    unflatten_matrices,
)
from smart_research_agent.math_foundations.types import (
    Matrix,
    Vector,
    matrix_shape,
    validate_matrix,
    validate_vector,
)
from smart_research_agent.transformer_core.verify import (
    GRADIENT_TOLERANCE as CORE_TOLERANCE,
)
from smart_research_agent.transformer_stack.assembly import (
    ASSEMBLY_CLASSES,
    ASSEMBLY_CONFIG_KEYS,
    ASSEMBLY_MODULES,
    assembly_facts,
    assembly_script,
)
from smart_research_agent.transformer_stack.errors import NumericError, ParameterError
from smart_research_agent.transformer_stack.layers import (
    make_stack_parameters,
    stack_forward,
    stack_loss,
    stack_loss_gradient,
    zero_attention,
    zero_branches,
)
from smart_research_agent.transformer_stack.types import (
    GRAD_STACK_INPUTS,
    KIND_LAYER,
    KIND_STACK,
    LAYER_GRADIENT_TARGETS,
    PROPERTY_ASSEMBLY_CARRIES_SHAPE,
    PROPERTY_DESCRIPTIONS,
    PROPERTY_DETERMINISTIC,
    PROPERTY_IDENTITY_WHEN_BRANCHES_VANISH,
    PROPERTY_MATCHES_LOOP,
    PROPERTY_PARAMETER_COUNT_MATCHES,
    PROPERTY_SHAPE_PRESERVED,
    STACK_GRADIENT_TARGETS,
    STACK_NOTES,
    STACK_PROPERTIES,
    STACK_STAGES,
    STAGE_DESCRIPTIONS,
    STAGE_SHAPES,
    StackForward,
    StackGradients,
    StackParameters,
    StackShape,
    _checked_layer_index,
    _checked_tolerance,
    frobenius,
)

#: 本课的梯度容差（**与 day075~079 同值**：口径换一次，diff 就要说得清哪一次换了）.
GRADIENT_TOLERANCE = 1e-6

if GRADIENT_TOLERANCE != CORE_TOLERANCE:  # pragma: no cover - 只在有人改常量时触发
    raise NumericError(
        f"本课的梯度容差 {GRADIENT_TOLERANCE} 与 day075 的 {CORE_TOLERANCE} 不一致。"
    )


def _scaled_error(left: float, right: float) -> float:
    """逐点相对误差（分母 ``max(1, |左|, |右|)``，与 day075~079 同口径）."""
    if not (math.isfinite(left) and math.isfinite(right)):
        return math.inf
    return abs(left - right) / max(1.0, abs(left), abs(right))


def _compare(
    left: Matrix, right: Matrix
) -> tuple[float, float, int]:
    """两个同形矩阵的 ``(最大绝对误差, 最大相对误差, 比较点数)``."""
    checked_left = validate_matrix(left, name="left")
    checked_right = validate_matrix(right, name="right")
    if matrix_shape(checked_left) != matrix_shape(checked_right):
        raise ParameterError(
            f"两边形状不同：{matrix_shape(checked_left)} 与 {matrix_shape(checked_right)}。"
        )
    worst_absolute = 0.0
    worst_scaled = 0.0
    points = 0
    for row_left, row_right in zip(checked_left, checked_right, strict=True):
        for value_left, value_right in zip(row_left, row_right, strict=True):
            points += 1
            worst_absolute = max(worst_absolute, abs(value_left - value_right))
            worst_scaled = max(worst_scaled, _scaled_error(value_left, value_right))
    return worst_absolute, worst_scaled, points


# ---------------------------------------------------------------------- 梯度报告


@dataclass(frozen=True)
class GradientOutcome:
    """一项梯度校验的读数（通用形状：名字由名单给出）."""

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
            raise NumericError(
                f"compared_points 必须是非负整数，收到 {self.compared_points!r}。"
            )
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
        """一行说明：``[ok] stack_inputs 最大相对误差 3.1e-11（绝对 2.0e-11，6 个逐点）``."""
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
        if self.kind not in STACK_GRADIENT_TARGETS:
            raise ParameterError(
                f"未知的报告种类 {self.kind!r}：可选 {', '.join(STACK_GRADIENT_TARGETS)}。"
            )
        resolved = tuple(self.outcomes)
        names = [item.target for item in resolved]
        if names != list(STACK_GRADIENT_TARGETS[self.kind]):
            raise NumericError(
                f"{self.kind} 这份报告的名单对不上：\n"
                f"  给的   {names}\n"
                f"  要的   {list(STACK_GRADIENT_TARGETS[self.kind])}\n"
                "缺项的报告在'全绿'时看起来与完整的一样——因此这里逐项对名单。"
            )
        if resolved and resolved[0].tolerance != self.tolerance:
            raise NumericError("报告里的容差与这一项的容差不一致。")
        object.__setattr__(self, "outcomes", resolved)
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def ok(self) -> bool:
        """全绿."""
        return all(item.passed for item in self.outcomes)

    @property
    def failures(self) -> tuple[GradientOutcome, ...]:
        """没通过的那些项."""
        return tuple(item for item in self.outcomes if not item.passed)

    @property
    def worst_scaled_error(self) -> float:
        """所有项里最大的相对误差."""
        return max(item.max_scaled_error for item in self.outcomes)

    @property
    def passed_targets(self) -> tuple[str, ...]:
        """通过的那些项的名字（按名单顺序）."""
        return tuple(item.target for item in self.outcomes if item.passed)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "kind": self.kind,
            "ok": self.ok,
            "worst_scaled_error": self.worst_scaled_error,
            "outcomes": [item.to_dict() for item in self.outcomes],
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``stack 梯度校验 1 项：通过 1、失败 0 | 最大相对误差 3.1e-11``."""
        failed = len(self.failures)
        return (
            f"{self.kind} 梯度校验 {len(self.outcomes)} 项：通过 "
            f"{len(self.outcomes) - failed}、失败 {failed} | "
            f"最大相对误差 {self.worst_scaled_error:.2e}"
        )

    def detail_lines(self) -> tuple[str, ...]:
        """逐项一行（演示脚本按节打印它）."""
        lines = [f"    {item.target} {item.max_scaled_error:.2e}" for item in self.outcomes]
        return tuple(lines)


def _report(
    kind: str,
    analytic: dict[str, Matrix],
    numeric: dict[str, Matrix],
    *,
    tolerance: float,
    step: float,
    notes: tuple[str, ...],
) -> GradientReport:
    """按名单逐项对照并组装报告（两侧的名单与顺序都来自 ``STACK_GRADIENT_TARGETS``）."""
    outcomes: list[GradientOutcome] = []
    for name in STACK_GRADIENT_TARGETS[kind]:
        worst_absolute, worst_scaled, points = _compare(analytic[name], numeric[name])
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


# ---------------------------------------------------------------------- 数值梯度


def numerical_stack_input_gradient(
    params: StackParameters,
    inputs: Matrix,
    target: Matrix,
    *,
    placement: str = NORM_PRE,
    use_residual: bool = True,
    activation: str = ACTIVATION_RELU,
    step: float = DEFAULT_STEP,
) -> Matrix:
    """整条链的输入梯度（中心差分，**穿过全部 N 层**）."""
    resolved_placement = _checked_placement(placement)
    resolved_activation = _checked_activation(activation)
    checked_inputs = validate_matrix(inputs, name="inputs")
    flat_inputs, input_shapes = flatten_matrices((checked_inputs,))

    def objective(theta: Vector) -> float:
        """给定压平后的输入，返回整条链的损失."""
        rebuilt = unflatten_matrices(theta, input_shapes)[0]
        return stack_loss(
            params,
            rebuilt,
            target,
            placement=resolved_placement,
            use_residual=use_residual,
            activation=resolved_activation,
        )

    numeric = gradient(objective, flat_inputs, step=step)
    return unflatten_matrices(numeric, input_shapes)[0]


def check_stack_input_gradient(
    params: StackParameters,
    inputs: Matrix,
    target: Matrix,
    *,
    placement: str = NORM_PRE,
    use_residual: bool = True,
    activation: str = ACTIVATION_RELU,
    step: float = DEFAULT_STEP,
    tolerance: float = GRADIENT_TOLERANCE,
) -> GradientReport:
    """整条链的输入梯度：解析 vs 数值（**今天钉住链式反向的那一条**）."""
    resolved_tolerance = _checked_tolerance(tolerance)
    _forward, grads = stack_loss_gradient(
        params,
        inputs,
        target,
        placement=placement,
        use_residual=use_residual,
        activation=activation,
    )
    numeric = numerical_stack_input_gradient(
        params,
        inputs,
        target,
        placement=placement,
        use_residual=use_residual,
        activation=activation,
        step=step,
    )
    return _report(
        KIND_STACK,
        {GRAD_STACK_INPUTS: grads.grad_inputs},
        {GRAD_STACK_INPUTS: numeric},
        tolerance=resolved_tolerance,
        step=step,
        notes=(
            f"链长 {params.layers} 层：这条检查会穿过全部 {params.layers} 环",
            "解析侧来自 stack_backward；数值侧来自中心差分",
        ),
    )


def numerical_layer_parameter_gradients(
    params: StackParameters,
    inputs: Matrix,
    target: Matrix,
    layer: int,
    *,
    placement: str = NORM_PRE,
    use_residual: bool = True,
    activation: str = ACTIVATION_RELU,
    step: float = DEFAULT_STEP,
) -> tuple[Matrix, ...]:
    """第 ``layer`` 层八块参数的数值梯度（**其余层不动**）.

    数值侧的做法是标准的“扰动一块、重跑整条链”：因此它天然知道
    “这一层的改动要穿过它上面的所有层才影响损失”——而这正是链式反向必须算对的东西。
    """
    resolved_layer = _checked_layer_index(layer, params.layers)
    resolved_placement = _checked_placement(placement)
    resolved_activation = _checked_activation(activation)
    layer_params = params.blocks[resolved_layer]
    flat, shapes = layer_params.flatten()

    def objective(theta: Vector) -> float:
        """给定压平后的这一层参数，返回整条链的损失."""
        rebuilt = params.replace_layer(
            resolved_layer, BlockParameters.unflatten(theta, shapes)
        )
        return stack_loss(
            params=rebuilt,
            inputs=inputs,
            target=target,
            placement=resolved_placement,
            use_residual=use_residual,
            activation=resolved_activation,
        )

    numeric_flat = gradient(objective, flat, step=step)
    return unflatten_matrices(numeric_flat, shapes)


def check_layer_parameter_gradients(
    params: StackParameters,
    inputs: Matrix,
    target: Matrix,
    layer: int = 0,
    *,
    placement: str = NORM_PRE,
    use_residual: bool = True,
    activation: str = ACTIVATION_RELU,
    step: float = DEFAULT_STEP,
    tolerance: float = GRADIENT_TOLERANCE,
) -> GradientReport:
    """第 ``layer`` 层八块参数：解析 vs 数值（**每一环都接到正确上游**的那一条）."""
    resolved_layer = _checked_layer_index(layer, params.layers)
    resolved_tolerance = _checked_tolerance(tolerance)
    _forward, grads = stack_loss_gradient(
        params,
        inputs,
        target,
        placement=placement,
        use_residual=use_residual,
        activation=activation,
    )
    analytic = grads.layer_at(resolved_layer).as_dict()
    numeric_parts = numerical_layer_parameter_gradients(
        params,
        inputs,
        target,
        resolved_layer,
        placement=placement,
        use_residual=use_residual,
        activation=activation,
        step=step,
    )
    order = [name for name in BLOCK_GRADIENT_TARGETS if name != GRAD_INPUTS]
    numeric = {
        name: numeric_parts[index] for index, name in enumerate(order)
    }
    return _report(
        KIND_LAYER,
        {name: analytic[name] for name in LAYER_GRADIENT_TARGETS},
        numeric,
        tolerance=resolved_tolerance,
        step=step,
        notes=(
            f"检查的是第 {resolved_layer} 层（共 {params.layers} 层）",
            "中间层的参数梯度必须包含'改动要穿过上面所有层'这一条链——只看最后一层看不出错",
            "名单里没有 'inputs'：中间层的输入不是自变量，它由前一层算出来",
        ),
    )


# ---------------------------------------------------------------------- 性质报告


@dataclass(frozen=True)
class PropertyOutcome:
    """一条性质的读数."""

    name: str
    passed: bool
    evidence: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.name not in STACK_PROPERTIES:
            raise ParameterError(
                f"未知的性质名 {self.name!r}：可选 {', '.join(STACK_PROPERTIES)}。"
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
        """一行说明：``[ok] stack_is_deterministic 两次调用逐位相等``."""
        mark = "ok" if self.passed else "!!"
        return f"[{mark}] {self.name} {self.evidence}"


@dataclass(frozen=True)
class PropertyReport:
    """六条性质的报告."""

    outcomes: tuple[PropertyOutcome, ...]
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        resolved = tuple(self.outcomes)
        if not resolved:
            raise ParameterError("性质报告不能为空。")
        names = {item.name for item in resolved}
        if names != set(STACK_PROPERTIES):
            raise NumericError(
                f"性质报告的名单不完整：缺 {sorted(set(STACK_PROPERTIES) - names)}。"
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
        """一行说明：``性质检查 6 项：通过 6、失败 0``."""
        failed = len(self.failures)
        return (
            f"性质检查 {len(self.outcomes)} 项：通过 "
            f"{len(self.outcomes) - failed}、失败 {failed}"
        )


def _outcome(name: str, passed: bool, evidence: str, detail: str = "") -> PropertyOutcome:
    """造一条性质读数."""
    return PropertyOutcome(name=name, passed=passed, evidence=evidence, detail=detail)


# ---------------------------------------------------------------------- 六条性质


def check_stack_preserves_shape(forward: StackForward) -> PropertyOutcome:
    """逐层保形 + 链上传递**逐位**一致（两个 ``==``，没有任何容差）."""
    if not isinstance(forward, StackForward):
        raise ParameterError(f"forward 必须是 StackForward，收到 {type(forward).__name__}。")
    shape_ok = all(
        matrix_shape(layer.block.inputs) == matrix_shape(layer.block.output)
        for layer in forward.layers
    )
    carry_ok = all(
        forward.layers[index].block.output == forward.layers[index + 1].block.inputs
        for index in range(forward.depth - 1)
    )
    exit_ok = forward.layers[-1].block.output == forward.output
    return _outcome(
        PROPERTY_SHAPE_PRESERVED,
        shape_ok and carry_ok and exit_ok,
        f"{forward.depth} 层逐层同形、{max(forward.depth - 1, 0)} 处传递逐位一致、出口逐位一致",
        "判据是 ==（形状是整数、传递是同一个值），因此它没有容差可以调",
    )


def check_stack_matches_loop(
    params: StackParameters,
    inputs: Matrix,
    forward: StackForward,
    *,
    placement: str = NORM_PRE,
    use_residual: bool = True,
    activation: str = ACTIVATION_RELU,
) -> PropertyOutcome:
    """与**手写逐层调用**逐位一致（本课最便宜、也最值钱的一条）."""
    resolved_placement = _checked_placement(placement)
    resolved_activation = _checked_activation(activation)
    checked = validate_matrix(inputs, name="inputs")
    current = checked
    for index, (block_params, attention_params) in enumerate(
        zip(params.blocks, params.attentions, strict=True)
    ):
        attention = block_attention(
            block_params, current, attention_params, placement=resolved_placement
        )
        rebuilt = encoder_block(
            block_params,
            current,
            attention,
            placement=resolved_placement,
            use_residual=use_residual,
            activation=resolved_activation,
        )
        if rebuilt.output != forward.layer_at(index).block.output:
            return _outcome(
                PROPERTY_MATCHES_LOOP,
                False,
                f"第 {index} 层的手写输出与 stack_forward 的输出不符",
                "两条路径必须做同一串浮点运算：不一致说明中间多过一次运算或参数错位",
            )
        current = rebuilt.output
    return _outcome(
        PROPERTY_MATCHES_LOOP,
        current == forward.output,
        f"{forward.depth} 层手写循环与 stack_forward 逐位相等",
        "这一条把'只是套了一个 for'从一句话变成一条可失败的断言",
    )


def check_stack_is_deterministic(
    params: StackParameters,
    inputs: Matrix,
    forward: StackForward,
    *,
    placement: str = NORM_PRE,
    use_residual: bool = True,
    activation: str = ACTIVATION_RELU,
) -> PropertyOutcome:
    """同一组参数跑两次：输出与**每一行读数**都逐位相同."""
    again = stack_forward(
        params,
        inputs,
        placement=placement,
        use_residual=use_residual,
        activation=activation,
    )
    output_ok = again.output == forward.output
    census_ok = tuple(row.to_dict() for row in again.censuses) == tuple(
        row.to_dict() for row in forward.censuses
    )
    return _outcome(
        PROPERTY_DETERMINISTIC,
        output_ok and census_ok,
        f"输出逐位相等 {output_ok}、{forward.depth} 行读数逐位相等 {census_ok}",
        "本包没有随机数：确定性不是运气，而是'没有 rand' 的直接后果",
    )


def check_stack_identity_when_branches_vanishes(
    params: StackParameters,
    inputs: Matrix,
    *,
    placement: str = NORM_PRE,
    activation: str = ACTIVATION_RELU,
) -> PropertyOutcome:
    """把**每一层**的两个分支都置零（含注意力），整条链退化成恒等映射.

    三个开关必须一起关：前馈的输出层、以及注意力那四个投影。
    只关其中一个都得不到恒等——这正是 day079 5.2 节那个坑在堆叠里的样子。
    """
    resolved_placement = _checked_placement(placement)
    resolved_activation = _checked_activation(activation)
    deflated = StackParameters(
        blocks=tuple(zero_branches(item) for item in params.blocks),
        attentions=tuple(zero_attention(item) for item in params.attentions),
    )
    forward = stack_forward(
        deflated,
        inputs,
        placement=resolved_placement,
        use_residual=True,
        activation=resolved_activation,
    )
    checked = validate_matrix(inputs, name="inputs")
    return _outcome(
        PROPERTY_IDENTITY_WHEN_BRANCHES_VANISH,
        forward.output == checked,
        f"{forward.depth} 层的两个分支全零后输出与输入逐位相等",
        "残差开 + 分支全零 ⇒ 每一层是恒等 ⇒ 整条链是恒等（IEEE 下 x+0 精确）",
    )


def check_parameter_count_matches(
    shape: StackShape, params: StackParameters
) -> PropertyOutcome:
    """解析式算出的参数量与实际构造出来的逐块数一致（整数比较）."""
    analytic = shape.total_parameter_count
    measured = params.parameter_count
    layer_ok = shape.layer_parameter_count * shape.layers == analytic
    depth_ok = params.layers == shape.layers
    return _outcome(
        PROPERTY_PARAMETER_COUNT_MATCHES,
        analytic == measured and layer_ok and depth_ok,
        f"解析式 {analytic}、逐块数 {measured}、层数 {params.layers}",
        "解析式与实测各写一遍：只写一边时，'公式错了'与'构造漏了一块'无法区分",
    )


def check_assembly_carries_shape(
    shape: StackShape, script: str
) -> PropertyOutcome:
    """生成的 PyTorch 脚本带着**同一个**形状（逐键核对 ``CONFIG``）."""
    facts = assembly_facts(script)
    config = facts["config"]
    missing = [key for key in ASSEMBLY_CONFIG_KEYS if key not in config]
    expected = {
        "hidden": shape.hidden,
        "ffn": shape.ffn,
        "tokens": shape.tokens,
        "layers": shape.layers,
        "analytic_total_parameters": shape.total_parameter_count,
    }
    mismatched = [
        key for key, value in expected.items() if config.get(key) != value
    ]
    classes_ok = all(name in facts["classes"] for name in ASSEMBLY_CLASSES)
    modules_ok = all(name in facts["modules"] for name in ASSEMBLY_MODULES)
    return _outcome(
        PROPERTY_ASSEMBLY_CARRIES_SHAPE,
        not missing and not mismatched and classes_ok and modules_ok,
        f"CONFIG 缺 {missing or '无'}、对不上的 {mismatched or '无'}、"
        f"类 {classes_ok}、nn 模块 {modules_ok}",
        f"脚本 {facts['line_count']} 行；d/d_ff/n/N 全部来自这里，源码里没有手工数字",
    )


def check_properties(
    shape: StackShape,
    params: StackParameters,
    inputs: Matrix,
    *,
    placement: str = NORM_PRE,
    use_residual: bool = True,
    activation: str = ACTIVATION_RELU,
    script: str | None = None,
    notes: Sequence[str] = STACK_NOTES,
) -> PropertyReport:
    """跑满六条性质（``script`` 缺省时按 ``shape`` 现生成一份）."""
    resolved_script = (
        assembly_script(shape, placement=placement, activation=activation)
        if script is None
        else script
    )
    forward = stack_forward(
        params,
        inputs,
        placement=placement,
        use_residual=use_residual,
        activation=activation,
    )
    outcomes = (
        check_stack_preserves_shape(forward),
        check_stack_matches_loop(
            params,
            inputs,
            forward,
            placement=placement,
            use_residual=use_residual,
            activation=activation,
        ),
        check_stack_is_deterministic(
            params,
            inputs,
            forward,
            placement=placement,
            use_residual=use_residual,
            activation=activation,
        ),
        check_stack_identity_when_branches_vanishes(
            params, inputs, placement=placement, activation=activation
        ),
        check_parameter_count_matches(shape, params),
        check_assembly_carries_shape(shape, resolved_script),
    )
    return PropertyReport(outcomes=outcomes, notes=tuple(str(item) for item in notes))


def stack_stage_lines() -> tuple[str, ...]:
    """五个阶段各一行（演示脚本开头打印它，教材第 2 章用的是同一张表）."""
    return tuple(
        f"{name:<8} | {STAGE_DESCRIPTIONS[name]} —— {STAGE_SHAPES[name]}"
        for name in STACK_STAGES
    )


def gradient_summary(report: GradientReport) -> str:
    """把一份梯度报告压成一行（演示脚本按节打印它）."""
    if not isinstance(report, GradientReport):
        raise ParameterError(f"report 必须是 GradientReport，收到 {type(report).__name__}。")
    return report.summary_line()


def parse_script_module_names(script: str) -> tuple[str, ...]:
    """从脚本里取出**所有**被调用的属性名（比 ``nn`` 那一族更宽的读数）.

    它存在的理由是 ``assembly_facts`` 只认 ``nn.xxx`` 这一种写法；
    而“那段脚本到底调了哪些名字”在调试生成器时是一个更宽的读数。
    """
    try:
        tree = ast.parse(script)
    except SyntaxError as error:  # pragma: no cover - 生成器不会产出这种文本
        raise ParameterError(f"脚本不是合法 Python：{error}") from error
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    return tuple(sorted(names))


def frobenius_profile(forward: StackForward) -> tuple[float, ...]:
    """逐层出口范数（与 ``StackForward.gain_profile`` 同口径，供演示脚本使用）."""
    if not isinstance(forward, StackForward):
        raise ParameterError(f"forward 必须是 StackForward，收到 {type(forward).__name__}。")
    return tuple(row.output_norm for row in forward.censuses)


def relative_gradient_change(norms: Sequence[float]) -> tuple[float, ...]:
    """逐层梯度的**相邻比值** ``‖dx_{i+1}‖ / ‖dx_i‖``（下标 i 从 0 开始）.

    “第几层开始塌”这句话在一条比值序列上才有定义：比值持续小于 1 的那一段
    就是衰减段。``norms`` 里为零的项会让比值没有定义，本函数对那一处返回 ``0.0``
    （与 ``LayerCensus.gain`` 同一个约定：**不用 nan 污染整张表**）。
    """
    checked = tuple(float(item) for item in norms)
    for value in checked:
        if not math.isfinite(value) or value < 0.0:
            raise NumericError(f"范数序列必须是非负有限数，收到 {value!r}。")
    ratios: list[float] = []
    for index in range(len(checked) - 1):
        previous = checked[index]
        ratios.append(0.0 if previous == 0.0 else checked[index + 1] / previous)
    return tuple(ratios)


def layer_norm_sequence(grads: StackGradients) -> tuple[float, ...]:
    """梯度账里的逐层范数（转发 ``StackGradients.norms``，顺带做一次类型检查）."""
    if not isinstance(grads, StackGradients):
        raise ParameterError(f"grads 必须是 StackGradients，收到 {type(grads).__name__}。")
    return grads.norms()


def census_of_layer(forward: StackForward, index: int) -> dict[str, Any]:
    """第 ``index`` 层读数的字典形状（演示脚本逐层打印它）."""
    return forward.census_at(index).to_dict()


def make_samples(
    *,
    layers: int = 4,
    hidden: int = 6,
    ffn_ratio: int = 4,
    tokens: int = 4,
    seed: int = 7,
) -> tuple[StackShape, StackParameters, Matrix, Matrix]:
    """一套固定的样本：形状 + 参数 + 输入 + 目标（演示与测试共用同一组）."""
    from smart_research_agent.transformer_stack.layers import make_shape

    shape = make_shape(layers=layers, hidden=hidden, tokens=tokens, ffn_ratio=ffn_ratio)
    params = make_stack_parameters(shape, seed=seed)
    inputs = tuple(
        tuple(float((row * hidden + column) % 7) / 7.0 - 0.4 for column in range(hidden))
        for row in range(tokens)
    )
    target = tuple(
        tuple(float((row * hidden + column) % 5) / 5.0 - 0.3 for column in range(hidden))
        for row in range(tokens)
    )
    return shape, params, inputs, target


def matrix_norm_of(matrix: Matrix) -> float:
    """一个矩阵的 Frobenius 范数（转发 :func:`types.frobenius`）."""
    return frobenius(matrix)


def checked_vector(values: Sequence[float]) -> Vector:
    """把一串数校验成 :data:`Vector`（演示脚本读外部数据时用）."""
    return validate_vector(values, name="values")


__all__ = [
    "GRADIENT_TOLERANCE",
    "STAGE_ITEMS",
    "GradientOutcome",
    "GradientReport",
    "PropertyOutcome",
    "PropertyReport",
    "census_of_layer",
    "check_assembly_carries_shape",
    "check_layer_parameter_gradients",
    "check_parameter_count_matches",
    "check_properties",
    "check_stack_identity_when_branches_vanishes",
    "check_stack_input_gradient",
    "check_stack_is_deterministic",
    "check_stack_matches_loop",
    "check_stack_preserves_shape",
    "checked_vector",
    "frobenius_profile",
    "gradient_summary",
    "layer_norm_sequence",
    "make_samples",
    "matrix_norm_of",
    "numerical_layer_parameter_gradients",
    "numerical_stack_input_gradient",
    "parse_script_module_names",
    "relative_gradient_change",
    "stack_stage_lines",
]
