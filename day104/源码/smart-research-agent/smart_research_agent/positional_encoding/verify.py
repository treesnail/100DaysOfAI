"""``positional_encoding`` 的梯度校验与性质检查（day078 / M7-D3）.

```text
五项梯度校验（day075）    w_output / w_value / w_query / w_key / inputs
六项梯度校验（本课）      上面五项 + **table**（位置表）
```

多出来的那一项就是这一课的全部内容：**加法注入的反向只有两步，而两步都容易写错**。

```text
dx         = dInjected          （**逐位**：加法注入的偏导数恰好是 1）
dTable[p]  = Σ_{i: positions[i]=p} dInjected[i]（**按位置累加**）
```

第二步有一个“不报错”的错误写法（按行覆盖），它只在**位置重复**时出错——
一批两条序列、或者一条序列里同一个位置被用了两次，都会触发。
本模块因此把“覆盖写法会差多少”也做成一条读数（第 5 条性质）。

## 七条性质与它们各自的**前提**

| 性质 | 前提 | 失败意味着 |
|------|------|-----------|
| `rows_have_constant_norm` | 对齐频率的偶数维正弦表 | 正弦表的公式配对错了（``sin² + cos² ≠ 1``） |
| `dot_product_depends_only_on_offset` | 同上 | “相对距离”不再能从编码里读出来 |
| `shift_is_an_orthogonal_rotation` | 同上 | 上一条的构造性理由不成立（位移不再是旋转） |
| `additive_injection_breaks_permutation_equivariance` | **非因果**前向 | 位置编码没有起作用（换序仍然等变） |
| `additive_backward_is_the_identity` | 任何表 | ``dx`` 被缩放 / 表梯度按行覆盖而不是累加 |
| `learnable_table_length_is_a_hard_bound` | 任何表 | 越界被静默截断（模型看到另一个位置） |
| `staggered_pairing_breaks_both_laws` | 对齐频率的偶数维正弦表 | 判据太松：错位实现也能“通过” |

前三条与第七条只对**对齐频率的正弦表**成立——用可学习表或错位表时它们会失败，
而**那是正确的结果**（报告里那几条的 ``passed=False`` 正是“这张表不满足它”的读数）。
其余四条对任何表都成立。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from smart_research_agent.math_foundations.calculus import DEFAULT_STEP, gradient
from smart_research_agent.math_foundations.optim import flatten_matrices, unflatten_matrices
from smart_research_agent.math_foundations.types import (
    Matrix,
    Vector,
    matrix_shape,
    validate_matrix,
)
from smart_research_agent.positional_encoding.errors import (
    GradientError,
    NumericError,
    ParameterError,
    RangeError,
    ShapeError,
)
from smart_research_agent.positional_encoding.layers import (
    closed_form_offset_inner,
    positional_backward,
    positional_forward,
    positional_loss_from_flat,
    rotate,
    sinusoidal_table,
    staggered_table,
)
from smart_research_agent.positional_encoding.types import (
    OFFSET_TOLERANCE,
    POSITIONAL_GRADIENT_DESCRIPTIONS,
    POSITIONAL_GRADIENT_FORMULAS,
    POSITIONAL_GRADIENT_TARGETS,
    POSITIONAL_PROPERTIES,
    POSITIONAL_PROPERTY_DESCRIPTIONS,
    PROPERTY_ADDITIVE_BACKWARD,
    PROPERTY_BREAKS_EQUIVARIANCE,
    PROPERTY_CONSTANT_NORM,
    PROPERTY_LENGTH_IS_A_HARD_BOUND,
    PROPERTY_OFFSET_ONLY,
    PROPERTY_SHIFT_IS_ROTATION,
    PROPERTY_STAGGERED_BREAKS_THE_LAWS,
    ROW_NORM_TOLERANCE,
    EncodingTable,
    PositionalForward,
    PositionalGradients,
    flatten_parameters,
    validate_settings,
)
from smart_research_agent.transformer_core.layers import attention_backward, self_attention
from smart_research_agent.transformer_core.types import (
    AttentionParams,
    relative_matrix_error,
)
from smart_research_agent.transformer_core.verify import GRADIENT_TOLERANCE as CORE_TOLERANCE

#: 梯度校验的容差（**与 day075/076 同值**，导入期把三者钉成相等）.
GRADIENT_TOLERANCE = 1e-6

if GRADIENT_TOLERANCE != CORE_TOLERANCE:  # pragma: no cover - 只在有人改常量时触发
    raise NumericError(
        f"本课的梯度容差 {GRADIENT_TOLERANCE} 与 day075 的 {CORE_TOLERANCE} 不一致："
        "三天用同一条判据，'这次的误差比上次大'才是一个有意义的比较。"
    )

#: 位移律的判据（与 ``types.OFFSET_TOLERANCE`` 同值，此处重新导出便于阅读）.
OFFSET_LAW_TOLERANCE = OFFSET_TOLERANCE

#: 旋转恒等式的判据：``R_δ·PE(p)`` 与 ``PE(p+δ)`` 是两条不同的求和路径，差最后几位.
ROTATION_TOLERANCE = 1e-12


def permute_rows(matrix: Matrix, permutation: Sequence[int]) -> Matrix:
    """按置换把行重排（``permutation[i]`` 是“新第 i 行来自旧的第几行”）."""
    checked = validate_matrix(matrix, name="matrix")
    rows = matrix_shape(checked)[0]
    if len(permutation) != rows:
        raise ShapeError(f"置换长度 {len(permutation)} 与行数 {rows} 不一致。")
    if sorted(permutation) != list(range(rows)):
        raise ParameterError(f"置换 {tuple(permutation)} 不是 0..{rows - 1} 的一个排列。")
    return tuple(checked[index] for index in permutation)


def default_permutation(rows: int) -> tuple[int, ...]:
    """缺省置换：**倒序**（它同时也是“最不像恒等”的那个置换）."""
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ParameterError(f"rows 必须是 >= 1 的整数，收到 {rows!r}。")
    return tuple(reversed(range(rows)))


def _scaled_error(left: float, right: float) -> float:
    """逐点相对误差（分母 ``max(1, |左|, |右|)``）——与 day075 的口径一致."""
    if not (math.isfinite(left) and math.isfinite(right)):
        return math.inf
    return abs(left - right) / max(1.0, abs(left), abs(right))


def _compare(target: str, analytic: Matrix, numeric: Matrix) -> tuple[float, float, int]:
    """比较两块梯度，返回 ``(最大绝对误差, 最大相对误差, 比较的逐点数)``."""
    left = validate_matrix(analytic, name="analytic")
    right = validate_matrix(numeric, name="numeric")
    if matrix_shape(left) != matrix_shape(right):
        raise ShapeError(
            f"{target}：解析梯度 {matrix_shape(left)} 与数值梯度 {matrix_shape(right)} "
            "形状不一致——形状不同时“哪一项对不上”这个问题本身没有定义。"
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
class PositionalGradientOutcome:
    """一项梯度校验的读数.

    ```text
    target                校验项（w_output / w_value / … / table / inputs）
    max_absolute_error    最大绝对差 |解析 − 数值|
    max_scaled_error      最大相对差（分母 max(1, |解析|, |数值|)）
    tolerance             这一次比较用的判据
    compared_points       比较了多少个逐点
    message               失败时的一句话（通过时为空串）
    ```
    """

    target: str
    max_absolute_error: float
    max_scaled_error: float
    tolerance: float
    compared_points: int
    message: str = ""

    def __post_init__(self) -> None:
        if self.target not in POSITIONAL_GRADIENT_TARGETS:
            raise ParameterError(
                f"未知的梯度校验项 {self.target!r}：可选 "
                f"{', '.join(POSITIONAL_GRADIENT_TARGETS)}。"
            )
        for name in ("max_absolute_error", "max_scaled_error"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or math.isnan(value) or value < 0:
                raise NumericError(f"{name} 必须是非负的数，收到 {value!r}。")
        validate_settings(self.tolerance)
        if not isinstance(self.compared_points, int) or self.compared_points < 0:
            raise NumericError(f"compared_points 必须是非负整数，收到 {self.compared_points!r}。")
        object.__setattr__(self, "message", str(self.message))

    @property
    def passed(self) -> bool:
        """这一项是否通过（**用相对误差与容差比**）."""
        return self.max_scaled_error <= self.tolerance

    @property
    def formula(self) -> str:
        """这一项的公式（来自 ``POSITIONAL_GRADIENT_FORMULAS``）."""
        return POSITIONAL_GRADIENT_FORMULAS[self.target]

    @property
    def description(self) -> str:
        """这一项在守什么（来自 ``POSITIONAL_GRADIENT_DESCRIPTIONS``）."""
        return POSITIONAL_GRADIENT_DESCRIPTIONS[self.target]

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "target": self.target,
            "passed": self.passed,
            "max_absolute_error": self.max_absolute_error,
            "max_scaled_error": self.max_scaled_error,
            "tolerance": self.tolerance,
            "compared_points": self.compared_points,
            "formula": self.formula,
            "message": self.message,
        }

    def summary_line(self) -> str:
        """一行说明：``[ok] w_query 最大相对误差 1.5e-11（比较 36 个逐点，容差 1e-06）``."""
        mark = "ok" if self.passed else "!!"
        return (
            f"[{mark}] {self.target:<9} 最大相对误差 {self.max_scaled_error:.2e}"
            f"（绝对 {self.max_absolute_error:.2e}，比较 {self.compared_points} 个逐点，"
            f"容差 {self.tolerance:g}）"
        )


@dataclass(frozen=True)
class PositionalGradientReport:
    """六项梯度校验的报告（**它属于控制流**：不通过就该停下来）."""

    outcomes: tuple[PositionalGradientOutcome, ...]
    tolerance: float = GRADIENT_TOLERANCE
    step: float = DEFAULT_STEP
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        resolved = tuple(self.outcomes)
        if not resolved:
            raise ParameterError("梯度报告不能为空：没有校验项的报告没有意义。")
        names = {item.target for item in resolved}
        if names != set(POSITIONAL_GRADIENT_TARGETS):
            missing = sorted(set(POSITIONAL_GRADIENT_TARGETS) - names)
            extra = sorted(names - set(POSITIONAL_GRADIENT_TARGETS))
            raise NumericError(
                f"梯度报告的校验项与名单不一致：缺 {missing}、多 {extra}——"
                "少一项的报告在'全绿'时看起来与完整报告一模一样。"
            )
        validate_settings(self.tolerance)
        object.__setattr__(self, "outcomes", resolved)
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def ok(self) -> bool:
        """六项是否全部通过."""
        return all(item.passed for item in self.outcomes)

    @property
    def failures(self) -> tuple[PositionalGradientOutcome, ...]:
        """没通过的那些项."""
        return tuple(item for item in self.outcomes if not item.passed)

    @property
    def worst_scaled_error(self) -> float:
        """六项里最大的相对误差（“这次反向离数值差分有多远”）."""
        return max(item.max_scaled_error for item in self.outcomes)

    def raise_if_failed(self) -> None:
        """有任意一项不通过就抛 :class:`GradientError`.

        梯度校验属于**控制流**而不是报告：一条对不上的梯度意味着这次训练从头到尾
        都不可信——把它记在报告里继续往下跑，只会得到一条“看起来很平滑”的错误曲线。
        """
        if not self.ok:
            detail = "；".join(
                f"{item.target}={item.max_scaled_error:.2e}" for item in self.failures
            )
            raise GradientError(
                f"梯度校验未通过（容差 {self.tolerance:g}）：{detail}——"
                "先确认两边算的是**同一个函数**（损失口径、位置序列、掩码），再怀疑推导。"
            )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "ok": self.ok,
            "tolerance": self.tolerance,
            "step": self.step,
            "worst_scaled_error": self.worst_scaled_error,
            "outcomes": [item.to_dict() for item in self.outcomes],
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``梯度校验 6 项：通过 6、失败 0 | 最大相对误差 7.4e-11``."""
        passed = sum(1 for item in self.outcomes if item.passed)
        return (
            f"梯度校验 {len(self.outcomes)} 项：通过 {passed}、失败 "
            f"{len(self.outcomes) - passed} | 最大相对误差 {self.worst_scaled_error:.2e}"
            f"（容差 {self.tolerance:g}）"
        )


def numerical_positional_gradients(
    params: AttentionParams,
    table: EncodingTable,
    inputs: Matrix,
    *,
    target: Matrix,
    positions: tuple[int, ...] | None = None,
    supervised: Sequence[int] | None = None,
    causal: bool = False,
    step: float = DEFAULT_STEP,
) -> PositionalGradients:
    """用**中心差分**算出六块“完整”的梯度（数值侧的唯一入口）.

    两趟差分：

    ```text
    第一趟  扰动 (四个投影 + 位置表) —— 五块，由 flatten_parameters 压成一串
    第二趟  扰动 inputs           —— 一块，由 flatten_matrices 压成一串
    ```

    为什么分成两趟而不是把 inputs 也压进同一串：**inputs 不是参数**。
    把它混进参数向量会让“优化器在更新什么”这件事变得含糊，
    也会让“表那一项”的位置随着样本长度而漂移——而它的位置正是这一课要盯的东西。
    """
    objective = positional_loss_from_flat(
        params,
        table,
        inputs,
        positions=positions,
        target=target,
        supervised=supervised,
        causal=causal,
    )
    _flat, shapes = flatten_parameters(params, table)
    numeric_flat = gradient(objective, _flat, step=step)
    numeric_params, numeric_table = numeric_params_resolve(numeric_flat, shapes)

    checked_inputs = validate_matrix(inputs, name="inputs")
    flat_inputs, input_shapes = flatten_matrices((checked_inputs,))

    def input_objective(theta: Vector) -> float:
        """给定压平后的 inputs，返回这一次前向的损失."""
        rebuilt = unflatten_matrices(theta, input_shapes)[0]
        forward = positional_forward(
            params,
            table,
            rebuilt,
            positions=positions,
            target=target,
            supervised=supervised,
            causal=causal,
        )
        if forward.loss is None:  # pragma: no cover - target 已给定
            raise NumericError("损失没有被算出来。")
        return forward.loss

    numeric_inputs = unflatten_matrices(
        gradient(input_objective, flat_inputs, step=step), input_shapes
    )[0]
    return PositionalGradients(
        grad_table=numeric_table,
        grad_inputs=numeric_inputs,
        grad_w_query=numeric_params.w_query,
        grad_w_key=numeric_params.w_key,
        grad_w_value=numeric_params.w_value,
        grad_w_output=numeric_params.w_output,
    )


def numeric_params_resolve(
    flat: Vector, shapes: tuple[tuple[int, int], ...]
) -> tuple[AttentionParams, Matrix]:
    """把压平的参数还原成“四个投影 + 表”（数值侧的内部辅助）."""
    matrices = unflatten_matrices(flat, shapes)
    if len(matrices) != 5:
        raise ShapeError(f"参数块个数 {len(matrices)} 与预期的 5 个不一致。")
    params = AttentionParams(
        w_query=matrices[0],
        w_key=matrices[1],
        w_value=matrices[2],
        w_output=matrices[3],
    )
    return params, matrices[4]


def check_positional_gradients(
    params: AttentionParams,
    table: EncodingTable,
    inputs: Matrix,
    *,
    target: Matrix,
    positions: tuple[int, ...] | None = None,
    supervised: Sequence[int] | None = None,
    causal: bool = False,
    step: float = DEFAULT_STEP,
    tolerance: float = GRADIENT_TOLERANCE,
) -> PositionalGradientReport:
    """六项逐点对照（解析 vs 数值），返回一份报告.

    **三处口径必须两边一致**（day076 的教训，本课再添一处）：

    ```text
    损失口径   全行 MSE vs 监督行 MSE        —— 本包用 loss_gradient() 统一，两处不可能分家
    positions  解析侧给了显式位置序列而数值侧没有 —— 两边算的是“不同的注入”
    causal     一个因果、一个全开            —— 两边算的是“不同的函数”
    ```

    后两条都在同一个函数里由**同一组参数**传给两侧，因此“忘了传”这件事在结构上
    不会发生——这是 day076 那次返工留下的最直接的改进。
    """
    validate_settings(tolerance)
    forward = positional_forward(
        params,
        table,
        inputs,
        positions=positions,
        target=target,
        supervised=supervised,
        causal=causal,
    )
    analytic = positional_backward(forward, loss_gradient_of(forward))
    numeric = numerical_positional_gradients(
        params,
        table,
        inputs,
        target=target,
        positions=positions,
        supervised=supervised,
        causal=causal,
        step=step,
    )
    analytic_blocks = analytic.as_dict()
    numeric_blocks = numeric.as_dict()
    outcomes: list[PositionalGradientOutcome] = []
    for name in POSITIONAL_GRADIENT_TARGETS:
        worst_absolute, worst_scaled, points = _compare(
            name, analytic_blocks[name], numeric_blocks[name]
        )
        message = ""
        if worst_scaled > tolerance:
            message = (
                f"{name} 的解析梯度与数值差分差 {worst_scaled:.2e}（容差 {tolerance:g}）："
                "先确认两边算的是同一个函数（损失口径 / 位置序列 / 掩码），再怀疑推导。"
            )
        outcomes.append(
            PositionalGradientOutcome(
                target=name,
                max_absolute_error=worst_absolute,
                max_scaled_error=worst_scaled,
                tolerance=tolerance,
                compared_points=points,
                message=message,
            )
        )
    notes = [
        f"数值侧用中心差分，步长 {step:g}（day074 的 calculus.gradient）",
        "两侧共用同一个前向构造函数：位置序列与掩码口径不可能分家",
    ]
    if positions is not None and len(set(positions)) != len(positions):
        notes.append(
            "位置序列里有重复：**表的那一项**在这里才真正被区分——"
            "按行覆盖的写法会与数值差分对不上"
        )
    return PositionalGradientReport(
        outcomes=tuple(outcomes), tolerance=tolerance, step=step, notes=tuple(notes)
    )


def loss_gradient_of(forward: PositionalForward) -> Matrix:
    """损失对输出的梯度（转发 :func:`layers.loss_gradient`，便于本模块单点引用）."""
    from smart_research_agent.positional_encoding.layers import loss_gradient

    return loss_gradient(forward)


# ---------------------------------------------------------------------- 性质


@dataclass(frozen=True)
class PositionalPropertyOutcome:
    """一条性质的读数."""

    name: str
    passed: bool
    evidence: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.name not in POSITIONAL_PROPERTIES:
            raise ParameterError(
                f"未知的性质名 {self.name!r}：可选 {', '.join(POSITIONAL_PROPERTIES)}。"
            )
        object.__setattr__(self, "evidence", str(self.evidence))
        object.__setattr__(self, "detail", str(self.detail))

    @property
    def description(self) -> str:
        """这一条在说什么（来自 ``POSITIONAL_PROPERTY_DESCRIPTIONS``）."""
        return POSITIONAL_PROPERTY_DESCRIPTIONS[self.name]

    @property
    def skipped(self) -> bool:
        """这一次是否**跳过**（“没有出现过的状态”与“处理了且通过”必须分开）."""
        return "跳过" in self.evidence

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "name": self.name,
            "passed": self.passed,
            "skipped": self.skipped,
            "evidence": self.evidence,
            "detail": self.detail,
            "description": self.description,
        }

    def summary_line(self) -> str:
        """一行说明：``[ok] rows_have_constant_norm 最大偏差 0.00e+00（检查 4 行）``."""
        mark = "skip" if self.skipped else ("ok" if self.passed else "!!")
        return f"[{mark}] {self.name} {self.evidence}"


@dataclass(frozen=True)
class PositionalPropertyReport:
    """七条性质的报告."""

    outcomes: tuple[PositionalPropertyOutcome, ...]
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        resolved = tuple(self.outcomes)
        if not resolved:
            raise ParameterError("性质报告不能为空。")
        names = {item.name for item in resolved}
        if names != set(POSITIONAL_PROPERTIES):
            missing = sorted(set(POSITIONAL_PROPERTIES) - names)
            raise NumericError(f"性质报告的名单不完整：缺 {missing}——缺项的'全绿'是假的。")
        object.__setattr__(self, "outcomes", resolved)
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def ok(self) -> bool:
        """所有**非跳过**的性质是否都通过."""
        return all(item.passed for item in self.outcomes)

    @property
    def failures(self) -> tuple[PositionalPropertyOutcome, ...]:
        """没通过的那些条（跳过的算通过）."""
        return tuple(item for item in self.outcomes if not item.passed)

    @property
    def skipped(self) -> tuple[PositionalPropertyOutcome, ...]:
        """这一次被跳过的那些条."""
        return tuple(item for item in self.outcomes if item.skipped)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "ok": self.ok,
            "outcomes": [item.to_dict() for item in self.outcomes],
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``性质检查 7 项：通过 7、失败 0、跳过 0``."""
        passed = sum(1 for item in self.outcomes if item.passed and not item.skipped)
        return (
            f"性质检查 {len(self.outcomes)} 项：通过 {passed}、失败 {len(self.failures)}、"
            f"跳过 {len(self.skipped)}"
        )


def check_row_norm_is_constant(
    table: EncodingTable, *, tolerance: float = ROW_NORM_TOLERANCE
) -> PositionalPropertyOutcome:
    """前三条的公共前提：**对齐频率的偶数维正弦表**才有 ``sin² + cos² = 1``."""
    validate_settings(tolerance)
    target = math.sqrt(table.dimension / 2.0)
    worst = max(abs(value - target) for value in table.norms)
    return PositionalPropertyOutcome(
        name=PROPERTY_CONSTANT_NORM,
        passed=worst <= tolerance,
        evidence=(
            f"检查 {table.positions} 行（{table.kind}/{table.pairing or '—'}），"
            f"每行范数与 √(d/2)={target:.6f} 的最大偏差 {worst:.2e}（容差 {tolerance:g}）"
        ),
        detail=(
            "只有对齐频率时才有 sin² + cos² = 1；"
            "错位频率（day073 的写法）会让范数随位置变化"
        ),
    )


def check_offset_law(
    table: EncodingTable,
    *,
    offsets: Sequence[int] = (1, 2, 3, 5),
    tolerance: float = OFFSET_LAW_TOLERANCE,
) -> PositionalPropertyOutcome:
    """``<PE(p), PE(q)>`` 只依赖 ``p − q``（位移律），并与闭式对得上.

    每一对 ``(p, p+δ)`` 的实测内积都要等于同一个数 ``Σ_i cos(δ/f_i)``——
    前者是“两行做点积”，后者是按频率对求和，**两条不同的求和路径**。
    """
    validate_settings(tolerance)
    if not table.pairing_is_aligned:
        return PositionalPropertyOutcome(
            name=PROPERTY_OFFSET_ONLY,
            passed=False,
            evidence=(
                f"本次表是 {table.kind}/{table.pairing or '—'}："
                "位移律的闭式只对**对齐频率的正弦表**成立（跳过不成立）"
            ),
            detail="可学习表的行是自由参数、错位表的 sin 与 cos 不在同一频率上",
        )
    if table.base is None:  # pragma: no cover - 对齐正弦表必然带 base
        raise NumericError("对齐正弦表缺少频率基数。")
    worst = 0.0
    pairs = 0
    for offset in offsets:
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 1:
            raise ParameterError(f"offset 必须是 >= 1 的整数，收到 {offset!r}。")
        closed = closed_form_offset_inner(
            table.dimension, offset, base=table.base
        )
        for start in range(table.positions - offset):
            worst = max(worst, abs(table.inner_product(start, start + offset) - closed))
            pairs += 1
    return PositionalPropertyOutcome(
        name=PROPERTY_OFFSET_ONLY,
        passed=worst <= tolerance,
        evidence=(
            f"检查 {pairs} 对位置（δ ∈ {tuple(offsets)}），"
            f"与闭式 Σ_i cos(δ/f_i) 的最大偏差 {worst:.2e}（容差 {tolerance:g}）"
        ),
        detail="实测是'两行做点积'、闭式是'按频率对求和'——两条路径必须给出同一个数",
    )


def check_shift_is_rotation(
    table: EncodingTable,
    *,
    offsets: Sequence[int] = (1, 2, 3),
    tolerance: float = ROTATION_TOLERANCE,
) -> PositionalPropertyOutcome:
    """``PE(p + δ)`` 逐位等于 ``R_δ · PE(p)``——位移律的**构造性理由**.

    它比上一条更强：上一条只说“内积一样”，这一条说“整行向量是同一批向量旋转过去”。
    因此这一条通过之后，位移律**不可能**不成立（``R_δ`` 是正交的）。
    """
    validate_settings(tolerance)
    if not table.pairing_is_aligned:
        return PositionalPropertyOutcome(
            name=PROPERTY_SHIFT_IS_ROTATION,
            passed=False,
            evidence=(
                f"本次表是 {table.kind}/{table.pairing or '—'}："
                "不存在一个统一的 ω 让 sin/cos 一起旋转（跳过不成立）"
            ),
            detail="旋转因子要求每一对 sin/cos 用同一个频率",
        )
    worst = 0.0
    checked = 0
    for offset in offsets:
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 1:
            raise ParameterError(f"offset 必须是 >= 1 的整数，收到 {offset!r}。")
        rotated = rotate(table, offset)
        for start in range(table.positions - offset):
            for a, b in zip(rotated[start], table.row(start + offset), strict=True):
                worst = max(worst, abs(a - b))
                checked += 1
    return PositionalPropertyOutcome(
        name=PROPERTY_SHIFT_IS_ROTATION,
        passed=worst <= tolerance,
        evidence=(
            f"检查 {checked} 个逐点（δ ∈ {tuple(offsets)}），"
            f"|R_δ·PE(p) − PE(p+δ)| 的最大值 {worst:.2e}（容差 {tolerance:g}）"
        ),
        detail="R_δ 是分块正交旋转 ⇒ 它保持范数与内积 ⇒ 位移律自动成立",
    )


@dataclass(frozen=True)
class SymmetryBreak:
    """“位置编码打破了置换等变性”这件事的读数.

    ```text
    permutation    这一次用的置换
    base_gap       不加位置编码时的置换缺口（**恰好 0.0**——day075 的那条性质）
    injected_gap   加了位置编码之后的置换缺口（> 0）
    ```

    两个数**必须一起看**：单看 ``injected_gap > 0`` 无法区分
    “位置编码起了作用”与“这一次前向本来就不等变”（比如开了因果掩码）。
    """

    permutation: tuple[int, ...]
    base_gap: float
    injected_gap: float
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        for name in ("base_gap", "injected_gap"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or math.isnan(value) or value < 0:
                raise NumericError(f"{name} 必须是非负的数，收到 {value!r}。")
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def broken(self) -> bool:
        """基准层**逐位**等变、而注入之后不再等变——这才是“位置编码起了作用”."""
        return self.base_gap == 0.0 and self.injected_gap > 1e-6

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "permutation": list(self.permutation),
            "base_gap": self.base_gap,
            "injected_gap": self.injected_gap,
            "broken": self.broken,
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``置换 (3, 2, 1, 0) | 无位置编码缺口 0.00e+00 | 注入后缺口 1.23e+00``."""
        return (
            f"置换 {tuple(self.permutation)} | 无位置编码缺口 {self.base_gap:.2e} | "
            f"注入后缺口 {self.injected_gap:.2e}"
        )


def symmetry_break(
    params: AttentionParams,
    table: EncodingTable,
    inputs: Matrix,
    *,
    positions: tuple[int, ...] | None = None,
    permutation: Sequence[int] | None = None,
) -> SymmetryBreak:
    """量一次“注入前后置换等变性的差别”（**不开掩码**）.

    ```text
    基准层   g(x) = self_attention(params, x)          → ‖g(πx) − π g(x)‖ 恰好 0.0
    本层     f(x) = positional_forward(params, table, x) → ‖f(πx) − π f(x)‖ > 0
    ```

    基准层的缺口**恰好**是 0（不是“很小”）：无掩码时 ``QKᵀ`` 只是行列跟着换，
    每一行 softmax 的求和项**顺序完全相同**，因此浮点上也逐位相等。
    """
    checked_inputs = validate_matrix(inputs, name="inputs")
    rows = matrix_shape(checked_inputs)[0]
    resolved = (
        default_permutation(rows)
        if permutation is None
        else tuple(permutation)
    )
    if sorted(resolved) != list(range(rows)):
        raise ParameterError(f"置换 {resolved} 不是 0..{rows - 1} 的一个排列。")
    permuted_inputs = permute_rows(checked_inputs, resolved)
    base = self_attention(params, checked_inputs, causal=False)
    base_permuted = self_attention(params, permuted_inputs, causal=False)
    base_gap = relative_matrix_error(permute_rows(base.output, resolved), base_permuted.output)
    injected = positional_forward(
        params, table, checked_inputs, positions=positions, causal=False
    )
    injected_permuted = positional_forward(
        params, table, permuted_inputs, positions=positions, causal=False
    )
    injected_gap = relative_matrix_error(
        permute_rows(injected.attention.output, resolved), injected_permuted.attention.output
    )
    return SymmetryBreak(
        permutation=resolved,
        base_gap=base_gap,
        injected_gap=injected_gap,
        notes=(
            "基准层逐位为 0：无掩码时注意力是置换等变的（day075 的性质）",
            "注入后非 0：位置那部分是**按行号**加进去的，换序之后它没有跟着换",
        ),
    )


def check_injection_breaks_equivariance(
    params: AttentionParams,
    table: EncodingTable,
    inputs: Matrix,
    *,
    positions: tuple[int, ...] | None = None,
    causal: bool = False,
    permutation: Sequence[int] | None = None,
) -> PositionalPropertyOutcome:
    """注入之后置换**不再**等变——**这是本课要的效果，不是 bug**.

    ``causal=True`` 时返回 **“跳过”**：因果掩码本身就已经让基准层不等变了，
    “位置编码有没有起作用”在那里无法被单独验证
    （“没有出现过的状态”与“处理了且通过”必须分开，这是 day075/076 同一条纪律）。
    """
    if causal:
        return PositionalPropertyOutcome(
            name=PROPERTY_BREAKS_EQUIVARIANCE,
            passed=True,
            evidence="本次前向启用了因果掩码（跳过）——基准层本身就不再等变，无法单独验证",
            detail="要验证这一条请用无掩码的前向",
        )
    report = symmetry_break(
        params, table, inputs, positions=positions, permutation=permutation
    )
    return PositionalPropertyOutcome(
        name=PROPERTY_BREAKS_EQUIVARIANCE,
        passed=report.broken,
        evidence=f"{report.summary_line()}（基准层逐位为 0 要求 == 0.0）",
        detail="判据是双向的：基准层必须**恰好** 0.0，注入后必须严格大于 0",
    )


def _accumulated_table_gradient(
    row_gradients: Matrix,
    positions: tuple[int, ...],
    *,
    table_positions: int,
) -> Matrix:
    """按位置累加的**独立实现**（用 ``math.fsum`` + 逐位置筛选，求和顺序也不同）.

    它与 :func:`types.scatter_add_rows` 是两条路径，因此可以用它交叉验证后者。
    """
    checked = validate_matrix(row_gradients, name="row_gradients")
    width = matrix_shape(checked)[1]
    rows: list[Vector] = []
    for position in range(table_positions):
        chosen = [
            checked[index] for index, value in enumerate(positions) if value == position
        ]
        rows.append(
            tuple(
                math.fsum(row[column] for row in chosen) if chosen else 0.0
                for column in range(width)
            )
        )
    return tuple(rows)


def _overwrite_table_gradient(
    row_gradients: Matrix,
    positions: tuple[int, ...],
    *,
    table_positions: int,
) -> Matrix:
    """**故意写错的**版本：按行覆盖（最后一次写入获胜）——只用来量“会差多少”."""
    checked = validate_matrix(row_gradients, name="row_gradients")
    width = matrix_shape(checked)[1]
    rows: list[list[float]] = [[0.0] * width for _ in range(table_positions)]
    for row, position in zip(checked, positions, strict=True):
        rows[position] = list(row)
    return tuple(tuple(row) for row in rows)


def check_additive_backward(forward: PositionalForward) -> PositionalPropertyOutcome:
    """加法注入的反向：``dx`` **逐位**传回、``dTable`` **按位置累加**.

    三条判据一起给出，因为它们各自能挡住一种“不报错”的错误写法：

    ```text
    ① dx == dInjected（逐位）     写成 dx = κ·dInjected 时形状全对、只是梯度被缩放
    ② dTable 与独立实现一致        写成“按行覆盖”时形状全对、只在位置重复时出错
    ③ 覆盖写法与累加写法的差       位置互不相同时必须**恰好为 0**（否则判据太松）
    ```
    """
    if forward.target is None:
        raise ParameterError(
            "这一份前向没有 target：没有目标就没有损失梯度，这一条性质无法检查。"
        )
    grad = loss_gradient_of(forward)
    analytic = positional_backward(forward, grad)
    inner = attention_backward(forward.attention, grad)
    identity_gap = max(
        abs(a - b)
        for left, right in zip(analytic.grad_inputs, inner.grad_inputs, strict=True)
        for a, b in zip(left, right, strict=True)
    )
    independent = _accumulated_table_gradient(
        inner.grad_inputs, forward.positions, table_positions=forward.table.positions
    )
    accumulate_gap = relative_matrix_error(analytic.grad_table, independent)
    overwrite = _overwrite_table_gradient(
        inner.grad_inputs, forward.positions, table_positions=forward.table.positions
    )
    overwrite_gap = relative_matrix_error(analytic.grad_table, overwrite)
    duplicates = len(forward.positions) - len(set(forward.positions))
    consistent = overwrite_gap == 0.0 if duplicates == 0 else overwrite_gap > 1e-6
    passed = identity_gap == 0.0 and accumulate_gap <= 1e-12 and consistent
    return PositionalPropertyOutcome(
        name=PROPERTY_ADDITIVE_BACKWARD,
        passed=passed,
        evidence=(
            f"dx 与 dInjected 的最大偏差 {identity_gap:.2e}（要求 **恰好 0.0**）；"
            f"dTable 与独立实现的最大相对偏差 {accumulate_gap:.2e}；"
            f"重复位置 {duplicates} 处，与'按行覆盖'的差 {overwrite_gap:.2e}"
        ),
        detail=(
            "位置互不相同时覆盖写法**恰好**给出同样的表梯度（差 0.00e+00），"
            "位置重复时两者必然不同——这就是那个'漏掉不会报错'的坑"
        ),
    )


def check_table_length_is_a_hard_bound(table: EncodingTable) -> PositionalPropertyOutcome:
    """可学习表对越界位置**拒绝**，而正弦表对同一个位置**算得出来**.

    这一条不是防御性代码，而是**两种编码的语义差别**：
    正弦表的每一行是一个公式的取值，因此表长之外仍然有定义；
    可学习表的每一行是一个参数，因此“第 20 行的参数”根本不存在。
    """
    probe = table.positions + 50
    if table.kind == "sinusoidal":
        if not table.extends_to(probe):
            return PositionalPropertyOutcome(
                name=PROPERTY_LENGTH_IS_A_HARD_BOUND,
                passed=False,
                evidence=f"正弦表声称不支持第 {probe} 位——但它的每一行都是公式的取值",
                detail="正弦表的 extends_to() 应当对任意非负位置返回 True",
            )
        row = table.table[0]
        fresh = sinusoidal_table(
            probe + 1, table.dimension, base=table.base or 10000.0
        ).row(probe)
        passed = len(fresh) == len(row)
        return PositionalPropertyOutcome(
            name=PROPERTY_LENGTH_IS_A_HARD_BOUND,
            passed=passed,
            evidence=(
                f"正弦表：第 {table.positions} 行之外的第 {probe} 位**算得出来**"
                f"（{len(fresh)} 维），表长不是边界"
            ),
            detail="外推是算出来的，不是猜出来的——因为每一行都是公式的取值",
        )
    try:
        table.row(probe)
    except RangeError:
        return PositionalPropertyOutcome(
            name=PROPERTY_LENGTH_IS_A_HARD_BOUND,
            passed=True,
            evidence=(
                f"可学习表：第 {probe} 位抛 RangeError（表只有 {table.positions} 行）——"
                "拒绝而不是外推"
            ),
            detail="越界被静默截断会让模型看到**另一个位置**，而报告里仍然写着'我查了第 20 位'",
        )
    return PositionalPropertyOutcome(
        name=PROPERTY_LENGTH_IS_A_HARD_BOUND,
        passed=False,
        evidence=f"可学习表声称支持第 {probe} 位，但表只有 {table.positions} 行",
        detail="可学习表必须对越界位置拒绝",
    )


def check_staggered_pairing_breaks_both_laws(
    table: EncodingTable,
    *,
    base: float | None = None,
) -> PositionalPropertyOutcome:
    """把 ``cos`` 换成“隔壁”的频率之后，**行范数恒定**与**位移律**一起失效.

    这是本课的“判据要能反向检验”那一条：一个只在正例上测过的性质，
    无法区分“实现对了”与“判据太松”。这里对同一组尺寸的错位表跑同样的两条判据，
    并给出**实测的偏差量级**——它必须明显大于容差。
    """
    if not table.pairing_is_aligned:
        return PositionalPropertyOutcome(
            name=PROPERTY_STAGGERED_BREAKS_THE_LAWS,
            passed=False,
            evidence=(
                f"本次表是 {table.kind}/{table.pairing or '—'}："
                "这一条要求一张对齐频率的正弦表作为对照基准（跳过不成立）"
            ),
            detail="对照必须有一侧是'已知正确'的，否则这条判据只是在自我循环",
        )
    resolved_base = base if base is not None else (table.base or 10000.0)
    staggered = staggered_table(
        table.positions, table.dimension, base=resolved_base
    )
    norm_target = math.sqrt(table.dimension / 2.0)
    norm_deviation = max(
        abs(value - norm_target) for value in staggered.norms
    )
    offset_deviation = 0.0
    for offset in (1, 2, 3):
        closed = closed_form_offset_inner(table.dimension, offset, base=resolved_base)
        for start in range(staggered.positions - offset):
            offset_deviation = max(
                offset_deviation,
                abs(staggered.inner_product(start, start + offset) - closed),
            )
    passed = (
        norm_deviation > ROW_NORM_TOLERANCE
        and offset_deviation > OFFSET_LAW_TOLERANCE
        and table.aligned_within
    )
    return PositionalPropertyOutcome(
        name=PROPERTY_STAGGERED_BREAKS_THE_LAWS,
        passed=passed,
        evidence=(
            f"错位表的行范数偏差 {norm_deviation:.2e}、位移律偏差 {offset_deviation:.2e}"
            f"（对照：对齐表行范数恒定 = {table.aligned_within}）——两条判据都抓得住它"
        ),
        detail="错位表不是'错误实现'，而是 day073 的写法；它必须被这两条判据挡住",
    )


def check_properties(
    params: AttentionParams,
    table: EncodingTable,
    inputs: Matrix,
    *,
    positions: tuple[int, ...] | None = None,
    target: Matrix | None = None,
    supervised: Sequence[int] | None = None,
    causal: bool = False,
) -> PositionalPropertyReport:
    """跑完七条性质，返回一份报告.

    ``causal=True`` 时第 4 条返回“跳过”；用可学习表或错位表时前三条与第七条
    会**不通过**——那是“这张表不满足它”的读数，而不是实现错了。
    """
    forward = positional_forward(
        params,
        table,
        inputs,
        positions=positions,
        target=target,
        supervised=supervised,
        causal=causal,
    )
    outcomes = (
        check_row_norm_is_constant(table),
        check_offset_law(table),
        check_shift_is_rotation(table),
        check_injection_breaks_equivariance(
            params, table, inputs, positions=positions, causal=causal
        ),
        check_additive_backward(forward)
        if target is not None
        else PositionalPropertyOutcome(
            name=PROPERTY_ADDITIVE_BACKWARD,
            passed=True,
            evidence="本次前向没有 target（跳过）——没有损失梯度就无法检查反向",
            detail="要检查这一条请给 target",
        ),
        check_table_length_is_a_hard_bound(table),
        check_staggered_pairing_breaks_both_laws(table),
    )
    notes = [
        "前三条与第七条只对**对齐频率的偶数维正弦表**成立；其余四条对任何表都成立",
        "第 4 条在 causal=True 时是'跳过'而不是'通过'",
    ]
    return PositionalPropertyReport(outcomes=outcomes, notes=tuple(notes))


__all__ = [
    "GRADIENT_TOLERANCE",
    "OFFSET_LAW_TOLERANCE",
    "ROTATION_TOLERANCE",
    "PositionalGradientOutcome",
    "PositionalGradientReport",
    "PositionalPropertyOutcome",
    "PositionalPropertyReport",
    "SymmetryBreak",
    "check_additive_backward",
    "check_injection_breaks_equivariance",
    "check_offset_law",
    "check_positional_gradients",
    "check_properties",
    "check_row_norm_is_constant",
    "check_shift_is_rotation",
    "check_staggered_pairing_breaks_both_laws",
    "check_table_length_is_a_hard_bound",
    "default_permutation",
    "loss_gradient_of",
    "numeric_params_resolve",
    "numerical_positional_gradients",
    "permute_rows",
    "symmetry_break",
]
