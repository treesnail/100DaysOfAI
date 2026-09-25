"""梯度校验、性质检查与"注意力 vs 检索"的类比（day075 / M7-D1）.

这一层回答三个问题，它们分别对应三种完全不同的怀疑：

```text
① 反向传播写对了吗？     与**数值差分**逐点对照（五项：四个参数矩阵 + 输入）
② 前向的账算对了吗？     校验四条性质（行和为 1 / 非负 / 因果无泄漏 / 置换等变）
③ 注意力与检索是什么关系？把同一批输入按两种方式排序，看它们的重合与秩相关
```

## 为什么梯度校验必须"逐项"而不是"整体"

一个整体的误差数字无法回答"是哪一块错了"。五项分开之后，
每一项都能对自己那一句推导负责：

```text
w_output   反向的第一站（它只吃损失对输出的梯度）—— 错了说明"最外层"写错了
w_value    把"输出该混哪些位置"翻译成"value 该怎么调"
w_query    梯度要穿过 softmax（这一块错了，多半是 softmax 的反向写错了）
w_key      与 query 共享同一份软打分的梯度
inputs     三条链之和（少了任何一条都不会报错，只会偏小）
```

其中 ``inputs`` 那一项是**唯一能发现"漏了一条链"的检查**：
前四项都正确的实现，完全可能在 ``grad_inputs`` 里只写了 q 与 v 两条链——
而它不会报错，只会让"靠近输入的那几层学得慢"。

## 数值差分怎么用（day074 的零件在这里兑现）

```text
参数：把四个矩阵压平成一串数（day074 的 flatten_matrices）
      再对每一个分量做中心差分（day074 的 calculus.gradient）
输入：同样处理（形状表只有一块）
```

代价是 ``2 × 参数量`` 次前向。本课的参数是 4 个小矩阵（几十到几百个），
因此它完全跑得动——而这也正是"数值差分只适合校验、不适合训练"的量化理由。

## 四项性质各自在说什么

```text
行和为 1        权重是一个**条件分布**，不是一组打分
非负            负权重会让"加权平均"在某处变成减法
因果无泄漏      上三角恰好 0.0 —— "看不到未来"在数字上就是"权重为 0"
置换等变        把输入行换序，输出行按同样方式换序（**因果掩码会破坏它**）
```

最后一条是 day078（位置编码）的伏笔：**注意力本身不知道顺序**，
因此"猫追老鼠"与"老鼠追猫"在没有位置编码时是同一件事。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.math_foundations.calculus import DEFAULT_STEP, gradient
from smart_research_agent.math_foundations.linalg import cosine
from smart_research_agent.math_foundations.optim import (
    flatten_matrices,
    unflatten_matrices,
)
from smart_research_agent.math_foundations.types import (
    Matrix,
    Vector,
    matrix_shape,
    validate_matrix,
)
from smart_research_agent.transformer_core.errors import (
    GradientError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.transformer_core.layers import (
    attention_backward,
    masked_mean_squared_error,
    mean_squared_error,
    mse_gradient,
    self_attention,
)
from smart_research_agent.transformer_core.types import (
    DEFAULT_SUM_TOLERANCE,
    GRADIENT_TARGETS,
    GRADIENT_TARGET_FORMULAS,
    PROPERTY_CHECKS,
    PROPERTY_CHECK_DESCRIPTIONS,
    AttentionForward,
    AttentionParams,
    ParameterGradients,
    matrix_max_absolute,
    relative_matrix_error,
)

#: 梯度校验的缺省容差。为什么是 ``1e-6`` 而不是 day074 的 ``1e-8``：
#: 这里的"函数值"是**损失**（量级约 0.1~1），而它由一串矩阵乘法叠出来，
#: 每次求值累积的舍入在 ``1e-16 × 运算次数`` 量级；
#: 中心差分再放大 ``1/h``。分辨率量级 ``1e-10 ~ 1e-9``，
#: 取 ``1e-6`` 留出三个数量级的余量（"真的写错了"通常是 ``1e-2`` 以上）。
GRADIENT_TOLERANCE = 1e-6


def default_permutation(rows: int) -> tuple[int, ...]:
    """按行数给出一个确定性的置换：**把顺序整个倒过来**.

    为什么要按行数生成而不是写一个常量：置换必须是 ``0..rows-1`` 的一个排列，
    而样本长度会变（induction 任务是 5 行、单元测试可能是 3 行）。
    写死一个常量在长度对不上时会抛 ``ParameterError``——
    一个"单元测试过了、示例脚本崩了"的错误，而它看起来像"样本有问题"。

    为什么选"倒序"而不是某个更花哨的置换：倒序对任何长度都是合法排列，
    且它在因果掩码下**一定**会改变"每个位置能看到什么"——
    这正是 ``permutation_gap`` 想暴露的那件事。
    """
    if isinstance(rows, bool) or not isinstance(rows, int):
        raise ParameterError(f"行数必须是整数，收到 {rows!r}。")
    if rows < 1:
        raise ParameterError(f"行数必须 >= 1，收到 {rows}。")
    return tuple(range(rows - 1, -1, -1))


# --------------------------------------------------------------------------- #
# 数值梯度
# --------------------------------------------------------------------------- #


def numerical_parameter_gradients(
    params: AttentionParams,
    inputs: Matrix,
    target: Matrix,
    *,
    causal: bool = True,
    step: float = DEFAULT_STEP,
    supervised: Sequence[int] | None = None,
) -> ParameterGradients:
    """用中心差分算出"完整的"梯度：四块参数梯度 + 输入梯度.

    这是校验的**独立一侧**：它只依赖"前向 + 损失"，不依赖任何推导。
    因此它与 :func:`layers.attention_backward` 对上，才算把反向传播钉住了。

    ``supervised`` 非空时损失换成**只在那些行上**的 MSE
    （与 ``layers.masked_mean_squared_error`` 同一个口径）。

    **这一条不是可选的礼貌参数，而是一次真实的踩坑**：第一版没有它，
    于是"数值梯度"算的是**全行** MSE，而"解析梯度"算的是**监督行** MSE——
    两者差了 7.5e-3（梯度范数只有 4.8e-2，也就是差 15%），
    而它们**都是对的**。一个"两边都对却对不上"的对照，
    比一个"有一边错"的对照更难查：它会让人先去怀疑推导。

    分母也必须是同一个：监督行 MSE 的分母是"监督行数 × 列数"，
    用全部行数当分母会让数值侧的梯度**整体偏小**（第 5 章的
    ``masked_cross_entropy`` 讲过同一个坑）。
    """
    checked_inputs = validate_matrix(inputs, name="inputs")
    checked_target = validate_matrix(target, name="target")
    if matrix_shape(checked_inputs) != matrix_shape(checked_target):
        raise ShapeError(
            f"输入形状 {matrix_shape(checked_inputs)} 与目标形状 "
            f"{matrix_shape(checked_target)} 不一致。"
        )
    selected = None if supervised is None else tuple(supervised)

    def loss(output: Matrix) -> float:
        """按 ``supervised`` 选择"全行 MSE"或"监督行 MSE"（**两侧必须同一个**）."""
        if selected is None:
            return mean_squared_error(output, checked_target)
        return masked_mean_squared_error(output, checked_target, selected)

    flat, shapes = params.flatten()

    def loss_with_parameters(candidate: Vector) -> float:
        """把候选参数还原成 AttentionParams 之后算一次损失."""
        rebuilt = AttentionParams.unflatten(candidate, shapes)
        return loss(self_attention(rebuilt, checked_inputs, causal=causal).output)

    numeric_flat = gradient(loss_with_parameters, flat, step=step)
    parts = unflatten_matrices(numeric_flat, shapes)

    input_flat, input_shapes = flatten_matrices([checked_inputs])

    def loss_with_inputs(candidate: Vector) -> float:
        """固定参数、只动输入时的损失."""
        rebuilt_inputs = unflatten_matrices(candidate, input_shapes)[0]
        return loss(self_attention(params, rebuilt_inputs, causal=causal).output)

    numeric_inputs = unflatten_matrices(
        gradient(loss_with_inputs, input_flat, step=step), input_shapes
    )[0]

    return ParameterGradients(
        grad_w_query=parts[0],
        grad_w_key=parts[1],
        grad_w_value=parts[2],
        grad_w_output=parts[3],
        grad_inputs=numeric_inputs,
    )


# --------------------------------------------------------------------------- #
# 梯度校验报告
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GradientCheckOutcome:
    """一项梯度校验的结论：解析 vs 数值的误差、容差与结论.

    ``max_absolute_error`` 给的是**证据**（误差多大），
    ``max_scaled_error`` 才是**判据**（按量级缩放后的误差）——
    两者的关系与 day074 的 ``gradcheck`` 完全一致。
    """

    target: str
    max_absolute_error: float
    max_scaled_error: float
    tolerance: float
    compared_points: int
    message: str

    def __post_init__(self) -> None:
        if self.target not in GRADIENT_TARGETS:
            raise GradientError(
                f"不认识的梯度校验项 {self.target!r}：可用取值 {list(GRADIENT_TARGETS)}。"
            )
        for name, value in (
            ("最大绝对误差", self.max_absolute_error),
            ("最大相对误差", self.max_scaled_error),
        ):
            if math.isnan(value) or value < 0:
                raise NumericError(f"{name}不能为负或 NaN：{value}。")

    @property
    def passed(self) -> bool:
        """这一项是否在容差内."""
        return self.max_scaled_error <= self.tolerance

    @property
    def formula(self) -> str:
        """这一项对照的解析梯度公式."""
        return GRADIENT_TARGET_FORMULAS[self.target]

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "target": self.target,
            "formula": self.formula,
            "passed": self.passed,
            "max_absolute_error": self.max_absolute_error,
            "max_scaled_error": self.max_scaled_error,
            "tolerance": self.tolerance,
            "compared_points": self.compared_points,
            "message": self.message,
        }

    def summary_line(self) -> str:
        """一行说明：``[ok] w_query 9 个点，最大相对误差 2.31e-11``."""
        mark = "ok" if self.passed else "FAIL"
        return (
            f"[{mark}] {self.target} {self.compared_points} 个点，"
            f"最大相对误差 {self.max_scaled_error:.2e}"
        )


@dataclass(frozen=True)
class GradientCheckReport:
    """五项梯度校验的汇总（:func:`check_attention_gradients` 的产物）."""

    outcomes: tuple[GradientCheckOutcome, ...]
    tolerance: float = GRADIENT_TOLERANCE
    step: float = DEFAULT_STEP
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        missing = set(GRADIENT_TARGETS) - {item.target for item in self.outcomes}
        if missing:
            raise GradientError(
                f"梯度校验报告缺少 {sorted(missing)}：少做的校验与'校验通过'"
                "在报告里长得一样，因此这里当场拒绝——"
                "一份不完整的报告比没有报告更危险。"
            )
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def ok(self) -> bool:
        """五项是否全部在容差内."""
        return all(item.passed for item in self.outcomes)

    @property
    def failures(self) -> tuple[GradientCheckOutcome, ...]:
        """需要改推导的那些项."""
        return tuple(item for item in self.outcomes if not item.passed)

    @property
    def worst_scaled_error(self) -> float:
        """五项里最大的相对误差（"这份报告的分辨率"）."""
        return max((item.max_scaled_error for item in self.outcomes), default=0.0)

    def raise_if_failed(self) -> None:
        """有任何一项不通过就抛 :class:`GradientError`.

        这一条是"梯度校验属于**控制流**而不是报告"的兑现：
        一条对不上的梯度意味着这次训练从头到尾都不可信。
        """
        if not self.ok:
            names = ", ".join(item.target for item in self.failures)
            raise GradientError(
                f"梯度校验未通过：{names}（最差相对误差 {self.worst_scaled_error:.3e}，"
                f"容差 {self.tolerance:.1e}）。一条对不上的梯度意味着这次训练不可信，"
                "因此这里不返回报告而是直接失败。"
            )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "ok": self.ok,
            "tolerance": self.tolerance,
            "step": self.step,
            "worst_scaled_error": self.worst_scaled_error,
            "failures": [item.target for item in self.failures],
            "outcomes": [item.to_dict() for item in self.outcomes],
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``梯度校验 5 项：通过 5、失败 0 | 最大相对误差 3.10e-11``."""
        passed = sum(1 for item in self.outcomes if item.passed)
        return (
            f"梯度校验 {len(self.outcomes)} 项：通过 {passed}、"
            f"失败 {len(self.failures)} | 最大相对误差 {self.worst_scaled_error:.2e}"
        )


def check_attention_gradients(
    params: AttentionParams,
    inputs: Matrix,
    target: Matrix,
    *,
    causal: bool = True,
    step: float = DEFAULT_STEP,
    tolerance: float = GRADIENT_TOLERANCE,
) -> GradientCheckReport:
    """五项逐点对照：解析梯度 vs 数值差分.

    实现只有五行：算一次前向、算一次解析反向、算一次数值梯度、
    按名字把五块拼起来逐点比。**"简单"正是它能当护栏的原因**——
    它不依赖被检查的那段代码。
    """
    if tolerance <= 0:
        raise ParameterError(f"容差必须为正，收到 {tolerance}。")
    forward = self_attention(params, inputs, causal=causal)
    checked_target = validate_matrix(target, name="target")
    if matrix_shape(forward.output) != matrix_shape(checked_target):
        raise ShapeError(
            f"目标形状 {matrix_shape(checked_target)} 与输出形状 "
            f"{matrix_shape(forward.output)} 不一致。"
        )
    analytic = attention_backward(forward, mse_gradient(forward.output, checked_target))
    numeric = numerical_parameter_gradients(
        params, inputs, checked_target, causal=causal, step=step
    )

    analytic_parts = {**analytic.as_dict(), "inputs": analytic.grad_inputs}
    numeric_parts = {**numeric.as_dict(), "inputs": numeric.grad_inputs}

    outcomes: list[GradientCheckOutcome] = []
    for name in GRADIENT_TARGETS:
        left = analytic_parts[name]
        right = numeric_parts[name]
        scaled = relative_matrix_error(left, right)
        absolute = _max_absolute_difference(left, right)
        rows, columns = matrix_shape(left)
        outcomes.append(
            GradientCheckOutcome(
                target=name,
                max_absolute_error=absolute,
                max_scaled_error=scaled,
                tolerance=tolerance,
                compared_points=rows * columns,
                message=(
                    f"逐点在容差 {tolerance:.1e} 内一致（最大绝对误差 {absolute:.2e}）"
                    if scaled <= tolerance
                    else f"最大相对误差 {scaled:.3e} 超过容差 {tolerance:.1e}"
                ),
            )
        )
    return GradientCheckReport(
        outcomes=tuple(outcomes),
        tolerance=tolerance,
        step=step,
        notes=(
            f"数值差分用中心差分、步长 {step:g}（day074 的 calculus.gradient）；"
            "它只依赖前向与损失，因此不依赖被检查的那段推导",
            "inputs 那一项是唯一能发现'漏了一条链'的检查："
            "前四项都对的实现完全可能在输入梯度里只写了 q 与 v 两条",
            f"容差 {tolerance:.1e} 比本层的分辨率（损失量级 1e-1、步长 {step:g} "
            "→ 约 1e-9）松三个数量级，而'真的写错了'通常差 1e-2 以上",
        ),
    )


def _max_absolute_difference(left: Matrix, right: Matrix) -> float:
    """两块同形矩阵的最大逐点绝对误差."""
    if matrix_shape(left) != matrix_shape(right):
        raise ShapeError("两块矩阵形状不同，无法比较。")
    worst = 0.0
    for left_row, right_row in zip(left, right, strict=True):
        for first, second in zip(left_row, right_row, strict=True):
            if not (math.isfinite(first) and math.isfinite(second)):
                return math.inf
            worst = max(worst, abs(first - second))
    return worst


# --------------------------------------------------------------------------- #
# 性质检查
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PropertyOutcome:
    """一条性质的结论：通过与否、证据、以及"这条性质在说什么"."""

    name: str
    passed: bool
    evidence: str
    detail: str

    def __post_init__(self) -> None:
        if self.name not in PROPERTY_CHECKS:
            raise NumericError(
                f"不认识的性质 {self.name!r}：可用取值 {list(PROPERTY_CHECKS)}。"
            )

    @property
    def description(self) -> str:
        """这条性质的一句话解释（来自口径表）."""
        return PROPERTY_CHECK_DESCRIPTIONS[self.name]

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "name": self.name,
            "description": self.description,
            "passed": self.passed,
            "evidence": self.evidence,
            "detail": self.detail,
        }

    def summary_line(self) -> str:
        """一行说明：``[ok] row_stochastic 行和最大偏差 0.00e+00``."""
        mark = "ok" if self.passed else "FAIL"
        return f"[{mark}] {self.name} {self.evidence}"


@dataclass(frozen=True)
class PropertyReport:
    """四条性质的汇总（:func:`check_properties` 的产物）."""

    outcomes: tuple[PropertyOutcome, ...]
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        missing = set(PROPERTY_CHECKS) - {item.name for item in self.outcomes}
        if missing:
            raise NumericError(
                f"性质报告缺少 {sorted(missing)}：少做一条与'这条通过'在报告里长得一样。"
            )
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def ok(self) -> bool:
        """四条是否全部通过."""
        return all(item.passed for item in self.outcomes)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "ok": self.ok,
            "outcomes": [item.to_dict() for item in self.outcomes],
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``性质检查 4 项：通过 4、失败 0``."""
        passed = sum(1 for item in self.outcomes if item.passed)
        return f"性质检查 {len(self.outcomes)} 项：通过 {passed}、失败 {len(self.outcomes) - passed}"


def check_row_stochastic(
    forward: AttentionForward, *, tolerance: float = DEFAULT_SUM_TOLERANCE
) -> PropertyOutcome:
    """每一行权重之和是否为 1（并给出**最大偏差**作为证据）."""
    deviations = [abs(math.fsum(row) - 1.0) for row in forward.weights]
    worst = max(deviations, default=0.0)
    return PropertyOutcome(
        name="row_stochastic",
        passed=worst <= tolerance,
        evidence=f"行和最大偏差 {worst:.2e}",
        detail=(
            "权重是一个**条件分布**（'这一行怎么看各个位置'），"
            "和不为 1 会让后续加权求和变成一次缩放错误的组合"
        ),
    )


def check_non_negative(forward: AttentionForward) -> PropertyOutcome:
    """每个权重是否非负."""
    worst = min((value for row in forward.weights for value in row), default=0.0)
    return PropertyOutcome(
        name="non_negative",
        passed=worst >= 0.0,
        evidence=f"最小权重 {worst:.2e}",
        detail="负权重会让'加权平均'在某处变成减法，而整行看起来仍然是一条曲线",
    )


def check_causal_no_leak(forward: AttentionForward) -> PropertyOutcome:
    """因果模式下上三角是否**恰好是 0.0**（"看不到未来"的可验证形式）."""
    if not forward.causal:
        return PropertyOutcome(
            name="causal_no_leak",
            passed=True,
            evidence="本次前向未启用因果掩码（跳过）",
            detail=(
                "未启用掩码时这一条不适用：它不是'通过'，而是'没有可检查的对象'——"
                "报告里必须把这两者区分开，否则'没开掩码'会伪装成'掩码正确'"
            ),
        )
    leaks: list[float] = []
    for row in range(forward.tokens):
        for column in range(row + 1, forward.tokens):
            leaks.append(abs(forward.weights[row][column]))
    worst = max(leaks, default=0.0)
    return PropertyOutcome(
        name="causal_no_leak",
        passed=worst == 0.0,
        evidence=f"上三角最大权重 {worst:.2e}",
        detail=(
            "位置 i 看不到它之后的任何位置——在数字上就是那一格**恰好是 0.0**"
            "（显式掩码，不是把打分写成 -inf）"
        ),
    )


def permutation_gap(
    params: AttentionParams,
    inputs: Matrix,
    *,
    causal: bool = False,
    permutation: tuple[int, ...] | None = None,
) -> float:
    """把输入行按 ``permutation`` 换序之后，输出行是否按同样方式换序.

    返回最大绝对差（0 表示严格等变）。
    ``causal=True`` 时它会**明显不为 0**——这不是 bug，而是这一条最值钱的地方：

    ```text
    无掩码     注意力是置换等变的 → 它不认识顺序
    因果掩码   掩码把"位置"写进了这张表 → 等变性被破坏
    ```

    也就是说：**顺序信息可以来自掩码，但掩码只提供了一种非常粗的顺序**
    （"谁在谁前面"）。要表达"位置 3 与位置 5 的距离"，仍然需要位置编码（day078）。
    """
    checked_inputs = validate_matrix(inputs, name="inputs")
    rows = len(checked_inputs)
    order = _checked_permutation(
        default_permutation(rows) if permutation is None else permutation, rows
    )
    permuted_inputs = tuple(checked_inputs[index] for index in order)
    original = self_attention(params, checked_inputs, causal=causal).output
    shuffled = self_attention(params, permuted_inputs, causal=causal).output
    worst = 0.0
    for position, source in enumerate(order):
        for column in range(len(original[source])):
            worst = max(worst, abs(shuffled[position][column] - original[source][column]))
    return worst


def _checked_permutation(permutation: tuple[int, ...], rows: int) -> tuple[int, ...]:
    """校验置换：必须是 0..rows-1 的一个排列（否则"换序"这件事没有定义）."""
    if sorted(permutation) != list(range(rows)):
        raise ParameterError(
            f"置换 {permutation} 不是 0..{rows - 1} 的一个排列："
            "换序的输入必须是同一批行，否则比较的不是'同一件事的两种写法'。"
        )
    return permutation


def check_permutation_equivariance(
    params: AttentionParams,
    inputs: Matrix,
    *,
    causal: bool = False,
    permutation: tuple[int, ...] | None = None,
    tolerance: float = 1e-9,
) -> PropertyOutcome:
    """无掩码时注意力是否置换等变（这是"需要位置编码"的直接证据）."""
    gap = permutation_gap(params, inputs, causal=causal, permutation=permutation)
    return PropertyOutcome(
        name="permutation_equivariance",
        passed=gap <= tolerance,
        evidence=f"置换前后的最大偏差 {gap:.2e}",
        detail=(
            "无掩码时把输入行换序，输出行按同样方式换序——"
            "**注意力本身不知道顺序**（因果掩码会破坏这条性质，见 permutation_gap 的说明）"
        ),
    )


def check_properties(
    params: AttentionParams,
    inputs: Matrix,
    *,
    causal: bool = True,
) -> PropertyReport:
    """跑完四条性质，装成一份 :class:`PropertyReport`（顺序 = 口径表）.

    注意"置换等变"这一条传的是 ``causal`` **的反面**：
    因果掩码下它本来就不该成立（那是设计而不是缺陷），
    因此这一条检查的是"没有掩码时成立"，而 `permutation_gap` 用来量"掩码破坏了它多少"。
    """
    forward = self_attention(params, inputs, causal=causal)
    outcomes = (
        check_row_stochastic(forward),
        check_non_negative(forward),
        check_causal_no_leak(forward),
        check_permutation_equivariance(params, inputs, causal=False),
    )
    return PropertyReport(
        outcomes=outcomes,
        notes=(
            "四条性质里前三条是**前向的账**（每一项都能独立失败），"
            "第四条是'这张表在换序下怎么变'——它引出位置编码（day078）",
            "置换等变一条在因果模式下**故意不适用**：掩码把顺序信息写进了表里，"
            f"实测偏差见 permutation_gap（本层的因果掩码下约 {permutation_gap(params, inputs, causal=True):.4f}）",
        ),
    )


# --------------------------------------------------------------------------- #
# 注意力 vs 检索：同一批输入的两种排序
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RetrievalComparison:
    """同一次前向里，"注意力权重"与"输入行之间的余弦"两种排序的一致程度.

    ```text
    peak_agreement    峰值位置一致的行数（两边的 argmax 是不是同一个位置）
    overlap           top-k 集合的重合总数（对所有行求和）
    overlap_ratio     overlap / (行数 × top_k)
    rank_correlation  两种打分的秩相关（Spearman，平均秩处理并列）
    ```

    **为什么不能只看 top-k 重合**：top-k 是一个离散量，
    k=1 时它只取值 0 或 1，很多差别会被它吞掉。秩相关是连续量，
    它回答的是"整体排序像不像"，而 top-k 回答的是"最相关的那几个是不是同一批"。
    两者一起看才能区分两种情况：

    ```text
    秩相关高、重合低   整体排序一致，但边界位置在换（k 太小或分数接近）
    秩相关低、重合高   只有最相关的那几个恰好相同（退化情形）
    ```
    """

    top_k: int
    rows: int
    peak_agreement: int
    overlap: int
    rank_correlation: float
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ParameterError(f"top_k 必须 >= 1，收到 {self.top_k}。")
        if self.rows < 1:
            raise ParameterError(f"行数必须 >= 1，收到 {self.rows}。")
        if not 0 <= self.peak_agreement <= self.rows:
            raise NumericError(
                f"峰值一致的行数 {self.peak_agreement} 落在 [0, {self.rows}] 之外。"
            )
        if not 0 <= self.overlap <= self.rows * self.top_k:
            raise NumericError(
                f"重合数 {self.overlap} 落在 [0, {self.rows * self.top_k}] 之外。"
            )
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def peak_ratio(self) -> float:
        """峰值一致的比例."""
        return self.peak_agreement / self.rows

    @property
    def overlap_ratio(self) -> float:
        """top-k 重合比例."""
        return self.overlap / (self.rows * self.top_k)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "top_k": self.top_k,
            "rows": self.rows,
            "peak_agreement": self.peak_agreement,
            "peak_ratio": self.peak_ratio,
            "overlap": self.overlap,
            "overlap_ratio": self.overlap_ratio,
            "rank_correlation": self.rank_correlation,
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``top-2：峰值一致 2/5、重合 7/10（70.0%）| 秩相关 0.9123``."""
        return (
            f"top-{self.top_k}：峰值一致 {self.peak_agreement}/{self.rows}、"
            f"重合 {self.overlap}/{self.rows * self.top_k}（{self.overlap_ratio:.1%}）| "
            f"秩相关 {self.rank_correlation:+.4f}"
        )


def compare_with_retrieval(forward: AttentionForward, *, top_k: int = 2) -> RetrievalComparison:
    """把"注意力在混谁"与"余弦检索会取谁"逐行对照.

    逐行的做法（**只在允许的位置上比**）：

    ```text
    检索侧    cos(input[i], input[j]) 对允许的 j 排序 —— 这是 day066~day071 的检索
    注意力侧  weights[i][j] 对允许的 j 排序 —— 这是这一课的注意力
    ```

    为什么必须"只在允许的位置上比"：因果掩码把未来位置的权重写成 0，
    而余弦并不知道掩码的存在。不排除它们时，"不一样"里会有很大一部分
    只是"掩码 vs 不掩码"的差别，而不是"两种相关性口径"的差别。
    """
    if top_k < 1:
        raise ParameterError(f"top_k 必须 >= 1，收到 {top_k}。")
    rows = forward.tokens
    if top_k > rows:
        raise ParameterError(
            f"top_k={top_k} 超过行数 {rows}：每一行最多只有 {rows} 个候选位置。"
        )
    peak_agreement = 0
    overlap = 0
    correlations: list[float] = []
    for row in range(rows):
        allowed = [column for column in range(rows) if forward.mask[row][column]]
        if len(allowed) < 2:
            # 只有一个候选时"排序"没有意义（第 0 行在因果掩码下就是这种情况）
            peak_agreement += 1
            overlap += min(top_k, len(allowed))
            continue
        attention_scores = [forward.weights[row][column] for column in allowed]
        similarity_scores = [
            cosine(forward.inputs[row], forward.inputs[column]) for column in allowed
        ]
        if _argmax_of(attention_scores) == _argmax_of(similarity_scores):
            peak_agreement += 1
        attention_top = set(_top_k_indices(attention_scores, top_k=min(top_k, len(allowed))))
        similarity_top = set(_top_k_indices(similarity_scores, top_k=min(top_k, len(allowed))))
        overlap += len(attention_top & similarity_top)
        correlations.append(rank_correlation(attention_scores, similarity_scores))
    average = math.fsum(correlations) / len(correlations) if correlations else 0.0
    return RetrievalComparison(
        top_k=top_k,
        rows=rows,
        peak_agreement=peak_agreement,
        overlap=overlap,
        rank_correlation=average,
        notes=(
            "对照只在**允许的位置**上做：掩码把未来位置的权重写成 0，"
            "而余弦并不知道掩码的存在——不排除它们时，"
            "'不一样'里会有很大一部分只是'掩码 vs 不掩码'的差别",
            "注意力是可微的混合（权重是一组概率），检索是离散的选择（取前 k 条）——"
            "两者都在回答'哪些内容与当前问题相关'，但只有前者能收到梯度",
        ),
    )


def _argmax_of(values: list[float]) -> int:
    """返回最大值的下标（并列时取最小的下标——**写下来**，否则并列情形不可复现）."""
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


def _top_k_indices(values: list[float], *, top_k: int) -> list[int]:
    """取最大的 k 个下标（**并列时按下标升序**，保证可复现）."""
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    return order[:top_k]


def rank_correlation(left: list[float], right: list[float]) -> float:
    """两个数列的秩相关（Spearman）：把两边都换成**平均秩**，再算 Pearson.

    两个必须说清的约定：

    ```text
    并列值        用"平均秩"（并列的 3 个数占据 2/3/4 名时每人记 3.0）
                  用"字典序"当排名的做法会让结果依赖于输入顺序
    零方差        任一边全部相同时相关性没有定义 → 返回 0.0 而不是抛异常
                  （"没有相关性可言"与"相关性为 0"在数值上一样，
                    但调用方需要拿到一个数才能继续，因此这里返回 0.0 并写在文档里）
    ```
    """
    if len(left) != len(right):
        raise ShapeError(f"两个数列长度不同：{len(left)} 与 {len(right)}。")
    if len(left) < 2:
        return 0.0
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    return _pearson(left_ranks, right_ranks)


def _average_ranks(values: list[float]) -> list[float]:
    """平均秩：值相同的元素共享它们所占名次的平均值."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2.0 + 1.0
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    """皮尔逊相关系数（任一边方差为 0 时返回 0.0）."""
    count = len(left)
    mean_left = math.fsum(left) / count
    mean_right = math.fsum(right) / count
    covariance = math.fsum(
        (first - mean_left) * (second - mean_right)
        for first, second in zip(left, right, strict=True)
    )
    variance_left = math.fsum((value - mean_left) ** 2 for value in left)
    variance_right = math.fsum((value - mean_right) ** 2 for value in right)
    if variance_left == 0.0 or variance_right == 0.0:
        return 0.0
    return covariance / math.sqrt(variance_left * variance_right)


__all__ = [
    "GRADIENT_TOLERANCE",
    "GradientCheckOutcome",
    "GradientCheckReport",
    "PropertyOutcome",
    "PropertyReport",
    "RetrievalComparison",
    "check_attention_gradients",
    "check_causal_no_leak",
    "check_non_negative",
    "check_permutation_equivariance",
    "check_properties",
    "check_row_stochastic",
    "compare_with_retrieval",
    "default_permutation",
    "numerical_parameter_gradients",
    "permutation_gap",
    "rank_correlation",
]
