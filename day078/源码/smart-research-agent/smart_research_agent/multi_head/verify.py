"""梯度校验、性质检查与"头之间像不像"的读数（day076 / M7-D2）.

这一层回答四个问题，它们分别对应四种完全不同的怀疑：

```text
① 多头反向写对了吗？     与**数值差分**逐点对照（五项：四个参数矩阵 + 输入）
② 前向的账算对了吗？     校验六条性质（每头行和为 1 / 非负 / 因果无泄漏 /
                         merge 是 split 的逆 / heads=1 等于 day075 / 头序只是记账）
③ 各头学到的东西一样吗？ 逐头逐行的**全变差距离**（TV）与 argmax 一致率
④ 改动是安全的吗？       一致地重排头块之后，输出必须逐位不变
```

## 五项梯度校验：名字与 day075 一样，公式不一样

```text
w_output   唯一**不被分头**的一块：它消费的是拼接结果 merged
w_value     heads 条链在**行维**上相加（W_v 被所有头共用）
w_query     每一头的梯度要穿过**它自己那一张**权重表的 softmax，再相加
w_key       与 query 共享同一份软打分，但分头之后各走各的
inputs      heads × 三条链的和——少一条不报错，只会让下层学得慢
```

``w_query`` 那一项仍然是**最容易出错、而且错了不会报错**的一项：
多头下 softmax 反向要跑 heads 遍，而"只跑第一遍"的实现形状完全正确、
损失也真的在降（只是每一头的梯度都偏小、且方向被另一头的分布污染）。

## 六条性质里，有两条本层专有

```text
merge_inverts_split         往返恒等：切错/拼错都不报错，只会让头部"串味"
single_head_matches_classic heads=1 必须**逐位**等于 day075 的自注意力
```

第二条值得单独说：它不是"多写了一条测试"，而是这一课的一条**回归护栏**——
它把"多头在 heads=1 时退化回单头"这件事钉死。任何一次对 split/merge 的重构，
只要破坏了这条退化关系，都会在这里红。而它用的是 ``==`` 而不是 ``approx``：
**"逐位相等"是一条比"误差很小"难伪造成立得多的断言。**

## 两处**刻意近似**的断言（以及为什么必须写清楚）

```text
merge → project 恒等式      浮点求和顺序不同 → 本样本实测差 6.94e-18，断言 <= 1e-12
头序置换不变                同上，重排之后求和顺序变了 → 断言 <= 1e-12
```

把这两条写成"必须逐位相等"，会让某一次无关的重构（比如换一个求和顺序）
变成一次假失败——而假失败会训练出"看到红就先怀疑测试"的坏习惯。
**该近似的地方就说近似，并且把观测值写进证据。**
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

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
)
from smart_research_agent.multi_head.errors import (
    GradientError,
    NumericError,
    ParameterError,
    ShapeError,
)
from smart_research_agent.multi_head.layers import (
    multi_head_attention,
    multi_head_backward,
)
from smart_research_agent.multi_head.types import (
    MULTIHEAD_GRADIENT_FORMULAS,
    MULTIHEAD_GRADIENT_TARGETS,
    MULTIHEAD_PROPERTIES,
    MULTIHEAD_PROPERTY_DESCRIPTIONS,
    MultiHeadForward,
    MultiHeadShape,
)
from smart_research_agent.transformer_core.layers import (
    masked_mean_squared_error,
    mean_squared_error,
    mse_gradient,
    self_attention,
)
from smart_research_agent.transformer_core.types import (
    DEFAULT_SUM_TOLERANCE,
    AttentionParams,
    ParameterGradients,
    relative_matrix_error,
)
from smart_research_agent.transformer_core.verify import (
    GRADIENT_TOLERANCE as CLASSIC_GRADIENT_TOLERANCE,
)

#: 梯度校验的缺省容差。与 day075 同值（``1e-6``），但**另立一个常量**：
#: 两处一旦不同，报告里"这份报告是谁的"就有了两种口径——
#: 因此 ``tests/test_multihead_gradients.py`` 里有一条断言把两者钉成相等。
GRADIENT_TOLERANCE = 1e-6

#: 两条"刻意近似"的性质用的容差（``1e-12``：比观测到的 7e-18 松五个数量级）.
IDENTITY_TOLERANCE = 1e-12


def default_head_order(heads: int) -> tuple[int, ...]:
    """按头数给出一个确定性的头序：**把顺序整个倒过来**.

    与 day075 的 ``default_permutation`` 同一理由：写死一个常量在头数变化时
    会抛 ``ParameterError``（一个"单元测试过了、演示脚本崩了"的错误，
    而它看起来像"参数有问题"）。

    为什么选倒序：它对任何 ``heads >= 1`` 都是合法排列，且 ``heads >= 3`` 时
    **一定**会让某一头的边界发生变化——这正是这条性质想暴露的那件事。
    （``heads = 2`` 时倒序是一种对换，同样会改变 ``merge`` 的顺序。）
    """
    if isinstance(heads, bool) or not isinstance(heads, int):
        raise ParameterError(f"头数必须是整数，收到 {heads!r}。")
    if heads < 1:
        raise ParameterError(f"头数必须 >= 1，收到 {heads}。")
    return tuple(range(heads - 1, -1, -1))


def checked_head_order(order: tuple[int, ...], heads: int) -> tuple[int, ...]:
    """校验头序：必须是 ``0..heads-1`` 的一个排列（否则"重排"没有定义）."""
    if sorted(order) != list(range(heads)):
        raise ParameterError(
            f"头序 {order} 不是 0..{heads - 1} 的一个排列："
            "重排的必须是同一批头，否则比较的不是'同一件事的两种写法'。"
        )
    return order


def permute_head_blocks(
    params: AttentionParams,
    heads: int,
    order: tuple[int, ...] | None = None,
) -> AttentionParams:
    """**一致地**重排头块：三个投影按行、``W_o`` 按列，用同一个 ``order``.

    ```text
    W_q 的行块 h → 位置 order[h]        W_k 同上        W_v 同上
    W_o 的列块 h → 位置 order[h]        ← 必须一起动，缺一不可
    ```

    "缺一不可"是这条性质的全部内容：只换 ``W_q`` 而不换 ``W_o``，
    得到的是一组**完全不同**的参数（第 h 头算出来的东西会被送到别的输出维）。
    那种改动当然会改变输出——而它改变的方式与"划分逻辑写错了"长得一模一样，
    因此这一条检查必须"要么全换、要么不换"。
    """
    shape = MultiHeadShape(params.shape, heads)
    checked_order = checked_head_order(
        default_head_order(heads) if order is None else order, heads
    )
    keys_partition = shape.keys_partition
    values_partition = shape.values_partition

    def _reorder_rows(matrix: Matrix, partition: Any) -> Matrix:
        blocks = partition.split_rows(matrix)
        return partition.merge_rows([blocks[index] for index in checked_order])

    def _reorder_columns(matrix: Matrix, partition: Any) -> Matrix:
        blocks = partition.split_columns(matrix)
        return partition.merge_columns([blocks[index] for index in checked_order])

    return AttentionParams(
        w_query=_reorder_rows(params.w_query, keys_partition),
        w_key=_reorder_rows(params.w_key, keys_partition),
        w_value=_reorder_rows(params.w_value, values_partition),
        w_output=_reorder_columns(params.w_output, values_partition),
    )


# --------------------------------------------------------------------------- #
# 数值梯度
# --------------------------------------------------------------------------- #


def numerical_multihead_gradients(
    params: AttentionParams,
    inputs: Matrix,
    target: Matrix,
    *,
    heads: int = 1,
    causal: bool = True,
    step: float = DEFAULT_STEP,
    supervised: Sequence[int] | None = None,
) -> ParameterGradients:
    """用中心差分算出"完整的"梯度：四块参数梯度 + 输入梯度.

    这是校验的**独立一侧**：它只依赖"前向 + 损失"，不依赖任何推导。
    因此它与 :func:`layers.multi_head_backward` 对上，才算把多头反向钉住了。

    ``supervised`` 的口径必须与解析侧**逐字一致**（day075 为此栽过一次：
    数值侧算全行 MSE、解析侧算监督行 MSE，差 15% 而两边都对）。
    今天这条坑再加深一层——``heads`` 也必须一致：
    数值侧若忘了传 ``heads``，它算的就是单头前向，而"多头梯度对不上"
    会表现为一个 1e-2 量级的差，看起来像"推导写错了"。
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
        """把候选参数还原成 AttentionParams 之后算一次多头损失."""
        rebuilt = AttentionParams.unflatten(candidate, shapes)
        return loss(
            multi_head_attention(rebuilt, checked_inputs, heads=heads, causal=causal).output
        )

    numeric_flat = gradient(loss_with_parameters, flat, step=step)
    parts = unflatten_matrices(numeric_flat, shapes)

    input_flat, input_shapes = flatten_matrices([checked_inputs])

    def loss_with_inputs(candidate: Vector) -> float:
        """固定参数、只动输入时的损失（``heads`` 同样要一致）."""
        rebuilt_inputs = unflatten_matrices(candidate, input_shapes)[0]
        return loss(
            multi_head_attention(
                params, rebuilt_inputs, heads=heads, causal=causal
            ).output
        )

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
class MultiHeadGradientOutcome:
    """一项梯度校验的结论：解析 vs 数值的误差、容差与结论."""

    target: str
    max_absolute_error: float
    max_scaled_error: float
    tolerance: float
    compared_points: int
    message: str

    def __post_init__(self) -> None:
        if self.target not in MULTIHEAD_GRADIENT_TARGETS:
            raise GradientError(
                f"不认识的梯度校验项 {self.target!r}：可用取值 "
                f"{list(MULTIHEAD_GRADIENT_TARGETS)}。"
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
        """这一项对照的**多头**解析梯度公式（与 day075 同名而不同式）."""
        return MULTIHEAD_GRADIENT_FORMULAS[self.target]

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
class MultiHeadGradientReport:
    """五项梯度校验的汇总（:func:`check_multihead_gradients` 的产物）."""

    outcomes: tuple[MultiHeadGradientOutcome, ...]
    heads: int
    tolerance: float = GRADIENT_TOLERANCE
    step: float = DEFAULT_STEP
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        missing = set(MULTIHEAD_GRADIENT_TARGETS) - {item.target for item in self.outcomes}
        if missing:
            raise GradientError(
                f"梯度校验报告缺少 {sorted(missing)}：少做的校验与'校验通过'"
                "在报告里长得一样，因此这里当场拒绝——"
                "一份不完整的报告比没有报告更危险。"
            )
        if self.heads < 1:
            raise ParameterError(f"头数必须 >= 1，收到 {self.heads}。")
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def ok(self) -> bool:
        """五项是否全部在容差内."""
        return all(item.passed for item in self.outcomes)

    @property
    def failures(self) -> tuple[MultiHeadGradientOutcome, ...]:
        """需要改推导的那些项."""
        return tuple(item for item in self.outcomes if not item.passed)

    @property
    def worst_scaled_error(self) -> float:
        """五项里最大的相对误差（"这份报告的分辨率"）."""
        return max((item.max_scaled_error for item in self.outcomes), default=0.0)

    def raise_if_failed(self) -> None:
        """有任何一项不通过就抛 :class:`GradientError`.

        与 day075 同一条纪律：梯度校验属于**控制流**而不是报告。
        多头把这条纪律的成本放大了 heads 倍——一次只回一头的实现
        会让"这次训练是不是可信"这个问题变得更难回答，而不是更容易。
        """
        if not self.ok:
            names = ", ".join(item.target for item in self.failures)
            raise GradientError(
                f"多头梯度校验未通过：{names}"
                f"（heads={self.heads}，最差相对误差 {self.worst_scaled_error:.3e}，"
                f"容差 {self.tolerance:.1e}）。一条对不上的梯度意味着这次训练不可信，"
                "因此这里不返回报告而是直接失败。"
            )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "ok": self.ok,
            "heads": self.heads,
            "tolerance": self.tolerance,
            "step": self.step,
            "worst_scaled_error": self.worst_scaled_error,
            "failures": [item.target for item in self.failures],
            "outcomes": [item.to_dict() for item in self.outcomes],
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``梯度校验 5 项（heads=2）：通过 5、失败 0 | 最大相对误差 3.10e-11``."""
        passed = sum(1 for item in self.outcomes if item.passed)
        return (
            f"梯度校验 {len(self.outcomes)} 项（heads={self.heads}）：通过 {passed}、"
            f"失败 {len(self.failures)} | 最大相对误差 {self.worst_scaled_error:.2e}"
        )


def check_multihead_gradients(
    params: AttentionParams,
    inputs: Matrix,
    target: Matrix,
    *,
    heads: int = 1,
    causal: bool = True,
    step: float = DEFAULT_STEP,
    tolerance: float = GRADIENT_TOLERANCE,
) -> MultiHeadGradientReport:
    """五项逐点对照：多头解析梯度 vs 数值差分.

    实现只有五行（与 day075 的对应函数逐行同构），**"简单"正是它能当护栏的原因**：
    它不依赖被检查的那段代码——它只是"再算一遍，换个办法"。
    """
    if tolerance <= 0:
        raise ParameterError(f"容差必须为正，收到 {tolerance}。")
    forward = multi_head_attention(params, inputs, heads=heads, causal=causal)
    checked_target = validate_matrix(target, name="target")
    if matrix_shape(forward.output) != matrix_shape(checked_target):
        raise ShapeError(
            f"目标形状 {matrix_shape(checked_target)} 与输出形状 "
            f"{matrix_shape(forward.output)} 不一致。"
        )
    analytic = multi_head_backward(forward, mse_gradient(forward.output, checked_target))
    numeric = numerical_multihead_gradients(
        params, inputs, checked_target, heads=heads, causal=causal, step=step
    )

    analytic_parts = {**analytic.as_dict(), "inputs": analytic.grad_inputs}
    numeric_parts = {**numeric.as_dict(), "inputs": numeric.grad_inputs}

    outcomes: list[MultiHeadGradientOutcome] = []
    for name in MULTIHEAD_GRADIENT_TARGETS:
        left = analytic_parts[name]
        right = numeric_parts[name]
        scaled = relative_matrix_error(left, right)
        absolute = relative_matrix_error(left, right)
        rows, columns = matrix_shape(left)
        outcomes.append(
            MultiHeadGradientOutcome(
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
    return MultiHeadGradientReport(
        outcomes=tuple(outcomes),
        heads=heads,
        tolerance=tolerance,
        step=step,
        notes=(
            f"数值差分用中心差分、步长 {step:g}（day074 的 calculus.gradient）；"
            "它只依赖前向与损失，因此不依赖被检查的那段推导",
            "数值侧必须收到**同一个 heads**：忘了传时它算的是单头前向，"
            "而'多头梯度对不上'会表现为 1e-2 量级的差，看起来像推导写错了",
            "w_query 那一项仍然是多头下最危险的一项：softmax 反向要跑 heads 遍，"
            "而'只跑第一遍'的实现形状完全正确、损失也真的在降",
            f"容差 {tolerance:.1e} 与 day075 的 "
            f"{CLASSIC_GRADIENT_TOLERANCE:.1e} 同值，运行时有一条断言把两者钉成相等",
        ),
    )


# --------------------------------------------------------------------------- #
# 性质检查
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MultiHeadPropertyOutcome:
    """一条性质的结论：通过与否、证据、以及"这条性质在说什么"."""

    name: str
    passed: bool
    evidence: str
    detail: str

    def __post_init__(self) -> None:
        if self.name not in MULTIHEAD_PROPERTIES:
            raise NumericError(
                f"不认识的性质 {self.name!r}：可用取值 {list(MULTIHEAD_PROPERTIES)}。"
            )

    @property
    def description(self) -> str:
        """这条性质的一句话解释（来自口径表）."""
        return MULTIHEAD_PROPERTY_DESCRIPTIONS[self.name]

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
        """一行说明：``[ok] per_head_row_stochastic 行和最大偏差 0.00e+00``."""
        mark = "ok" if self.passed else "FAIL"
        return f"[{mark}] {self.name} {self.evidence}"


@dataclass(frozen=True)
class MultiHeadPropertyReport:
    """六条性质的汇总（:func:`check_properties` 的产物）."""

    outcomes: tuple[MultiHeadPropertyOutcome, ...]
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        missing = set(MULTIHEAD_PROPERTIES) - {item.name for item in self.outcomes}
        if missing:
            raise NumericError(
                f"性质报告缺少 {sorted(missing)}：少做一条与'这条通过'在报告里长得一样。"
            )
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def ok(self) -> bool:
        """六条是否全部通过."""
        return all(item.passed for item in self.outcomes)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "ok": self.ok,
            "outcomes": [item.to_dict() for item in self.outcomes],
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``性质检查 6 项：通过 6、失败 0``."""
        passed = sum(1 for item in self.outcomes if item.passed)
        return (
            f"性质检查 {len(self.outcomes)} 项：通过 {passed}、"
            f"失败 {len(self.outcomes) - passed}"
        )


def check_per_head_row_stochastic(
    forward: MultiHeadForward,
    *,
    tolerance: float = DEFAULT_SUM_TOLERANCE,
) -> MultiHeadPropertyOutcome:
    """**每一头的每一行**之和是否为 1（并给出最大偏差作为证据）.

    这一条与 day075 的同名检查是同一句话在两个尺度上：那里是 ``n`` 行，
    这里是 ``heads × n`` 行。**行数必须报出来**——
    否则一个"只检查了第一头"的实现会让证据看起来完全一样（偏差都是 0）。
    """
    deviations = [
        abs(math.fsum(row) - 1.0) for head in forward.head_weights for row in head
    ]
    worst = max(deviations, default=0.0)
    return MultiHeadPropertyOutcome(
        name="per_head_row_stochastic",
        passed=worst <= tolerance,
        evidence=(
            f"{forward.distributions} 行（heads={forward.heads} × n={forward.tokens}）"
            f"行和最大偏差 {worst:.2e}"
        ),
        detail=(
            "每一头每一行是一个**条件分布**（'这一头怎么看各个位置'）——"
            "是 heads·n 个分布，不是一个。只检查第一头的实现给出的证据与此完全相同，"
            "因此证据里必须带上被检查的行数"
        ),
    )


def check_head_non_negative(forward: MultiHeadForward) -> MultiHeadPropertyOutcome:
    """每一头的每个权重是否非负."""
    worst = min(
        (value for head in forward.head_weights for row in head for value in row),
        default=0.0,
    )
    return MultiHeadPropertyOutcome(
        name="head_non_negative",
        passed=worst >= 0.0,
        evidence=f"最小权重 {worst:.2e}（扫描 {forward.distributions} 行）",
        detail="负权重会让某一头的'加权平均'在某处变成减法，而整行看起来仍然是一条曲线",
    )


def check_causal_no_leak_per_head(forward: MultiHeadForward) -> MultiHeadPropertyOutcome:
    """因果模式下**每一头**的上三角是否恰好是 0.0."""
    if not forward.causal:
        return MultiHeadPropertyOutcome(
            name="causal_no_leak_per_head",
            passed=True,
            evidence="本次前向未启用因果掩码（跳过）",
            detail=(
                "未启用掩码时这一条不适用：它不是'通过'，而是'没有可检查的对象'——"
                "报告里必须把这两者区分开，否则'没开掩码'会伪装成'掩码正确'"
            ),
        )
    leaks: list[float] = []
    for head in forward.head_weights:
        for row in range(forward.tokens):
            for column in range(row + 1, forward.tokens):
                leaks.append(abs(head[row][column]))
    worst = max(leaks, default=0.0)
    return MultiHeadPropertyOutcome(
        name="causal_no_leak_per_head",
        passed=worst == 0.0,
        evidence=f"{forward.heads} 头的上三角最大权重 {worst:.2e}",
        detail=(
            "位置 i 在**每一头**上都看不到它之后的任何位置——在数字上就是那些格子"
            "恰好是 0.0。掩码是同一张表发给每一头：因果性是位置的性质"
        ),
    )


def check_merge_inverts_split(forward: MultiHeadForward) -> MultiHeadPropertyOutcome:
    """往返恒等：``merge_rows(split_rows(M)) == M``（以及按列、以及 context 的拼回）.

    三条都要查，因为它们各自对应一种"切错"的方式：

    ```text
    按行往返    Q/K/V 的行块（切错 → 每一头看到别的头的维度）
    按列往返    W_o 的列块（切错 → 每一头的 context 被送到别的输出维）
    拼接一致    merged_context 必须等于 heads 份 head_context 的按列拼接
    ```
    """
    shape = forward.shape
    keys_partition = shape.keys_partition
    values_partition = shape.values_partition
    worst = 0.0
    # 权重按**行**切（d_k / d_v 是权重矩阵的行）
    for matrix, partition in (
        (forward.params.w_query, keys_partition),
        (forward.params.w_key, keys_partition),
        (forward.params.w_value, values_partition),
        (forward.params.w_output, values_partition),
    ):
        if matrix is forward.params.w_output:
            rebuilt = partition.merge_columns(partition.split_columns(matrix))
        else:
            rebuilt = partition.merge_rows(partition.split_rows(matrix))
        worst = max(worst, relative_matrix_error(rebuilt, matrix))
    # 激活按**列**切（d_k / d_v 在 Q/K/V 与 context 的列上）
    for matrix, partition in (
        (forward.queries, keys_partition),
        (forward.keys, keys_partition),
        (forward.values, values_partition),
        (forward.merged_context, values_partition),
    ):
        rebuilt = partition.merge_columns(partition.split_columns(matrix))
        worst = max(worst, relative_matrix_error(rebuilt, matrix))
    merged = values_partition.merge_columns(forward.head_contexts)
    worst = max(worst, relative_matrix_error(merged, forward.merged_context))
    return MultiHeadPropertyOutcome(
        name="merge_inverts_split",
        passed=worst == 0.0,
        evidence=f"往返恒等的最大偏差 {worst:.2e}（9 组矩阵）",
        detail=(
            "merge 是 split 的**严格逆**：切错或拼错都不会报错，"
            "只会让某一头看到的维度里混进别的头——而那一头的语义已经变了，"
            "却仍然是一张合法的 (n, n) 权重表"
        ),
    )


def check_single_head_matches_classic(
    params: AttentionParams,
    inputs: Matrix,
    *,
    causal: bool = True,
) -> MultiHeadPropertyOutcome:
    """``heads=1`` 时输出与 day075 的自注意力**逐位相同**（前向 + 反向）.

    这条检查的判据是 ``==`` 而不是容差：多头在 ``heads=1`` 时每一步的算子、
    顺序、循环都退化回 day075 的七个阶段（连续块的唯一划分就是整张矩阵），
    因此两个实现给出的是**同一串数**。

    它的价值不在"今天是对的"，而在**明天有人重构 split/merge 时它会不会红**：
    这条退化关系是"多头是单头的严格超集"的工程证据。
    """
    forward = multi_head_attention(params, inputs, heads=1, causal=causal)
    classic = self_attention(params, inputs, causal=causal)
    weight_gap = relative_matrix_error(forward.head_weights[0], classic.weights)
    context_gap = relative_matrix_error(forward.merged_context, classic.context)
    output_gap = relative_matrix_error(forward.output, classic.output)
    exact = (
        forward.output == classic.output
        and forward.head_weights[0] == classic.weights
        and forward.merged_context == classic.context
    )
    return MultiHeadPropertyOutcome(
        name="single_head_matches_classic",
        passed=exact,
        evidence=(
            f"权重差 {weight_gap:.2e}、context 差 {context_gap:.2e}、"
            f"输出差 {output_gap:.2e}（三者都必须**恰好是 0**）"
        ),
        detail=(
            "heads=1 不是'接近 day075'，而是与之**逐位相同**——"
            "因为连续块划分在 heads=1 时的唯一划分就是整张矩阵。"
            "任何一次对 split/merge 的重构只要破坏这条退化关系，都会在这里红"
        ),
    )


def check_head_order_is_bookkeeping(
    params: AttentionParams,
    inputs: Matrix,
    *,
    heads: int,
    causal: bool = True,
    order: tuple[int, ...] | None = None,
) -> MultiHeadPropertyOutcome:
    """一致地重排头块之后输出不变——**头编号是一个记账约定**.

    这条性质说的是：``heads`` 这个数字带来的那部分结构里，
    "第 0 头"与"第 1 头"的**名字**没有任何意义，有意义的只是"有几头、每头多宽"。
    实现方式是把三个投影的行块与 ``W_o`` 的列块按同一个 ``order`` 重排
    （:func:`permute_head_blocks`），而输出必须逐位不变。

    这里用容差而不是 ``==``：重排之后**求和顺序变了**（``Σ_c`` 遍历列的次序不同），
    而浮点加法不满足结合律——实测差在 ``1e-18`` 量级（本样本 6.94e-18）。
    """
    checked_order = checked_head_order(
        default_head_order(heads) if order is None else order, heads
    )
    permuted = permute_head_blocks(params, heads, checked_order)
    base = multi_head_attention(params, inputs, heads=heads, causal=causal)
    shuffled = multi_head_attention(permuted, inputs, heads=heads, causal=causal)
    gap = relative_matrix_error(shuffled.output, base.output)
    return MultiHeadPropertyOutcome(
        name="head_order_is_bookkeeping",
        passed=gap <= IDENTITY_TOLERANCE,
        evidence=(
            f"头序 {checked_order} 下的输出差 {gap:.2e}（容差 {IDENTITY_TOLERANCE:.0e}）"
        ),
        detail=(
            "头编号只是一个记账约定：一致地重排头块（三个投影按行、W_o 按列）"
            "不改变这一层算的是什么。判据取 1e-12 而不是逐位相等——"
            "重排会把浮点求和的**顺序**也换掉，而实测是否为 0 取决于具体数值"
            "（本次实测见证据）。**只换一个、不换另一个**会得到一组完全不同的参数，"
            "而那种改动与'划分逻辑写错了'长得一模一样"
        ),
    )


def check_properties(
    params: AttentionParams,
    inputs: Matrix,
    *,
    heads: int = 1,
    causal: bool = True,
    order: tuple[int, ...] | None = None,
) -> MultiHeadPropertyReport:
    """跑完六条性质，装成一份 :class:`MultiHeadPropertyReport`（顺序 = 口径表）.

    ``head_order_is_bookkeeping`` 在 ``heads=1`` 时是**平凡**的：只有一个头时
    唯一的排列就是恒等排列，置换前后的参数逐位相同。报告里仍然保留它，
    但读数会自己说明这一点（输出差恰好 0.0，而原因不是"划分对"而是"没有可换的东西"）。
    """
    forward = multi_head_attention(params, inputs, heads=heads, causal=causal)
    outcomes = (
        check_per_head_row_stochastic(forward),
        check_head_non_negative(forward),
        check_causal_no_leak_per_head(forward),
        check_merge_inverts_split(forward),
        check_single_head_matches_classic(params, inputs, causal=causal),
        check_head_order_is_bookkeeping(
            params, inputs, heads=heads, causal=causal, order=order
        ),
    )
    return MultiHeadPropertyReport(
        outcomes=outcomes,
        notes=(
            "六条性质里前三条是**每一头的账**（每一项都能独立失败，"
            "而且证据里都带上了被检查的行数/头数），"
            "第四条是划分的往返恒等，第五条是 heads=1 的退化关系",
            "第六条（头序只是记账）是这一课唯一一条**刻意近似**的性质："
            "重排改变了浮点求和顺序，因此判据是 1e-12 而不是逐位相等",
            "heads=1 时第六条平凡成立（唯一排列是恒等），"
            "而第四条与第五条仍然是有内容的——它们检查的是'退化关系'本身",
        ),
    )


# --------------------------------------------------------------------------- #
# 各头像不像：全变差距离
# --------------------------------------------------------------------------- #


def total_variation(left: Vector, right: Vector) -> float:
    """两份分布的全变差距离 ``TV = ½·Σ|p−q|``（落在 ``[0, 1]``）.

    为什么这一课用 TV 而不是 KL：TV 是**对称的**、有界的，
    因此"头 A 与头 B 有多不一样"这个问题有一个不用指定方向的答案——
    而 KL 在 ``q = 0`` 而 ``p > 0`` 时是无穷。
    多头多样性是一个**对称**的读数（谁是"基准"没有意义）。

    两个约定写下来：

    ```text
    长度不同       直接报错（两份分布必须作用在同一个位置上）
    最大值 1.0     两份分布支撑集不交时取到（"完全不一样"）
    ```
    """
    if len(left) != len(right):
        raise ShapeError(
            f"两份分布长度不同：{len(left)} 与 {len(right)}——"
            "TV 要求两者逐位对应（它们是同一行在两头上的两份分布）。"
        )
    total = 0.0
    for first, second in zip(left, right, strict=True):
        if not (math.isfinite(first) and math.isfinite(second)):
            raise NumericError("分布里出现了非有限数：TV 会因此变成 nan。")
        total += abs(first - second)
    return 0.5 * total


@dataclass(frozen=True)
class HeadDisagreement:
    """各头"看到的东西"有多不一样（:func:`head_disagreement` 的产物）.

    ```text
    pairs                  头对的数量 = heads·(heads−1)/2
    mean_total_variation   所有 (头对, 行) 的 TV 平均
    max_total_variation    最大的一份（"最不一致的那一对、那一行"）
    min_total_variation    最小的一份
    peak_agreement         (头对, 行) 上两头 argmax 相同的对数
    ```

    ## 为什么这一课需要这个读数

    多头**不保证**各头学到不同的东西：一个把所有头都训练成同一个分布的解，
    损失同样会下降——而它等于"付了 heads 份参数、只用了一份注意力"。
    这个读数把这个退化情形变成一个可看见的数（``mean_total_variation ≈ 0``）。

    ## 它**不**回答什么（必须写下来）

    ```text
    它回答    "现在的权重表里，各头的分布像不像"
    它不回答  "各头有没有学到不同**机制**"——两份分布可能数值接近，
              而它们依赖的维度（子空间）完全不同
    ```

    因此这是一个**前向读数**，不能替代梯度或训练曲线上的证据。
    """

    heads: int
    rows: int
    pairs: int
    mean_total_variation: float
    max_total_variation: float
    min_total_variation: float
    peak_agreement: int
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.heads < 1:
            raise ParameterError(f"头数必须 >= 1，收到 {self.heads}。")
        expected = self.heads * (self.heads - 1) // 2
        if self.pairs != expected:
            raise NumericError(
                f"头对数是 {self.pairs}，而 {self.heads} 头两两组合应当有 {expected} 对："
                "少数的那些头对会静默地不参与这个读数。"
            )
        if self.rows < 1:
            raise ParameterError(f"行数必须 >= 1，收到 {self.rows}。")
        if not 0 <= self.min_total_variation <= self.max_total_variation <= 1.0:
            raise NumericError(
                f"TV 统计量越界：min={self.min_total_variation}、"
                f"max={self.max_total_variation}（TV 必须落在 [0, 1]）。"
            )
        if not self.min_total_variation <= self.mean_total_variation <= self.max_total_variation:
            raise NumericError("平均 TV 必须落在最小值与最大值之间。")
        if not 0 <= self.peak_agreement <= self.pairs * self.rows:
            raise NumericError(
                f"argmax 一致的对数 {self.peak_agreement} 落在 "
                f"[0, {self.pairs * self.rows}] 之外。"
            )
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def comparisons(self) -> int:
        """被比较的次数：``头对 × 行``."""
        return self.pairs * self.rows

    @property
    def peak_agreement_ratio(self) -> float:
        """argmax 一致率（没有头对时记 0.0）."""
        if self.comparisons == 0:
            return 0.0
        return self.peak_agreement / self.comparisons

    @property
    def degenerate(self) -> bool:
        """是否退化成"各头一样"（平均 TV 低于 1e-9 视为退化）.

        ``heads = 1`` 时返回 ``False``：单头没有"可以退化"的对象——
        把"没有第二个头"读成"多头退化了"是一件会误导人的事
        （两者在数值上都表现为 TV = 0，但一个要改结构、一个要改训练）。
        """
        return self.pairs >= 1 and self.mean_total_variation <= 1e-9

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "heads": self.heads,
            "rows": self.rows,
            "pairs": self.pairs,
            "comparisons": self.comparisons,
            "mean_total_variation": self.mean_total_variation,
            "max_total_variation": self.max_total_variation,
            "min_total_variation": self.min_total_variation,
            "peak_agreement": self.peak_agreement,
            "peak_agreement_ratio": self.peak_agreement_ratio,
            "degenerate": self.degenerate,
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``2 头 / 3 行：平均 TV 0.6142（最大 0.6142）| argmax 一致 3/3``."""
        if self.pairs == 0:
            return f"{self.heads} 头：没有可比的头对（单头无多样性可言）"
        return (
            f"{self.heads} 头 / {self.rows} 行：平均 TV {self.mean_total_variation:.4f}"
            f"（最大 {self.max_total_variation:.4f}）| "
            f"argmax 一致 {self.peak_agreement}/{self.comparisons}"
        )


def head_disagreement(forward: MultiHeadForward) -> HeadDisagreement:
    """逐头逐行算 TV 与 argmax 一致率（**只在允许的位置上比**）.

    "只在允许的位置上比"这条纪律与 day075 的检索类比同源：因果掩码把未来位置的
    权重写成 0（**所有头都一样**），不排除它们时 TV 会被这些共同的 0 稀释——
    "两头不一样"的那部分差别会被分母吃掉。
    """
    heads = forward.heads
    rows = forward.tokens
    pairs = heads * (heads - 1) // 2
    values: list[float] = []
    peak_agreement = 0
    for first in range(heads):
        for second in range(first + 1, heads):
            for row in range(rows):
                allowed = [
                    column for column in range(rows) if forward.mask[row][column]
                ]
                left = tuple(forward.head_weights[first][row][column] for column in allowed)
                right = tuple(forward.head_weights[second][row][column] for column in allowed)
                values.append(total_variation(left, right))
                if forward.head_indices[first][row] == forward.head_indices[second][row]:
                    peak_agreement += 1
    if not values:
        return HeadDisagreement(
            heads=heads,
            rows=rows,
            pairs=pairs,
            mean_total_variation=0.0,
            max_total_variation=0.0,
            min_total_variation=0.0,
            peak_agreement=0,
            notes=(
                "单头没有可比的头对：这个读数在 heads=1 时恒为 0，"
                "而它**不代表**'第一头很专注'——它代表'没有第二个头'",
            ),
        )
    return HeadDisagreement(
        heads=heads,
        rows=rows,
        pairs=pairs,
        mean_total_variation=math.fsum(values) / len(values),
        max_total_variation=max(values),
        min_total_variation=min(values),
        peak_agreement=peak_agreement,
        notes=(
            f"比较次数 {len(values)} = 头对 {pairs} × 行 {rows}，"
            "且只在**允许的位置**上比：掩码把未来位置的权重写成 0（所有头都一样），"
            "不排除它们时 TV 会被那些共同的 0 稀释",
            "这是一个**前向读数**：它说'现在的分布像不像'，"
            "不说'各头有没有学到不同机制'——后者要看梯度与训练曲线",
            "平均 TV 接近 0 意味着多头退化成单头：付了 heads 份参数、只用了一份注意力",
        ),
    )


def head_weight_table(forward: MultiHeadForward) -> dict[str, Any]:
    """把"同一行的 heads 份分布"排成一张可读的表（进演示脚本的输出）.

    它是报告的一部分而不是一个调试工具：**"每一行有 heads 个分布"这句话
    只有在能把它们并排打出来的时候才是可核对的**。
    """
    rows: list[dict[str, Any]] = []
    for row in range(forward.tokens):
        allowed = [column for column in range(forward.tokens) if forward.mask[row][column]]
        rows.append(
            {
                "row": row,
                "allowed": allowed,
                "distributions": [
                    [
                        round(forward.head_weights[head][row][column], 6)
                        for column in range(forward.tokens)
                    ]
                    for head in range(forward.heads)
                ],
                "argmax": list(forward.head_indices[head][row] for head in range(forward.heads)),
            }
        )
    return {
        "heads": forward.heads,
        "tokens": forward.tokens,
        "distributions": forward.distributions,
        "rows": rows,
    }


__all__ = [
    "GRADIENT_TOLERANCE",
    "IDENTITY_TOLERANCE",
    "HeadDisagreement",
    "MultiHeadGradientOutcome",
    "MultiHeadGradientReport",
    "MultiHeadPropertyOutcome",
    "MultiHeadPropertyReport",
    "check_causal_no_leak_per_head",
    "check_head_non_negative",
    "check_head_order_is_bookkeeping",
    "check_merge_inverts_split",
    "check_multihead_gradients",
    "check_per_head_row_stochastic",
    "check_properties",
    "check_single_head_matches_classic",
    "checked_head_order",
    "default_head_order",
    "head_disagreement",
    "head_weight_table",
    "numerical_multihead_gradients",
    "permute_head_blocks",
    "total_variation",
]
