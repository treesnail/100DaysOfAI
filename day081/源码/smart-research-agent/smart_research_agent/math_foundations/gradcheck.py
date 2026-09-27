"""梯度对照：解析式与数值差分必须对得上（day074 / Math-D2）.

这一层与 ``bridge`` 是同一件事的第二个版本——``bridge`` 比的是**函数值**，
这里比的是**变化率**：

```text
bridge     我们的 cosine/softmax/sigmoid 与生产实现的**值**是否逐位一致
gradcheck  我们的**解析梯度**与对生产实现的**数值差分**是否逐点一致
```

## 为什么梯度的对照更值得做一次

函数值写错通常会被"看一眼"抓到（一个明显的偏差）；**梯度写错不会**：

```text
解析梯度少乘一个因子      训练仍然在下降，只是慢一点（看起来像 lr 设小了）
符号写反                  某些任务上仍然收敛（因为方向"大致"还是对的）
softmax 的雅可比漏掉一项  交叉熵的梯度仍然是 p − y（**恰好抵消**）
```

最后一条是这一课最想讲清的：``softmax`` 的雅可比是 ``p_i(δ_ij − p_j)``，
它与 ``−log p_y`` 的导数相乘之后**化简成了 ``p − onehot(y)``**——
中间那一大堆项全部抵消。于是"softmax 的雅可比写错了"这件事
在"交叉熵的梯度"上**看不见**（因为正确的实现也不直接用它）。
只有把 softmax 的雅可比单独拿出来与数值雅可比比，才能发现它错了。

## 两种结论，而不是三种

``bridge`` 有三种结论（``agrees`` / ``differs`` / ``rejected``）。这里只有两种：

```text
rejected 的含义是"一边拒绝了这个输入、另一边给了值"——那是两个**不同用途**的
         实现之间的约定差异（例如零向量的归一化）
本模块两侧都是**我们自己算的**（解析式与数值差分），输入由我们选定，
         因此不存在"一边拒绝一边放行"这种情况：
         数值差分被拒（函数在扰动点上返回非有限数）时，那只是"这个点不适合做对照"，
         它应当被**从样本里换掉**，而不是被记成一条"记录的差异"
```

把这一条写下来是有必要的：一个"没有出现过的状态"与"处理了但没发生"不一样，
而报告里两者都只会留下一片空白。

## 容差为什么要由"分辨率"决定，而不是拍一个数

数值差分自身的误差下限是 ``eps·|f|/(2h)``（见 :func:`difference_resolution`）——
它就是这个对照的**分辨率下限**：

```text
|f| ≈ 1      h = 1e-6   →  1.1e-10       容差 1e-8 是它的 90 倍，够松
|f| ≈ 30     h = 1e-6   →  3.3e-09       容差 1e-8 是它的 3 倍，仍然够
|f| ≈ 100    h = 1e-6   →  1.1e-08       因此判据必须**按量级缩放**
```

两个结论由此而来，它们都是**算出来的**而不是定出来的：

```text
① 容差取 1e-8（而不是 1e-12）：低于分辨率阈值的容差会产出一条
   "不一致"，而两边其实都没错——那种假警报比"漏报"更坏，
   因为它会让人去改一份本来正确的代码
② 判据按 |数值| 缩放（:func:`max_scaled_gap`）：导数为 100 的量与
   导数为 1 的量，用同一个绝对阈值去卡等于对前者更苛刻
```

而"真的写错了"通常差 ``1e-3`` 以上——与 1e-8 之间隔着五个数量级，
因此这个容差仍然有充足的区分度。在报告里，
"最大误差 1e-10"与"最大误差 1e-3"是两个**完全不同**的结论，
而它们各自都有一个具体数字可以指认。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.math_foundations.autograd import Scalar, value_and_grad
from smart_research_agent.math_foundations.calculus import (
    DEFAULT_STEP,
    ScalarFunction,
    gradient,
    jacobian,
)
from smart_research_agent.math_foundations.errors import MathError, NumericError
from smart_research_agent.math_foundations.linalg import softmax
from smart_research_agent.math_foundations.probability import perplexity_from_entropy
from smart_research_agent.math_foundations.types import (
    FLOAT_EPSILON,
    GRADIENT_AUTOGRAD_CHAIN,
    GRADIENT_CROSS_ENTROPY,
    GRADIENT_LOG_SIGMOID,
    GRADIENT_PERPLEXITY,
    GRADIENT_SIGMOID,
    GRADIENT_SOFTMAX,
    GRADIENT_STATUSES,
    GRADIENT_TARGETS,
    GRADIENT_TARGET_DESCRIPTIONS,
    GRADIENT_TARGET_FORMULAS,
    GRADIENT_TARGET_SOURCES,
    Matrix,
    Vector,
    validate_vector,
)

#: 梯度对照的容差（**相对判据**：见 :func:`max_scaled_gap` 的说明）.
#:
#: 为什么不是 1e-9：中心差分自身的误差下限是 ``eps·|f|/(2h)``
#: （见 :func:`difference_resolution`）。当 ``|f|`` 是 30 的量级时它是 ``3e-9``——
#: 也就是说 1e-9 这个容差**低于测量手段的分辨率**，
#: 于是报告里会出现一条"不一致"，而两边都没错。
#: 取 1e-8（比分辨率松一档），同时保留"真的写错了通常差 1e-3 以上"这条区分度。
GRADIENT_TOLERANCE = 1e-8

#: 对照用的打分样本（**写死在代码里**：随机输入会让"这次对上了"不可复现）.
#: 长度覆盖 1 / 2 / 3 / 4 四档，因为 softmax 的雅可比在长度为 1 时退化成全 0。
LOGIT_VECTORS: tuple[Vector, ...] = (
    (0.5,),
    (1.0, 2.0),
    (-1.0, 0.0, 3.0),
    (0.3, -0.7, 1.4, 2.0),
    (10.0, 10.0, 10.0),  # 三者相同：雅可比是对称的，容易看出"每一项差多少"
)

#: 交叉熵对照用的"正确答案下标"（与 LOGIT_VECTORS 一一对应）.
TARGET_INDICES: tuple[int, ...] = (0, 1, 2, 0, 1)

#: sigmoid / log_sigmoid 对照用的点（含 ±30 这类"已经饱和"的位置）.
SCALAR_POINTS: tuple[float, ...] = (0.0, 1.0, -1.0, 0.5, -0.5, 30.0, -30.0)

#: 困惑度对照用的输入（都是合法 loss：**严格大于 0** 且远小于 700，不会触发饱和分支）.
#:
#: 为什么没有 ``0.0``：中心差分要在两侧各取一个点，而 ``sft.loss.perplexity``
#: 的定义域是 ``[0, ∞)``——``L = 0`` 时左邻点 ``−h`` 落在定义域之外，
#: 生产实现会（正确地）抛 ``SFTLossError``。这不是"两边不一致"，
#: 而是"这个点不适合做双向差分"，因此它应当被**从样本里换掉**，
#: 而不是被记成一条"记录的差异"（见模块说明里"两种结论而不是三种"）。
PERPLEXITY_INPUTS: Vector = (0.25, math.log(2.0), 1.0, math.log(100.0), 3.0)

#: 自动微分对照用的复合表达式：``(名字, 表达式, 输入点)``.
AUTOGRAD_CASES: tuple[tuple[str, Callable[[list[Scalar]], Scalar], Vector], ...] = (
    # 单层：sigmoid 与内层线性（链式法则只有一段）
    ("sigmoid(3x+1)", lambda xs: (xs[0] * 3.0 + 1.0).sigmoid(), (0.7,)),
    # 两层：softplus（log(1+e^x)）+ 平方（外层与内层的点必须各自取对）
    ("log(1+exp(x)) + x²", lambda xs: (xs[0].exp() + 1.0).log() + xs[0] * xs[0], (0.4,)),
    # 用到两次同一个输入：x·x 的两条边都要把梯度累加回来（dy/dx = 2x）
    ("tanh(x)·relu(x+1)", lambda xs: xs[0].tanh() * (xs[0] + 1.0).relu(), (0.25,)),
    # 两个输入、含除法与 exp：分子分母都依赖 y
    (
        "(x·y + exp(y)) / (1 + x²)",
        lambda xs: (xs[0] * xs[1] + xs[1].exp()) / (1.0 + xs[0] * xs[0]),
        (0.6, -0.8),
    ),
    # 三个输入、含 sqrt 与 sin：链式法则在多层复合下仍然只是"乘一串局部导数"
    (
        "sqrt(sin(x)+2)·y + z²",
        lambda xs: ((xs[0].sin() + 2.0).sqrt()) * xs[1] + xs[2] * xs[2],
        (0.9, 1.3, -0.4),
    ),
)


@dataclass(frozen=True)
class GradientOutcome:
    """一项梯度对照的结论：最大逐点误差 + 解析式 + 数值式的来源.

    ``max_absolute_error`` 是"逐点比较"的分辨率——报告里给的是**误差**，
    而不是一句"一致"："一致"是结论，误差是证据（与 ``bridge.CheckOutcome`` 同一取向）。
    """

    target: str
    status: str
    max_absolute_error: float
    compared_points: int
    message: str
    source: str
    tolerance: float = GRADIENT_TOLERANCE
    max_scaled_error: float = 0.0

    def __post_init__(self) -> None:
        if self.target not in GRADIENT_TARGETS:
            raise MathError(
                f"不认识的梯度对照项 {self.target!r}：可用取值 {list(GRADIENT_TARGETS)}。"
            )
        if self.status not in GRADIENT_STATUSES:
            raise MathError(
                f"不认识的梯度对照结论 {self.status!r}：只允许 {list(GRADIENT_STATUSES)}"
                "（比函数值对照少一种：这里两侧都是我们自己算的，"
                "不存在'一边拒绝一边放行'）。"
            )
        if self.max_absolute_error < 0 or math.isnan(self.max_absolute_error):
            raise NumericError(
                f"最大绝对误差不能为负或 NaN：{self.max_absolute_error}。"
            )
        if self.max_scaled_error < 0 or math.isnan(self.max_scaled_error):
            raise NumericError(
                f"最大相对误差不能为负或 NaN：{self.max_scaled_error}。"
            )

    @property
    def formula(self) -> str:
        """这一项对照的解析梯度公式（报告里要能读出"比的是哪个式子"）."""
        return GRADIENT_TARGET_FORMULAS[self.target]

    @property
    def ok(self) -> bool:
        """这一项是否没有分歧（``differs`` 才算分歧）."""
        return self.status != "differs"

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "target": self.target,
            "description": GRADIENT_TARGET_DESCRIPTIONS[self.target],
            "formula": self.formula,
            "source": self.source,
            "status": self.status,
            "ok": self.ok,
            "max_absolute_error": self.max_absolute_error,
            "max_scaled_error": self.max_scaled_error,
            "compared_points": self.compared_points,
            "tolerance": self.tolerance,
            "message": self.message,
        }

    def summary_line(self) -> str:
        """一行说明：``[agrees] softmax 39 个点，最大误差 1.82e-10``."""
        return (
            f"[{self.status}] {self.target} {self.compared_points} 个点，"
            f"最大误差 {self.max_absolute_error:.2e}"
            f"（相对 {self.max_scaled_error:.2e}）"
        )


@dataclass(frozen=True)
class GradientReport:
    """六项梯度对照的汇总（:func:`check_gradients_all` 的产物）."""

    outcomes: tuple[GradientOutcome, ...]
    tolerance: float = GRADIENT_TOLERANCE
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        missing = set(GRADIENT_TARGETS) - {item.target for item in self.outcomes}
        if missing:
            raise MathError(
                f"梯度对照报告缺少 {sorted(missing)}：少做的对照与'对照通过'"
                "在报告里长得一样，因此这里当场拒绝——"
                "一份不完整的报告比没有报告更危险。"
            )
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def disagreements(self) -> tuple[GradientOutcome, ...]:
        """真正需要改代码的那些（``differs``）."""
        return tuple(item for item in self.outcomes if item.status == "differs")

    @property
    def ok(self) -> bool:
        """没有任何分歧."""
        return not self.disagreements

    @property
    def worst_error(self) -> float:
        """六项里最大的那一个误差（"这份报告的分辨率"）."""
        return max((item.max_absolute_error for item in self.outcomes), default=0.0)

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "tolerance": self.tolerance,
            "ok": self.ok,
            "worst_error": self.worst_error,
            "disagreements": [item.target for item in self.disagreements],
            "outcomes": [item.to_dict() for item in self.outcomes],
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``梯度对照 6 项：一致 6、分歧 0 | 最大误差 3.10e-11``."""
        agrees = sum(1 for item in self.outcomes if item.status == "agrees")
        return (
            f"梯度对照 {len(self.outcomes)} 项：一致 {agrees}、"
            f"分歧 {len(self.disagreements)} | 最大误差 {self.worst_error:.2e}"
        )


def _flat(matrix: Matrix) -> tuple[float, ...]:
    """把矩阵按行优先压平（对照时两侧必须用**同一个顺序**）."""
    return tuple(value for row in matrix for value in row)


def difference_resolution(*, magnitude: float, step: float = DEFAULT_STEP) -> float:
    """数值差分在"函数值量级为 ``magnitude``"时的**分辨率下限** ``eps·|f|/(2h)``.

    这个数字来自一条具体的浮点事实：``f(x+h)`` 与 ``f(x−h)`` 各带一个
    ``eps·|f|`` 量级的舍入误差，两者相减之后误差**不会抵消**，
    再除以 ``2h`` 就得到导数上的误差下限。

    ```text
    |f| ≈ 1        h = 1e-6    →  分辨率 ≈ 1.1e-10
    |f| ≈ 30       h = 1e-6    →  分辨率 ≈ 3.3e-09   ← 容差必须比它松
    |f| ≈ 100      h = 1e-6    →  分辨率 ≈ 1.1e-08
    ```

    它把"容差该取多少"从一个拍脑袋的数字变成一次可以验算的计算——
    这也是本模块的容差取 ``1e-8`` 而不是更小的**唯一**理由。
    """
    if not math.isfinite(magnitude):
        raise NumericError(f"量级必须是有限实数，收到 {magnitude!r}。")
    if not math.isfinite(step) or step <= 0:
        raise NumericError(f"步长必须是正的有限数，收到 {step!r}。")
    return FLOAT_EPSILON * abs(magnitude) / (2.0 * step)


def max_absolute_gap(left: Vector, right: Vector) -> float:
    """两个等长向量的最大逐点绝对误差（长度不同直接 ``inf``）.

    "长度不同"返回 ``inf`` 而不是抛异常：对照的价值在于**给出结论**，
    而"长度都不一样"就是最严重的那种不一致（它说明两侧算的不是同一件事）。
    """
    if len(left) != len(right):
        return math.inf
    worst = 0.0
    for first, second in zip(left, right):
        if not (math.isfinite(first) and math.isfinite(second)):
            return math.inf
        worst = max(worst, abs(first - second))
    return worst


def max_scaled_gap(left: Vector, right: Vector) -> float:
    """最大**相对**逐点误差 ``|l − r| / max(1, |r|)``（对照实际使用的判据）.

    为什么判据要缩放：一个"导数是 100"的量，绝对误差 1e-8 是精确到 10 位；
    而一个"导数是 1"的量，同样的绝对误差只精确到 8 位。
    数值差分的误差下限正比于**函数值的量级**（见 :func:`difference_resolution`），
    因此用绝对阈值去卡不同量级的量，等于对大数更苛刻——
    那会产出一条"不一致"，而它其实是"测量手段的极限"。

    分母取 ``max(1, |r|)`` 与 :func:`types.close` 同一口径：
    0 附近自动退化成绝对判据。
    """
    if len(left) != len(right):
        return math.inf
    worst = 0.0
    for first, second in zip(left, right):
        if not (math.isfinite(first) and math.isfinite(second)):
            return math.inf
        scale = max(1.0, abs(second))
        worst = max(worst, abs(first - second) / scale)
    return worst


def _verdict(
    target: str,
    analytic: Vector,
    numeric: Vector,
    *,
    tolerance: float,
    message: str,
) -> GradientOutcome:
    """按最大**相对**逐点误差给出一项对照的结论（**只在这一处比较**）."""
    gap = max_absolute_gap(analytic, numeric)
    scaled = max_scaled_gap(analytic, numeric)
    status = "agrees" if scaled <= tolerance else "differs"
    detail = (
        f"{message}；逐点在容差 {tolerance:.1e} 内一致（最大绝对误差 {gap:.2e}）"
        if status == "agrees"
        else f"{message}；最大相对误差 {scaled:.3e} 超过容差 {tolerance:.1e}"
    )
    return GradientOutcome(
        target=target,
        status=status,
        max_absolute_error=gap,
        compared_points=len(analytic),
        message=detail,
        source=GRADIENT_TARGET_SOURCES[target],
        tolerance=tolerance,
        max_scaled_error=scaled,
    )


# --------------------------------------------------------------------------- #
# 六项对照
# --------------------------------------------------------------------------- #


def check_softmax_jacobian(*, step: float = DEFAULT_STEP, tolerance: float = GRADIENT_TOLERANCE) -> GradientOutcome:
    """``softmax`` 的雅可比：解析式 ``p_i(δ_ij − p_j)`` vs 数值雅可比.

    这一项单独存在的原因见模块说明：**它在交叉熵的梯度里会被抵消掉**，
    因此在"交叉熵对不对"这个问题上，它错与不错给出的答案一样。
    只有把它单独比一次，"softmax 的雅可比"这个知识才是有护栏的。
    """
    analytic: list[float] = []
    numeric: list[float] = []
    for logits in LOGIT_VECTORS:
        probabilities = softmax(logits)
        expected = [
            [
                probabilities[row] * ((1.0 if row == column else 0.0) - probabilities[column])
                for column in range(len(probabilities))
            ]
            for row in range(len(probabilities))
        ]
        observed = jacobian(lambda vector: softmax(vector), logits, step=step)
        analytic.extend(_flat(tuple(tuple(row) for row in expected)))
        numeric.extend(_flat(observed))
    return _verdict(
        GRADIENT_SOFTMAX,
        tuple(analytic),
        tuple(numeric),
        tolerance=tolerance,
        message=f"{len(LOGIT_VECTORS)} 组打分上的完整雅可比（解析 vs 数值）",
    )


def check_cross_entropy_gradient(
    *, step: float = DEFAULT_STEP, tolerance: float = GRADIENT_TOLERANCE
) -> GradientOutcome:
    """``∂(−log p_y)/∂z = p − onehot(y)`` vs 对**生产实现**的数值差分.

    这一项是"day054 那一行公式"的护栏：``alignment`` 的 DPO 目标里
    对 log 概率求导用的就是这个结论。生产实现是 ``sft.loss.cross_entropy``——
    它的入参是**打分**（不是概率），因此数值差分也是对打分做的。
    """
    from smart_research_agent.sft.loss import cross_entropy as training_cross_entropy

    analytic: list[float] = []
    numeric: list[float] = []
    for logits, target in zip(LOGIT_VECTORS, TARGET_INDICES):
        probabilities = softmax(logits)
        analytic.extend(
            probability - (1.0 if index == target else 0.0)
            for index, probability in enumerate(probabilities)
        )
        numeric.extend(
            gradient(
                lambda vector, position=target: training_cross_entropy(
                    list(vector), position
                ),
                logits,
                step=step,
            )
        )
    return _verdict(
        GRADIENT_CROSS_ENTROPY,
        tuple(analytic),
        tuple(numeric),
        tolerance=tolerance,
        message=f"{len(LOGIT_VECTORS)} 组 (打分, 正确下标) 上的梯度",
    )


def _production_sigmoid() -> ScalarFunction:
    """生产侧的 sigmoid（``alignment.objectives.sigmoid``），包成标量函数."""
    from smart_research_agent.alignment.objectives import sigmoid as production_sigmoid

    return lambda value: production_sigmoid(value)


def _production_log_sigmoid() -> ScalarFunction:
    """生产侧的 log-sigmoid（``alignment.objectives.log_sigmoid``）."""
    from smart_research_agent.alignment.objectives import log_sigmoid as production

    return lambda value: production(value)


def check_sigmoid_gradient(
    *, step: float = DEFAULT_STEP, tolerance: float = GRADIENT_TOLERANCE
) -> GradientOutcome:
    """``σ'(x) = σ(x)(1−σ(x))`` vs 对 ``alignment.objectives.sigmoid`` 的数值差分."""
    production = _production_sigmoid()
    analytic: list[float] = []
    numeric: list[float] = []
    for point in SCALAR_POINTS:
        value = production(point)
        analytic.append(value * (1.0 - value))
        numeric.append(
            gradient(lambda vector: production(vector[0]), (point,), step=step)[0]
        )
    return _verdict(
        GRADIENT_SIGMOID,
        tuple(analytic),
        tuple(numeric),
        tolerance=tolerance,
        message=f"{len(SCALAR_POINTS)} 个点上的一阶导数",
    )


def check_log_sigmoid_gradient(
    *, step: float = DEFAULT_STEP, tolerance: float = GRADIENT_TOLERANCE
) -> GradientOutcome:
    """``(log σ)'(x) = σ(−x)`` vs 对 ``alignment.objectives.log_sigmoid`` 的数值差分.

    这一项在 ``x = ±30`` 上仍然成立，而它检验的是一件具体的事：
    **log-sigmoid 的稳定实现不能把"稳定"做成"导数为 0"**。
    朴素写法 ``log(1/(1+e^{−x}))`` 在 ``x = −30`` 时会把 ``1+e^{30}`` 算成一个大数再取 log，
    结果 1e-9 量级——导数在数值差分下会变成一串噪声。
    """
    production = _production_log_sigmoid()
    analytic: list[float] = []
    numeric: list[float] = []
    for point in SCALAR_POINTS:
        analytic.append(_sigmoid(-point))
        numeric.append(
            gradient(lambda vector: production(vector[0]), (point,), step=step)[0]
        )
    return _verdict(
        GRADIENT_LOG_SIGMOID,
        tuple(analytic),
        tuple(numeric),
        tolerance=tolerance,
        message=f"{len(SCALAR_POINTS)} 个点上的一阶导数（含 ±30 的饱和区）",
    )


def check_perplexity_gradient(
    *, step: float = DEFAULT_STEP, tolerance: float = GRADIENT_TOLERANCE
) -> GradientOutcome:
    """``d(exp L)/dL = exp L`` vs 对 ``sft.loss.perplexity`` 的数值差分.

    这一项看着平凡（就是一个指数），它出现的原因是一处**真实的耦合**：
    困惑度与 loss 是两个单位不同的数字，训练日志里常同时出现；
    而"困惑度对 loss 的导数就是困惑度本身"意味着
    **loss 每涨 1 nat，困惑度就乘以 e**——这条直觉只在导数正确时才成立。
    """
    from smart_research_agent.sft.loss import perplexity as training_perplexity

    analytic = tuple(perplexity_from_entropy(value) for value in PERPLEXITY_INPUTS)
    numeric = tuple(
        gradient(lambda vector: training_perplexity(vector[0]), (value,), step=step)[0]
        for value in PERPLEXITY_INPUTS
    )
    return _verdict(
        GRADIENT_PERPLEXITY,
        analytic,
        numeric,
        tolerance=tolerance,
        message=f"{len(PERPLEXITY_INPUTS)} 个 loss 上的导数",
    )


def check_autograd_chain(
    *, step: float = DEFAULT_STEP, tolerance: float = GRADIENT_TOLERANCE
) -> GradientOutcome:
    """自动微分 vs 数值差分：验证"链式法则被自动跑对了".

    这是六项里唯一"与生产代码无关"的一项，也是这一课的核心：
    前面五项比的都是单个函数的导数，这一项比的是**复合**——
    ``AUTOGRAD_CASES`` 里每一串表达式都含 2~4 层复合，
    其中 ``x·x`` 与 ``x + x`` 那种"同一个输入走两条路径"的写法**故意留着**：
    它正是"梯度必须累加"的检验点（漏了累加的实现会在这里少算一半）。
    """
    analytic: list[float] = []
    numeric: list[float] = []
    for _name, expression, point in AUTOGRAD_CASES:
        _value, grads = value_and_grad(expression, point)
        analytic.extend(grads)
        numeric.extend(
            gradient(lambda vector: expression([Scalar(value) for value in vector]).value, point, step=step)
        )
    return _verdict(
        GRADIENT_AUTOGRAD_CHAIN,
        tuple(analytic),
        tuple(numeric),
        tolerance=tolerance,
        message=f"{len(AUTOGRAD_CASES)} 个复合表达式上的全部偏导数",
    )


def _sigmoid(value: float) -> float:
    """数值稳定的 sigmoid（与 ``autograd`` / ``bridge`` 同一实现口径）.

    在这里重写一遍而不是从 ``bridge`` 导入，是因为对照的**解析侧**
    一旦与别处共用实现，"两边一起错"就变成了可能：
    解析侧独立写一次，才有"两条独立路径互相验证"可言。
    """
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


#: 六项对照的调度表（键与 :data:`types.GRADIENT_TARGETS` 逐键对齐）.
GRADIENT_CHECKS: dict[str, Callable[..., GradientOutcome]] = {
    GRADIENT_SOFTMAX: check_softmax_jacobian,
    GRADIENT_CROSS_ENTROPY: check_cross_entropy_gradient,
    GRADIENT_SIGMOID: check_sigmoid_gradient,
    GRADIENT_LOG_SIGMOID: check_log_sigmoid_gradient,
    GRADIENT_PERPLEXITY: check_perplexity_gradient,
    GRADIENT_AUTOGRAD_CHAIN: check_autograd_chain,
}

if set(GRADIENT_CHECKS) != set(GRADIENT_TARGETS):  # pragma: no cover - 导入期不变式
    raise MathError(
        "梯度对照调度表与 GRADIENT_TARGETS 不一致：少一项的后果是那一项**静默地不跑**，"
        "而'没跑'与'跑过了、一致'在报告里长得一样。"
    )


def check_gradients_all(
    *,
    step: float = DEFAULT_STEP,
    tolerance: float = GRADIENT_TOLERANCE,
) -> GradientReport:
    """跑完六项梯度对照，装成一份 :class:`GradientReport`（**顺序 = GRADIENT_TARGETS**）.

    ``tolerance`` 与 ``step`` 都进函数参数而不是配置：它们只影响**这份报告**
    的判定口径与分辨率（"换一套判据"是一次显式决定）。
    """
    if tolerance <= 0:
        raise NumericError(f"容差必须为正，收到 {tolerance}。")
    outcomes = tuple(
        GRADIENT_CHECKS[target](step=step, tolerance=tolerance) for target in GRADIENT_TARGETS
    )
    return GradientReport(
        outcomes=outcomes,
        tolerance=tolerance,
        notes=(
            "对照的是'解析推导'与'数值差分'：前者快且精确但可能写错，"
            "后者慢且只到 1e-9 但**不依赖任何推导**——两者对上才算把链式法则说清楚了",
            "softmax 的雅可比被单独比一次，是因为它在交叉熵的梯度里会被抵消："
            "雅可比写错时，交叉熵的梯度**仍然是 p − onehot(y)**",
            "autograd_chain 里的表达式刻意包含 x·x 与 x + x："
            "'同一个值走两条路径'是梯度必须累加的地方，漏了累加会在这里少算一半",
            f"容差 {tolerance:.1e} 与判据的缩放口径都由分辨率决定："
            f"|f| = 100 时中心差分的分辨率下限是 "
            f"{difference_resolution(magnitude=100.0, step=step):.2e}——"
            "低于它的容差只会产出假警报",
        ),
    )


def sample_vectors() -> tuple[Vector, ...]:
    """演示与测试共用的打分样本（**同一份**，避免两处各写一份而悄悄分家）."""
    return LOGIT_VECTORS


__all__ = [
    "AUTOGRAD_CASES",
    "FLOAT_EPSILON",
    "GRADIENT_CHECKS",
    "GRADIENT_TOLERANCE",
    "LOGIT_VECTORS",
    "PERPLEXITY_INPUTS",
    "SCALAR_POINTS",
    "TARGET_INDICES",
    "GradientOutcome",
    "GradientReport",
    "check_autograd_chain",
    "check_cross_entropy_gradient",
    "check_gradients_all",
    "check_log_sigmoid_gradient",
    "check_perplexity_gradient",
    "check_sigmoid_gradient",
    "check_softmax_jacobian",
    "difference_resolution",
    "max_absolute_gap",
    "max_scaled_gap",
    "sample_vectors",
]
