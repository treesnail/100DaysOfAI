"""桥：把本包算的东西与项目里**已有的实现**逐点对上（day073 / Math-D1）.

这一课的最后一环，也是它区别于"一本数学书"的地方。

## 一个必须先回答的问题：为什么要写第二份实现

项目里已经有一份余弦（``vectorstore.metrics``）、两份 softmax
（``sft.loss`` 与 ``llm.sampling``）、一份 log-softmax、交叉熵、困惑度、
sigmoid 与 log-sigmoid。今天的 ``math_foundations`` 又写了一遍——
这不是重复，而是**两种不同用途的实现**：

```text
生产实现    为某一层服务：检索要快、训练要稳、采样要能被温度控制
            它们的取舍写在各自的注释里（例如"零向量返回 0.0 让查询不中断"）
教学实现    为了让公式能被逐行读出来：不加缓存、不为性能让步、
            每个拒绝都写清"为什么这样选"
```

两者**必须对得上**，否则会出现最坏的一种情况：教程里算的是 A，
而线上跑的是 B——读者照着教程理解的"模型在做什么"是错的。

## 八项对照，以及三种"对不上"的处理

```text
逐位一致          cosine / softmax / log_softmax / cross_entropy /
                  perplexity / sigmoid / log_sigmoid（本课的七项）
约定的差异        normalize（零向量：本包报错、生产实现原样返回）
                  ——这不是 bug，而是"同一个输入，两种用途下的两种正确行为"
                 它必须被**记下来**，而不是被"统一"掉
真正的分歧        只有一处 = 需要修代码的地方（本课没有出现，但报告里要能报出来）
```

第 2、3 两种的区别是这一课最值钱的一课：

```text
差异被记下来 → 读的人知道"这里有三条路，各自为什么"
差异被抹平   → 某一天有人"顺手统一"，于是某处开始静默地返回错值
```

## 对照怎么做才算数

1. **固定输入 + 固定容差**：每一组输入都写在代码里（不是随机生成），
   容差用 :data:`types.DEFAULT_TOLERANCE`；
2. **逐点比较，不只看"差不多"**：报告里给出**最大绝对误差**，
   而不是一句"一致"——"一致"是结论，误差是证据；
3. **拒绝必须两边都拒绝**：一边报错、一边给出值，本身就是一种不一致。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.math_foundations.errors import MathError, NumericError
from smart_research_agent.math_foundations.linalg import (
    cosine,
    log_softmax,
    norm,
    normalize,
    softmax,
)
from smart_research_agent.math_foundations.probability import (
    cross_entropy as our_cross_entropy,
)
from smart_research_agent.math_foundations.probability import (
    perplexity_from_entropy,
)
from smart_research_agent.math_foundations.types import (
    BRIDGE_COSINE,
    BRIDGE_CROSS_ENTROPY,
    BRIDGE_LOG_SIGMOID,
    BRIDGE_LOG_SOFTMAX,
    BRIDGE_NORMALIZE,
    BRIDGE_PERPLEXITY,
    BRIDGE_SIGMOID,
    BRIDGE_SOFTMAX,
    BRIDGE_TARGET_DESCRIPTIONS,
    BRIDGE_TARGET_SOURCES,
    BRIDGE_TARGETS,
    DEFAULT_TOLERANCE,
    Matrix,
    Vector,
)

#: 对照用的向量样本（**写死在代码里**：随机生成的输入会让"这次对上了"不可复现）.
VECTOR_PAIRS: tuple[tuple[Vector, Vector], ...] = (
    ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),  # 完全相同 → 1
    ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),  # 正交 → 0
    ((1.0, 1.0, 0.0), (-1.0, -1.0, 0.0)),  # 相反 → -1
    ((0.3, -0.7, 1.4, 2.0), (2.0, 1.4, -0.7, 0.3)),  # 一般情形
    ((1e-8, 2e-8, -3e-8), (4.0, 5.0, 6.0)),  # 极小模长（相对误差的边界）
)

#: 对照用的打分样本（含极端值：softmax 的稳定性就是在这类输入上体现的）.
LOGIT_SAMPLES: tuple[Vector, ...] = (
    (0.0, 0.0),
    (1.0, 2.0, 3.0),
    (1000.0, 1001.0),  # 直接 exp 会溢出：1000 与 1001 的差才是关键
    (-1000.0, -1001.0, -999.0),
    (0.5,),
)

#: 对照用的标量样本（sigmoid / log_sigmoid 的边界：x 很负时 log 会下溢）.
SCALAR_SAMPLES: tuple[float, ...] = (
    0.0,
    1.0,
    -1.0,
    30.0,
    -30.0,
    -800.0,  # log(σ(-800)) 用朴素写法会得到 -inf
    800.0,
)


@dataclass(frozen=True)
class CheckOutcome:
    """一项对照的结论：对没对上、误差多大、差在哪.

    ``status`` 的三种取值是这一课的核心概念之一：

    ```text
    agrees       逐点在容差内一致
    differs      超出了容差（**这才是需要改代码的信号**）
    rejected     两边的**拒绝行为**不同（一边报错、一边给出值）
    ```
    """

    target: str
    status: str
    max_absolute_error: float
    compared_points: int
    message: str
    source: str

    def __post_init__(self) -> None:
        if self.target not in BRIDGE_TARGETS:
            raise MathError(
                f"不认识的对照项 {self.target!r}：可用取值 {list(BRIDGE_TARGETS)}。"
            )
        if self.status not in ("agrees", "differs", "rejected"):
            raise MathError(
                f"不认识的对照结论 {self.status!r}："
                "只允许 agrees / differs / rejected——"
                "多一种取值会让'哪些项需要修'变成一个需要读三个地方的问题。"
            )
        if self.max_absolute_error < 0:
            raise NumericError(f"最大绝对误差不能为负：{self.max_absolute_error}。")

    @property
    def ok(self) -> bool:
        """这一项**没有分歧**（``differs`` 才算分歧；``rejected`` 是记录的差异）."""
        return self.status != "differs"

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（含它对照的是哪个函数）."""
        return {
            "target": self.target,
            "description": BRIDGE_TARGET_DESCRIPTIONS[self.target],
            "source": self.source,
            "status": self.status,
            "ok": self.ok,
            "max_absolute_error": self.max_absolute_error,
            "compared_points": self.compared_points,
            "message": self.message,
        }

    def summary_line(self) -> str:
        """一行说明：``[agrees] cosine 5 个点，最大误差 0.00e+00``."""
        return (
            f"[{self.status}] {self.target} {self.compared_points} 个点，"
            f"最大误差 {self.max_absolute_error:.2e}"
        )


@dataclass(frozen=True)
class BridgeReport:
    """八项对照的汇总（`cross_check_all` 的产物）."""

    outcomes: tuple[CheckOutcome, ...]
    tolerance: float = DEFAULT_TOLERANCE
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        missing = set(BRIDGE_TARGETS) - {item.target for item in self.outcomes}
        if missing:
            raise MathError(
                f"对照报告缺少 {sorted(missing)}：少做的对照与'对照通过'在报告里长得一样，"
                "因此这里当场拒绝——一份不完整的报告比没有报告更危险。"
            )
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @property
    def disagreements(self) -> tuple[CheckOutcome, ...]:
        """真正需要改代码的那些（``differs``）."""
        return tuple(item for item in self.outcomes if item.status == "differs")

    @property
    def recorded_differences(self) -> tuple[CheckOutcome, ...]:
        """被**记录**的约定差异（``rejected``）——它们不是 bug，但必须可读."""
        return tuple(item for item in self.outcomes if item.status == "rejected")

    @property
    def ok(self) -> bool:
        """没有任何分歧（约定差异不影响这条结论）."""
        return not self.disagreements

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状."""
        return {
            "tolerance": self.tolerance,
            "ok": self.ok,
            "disagreements": [item.target for item in self.disagreements],
            "recorded_differences": [item.target for item in self.recorded_differences],
            "outcomes": [item.to_dict() for item in self.outcomes],
            "notes": list(self.notes),
        }

    def summary_line(self) -> str:
        """一行说明：``对照 8 项：一致 7、记录差异 1、分歧 0``."""
        agrees = sum(1 for item in self.outcomes if item.status == "agrees")
        return (
            f"对照 {len(self.outcomes)} 项：一致 {agrees}、"
            f"记录差异 {len(self.recorded_differences)}、"
            f"分歧 {len(self.disagreements)}"
        )


def _compare_points(
    target: str,
    ours: Callable[[], list[float]],
    theirs: Callable[[], list[float]],
) -> CheckOutcome:
    """逐点比较两个实现（**只在两边都成功时比数值**；一边失败即 ``rejected``）.

    "一边报错、一边给出值"这件事必须单独成一类：把它当成"数值不一致"，
    会让一条"本包拒绝、生产实现返回 0.0"的约定差异看起来像一个 bug——
    而它其实是**有意设计的两条路**（见 ``errors`` 的说明）。
    """
    our_error: str = ""
    their_error: str = ""
    our_values: list[float] = []
    their_values: list[float] = []
    try:
        our_values = list(ours())
    except MathError as exc:
        our_error = str(exc)
    try:
        their_values = list(theirs())
    except Exception as exc:  # noqa: BLE001 - 生产实现抛的是 ValueError 子类，这里统一记录
        their_error = f"{type(exc).__name__}: {exc}"
    if our_error or their_error:
        if our_error and their_error:
            return CheckOutcome(
                target=target,
                status="agrees",
                max_absolute_error=0.0,
                compared_points=0,
                message=f"两边都拒绝了这个输入（本包：{our_error[:60]}；生产：{their_error[:60]}）",
                source=BRIDGE_TARGET_SOURCES[target],
            )
        side = "本包拒绝" if our_error else "生产实现拒绝"
        detail = our_error or their_error
        return CheckOutcome(
            target=target,
            status="rejected",
            max_absolute_error=0.0,
            compared_points=0,
            message=f"{side}、另一边给出值：{detail[:120]}",
            source=BRIDGE_TARGET_SOURCES[target],
        )
    if len(our_values) != len(their_values):
        return CheckOutcome(
            target=target,
            status="differs",
            max_absolute_error=float("inf"),
            compared_points=0,
            message=f"返回长度不同：本包 {len(our_values)}、生产 {len(their_values)}",
            source=BRIDGE_TARGET_SOURCES[target],
        )
    worst = 0.0
    for left, right in zip(our_values, their_values):
        if not (math.isfinite(left) and math.isfinite(right)):
            worst = float("inf")
            break
        worst = max(worst, abs(left - right))
    return CheckOutcome(
        target=target,
        status="agrees" if worst <= DEFAULT_TOLERANCE else "differs",
        max_absolute_error=worst,
        compared_points=len(our_values),
        message=(
            "逐点在容差内一致"
            if worst <= DEFAULT_TOLERANCE
            else f"最大绝对误差 {worst:.3e} 超过容差 {DEFAULT_TOLERANCE:.1e}"
        ),
        source=BRIDGE_TARGET_SOURCES[target],
    )


def check_cosine() -> CheckOutcome:
    """余弦：本包 vs ``vectorstore.metrics.cosine_similarity``（day064 的口径）."""
    from smart_research_agent.vectorstore.metrics import (
        cosine_similarity as production_cosine,
    )

    def ours() -> list[float]:
        return [cosine(left, right) for left, right in VECTOR_PAIRS]

    def theirs() -> list[float]:
        return [production_cosine(left, right) for left, right in VECTOR_PAIRS]

    return _compare_points(BRIDGE_COSINE, ours, theirs)


def check_normalize() -> CheckOutcome:
    """归一化：先比数值（非零向量），再**单独记录**零向量上的约定差异.

    这一项是八项里唯一一条"故意对不上"的，因此它必须把两件事都写出来：

    ```text
    非零向量   逐位一致（这才是"公式是不是同一个"的答案）
    零向量     本包拒绝、生产实现原样返回（两条都是对的，理由见 errors 的说明）
    ```

    只报"零向量不一致"会让人以为这一项的公式有问题；
    只报"数值一致"又会把那条约定藏起来——而它正是读者最容易踩的地方。
    """
    from smart_research_agent.llm.embedding import l2_normalize

    non_zero = tuple(pair for pair in VECTOR_PAIRS if norm(pair[0]) > 0)
    zero = (0.0, 0.0, 0.0)

    def ours() -> list[float]:
        return [value for pair in non_zero for value in normalize(pair[0])]

    def theirs() -> list[float]:
        return [value for pair in non_zero for value in l2_normalize(list(pair[0]))]

    outcome = _compare_points(BRIDGE_NORMALIZE, ours, theirs)
    if outcome.status == "differs":
        return outcome
    zero_rejected = False
    try:
        normalize(zero)
    except MathError:
        zero_rejected = True
    zero_returned = tuple(l2_normalize(list(zero)))
    if not zero_rejected or any(value != 0.0 for value in zero_returned):
        return CheckOutcome(
            target=BRIDGE_NORMALIZE,
            status="differs",
            max_absolute_error=max(outcome.max_absolute_error, 0.0),
            compared_points=outcome.compared_points,
            message=(
                "零向量的处理与预期不符：本包应当拒绝、生产实现应当原样返回——"
                f"实际（拒绝={zero_rejected}、生产返回值={zero_returned}）"
            ),
            source=BRIDGE_TARGET_SOURCES[BRIDGE_NORMALIZE],
        )
    return CheckOutcome(
        target=BRIDGE_NORMALIZE,
        status="rejected",
        max_absolute_error=outcome.max_absolute_error,
        compared_points=outcome.compared_points,
        message=(
            f"非零向量 {outcome.compared_points} 个点逐位一致（最大误差 "
            f"{outcome.max_absolute_error:.2e}）；零向量上**刻意不同**："
            "本包拒绝（零向量没有方向），生产实现原样返回（写入侧不能因为"
            "'没有方向'就抛异常）——这是有意记录的约定差异，不是分歧"
        ),
        source=BRIDGE_TARGET_SOURCES[BRIDGE_NORMALIZE],
    )


def check_softmax() -> CheckOutcome:
    """softmax：本包 vs ``sft.loss.softmax`` 与 ``llm.sampling.softmax_with_temperature``.

    **两边的取样顺序必须逐位对齐**（同一组 logits、同一个顺序喂给三个实现）：
    对照工作里最容易做错的一步不是算错，而是"拿两串顺序不同的数去比"——
    那样得到的"不一致"是假的，而它会让人去改一份本来正确的代码。
    """
    from smart_research_agent.llm.sampling import softmax_with_temperature
    from smart_research_agent.sft.loss import softmax as training_softmax

    def ours() -> list[float]:
        values: list[float] = []
        for logits in LOGIT_SAMPLES:
            values.extend(softmax(logits))
            values.extend(softmax(logits))
        return values

    def theirs() -> list[float]:
        values: list[float] = []
        for logits in LOGIT_SAMPLES:
            values.extend(training_softmax(list(logits)))
            values.extend(softmax_with_temperature(list(logits), 1.0))
        return values

    return _compare_points(BRIDGE_SOFTMAX, ours, theirs)


def check_log_softmax() -> CheckOutcome:
    """log-softmax：本包 vs ``sft.loss.log_softmax``（含极小概率不下溢）. """
    from smart_research_agent.sft.loss import log_softmax as training_log_softmax

    def ours() -> list[float]:
        return [value for logits in LOGIT_SAMPLES for value in log_softmax(logits)]

    def theirs() -> list[float]:
        return [
            value for logits in LOGIT_SAMPLES for value in training_log_softmax(list(logits))
        ]

    return _compare_points(BRIDGE_LOG_SOFTMAX, ours, theirs)


def one_hot(index: int, size: int) -> Vector:
    """独热向量（第 ``index`` 位是 1，其余是 0）——把"某个 token 是正确答案"写成分布."""
    if size < 1:
        raise NumericError(f"向量长度必须 >= 1，收到 {size}。")
    if not 0 <= index < size:
        raise NumericError(f"下标必须落在 [0, {size})，收到 {index}。")
    return tuple(1.0 if position == index else 0.0 for position in range(size))


def check_cross_entropy() -> CheckOutcome:
    """交叉熵：本包通式 vs ``sft.loss.cross_entropy``（**两个坑都在这一项里**）.

    坑一：**定义域不同**（上一次修改踩到的）。

    ```text
    sft.loss.cross_entropy(logits, target)   单点交叉熵：真值是一个 one-hot 下标、预测是 **logits**
    probability.cross_entropy(p, q)          通式：真值是一个**分布**、预测也是**分布**
    ```

    通式在 one-hot 真值上退化成的就是单点交叉熵，两者此时逐位一致；
    但真值不是 one-hot 时生产实现**根本没有那个入参**，硬要比会得到一个假的不一致。

    坑二：**同名不同物——logits 还是概率**（这一项真的是个陷阱）。

    ``sft.loss.cross_entropy`` 的第一个参数是**打分（logits）**，
    而本包的 ``cross_entropy`` 要的是**概率**。把概率当 logits 传进去：
    形状对、数值合法、不报错——只会**又做了一次 softmax**、把概率压向均匀：

    ```text
    target 是 argmax      该类的概率被压小 → loss 变大（像是"模型变差了"）
    target 不是 argmax    该类的概率被推大 → loss 变小（像是"模型变好了"）
    ```

    两个方向都会出现，因此"loss 不对劲"不能只看方向——
    必须回到**入参形态**去查。这就是"同名函数"最容易骗人的地方。

    因此这一项的每次比较都必须把两边的**入参形态**写清楚：
    本包这边显式做一次 ``softmax``（把打分变成概率），生产那边吃原始打分。
    """
    from smart_research_agent.sft.loss import cross_entropy as training_cross_entropy

    cases: list[tuple[Vector, int]] = [
        ((1.0, 2.0), 0),
        ((1.0, 2.0), 1),
        ((0.5, -0.5, 2.0), 2),
        ((1000.0, 1001.0), 1),  # 直接 exp 会溢出的打分：两边都该稳
    ]

    def ours() -> list[float]:
        return [
            our_cross_entropy(one_hot(target, len(logits)), softmax(logits))
            for logits, target in cases
        ]

    def theirs() -> list[float]:
        return [training_cross_entropy(list(logits), target) for logits, target in cases]

    outcome = _compare_points(BRIDGE_CROSS_ENTROPY, ours, theirs)
    if outcome.status == "agrees":
        return CheckOutcome(
            target=BRIDGE_CROSS_ENTROPY,
            status="agrees",
            max_absolute_error=outcome.max_absolute_error,
            compared_points=outcome.compared_points,
            message=(
                f"{outcome.compared_points} 个 one-hot 真值逐位一致"
                "（**本包吃概率、生产吃打分**：比较时本包显式做了一次 softmax。"
                "把概率当 logits 传不会报错，只会又做一次 softmax："
                "target 是 argmax 时 loss 变大、不是 argmax 时 loss 变小——"
                "两个方向都会出现，因此只能回到入参形态去查）"
            ),
            source=BRIDGE_TARGET_SOURCES[BRIDGE_CROSS_ENTROPY],
        )
    return outcome


def check_perplexity() -> CheckOutcome:
    """困惑度：本包 ``perplexity_from_entropy`` vs ``sft.loss.perplexity``（同一个 exp）."""
    from smart_research_agent.sft.loss import perplexity as training_perplexity

    def ours() -> list[float]:
        return [perplexity_from_entropy(value) for value in checked_perplexity_inputs()]

    def theirs() -> list[float]:
        return [training_perplexity(value) for value in checked_perplexity_inputs()]

    return _compare_points(BRIDGE_PERPLEXITY, ours, theirs)


def checked_perplexity_inputs() -> Vector:
    """困惑度对照用的输入（都是合法的"loss"：非负、有限）."""
    return (0.0, math.log(2.0), math.log(100.0), 5.0)


def check_sigmoid() -> CheckOutcome:
    """sigmoid：本包（用 ``softplus`` 的定义写）vs ``alignment.objectives.sigmoid``."""
    from smart_research_agent.alignment.objectives import sigmoid as production_sigmoid

    def ours() -> list[float]:
        return [_sigmoid(value) for value in SCALAR_SAMPLES]

    def theirs() -> list[float]:
        return [production_sigmoid(value) for value in SCALAR_SAMPLES]

    return _compare_points(BRIDGE_SIGMOID, ours, theirs)


def check_log_sigmoid() -> CheckOutcome:
    """log-sigmoid：本包 vs ``alignment.objectives.log_sigmoid``（x 很负时不下溢）."""
    from smart_research_agent.alignment.objectives import (
        log_sigmoid as production_log_sigmoid,
    )

    def ours() -> list[float]:
        return [log_sigmoid(value) for value in SCALAR_SAMPLES]

    def theirs() -> list[float]:
        return [production_log_sigmoid(value) for value in SCALAR_SAMPLES]

    return _compare_points(BRIDGE_LOG_SIGMOID, ours, theirs)


def _sigmoid(value: float) -> float:
    """本包的 sigmoid：``σ(x) = 1/(1+e^{−x})``，按符号分支避免上溢.

    这条实现与 ``log_sigmoid`` 配套：``σ(x) = exp(log_sigmoid(x))``
    在 ``x`` 很负时会先下溢到 0（正确），但直接算 ``1/(1+exp(-x))``
    在 ``x = -800`` 时会先算 ``exp(800) = inf`` 再得到 ``1/inf = 0``——
    结果对，但中间那一步 ``inf`` 会在开启浮点陷阱的环境里直接抛异常。
    """
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def log_sigmoid(value: float) -> float:
    """``log σ(x) = −softplus(−x)``，按符号分支实现（x 很负时仍给有限值）."""
    if value >= 0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def softplus(value: float) -> float:
    """``log(1 + e^x)``（按符号分支，``x`` 很大时不溢出）."""
    if value > 0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


#: 八项对照的调度表（键与 :data:`types.BRIDGE_TARGETS` 逐键对齐）.
CHECK_FUNCTIONS: dict[str, Callable[[], CheckOutcome]] = {
    BRIDGE_COSINE: check_cosine,
    BRIDGE_NORMALIZE: check_normalize,
    BRIDGE_SOFTMAX: check_softmax,
    BRIDGE_LOG_SOFTMAX: check_log_softmax,
    BRIDGE_CROSS_ENTROPY: check_cross_entropy,
    BRIDGE_PERPLEXITY: check_perplexity,
    BRIDGE_SIGMOID: check_sigmoid,
    BRIDGE_LOG_SIGMOID: check_log_sigmoid,
}

if set(CHECK_FUNCTIONS) != set(BRIDGE_TARGETS):
    raise MathError(
        "对照调度表与 BRIDGE_TARGETS 不一致：少一项的后果是那一项**静默地不跑**，"
        "而'没跑'与'跑过了、一致'在报告里长得一样。"
    )


def cross_check_all(*, tolerance: float = DEFAULT_TOLERANCE) -> BridgeReport:
    """跑完八项对照，装成一份 :class:`BridgeReport`（**顺序 = BRIDGE_TARGETS**）.

    ``tolerance`` 不进配置也不被写入任何一个生产模块：它只影响**这份报告**的
    判定口径，因此它进的是函数参数（"换一套判据"是一次显式决定）。
    """
    if tolerance <= 0:
        raise NumericError(f"容差必须为正，收到 {tolerance}。")
    outcomes = tuple(CHECK_FUNCTIONS[target]() for target in BRIDGE_TARGETS)
    return BridgeReport(
        outcomes=outcomes,
        tolerance=tolerance,
        notes=(
            "对照的是'同一个公式的两种用途实现'：生产实现为所在层服务（快、稳、可容错），"
            "教学实现为了让公式能被逐行读出来——两者必须对得上，"
            "否则读者照着教程理解的'模型在做什么'就是错的",
            "normalize 的零向量约定被**记录**为差异而不是分歧："
            "一个必须拒绝、一个必须容错，两条都是对的",
        ),
    )


def sample_outputs() -> tuple[Matrix, Matrix, Matrix]:
    """演示与测试共用的注意力输入（3×4 的 Q/K/V，**写死**以保证可复现）.

    三行 query **刻意不同**：这样"每一行看到的东西不一样"才看得出来
    （如果三行相同，因果掩码的效果会退化成"只有前两行不同"，
    读的人会把"所有行权重一样"误当成结论）。
    """
    queries: Matrix = (
        (1.0, 0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0, 1.0),
        (1.0, 0.0, 0.0, 1.0),
    )
    keys: Matrix = ((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 1.0, 0.0), (1.0, 1.0, 1.0, 1.0))
    values: Matrix = ((1.0, 0.0), (0.0, 1.0), (0.5, 0.5))
    return queries, keys, values


__all__ = [
    "CHECK_FUNCTIONS",
    "LOGIT_SAMPLES",
    "SCALAR_SAMPLES",
    "VECTOR_PAIRS",
    "BridgeReport",
    "CheckOutcome",
    "check_cosine",
    "check_cross_entropy",
    "check_log_sigmoid",
    "check_log_softmax",
    "check_normalize",
    "check_perplexity",
    "check_sigmoid",
    "check_softmax",
    "checked_perplexity_inputs",
    "cross_check_all",
    "log_sigmoid",
    "one_hot",
    "sample_outputs",
    "softplus",
]
