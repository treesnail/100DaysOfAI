"""概率：从一个分布到一张联合分布表（day073 / Math-D1）.

这一课的第二半。它要回答的是**"LLM 输出那个 softmax 到底是什么"**：

```text
一个 token 的下一词分布      Distribution：一串和为 1 的概率 + 它们的名字
"模型有多犹豫"               entropy / perplexity
"换了模型之后分布差多少"      cross_entropy / kl_divergence
"给定证据，原因是什么"       JointTable + 贝叶斯后验
"这条答案的期望长度是多少"    expectation / variance（把 token 个数当成取值）
```

## 四条纪律

**1. 单位一律 nats。**
熵、交叉熵、KL 全部与 ``math.log`` 同底（自然对数）。这不是口味问题：
``sft.loss`` 的交叉熵是 nats、``perplexity = exp(loss)`` 也是 nats，
若这里改用 bits，同一个数字在两处报告里会差一个 ``ln 2`` 倍，
而它们各自的说明都"看起来对"。

**2. 归一化是公理，不是建议。**
概率之和必须为 1（容差内，见 ``types.check_distribution``）。
``softmax`` 的输出天然满足，但**手写的概率表常常不满足**——
一张和为 0.98 的表会在几十步采样之后把概率质量全推到最后一个事件上。

**3. 不可能的观测要报错，不许返回 inf 或 nan。**
``cross_entropy(p, q)`` 里若某个 ``p_i > 0`` 而 ``q_i == 0``，
数学上交叉熵是 ``+inf``、KL 是 ``+inf``。本包**当场拒绝**：

```text
返回 inf 的后果   下游的均值变成 inf、图表断线、"这次评估没有结论"
                   而根因（那个把概率写成 0 的实现）永远查不出来
```

**4. 随机性必须可注入、可复现。**
``uniforms(count, seed)`` 用线性同余生成器（LCG），**不是密码学安全的**，
但它是确定性的：同一个种子永远同一串数，因此"按这个分布采 1000 次"
这条结论可以被逐位复现。需要真随机时请传自己的 ``u`` 序列进来。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from smart_research_agent.math_foundations.errors import (
    NumericError,
    ParameterError,
    TableError,
)
from smart_research_agent.math_foundations.linalg import softmax
from smart_research_agent.math_foundations.types import (
    DEFAULT_SUM_TOLERANCE,
    Matrix,
    Vector,
    check_distribution,
    validate_matrix,
)

#: LCG 的乘子与增量（取自 Numerical Recipes 的那一组；**只要确定性**，不追求统计最优）.
LCG_MULTIPLIER = 1664525
LCG_INCREMENT = 1013904223
LCG_MODULUS = 2**32


# --------------------------------------------------------------------------- #
# 一、一个分布能回答什么
# --------------------------------------------------------------------------- #


def entropy(probabilities: Vector, *, tolerance: float = DEFAULT_SUM_TOLERANCE) -> float:
    """香农熵 ``H(p) = −Σ p_i ln p_i``（单位 **nats**）.

    两条边界约定必须写清：

    ```text
    p_i = 0      这一项的贡献是 0（lim p→0 p ln p = 0），**不是** 0·(−inf) = nan
    均匀分布     熵最大 = ln(n)：n 个等可能的事件之间"最没主意"
    确定性分布   熵最小 = 0：全押一个事件，没有任何不确定性
    ```

    "0 的贡献是 0"这一条是这一课最容易写错的地方：直接写
    ``-p * math.log(p)`` 会在 ``p = 0`` 时得到 ``nan``，
    而一个 ``nan`` 会顺着"平均熵"污染整张表。
    """
    checked = check_distribution(probabilities, tolerance=tolerance)
    total = 0.0
    for value in checked:
        if value > 0.0:
            total += value * math.log(value)
    return -total


def max_entropy(size: int) -> float:
    """``size`` 个等可能事件的最大熵 ``ln(size)``（"完全没主意"时的上界）."""
    if size < 1:
        raise ParameterError(f"事件个数必须 >= 1，收到 {size}。")
    return math.log(size)


def cross_entropy(
    truth: Vector,
    prediction: Vector,
    *,
    tolerance: float = DEFAULT_SUM_TOLERANCE,
) -> float:
    """交叉熵 ``H(p, q) = −Σ p_i ln q_i``（``p`` 是真值，``q`` 是预测）.

    三条性质（也是这一课为什么要它）：

    ```text
    非对称      H(p, q) ≠ H(q, p)：它衡量"用 q 编码 p 的代价"
    ≥ 熵        H(p, q) ≥ H(p)，等号只在 p == q 时成立（Gibbs 不等式）
    等于熵+KL   H(p, q) = H(p) + KL(p‖q)
    ```

    ``p_i > 0`` 而 ``q_i == 0`` 时报错（数学上是 +inf）：让 inf 流下去，
    下游的均值会变成 inf、图表会断线，而根因（那个把概率写成 0 的实现）查不出来。
    """
    checked_truth = check_distribution(truth, tolerance=tolerance)
    checked_prediction = check_distribution(prediction, tolerance=tolerance)
    if len(checked_truth) != len(checked_prediction):
        raise NumericError(
            f"真值与预测的长度不一致：{len(checked_truth)} != {len(checked_prediction)}。"
        )
    total = 0.0
    for index, (actual, guessed) in enumerate(zip(checked_truth, checked_prediction)):
        if actual == 0.0:
            continue
        if guessed <= 0.0:
            raise NumericError(
                f"第 {index} 个事件的真值概率是 {actual}，而预测概率是 {guessed}："
                "交叉熵在数学上是 +inf。本包不留这个 inf——"
                "inf 流到均值里会让整张表变成空，而根因是那个给出 0 概率的实现。"
            )
        total += actual * math.log(guessed)
    return -total


def kl_divergence(
    truth: Vector,
    prediction: Vector,
    *,
    tolerance: float = DEFAULT_SUM_TOLERANCE,
) -> float:
    """KL 散度 ``KL(p‖q) = Σ p_i ln(p_i / q_i)``（同样**非对称**，且 ≥ 0）.

    ``KL = 交叉熵 − 熵`` 是它的定义式之一，本函数直接按原式算——
    两处各算一遍的代价不是"多几行"，而是两条路径的浮点误差会在
    ``p == q`` 处给出两个不同的"应该是 0"的数。
    """
    return cross_entropy(truth, prediction, tolerance=tolerance) - entropy(
        truth, tolerance=tolerance
    )


def perplexity_from_entropy(value: float) -> float:
    """把熵（nats）换成困惑度 ``exp(H)``："平均在多少个等可能的候选之间犹豫"."""
    if not math.isfinite(value):
        raise NumericError(f"熵必须是有限实数，收到 {value!r}。")
    if value < 0:
        raise NumericError(f"熵不能为负，收到 {value}：负的熵说明上游算错了。")
    if value > 700:  # pragma: no cover - 防御式分支：exp(709) 已接近 float 上限
        return math.inf
    return math.exp(value)


def expectation(values: Vector, probabilities: Vector) -> float:
    """期望 ``E[X] = Σ p_i x_i``（``values`` 是每个事件的取值）."""
    checked_values, checked_probabilities = _paired(values, probabilities)
    return math.fsum(
        value * probability for value, probability in zip(checked_values, checked_probabilities)
    )


def variance(values: Vector, probabilities: Vector) -> float:
    """方差 ``Var[X] = E[X²] − (E[X])²``（按定义算，不用"平移公式"化简）.

    直接算 ``E[(X − μ)²]`` 而不是 ``E[X²] − μ²``：两个式子在数学上等价，
    但后者在 ``E[X²]`` 与 ``μ²`` 接近时会发生**灾难性抵消**
    （两个大数相减得到一个小数，有效位全丢）——而那正是"同一个分布算两次
    方差得到两个值"的常见原因。
    """
    checked_values, checked_probabilities = _paired(values, probabilities)
    mean = math.fsum(
        value * probability
        for value, probability in zip(checked_values, checked_probabilities)
    )
    return math.fsum(
        probability * (value - mean) ** 2
        for value, probability in zip(checked_values, checked_probabilities)
    )


def _paired(values: Vector, probabilities: Vector) -> tuple[Vector, Vector]:
    """把取值与概率配成对（长度必须一致，概率必须是一个合法分布）."""
    from smart_research_agent.math_foundations.types import validate_vector

    checked_values = validate_vector(values, name="取值")
    checked_probabilities = check_distribution(probabilities)
    if len(checked_values) != len(checked_probabilities):
        raise NumericError(
            f"取值的个数 {len(checked_values)} 与概率个数 "
            f"{len(checked_probabilities)} 不一致："
            "zip 会静默截断成较短的那一边，于是期望只统计了前几个事件。"
        )
    return checked_values, checked_probabilities


def sample_index(probabilities: Vector, *, u: float) -> int:
    """按累积概率采一个下标（``u ∈ [0, 1)`` 由调用方给，因此结果可复现）.

    做法是"把区间 [0,1) 按概率切成若干段，看 u 落在哪一段"：

    ```text
    p = [0.5, 0.3, 0.2]   →   段边界 0.5 | 0.8 | 1.0
    u = 0.42 → 0；u = 0.79 → 1；u = 0.99 → 2
    ```

    边界约定：``u`` 恰好等于某个边界时取**后**一段（等价于 ``p_i`` 覆盖
    ``[c_{i-1}, c_i)``）。这条约定必须写死，否则"u = 0.5 该采哪个"
    会变成两个实现对不上的地方。
    """
    checked = check_distribution(probabilities)
    if not math.isfinite(u) or not 0.0 <= u < 1.0:
        raise ParameterError(f"u 必须落在 [0, 1)，收到 {u!r}。")
    cumulative = 0.0
    for index, value in enumerate(checked):
        cumulative += value
        if u < cumulative:
            return index
    return len(checked) - 1


def uniforms(count: int, *, seed: int = 42) -> Vector:
    """一串确定性的 ``[0, 1)`` 均匀数（LCG，**不是密码学安全的**）.

    存在的理由是"让采样这件事可以被逐位复现"：同一个种子永远同一串数，
    于是"按这个分布采 10 万次得到什么直方图"是一条可以被复核的结论。
    真要随机数请传自己的 ``u`` 序列进 :func:`sample_index`。
    """
    if count < 0:
        raise ParameterError(f"count 不能为负，收到 {count}。")
    state = int(seed) % LCG_MODULUS
    values: list[float] = []
    for _ in range(count):
        state = (LCG_MULTIPLIER * state + LCG_INCREMENT) % LCG_MODULUS
        values.append(state / LCG_MODULUS)
    return tuple(values)


# --------------------------------------------------------------------------- #
# 二、一个分布（带名字）
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Distribution:
    """一个离散分布：概率 + 每个事件的名字（名字可选，用于报告可读）.

    ``labels`` 缺省时用 ``"0"`` / ``"1"`` … 补齐——**补名而不是补概率**：
    名字只是给人看的，概率才是那个必须被校验的东西
    （见 ``types.check_distribution`` 的四条判据）。
    """

    probabilities: Vector
    labels: tuple[str, ...] = ()
    name: str = "distribution"
    tolerance: float = DEFAULT_SUM_TOLERANCE

    def __post_init__(self) -> None:
        checked = check_distribution(
            tuple(float(value) for value in self.probabilities),
            labels=self.labels,
            tolerance=self.tolerance,
        )
        object.__setattr__(self, "probabilities", checked)
        if not self.labels:
            object.__setattr__(
                self, "labels", tuple(str(index) for index in range(len(checked)))
            )

    # ------------------------------------------------------------------ 构造

    @classmethod
    def uniform(cls, size: int, *, labels: tuple[str, ...] = (), name: str = "uniform"):
        """均匀分布（``size`` 个等可能事件）."""
        if size < 1:
            raise ParameterError(f"事件个数必须 >= 1，收到 {size}。")
        return cls(
            probabilities=tuple(1.0 / size for _ in range(size)),
            labels=labels,
            name=name,
        )

    @classmethod
    def from_logits(cls, logits: Vector, *, temperature: float = 1.0, name: str = "softmax"):
        """由打分（logits）经 softmax 得到分布（**这就是 LLM 下一词分布的做法**）."""
        return cls(
            probabilities=softmax(logits, temperature=temperature),
            labels=tuple(str(index) for index in range(len(logits))),
            name=name,
        )

    @classmethod
    def from_counts(
        cls,
        counts: Vector,
        *,
        labels: tuple[str, ...] = (),
        name: str = "counts",
    ):
        """由计数归一化成分布（计数必须非负、不能全零）."""
        from smart_research_agent.math_foundations.types import validate_vector

        checked = validate_vector(counts, name="counts")
        total = math.fsum(checked)
        if any(value < 0 for value in checked):
            raise NumericError("计数不能为负。")
        if total <= 0:
            raise NumericError(
                "计数之和为 0，无法归一化成分布："
                "'一个事件都没发生过'不是一个分布，它是'没有数据'——请在上游分开处理。"
            )
        return cls(
            probabilities=tuple(value / total for value in checked),
            labels=labels,
            name=name,
        )

    # ------------------------------------------------------------------ 查询

    @property
    def size(self) -> int:
        """事件个数."""
        return len(self.probabilities)

    def top_k(self, k: int) -> tuple[tuple[str, float], ...]:
        """按概率降序取前 ``k`` 个 ``(名字, 概率)``（并列时按名字升序）."""
        if k < 1 or k > self.size:
            raise ParameterError(f"k 必须落在 [1, {self.size}]，收到 {k}。")
        order = sorted(
            range(self.size),
            key=lambda index: (-self.probabilities[index], self.labels[index]),
        )
        return tuple((self.labels[index], self.probabilities[index]) for index in order[:k])

    def entropy(self) -> float:
        """香农熵（nats）."""
        return entropy(self.probabilities, tolerance=self.tolerance)

    def perplexity(self) -> float:
        """困惑度 ``exp(H)``（"平均在多少个等可能的候选之间犹豫"）."""
        return perplexity_from_entropy(self.entropy())

    def expectation(self, values: Vector) -> float:
        """按这个分布对 ``values`` 求期望."""
        return expectation(values, self.probabilities)

    def variance(self, values: Vector) -> float:
        """按这个分布对 ``values`` 求方差."""
        return variance(values, self.probabilities)

    def cross_entropy(self, other: Distribution) -> float:
        """``H(self, other)``：用 ``other`` 去编码 ``self`` 的代价."""
        return cross_entropy(self.probabilities, other.probabilities, tolerance=self.tolerance)

    def kl(self, other: Distribution) -> float:
        """``KL(self ‖ other)``（**非对称**：注意参数顺序）."""
        return kl_divergence(self.probabilities, other.probabilities, tolerance=self.tolerance)

    def sample(self, *, u: float) -> str:
        """按累积概率采一个事件名（``u`` 由调用方给，因此可复现）."""
        return self.labels[sample_index(self.probabilities, u=u)]

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（含三条派生量）."""
        return {
            "name": self.name,
            "size": self.size,
            "entropy": self.entropy(),
            "max_entropy": max_entropy(self.size),
            "perplexity": self.perplexity(),
            "top3": [list(item) for item in self.top_k(min(3, self.size))],
            "probabilities": list(self.probabilities),
            "labels": list(self.labels),
        }

    def summary_line(self) -> str:
        """一行说明：``softmax | 3 个事件 | H=0.7123 nats | 困惑度 2.0386``."""
        return (
            f"{self.name} | {self.size} 个事件 | H={self.entropy():.4f} nats | "
            f"困惑度 {self.perplexity():.4f}"
        )


# --------------------------------------------------------------------------- #
# 三、一张联合分布表
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class JointTable:
    """联合分布 ``P(A, B)`` 的表格形式（行 = A 的事件，列 = B 的事件）.

    它回答的问题与 `Distribution` 不同：不是"有多不确定"，
    而是**"两件事怎么一起发生"**——而"怎么一起发生"里包含了因果方向的信息
    （``P(A|B)`` 与 ``P(B|A)`` 一般不等）。
    """

    cells: Matrix
    names_a: tuple[str, ...]
    names_b: tuple[str, ...]
    name: str = "joint"
    tolerance: float = DEFAULT_SUM_TOLERANCE

    def __post_init__(self) -> None:
        checked = validate_matrix(
            tuple(tuple(float(value) for value in row) for row in self.cells), name="联合分布表"
        )
        rows, columns = len(checked), len(checked[0])
        if len(self.names_a) != rows:
            raise TableError(
                f"A 的事件名有 {len(self.names_a)} 个，表有 {rows} 行："
                "对不上的后果是'哪个概率属于哪个事件'说不清，"
                "而 zip 会静默截断成较短的那一边。"
            )
        if len(self.names_b) != columns:
            raise TableError(
                f"B 的事件名有 {len(self.names_b)} 个，表有 {columns} 列。"
            )
        if len(set(self.names_a)) != rows or len(set(self.names_b)) != columns:
            raise TableError(
                "事件名必须互不相同：重名会让'P(A=a|B=b) 是多少'有两个候选答案。"
            )
        for value in (item for row in checked for item in row):
            if value < 0:
                raise TableError(f"联合概率不能为负（收到 {value}）。")
        total = math.fsum(item for row in checked for item in row)
        if abs(total - 1.0) > self.tolerance:
            raise NumericError(
                f"联合分布的总和是 {total!r}，不是 1（容差 {self.tolerance}）："
                "合不拢的表算出来的条件概率与贝叶斯后验全是错的，"
                "而它们看起来仍然是一组和为 1 的数。"
            )
        object.__setattr__(self, "cells", checked)

    # ------------------------------------------------------------------ 边缘

    def marginal_a(self) -> Distribution:
        """A 的边缘分布 ``P(A) = Σ_b P(A, B=b)``（把每一行加起来）."""
        return Distribution(
            probabilities=tuple(math.fsum(row) for row in self.cells),
            labels=self.names_a,
            name=f"{self.name}.marginal_a",
            tolerance=self.tolerance,
        )

    def marginal_b(self) -> Distribution:
        """B 的边缘分布 ``P(B) = Σ_a P(A=a, B)``（把每一列加起来）."""
        columns = len(self.names_b)
        return Distribution(
            probabilities=tuple(
                math.fsum(self.cells[row][column] for row in range(len(self.names_a)))
                for column in range(columns)
            ),
            labels=self.names_b,
            name=f"{self.name}.marginal_b",
            tolerance=self.tolerance,
        )

    # ------------------------------------------------------------------ 条件

    def conditional_b_given_a(self, index: int) -> Distribution:
        """``P(B | A = index)``（行归一化；该行全零时报错）."""
        self._require_index(index, self.names_a, axis="A")
        row = self.cells[index]
        total = math.fsum(row)
        if total <= 0:
            raise TableError(
                f"P(A = {self.names_a[index]}) = 0，条件分布没有定义："
                "在一个概率为零的事件上做条件，等于在问'一个不会发生的事发生时会怎样'。"
            )
        return Distribution(
            probabilities=tuple(value / total for value in row),
            labels=self.names_b,
            name=f"{self.name}.P(B|A={self.names_a[index]})",
            tolerance=self.tolerance,
        )

    def conditional_a_given_b(self, index: int) -> Distribution:
        """``P(A | B = index)``（列归一化；该列全零时报错）."""
        self._require_index(index, self.names_b, axis="B")
        column = tuple(row[index] for row in self.cells)
        total = math.fsum(column)
        if total <= 0:
            raise TableError(f"P(B = {self.names_b[index]}) = 0，条件分布没有定义。")
        return Distribution(
            probabilities=tuple(value / total for value in column),
            labels=self.names_a,
            name=f"{self.name}.P(A|B={self.names_b[index]})",
            tolerance=self.tolerance,
        )

    def posterior_a_given_b(
        self, index: int, *, prior: Distribution | None = None
    ) -> Distribution:
        """贝叶斯后验 ``P(A | B = index)``.

        ```text
        P(A = a | B = b) = P(B = b | A = a) · P(A = a) / P(B = b)
                            └── 似然（表里读） ──┘ └ 先验 ┘   └ 证据 ┘
        ```

        ``prior`` 缺省用这张表的边缘 ``P(A)``——此时**算出来的后验必然等于
        表的列条件分布**（两者用的是同一份数据）。这不是巧合，而是一条
        可以断言的自洽性：若两者不等，说明似然或证据有一处取错了。
        传入自己的先验时，后验会随之改变——这正是"先验影响结论"的那件事。

        似然为 0 而先验不为 0 时报错，而不是给出"后验全是 0"：
        一个全零的后验看起来像"这个原因不可能"，而真相是"这份数据不支持它"。
        """
        self._require_index(index, self.names_b, axis="B")
        resolved_prior = prior if prior is not None else self.marginal_a()
        if resolved_prior.size != len(self.names_a):
            raise TableError(
                f"先验的事件个数 {resolved_prior.size} 与 A 的事件个数 "
                f"{len(self.names_a)} 不一致。"
            )
        likelihood = tuple(self.conditional_b_given_a(row).probabilities[index]
                           for row in range(len(self.names_a)))
        evidence = math.fsum(
            prior_value * like
            for prior_value, like in zip(resolved_prior.probabilities, likelihood)
        )
        if evidence <= 0:
            raise NumericError(
                f"证据 P(B = {self.names_b[index]}) 为 0：无法做贝叶斯更新——"
                "一个概率为零的观测不会让任何假设变得可能（它只会让整条式子变成 0/0）。"
            )
        posterior: list[float] = []
        for prior_value, like in zip(resolved_prior.probabilities, likelihood):
            if prior_value > 0 and like == 0:
                raise NumericError(
                    f"似然 P(B = {self.names_b[index]} | A) 为 0，而先验不为 0："
                    "严格按公式会得到'这个原因彻底不可能'，"
                    "但那只是这一份数据不支持它——请改用平滑后的似然，而不是接受一个 0。"
                )
            posterior.append(prior_value * like / evidence)
        return Distribution(
            probabilities=tuple(posterior),
            labels=self.names_a,
            name=f"{self.name}.posterior(A|B={self.names_b[index]})",
            tolerance=self.tolerance,
        )

    # ------------------------------------------------------------------ 相关性

    def independence_gap(self) -> float:
        """与"独立"的最大偏差 ``max |P(a,b) − P(a)P(b)|``（0 表示独立）."""
        marginal_a = self.marginal_a().probabilities
        marginal_b = self.marginal_b().probabilities
        worst = 0.0
        for row, probability_a in enumerate(marginal_a):
            for column, probability_b in enumerate(marginal_b):
                gap = abs(self.cells[row][column] - probability_a * probability_b)
                worst = max(worst, gap)
        return worst

    def is_independent(self, *, tolerance: float = 1e-9) -> bool:
        """所有格子的偏差都在容差内（判据只有一处实现：它读 :meth:`independence_gap`）."""
        if tolerance <= 0:
            raise ParameterError(f"容差必须为正，收到 {tolerance}。")
        return self.independence_gap() <= tolerance

    def mutual_information(self) -> float:
        """互信息 ``I(A;B) = H(A) + H(B) − H(A,B)``（独立时为 0，单位 nats）.

        它比"最大格子偏差"更有信息量：偏差是逐格的，而互信息是**整体**的。
        一个"每格都差一点"的表在 :meth:`independence_gap` 上可能看着不大，
        但它的互信息会明显大于 0。
        """
        joint = entropy(
            tuple(value for row in self.cells for value in row), tolerance=self.tolerance
        )
        return (
            self.marginal_a().entropy()
            + self.marginal_b().entropy()
            - joint
        )

    def to_dict(self) -> dict[str, Any]:
        """可 json.dumps 的形状（含边缘、互信息与最大偏差）."""
        return {
            "name": self.name,
            "shape": [len(self.names_a), len(self.names_b)],
            "names_a": list(self.names_a),
            "names_b": list(self.names_b),
            "cells": [list(row) for row in self.cells],
            "marginal_a": list(self.marginal_a().probabilities),
            "marginal_b": list(self.marginal_b().probabilities),
            "mutual_information": self.mutual_information(),
            "independence_gap": self.independence_gap(),
        }

    def summary_line(self) -> str:
        """一行说明：``joint | 2×2 | I=0.1234 nats | 最大偏差 0.0600``."""
        return (
            f"{self.name} | {len(self.names_a)}×{len(self.names_b)} | "
            f"I={self.mutual_information():.4f} nats | "
            f"最大偏差 {self.independence_gap():.4f}"
        )

    # ------------------------------------------------------------------ 内部

    def _require_index(self, index: int, names: tuple[str, ...], *, axis: str) -> None:
        if not 0 <= index < len(names):
            raise ParameterError(
                f"{axis} 的下标必须落在 [0, {len(names)})，收到 {index}。"
            )


__all__ = [
    "LCG_INCREMENT",
    "LCG_MODULUS",
    "LCG_MULTIPLIER",
    "Distribution",
    "JointTable",
    "cross_entropy",
    "entropy",
    "expectation",
    "kl_divergence",
    "max_entropy",
    "perplexity_from_entropy",
    "sample_index",
    "uniforms",
    "variance",
]
