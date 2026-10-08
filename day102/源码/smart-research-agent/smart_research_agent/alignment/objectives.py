"""两个对齐目标的数学：RLHF（PPO）与 DPO（M5-D6）.

对齐（alignment）要解决的是 SFT 解决不了的问题：**同一个提问有很多"都说得通"
的回答**，SFT 只能教模型模仿训练集里的那一个，而偏好数据能告诉模型
"在这些说得通的答法里，哪一种更被偏好"。

两种主流做法，差别在于"**偏好是怎么进到策略里的**"：

.. code-block:: text

    RLHF（PPO）：偏好 → 训一个奖励模型 r(x,y) → 用强化学习最大化
                 E[r] − β·KL(π_θ ‖ π_ref)
    DPO       ：偏好 → 直接在策略上做监督式优化
                 −log σ( β·[(logπ_θ(y_w) − logπ_θ(y_l)) − (logπ_ref(y_w) − logπ_ref(y_l))] )

本模块把两者都写成**可手算的标量函数**，因为这一课的核心不是"跑起来"，
而是三件事能被算出来：

1. **未训练时 DPO 的 loss 恰好是 ln 2**（`0.6931`）：此时策略与参考模型
   是同一个模型，括号里的差为 0，`−log σ(0) = ln 2`。这条"起点检查"极其有用——
   **如果你的 DPO loss 一开始不是 ln 2，那么你的 ref 或 β 接错了**；
2. **DPO 的隐式奖励**：`r(x,y) = β·log(π_θ(y|x) / π_ref(y|x))`。它说明 DPO
   并没有"不要奖励模型"，而是**把奖励做成了策略与参考模型的对数比**——
   于是不需要单独训一个奖励模型，也不需要 rollout；
3. **RLHF 的目标里有 KL 惩罚**，而 DPO 把它藏进了"与参考模型的对数比"里。
   两者是同一个约束的两种写法（见 ``objective_requirements`` 的对照表）。

数值稳定性：``sigmoid`` / ``log_sigmoid`` 都按符号分支实现，**不做截断**——
截断会把"margin 很大"这种正常情况变成一个假的常数，而真正的解法是
在数学上避免 ``exp`` 溢出（``-log σ(x) = softplus(-x)``）。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from smart_research_agent.finetune_eval.metrics import FinetuneEvalError

#: margin 为 0 时的 DPO loss：``−log σ(0) = ln 2``.
#:
#: 它是一条**起点检查**：未训练时策略就是参考模型，括号里的差为 0，
#: 于是 loss 必然等于这个值。实测见 ``scripts/alignment_demo.py`` 的第 1 步。
ZERO_MARGIN_LOSS = math.log(2.0)

#: KL 估计器的名字（三者的偏差/方差特性不同，见 ``kl_estimators``）.
KL_ESTIMATORS: tuple[str, ...] = ("k1", "k2", "k3")

#: ``beta`` 的常用量级说明（进文档与告警）.
#:
#: β 同时出现在两个地方：它是 DPO loss 里的"温度"，也是梯度系数
#: ``−β·σ(−βm)`` 的一部分。**"β 越大策略偏离参考模型越少"这句常见说法在本课
#: 的实测里并不成立**：固定学习率与步数时，KL 随 β 先升后降（``0.05→0.0125``、
#: ``0.1→0.0449``、``0.5→0.4460``、``1.0→0.5940``、``2.0→0.4987``）——
#: β 大则早期步长大，但 margin 一大梯度就迅速衰减，反而早早刹住。
#: 所以 β 只能实测标定（见 ``docs/alignment.md`` 的 KL 表）。
BETA_SOFT_RANGE = (0.01, 5.0)


def sigmoid(value: float) -> float:
    """数值稳定的 sigmoid（按符号分支，避免 ``exp`` 上溢）."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def log_sigmoid(value: float) -> float:
    """``log σ(x)`` = ``−softplus(−x)``，按符号分支实现.

    两种写法在数学上等价，但直接算 ``math.log(sigmoid(x))`` 在 ``x`` 很负时
    会先下溢成 ``0.0`` 再取对数得到 ``−inf``；本实现返回的是有限的
    ``x − log1p(exp(x))``。
    """
    if value >= 0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def softplus(value: float) -> float:
    """``log(1 + exp(x))``（同样按符号分支，``x`` 很大时不溢出）."""
    if value > 0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def dpo_margin(
    policy_chosen: float,
    policy_rejected: float,
    reference_chosen: float,
    reference_rejected: float,
) -> float:
    """DPO 的括号：``Δ_policy − Δ_ref``（**不含 β**）.

    四个参数都是**对数概率**（通常是把整条序列的逐 token 对数概率相加得到的
    序列对数概率）。把 ``β`` 留到 loss 里乘，是为了让这个量保持"纯数据"的
    语义：换 β 不改变 margin，只改变它对 loss 的影响强度。
    """
    for name, value in (
        ("policy_chosen", policy_chosen),
        ("policy_rejected", policy_rejected),
        ("reference_chosen", reference_chosen),
        ("reference_rejected", reference_rejected),
    ):
        if not math.isfinite(value):
            raise FinetuneEvalError(f"{name} 必须是有限数，收到 {value}")
    return (policy_chosen - policy_rejected) - (reference_chosen - reference_rejected)


def dpo_loss_from_margin(margin: float, *, beta: float) -> float:
    """``L = −log σ(β·margin)``.

    margin 为 0 时恒为 ``ln 2``；margin 很大时趋近 0；margin 很负时趋近
    ``−β·margin``（线性增长）。第三种情况是**必须看得见的**：
    一条被写错的偏好样本（chosen 其实更差）会给出很大的 loss，
    而它不该被任何"数值保护"悄悄压平。
    """
    _check_beta(beta)
    return -log_sigmoid(beta * margin)


def dpo_loss(
    *,
    policy_chosen: float,
    policy_rejected: float,
    reference_chosen: float,
    reference_rejected: float,
    beta: float,
) -> float:
    """完整签名的 DPO loss（四个对数概率 + β）."""
    return dpo_loss_from_margin(
        dpo_margin(policy_chosen, policy_rejected, reference_chosen, reference_rejected),
        beta=beta,
    )


def dpo_gradient(margin: float, *, beta: float) -> float:
    """``∂L/∂margin = −β·σ(−β·margin) = −β·(1 − σ(β·margin))``.

    它是本课"手算梯度"的落点：margin 为 0 时等于 ``−β/2``（一半的步长），
    margin 很大时趋近 0（已经拉开差距，梯度自然变小）——**这正是"越接近最优、
    梯度越小"的自适应行为**，也是 DPO 比"用正确/错误做监督"更稳的原因。
    """
    _check_beta(beta)
    return -beta * sigmoid(-beta * margin)


def dpo_probability(margin: float, *, beta: float) -> float:
    """模型认为 chosen 更被偏好的概率：``σ(β·margin)``.

    它的期望语义是"偏好正确率"：在留出的偏好对上统计它是否大于 0.5，
    就得到偏好准确率（见 ``reward.preference_accuracy``）。
    """
    _check_beta(beta)
    return sigmoid(beta * margin)


def implicit_reward(logprob: float, reference_logprob: float, *, beta: float) -> float:
    """DPO 的隐式奖励：``r(x,y) = β·(logπ_θ(y|x) − logπ_ref(y|x))``.

    **这是 DPO 名字里没有明说、但最重要的那一半**：它并不是"不需要奖励"，
    而是把奖励定义成策略与参考模型的对数概率比。由此得到三个推论：

    - 参考模型一动，全部隐式奖励一起平移（所以参考模型必须冻结）；
    - ``β`` 同时缩放奖励与 KL 约束，两者没法独立调；
    - 训练初期（π_θ = π_ref）所有隐式奖励都是 0，**排序信息为零**——
      这就是 loss 恰好为 ln 2 的另一面。
    """
    _check_beta(beta)
    if not math.isfinite(logprob) or not math.isfinite(reference_logprob):
        raise FinetuneEvalError("对数概率必须是有限数")
    return beta * (logprob - reference_logprob)


def implicit_reward_margin(
    *,
    policy_chosen: float,
    policy_rejected: float,
    reference_chosen: float,
    reference_rejected: float,
    beta: float,
) -> float:
    """隐式奖励之差 ``r(x,y_w) − r(x,y_l) = β·margin``.

    它与 ``dpo_probability`` 是同一件事的两种写法：``σ(r_w − r_l)`` 就是
    DPO 对"chosen 更被偏好"的预测概率。
    """
    return beta * dpo_margin(
        policy_chosen, policy_rejected, reference_chosen, reference_rejected
    )


def kl_estimators(logprob: float, reference_logprob: float) -> dict[str, float]:
    """三种逐样本 KL 估计器（``KL(π_θ ‖ π_ref)``），都由对数概率算出.

    .. code-block:: text

        k1 = log p − log q
        k2 = 0.5 · (log p − log q)²
        k3 = exp(r) − r − 1       其中 r = log q − log p

    三者的取舍（这是 RLHF 里最常见的"看起来都对"的坑）：

    - ``k1`` **有偏且可能为负**——单样本的 log 比可以是负数，于是"KL 惩罚"
      偶尔会奖励偏离参考模型的行为；
    - ``k2`` 恒非负，但方差大（它把 log 比平方，放大了离群样本）；
    - ``k3`` 恒非负、方差最小、且是无偏估计（Schulman 提出的那一个），
      代价是它需要 ``exp``，极端情况下要小心溢出。

    本课报告里**三个都报**，因为"只报一个"会掩盖它们的分歧：
    在训练早期，k1 常常给出负数，而 k3 已经稳定为正。
    """
    if not math.isfinite(logprob) or not math.isfinite(reference_logprob):
        raise FinetuneEvalError("对数概率必须是有限数")
    log_ratio = logprob - reference_logprob
    shifted = reference_logprob - logprob
    return {
        "k1": log_ratio,
        "k2": 0.5 * log_ratio * log_ratio,
        "k3": math.expm1(shifted) - shifted,
    }


def mean_kl(
    logprobs: Sequence[float],
    reference_logprobs: Sequence[float],
    *,
    estimator: str = "k3",
) -> float:
    """一批样本上的平均 KL（缺省用 ``k3``，理由见 ``kl_estimators``）."""
    if len(logprobs) != len(reference_logprobs):
        raise FinetuneEvalError(
            f"两串对数概率长度必须一致：{len(logprobs)} != {len(reference_logprobs)}"
        )
    if not logprobs:
        raise FinetuneEvalError("不能对空集合求平均 KL")
    if estimator not in KL_ESTIMATORS:
        raise FinetuneEvalError(
            f"未知的 KL 估计器 {estimator!r}，可选：{', '.join(KL_ESTIMATORS)}"
        )
    values = [
        kl_estimators(logprob, reference)[estimator]
        for logprob, reference in zip(logprobs, reference_logprobs, strict=True)
    ]
    return sum(values) / len(values)


def reward_with_kl_shaping(
    reward: float,
    *,
    logprob: float,
    reference_logprob: float,
    beta: float,
) -> float:
    """逐样本的 RLHF 目标：``r(x,y) − β·(log π_θ − log π_ref)``.

    这是 PPO 里"奖励塑形"的写法：把 KL 惩罚**折进奖励**，于是可以继续用
    现成的强化学习算法。它与 DPO 的关系值得说清——把上式对策略取期望、
    并把"奖励模型"换成 DPO 的隐式奖励，两条路径在数学上会汇合到同一个
    最优策略（``π* ∝ π_ref·exp(r/β)``）。
    """
    _check_beta(beta)
    return reward - beta * (logprob - reference_logprob)


@dataclass(frozen=True)
class RLHFObjective:
    """一次 RLHF（PPO）目标的读数."""

    samples: int
    mean_reward: float
    kl: float
    beta: float
    estimator: str = "k3"

    @property
    def objective(self) -> float:
        """``E[r] − β·KL``（PPO 真正在最大化的量）."""
        return self.mean_reward - self.beta * self.kl

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"样本 {self.samples} | 平均奖励 {self.mean_reward:.6f} | "
            f"KL({self.estimator}) {self.kl:.6f} | β {self.beta} | "
            f"目标 {self.objective:.6f}"
        )

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典."""
        return {
            "samples": self.samples,
            "mean_reward": self.mean_reward,
            "kl": self.kl,
            "beta": self.beta,
            "estimator": self.estimator,
            "objective": self.objective,
        }


def rlhf_ppo_objective(
    rewards: Sequence[float],
    logprobs: Sequence[float],
    reference_logprobs: Sequence[float],
    *,
    beta: float,
    estimator: str = "k3",
) -> RLHFObjective:
    """横向汇总：``E[r] − β·KL``（PPO 的目标函数）."""
    _check_beta(beta)
    if len(rewards) != len(logprobs):
        raise FinetuneEvalError(
            f"奖励与对数概率长度必须一致：{len(rewards)} != {len(logprobs)}"
        )
    if not rewards:
        raise FinetuneEvalError("RLHF 目标需要至少一个样本")
    kl = mean_kl(logprobs, reference_logprobs, estimator=estimator)
    return RLHFObjective(
        samples=len(rewards),
        mean_reward=sum(rewards) / len(rewards),
        kl=kl,
        beta=beta,
        estimator=estimator,
    )


def objective_requirements() -> list[dict[str, Any]]:
    """两个对齐目标的对照表（**做成数据**，因此可以被测试与渲染）.

    本课把这张表写成代码里的数据而不是文档里的表格，理由与 day052 把
    accelerate 配置写成数据一样：**文档里的表格会腐烂，代码里的表格会被测试**。
    """
    return [
        {
            "objective": "RLHF（PPO）",
            "stages": 3,
            "stage_detail": "训奖励模型 → 采样 rollout → 强化学习更新",
            "models_needed": ["policy", "reference", "reward", "value"],
            "preference_usage": "先拟合成分数，再用分数训练策略",
            "online_sampling": True,
            "kl_penalty": "显式（在目标里，β·KL）",
            "main_failure": "奖励模型被过优化（reward hacking）；四个模型一起调",
            "signal_reusable": True,
        },
        {
            "objective": "DPO",
            "stages": 1,
            "stage_detail": "直接在偏好对上做监督式优化",
            "models_needed": ["policy", "reference"],
            "preference_usage": "直接进 loss（chosen / rejected 的对数概率差）",
            "online_sampling": False,
            "kl_penalty": "隐式（藏在与参考模型的对数比里）",
            "main_failure": "过优化（margin 一路涨而验证准确率下降）；对数据质量极敏感",
            "signal_reusable": False,
        },
    ]


def _check_beta(beta: float) -> None:
    """β 必须是正的有限数（它同时是温度与 KL 权重，为 0 时目标无定义）."""
    if not math.isfinite(beta) or beta <= 0:
        raise FinetuneEvalError(f"beta 必须是正的有限数，收到 {beta}")


__all__ = [
    "BETA_SOFT_RANGE",
    "KL_ESTIMATORS",
    "ZERO_MARGIN_LOSS",
    "RLHFObjective",
    "dpo_gradient",
    "dpo_loss",
    "dpo_loss_from_margin",
    "dpo_margin",
    "dpo_probability",
    "implicit_reward",
    "implicit_reward_margin",
    "kl_estimators",
    "log_sigmoid",
    "mean_kl",
    "objective_requirements",
    "reward_with_kl_shaping",
    "rlhf_ppo_objective",
    "sigmoid",
    "softplus",
]
