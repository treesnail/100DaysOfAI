"""奖励、优势与"离线的对齐信号"（M5-D6）.

RLHF 里最贵的三样东西是：一张偏好标注表、一个奖励模型、以及**在线采样**
（rollout）。本模块把"能在离线、确定、可手算的条件下量出来的那些信号"集中起来，
它们正好覆盖了对齐训练真正需要盯住的三件事：

============================  ==================================================
``preference_accuracy``       留出偏好对上的**偏好准确率**（"模型更偏好 chosen 吗"）
``policy_kl``                 策略相对参考模型的 KL（偏离了多少）
``gae_advantages``            优势估计（PPO 真正用来加权梯度的那一项）
============================  ==================================================

一个必须先说清的边界：**这里的"奖励"不是训练出来的奖励模型**，而是
DPO 的隐式奖励（``β·log(π_θ/π_ref)``，见 ``objectives.implicit_reward``）。
它是一个**闭式**的量：只要有两个模型与一段文本，就能算出来，不需要采样、
不需要 rollout、也没有额外参数。这正是 DPO 相比 PPO 在工程上"便宜"的地方——
**便宜的地方也正是它信息少的地方**（奖励信号无法被复用到别的用途）。

优势估计（``gae_advantages``）留在这里而不是删掉，是因为它是理解 PPO 目标的
必要一环：PPO 用优势而不是回报来加权策略梯度，而优势里那个 ``γλ`` 决定了
"奖励归因到多远"。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from smart_research_agent.alignment.objectives import (
    dpo_margin,
    implicit_reward_margin,
    mean_kl,
)
from smart_research_agent.alignment.preference import PreferencePair
from smart_research_agent.finetune_eval.metrics import FinetuneEvalError
from smart_research_agent.finetune_eval.model_probe import ContextModel, sequence_logprob
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: GAE 的两个折减系数：``gamma`` 折未来奖励、``lambda`` 折优势的自举.
#: 两个都取 1 时退化为"回报减基线"（无折减）；取 0 时只看即时奖励。
DEFAULT_GAMMA = 1.0
DEFAULT_LAMBDA = 0.95

#: 过优化体检的默认 KL 预算.
#:
#: 它定义在**本模块**而不是 ``dpo_trainer``：KL 预算是"读历史"的判据，
#: 属于本模块的职责；训练器只是把历史交出来。放在训练器里会造成
#: "读历史的模块需要 import 训练器"的循环导入——本课实现时真的踩到了。
DEFAULT_KL_BUDGET = 0.5


@dataclass(frozen=True)
class PairReward:
    """一条偏好对上的四个对数概率与由它们算出的三个量."""

    pair_id: str
    dimension: str
    policy_chosen: float
    policy_rejected: float
    reference_chosen: float
    reference_rejected: float
    chosen_tokens: int
    rejected_tokens: int
    beta: float

    @property
    def margin(self) -> float:
        """DPO 的括号 ``Δ_policy − Δ_ref``（不含 β）."""
        return dpo_margin(
            self.policy_chosen,
            self.policy_rejected,
            self.reference_chosen,
            self.reference_rejected,
        )

    @property
    def reward_margin(self) -> float:
        """隐式奖励之差 ``β·margin``（> 0 表示模型更偏好 chosen）."""
        return implicit_reward_margin(
            policy_chosen=self.policy_chosen,
            policy_rejected=self.policy_rejected,
            reference_chosen=self.reference_chosen,
            reference_rejected=self.reference_rejected,
            beta=self.beta,
        )

    @property
    def correct(self) -> bool:
        """这一对是否被"判对"（隐式奖励之差为正）."""
        return self.reward_margin > 0

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典."""
        return {
            "pair_id": self.pair_id,
            "dimension": self.dimension,
            "policy_chosen": self.policy_chosen,
            "policy_rejected": self.policy_rejected,
            "reference_chosen": self.reference_chosen,
            "reference_rejected": self.reference_rejected,
            "chosen_tokens": self.chosen_tokens,
            "rejected_tokens": self.rejected_tokens,
            "beta": self.beta,
            "margin": self.margin,
            "reward_margin": self.reward_margin,
            "correct": self.correct,
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        state = "判对" if self.correct else "判错"
        return (
            f"{self.pair_id}（{self.dimension}）margin {self.margin:+.6f} | "
            f"隐式奖励差 {self.reward_margin:+.6f} | {state}"
        )


def pair_reward(
    policy: ContextModel,
    reference: ContextModel,
    tokenizer: Any,
    pair: PreferencePair,
    *,
    beta: float,
) -> PairReward:
    """算一条偏好对的四个对数概率与隐式奖励差.

    四个量都来自 ``sequence_logprob``（day053 的白盒探针用的同一个函数），
    因此口径天然一致：**除第一个 token 外的每一个位置都计分**，分母交出来
    （``chosen_tokens`` / ``rejected_tokens``）。

    这里有一处容易被忽略的语义：chosen 与 rejected 的长度通常不同，
    所以"对数概率之和"不可直接互相比较——长度越长，和越负。
    这正是 DPO 要取**差**（而不是比值或平均）的原因之一：
    但严格说，长度差异仍然会通过"多出来的那几个 token"影响 margin，
    这也是长度偏置会渗进 DPO 的路径（见 ``preference.length_bias_report``）。
    """
    chosen_ids = tokenizer.encode(pair.chosen)
    rejected_ids = tokenizer.encode(pair.rejected)
    policy_chosen, chosen_tokens = sequence_logprob(policy, chosen_ids)
    policy_rejected, rejected_tokens = sequence_logprob(policy, rejected_ids)
    reference_chosen, _ = sequence_logprob(reference, chosen_ids)
    reference_rejected, _ = sequence_logprob(reference, rejected_ids)
    return PairReward(
        pair_id=pair.id,
        dimension=pair.dimension,
        policy_chosen=policy_chosen,
        policy_rejected=policy_rejected,
        reference_chosen=reference_chosen,
        reference_rejected=reference_rejected,
        chosen_tokens=chosen_tokens,
        rejected_tokens=rejected_tokens,
        beta=beta,
    )


def reward_table(
    policy: ContextModel,
    reference: ContextModel,
    tokenizer: Any,
    pairs: Sequence[PreferencePair],
    *,
    beta: float,
) -> list[PairReward]:
    """对一批偏好对逐条算隐式奖励（顺序与输入一致，报告才能逐行比对）."""
    if not pairs:
        raise FinetuneEvalError("不能对空偏好数据集算奖励")
    return [pair_reward(policy, reference, tokenizer, pair, beta=beta) for pair in pairs]


def preference_accuracy(
    policy: ContextModel,
    reference: ContextModel,
    tokenizer: Any,
    pairs: Sequence[PreferencePair],
    *,
    beta: float,
) -> dict[str, Any]:
    """留出偏好对上的**偏好准确率**（对齐训练唯一的离线判据）.

    判据是"隐式奖励之差 > 0"，也就是 ``σ(β·margin) > 0.5``。

    **为什么必须用留出数据**：训练集上的 margin 会单调地涨（那是被直接
    优化的量），而它涨到什么时候开始有害，只有留出数据能回答。
    用训练集 margin 当早停判据，等价于"训练 loss 降到 0 就停"——
    对齐任务里这两件事之间的差距比 SFT 大得多。
    """
    if not pairs:
        raise FinetuneEvalError("不能对空偏好数据集算偏好准确率")
    rewards = reward_table(policy, reference, tokenizer, pairs, beta=beta)
    correct = sum(1 for item in rewards if item.correct)
    correct_by_dimension: dict[str, list[bool]] = {}
    for item in rewards:
        correct_by_dimension.setdefault(item.dimension, []).append(item.correct)
    return {
        "total": len(rewards),
        "correct": correct,
        "accuracy": correct / len(rewards),
        "mean_margin": sum(item.margin for item in rewards) / len(rewards),
        "mean_reward_margin": sum(item.reward_margin for item in rewards) / len(rewards),
        "by_dimension": {
            dimension: {
                "total": len(flags),
                "correct": sum(1 for flag in flags if flag),
                "accuracy": sum(1 for flag in flags if flag) / len(flags),
            }
            for dimension, flags in sorted(correct_by_dimension.items())
        },
        "pairs": [item.to_dict() for item in rewards],
    }


def policy_kl(
    policy: ContextModel,
    reference: ContextModel,
    tokenizer: Any,
    pairs: Sequence[PreferencePair],
    *,
    estimator: str = "k3",
) -> float:
    """策略相对参考模型的平均 KL（chosen 与 rejected 一起统计）.

    用同一批文本在**两个模型**上各算一次对数概率，再取 ``k3`` 估计器的均值。
    它是"对齐把模型推离了多远"的直接度量。

    **"β 越大 KL 越小"这句话是错的**——本课实测（``lr=0.5``、42 步、同一批
    训练对）KL 随 β 先升后降::

        β = 0.05 → KL 0.012482        β = 0.5 → KL 0.446000
        β = 0.1  → KL 0.044928        β = 1.0 → KL 0.593957
        β = 2.0  → KL 0.498689

    原因是 β 同时出现在两个地方：它是 loss 里的温度（**也**是梯度系数的
    一部分 ``−β·σ(−βm)``）。β 小则每一步走得少；β 大则早期步长大，但 margin
    一大 ``σ(−βm)`` 就迅速衰减，**反而早早刹住**。两个效应叠加的结果就是
    上面这张表。所以"β 与 KL 的关系"没有一句能写对的话，只能实测——
    这也是本课把这张表放进文档而不是放进一句结论里的原因。
    """
    logprobs: list[float] = []
    references: list[float] = []
    for pair in pairs:
        for text in (pair.chosen, pair.rejected):
            ids = tokenizer.encode(text)
            value, _ = sequence_logprob(policy, ids)
            base, _ = sequence_logprob(reference, ids)
            logprobs.append(value)
            references.append(base)
    return mean_kl(logprobs, references, estimator=estimator)


def reward_normalize(values: Sequence[float]) -> tuple[list[float], float, float]:
    """奖励的标准化（白化）:``(r − μ) / σ``，返回 ``(标准化值, μ, σ)``.

    ``σ = 0`` 时**返回全 0**，而不是抛异常或除以 0：常数奖励意味着"这一批
    样本没有区分度"，标准化后的唯一诚实答案是"都是 0"，让调用方在上层
    发现"优势全为 0、梯度也就全为 0"这件事。
    """
    if not values:
        raise FinetuneEvalError("不能对空序列做标准化")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std == 0.0:
        return ([0.0] * len(values), mean, 0.0)
    return ([(value - mean) / std for value in values], mean, std)


def gae_advantages(
    rewards: Sequence[float],
    values: Sequence[float],
    *,
    gamma: float = DEFAULT_GAMMA,
    lam: float = DEFAULT_LAMBDA,
) -> list[float]:
    """广义优势估计（GAE）：``δ_t = r_t + γV(s_{t+1}) − V(s_t)``，``A_t = δ_t + γλ·A_{t+1}``.

    三条实现口径写在这里，因为它们都会显著改变数值：

    1. **最后一步之后的价值取 0**（``V(s_T) = 0``）：这是"回合结束"的约定，
       等价的写法是让调用方多传一个落地的 ``V(s_T)``；
    2. **从后往前累加**（自举），所以 ``A_t`` 依赖 ``A_{t+1}``——
       这个递推式就是"用未来校正现在"的全部内容；
    3. ``γ = λ = 1`` 时退化为"回报减基线"：``A_t = Σ_{k≥t} r_k − V(s_t)``。

    长度不一致直接报错：优势与奖励一一对应，"少一个"会让所有下标错位，
    而错位之后算出来的优势**看起来完全正常**。
    """
    if len(rewards) != len(values):
        raise FinetuneEvalError(
            f"奖励与价值长度必须一致：{len(rewards)} != {len(values)}"
        )
    if not rewards:
        raise FinetuneEvalError("GAE 需要至少一个时间步")
    if not 0.0 <= gamma <= 1.0:
        raise FinetuneEvalError(f"gamma 必须落在 [0, 1]，收到 {gamma}")
    if not 0.0 <= lam <= 1.0:
        raise FinetuneEvalError(f"lambda 必须落在 [0, 1]，收到 {lam}")
    advantages = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_value = values[index + 1] if index + 1 < len(values) else 0.0
        delta = rewards[index] + gamma * next_value - values[index]
        running = delta + gamma * lam * running
        advantages[index] = running
    return advantages


def advantage_report(advantages: Sequence[float]) -> dict[str, Any]:
    """优势的汇总（均值 / 标准差 / 正负比例）.

    ``positive_fraction`` 是"这一批里有多少步的优势为正"：它与均值一起看
    才能区分"整体偏好某一方向"与"两极分化"（后者会让策略梯度的方差很大）。
    """
    if not advantages:
        raise FinetuneEvalError("不能对空优势序列做汇总")
    mean = sum(advantages) / len(advantages)
    variance = sum((value - mean) ** 2 for value in advantages) / len(advantages)
    positives = sum(1 for value in advantages if value > 0)
    return {
        "steps": len(advantages),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(advantages),
        "max": max(advantages),
        "positive_fraction": positives / len(advantages),
    }


def over_optimization_flags(
    history: Sequence[dict[str, Any]],
    *,
    kl_budget: float = DEFAULT_KL_BUDGET,
) -> list[str]:
    """过优化体检的**人类可读清单**（判据见 ``optimization_summary``）."""
    return optimization_summary(history, kl_budget=kl_budget)["flags"]


def optimization_summary(
    history: Sequence[dict[str, Any]],
    *,
    kl_budget: float = DEFAULT_KL_BUDGET,
) -> dict[str, Any]:
    """从训练历史里读出"训练到哪一步该停"，并把三条判据的结论都交出来.

    三条判据（**每一条都对应一种可以被观测到的故障**）：

    ========================  ==========================================================================
    ``over_optimized``        留出 margin 从峰值回落，而训练 margin 仍在涨——过优化的定义式迹象
    ``kl_exceeded``           最终 KL 超出预算——策略偏离参考模型过远，格式与引用能力会一起退化
    ``accuracy_saturated``    留出准确率自某步起不再创新高——继续训练买不到留出收益
    ========================  ==========================================================================

    为什么要有这个"把结论都交出来"的版本，而不只是一个 flag 清单：
    **空的清单是有歧义的**——它既可能表示"没有过优化"，也可能表示"历史太短、
    什么都没测出来"。``optimization_summary`` 把三个布尔值与峰值位置都写出来，
    于是"没触发"与"没测到"被分开了。

    本课实测（参考模型、7 条训练对、5 条留出对）：

    - 42 步（``beta=0.1`` / ``lr=0.5``）：留出 margin ``+0.0794``、KL ``0.0449``、
      准确率 ``0.8`` —— 三条判据都**没有**触发；
    - 140 步（``beta=0.1`` / ``lr=2.0``，步数是前者的 3.3 倍）：留出 margin 涨到
      ``+0.9089``（11.4 倍）、KL 涨到 ``2.4276``（**54 倍**，超预算 4.9 倍），
      而准确率**仍然停在 0.8** —— ``kl_exceeded`` 与 ``accuracy_saturated`` 同时触发。
      这就是"继续训练买到了什么"的答案：**只买到了 KL**。
    """
    if not history:
        return {
            "steps": 0,
            "valid_margin_peak": 0.0,
            "valid_margin_peak_step": 0,
            "valid_margin_final": 0.0,
            "valid_accuracy_best": 0.0,
            "valid_accuracy_best_step": 0,
            "valid_accuracy_final": 0.0,
            "kl_final": 0.0,
            "over_optimized": False,
            "kl_exceeded": False,
            "accuracy_saturated": False,
            "flags": [],
        }
    margins = [entry.get("valid_margin", 0.0) for entry in history]
    accuracies = [entry.get("valid_accuracy", 0.0) for entry in history]
    train_margins = [entry.get("mean_margin", 0.0) for entry in history]
    peak_index = max(range(len(margins)), key=lambda index: margins[index])
    best_accuracy_index = max(range(len(accuracies)), key=lambda index: accuracies[index])
    final_kl = history[-1].get("kl", 0.0)
    over_optimized = peak_index < len(margins) - 1 and margins[-1] < margins[peak_index] - 1e-12
    accuracy_saturated = best_accuracy_index < len(accuracies) - 1
    flags: list[str] = []
    if over_optimized:
        flags.append(
            f"留出 margin 在第 {history[peak_index].get('step', peak_index)} 步达到峰值 "
            f"{margins[peak_index]:+.6f} 后回落到 {margins[-1]:+.6f}：典型过优化，"
            f"早停点应在峰值附近（训练 margin 此时仍有 "
            f"{abs(train_margins[-1]):.6f}，**它不会告诉你这件事**）"
        )
    if final_kl > kl_budget:
        flags.append(
            f"最终 KL {final_kl:.6f} 超出预算 {kl_budget}：策略偏离参考模型过远，"
            "格式与引用能力可能一起退化"
        )
    if accuracy_saturated:
        flags.append(
            f"留出准确率自第 {history[best_accuracy_index].get('step', best_accuracy_index)} 步"
            f"起不再创新高（停在 {accuracies[best_accuracy_index]:.4f}）："
            f"继续训练买不到留出收益，而 KL 已从 "
            f"{history[best_accuracy_index].get('kl', 0.0):.6f} 涨到 {final_kl:.6f}"
        )
    return {
        "steps": len(history),
        "valid_margin_peak": margins[peak_index],
        "valid_margin_peak_step": history[peak_index].get("step", peak_index),
        "valid_margin_final": margins[-1],
        "valid_accuracy_best": accuracies[best_accuracy_index],
        "valid_accuracy_best_step": history[best_accuracy_index].get("step", best_accuracy_index),
        "valid_accuracy_final": accuracies[-1],
        "kl_final": final_kl,
        "over_optimized": over_optimized,
        "kl_exceeded": final_kl > kl_budget,
        "accuracy_saturated": accuracy_saturated,
        "flags": flags,
    }


__all__ = [
    "DEFAULT_GAMMA",
    "DEFAULT_KL_BUDGET",
    "DEFAULT_LAMBDA",
    "PairReward",
    "advantage_report",
    "gae_advantages",
    "optimization_summary",
    "over_optimization_flags",
    "pair_reward",
    "policy_kl",
    "preference_accuracy",
    "reward_normalize",
    "reward_table",
]
