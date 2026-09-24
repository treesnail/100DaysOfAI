"""对齐策略：给 SmartResearch Agent 的输出风格定"要买什么、要多少样本"（M5-D6）.

前三个模块解决"怎么算"，这一模块解决"**做什么**"。对齐与 SFT 最大的区别
在于它的**验收标准不是"会不会"**，而是"在两个都说得通的答法里更偏好哪一个"——
这意味着两件事：

1. **必须先说清偏好的是哪几个维度**。同一份标注在两个维度上会给出相反的
   结论（"信息更全"vs"更简洁"），维度不明确时标注者之间的一致度会低到
   数据不可用。本课固定五个维度（见 ``preference.ALIGNMENT_DIMENSIONS``）。
2. **样本量要提前算**。偏好数据的边际成本很高（每条都要人读两个回答），
   而"要多少条才够"是一个纯粹的样本量问题——本模块把它算出来，
   并把"当前的收集量离目标差多少"作为开工与否的判据。

一个必须写清楚的近似：``pairs_for_margin`` 用的是"**二元比例 vs 0.5**"的
样本量公式（正态近似），它回答的是"想在偏好准确率上分辨出 60% 与 50%，
需要多少条**独立**样本"。真实情况比这复杂（配对、分层、标注噪声），
所以本课把它当作**下界**而不是承诺——算出来的数偏小，不会偏大。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from smart_research_agent.alignment.preference import (
    ALIGNMENT_DIMENSIONS,
    DIMENSION_GOALS,
)
from smart_research_agent.finetune_eval.metrics import FinetuneEvalError

#: 单侧正态分位数表（按置信水平 / 功效查表，**不引 scipy**）.
#:
#: 用表而不是库，与 day053 拒绝统计库是同一条理由：这一课的这一节要能
#: 被手算核对，而 ``z = 1.96`` 是每个人都能在教材里找到的数字。
Z_BY_ALPHA: dict[float, float] = {
    0.10: 1.6449,
    0.05: 1.9600,
    0.02: 2.3263,
    0.01: 2.5758,
}

Z_BY_POWER: dict[float, float] = {
    0.80: 0.8416,
    0.90: 1.2816,
    0.95: 1.6449,
}

#: 默认目标：让偏好准确率从"抛硬币"提升到 60% 可被检出.
DEFAULT_TARGET_ACCURACY = 0.6

#: 默认 β 与学习率（本课的标定值，见 ``docs/alignment.md``）.
DEFAULT_BETA = 0.1
DEFAULT_LEARNING_RATE = 0.5

#: 对齐的四个主要风险（做成数据：可以被测试、被 API 返回、被渲染）.
ALIGNMENT_RISKS: tuple[dict[str, str], ...] = (
    {
        "name": "长度偏置",
        "symptom": "chosen 系统性比 rejected 长，模型学会『越长越好』",
        "mitigation": "重写长回复让两侧长度可比；用 length_bias_report 在开工前粗筛",
        "detect": "preference.length_bias_report 的 chosen_longer_fraction",
    },
    {
        "name": "过优化",
        "symptom": "训练 margin 一路涨，留出偏好准确率先升后降；格式与出处开始丢",
        "mitigation": "按验证准确率峰值早停；把 KL 预算写进门禁",
        "detect": "reward.over_optimization_flags 的三条判据",
    },
    {
        "name": "维度混淆",
        "symptom": "标注者按各自的维度打分，一致度低（kappa < 0.4）",
        "mitigation": "每条样本标注 dimension；双标一小批算 kappa 再放量",
        "detect": "strategy.annotator_agreement 的 kappa",
    },
    {
        "name": "参考模型漂移",
        "symptom": "隐式奖励整体平移，前后两次训练的 margin 不可比",
        "mitigation": "参考模型冻结并在报告里记内容哈希；换基座就要重训",
        "detect": "报告里的 reference 哈希与偏好数据指纹",
    },
)


@dataclass(frozen=True)
class SampleBudget:
    """一个维度上的样本量目标与实际收集量."""

    dimension: str
    collected: int
    required: int
    goal: str = ""

    @property
    def gap(self) -> int:
        """还差多少条（已达标时为 0，不返回负数——"超出目标"不是缺口）."""
        return max(0, self.required - self.collected)

    @property
    def ready(self) -> bool:
        """这个维度的样本量是否已经够开工."""
        return self.collected >= self.required

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        state = "可开工" if self.ready else f"还差 {self.gap} 条"
        return f"{self.dimension}: {self.collected}/{self.required}（{state}）"

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典."""
        return {
            "dimension": self.dimension,
            "collected": self.collected,
            "required": self.required,
            "gap": self.gap,
            "ready": self.ready,
            "goal": self.goal,
        }


def z_for_confidence(alpha: float) -> float:
    """按显著性水平查单侧正态分位数（只支持表里的取值）."""
    if alpha not in Z_BY_ALPHA:
        raise FinetuneEvalError(
            f"不支持的 alpha {alpha}，可选：{sorted(Z_BY_ALPHA)}（刻意查表而不是引统计库）"
        )
    return Z_BY_ALPHA[alpha]


def z_for_power(power: float) -> float:
    """按检验功效查单侧正态分位数（只支持表里的取值）."""
    if power not in Z_BY_POWER:
        raise FinetuneEvalError(
            f"不支持的 power {power}，可选：{sorted(Z_BY_POWER)}"
        )
    return Z_BY_POWER[power]


def pairs_for_margin(
    *,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    alpha: float = 0.05,
    power: float = 0.80,
    baseline: float = 0.5,
) -> int:
    """要多少条偏好对，才能在统计上分辨出目标准确率与基线（默认 50%）.

    .. code-block:: text

        n = ⌈ (z_{α/2} + z_β)² · p(1−p) / (p − baseline)² ⌉

    实测（``alpha = 0.05`` → ``z = 1.96``；``power = 0.80`` → ``z = 0.8416``；
    ``(1.96 + 0.8416)² = 7.8489`` 是分子里的公因子）::

        p = 0.55 → 7.8489 × 0.2475 / 0.0025 = 777.04  → 778 条
        p = 0.60 → 7.8489 × 0.2400 / 0.0100 = 188.37  → 189 条
        p = 0.65 → 7.8489 × 0.2275 / 0.0225 =  79.36  →  80 条
        p = 0.70 → 7.8489 × 0.2100 / 0.0400 =  41.21  →  42 条

    这张表本身就是结论：**"提升 5 个百分点"要比"提升 20 个百分点"多花
    一个数量级的标注量**。所以小团队的现实选择通常是"先对齐一个明显的
    维度（拒答、格式），不要一上来追 5% 的提升"。
    """
    if not 0.0 < baseline < 1.0:
        raise FinetuneEvalError(f"baseline 必须落在 (0, 1)，收到 {baseline}")
    if not baseline < target_accuracy < 1.0:
        raise FinetuneEvalError(
            f"target_accuracy 必须落在 ({baseline}, 1)，收到 {target_accuracy}"
        )
    z_alpha = z_for_confidence(alpha)
    z_power = z_for_power(power)
    variance = target_accuracy * (1 - target_accuracy)
    effect = (target_accuracy - baseline) ** 2
    return math.ceil((z_alpha + z_power) ** 2 * variance / effect)


def annotator_agreement(labels_a: Sequence[str], labels_b: Sequence[str]) -> dict[str, Any]:
    """两位标注者的一致性：观察一致度、期望一致度与 Cohen's κ.

    .. code-block:: text

        p_o = 一致条数 / n
        p_e = Σ_c  (A 用 c 的比例) × (B 用 c 的比例)
        κ   = (p_o − p_e) / (1 − p_e)

    ``p_e`` 是"两人各自独立按比例乱标也会凑出的一致度"。减去它，是因为
    **在一个答案 90% 都合格的标注任务里，90% 的观察一致度等于零信息**。

    分档用 Landis & Koch 的常用区间（<0 差 / 0~0.2 轻微 / 0.2~0.4 一般 /
    0.4~0.6 中等 / 0.6~0.8 显著 / >0.8 几乎完全）。

    一处边界要写清楚：当双方都只用了同一个标签时 ``p_e = 1``，κ 的分母为 0。
    本函数返回 ``kappa = 0.0`` 并标 ``degenerate = True``——
    **"完全一致但无信息"（例如两人都全打了 A）不该被报告成 κ = 1.0**。
    """
    if len(labels_a) != len(labels_b):
        raise FinetuneEvalError(
            f"两份标注长度必须一致：{len(labels_a)} != {len(labels_b)}"
        )
    if len(labels_a) < 2:
        raise FinetuneEvalError("至少需要 2 条标注才能算一致性")
    total = len(labels_a)
    observed = sum(1 for left, right in zip(labels_a, labels_b, strict=True) if left == right)
    p_observed = observed / total
    categories = sorted(set(labels_a) | set(labels_b))
    p_expected = 0.0
    for category in categories:
        share_a = sum(1 for label in labels_a if label == category) / total
        share_b = sum(1 for label in labels_b if label == category) / total
        p_expected += share_a * share_b
    if abs(1.0 - p_expected) < 1e-12:
        return {
            "total": total,
            "observed_agreement": p_observed,
            "expected_agreement": p_expected,
            "kappa": 0.0,
            "degenerate": True,
            "interpretation": "完全一致但无信息（双方只用了同一个标签），κ 无定义，取 0.0",
            "categories": categories,
        }
    kappa = (p_observed - p_expected) / (1.0 - p_expected)
    return {
        "total": total,
        "observed_agreement": p_observed,
        "expected_agreement": p_expected,
        "kappa": kappa,
        "degenerate": False,
        "interpretation": interpret_kappa(kappa),
        "categories": categories,
    }


def interpret_kappa(kappa: float) -> str:
    """把 κ 映射成 Landis & Koch 的常用文字分档."""
    if kappa < 0.0:
        return "差（比随机还差）"
    if kappa < 0.2:
        return "轻微一致"
    if kappa < 0.4:
        return "一般一致"
    if kappa < 0.6:
        return "中等一致"
    if kappa < 0.8:
        return "显著一致"
    return "几乎完全一致"


def dimension_table() -> list[dict[str, Any]]:
    """五个对齐维度的目标说明（进 API 响应与文档）."""
    return [
        {"dimension": dimension, "goal": DIMENSION_GOALS[dimension]}
        for dimension in ALIGNMENT_DIMENSIONS
    ]


def alignment_plan(
    *,
    collected: Mapping[str, int] | None = None,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    alpha: float = 0.05,
    power: float = 0.80,
    beta: float = DEFAULT_BETA,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    kl_budget: float = 0.5,
) -> dict[str, Any]:
    """把"要买什么、要多少、怎么配"打成一份可执行的计划.

    ``collected`` 是各维度已收集的偏好对数量。计划里每个维度都给出
    ``required`` 与 ``gap``，并给出 ``ready`` 总判据——
    **"数据量够不够开工"这件事必须有一个明确的答案**，否则最常见的
    做法是"先随便标几条跑起来再说"，而那样得到的对齐结果无法解释。
    """
    counts = dict(collected or {})
    unknown = sorted(set(counts) - set(ALIGNMENT_DIMENSIONS))
    if unknown:
        raise FinetuneEvalError(
            f"未知的对齐维度：{', '.join(unknown)}。可用维度：{', '.join(ALIGNMENT_DIMENSIONS)}"
        )
    for dimension, value in counts.items():
        if value < 0:
            raise FinetuneEvalError(f"维度 {dimension} 的收集量不能为负数，收到 {value}")
    required = pairs_for_margin(
        target_accuracy=target_accuracy, alpha=alpha, power=power
    )
    budgets = [
        SampleBudget(
            dimension=dimension,
            collected=int(counts.get(dimension, 0)),
            required=required,
            goal=DIMENSION_GOALS[dimension],
        )
        for dimension in ALIGNMENT_DIMENSIONS
    ]
    total_collected = sum(item.collected for item in budgets)
    total_required = required * len(ALIGNMENT_DIMENSIONS)
    return {
        "target_accuracy": target_accuracy,
        "alpha": alpha,
        "power": power,
        "per_dimension_required": required,
        "budgets": [item.to_dict() for item in budgets],
        "summary": {
            "collected": total_collected,
            "required": total_required,
            "gap": max(0, total_required - total_collected),
            "ready_dimensions": sum(1 for item in budgets if item.ready),
            "dimensions": len(budgets),
            "ready": all(item.ready for item in budgets),
        },
        "dpo_config": {
            "beta": beta,
            "learning_rate": learning_rate,
            "valid_ratio": 0.25,
            "kl_budget": kl_budget,
            "note": "β 与 KL 不是单调关系（固定步数时 KL 随 β 先升后降），"
            "必须像学习率一样实测标定，不要照抄别人的 β",
        },
        "dimensions": dimension_table(),
        "risks": [dict(risk) for risk in ALIGNMENT_RISKS],
    }


def strategy_notes() -> list[str]:
    """一组"开工前先读一遍"的经验（与 ``ALIGNMENT_RISKS`` 互补：这里是流程建议）."""
    return [
        "先对齐**明显的**维度（拒答、格式、不编造），再考虑 5% 级别的风格偏好——"
        "后者需要的标注量高一个数量级（见 pairs_for_margin 的表）。",
        "第 0 天就冻结参考模型并记它的哈希：参考模型一动，之前所有 margin 都不可比。",
        "双标 10% 的样本算 κ，低于 0.4 说明维度定义还没讲清，此时放量等于放大噪声。",
        "验证集按维度分层留出（至少每维 1 条），否则「何时停」这个问题没有数据可答。",
        "把 KL 与验证准确率一起报：**只看训练 loss 的 DPO 报告是不完整的**。",
    ]


__all__ = [
    "ALIGNMENT_RISKS",
    "DEFAULT_BETA",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_TARGET_ACCURACY",
    "Z_BY_ALPHA",
    "Z_BY_POWER",
    "SampleBudget",
    "alignment_plan",
    "annotator_agreement",
    "dimension_table",
    "interpret_kappa",
    "pairs_for_margin",
    "strategy_notes",
    "z_for_confidence",
    "z_for_power",
]
