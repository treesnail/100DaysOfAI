"""RLHF / DPO 原理（M5-D6）：把"偏好"变成可以被手算核对的目标函数.

M5 前五天的线索是一条清晰的升级路线：day048 造数据、day050 学会跑训练
循环、day051 把参数量压到 2.787%、day052 把产物做成可交付物、day053 把
"有没有变好"做成可复现的数字。今天补上最后一块——**当"更好"没有唯一答案
时怎么办**。

.. code-block:: text

    SFT    ：教"该怎么答"        → 数据是 (输入, 唯一输出)
    评估    ：判"答得对不对"      → 数据是 (输入, 参考答案 + 判定规则)
    对齐    ：选"更喜欢哪一种"    → 数据是 (同一个提问, chosen, rejected)

五层结构，每层只干一件事：

==============================  ====================================================
``preference``                  偏好数据的格式、四道体检、分层切分（chosen / rejected）
``objectives``                  RLHF（PPO）与 DPO 的目标函数、隐式奖励、KL 估计器
``reward``                      离线可算的对齐信号：偏好准确率、policy KL、GAE 优势
``dpo_trainer``                 在参考模型上**真的走几步 DPO**（只用公开 API）
``strategy``                    五个对齐维度、样本量算术、标注一致性（κ）、风险表
==============================  ====================================================

贯穿全包的三个约定：

1. **分母交出来**。``dpo_step`` 返回 ``positions``、``preference_accuracy``
   返回 ``total`` / ``correct``、``rlhf_ppo_objective`` 返回 ``kl`` 与
   ``samples``——与 day050 的 ``masked_cross_entropy``、day053 的
   ``fact_recall`` 是同一条纪律。
2. **起点可自检**。未训练时策略与参考模型逐位相同，因此 **DPO loss 必然
   等于 ``ln 2``**（``ZERO_MARGIN_LOSS``）。报告里对照它，就能发现
   "ref 或 β 接错了"这类接线错误——它们在训练日志里只表现为"loss 有点怪"。
3. **历史比最终值重要**。``train_dpo`` 返回逐步历史（训练 margin / 训练 loss /
   **留出验证准确率** / KL）。DPO 最常见的失败是"loss 一路降、模型一路坏"，
   而只看最终 loss 的报告看不见它。

边界也要说清楚：``reward`` 里的"奖励"是 DPO 的**隐式奖励**（闭式、
不需要奖励模型），它不是 PPO 里那个训练出来的奖励模型；
``objectives.objective_requirements`` 把两者的差别做成了数据。
"""

from __future__ import annotations

from smart_research_agent.alignment.dpo_trainer import (
    DEFAULT_KL_BUDGET,
    DPOReport,
    DPOStepRecord,
    TrainableModel,
    dpo_step,
    train_dpo,
)
from smart_research_agent.alignment.objectives import (
    BETA_SOFT_RANGE,
    KL_ESTIMATORS,
    ZERO_MARGIN_LOSS,
    RLHFObjective,
    dpo_gradient,
    dpo_loss,
    dpo_loss_from_margin,
    dpo_margin,
    dpo_probability,
    implicit_reward,
    implicit_reward_margin,
    kl_estimators,
    log_sigmoid,
    mean_kl,
    objective_requirements,
    reward_with_kl_shaping,
    rlhf_ppo_objective,
    sigmoid,
    softplus,
)
from smart_research_agent.alignment.preference import (
    ALIGNMENT_DIMENSIONS,
    DEFAULT_VALID_RATIO,
    DIMENSION_GOALS,
    MAX_LENGTH_RATIO,
    NEAR_DUPLICATE_NGRAM,
    NEAR_DUPLICATE_THRESHOLD,
    SEED_PAIRS,
    PreferencePair,
    dedupe_pairs,
    length_bias_report,
    near_duplicates,
    preference_fingerprint,
    preference_stats,
    read_preferences,
    split_preferences,
    validate_pairs,
    write_preferences,
)
from smart_research_agent.alignment.pipeline import (
    ZERO_MARGIN_TOLERANCE,
    AlignmentOutcome,
    model_hash,
    run_alignment,
)
from smart_research_agent.alignment.reward import (
    DEFAULT_GAMMA,
    DEFAULT_LAMBDA,
    PairReward,
    advantage_report,
    gae_advantages,
    optimization_summary,
    over_optimization_flags,
    pair_reward,
    policy_kl,
    preference_accuracy,
    reward_normalize,
    reward_table,
)
from smart_research_agent.alignment.strategy import (
    ALIGNMENT_RISKS,
    DEFAULT_BETA,
    DEFAULT_LEARNING_RATE,
    DEFAULT_TARGET_ACCURACY,
    Z_BY_ALPHA,
    Z_BY_POWER,
    SampleBudget,
    alignment_plan,
    annotator_agreement,
    dimension_table,
    interpret_kappa,
    pairs_for_margin,
    strategy_notes,
    z_for_confidence,
    z_for_power,
)

__all__ = [
    "ALIGNMENT_DIMENSIONS",
    "ALIGNMENT_RISKS",
    "BETA_SOFT_RANGE",
    "DEFAULT_BETA",
    "DEFAULT_GAMMA",
    "DEFAULT_KL_BUDGET",
    "DEFAULT_LAMBDA",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_TARGET_ACCURACY",
    "DEFAULT_VALID_RATIO",
    "DIMENSION_GOALS",
    "KL_ESTIMATORS",
    "MAX_LENGTH_RATIO",
    "NEAR_DUPLICATE_NGRAM",
    "NEAR_DUPLICATE_THRESHOLD",
    "SEED_PAIRS",
    "ZERO_MARGIN_LOSS",
    "ZERO_MARGIN_TOLERANCE",
    "Z_BY_ALPHA",
    "Z_BY_POWER",
    "AlignmentOutcome",
    "DPOReport",
    "DPOStepRecord",
    "PairReward",
    "PreferencePair",
    "RLHFObjective",
    "SampleBudget",
    "TrainableModel",
    "advantage_report",
    "alignment_plan",
    "annotator_agreement",
    "dedupe_pairs",
    "dimension_table",
    "dpo_gradient",
    "dpo_loss",
    "dpo_loss_from_margin",
    "dpo_margin",
    "dpo_probability",
    "dpo_step",
    "gae_advantages",
    "implicit_reward",
    "implicit_reward_margin",
    "interpret_kappa",
    "kl_estimators",
    "length_bias_report",
    "log_sigmoid",
    "mean_kl",
    "model_hash",
    "near_duplicates",
    "objective_requirements",
    "optimization_summary",
    "over_optimization_flags",
    "pair_reward",
    "pairs_for_margin",
    "policy_kl",
    "preference_accuracy",
    "preference_fingerprint",
    "preference_stats",
    "read_preferences",
    "reward_normalize",
    "reward_table",
    "reward_with_kl_shaping",
    "rlhf_ppo_objective",
    "run_alignment",
    "sigmoid",
    "softplus",
    "split_preferences",
    "strategy_notes",
    "train_dpo",
    "validate_pairs",
    "write_preferences",
    "z_for_confidence",
    "z_for_power",
]
