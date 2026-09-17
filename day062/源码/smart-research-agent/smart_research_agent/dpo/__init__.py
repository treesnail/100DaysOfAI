"""DPO 对齐实践包（M5-D7）：偏好数据 → 训练配置 → 损失函数 → 对齐训练.

day054 讲清了 RLHF/DPO 的**原理**（偏好对、参考模型、隐式奖励、β 与 KL），
本课把原理落成一次**真的能跑的训练**。包内四个模块对应流水线上的四段::

    config.DPOTrainingConfig     训练配置（TRL ``DPOConfig`` 的离线镜像 + 步数算术）
        -> dataset.build_preferences()   偏好数据构造（含"金标准必须最好"体检）
        -> losses.loss_value()           三种损失函数的取值与梯度（含起点自检）
        -> (trainer / pipeline)          训练与对齐后评估

边界是刻意的，与 ``sft`` / ``peft`` 两个包保持同一条纪律：

- ``losses`` 是**纯函数**：给定 margin 与 β 就能算出 loss 与梯度，因此
  "起点 m=0 的期望 loss"可以闭式算出并用来做训练前自检；
- ``config`` 是**纯数据 + 纯算术**：``step_plan()`` 在开训前就把
  "会做多少次参数更新、什么时候评估"算准，不需要先跑一轮再数；
- ``dataset`` 只依赖磁盘上的 jsonl 与 ``finetune``/``alignment`` 的类型，
  不触碰网络——所以整条链路离线可测、可复现；
- 三个模块共用一个异常基类 ``DPOError``（继承 ``ValueError``），
  于是路由层 ``except ValueError`` 就能统一映射成 400。
"""

from __future__ import annotations

from smart_research_agent.dpo.config import (
    DEFAULT_ACCUMULATION_STEPS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_STEPS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOGGING_STEPS,
    DEFAULT_MAX_COMPLETION_LENGTH,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MAX_PROMPT_LENGTH,
    DEFAULT_PATIENCE,
    DEFAULT_SAVE_STEPS,
    DEFAULT_WARMUP_RATIO,
    OFFLINE_LEARNING_RATE,
    SNAPSHOT,
    DPOTrainingConfig,
    warmup_floor,
)
from smart_research_agent.dpo.dataset import (
    DEFAULT_SAFETY_PAIRS,
    DEFAULT_SEED_EXAMPLES,
    DEFAULT_SELECTION,
    DEGRADATION_OPS,
    MIN_PHRASE_CHARS,
    MIN_SCORE_GAP,
    OP_DIMENSION,
    OP_REASONS,
    OVERLENGTH_PENALTY,
    PROJECT_ROOT,
    SELECTION_MODES,
    TRL_COLUMNS,
    CandidateScore,
    PreferenceBuildReport,
    build_preferences,
    candidates_for,
    dataset_report,
    degrade,
    key_phrases,
    load_seed_examples,
    pick_rejected,
    read_safety_pairs,
    read_trl_dataset,
    safety_category_table,
    score_candidate,
    to_trl_rows,
    write_trl_dataset,
)
from smart_research_agent.dpo.errors import DPOError
from smart_research_agent.dpo.losses import (
    DEFAULT_LOSS_TYPE,
    GRADIENT_CHECK_STEP,
    LOSS_TYPES,
    check_loss_type,
    describe_loss,
    expected_zero_margin_tolerance,
    hinge_dpo_gradient,
    hinge_dpo_loss,
    ipo_dpo_gradient,
    ipo_dpo_loss,
    loss_gradient,
    loss_table,
    loss_value,
    numeric_gradient,
    sigmoid_dpo_gradient,
    sigmoid_dpo_loss,
    zero_margin_loss,
    zero_margin_losses,
)

__all__ = [
    "DEFAULT_ACCUMULATION_STEPS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_EVAL_STEPS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_LOGGING_STEPS",
    "DEFAULT_LOSS_TYPE",
    "DEFAULT_MAX_COMPLETION_LENGTH",
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_MAX_PROMPT_LENGTH",
    "DEFAULT_PATIENCE",
    "DEFAULT_SAFETY_PAIRS",
    "DEFAULT_SAVE_STEPS",
    "DEFAULT_SEED_EXAMPLES",
    "DEFAULT_SELECTION",
    "DEFAULT_WARMUP_RATIO",
    "DEGRADATION_OPS",
    "GRADIENT_CHECK_STEP",
    "LOSS_TYPES",
    "MIN_PHRASE_CHARS",
    "MIN_SCORE_GAP",
    "OFFLINE_LEARNING_RATE",
    "OP_DIMENSION",
    "OP_REASONS",
    "OVERLENGTH_PENALTY",
    "PROJECT_ROOT",
    "SELECTION_MODES",
    "SNAPSHOT",
    "TRL_COLUMNS",
    "CandidateScore",
    "DPOError",
    "DPOTrainingConfig",
    "PreferenceBuildReport",
    "build_preferences",
    "candidates_for",
    "check_loss_type",
    "dataset_report",
    "degrade",
    "describe_loss",
    "expected_zero_margin_tolerance",
    "hinge_dpo_gradient",
    "hinge_dpo_loss",
    "ipo_dpo_gradient",
    "ipo_dpo_loss",
    "key_phrases",
    "load_seed_examples",
    "loss_gradient",
    "loss_table",
    "loss_value",
    "numeric_gradient",
    "pick_rejected",
    "read_safety_pairs",
    "read_trl_dataset",
    "safety_category_table",
    "score_candidate",
    "sigmoid_dpo_gradient",
    "sigmoid_dpo_loss",
    "to_trl_rows",
    "warmup_floor",
    "write_trl_dataset",
    "zero_margin_loss",
    "zero_margin_losses",
]
