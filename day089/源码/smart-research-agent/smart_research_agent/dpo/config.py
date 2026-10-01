"""DPO 训练配置：把"要跑一次 DPO"翻译成**可以被手算核对的步数**（M5-D7）.

day050 的 ``sft/args.py``、day052 的 ``peft/accelerate.py`` 都在同一件事上
花了一整节：**先把机器要跑多少步算清楚，再去跑**。本课沿用同一条纪律，
但 DPO 多出两个只有它才有的算术：

.. code-block:: text

    micro-batch       = per_device_train_batch_size  条偏好对
    一次参数更新       = gradient_accumulation_steps 个 micro-batch
    一个 epoch 的更新次数 = (偏好对数 // micro-batch) // 累积步数
    warmup_steps      = int(total_steps * warmup_ratio)

按本课数据集实测（16 条偏好对 → 按维度分层切分后训练 12 / 留出 4）::

    train_pairs                        = 12
    micro-batches/epoch = 12 // 2       = 6
    optimizer steps/epoch = 6 // 2      = 3
    total_steps = 3 × 3 个完整 epoch     = 9
    warmup_steps = int(9 × 0.03)        = 0     ← 退化，理由见下

**``warmup_steps = 0`` 是一个必须解释的数字**，而不是"配置填错了"：9 步的
训练里，3% 是 0.27 步，取整为 0。warmup 的意义是"前若干步慢慢把学习率
升上来"，在 9 步的规模上它没有可操作的空间——**要么把 warmup_ratio 提到
``1/9 ≈ 11.11%`` 以上，要么承认这次训练不需要 warmup**。这与 day049
为 day050 预先算出的 ``warmup_steps = 0`` 是同一个结论。

## 两个学习率，不是笔误

``learning_rate`` 默认 ``5e-6``：这是**真实 TRL 脚本**里的量级（DPO 通常
比 SFT 小一个数量级，因为策略只允许相对参考模型缓慢移动）。
``offline_learning_rate`` 默认 ``0.5``：这是**本课离线参考模型**上的标定值
（day054 在同一个 bigram 模型上量出来的).

两者相差五个数量级而都"正确"，理由与 day050 的 ``REFERENCE_LEARNING_RATE``
完全相同：**学习率没有绝对尺度**，它由参数量、批大小、优化器共同决定。
拿 7B 模型的 5e-6 去跑一个 557×557 的纯 SGD 模型，结果是 loss 一位小数都不动。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from smart_research_agent.alignment.strategy import DEFAULT_BETA
from smart_research_agent.dpo.errors import DPOError
from smart_research_agent.dpo.losses import (
    DEFAULT_LOSS_TYPE,
    LOSS_TYPES,
    check_loss_type,
    loss_table,
    zero_margin_losses,
)
from smart_research_agent.peft.config import LoRAConfig

#: 真实 TRL 脚本里的学习率量级（比 SFT 小一个数量级）.
DEFAULT_LEARNING_RATE = 5e-6

#: 离线参考模型上的标定学习率（沿用 day054 在同一个小模型上的量测值）.
OFFLINE_LEARNING_RATE = 0.5

#: 其余默认值：与 day049 为 day050 预先算出的那套超参保持同源（batch 2、
#: 累积 2、3 个 epoch），这样两课的数字可以直接对照。
DEFAULT_EPOCHS = 3.0
DEFAULT_BATCH_SIZE = 2
DEFAULT_ACCUMULATION_STEPS = 2
DEFAULT_WARMUP_RATIO = 0.03
DEFAULT_MAX_LENGTH = 512
DEFAULT_MAX_PROMPT_LENGTH = 256
DEFAULT_MAX_COMPLETION_LENGTH = 256
DEFAULT_SAVE_STEPS = 8
DEFAULT_EVAL_STEPS = 4
DEFAULT_LOGGING_STEPS = 4

#: 早停耐心：连续多少个"评估点"没有刷新最佳留出 margin 就停.
#:
#: 与 day054 的过优化告警是两套东西，必须分开说：``optimization_summary``
#: 是**事后读历史**（训练跑完后判"是否已经过优化"），``patience`` 是
#: **训练中在线决策**（要不要现在就停）。前者可以看见"已经太晚了"，
#: 后者才真的省下机器时间。
DEFAULT_PATIENCE = 3

#: 配置所属的课程快照标签（写进产物与 API 响应，便于溯源）.
SNAPSHOT = "day055"


@dataclass(frozen=True)
class DPOTrainingConfig:
    """一次 DPO 训练的全部配置（TRL ``DPOConfig`` 的离线镜像 + 本课的算术）.

    ``to_dpo_config()`` 产出的键名与 TRL ``DPOConfig`` **逐字对齐**，
    因此脚本里可以直接 ``DPOConfig(**cfg.to_dpo_config())``，不需要一张
    "我们的字段 → TRL 字段"的翻译表——翻译表就是漂移的来源（day050 的
    ``to_hf_dict`` 与 day051 的 ``to_peft_dict`` 都是同一条理由）。
    """

    #: KL 约束强度（与 day054 的 ``DEFAULT_BETA`` 同源）
    beta: float = DEFAULT_BETA
    #: 损失函数：``sigmoid`` / ``hinge`` / ``ipo``（见 ``losses.loss_table``）
    loss_type: str = DEFAULT_LOSS_TYPE
    #: 真实脚本用的学习率
    learning_rate: float = DEFAULT_LEARNING_RATE
    #: 离线参考模型用的学习率（见模块文档"两个学习率"）
    offline_learning_rate: float = OFFLINE_LEARNING_RATE
    #: 训练轮数（允许小数；步数算术只计**完整 epoch**）
    num_train_epochs: float = DEFAULT_EPOCHS
    #: 单设备 micro-batch 大小（单位是"偏好对"，不是样本条数）
    per_device_train_batch_size: int = DEFAULT_BATCH_SIZE
    #: 梯度累积步数
    gradient_accumulation_steps: int = DEFAULT_ACCUMULATION_STEPS
    #: warmup 占比
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    #: 整条序列（prompt + completion）的长度上限
    max_length: int = DEFAULT_MAX_LENGTH
    #: prompt 段长度上限
    max_prompt_length: int = DEFAULT_MAX_PROMPT_LENGTH
    #: completion 段长度上限
    max_completion_length: int = DEFAULT_MAX_COMPLETION_LENGTH
    #: TRL 侧检查点保存间隔（单位：优化器步）
    save_steps: int = DEFAULT_SAVE_STEPS
    #: TRL 侧评估间隔（单位：优化器步）
    eval_steps: int = DEFAULT_EVAL_STEPS
    #: 日志间隔
    logging_steps: int = DEFAULT_LOGGING_STEPS
    #: 早停耐心（离线训练器使用）
    early_stopping_patience: int = DEFAULT_PATIENCE
    #: 标签平滑（TRL 支持；非 0 时等价于"承认标注有噪声"）
    label_smoothing: float = 0.0
    #: 产物目录
    output_dir: str = "outputs/dpo/final"
    #: 随机种子（切分、shuffle、初始化共用）
    seed: int = 42
    #: 是否用 LoRA 适配器（对应 TRL 的 ``peft_config``）
    use_lora: bool = True
    #: LoRA 配置（复用 day051 的 ``LoRAConfig``，默认值与 peft 文档一致）
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ 校验
    def validate(self) -> None:
        """把不合法的配置在**构造期**就拦下（不留给训练循环去炸）.

        每一条判据都对应一个真实的失败：
        ``beta <= 0`` 让 KL 约束消失；``loss_type`` 打错字会让 TRL 抛
        ``ValueError``（在它内部，栈很深）；``batch/accum <= 0`` 会得到
        除零或"永远不更新"；``max_prompt_length + max_completion_length``
        超过 ``max_length`` 意味着**每一批都要截断**——那不会报错，只会让
        一部分答案在训练时被悄悄切掉，而模型学到的是"答一半就停"。
        """
        check_loss_type(self.loss_type)
        if self.beta <= 0:
            raise DPOError(f"beta 必须为正数，收到 {self.beta}")
        if self.learning_rate <= 0:
            raise DPOError(f"learning_rate 必须为正数，收到 {self.learning_rate}")
        if self.offline_learning_rate <= 0:
            raise DPOError(
                f"offline_learning_rate 必须为正数，收到 {self.offline_learning_rate}"
            )
        if self.num_train_epochs <= 0:
            raise DPOError(f"num_train_epochs 必须为正数，收到 {self.num_train_epochs}")
        if self.per_device_train_batch_size <= 0:
            raise DPOError(
                f"per_device_train_batch_size 必须为正整数，收到 "
                f"{self.per_device_train_batch_size}"
            )
        if self.gradient_accumulation_steps <= 0:
            raise DPOError(
                f"gradient_accumulation_steps 必须为正整数，收到 "
                f"{self.gradient_accumulation_steps}"
            )
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise DPOError(f"warmup_ratio 必须落在 [0, 1)，收到 {self.warmup_ratio}")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise DPOError(f"label_smoothing 必须落在 [0, 1)，收到 {self.label_smoothing}")
        for name, value in (
            ("max_length", self.max_length),
            ("max_prompt_length", self.max_prompt_length),
            ("max_completion_length", self.max_completion_length),
            ("save_steps", self.save_steps),
            ("eval_steps", self.eval_steps),
            ("logging_steps", self.logging_steps),
        ):
            if value <= 0:
                raise DPOError(f"{name} 必须为正整数，收到 {value}")
        if self.max_prompt_length + self.max_completion_length > self.max_length:
            raise DPOError(
                f"max_prompt_length({self.max_prompt_length}) + "
                f"max_completion_length({self.max_completion_length}) 超过 "
                f"max_length({self.max_length})：每一批都会被截断，"
                "模型会学到『答一半就停』"
            )
        if self.early_stopping_patience < 1:
            raise DPOError(
                f"early_stopping_patience 至少为 1，收到 {self.early_stopping_patience}"
            )
        if not self.output_dir.strip():
            raise DPOError("output_dir 不能为空")
        if not isinstance(self.lora, LoRAConfig):  # pragma: no cover - 类型注解已保证
            raise DPOError(f"lora 必须是 LoRAConfig，收到 {type(self.lora).__name__}")

    # ------------------------------------------------------------------ 派生量
    @property
    def effective_batch_size(self) -> int:
        """一次参数更新实际消费的偏好对数（micro-batch × 累积）."""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    @property
    def complete_epochs(self) -> int:
        """完整 epoch 数（小数部分不产生任何参数更新，因此不计）."""
        return int(self.num_train_epochs)

    @property
    def zero_margin_loss(self) -> float:
        """本损失函数在起点（``m = 0``）的期望 loss（自检用）."""
        return zero_margin_losses(beta=self.beta)[self.loss_type]

    def step_plan(self, train_pairs: int) -> dict[str, Any]:
        """步数算术：一次训练会做多少次参数更新、什么时候评估.

        **口径必须写清楚**（否则数字对不上是必然的）：本课程按
        "每个 epoch 各自取整，再乘以完整 epoch 数"计算，与 day049 为
        day050 预先算出的数字同一口径。换成"先把总 micro-batch 数算出来
        再除以累积步数"会得到不同的结果（同一组超参：本口径 9 步，
        另一口径 11 步）——**两种口径都有人用，关键是报告里写清用的是哪种**。
        """
        if train_pairs <= 0:
            raise DPOError(f"train_pairs 必须为正整数，收到 {train_pairs}")
        if self.complete_epochs < 1:
            raise DPOError(
                f"num_train_epochs={self.num_train_epochs} 的完整 epoch 数为 0："
                "不会发生任何参数更新"
            )
        micro_per_epoch = train_pairs // self.per_device_train_batch_size
        if micro_per_epoch == 0:
            raise DPOError(
                f"训练集 {train_pairs} 条不足一个 micro-batch"
                f"（{self.per_device_train_batch_size} 条）：一个更新都不会发生"
            )
        steps_per_epoch = micro_per_epoch // self.gradient_accumulation_steps
        if steps_per_epoch == 0:
            raise DPOError(
                f"每个 epoch 只有 {micro_per_epoch} 个 micro-batch，"
                f"凑不满 {self.gradient_accumulation_steps} 次累积："
                "要么减小累积步数，要么增加数据"
            )
        total_steps = steps_per_epoch * self.complete_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        return {
            "train_pairs": train_pairs,
            "micro_batches_per_epoch": micro_per_epoch,
            "optimizer_steps_per_epoch": steps_per_epoch,
            "complete_epochs": self.complete_epochs,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "effective_batch_size": self.effective_batch_size,
            "pairs_consumed": micro_per_epoch * self.per_device_train_batch_size
            * self.complete_epochs,
            "pairs_dropped_per_epoch": train_pairs
            % self.per_device_train_batch_size,
            "evaluations": max(1, total_steps // self.eval_steps),
            "warmup_degenerate": warmup_steps == 0,
        }

    # --------------------------------------------------------------- 投影与序列化
    def to_dpo_config(self) -> dict[str, Any]:
        """投影为 TRL ``DPOConfig`` 的构造参数（键名逐字对齐）.

        三项固定值单独说明：``gradient_checkpointing=True``（显存换时间，
        对齐训练几乎总要开）、``bf16=True``、``report_to="none"``
        （本课程不依赖任何外部实验追踪服务）。
        """
        return {
            "output_dir": self.output_dir,
            "beta": self.beta,
            "loss_type": self.loss_type,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "warmup_ratio": self.warmup_ratio,
            "max_length": self.max_length,
            "max_prompt_length": self.max_prompt_length,
            "max_completion_length": self.max_completion_length,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
            "label_smoothing": self.label_smoothing,
            "seed": self.seed,
            "gradient_checkpointing": True,
            "bf16": True,
            "report_to": "none",
        }

    def to_lora_config(self) -> dict[str, Any]:
        """投影为 ``peft.LoraConfig`` 的构造参数（复用 day051 的配置对象）."""
        if not self.use_lora:
            raise DPOError("use_lora=False 时没有 LoRA 配置可投影")
        return self.lora.to_peft_dict()

    def to_dict(self) -> dict[str, Any]:
        """完整配置（含派生量），用于日志、落盘与 API 响应."""
        payload = asdict(self)
        payload["lora"] = self.lora.to_dict()
        payload["output_dir"] = self.output_dir
        return {
            **payload,
            "snapshot": SNAPSHOT,
            "loss_types": list(LOSS_TYPES),
            "effective_batch_size": self.effective_batch_size,
            "zero_margin_loss": round(self.zero_margin_loss, 6),
            "dpo_config": self.to_dpo_config(),
        }

    def plan(self, train_pairs: int) -> dict[str, Any]:
        """配置 + 步数算术 + 三种损失的起点值（开工前的完整预备）."""
        return {
            "snapshot": SNAPSHOT,
            "loss_type": self.loss_type,
            "beta": self.beta,
            "loss_table": loss_table(beta=self.beta),
            "zero_margin_losses": {
                name: round(value, 6)
                for name, value in zero_margin_losses(beta=self.beta).items()
            },
            "steps": self.step_plan(train_pairs),
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"DPO 配置 | loss={self.loss_type} | β={self.beta} | lr={self.learning_rate:g}"
            f"（离线 {self.offline_learning_rate:g}） | "
            f"{self.complete_epochs} epoch × batch {self.per_device_train_batch_size} "
            f"× 累积 {self.gradient_accumulation_steps} "
            f"= 有效批 {self.effective_batch_size} 对/更新"
        )


def warmup_floor(total_steps: int) -> float:
    """让 warmup 真正生效所需的**最小** ``warmup_ratio``：``1 / total_steps``.

    它把"``warmup_steps = 0`` 是不是配置写错了"变成一个可以回答的问题：
    本课 9 步对应的下限是 ``1/9 ≈ 11.11%``，而默认的 3% 必然退化成 0。
    """
    if total_steps <= 0:
        raise DPOError(f"total_steps 必须为正整数，收到 {total_steps}")
    return 1.0 / total_steps


__all__ = [
    "DEFAULT_ACCUMULATION_STEPS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_EVAL_STEPS",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_LOGGING_STEPS",
    "DEFAULT_MAX_COMPLETION_LENGTH",
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_MAX_PROMPT_LENGTH",
    "DEFAULT_PATIENCE",
    "DEFAULT_SAVE_STEPS",
    "DEFAULT_WARMUP_RATIO",
    "OFFLINE_LEARNING_RATE",
    "SNAPSHOT",
    "DPOTrainingConfig",
    "warmup_floor",
]
