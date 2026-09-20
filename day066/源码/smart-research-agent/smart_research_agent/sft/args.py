"""SFT 训练超参与派生量（M5-D2）.

微调脚本里最容易出错的不是模型代码，而是**超参之间的算术关系**：
学习率、批大小、梯度累积、epoch、warmup、评估/保存间隔，这七个数字
互相约束，写错一个不会报错，只会让训练"看起来在跑，其实没学到东西"。

本模块把这件事做成两件可测试的事：

1. **``SFTTrainingArgs``**：一份超参对象，字段名与 HuggingFace
   ``TrainingArguments`` **逐字对齐**（``learning_rate`` /
   ``per_device_train_batch_size`` / ``gradient_accumulation_steps`` /
   ``num_train_epochs`` / ``warmup_ratio`` / ``lr_scheduler_type`` /
   ``max_grad_norm`` / ``eval_strategy`` …）。对齐的理由很实际：
   ``to_hf_dict()`` 的输出可以**直接展开成 ``TrainingArguments(**kwargs)``**，
   不需要一张"翻译表"——而翻译表就是漂移的来源。
2. **``plan_training``**：把"这批数据 + 这组超参"会得到什么**算出来**，
   并把判据写成 ``warnings``（而不是散落在注释里）。day049 已经证明
   这些警告是真会命中的：``total_steps = 9`` 时 ``warmup_ratio = 0.03``
   必然取整为 0。

学习率调度按 HF ``get_{linear,cosine}_schedule_with_warmup`` 的分段定义
实现（warmup 段线性升到峰值、其后半周期余弦/线性降到 0），因此
``lr_at`` 的输出可以与训练日志里的 ``learning_rate`` 逐点对照。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

#: HuggingFace ``SchedulerType`` 的取值全集（``lr_at`` 只实现前四种，
#: 其余交给 transformers 自己的 ``get_scheduler``）
SCHEDULER_TYPES: tuple[str, ...] = (
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
)

#: ``lr_at`` 本课实现了调度曲线的调度器（其余类型会抛 ``SFTConfigError``）
IMPLEMENTED_SCHEDULERS: tuple[str, ...] = (
    "linear",
    "cosine",
    "constant",
    "constant_with_warmup",
)

#: ``eval_strategy`` / ``save_strategy`` 的取值（与 HF 一致）
STRATEGY_STEPS = "steps"
STRATEGY_EPOCH = "epoch"
STRATEGY_NO = "no"
STRATEGIES: tuple[str, ...] = (STRATEGY_STEPS, STRATEGY_EPOCH, STRATEGY_NO)

#: 全参 SFT 常用的学习率经验区间（用于生成告警，不是硬约束）
LR_SOFT_RANGE: tuple[float, float] = (1e-5, 5e-4)
#: LoRA/PEFT 常用的学习率经验区间（只训新增参数，可以更大）
LR_SOFT_RANGE_PEFT: tuple[float, float] = (1e-4, 5e-4)
#: **参考模型**（``ReferenceSFTModel``，557×557 的 bigram + 纯 SGD）的经验区间.
#:
#: 它与 ``LR_SOFT_RANGE`` 相差约四个数量级，这不是"经验不准"，而是
#: **学习率没有绝对尺度**：它必须与模型规模、批大小、优化器一起标定。
#: 7B 全参 + AdamW 的合理量级是 1e-5 ~ 5e-4；而一个 557×557、用纯 SGD、
#: 每次更新要在一千多个监督 token 上求均值的模型，合理量级是 1 ~ 50。
#: 把这条差异显式写成第二个区间，是为了让"告警阈值必须与模型规模匹配"
#: 这件事**有代码可依**——否则一条针对 7B 的告警会在参考模型上永远误报，
#: 而**天天误报的告警一定会被忽略**（与 day046 性能基线判定下限同一条纪律）。
LR_SOFT_RANGE_REFERENCE: tuple[float, float] = (1.0, 50.0)

#: "总步数少于这个值时训练几乎无效"的经验阈值（day049 实测 total_steps = 9）
MIN_MEANINGFUL_STEPS = 20


class SFTConfigError(ValueError):
    """超参非法（继承 ``ValueError``，便于调用方按类型捕获）.

    非法值一律在这里拒绝，**不留到训练循环里报错**：训练可能跑在远程
    机器上，在那里才发现 ``learning_rate = 0`` 是最昂贵的失败方式。
    """


@dataclass
class SFTTrainingArgs:
    """一份 SFT 超参（字段名与 ``TrainingArguments`` 逐字对齐）.

    缺省值取自本课程的数据规模与常见 SFT 实践（全参微调 7B 量级）：

    - ``learning_rate = 2e-4``：全参 SFT 偏小、LoRA 偏小-适中，是一个
      两边都不会立刻发散的保守值（``LR_SOFT_RANGE`` 给出判断区间）；
    - ``gradient_accumulation_steps = 4`` 与 ``per_device_train_batch_size = 2``
      组合出有效批 8 条——**在小数据集上这是唯一能把批大小做大的手段**
      （单卡显存只够 batch=2，靠累积换批大小）；
    - ``warmup_ratio = 0.03`` 是社区常用值，但 day049 已经算过它在
      ``total_steps = 9`` 时会退化为 0——所以 ``plan_training`` 会告警。
    """

    #: 产物目录（HF 同名参数）
    output_dir: str = "outputs/sft"
    #: 训练轮数（可为小数，HF 支持 ``num_train_epochs=0.5``）
    num_train_epochs: float = 3.0
    #: 单设备训练批大小（显存的主要决定项）
    per_device_train_batch_size: int = 2
    #: 单设备评估批大小（可大于训练批：评估不存梯度与优化器状态）
    per_device_eval_batch_size: int = 4
    #: 梯度累积步数：攒够这么多个 micro-batch 才做一次参数更新
    gradient_accumulation_steps: int = 4
    #: 学习率（与有效批大小是一对需要联调的参数）
    learning_rate: float = 2e-4
    #: 权重衰减（AdamW 的 L2 项；SFT 常用 0.01）
    weight_decay: float = 0.01
    #: warmup 占总步数的比例（取整为步数；见 ``plan_training`` 的告警）
    warmup_ratio: float = 0.03
    #: 学习率调度器类型（HF ``SchedulerType`` 的字符串取值）
    lr_scheduler_type: str = "cosine"
    #: 梯度裁剪的最大范数（0 表示不裁剪）
    max_grad_norm: float = 1.0
    #: 每个 epoch 之前是否打乱训练数据（**必须**，否则同分布样本连续出现）
    shuffle: bool = True
    #: 打乱顺序的随机种子（与 ``seed`` 分开，便于"换数据顺序"的对照实验）
    data_seed: int = 42
    #: 全局随机种子（权重初始化等）
    seed: int = 42
    #: 单条样本的最大 token 数（HF ``TrainingArguments`` 无此项，属 TRL
    #: ``SFTConfig``；放在本对象里是因为它决定"答案能不能放下"）
    #:
    #: 缺省 384 不是拍脑袋来的：本课程数据集**渲染并分词之后**的实测长度
    #: 分布是 ``min=219 / p50=235 / p90=276 / p95=302 / max=320``，向上取整
    #: 到 32 的倍数即 320；给一点余量取 384。**注意它比 day048 画像里的
    #: 84.24 token/条 大了近三倍**——因为渲染加上了系统提示词与角色标记，
    #: 而字符级分词让中文约 1 字 1 token。用 ``suggest_max_length()`` 可以
    #: 从数据直接算出来，不要照抄别人的值。
    max_length: int = 384
    #: 截断策略：``keep-answer`` / ``head``（见 ``sft.encoding``）
    truncation: str = "keep-answer"
    #: 每多少步记录一次日志
    logging_steps: int = 10
    #: 评估策略：``steps`` / ``epoch`` / ``no``
    eval_strategy: str = STRATEGY_NO
    #: 每多少步评估一次（仅 ``eval_strategy = "steps"`` 时生效）
    eval_steps: int = 50
    #: 保存策略：``steps`` / ``epoch`` / ``no``
    save_strategy: str = STRATEGY_NO
    #: 每多少步保存一次（仅 ``save_strategy = "steps"`` 时生效）
    save_steps: int = 50
    #: 最多保留多少个检查点（``None`` 表示不限制）
    save_total_limit: int | None = 2
    #: 是否用 bf16 混合精度（Ampere 及之后推荐；与 ``fp16`` 互斥）
    bf16: bool = False
    #: 是否用 fp16 混合精度（与 ``bf16`` 互斥）
    fp16: bool = False
    #: 是否启用梯度检查点（用时间换显存；长序列训练常用）
    gradient_checkpointing: bool = False
    #: 优化器实现名（HF ``optim`` 的取值，如 ``adamw_torch``）
    optim: str = "adamw_torch"
    #: 是否按长度分组采样——**本课刻意缺省关闭**：它会改变 batch 组成，
    #: 让"第 N 步看到哪些样本"不可复现；小数据集上收益也很小
    group_by_length: bool = False
    #: 是否移除数据集中未被模型签名使用的列（传入预编码数据集时必须关掉）
    remove_unused_columns: bool = False

    def validate(self) -> None:
        """硬校验：任何非法取值都在这里抛出 ``SFTConfigError``."""
        if not self.output_dir.strip():
            raise SFTConfigError("output_dir 不能为空：没有产物目录就无法保存检查点")
        if self.num_train_epochs <= 0:
            raise SFTConfigError(f"num_train_epochs 必须为正数，收到 {self.num_train_epochs}")
        if self.per_device_train_batch_size <= 0:
            raise SFTConfigError(
                f"per_device_train_batch_size 必须为正整数，收到 {self.per_device_train_batch_size}"
            )
        if self.per_device_eval_batch_size <= 0:
            raise SFTConfigError(
                f"per_device_eval_batch_size 必须为正整数，收到 {self.per_device_eval_batch_size}"
            )
        if self.gradient_accumulation_steps <= 0:
            raise SFTConfigError(
                f"gradient_accumulation_steps 必须为正整数，收到 {self.gradient_accumulation_steps}"
            )
        if self.learning_rate <= 0:
            raise SFTConfigError(f"learning_rate 必须为正数，收到 {self.learning_rate}")
        if self.weight_decay < 0:
            raise SFTConfigError(f"weight_decay 不能为负数，收到 {self.weight_decay}")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise SFTConfigError(
                f"warmup_ratio 必须落在 [0, 1) 区间，收到 {self.warmup_ratio}"
            )
        if self.max_grad_norm < 0:
            raise SFTConfigError(f"max_grad_norm 不能为负数，收到 {self.max_grad_norm}")
        if self.max_length < 2:
            raise SFTConfigError(
                f"max_length 至少为 2（1 个上下文 + 1 个监督 token），收到 {self.max_length}"
            )
        if self.lr_scheduler_type not in SCHEDULER_TYPES:
            raise SFTConfigError(
                f"未知的 lr_scheduler_type {self.lr_scheduler_type!r}，"
                f"可选：{', '.join(SCHEDULER_TYPES)}"
            )
        strategies = (("eval_strategy", self.eval_strategy), ("save_strategy", self.save_strategy))
        for name, value in strategies:
            if value not in STRATEGIES:
                raise SFTConfigError(
                    f"未知的 {name} {value!r}，可选：{', '.join(STRATEGIES)}"
                )
        if self.logging_steps <= 0:
            raise SFTConfigError(f"logging_steps 必须为正整数，收到 {self.logging_steps}")
        if self.eval_steps <= 0 or self.save_steps <= 0:
            raise SFTConfigError("eval_steps / save_steps 必须为正整数")
        if self.save_total_limit is not None and self.save_total_limit <= 0:
            raise SFTConfigError("save_total_limit 为 None 或正整数（0 无意义）")
        if self.bf16 and self.fp16:
            raise SFTConfigError(
                "bf16 与 fp16 不能同时启用：两者会争用同一套 autocast 上下文，"
                "必须二选一"
            )
        if self.truncation not in ("keep-answer", "head"):
            raise SFTConfigError(
                f"未知的截断策略 {self.truncation!r}，可选：keep-answer, head"
            )

    @property
    def effective_batch_size(self) -> int:
        """有效批大小 = ``per_device_train_batch_size × gradient_accumulation_steps``.

        单卡场景下即"一次参数更新用了多少条样本"。它是与学习率联调的那个
        量：**批大小翻倍通常意味着学习率可以同步放大**（线性缩放经验法则），
        反之则要收小。
        """
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def micro_batches_per_epoch(self, train_size: int) -> int:
        """一个 epoch 里的 micro-batch 数（丢弃不满一批的尾巴）."""
        return train_size // self.per_device_train_batch_size

    def steps_per_epoch(self, train_size: int) -> int:
        """一个 epoch 里的**优化器更新次数**（micro-batch 数 // 累积步数）.

        口径是 ``drop_last``：不足一个累积窗口的尾部 micro-batch 丢掉，
        这样"步数"有纯算术定义，与 day049 预先算出的数字一致。
        ``plan_training`` 会同时给出"不丢尾巴"的口径供对照。
        """
        return self.micro_batches_per_epoch(train_size) // self.gradient_accumulation_steps

    def total_steps(self, train_size: int) -> int:
        """总优化器步数（下界 1：即使数据极少也要至少更新一次）."""
        return max(1, int(self.steps_per_epoch(train_size) * self.num_train_epochs))

    def warmup_steps(self, train_size: int) -> int:
        """warmup 步数 = ``int(total_steps × warmup_ratio)``（与 HF 同样的取整）."""
        return int(self.total_steps(train_size) * self.warmup_ratio)

    def lr_at(self, step: int, total_steps: int) -> float:
        """第 ``step`` 步的学习率（HF ``get_*_schedule_with_warmup`` 的分段定义）.

        分段（``warmup`` 段线性升到峰值，其后按调度器衰减）::

            step <  warmup :  lr * step / warmup
            warmup <= step :  lr * decay(progress)
            progress = (step - warmup) / max(1, total_steps - warmup)

        - ``cosine``：``0.5 * (1 + cos(pi * progress))``（半周期余弦）
        - ``linear``：``1 - progress``
        - ``constant``：恒为 1（warmup 段仍然线性升，这与 HF 一致）
        - ``constant_with_warmup``：等同 ``constant`` 实现

        注意 ``step = 0`` 且 ``warmup > 0`` 时返回 **0.0**。这不是 bug：
        ``torch.optim.lr_scheduler.LambdaLR`` 在**构造时就会求值一次**
        ``lr_lambda(0)``（``last_epoch=-1`` 的初始化行为），因此第一次
        参数更新用到的学习率确实是 0——也就是说 **warmup 段的第 0 步是
        一次空更新**。训练循环必须照常执行它并清零梯度，否则这段梯度
        会漏进下一个窗口。
        """
        if total_steps <= 0:
            raise SFTConfigError(f"total_steps 必须为正整数，收到 {total_steps}")
        if self.lr_scheduler_type not in IMPLEMENTED_SCHEDULERS:
            raise SFTConfigError(
                f"本课的参考实现未覆盖调度器 {self.lr_scheduler_type!r}，"
                f"只实现了 {', '.join(IMPLEMENTED_SCHEDULERS)}；其余请交给 "
                "transformers 的 get_scheduler"
            )
        step = max(0, min(step, total_steps))
        warmup = self.warmup_from_steps(total_steps)
        if warmup > 0 and step < warmup:
            return self.learning_rate * step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(1.0, max(0.0, progress))
        if self.lr_scheduler_type == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        elif self.lr_scheduler_type == "linear":
            factor = 1.0 - progress
        else:  # constant / constant_with_warmup
            factor = 1.0
        return self.learning_rate * factor

    def warmup_from_steps(self, total_steps: int) -> int:
        """按比例换算 warmup 步数（供 ``lr_at`` 复用，避免两处取整不一致）."""
        return int(total_steps * self.warmup_ratio)

    def to_hf_dict(self) -> dict[str, Any]:
        """投影为 ``transformers.TrainingArguments`` 的构造参数.

        字段名与 HF 逐字对齐，因此可以::

            from transformers import TrainingArguments
            training_args = TrainingArguments(**args.to_hf_dict())

        刻意**不包含** ``max_length`` / ``truncation``：前者属于 TRL
        ``SFTConfig``（且 ``TrainingArguments`` 不认这个参数，传进去会抛
        ``TypeError``），后者是本课的数据准备参数，与训练器无关。
        ``eval_strategy`` 用的是 HF 4.41 之后的名字——旧名
        ``evaluation_strategy`` 已废弃并在 v4.46 移除。
        """
        return {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "lr_scheduler_type": self.lr_scheduler_type,
            "max_grad_norm": self.max_grad_norm,
            "logging_steps": self.logging_steps,
            "eval_strategy": self.eval_strategy,
            "eval_steps": self.eval_steps,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "optim": self.optim,
            "seed": self.seed,
            "data_seed": self.data_seed,
            "group_by_length": self.group_by_length,
            "remove_unused_columns": self.remove_unused_columns,
        }

    def to_trl_dict(self) -> dict[str, Any]:
        """投影为 TRL ``SFTConfig`` 的构造参数（= HF 参数 + 三个 SFT 专属项）.

        三个专属项在本课里都有对应实现，且语义已对齐官方文档：

        - ``max_length``：TRL 从 ``max_seq_length`` 改名而来，指"单条样本
          的最大 token 数"，超过即截断；
        - ``packing``：**本课缺省关闭**。它把多条短样本拼进同一条序列以
          提高吞吐，但会改变样本边界与 loss 的分母口径，让"每步看到什么"
          不再可复现——小数据集上收益远小于可解释性损失；
        - ``assistant_only_loss``：只在 assistant 消息上算 loss，正是本课
          label mask（prompt 段置 -100）在 TRL 里的等价开关。
        """
        payload = self.to_hf_dict()
        payload.update(
            {
                "max_length": self.max_length,
                "packing": False,
                "completion_only_loss": True,
                "assistant_only_loss": True,
            }
        )
        return payload

    def to_dict(self) -> dict[str, Any]:
        """完整超参的字典（含不属于 HF 的字段），用于日志与检查点落盘."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SFTTrainingArgs:
        """从 ``to_dict`` 的产物还原（未知键忽略，便于向前兼容）."""
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in payload.items() if key in known})

    def with_overrides(self, **changes: Any) -> SFTTrainingArgs:
        """返回一份改了若干字段的新对象（不改原对象）.

        与 day048 ``TrainingExample`` 用 ``dataclasses.replace`` 生成清洗后
        新样本是同一种纪律：**原始超参永远可回溯**，实验记录才对得上。
        """
        return replace(self, **changes)


@dataclass
class TrainingPlan:
    """"这批数据 + 这组超参"会得到什么：一次训练的全部派生量.

    ``warnings`` 与 ``steps`` 分开放，理由与 day048 ``recommend_method``
    的 ``reasons`` / ``warnings`` 相同：一个描述"会怎样"，一个描述
    "还要注意什么"。
    """

    train_size: int
    eval_size: int
    micro_batches_per_epoch: int
    #: drop_last 口径（本课的训练循环采用）：尾部落不满一个累积窗口时丢弃
    steps_per_epoch: int
    total_steps: int
    #: 不丢尾巴口径（HF Trainer 会 flush 尾部梯度）：``ceil(micro / accum)``
    steps_per_epoch_flushing: int
    total_steps_flushing: int
    warmup_steps: int
    effective_batch_size: int
    logging_steps: int
    eval_steps: int
    save_steps: int
    epoch_logging_times: int
    epoch_eval_times: int
    epoch_save_times: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """投影为可直接 json.dumps 的字典（API 响应直接用）."""
        return asdict(self)


def plan_training(
    args: SFTTrainingArgs,
    *,
    train_size: int,
    eval_size: int = 0,
    dataset_avg_tokens: float | None = None,
    dataset_max_tokens: float | None = None,
    lr_range: tuple[float, float] | None = None,
) -> TrainingPlan:
    """算出一次训练的全部派生量，并把风险写成 ``warnings``.

    ``warnings`` 里每一条都对应一个**真实会发生的失败模式**（括号内是
    day049 在本课程数据集上实测命中的结果）：

    1. drop_last 口径下每个 epoch 有 0 次更新（累积窗口比数据还大）；
    2. 总步数过少（``total_steps = 9``）——训练几乎不产生有效变化；
    3. warmup 比例退化（``warmup_ratio = 0.03`` 在 ``total_steps = 9`` 下
       取整为 0）——warmup 不生效，学习率从第 1 步就接近峰值；
    4. ``eval_steps`` / ``save_steps`` 大于总步数——训练过程中永远不会
       触发评估/保存，只有最后一刻才有结果；
    5. 未提供评估集——无法判断是否过拟合（这是 SFT 最常见的自欺）；
    6. 学习率超出经验区间——容易发散或灾难性遗忘；
    7. ``max_length`` 与真实长度分布不匹配（太小 → 截断答案；太大 → 纯浪费）；
    8. 数据量小于 500 条——与 day048 的选型规则同一条经验。

    ``lr_range`` 缺省取 ``LR_SOFT_RANGE``（7B 全参量级）；跑参考模型时
    传 ``LR_SOFT_RANGE_REFERENCE``——**告警阈值必须与模型规模匹配**，
    否则它只会变成噪声。
    """
    if train_size <= 0:
        raise SFTConfigError(f"train_size 必须为正整数，收到 {train_size}")
    if eval_size < 0:
        raise SFTConfigError(f"eval_size 不能为负数，收到 {eval_size}")
    args.validate()
    effective_lr_range = lr_range if lr_range is not None else LR_SOFT_RANGE

    micro = args.micro_batches_per_epoch(train_size)
    per_epoch = args.steps_per_epoch(train_size)
    total = args.total_steps(train_size)
    per_epoch_flushing = math.ceil(micro / args.gradient_accumulation_steps) if micro else 0
    total_flushing = max(1, int(per_epoch_flushing * args.num_train_epochs))
    warmup = args.warmup_steps(train_size)

    logging_steps = args.logging_steps
    eval_steps = args.eval_steps if args.eval_strategy == STRATEGY_STEPS else 0
    save_steps = args.save_steps if args.save_strategy == STRATEGY_STEPS else 0

    epoch_logging_times = (
        per_epoch // logging_steps if logging_steps and args.logging_steps <= total else 0
    )
    epoch_eval_times = per_epoch // eval_steps if eval_steps else 0
    epoch_save_times = per_epoch // save_steps if save_steps else 0

    warnings: list[str] = []
    if per_epoch == 0:
        warnings.append(
            f"drop_last 口径下每个 epoch 有 0 次参数更新：{micro} 个 micro-batch 不足一个累积窗口"
            f"（gradient_accumulation_steps={args.gradient_accumulation_steps}）。"
            "请调小累积步数、调小批大小，或补充训练数据。"
        )
    if total < MIN_MEANINGFUL_STEPS:
        warnings.append(
            f"总优化器步数仅 {total} 步（< {MIN_MEANINGFUL_STEPS}）：训练几乎不会产生有效变化。"
            "这通常意味着数据集太小或 batch 太大——先补数据，而不是调学习率。"
        )
    if args.warmup_ratio > 0 and warmup == 0:
        floor = 1 / total
        warnings.append(
            f"warmup_ratio={args.warmup_ratio} 在 total_steps={total} 下取整为 0：warmup "
            f"不会生效，学习率从第 1 步就接近峰值。要让 warmup 至少 1 步，比例需 ≥ "
            f"1/{total} ≈ {floor:.2%}，或改用更小的学习率。"
        )
    if eval_steps and eval_steps > total:
        warnings.append(
            f"eval_steps={eval_steps} 大于 total_steps={total}：训练过程中不会执行任何评估，"
            "eval_loss 只在末尾可得，无法观察过拟合的起点。"
        )
    if save_steps and save_steps > total:
        warnings.append(
            f"save_steps={save_steps} 大于 total_steps={total}：训练过程中不会保存中间检查点，"
            "中断即前功尽弃。"
        )
    if eval_size == 0:
        warnings.append(
            "未提供评估集（eval_size=0）：无法判断模型是否过拟合，也无法比较两次微调的优劣。"
        )
    low, high = effective_lr_range
    if not low <= args.learning_rate <= high:
        warnings.append(
            f"learning_rate={args.learning_rate:g} 超出经验区间 [{low:g}, {high:g}]："
            "偏小会学不动（loss 曲线几乎水平），偏大会发散或灾难性遗忘。"
            "注意经验区间是**按模型规模标定**的，换模型必须重新标定。"
        )
    if dataset_max_tokens is not None and dataset_max_tokens > args.max_length:
        warnings.append(
            f"max_length={args.max_length} 小于渲染后最长样本的 {dataset_max_tokens:.0f} token："
            "会有样本被截断（keep-answer 策略下答案放不下会直接报错）。"
            "请用 suggest_max_length() 按长度分位数定值。"
        )
    padding_blown = (
        dataset_avg_tokens is not None
        and dataset_avg_tokens > 0
        and args.max_length > dataset_avg_tokens * 4
    )
    if padding_blown:
        waste = 1 - dataset_avg_tokens / args.max_length
        warnings.append(
            f"max_length={args.max_length} 远大于数据平均长度 {dataset_avg_tokens:.1f} token："
            f"约 {waste:.1%} 的位置是 padding，而 padding 同样参与注意力计算（平方复杂度）。"
            "建议按长度分位数取值。"
        )
    if train_size < 500:
        warnings.append(
            f"训练样本仅 {train_size} 条：与 day048 的选型规则一致，"
            "500 条以下先补数据比换方法有效。"
        )
    if per_epoch_flushing != per_epoch:
        warnings.append(
            f"两种口径的步数不同：drop_last（本课训练循环）= {total} 步，"
            f"flush 尾部（HF Trainer 的默认行为）= {total_flushing} 步。"
            "对照实验时必须说明用的是哪一种，否则「步数」不可比。"
        )

    return TrainingPlan(
        train_size=train_size,
        eval_size=eval_size,
        micro_batches_per_epoch=micro,
        steps_per_epoch=per_epoch,
        total_steps=total,
        steps_per_epoch_flushing=per_epoch_flushing,
        total_steps_flushing=total_flushing,
        warmup_steps=warmup,
        effective_batch_size=args.effective_batch_size,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        save_steps=save_steps,
        epoch_logging_times=epoch_logging_times,
        epoch_eval_times=epoch_eval_times,
        epoch_save_times=epoch_save_times,
        warnings=warnings,
    )


def lr_curve(args: SFTTrainingArgs, train_size: int, *, points: int = 6) -> list[tuple[int, float]]:
    """采样学习率曲线（用于 demo 打印与测试断言）.

    返回 ``[(step, lr), ...]``，覆盖 ``0`` 到 ``total_steps`` 的 ``points``
    个等分点（含两端）。曲线形状是判断"调度器接对了没有"最直观的方式：
    cosine 应当先升后单调降、linear 应当先升后线性降、constant 应当先升后持平。
    """
    if points < 2:
        raise SFTConfigError(f"points 至少为 2（要含两端），收到 {points}")
    total = args.total_steps(train_size)
    steps = sorted({round(total * i / (points - 1)) for i in range(points)})
    return [(step, args.lr_at(step, total)) for step in steps]
