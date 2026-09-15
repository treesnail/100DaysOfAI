"""对齐流水线：一条调用串起"数据体检 → 切分 → 起点自检 → DPO → 过优化体检"（M5-D6）.

与 day053 的 ``run_finetune_evaluation`` 一样，这个模块存在的理由是
**"每一步单独跑通"不等于"串起来跑通"**。对齐这条链路有几个顺序上的硬要求，
而且它们错了都不会报错，只会给出一个看起来合理的报告：

1. **切分必须在训练之前**（``split_preferences``）——而且验证集不能为空；
2. **参考模型必须在训练之前冻结**——它一旦跟着更新，"隐式奖励"这个量的
   基准就漂了，前后两次训练的 margin 不再可比；
3. **起点自检必须在第一步之前**——策略与参考模型此时逐位相同，
   ``loss`` 必须等于 ``ln 2``；等到走完几步再检查就查不出接线错误了；
4. **过优化体检必须在训练之后**（``optimization_summary``），
   而它的输入是**逐步历史**——只留下最终 loss 的训练器做不了这件事。

本模块还额外给出三样东西，因为它们是"能不能开工"的判据：

- **偏好数据指纹**：这次对齐用的是哪一份数据；
- **参考模型哈希**：这次对齐偏离的是哪一个基准（用参数的哈希表示，
  比"我们用的是同一个模型"这句话可靠）；
- **样本量差距**：当前收集量离统计上可分辨的目标还差多少（``strategy``）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from smart_research_agent.alignment.dpo_trainer import (
    DEFAULT_KL_BUDGET,
    DPOReport,
    train_dpo,
)
from smart_research_agent.alignment.objectives import ZERO_MARGIN_LOSS
from smart_research_agent.alignment.preference import (
    DEFAULT_VALID_RATIO,
    SEED_PAIRS,
    PreferencePair,
    preference_fingerprint,
    preference_stats,
    split_preferences,
)
from smart_research_agent.alignment.reward import (
    optimization_summary,
    preference_accuracy,
)
from smart_research_agent.alignment.strategy import (
    DEFAULT_BETA,
    DEFAULT_LEARNING_RATE,
    DEFAULT_TARGET_ACCURACY,
    alignment_plan,
)
from smart_research_agent.finetune_eval.metrics import FinetuneEvalError
from smart_research_agent.sft.encoding import CharTokenizer
from smart_research_agent.sft.reference_model import ModelState, ReferenceSFTModel
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 起点自检的容差：loss 与 ``ln 2`` 的差超过它就判"接线有问题".
#:
#: 取 ``1e-9`` 而不是 0：``ln 2`` 是浮点运算的结果，同一个式子换个加法顺序
#: 会差最后一位。这个容差足够紧（比任何真实的接线错误小十几个数量级），
#: 又不会因为浮点噪声误报。
ZERO_MARGIN_TOLERANCE = 1e-9


@dataclass
class AlignmentOutcome:
    """一次完整对齐流水线的产物（所有中间证据都留着）."""

    stats: dict[str, Any]
    train_ids: list[str] = field(default_factory=list)
    valid_ids: list[str] = field(default_factory=list)
    plan: dict[str, Any] = field(default_factory=dict)
    reference_hash: str = ""
    initial_loss: float = 0.0
    zero_margin_loss: float = ZERO_MARGIN_LOSS
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    report: DPOReport | None = None
    optimization: dict[str, Any] = field(default_factory=dict)

    @property
    def initial_check_passed(self) -> bool:
        """起点自检：未训练时的 loss 是否等于 ``ln 2``."""
        return abs(self.initial_loss - ZERO_MARGIN_LOSS) <= ZERO_MARGIN_TOLERANCE

    @property
    def accuracy_gain(self) -> float:
        """留出偏好准确率的绝对提升."""
        return float(self.after.get("accuracy", 0.0)) - float(self.before.get("accuracy", 0.0))

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典."""
        return {
            "stats": self.stats,
            "train_ids": list(self.train_ids),
            "valid_ids": list(self.valid_ids),
            "plan": self.plan,
            "reference_hash": self.reference_hash,
            "initial_loss": self.initial_loss,
            "zero_margin_loss": self.zero_margin_loss,
            "initial_check_passed": self.initial_check_passed,
            "before": self.before,
            "after": self.after,
            "accuracy_gain": self.accuracy_gain,
            "report": None if self.report is None else self.report.to_dict(),
            "optimization": self.optimization,
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        if self.report is None:  # pragma: no cover - 防御式
            raise FinetuneEvalError("流水线尚未完成：report 缺失")
        return (
            f"对齐 | 偏好数据 {self.stats['fingerprint']} | 参考模型 {self.reference_hash} | "
            f"{len(self.train_ids)}/{len(self.valid_ids)} 对（训练/留出）| "
            f"起点自检 {'通过' if self.initial_check_passed else '未通过'} | "
            f"留出准确率 {self.before.get('accuracy', 0.0):.4f} → "
            f"{self.after.get('accuracy', 0.0):.4f}（{self.accuracy_gain:+.4f}）| "
            f"KL {self.report.records[-1].kl if self.report.records else 0.0:.6f}"
        )


def model_hash(state: ModelState) -> str:
    """模型参数的短哈希（前 12 位）.

    用参数本身做标识，而不是用模型名：**"我们用的是同一个参考模型"这句话
    无法被核对**，而一串哈希可以。它与 day052 的适配器哈希、day053 的
    评估集指纹是同一条纪律的第三次出现。
    """
    payload = repr((state.vocab_size, state.weights, state.bias)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def run_alignment(
    *,
    pairs: list[PreferencePair] | None = None,
    tokenizer: Any | None = None,
    reference: ReferenceSFTModel | None = None,
    valid_ratio: float = DEFAULT_VALID_RATIO,
    beta: float = DEFAULT_BETA,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    epochs: int = 6,
    seed: int = 42,
    kl_budget: float = DEFAULT_KL_BUDGET,
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
) -> AlignmentOutcome:
    """跑完整条对齐流水线（**顺序被固定在这里，不允许调用方自己拼**）.

    ``tokenizer`` 与 ``reference`` 可以注入：真实场景用它们接上自己的模型与
    分词器；不传时用本课语料现造 ``CharTokenizer`` 并新建参考模型——
    后一条路径让这个函数在离线环境里可以**一行调用跑完**。
    """
    preference_pairs = list(SEED_PAIRS) if pairs is None else list(pairs)
    stats = preference_stats(preference_pairs)
    train_pairs, valid_pairs = split_preferences(
        preference_pairs, valid_ratio=valid_ratio, seed=seed
    )
    active_tokenizer = (
        tokenizer
        if tokenizer is not None
        else CharTokenizer.from_texts(
            [
                pair.prompt + pair.chosen + pair.rejected
                for pair in preference_pairs
            ]
        )
    )
    reference_model = (
        reference
        if reference is not None
        else ReferenceSFTModel(active_tokenizer.vocab_size, seed=seed)
    )
    frozen_state = reference_model.state_dict()
    policy = ReferenceSFTModel.from_state(frozen_state)
    from smart_research_agent.alignment.dpo_trainer import dpo_step

    initial = dpo_step(
        policy,
        reference_model,
        active_tokenizer,
        train_pairs,
        beta=beta,
        learning_rate=0.0,
    )
    before = preference_accuracy(
        policy, reference_model, active_tokenizer, valid_pairs, beta=beta
    )
    report = train_dpo(
        policy,
        reference_model,
        active_tokenizer,
        train_pairs,
        valid_pairs=valid_pairs,
        beta=beta,
        learning_rate=learning_rate,
        epochs=epochs,
        seed=seed,
        kl_budget=kl_budget,
    )
    after = preference_accuracy(
        policy, reference_model, active_tokenizer, valid_pairs, beta=beta
    )
    optimization = optimization_summary(
        [record.to_dict() for record in report.records], kl_budget=kl_budget
    )
    collected = {
        dimension: sum(
            1 for pair in preference_pairs if pair.dimension == dimension
        )
        for dimension in stats["by_dimension"]
    }
    plan = alignment_plan(
        collected=collected,
        target_accuracy=target_accuracy,
        beta=beta,
        learning_rate=learning_rate,
        kl_budget=kl_budget,
    )
    outcome = AlignmentOutcome(
        stats=stats,
        train_ids=[pair.id for pair in train_pairs],
        valid_ids=[pair.id for pair in valid_pairs],
        plan=plan,
        reference_hash=model_hash(frozen_state),
        initial_loss=initial["mean_loss"],
        before=before,
        after=after,
        report=report,
        optimization=optimization,
    )
    logger.info("对齐流水线完成：%s", outcome.summary_line())
    return outcome


__all__ = [
    "ZERO_MARGIN_TOLERANCE",
    "AlignmentOutcome",
    "model_hash",
    "run_alignment",
]
