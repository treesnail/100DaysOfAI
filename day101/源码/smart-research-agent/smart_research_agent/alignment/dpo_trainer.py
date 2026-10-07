"""在参考模型上**真的走几步 DPO**（M5-D6）.

这一课如果只讲公式，学员看到的仍然是一个 `loss` 数字。所以与 day050~day052
同一套做法：**让它在离线、确定、零依赖的条件下真的发生一次**，而且每一步
都能被手算核对。

用的是 day050 那个上下文长度为 1 的 bigram 参考模型（``ReferenceSFTModel``）。
策略与参考模型是**两个独立的对象**：

.. code-block:: text

    参考模型（reference）  冻结，永不更新，它的对数概率是隐式奖励的基准
    策略（policy）         初始化为参考模型的副本，DPO 只更新它

于是"未训练时 DPO loss = ln 2"这条起点检查成立：此时两者逐位相同，
括号里的差恰好为 0。

## 梯度是怎么来的

.. code-block:: text

    L = −log σ(β·m)              m = (l_c − l_r) − (r_c − r_r)
    ∂L/∂l_c = −β·σ(−βm) =: g     ∂L/∂l_r = +β·σ(−βm) = −g
    ∂logp/∂logits_t = onehot(y_t) − softmax(logits_t)      ← 注意方向
    ⇒ chosen 的每个位置累加 (−g)·(softmax − onehot)
      rejected 的每个位置累加 (+g)·(softmax − onehot)
    更新：θ ← θ − lr·(Σgrad) / N，N = chosen 与 rejected 参与计分的位置数之和

**上面那一行的符号是本课踩过的坑**：``softmax − onehot`` 是**交叉熵**的
梯度（``∂(−log p)/∂logits``），而 DPO 需要的是**对数概率**的梯度，两者
相差一个负号。于是"把 chosen 的梯度乘 g 直接累加"会让 p(chosen) 一路
**下降**、p(rejected) 一路上升——loss 从 ``ln 2`` 单调涨到几十，
而代码没有任何地方报错。本课实测（符号修正前）：12 步后 loss 0.693 →
0.716、验证准确率从 0 掉到 0.2；修正后方向才对。

这也解释了为什么起点检查必须写成断言：**起点 loss = ln 2 在两种符号下
都成立**（那时 margin 恰好为 0，方向还没显形），只有走几步才能看出来。

**分母仍然是"参与计分的位置数"**（day050 的纪律第三次出现）：用批数或
序列长度做分母会让有效学习率被悄悄缩放，而日志里的 ``learning_rate``
看不出任何异常。

## 只用公开 API

整个训练器**没有碰任何私有字段**：读参数用 ``state_dict()``、写参数用
``load_state()``、前向用 ``logits()``。这不是洁癖——它意味着"换一个模型
就能换一个训练对象"（day051 的 ``LoRAReferenceModel`` 同样满足这组接口），
也意味着这一课的测试不需要任何 mock。
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from smart_research_agent.alignment.objectives import (
    dpo_gradient,
    dpo_loss_from_margin,
    dpo_margin,
)
from smart_research_agent.alignment.preference import PreferencePair
from smart_research_agent.alignment.reward import (
    DEFAULT_KL_BUDGET,
    policy_kl,
    preference_accuracy,
)
from smart_research_agent.finetune_eval.metrics import FinetuneEvalError
from smart_research_agent.finetune_eval.model_probe import ContextModel
from smart_research_agent.sft.loss import softmax
from smart_research_agent.sft.reference_model import ModelState
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)


class TrainableModel(ContextModel, Protocol):
    """可训练模型的最小接口：探针的两个成员 + 读写参数.

    ``ReferenceSFTModel`` 与 ``LoRAReferenceModel`` 都满足它。刻意不声明
    "梯度缓冲"之类的成员：本训练器自己算梯度、自己写回参数，
    **因此它对模型的内部结构一无所知**。
    """

    def state_dict(self) -> ModelState:  # pragma: no cover - Protocol 声明
        """导出可序列化状态."""
        ...

    def load_state(self, state: ModelState) -> None:  # pragma: no cover - Protocol 声明
        """载入状态."""
        ...


@dataclass(frozen=True)
class DPOStepRecord:
    """一次参数更新之后的读数（训练历史的元素）.

    四个量里**只有最后两个是"留出"的信号**：

    - ``mean_margin`` 是**训练集**上的 margin，它被直接优化，**必然单调变好**；
    - ``valid_accuracy`` 是留出集上的 5 值量（本课 5 条验证对 → 0.2 一格），
      粗但不可骗；
    - ``valid_margin`` 是留出集上的**连续** margin，粒度比准确率细得多——
      本课因此把它作为"该在哪停"的主判据，准确率作为交叉验证。
    """

    step: int
    mean_margin: float
    mean_loss: float
    valid_accuracy: float
    kl: float
    learning_rate: float
    valid_margin: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典."""
        return {
            "step": self.step,
            "mean_margin": self.mean_margin,
            "mean_loss": self.mean_loss,
            "valid_accuracy": self.valid_accuracy,
            "valid_margin": self.valid_margin,
            "kl": self.kl,
            "learning_rate": self.learning_rate,
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"step {self.step:3d} | 训练 margin {self.mean_margin:+.6f} | "
            f"loss {self.mean_loss:.6f} | 验证 margin {self.valid_margin:+.6f} "
            f"准确率 {self.valid_accuracy:.4f} | KL {self.kl:.6f}"
        )


@dataclass
class DPOReport:
    """一次 DPO 训练的完整历史（**历史比最终值重要**）."""

    beta: float
    learning_rate: float
    epochs: int
    train_pairs: int
    valid_pairs: int
    records: list[DPOStepRecord] = field(default_factory=list)
    initial_loss: float = 0.0
    zero_margin_loss: float = 0.0

    @property
    def steps(self) -> int:
        """参数更新次数."""
        return len(self.records)

    @property
    def final_loss(self) -> float:
        """最后一次更新的平均 loss（空历史时为 0.0）."""
        return self.records[-1].mean_loss if self.records else 0.0

    @property
    def loss_drop(self) -> float:
        """``初始 loss − 最终 loss``（负值表示 loss 反而涨了）."""
        return self.initial_loss - self.final_loss

    @property
    def best_valid_accuracy(self) -> float:
        """历史里最高的验证准确率（早停点的候选）."""
        return max((record.valid_accuracy for record in self.records), default=0.0)

    @property
    def best_step(self) -> int:
        """验证准确率最高时的步数（平局取最早的）."""
        if not self.records:
            return 0
        return max(self.records, key=lambda record: record.valid_accuracy).step

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典."""
        return {
            "beta": self.beta,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "train_pairs": self.train_pairs,
            "valid_pairs": self.valid_pairs,
            "steps": self.steps,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "loss_drop": self.loss_drop,
            "zero_margin_loss": self.zero_margin_loss,
            "best_valid_accuracy": self.best_valid_accuracy,
            "best_step": self.best_step,
            "records": [record.to_dict() for record in self.records],
        }

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"DPO | β {self.beta} | lr {self.learning_rate} | {self.steps} 步 | "
            f"loss {self.initial_loss:.6f} → {self.final_loss:.6f}"
            f"（{self.loss_drop:+.6f}） | 最佳验证准确率 {self.best_valid_accuracy:.4f}"
            f"@step {self.best_step}"
        )


def _pair_margin(
    policy: ContextModel,
    reference: ContextModel,
    tokenizer: Any,
    pair: PreferencePair,
) -> float:
    """一条偏好对上的 margin（四个对数概率之差，不含 β）."""
    from smart_research_agent.finetune_eval.model_probe import sequence_logprob

    chosen_ids = tokenizer.encode(pair.chosen)
    rejected_ids = tokenizer.encode(pair.rejected)
    policy_chosen, _ = sequence_logprob(policy, chosen_ids)
    policy_rejected, _ = sequence_logprob(policy, rejected_ids)
    reference_chosen, _ = sequence_logprob(reference, chosen_ids)
    reference_rejected, _ = sequence_logprob(reference, rejected_ids)
    return dpo_margin(policy_chosen, policy_rejected, reference_chosen, reference_rejected)


def _accumulate_sequence_gradient(
    model: ContextModel,
    token_ids: Sequence[int],
    *,
    scale: float,
    grad_weights: list[list[float]],
    grad_bias: list[float],
) -> int:
    """把一条序列的交叉熵梯度按 ``scale`` 累加进缓冲，返回参与计分的位置数.

    逐位置做的事与 day050 的 ``ReferenceSFTModel.accumulate`` 是同一件事，
    但**梯度方向必须按"对数概率"取**：

    .. code-block:: text

        ∂(−log p(y_t))/∂logits_t = softmax(logits_t) − onehot(y_t)     ← 交叉熵
        ∂log p(y_t)/∂logits_t    = onehot(y_t) − softmax(logits_t)     ← 对数概率

    调用方传进来的 ``scale`` 是"**交叉熵梯度的系数**"。因此对 chosen 要传
    ``−g``、对 rejected 要传 ``+g``（``g`` 是 loss 对 margin 的导数）——
    写成"chosen 传 g"会让 p(chosen) 一路下降，而 loss 只是缓慢上涨，
    看起来像"学习率太小"。
    """
    count = 0
    for index in range(1, len(token_ids)):
        target = token_ids[index]
        context = token_ids[index - 1]
        logits = model.logits(context)
        probabilities = softmax(logits)
        row = grad_weights[context]
        for vocab_index in range(len(probabilities)):
            indicator = 1.0 if vocab_index == target else 0.0
            gradient = scale * (probabilities[vocab_index] - indicator)
            row[vocab_index] += gradient
            grad_bias[vocab_index] += gradient
        count += 1
    if count == 0:
        raise FinetuneEvalError(
            "序列太短（没有可计分的位置）：DPO 需要对数概率，长度为 1 的序列没有它"
        )
    return count


def dpo_step(
    policy: TrainableModel,
    reference: ContextModel,
    tokenizer: Any,
    pairs: Sequence[PreferencePair],
    *,
    beta: float,
    learning_rate: float,
) -> dict[str, Any]:
    """在一个批（这里是若干偏好对）上做**一次** DPO 参数更新.

    返回这一批的读数：平均 loss、平均 margin、参与计分的位置数、以及
    梯度系数 ``g`` 的均值。**``positions`` 就是分母**，与 day050 一样交出来。

    ``learning_rate = 0`` 是**允许**的，语义是"只测量、不更新"：起点检查
    （未训练时的 loss 应当等于 ``ln 2``）与"训练中途只读一次评估"都靠它。
    把它做成同一段代码而不是另写一个测量函数，是为了保证**测量与训练
    走的是同一条计算路径**——两条路径迟早会漂移，而漂移了不会报错。
    """
    if not pairs:
        raise FinetuneEvalError("DPO 的一个批至少要有一条偏好对")
    if learning_rate < 0:
        raise FinetuneEvalError(f"learning_rate 不能为负数，收到 {learning_rate}")
    vocab_size = policy.vocab_size
    grad_weights = [[0.0] * vocab_size for _ in range(vocab_size)]
    grad_bias = [0.0] * vocab_size
    positions = 0
    loss_sum = 0.0
    margin_sum = 0.0
    gradient_sum = 0.0
    for pair in pairs:
        margin = _pair_margin(policy, reference, tokenizer, pair)
        loss = dpo_loss_from_margin(margin, beta=beta)
        coefficient = dpo_gradient(margin, beta=beta)
        chosen_ids = tokenizer.encode(pair.chosen)
        rejected_ids = tokenizer.encode(pair.rejected)
        positions += _accumulate_sequence_gradient(
            policy, chosen_ids, scale=-coefficient, grad_weights=grad_weights,
            grad_bias=grad_bias,
        )
        positions += _accumulate_sequence_gradient(
            policy, rejected_ids, scale=coefficient, grad_weights=grad_weights,
            grad_bias=grad_bias,
        )
        loss_sum += loss
        margin_sum += margin
        gradient_sum += coefficient
    state = policy.state_dict()
    if state.vocab_size != vocab_size:  # pragma: no cover - 防御式
        raise FinetuneEvalError(
            f"模型状态词表与当前词表不一致：{state.vocab_size} != {vocab_size}"
        )
    if learning_rate > 0:
        scale = learning_rate / positions
        new_weights = tuple(
            tuple(
                state.weights[row][column] - scale * grad_weights[row][column]
                for column in range(vocab_size)
            )
            for row in range(vocab_size)
        )
        new_bias = tuple(
            state.bias[index] - scale * grad_bias[index] for index in range(vocab_size)
        )
        policy.load_state(
            ModelState(
                vocab_size=vocab_size,
                weights=new_weights,
                bias=new_bias,
                updates=state.updates + 1,
            )
        )
    count = len(pairs)
    return {
        "mean_loss": loss_sum / count,
        "mean_margin": margin_sum / count,
        "mean_gradient": gradient_sum / count,
        "positions": positions,
        # 只测量（lr=0）时 updates 不变：**"测了一次"不是"更新了一次"**，
        # 把它算成一次更新会让"这个模型训了多少步"这个数字失去意义。
        "updates": policy.state_dict().updates,
        "updated": learning_rate > 0,
    }


def train_dpo(
    policy: TrainableModel,
    reference: ContextModel,
    tokenizer: Any,
    train_pairs: Sequence[PreferencePair],
    *,
    valid_pairs: Sequence[PreferencePair],
    beta: float,
    learning_rate: float,
    epochs: int = 1,
    seed: int = 42,
    kl_budget: float = DEFAULT_KL_BUDGET,
) -> DPOReport:
    """跑一遍最小 DPO 训练，返回**逐步历史**（而不是只返回最终 loss）.

    历史里每一步都记四样东西：训练 margin、训练 loss、**留出验证准确率**、KL。
    少了最后两样，这个训练过程就只剩"loss 在降"——而 DPO 最常见的失败
    恰恰是"loss 一路降、模型一路坏"。

    ``initial_loss`` 记的是**更新之前**那一次的平均 loss：它应当等于
    ``ln 2``（策略与参考模型此时逐位相同）。这一条会被写进报告并对照
    ``ZERO_MARGIN_LOSS``——**一个能自己验证起点的训练循环，比一个只能
    报最终 loss 的循环可信得多**。
    """
    if not train_pairs:
        raise FinetuneEvalError("DPO 训练至少需要一条训练偏好对")
    if not valid_pairs:
        raise FinetuneEvalError(
            "DPO 训练至少需要一条留出偏好对：没有验证集就无法判断何时停止"
        )
    if epochs < 1:
        raise FinetuneEvalError(f"epochs 必须为正整数，收到 {epochs}")
    from smart_research_agent.alignment.objectives import ZERO_MARGIN_LOSS

    report = DPOReport(
        beta=beta,
        learning_rate=learning_rate,
        epochs=epochs,
        train_pairs=len(train_pairs),
        valid_pairs=len(valid_pairs),
        zero_margin_loss=ZERO_MARGIN_LOSS,
    )
    order = list(train_pairs)
    random.Random(seed).shuffle(order)
    initial = dpo_step(
        policy,
        reference,
        tokenizer,
        order,
        beta=beta,
        learning_rate=0.0,
    )
    report.initial_loss = initial["mean_loss"]
    step_index = 0
    for _ in range(epochs):
        for pair in order:
            step_index += 1
            result = dpo_step(
                policy,
                reference,
                tokenizer,
                [pair],
                beta=beta,
                learning_rate=learning_rate,
            )
            accuracy = preference_accuracy(
                policy, reference, tokenizer, valid_pairs, beta=beta
            )
            kl = policy_kl(policy, reference, tokenizer, train_pairs)
            record = DPOStepRecord(
                step=step_index,
                mean_margin=result["mean_margin"],
                mean_loss=result["mean_loss"],
                valid_accuracy=accuracy["accuracy"],
                valid_margin=accuracy["mean_margin"],
                kl=kl,
                learning_rate=learning_rate,
            )
            report.records.append(record)
            if step_index % 4 == 0 or step_index == 1:
                logger.info("DPO %s", record.summary_line())
    logger.info(
        "DPO 训练完成：%s | KL 预算 %s",
        report.summary_line(),
        kl_budget,
    )
    return report


__all__ = [
    "DEFAULT_KL_BUDGET",
    "DPOReport",
    "DPOStepRecord",
    "TrainableModel",
    "dpo_step",
    "train_dpo",
]
