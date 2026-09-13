"""SFT 的参考模型：一个**真的会训练**的极小 next-token 模型（M5-D2）.

本模块存在的理由只有一个：**让训练循环在离线、确定、零依赖的条件下
真的发生一次**。

如果 SFT 这一课只讲"怎么调 ``SFTTrainer``"，那么学员看到的永远是
"提交任务 → 等 40 分钟 → 看到一个 loss 数字"。而 loss 到底该从多少
降到多少、梯度长什么样、为什么分母要只数监督 token——这些**必须能被
手算验证**的东西，全都被框架盖住了。

所以这里实现一个上下文长度为 1 的 next-token 语言模型（bigram + softmax）：

    参数：  W ∈ R^{V×V}（每个上下文 token 一行 logits）、b ∈ R^V
    前向：  logits = W[ctx] + b ； probs = softmax(logits)
    损失：  L = -(1/N) Σ_{t∈S} log probs[y_t]     S = 监督位置，N = |S|
    梯度：  ∂L/∂logits_t = (probs_t - onehot(y_t)) / N
    更新：  W[ctx] -= lr * Σ ∂L/∂W[ctx]  ；  b -= lr * Σ ∂L/∂b

这个梯度是**解析求出来的**，不是数值近似：``logits = W[ctx] + b`` 对
``W[ctx]`` 与 ``b`` 的偏导都是 1，所以 ``∂L/∂logits`` 就是梯度本身。

它能验证的结论（也正是 day050 教程要讲的）：

1. **prompt 段的 -100 确实在起作用**：把同一批样本分两次训练，一次
   屏蔽 prompt、一次不屏蔽，前者的 loss 下降明显更快——因为 23% 的
   监督信号不再被浪费在"预测用户提问"上；
2. **loss 的分母只数监督 token**：手算 ``-log p`` 的平均值与实现输出
   逐位一致；
3. **训练真的在学**：训练集上的 loss 会从 ``ln(V)`` 附近（≈ 均匀分布）
   单调下降到明显更低——``ln(V)`` 是"完全瞎猜"的参考值。

它**不是**用来训出可用模型的（bigram 没有能力做任何有意义的生成）。
真实训练路径见 ``sft/hf_script.py`` 生成的 ``transformers`` + ``Trainer``
脚本；本模型的接口与那个脚本的 ``train_step`` 语义一一对应。
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from smart_research_agent.sft.encoding import IGNORE_INDEX, Batch
from smart_research_agent.sft.loss import SFTLossError, log_softmax, softmax

#: 参考模型的推荐学习率（实测标定，**不是从 7B 微调的经验值抄来的**）.
#:
#: 为什么必须单独标定：学习率**没有绝对尺度**，它由模型规模、批大小、
#: 优化器共同决定。本课程数据集上实测（30 条训练样本、9 次参数更新）::
#:
#:     lr = 2e-4  ->  loss 6.3212 -> 6.3212   降幅 0.00%   （完全学不动）
#:     lr = 2.0   ->  loss 6.3212 -> 6.2128   降幅 1.72%
#:     lr = 4.0   ->  loss 6.3212 -> 6.1070   降幅 3.39%
#:     lr = 8.0   ->  loss 6.3212 -> 5.9048   降幅 6.59%
#:     lr = 20.0  ->  loss 6.3212 -> 5.3933   降幅 14.68%
#:
#: ``2e-4`` 是 7B 全参 + AdamW 的合理量级；对 557×557、纯 SGD、每次更新
#: 在上千个监督 token 上求均值的模型，它小了四个数量级——**梯度幅度与
#: 参数量、批大小一起决定了步长**。8.0 是"9 步内肉眼可见下降"的起点，
#: 也是本课 demo 的默认值。``LR_SOFT_RANGE_REFERENCE`` 给出对应的告警区间。
REFERENCE_LEARNING_RATE = 8.0


@dataclass(frozen=True)
class ModelState:
    """模型的可序列化状态（检查点用）."""

    vocab_size: int
    #: ``V × V`` 的行主序 logits 矩阵
    weights: tuple[tuple[float, ...], ...]
    #: 长度 V 的偏置
    bias: tuple[float, ...]
    #: 已完成的参数更新次数
    updates: int

    def to_dict(self) -> dict:
        """投影为可 json.dumps 的字典."""
        return {
            "vocab_size": self.vocab_size,
            "weights": [list(row) for row in self.weights],
            "bias": list(self.bias),
            "updates": self.updates,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ModelState:
        """从 ``to_dict`` 的产物还原."""
        return cls(
            vocab_size=int(payload["vocab_size"]),
            weights=tuple(tuple(float(v) for v in row) for row in payload["weights"]),
            bias=tuple(float(v) for v in payload["bias"]),
            updates=int(payload.get("updates", 0)),
        )


class ReferenceSFTModel:
    """上下文长度 1 的 next-token 模型（真实梯度下降，非模拟）.

    梯度累积的接口与 PyTorch 的语义对齐，但**分成两步**，因为它必须能
    被训练循环显式控制：

    - ``accumulate(batch)``：前向 + 反向，把梯度**累加**到内部缓冲，返回
      ``(loss 之和, 监督 token 数)``。多次调用即多轮 micro-batch。
    - ``apply_update(learning_rate)``：把累加的梯度按 ``1/N`` 缩放到均值、
      应用一次 SGD，然后清零缓冲。

    为什么要"梯度项先累加、最后统一除以 N"而不是"每个 micro-batch 各算
    均值再加权"：**前者是精确的**（等于把整个累积窗口当成一个大 batch），
    后者在窗口内各 micro-batch 监督 token 数不等时会引入偏差。
    """

    def __init__(self, vocab_size: int, *, seed: int = 42, init_scale: float = 0.02):
        if vocab_size < 2:
            raise ValueError(f"vocab_size 至少为 2，收到 {vocab_size}")
        if init_scale <= 0:
            raise ValueError(f"init_scale 必须为正数，收到 {init_scale}")
        self._vocab_size = vocab_size
        self._seed = seed
        self._init_scale = init_scale
        self._updates = 0

        rng = random.Random(seed)
        self._weights: list[list[float]] = [
            [rng.uniform(-init_scale, init_scale) for _ in range(vocab_size)]
            for _ in range(vocab_size)
        ]
        self._bias: list[float] = [0.0] * vocab_size
        self._zero_grad()

    # ------------------------------------------------------------------ 属性
    @property
    def vocab_size(self) -> int:
        """词表大小."""
        return self._vocab_size

    @property
    def updates(self) -> int:
        """已完成的参数更新次数（= 优化器步数）."""
        return self._updates

    @property
    def uniform_loss(self) -> float:
        """完全瞎猜时的参考 loss：``ln(V)``.

        它是判断"训练有没有真的在学"的第一条基线：loss 长期停在
        ``ln(V)`` 附近，说明模型学到的分布仍接近均匀分布。
        """
        return math.log(self._vocab_size)

    # -------------------------------------------------------------- 前向 / 梯度
    def logits(self, context_token: int) -> list[float]:
        """给定上下文 token 的 logits（越界抛 ``SFTLossError``）."""
        if not 0 <= context_token < self._vocab_size:
            raise SFTLossError(
                f"上下文 token {context_token} 落在词表范围 [0, {self._vocab_size}) 之外"
            )
        row = self._weights[context_token]
        bias = self._bias
        return [row[v] + bias[v] for v in range(self._vocab_size)]

    def predict_next(self, context_token: int) -> list[float]:
        """给定上下文 token 的下一 token 概率分布（softmax 之后的 logits）."""
        return softmax(self.logits(context_token))

    def _zero_grad(self) -> None:
        """清零梯度缓冲与待处理计数."""
        size = self._vocab_size
        self._grad_weights: list[list[float]] = [[0.0] * size for _ in range(size)]
        self._grad_bias: list[float] = [0.0] * size
        self._pending_tokens = 0

    def zero_grad(self) -> None:
        """公开的清零入口（丢弃未应用的累积梯度）."""
        self._zero_grad()

    def accumulate(self, batch: Batch) -> tuple[float, int]:
        """前向 + 反向，把该批的梯度累加进缓冲，返回 ``(loss 之和, 监督 token 数)``.

        对每条样本、每个位置 ``t >= 1``：上下文是 ``input_ids[t-1]``、
        目标写在 ``labels[t]``。``labels[t] == -100`` 的位置**完全跳过**——
        这就是 SFT 的 label mask 在整个训练链路里的落点，它只在这里出现
        一次，其余所有地方（数据准备、批次组装）都只是**让这个值存在**。

        位置 ``t = 0`` 被跳过：它没有任何前文，构不成 next-token 预测任务。
        """
        loss_sum = 0.0
        count = 0
        for ids, labels in zip(batch.input_ids, batch.labels):
            if len(ids) != len(labels):  # pragma: no cover - Batch 构造处已保证等长
                raise SFTLossError(f"input_ids 与 labels 长度不一致：{len(ids)} != {len(labels)}")
            for t in range(1, len(ids)):
                target = labels[t]
                if target == IGNORE_INDEX:
                    continue
                context = ids[t - 1]
                row = self._weights[context]
                bias = self._bias
                logits = [row[v] + bias[v] for v in range(self._vocab_size)]
                log_probs = log_softmax(logits)
                if not 0 <= target < self._vocab_size:
                    raise SFTLossError(
                        f"标签 {target} 落在词表范围 [0, {self._vocab_size}) 之外"
                    )
                loss_sum += -log_probs[target]
                count += 1
                # ∂L/∂logits = (softmax(logits) - onehot(target))，此处**不除 N**：
                # N 要等整个累积窗口数完才确定（见 apply_update）
                for v in range(self._vocab_size):
                    probability = math.exp(log_probs[v])
                    indicator = 1.0 if v == target else 0.0
                    gradient = probability - indicator
                    self._grad_weights[context][v] += gradient
                    self._grad_bias[v] += gradient
        if count == 0:
            raise SFTLossError(
                "本批没有任何监督位置：请检查 labels 是否全被 ignore_index 屏蔽"
            )
        self._pending_tokens += count
        return loss_sum, count

    def apply_update(self, learning_rate: float) -> None:
        """按 ``1/N`` 缩放累加梯度并做一次 SGD，然后清零缓冲.

        ``N`` 是**自上次更新以来累积的监督 token 数**，不是样本数、也不是
        token 总数——这正是"SFT 的 loss 分母只数监督 token"在梯度层面的
        落点。用错分母会让有效学习率被悄悄缩放（见 ``loss.py`` 的说明）。
        """
        if learning_rate <= 0:
            raise SFTLossError(f"learning_rate 必须为正数，收到 {learning_rate}")
        if self._pending_tokens == 0:
            raise SFTLossError("没有待应用的梯度：apply_update 之前必须先 accumulate")
        scale = 1.0 / self._pending_tokens
        for context in range(self._vocab_size):
            grad_row = self._grad_weights[context]
            row = self._weights[context]
            for v in range(self._vocab_size):
                row[v] -= learning_rate * grad_row[v] * scale
        for v in range(self._vocab_size):
            self._bias[v] -= learning_rate * self._grad_bias[v] * scale
        self._updates += 1
        self._zero_grad()

    def step(self, batch: Batch, *, learning_rate: float) -> tuple[float, int]:
        """``accumulate`` + ``apply_update`` 的便捷组合（单批更新）."""
        loss_sum, count = self.accumulate(batch)
        self.apply_update(learning_rate)
        return loss_sum, count

    # ------------------------------------------------------------------ 评估
    def evaluate(self, batches: Iterable[Batch]) -> tuple[float, int]:
        """只前向、不更新，返回 ``(平均 loss, 监督 token 数)``.

        评估必须**不污染梯度缓冲**，否则"训练 — 评估 — 继续训练"会把评估
        的梯度混进下一次更新（这是真实框架里也偶尔出现的 bug，因为评估
        通常忘记 ``zero_grad``）。本实现的做法是评估前记下缓冲状态、
        评估后恢复——比"相信没人会忘记"更可靠。
        """
        saved_weights = [row[:] for row in self._grad_weights]
        saved_bias = self._grad_bias[:]
        saved_tokens = self._pending_tokens
        loss_sum = 0.0
        count = 0
        for batch in batches:
            batch_loss, batch_count = self.accumulate(batch)
            loss_sum += batch_loss
            count += batch_count
        self._grad_weights = saved_weights
        self._grad_bias = saved_bias
        self._pending_tokens = saved_tokens
        if count == 0:
            raise SFTLossError("评估集没有任何监督位置，loss 无定义")
        return loss_sum / count, count

    # --------------------------------------------------------------- 序列化
    def state_dict(self) -> ModelState:
        """导出可序列化状态."""
        return ModelState(
            vocab_size=self._vocab_size,
            weights=tuple(tuple(row) for row in self._weights),
            bias=tuple(self._bias),
            updates=self._updates,
        )

    def load_state(self, state: ModelState) -> None:
        """载入状态（词表大小不一致直接拒绝，避免静默错位）."""
        if state.vocab_size != self._vocab_size:
            raise SFTLossError(
                f"词表大小不一致：模型 {self._vocab_size} vs 状态 {state.vocab_size}"
            )
        self._weights = [list(row) for row in state.weights]
        self._bias = list(state.bias)
        self._updates = state.updates
        self._zero_grad()

    @classmethod
    def from_state(cls, state: ModelState) -> ReferenceSFTModel:
        """从状态重建模型（检查点恢复路径）."""
        model = cls(state.vocab_size, seed=0)
        model.load_state(state)
        return model


def top_k_next(
    model: ReferenceSFTModel, context_token: int, *, k: int = 3
) -> list[tuple[int, float]]:
    """给定上下文 token，返回概率最高的 ``k`` 个候选 ``(token_id, 概率)``.

    用途只有一个：**让"模型确实学到了东西"变得可见**。训练前后各调一次，
    概率最高的候选会从"几乎均匀分布"变成"明显偏向某个 token"。
    """
    if k <= 0:
        raise SFTLossError(f"k 必须为正整数，收到 {k}")
    probabilities = model.predict_next(context_token)
    ranked = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)
    return [(index, probabilities[index]) for index in ranked[:k]]


def token_frequencies(sequences: Sequence[Sequence[int]]) -> dict[int, int]:
    """统计一批 id 序列的 token 频次（demo 用来挑"最有信息量的上下文"）."""
    counts: dict[int, int] = {}
    for sequence in sequences:
        for token in sequence:
            counts[token] = counts.get(token, 0) + 1
    return counts
