"""SFT 的损失函数：带屏蔽位的交叉熵与困惑度（M5-D2）.

SFT 的 loss 公式与预训练**完全相同**，唯一区别是分母：

    预训练：  L = -(1/T)   Σ_{t=1..T}  log p(y_t | y_<t)
    SFT：     L = -(1/T')  Σ_{t∈S}     log p(y_t | y_<t)      S = 监督位置集合

其中 ``T' = |S|`` 是**监督 token 数**，不是序列长度。这个分母的选择是
SFT 里最容易被写错、且错了不报错的地方：

- 若分母误用序列长度 ``T``（含 prompt 与 padding），loss 会被系统性地
  **压低**——因为分子不变而分母变大。于是"loss 从 2.7 降到 0.9"看起来
  非常漂亮，但它同时意味着"梯度被缩小了 (T'/T) 倍"，学习率实际上被
  偷偷改小了一个数量级；
- 若分母只算非 padding 位置但没排除 prompt，loss 偏小且**随 prompt
  长度波动**——同一批数据里 prompt 越长的样本被压得越狠。

所以本模块把"分母只数监督 token"写成显式契约：``masked_cross_entropy``
返回 ``(loss, supervised_count)``，**把分母交出来**，让调用方能核对它。

数值稳定性：``softmax`` / ``log_softmax`` 一律**先减去最大 logit**。
不这么做时，logit 稍大（如 100）就会让 ``exp`` 溢出为 ``inf``，而
``inf / inf`` 得到 ``nan``——训练会从"梯度爆炸"变成"梯度是 nan"，
后者更难定位。减最大值不改变数学结果（softmax 对整体平移不变），
只是把指数控制在 ``exp(0) = 1`` 以内。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from smart_research_agent.sft.encoding import IGNORE_INDEX


class SFTLossError(ValueError):
    """损失无法计算（例如没有任何监督位置、logits 与 labels 长度不一致）.

    这类错误必须**显式失败**而不是返回 0：返回 0 会让"这批数据全被屏蔽了"
    伪装成"loss 完美"，而它恰恰是最需要被发现的训练故障。
    """


def softmax(logits: Sequence[float]) -> list[float]:
    """数值稳定的 softmax：先减最大值再取指数.

    返回值之和为 1（浮点误差内）。空输入抛 ``SFTLossError``——
    "对空 logits 求分布"没有定义，返回空列表会让下游静默算出 nan。
    """
    if not logits:
        raise SFTLossError("softmax 的输入不能为空")
    largest = max(logits)
    exps = [math.exp(value - largest) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def log_softmax(logits: Sequence[float]) -> list[float]:
    """数值稳定的 log-softmax：``logits - max - log(sum(exp(logits - max)))``.

    直接用 ``math.log(softmax(x))`` 在概率极小时会下溢为 ``-inf``，
    而 ``log_softmax`` 的表达式在同样的输入下仍然给出有限值。
    """
    if not logits:
        raise SFTLossError("log_softmax 的输入不能为空")
    largest = max(logits)
    shifted = [value - largest for value in logits]
    log_total = math.log(sum(math.exp(value) for value in shifted))
    return [value - log_total for value in shifted]


def cross_entropy(logits: Sequence[float], target: int) -> float:
    """单个位置的交叉熵 ``-log p(target)``（target 越界抛 ``SFTLossError``）."""
    if not 0 <= target < len(logits):
        raise SFTLossError(f"target={target} 落在 logits 范围 [0, {len(logits)}) 之外")
    return -log_softmax(logits)[target]


def masked_cross_entropy(
    logits_seq: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[float, int]:
    """带屏蔽位的平均交叉熵，返回 ``(loss, 监督 token 数)``.

    ``logits_seq[t]`` 是对位置 ``t`` 的下一 token 预测分布，与 ``labels[t]``
    位置对齐（**不做 shift**：shift 由模型内部完成，这与 transformers 的
    约定一致——模型的 ``labels`` 与 ``input_ids`` 等长，模型自己右移）。

    分母只数非 ``ignore_index`` 的位置。没有任何监督位置时抛
    ``SFTLossError``：这是一种**必须被发现的配置错误**（要么 mask 全错、
    要么 batch 拼错），绝不能返回 0.0 让它混过去。
    """
    if len(logits_seq) != len(labels):
        raise SFTLossError(
            f"logits 与 labels 长度不一致：{len(logits_seq)} != {len(labels)}"
        )
    total = 0.0
    count = 0
    for logits, target in zip(logits_seq, labels):
        if target == ignore_index:
            continue
        total += cross_entropy(logits, target)
        count += 1
    if count == 0:
        raise SFTLossError(
            "没有任何监督位置（全部被 ignore_index 屏蔽）：请检查 label mask 的构造"
        )
    return total / count, count


def masked_token_accuracy(
    logits_seq: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[float, int]:
    """监督位置上的**下一 token 命中率**，返回 ``(accuracy, 监督 token 数)``.

    它是比 loss 更直观的训练进度指标：loss 是"对正确答案给了多少概率"，
    accuracy 是"把正确答案排到第一位了吗"。两者一起看才能区分两种情况：

    - loss 降但 accuracy 不涨 → 模型对正确答案更自信了，但还没超过最强的
      竞争者（**在早期阶段很常见，属于正常**）；
    - accuracy 涨但 loss 不降 → 统计噪声或样本太少，需要看更多步。

    注意它**不是**生成质量的度量：token 级命中率高的模型完全可能生成
    语义错误的答案（尤其在本课这种极小数据上）。
    """
    if len(logits_seq) != len(labels):
        raise SFTLossError(
            f"logits 与 labels 长度不一致：{len(logits_seq)} != {len(labels)}"
        )
    hits = 0
    count = 0
    for logits, target in zip(logits_seq, labels):
        if target == ignore_index:
            continue
        count += 1
        if max(range(len(logits)), key=lambda index: logits[index]) == target:
            hits += 1
    if count == 0:
        raise SFTLossError("没有任何监督位置，准确率无定义")
    return hits / count, count


def perplexity(loss: float) -> float:
    """困惑度 ``exp(loss)``——"模型在每一步平均在多少个候选之间犹豫".

    报告里保留它是因为**它的单位比 loss 好解释**：loss 是 nats，
    困惑度是"候选个数"。loss = 0 对应困惑度 1（完全确定），
    loss = ln(vocab_size) 对应"相当于在词表里均匀瞎猜"。
    """
    if loss < 0:
        raise SFTLossError(f"loss 不能为负数，收到 {loss}")
    # 大 loss 时 exp 会溢出；困惑度此时已经大到没有信息量，直接饱和
    if loss > 700:  # pragma: no cover - 防御式分支：exp(709) 已接近 float 上限
        return math.inf
    return math.exp(loss)


def mean(values: Sequence[float]) -> float:
    """安全求均值（空序列抛 ``SFTLossError``，不返回 0）. """
    if not values:
        raise SFTLossError("不能对空序列求均值")
    return sum(values) / len(values)
