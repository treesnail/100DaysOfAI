"""模型侧探针：用**真的会训练**的参考模型给出微调前后的内部信号（M5-D5）.

``metrics.py`` 量的是"输出文本像不像",它是**黑盒**视角：只要模型给出同一段
文本就得同一个分数。微调评估还缺一个**白盒**视角——模型对领域答案到底
"熟不熟"。这两件事经常背离，而且背离的方式很有信息量：

- 词面指标涨、模型困惑度不降 → 输出变好看是因为温度 / 提示词变了，
  不是权重变了；
- 困惑度降、词面指标不涨 → 模型学会了领域用词，但还没学会"把事实说全"。

本模块提供两个可以从真实模型上算出来的量：

.. code-block:: text

    逐 token 平均对数概率  mean_logprob = (1/N) Σ_{t>=1} log p(y_t | y_{t-1})
    困惑度                perplexity  = exp(-mean_logprob)
    下一 token 命中率      accuracy    = #(argmax logits == y_t) / N

它们都能**手算核对**，因为参考模型是上下文长度为 1 的 bigram（day050）：
``p(·|y_{t-1})`` 就是 ``softmax(W[y_{t-1}] + b)``，一行算术。

模型侧探针与文本侧指标的另一个区别是**它必须用同一套词表**：``CharTokenizer``
的词表是从语料现造的，换一份语料就换一套 id。用 A 语料的 tokenizer 去
探 B 模型，得到的不是"模型不熟"，而是**毫无意义的数字**——所以这里对
越界 id 直接拒绝，而不是让它静默算出别的东西。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from smart_research_agent.finetune_eval.metrics import FinetuneEvalError
from smart_research_agent.finetune_eval.suites import EvalItem
from smart_research_agent.sft.encoding import IGNORE_INDEX, Batch
from smart_research_agent.sft.loss import log_softmax, perplexity
from smart_research_agent.utils.logger import get_logger

logger = get_logger(__name__)

#: 探针可选的文本字段：参考答案（缺省）或指令.
PROBE_FIELDS: tuple[str, ...] = ("reference", "instruction")

#: 探针至少要有的 token 数：只有 1 个 token 的序列没有"下一个 token"，
#: 对数概率无定义（分母为 0）.
MIN_PROBE_TOKENS = 2


class ContextModel(Protocol):
    """探针需要的最小模型接口（``ReferenceSFTModel`` 与 ``LoRAReferenceModel`` 都满足）.

    刻意只声明两个成员：**探针不应该知道模型是怎么实现的**——它只需要
    "给一个上下文 token，还我一个 logits 向量"。这样 day050 的基座模型
    与 day051 的 LoRA 模型可以被同一段代码测量，微调前后的差值才是
    同口径的。
    """

    @property
    def vocab_size(self) -> int:  # pragma: no cover - Protocol 声明
        """词表大小."""
        ...

    def logits(self, context_token: int) -> list[float]:  # pragma: no cover - Protocol 声明
        """给定上下文 token 的 logits."""
        ...


@dataclass(frozen=True)
class ProbeRecord:
    """单条用例上的模型侧读数."""

    item_id: str
    bucket: str
    difficulty: str
    tokens: int
    supervised: int
    logprob: float
    accuracy: float

    def __post_init__(self) -> None:
        """拒绝"没有监督位置"的记录（否则平均值会以 ``ZeroDivisionError`` 炸掉）.

        ``probe_items`` 走的是 ``sequence_logprob``，它保证 ``supervised >= 1``，
        所以这一条只在手工构造时才会命中。加它的理由是**错误类型要说人话**：
        ``ZeroDivisionError`` 会被读成"代码写错了"，而真正的问题是
        "这条记录没有可计分的位置"。
        """
        if self.supervised < 1:
            raise FinetuneEvalError(
                f"记录 {self.item_id} 的 supervised 必须为正整数，收到 {self.supervised}"
            )
        if self.tokens < MIN_PROBE_TOKENS:
            raise FinetuneEvalError(
                f"记录 {self.item_id} 的 tokens 至少为 {MIN_PROBE_TOKENS}，收到 {self.tokens}"
            )

    @property
    def mean_logprob(self) -> float:
        """逐 token 平均对数概率（构造期已保证分母为正）."""
        return self.logprob / self.supervised

    @property
    def perplexity(self) -> float:
        """困惑度 ``exp(-平均对数概率)``."""
        return perplexity(-self.mean_logprob)

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典."""
        return {
            "item_id": self.item_id,
            "bucket": self.bucket,
            "difficulty": self.difficulty,
            "tokens": self.tokens,
            "supervised": self.supervised,
            "logprob": self.logprob,
            "mean_logprob": self.mean_logprob,
            "perplexity": self.perplexity,
            "accuracy": self.accuracy,
        }


@dataclass(frozen=True)
class ProbeResult:
    """一次探针的全部读数与聚合（``label`` 用来区分"微调前 / 微调后"）."""

    label: str
    records: list[ProbeRecord] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """参与统计的 token 总数（分母）."""
        return sum(record.supervised for record in self.records)

    @property
    def mean_logprob(self) -> float:
        """**按 token 加权**的平均对数概率.

        为什么不用"先按条目平均、再对条目求平均"：那样会让短条目权重过大，
        而困惑度的定义是逐 token 的平均。两种平均在同一份数据上会给出
        不同数字，报告里必须说清用的是哪一种——**这里用的是逐 token 的**。
        """
        if not self.records:
            raise FinetuneEvalError("空探针结果没有平均对数概率")
        total = self.total_tokens
        if total == 0:  # pragma: no cover - 构造处已拒绝无监督位置的条目
            raise FinetuneEvalError("探针结果里没有任何监督位置")
        return sum(record.logprob for record in self.records) / total

    @property
    def perplexity(self) -> float:
        """整个评估集上的困惑度 ``exp(-平均对数概率)``."""
        return perplexity(-self.mean_logprob)

    @property
    def accuracy(self) -> float:
        """整个评估集上的下一 token 命中率（按 token 加权）."""
        total = self.total_tokens
        if total == 0:  # pragma: no cover - 同上
            raise FinetuneEvalError("探针结果里没有任何监督位置")
        return sum(record.accuracy * record.supervised for record in self.records) / total

    def by_bucket(self) -> dict[str, dict[str, float]]:
        """按桶聚合（每个桶内部仍是按 token 加权）."""
        grouped: dict[str, list[ProbeRecord]] = {}
        for record in self.records:
            grouped.setdefault(record.bucket, []).append(record)
        table: dict[str, dict[str, float]] = {}
        for bucket, members in sorted(grouped.items()):
            tokens = sum(member.supervised for member in members)
            logprob = sum(member.logprob for member in members)
            accuracy = sum(member.accuracy * member.supervised for member in members)
            table[bucket] = {
                "items": len(members),
                "supervised": tokens,
                "mean_logprob": logprob / tokens if tokens else 0.0,
                "perplexity": perplexity(-logprob / tokens) if tokens else 0.0,
                "accuracy": accuracy / tokens if tokens else 0.0,
            }
        return table

    def to_dict(self) -> dict[str, Any]:
        """投影为可 json.dumps 的字典（逐条 + 聚合）."""
        return {
            "label": self.label,
            "total_tokens": self.total_tokens,
            "mean_logprob": self.mean_logprob,
            "perplexity": self.perplexity,
            "accuracy": self.accuracy,
            "by_bucket": self.by_bucket(),
            "records": [record.to_dict() for record in self.records],
        }


def sequence_logprob(
    model: ContextModel,
    token_ids: Sequence[int],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[float, int]:
    """序列的对数概率之和与**监督位置数**（分母单独交出来）.

    位置 ``t = 0`` 被跳过（没有前文），``ignore_index`` 的位置被跳过——
    这与 day050 的 ``ReferenceSFTModel.accumulate`` 是同一套口径。

    **但它不等于训练 loss**，两者的分母差 1：训练时"答案的第一个 token"
    是被计分的（它的上下文是 prompt 的最后一个 token），而本函数在被喂
    "只有答案的 id"时从第二个 token 才开始计分。所以正确用法是
    **同口径对比（微调前 vs 微调后都用探针）**，而不是把探针读数与训练
    日志里的 loss 相减——那两个数字的分子分母都不一样，相减得到的差
    没有任何含义。

    ``ignore_index`` 出现在**目标**位置时跳过该位置；出现在**上下文**位置时
    直接报错（因为"以上一个被屏蔽的 token 为条件"没有定义，而静默跳过会
    让分母悄悄变小——分母一变，前后两次探针就不可比了）。

    越界 token 立刻报错：用错词表的 tokenizer 探针会得到"模型很差"的
    假象，而真实原因是 id 根本不属于这个模型。
    """
    if len(token_ids) < MIN_PROBE_TOKENS:
        raise FinetuneEvalError(
            f"序列至少需要 {MIN_PROBE_TOKENS} 个 token 才能算下一个 token 的对数概率，"
            f"收到 {len(token_ids)}"
        )
    total = 0.0
    count = 0
    for index in range(1, len(token_ids)):
        target = token_ids[index]
        if target == ignore_index:
            continue
        context = token_ids[index - 1]
        if not 0 <= context < model.vocab_size:
            raise FinetuneEvalError(
                f"上下文 token {context} 超出模型词表 [0, {model.vocab_size})："
                "探针与模型必须使用同一套词表"
            )
        if not 0 <= target < model.vocab_size:
            raise FinetuneEvalError(
                f"目标 token {target} 超出模型词表 [0, {model.vocab_size})："
                "探针与模型必须使用同一套词表"
            )
        total += log_softmax(model.logits(context))[target]
        count += 1
    if count == 0:
        raise FinetuneEvalError("序列里没有任何可计分的位置（全部被 ignore_index 屏蔽）")
    return total, count


def token_accuracy(model: ContextModel, token_ids: Sequence[int]) -> tuple[float, int]:
    """下一 token 命中率：``argmax logits == 目标 token`` 的比例（并返回分母）.

    它是比对数概率更直观的量：对数概率是"给正确答案多少概率"，命中率是
    "有没有把正确答案排到第一"。两者一起看才能区分"更自信了"与"更对了"。

    与 ``sequence_logprob`` 的**一处刻意的不对称**：本函数不做
    ``ignore_index`` 屏蔽（而是把越界的 ``-100`` 当成"id 不属于这个词表"
    报错）。理由是本函数的输入永远是"一段真实文本的 id"（``probe_items``
    只喂 tokenizer 的产物），屏蔽位在这里没有语义；要让它与
    ``sequence_logprob`` 完全同构，就得允许"以被屏蔽 token 为上下文"这种
    没有定义的情形混进来。
    """
    if len(token_ids) < MIN_PROBE_TOKENS:
        raise FinetuneEvalError(
            f"序列至少需要 {MIN_PROBE_TOKENS} 个 token 才能算命中率，收到 {len(token_ids)}"
        )
    hits = 0
    count = 0
    for index in range(1, len(token_ids)):
        context = token_ids[index - 1]
        target = token_ids[index]
        if not 0 <= context < model.vocab_size or not 0 <= target < model.vocab_size:
            raise FinetuneEvalError(
                f"token 超出模型词表 [0, {model.vocab_size})：探针与模型必须使用同一套词表"
            )
        logits = model.logits(context)
        best = max(range(len(logits)), key=lambda position: logits[position])
        if best == target:
            hits += 1
        count += 1
    if count == 0:  # pragma: no cover - 与 sequence_logprob 同构，长度已校验
        raise FinetuneEvalError("序列里没有任何可计分的位置")
    return hits / count, count


def encode_probe_batches(items: Sequence[EvalItem], tokenizer: Any) -> list[Batch]:
    """把用例编码成**单样本** Batch，并把 prompt 段屏蔽为 ``IGNORE_INDEX``.

    两处刻意的选择：

    1. **prompt 段被屏蔽**（day050 的核心纪律）。不屏蔽时 loss 里混进
       "预测用户提问"的信号，而探针量的是"模型对参考答案有多熟"——
       只有屏蔽之后，loss 的分子分母才与这个说法对齐。
    2. **单样本、不 padding**：探针与微调都逐条处理，padding 只会引入
       "同一批里最长的那条决定其余各条的显存占用"这个与结论无关的变量。
       ``Batch`` 是元组结构，本就允许不同批次的长度不同。

    返回值可以直接喂给 ``ReferenceSFTModel.accumulate`` / ``LoRAReferenceModel``，
    也可以交给 ``SFTTrainer``——**它就是一份合法的训练批次**。
    """
    batches: list[Batch] = []
    for item in items:
        prompt_ids = tokenizer.encode(item.instruction)
        answer_ids = tokenizer.encode(item.reference)
        ids = prompt_ids + answer_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
        batches.append(
            Batch(
                input_ids=(tuple(ids),),
                labels=(tuple(labels),),
                attention_mask=(tuple([1] * len(ids)),),
                pad_token_id=tokenizer.pad_token_id,
            )
        )
    return batches


def probe_items(
    model: ContextModel,
    tokenizer: Any,
    items: Sequence[EvalItem],
    *,
    label: str = "probe",
    probe_field: str = "reference",
) -> ProbeResult:
    """对评估集逐条探针，返回带聚合的 ``ProbeResult``.

    ``tokenizer`` 只要求有 ``encode(text) -> list[int]``（``CharTokenizer``
    满足）。用 ``probe_field`` 选择探"参考答案"还是"指令"：探答案是
    "模型对领域答案有多熟"，探指令是"模型对提问分布有多熟"——两者在
    微调前后会给出不同的变化幅度，这本身就是一条可报告的发现。
    """
    if probe_field not in PROBE_FIELDS:
        raise FinetuneEvalError(
            f"未知的探针字段 {probe_field!r}，可选：{', '.join(PROBE_FIELDS)}"
        )
    if not items:
        raise FinetuneEvalError("不能对空评估集做探针")
    records: list[ProbeRecord] = []
    for item in items:
        text = item.reference if probe_field == "reference" else item.instruction
        token_ids = tokenizer.encode(text)
        logprob, supervised = sequence_logprob(model, token_ids)
        accuracy, _ = token_accuracy(model, token_ids)
        records.append(
            ProbeRecord(
                item_id=item.id,
                bucket=item.bucket,
                difficulty=item.difficulty,
                tokens=len(token_ids),
                supervised=supervised,
                logprob=logprob,
                accuracy=accuracy,
            )
        )
    result = ProbeResult(label=label, records=records)
    logger.info(
        "探针 %s：%d 条 | 困惑度 %.4f | 命中率 %.4f",
        label,
        len(records),
        result.perplexity,
        result.accuracy,
    )
    return result


def compare_probes(before: ProbeResult, after: ProbeResult) -> dict[str, Any]:
    """微调前后的探针对比：困惑度降了多少、命中率涨了多少、逐条胜负如何.

    ``improved`` 的判据是**困惑度下降**（对数概率上升），不是命中率上升：
    在字符级词表上，命中率天然偏低（一个上下文在中文里往往有多个合理
    后继），它对小规模微调不敏感；而平均对数概率是连续的，能反映
    "把正确答案的概率抬高了"这件真实发生的事。
    """
    before_ids = {record.item_id for record in before.records}
    after_ids = {record.item_id for record in after.records}
    if before_ids != after_ids:
        missing = sorted(before_ids ^ after_ids)
        raise FinetuneEvalError(f"两次探针的用例集合不一致，差异：{', '.join(missing)}")
    after_by_id = {record.item_id: record for record in after.records}
    wins = 0
    losses = 0
    for record in before.records:
        counterpart = after_by_id[record.item_id]
        if counterpart.logprob > record.logprob:
            wins += 1
        elif counterpart.logprob < record.logprob:
            losses += 1
    perplexity_delta = before.perplexity - after.perplexity
    return {
        "items": len(before.records),
        "before": {
            "label": before.label,
            "perplexity": before.perplexity,
            "mean_logprob": before.mean_logprob,
            "accuracy": before.accuracy,
        },
        "after": {
            "label": after.label,
            "perplexity": after.perplexity,
            "mean_logprob": after.mean_logprob,
            "accuracy": after.accuracy,
        },
        "perplexity_delta": perplexity_delta,
        "perplexity_ratio": after.perplexity / before.perplexity,
        "accuracy_delta": after.accuracy - before.accuracy,
        "item_wins": wins,
        "item_losses": losses,
        "improved": perplexity_delta > 0,
    }


__all__ = [
    "MIN_PROBE_TOKENS",
    "PROBE_FIELDS",
    "ContextModel",
    "ProbeRecord",
    "ProbeResult",
    "compare_probes",
    "probe_items",
    "sequence_logprob",
    "token_accuracy",
]
