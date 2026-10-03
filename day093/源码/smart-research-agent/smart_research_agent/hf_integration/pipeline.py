"""``pipeline``：两条管线，**零数学**（day086 / M7-D10）.

真实库里的 ``pipeline("text-generation")`` 与 ``pipeline("feature-extraction")``
本身不含任何数学：它们只做"分词 → 前向 → 池化 / 生成 → 后处理"的编排。
本模块复刻这条分工，因此它带来一个立刻可用的结论：

```text
换管线不改模型，换模型也不改管线。
```

本模块只有三个函数，而三个函数各自对应一条**必须写下来的规矩**：

```text
text_generation     逐条跑（长度不齐、各自停在 eos）；settings **原样**透传给 day085
feature_extraction  可以合批；池化按 mask（见 features 模块）
run_both            同一份模型上跑两条管线：用来验证"换管线不改模型"
```

## 为什么生成不能合批

生成是**自回归**的：第 t 步的输出是第 t+1 步的输入。合批之后，
两条样本在第 t 步的"下一个 token"落在同一个矩阵乘法里——
但其中一条可能已经在第 t 步撞上了 eos（它该停下来），
而另一条还要继续。把它们绑在同一个循环里，只有两种结局：

```text
一起停        另一条被提前截断（它的 token 序列少了一截，而结果看起来完整）
不停          撞上 eos 的那条继续生成（它的输出里多了一段本该结束的内容）
```

因此本包**逐条跑**（与真实库在 ``batch_size=1`` 时的行为一致），
并把这件事写成一条读数（:mod:`study` 的生成表里会印出每条的长度）。
"""

from __future__ import annotations

from dataclasses import dataclass

from smart_research_agent.hf_integration.errors import ConfigError, ParameterError
from smart_research_agent.hf_integration.features import pool_forward
from smart_research_agent.hf_integration.forward import (
    ModelWeights,
    hidden_states,
    make_logits_fn,
)
from smart_research_agent.hf_integration.tokenizer import ByteBPETokenizer
from smart_research_agent.hf_integration.types import (
    ARCHITECTURE_GPT2,
    TASK_FEATURE_EXTRACTION,
    TASK_HEADS,
    TASK_KINDS,
    TASK_TEXT_GENERATION,
    ModelCard,
    PooledBatch,
    TextBatch,
)
from smart_research_agent.hf_source.generation import generate
from smart_research_agent.hf_source.types import GenerationResult, GenerationSettings

#: 两条管线各自需要哪种模型头（与 :data:`types.TASK_HEADS` 同一张表）.
HEAD_OF_TASK: dict[str, str] = dict(TASK_HEADS)


@dataclass(frozen=True)
class GenerationOutput:
    """一次生成的账：每条的 token 序列 + **解码出来的文本** + day085 的原始结果."""

    prompts: tuple[str, ...]
    texts: tuple[str, ...]
    results: tuple[GenerationResult, ...]
    strategy: str

    @property
    def batch_size(self) -> int:
        """几条提示词."""
        return len(self.prompts)

    def generated_lengths(self) -> tuple[int, ...]:
        """每一条**新生成**了多长（不等长是正常的：eos 让它们各自停下）."""
        return tuple(len(result.generated) for result in self.results)

    def to_dict(self) -> dict[str, object]:
        """摊平成可 JSON 化的字段."""
        return {
            "batch_size": self.batch_size,
            "strategy": self.strategy,
            "prompts": list(self.prompts),
            "texts": list(self.texts),
            "generated_lengths": list(self.generated_lengths()),
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True)
class FeatureOutput:
    """一次特征抽取的账：池化向量 + 每条的真实长度 + 这一步用的策略."""

    texts: tuple[str, ...]
    pooled: PooledBatch
    lengths: tuple[int, ...]
    batch: TextBatch

    def to_dict(self) -> dict[str, object]:
        """摊平成可 JSON 化的字段."""
        return {
            "texts": list(self.texts),
            "lengths": list(self.lengths),
            "pooling": self.pooled.to_dict(),
            "padding_ratio": self.batch.padding_ratio,
        }


def _check_task(task: str) -> None:
    """未知任务当场拒绝（回退到某条管线的后果是——一次生成被当成一次特征抽取）."""
    if task not in TASK_KINDS:
        raise ParameterError(
            f"未知的任务 {task!r}：本包只认 {list(TASK_KINDS)}。"
            "两条管线消费的是同一次前向的**不同部分**（logits vs hidden states），"
            "回退到其中一条会让调用方拿到形状对、语义错的结果。"
        )


def check_head_available(card: ModelCard, task: str, weights: ModelWeights) -> None:
    """这条管线需要的模型头在不在（**它决定了同一份主干能跑哪条管线**）.

    生成要一个"hidden → 词表"的输出投影；特征抽取只要主干。
    本包对 BERT 卡片跑生成时**当场拒绝**，而不是"用一个没训练过的池化输出硬凑"——
    后者会给出一个长度恰好等于词表的向量，而它的每一个数都没有意义。
    """
    _check_task(task)
    if task == TASK_TEXT_GENERATION:
        if weights.head is None or len(weights.head) != card.vocab:
            raise ConfigError(
                f"这份权重给不出长度 {card.vocab} 的 logits："
                "生成需要 lm_head（可与词嵌入共享）。"
            )
        if card.model_type != ARCHITECTURE_GPT2:
            raise ConfigError(
                f"本包只让 GPT-2 卡片跑生成：{card.model_type} 没有因果掩码，"
                "自回归生成在双向模型上会读到'未来'（那不会报错，只会给出一个"
                "每一步都偷看过答案的序列）。"
            )
        return
    return


def text_generation(
    card: ModelCard,
    weights: ModelWeights,
    tokenizer: ByteBPETokenizer,
    prompts: tuple[str, ...] | list[str],
    settings: GenerationSettings,
    *,
    skip_special_tokens: bool = True,
) -> GenerationOutput:
    """文本生成：逐条分词 → 逐条跑 day085 的 ``generate`` → 解码**新生成的那一段**.

    解码时故意**只解码新生成的那一段**（``result.generated``），
    因为把 prompt 一起拼回去会让"模型生成了什么"这件事消失在字符串里——
    而"它是不是把 prompt 原样吐了回来"恰恰是最需要被看见的一种失败。
    """
    check_head_available(card, TASK_TEXT_GENERATION, weights)
    if not prompts:
        raise ParameterError("text_generation 收到 0 条提示词：没有提示就没有生成。")
    logits_fn = make_logits_fn(card, weights)
    eos = tokenizer.eos_id
    resolved = settings if settings.eos_token is not None else _with_eos(settings, eos)
    texts: list[str] = []
    results: list[GenerationResult] = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt)
        if not ids:
            raise ParameterError(
                f"提示词 {prompt!r} 编出 0 个 token：空提示词无法作为生成的起点。"
            )
        result = generate(logits_fn, tuple(ids), resolved, vocab=card.vocab)
        results.append(result)
        texts.append(
            tokenizer.decode(result.generated, skip_special_tokens=skip_special_tokens)
        )
    return GenerationOutput(
        prompts=tuple(prompts),
        texts=tuple(texts),
        results=tuple(results),
        strategy=results[0].strategy if results else "",
    )


def _with_eos(settings: GenerationSettings, eos: int | None) -> GenerationSettings:
    """把 tokenizer 的 eos 填进生成配置（**不覆盖调用方显式给的那个**）."""
    if eos is None:
        return settings
    from dataclasses import replace

    return replace(settings, eos_token=eos)


def feature_extraction(
    card: ModelCard,
    weights: ModelWeights,
    tokenizer: ByteBPETokenizer,
    texts: tuple[str, ...] | list[str],
    *,
    pooling: str = "mean",
    normalize: bool = False,
    max_length: int | None = None,
    policy: str = "safe",
) -> FeatureOutput:
    """特征抽取：一次编码 + 一次（或逐条）前向 + 一次池化.

    ``max_length`` 给定时会先截断再对齐；不给时按这一批最长的那条对齐——
    两种选择在读数上不同（填充占比不同），因此它被印在 :class:`FeatureOutput` 里。
    """
    check_head_available(card, TASK_FEATURE_EXTRACTION, weights)
    if not texts:
        raise ParameterError("feature_extraction 收到 0 条文本：没有文本就没有向量。")
    batch = tokenizer.batch_encode(list(texts), max_length=max_length, padding=True)
    hidden = hidden_states(
        card,
        weights,
        tuple(row.input_ids for row in batch.rows),
        tuple(row.attention_mask for row in batch.rows),
        policy=policy,
    )
    pooled = pool_forward(hidden, pooling, normalize_vectors=normalize)
    if pooled.dim != card.hidden:
        raise ConfigError(
            f"池化出来的宽度 {pooled.dim} 与配置里的 hidden {card.hidden} 不一致："
            "它们必须是同一个数——否则下游的'同一份向量空间'是一个假前提。"
        )
    return FeatureOutput(
        texts=tuple(texts),
        pooled=pooled,
        lengths=hidden.lengths(),
        batch=batch,
    )


def run_both(
    card: ModelCard,
    weights: ModelWeights,
    tokenizer: ByteBPETokenizer,
    prompt: str,
    settings: GenerationSettings,
    *,
    pooling: str = "mean",
) -> tuple[GenerationOutput, FeatureOutput]:
    """同一份模型上同时跑两条管线（**用来证明"换管线不改模型"**）.

    它返回的两份结果都来自**同一份权重对象**：一次都没重新造过参数。
    因此"两条管线共用一次装载"这件事是可断言的（测试里查两个输出里的
    ``weights.seed`` 与 ``card.name`` 是同一个）。
    """
    generation = text_generation(card, weights, tokenizer, (prompt,), settings)
    features = feature_extraction(
        card, weights, tokenizer, (prompt,), pooling=pooling, policy="safe"
    )
    return generation, features


def head_line(card: ModelCard, task: str) -> str:
    """一行读数：这条管线要什么头（**"要什么头"就是"能跑哪条管线"**）."""
    _check_task(task)
    return f"{task}：需要 {HEAD_OF_TASK[task]} | 本卡片为 {card.model_type}"


__all__ = [
    "HEAD_OF_TASK",
    "FeatureOutput",
    "GenerationOutput",
    "check_head_available",
    "feature_extraction",
    "head_line",
    "run_both",
    "text_generation",
]
