"""SFT 编码：监督文本 → ``input_ids`` / ``labels`` / ``attention_mask``（M5-D2）.

本模块是全课的核心机制所在。一句话概括 SFT 的 loss 口径：

    **prompt 段的 label 置成 ``IGNORE_INDEX = -100``，只有答案段的 token
    参与交叉熵；padding 段同样置成 -100。**

为什么必须这样做：如果 prompt 也参与 loss，那么模型会被同时训练去
"预测用户的问题"——而用户的提问在推理时是**给定的输入**，模型不需要
（也不应该）学会生成它。把 prompt 计入 loss 有两个直接后果：

1. **浪费容量**：模型把一部分参数用于"学会像用户那样提问"，而这部分
   能力在推理时完全用不上；
2. **稀释信号**：本课程的数据集里 prompt 平均 26.8 字、答案平均 88.1 字，
   若 prompt 计入 loss，约 23% 的监督信号会被用在一个错误的目标上。

``-100`` 这个具体数值不是本课发明的：它是 PyTorch ``nn.CrossEntropyLoss``
的 ``ignore_index`` 默认值，也是 HuggingFace 生态里 SFT label masking 的
通行约定（TRL 的 ``SFTTrainer`` 用 ``completion_mask`` 把 prompt 位置屏蔽，
最终同样落到 -100）。**用同一个约定值，意味着本课的数据可以直接喂给
``transformers.Trainer`` 而不需要任何转换。**

分词器的选择：本模块自带一个**字符级 ``CharTokenizer``**（离线、确定、
可逆、零依赖），并把"分词器"抽象成一个只依赖 ``encode`` / ``vocab_size`` /
``pad_token_id`` 的窄接口。生产上把 ``AutoTokenizer`` 传进来即可，其余
代码一行不改——这与 day045"协议兼容胜过 SDK 封装"是同一种思路。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from smart_research_agent.sft.template import RenderedSample

#: 屏蔽位：PyTorch ``CrossEntropyLoss`` 的 ``ignore_index`` 默认值，
#: 也是 HuggingFace / TRL 生态的通行约定。**不要改成别的值**——一旦
#: 偏离这个约定，"训练数据能直接喂给 transformers.Trainer" 这条性质就没了。
IGNORE_INDEX = -100

#: padding 用的特殊 token 与 id（id 必须在词表内且固定，避免下游错位）
PAD_TOKEN = "<pad>"
#: 未登录字符占位符
UNK_TOKEN = "<unk>"
#: 特殊 token 全集；它们在词表里的位置即其 id
SPECIAL_TOKENS: tuple[str, ...] = (PAD_TOKEN, UNK_TOKEN)
#: ``<pad>`` 的 id：0 是多数实现（含 transformers）的默认 padding id
PAD_TOKEN_ID = 0
#: ``<unk>`` 的 id
UNK_TOKEN_ID = 1

#: 截断策略：``keep-answer`` 保留完整答案、从**左侧**截断 prompt；``head`` 直接取前 N 个 token
TRUNCATION_KEEP_ANSWER = "keep-answer"
TRUNCATION_HEAD = "head"
#: 支持的截断策略全集
SUPPORTED_TRUNCATIONS: tuple[str, ...] = (TRUNCATION_KEEP_ANSWER, TRUNCATION_HEAD)

#: 一条样本至少要 1 个上下文 token + 1 个监督 token，否则它对训练毫无贡献
MIN_USEFUL_TOKENS = 2


class SFTDataError(ValueError):
    """样本无法编码为可训练的监督样本（继承 ``ValueError``）.

    刻意让"答案放不下""整段被截掉"这类情况**在这里就报错**，而不是
    静默产出一条监督 token 为 0 的样本：**零监督样本是一条"消费算力但
    不产生梯度"的死数据**，它会让 loss 曲线看起来正常（其余批次在降），
    却让样本的实际利用率悄悄下降。这类缺陷比崩溃更难发现。
    """


class CharTokenizer:
    """字符级分词器：离线、确定、可逆（生产上替换为 ``AutoTokenizer``）.

    为什么本课自带一个而不是直接把 ``tiktoken`` 用起来：SFT 的编码路径
    要被**逐 token 精确断言**（"第几个 token 的 label 是 -100"），而
    ``tiktoken`` 需要下载词表文件（网络依赖 + 版本漂移），且它的 BPE
    合并会让"哪几个字符对应哪个 token"变得难以手算。字符级分词器让
    每一个 token 都能手数出来，**机制因此可被验证**。

    坏处也必须说清楚：字符级分词会让序列变长（中文约 1 字 1 token，
    英文约 1 字符 1 token，而 BPE 英文约 4 字符 1 token）。所以它只用于
    **验证机制**，不用于真实训练——真实训练的 token 预算由 day034 的
    ``TokenCounter`` 与 day048 的 ``DatasetStats`` 给出。
    """

    def __init__(self, tokens: Sequence[str]):
        # 去重且保序：调用方给的顺序直接决定 id，便于复现
        seen: list[str] = []
        for token in tokens:
            if token not in seen:
                seen.append(token)
        self._tokens: tuple[str, ...] = (*SPECIAL_TOKENS, *seen)
        self._index: dict[str, int] = {token: i for i, token in enumerate(self._tokens)}

    @classmethod
    def from_texts(cls, texts: Sequence[str]) -> CharTokenizer:
        """从一批文本构建词表（按字符首次出现顺序，保证可复现）."""
        return cls([char for text in texts for char in text])

    @property
    def tokens(self) -> tuple[str, ...]:
        """完整词表（特殊 token 在前）."""
        return self._tokens

    @property
    def vocab_size(self) -> int:
        """词表大小（= 特殊 token 数 + 语料字符数）."""
        return len(self._tokens)

    @property
    def pad_token_id(self) -> int:
        """padding id（``<pad>`` 的 id，恒为 0）."""
        return PAD_TOKEN_ID

    @property
    def ignore_index(self) -> int:
        """label 屏蔽位（恒为 -100）."""
        return IGNORE_INDEX

    def encode(self, text: str) -> list[int]:
        """把文本编码为 id 序列；未登录字符映射为 ``<unk>``."""
        unk = UNK_TOKEN_ID
        return [self._index.get(char, unk) for char in text]

    def decode(self, ids: Sequence[int]) -> str:
        """把 id 序列还原为文本（跳过特殊 token，未登录 id 也跳过）.

        ``decode`` 与 ``encode`` 对"词表内文本"构成往返：这是"分词器没接错"
        最直接的验证方式，也是本课测试里最爱用的一条断言。
        """
        specials = set(range(len(SPECIAL_TOKENS)))
        return "".join(
            self._tokens[i] for i in ids if 0 <= i < len(self._tokens) and i not in specials
        )

    def to_dict(self) -> dict:
        """导出为可 json.dumps 的字典（检查点需要它来保存词表）."""
        return {"tokens": list(self._tokens)}

    @classmethod
    def from_dict(cls, payload: dict) -> CharTokenizer:
        """从 ``to_dict`` 的产物还原（特殊 token 由构造器自动补回）."""
        tokens = list(payload.get("tokens") or [])
        # 去掉开头已经存在的前缀特殊 token，避免重复
        head = list(SPECIAL_TOKENS)
        if tokens[: len(head)] == head:
            tokens = tokens[len(head) :]
        return cls(tokens)

    def __len__(self) -> int:
        return len(self._tokens)


@dataclass(frozen=True)
class EncodedSample:
    """一条编码好的监督样本：三个等长序列 + 统计量.

    三个序列等长是不变式（``__post_init__`` 会校验）：``input_ids`` /
    ``labels`` / ``attention_mask`` 长度不一致是 SFT 里最典型的送错数据
    的方式，而它在 loss 计算时往往不会立刻报错（很多实现按 ``input_ids``
    的长度循环，多出来的 label 被静默忽略）。**在构造处校验，比在 loss
    处排查便宜得多。**
    """

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    #: 序列里属于**前缀（prompt）**的 token 数——与是否屏蔽无关
    prompt_tokens: int
    #: ``labels`` 里非 ``IGNORE_INDEX`` 的个数（= 真正参与 loss 的 token 数）
    supervised_tokens: int
    #: 是否屏蔽了 prompt（True = 标准 SFT；False 只用于"不屏蔽会怎样"的对照实验）
    masked_prompt: bool
    #: 是否发生了截断（prompt 被裁短，或 head 策略下答案被切掉）
    truncated: bool

    def __post_init__(self) -> None:
        lengths = {len(self.input_ids), len(self.labels), len(self.attention_mask)}
        if len(lengths) != 1:
            raise SFTDataError(f"三个序列必须等长，实际长度集合为 {sorted(lengths)}")
        if self.supervised_tokens <= 0:
            raise SFTDataError("监督 token 数为 0：这条样本不会产生任何梯度")
        if self.prompt_tokens < 1:
            raise SFTDataError("prompt token 数为 0：第一个监督 token 没有任何上下文")
        if self.masked_prompt and self.supervised_tokens != (
            len(self.input_ids) - self.prompt_tokens
        ):
            raise SFTDataError(
                "屏蔽了 prompt 时，监督 token 数必须等于监督区间的长度："
                f"{self.supervised_tokens} != {len(self.input_ids)} - {self.prompt_tokens}"
            )

    @property
    def total_tokens(self) -> int:
        """总 token 数（含 padding 之前）."""
        return len(self.input_ids)

    @property
    def masked_tokens(self) -> int:
        """被屏蔽的 token 数（不产生梯度，但**参与注意力计算**）."""
        return self.total_tokens - self.supervised_tokens

    @property
    def supervised_ratio(self) -> float:
        """监督 token 占比——SFT 里这个比例直接决定"算力花在哪"."""
        return self.supervised_tokens / self.total_tokens if self.total_tokens else 0.0


@dataclass(frozen=True)
class Batch:
    """一个已 padding 的批次（二维元组，便于日志与断言）.

    用元组而不是 list：批次一旦组装就不应再被就地修改——训练循环里
    "某个环节偷偷改了 labels"是最难查的一类 bug。
    """

    input_ids: tuple[tuple[int, ...], ...]
    labels: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]
    pad_token_id: int

    @property
    def batch_size(self) -> int:
        """批大小."""
        return len(self.input_ids)

    @property
    def max_length(self) -> int:
        """本批的 padding 长度（= batch 内最长样本的长度）."""
        return len(self.input_ids[0]) if self.input_ids else 0

    @property
    def lengths(self) -> tuple[int, ...]:
        """每条样本的真实长度（由 attention_mask 求和得到）."""
        return tuple(sum(mask) for mask in self.attention_mask)

    @property
    def supervised_tokens(self) -> int:
        """本批监督 token 总数."""
        return sum(
            1 for labels in self.labels for value in labels if value != IGNORE_INDEX
        )

    @property
    def padding_tokens(self) -> int:
        """本批 padding token 总数——**它们同样参与注意力计算**，
        所以这个数字是"算力有多少花在了空位上"的直接度量。
        """
        return self.batch_size * self.max_length - sum(self.lengths)


def encode_supervised(
    rendered: RenderedSample,
    tokenizer: CharTokenizer,
    *,
    max_length: int,
    truncation: str = TRUNCATION_KEEP_ANSWER,
    mask_prompt: bool = True,
    ignore_index: int = IGNORE_INDEX,
) -> EncodedSample:
    """把渲染结果编码为监督样本（本课最核心的一个函数）.

    关键实现细节：**前缀与监督区间分别编码再拼接**。

    ``tokenizer.encode(rendered.prompt_text) + tokenizer.encode(rendered.supervised_text)``
    而不是 ``tokenizer.encode(rendered.text)`` 再按字符偏移切片。原因在
    BPE 类分词器上很致命：跨边界的字符合并会产出一个"同时属于 prompt 与
    answer"的 token，切片时无论怎么切都不对——切给 prompt 就丢了答案的
    第一个词，切给 answer 就让模型学着预测一个它本不该预测的 token。
    按字符偏移分别编码，边界因此是**确定的**。

    ``truncation`` 两种策略：

    - ``keep-answer``（缺省）：**答案必须完整放下**。预算先给答案，剩下的
      给 prompt，并从**左侧**裁掉 prompt——靠近答案的那部分指令信息密度
      最高，丢掉开头的客套话代价最小。答案放不下则抛 ``SFTDataError``。
    - ``head``：朴素地取前 ``max_length`` 个 token。它可能把答案整个切掉，
      此时同样抛 ``SFTDataError``（宁可失败，也不产出零监督样本）。

    ``mask_prompt=True``（缺省）是本课的标准口径，prompt 段 label 置
    ``-100``。把它做成开关不是为了"多一个选项"，而是为了让
    **"不屏蔽会怎样"成为一个可运行的对照实验**（见
    ``tests/test_sft_trainer.py`` 与教程第六章的实测）。
    """
    if truncation not in SUPPORTED_TRUNCATIONS:
        raise SFTDataError(
            f"不支持的截断策略 {truncation!r}，可选：{', '.join(SUPPORTED_TRUNCATIONS)}"
        )
    if max_length < MIN_USEFUL_TOKENS:
        raise SFTDataError(
            f"max_length 至少为 {MIN_USEFUL_TOKENS}"
            f"（1 个上下文 + 1 个监督 token），收到 {max_length}"
        )

    prompt_ids = tokenizer.encode(rendered.prompt_text)
    answer_ids = tokenizer.encode(rendered.supervised_text)
    if not prompt_ids:
        raise SFTDataError("前缀编码后为空：prompt 至少要贡献 1 个上下文 token")
    if not answer_ids:  # pragma: no cover - 防御式分支：RenderedSample 保证监督区间非空
        raise SFTDataError("监督区间编码后为空：答案至少要贡献 1 个监督 token")

    if truncation == TRUNCATION_KEEP_ANSWER:
        budget = max_length - len(answer_ids)
        if budget < 1:
            raise SFTDataError(
                f"答案需要 {len(answer_ids)} 个 token，加上 1 个上下文 token 已超过 "
                f"max_length={max_length}：请调大 max_length 或先用 suggest_max_length 检查长度分布"
            )
        if budget < len(prompt_ids):
            kept_prompt = prompt_ids[len(prompt_ids) - budget :]
            truncated = True
        else:
            kept_prompt = prompt_ids
            truncated = False
        ids = kept_prompt + answer_ids
        prompt_labels = (
            [ignore_index] * len(kept_prompt) if mask_prompt else list(kept_prompt)
        )
        labels = prompt_labels + list(answer_ids)
        prompt_tokens = len(kept_prompt)
    else:  # TRUNCATION_HEAD
        full = prompt_ids + answer_ids
        kept = full[:max_length]
        prompt_kept = min(len(prompt_ids), max_length)
        supervised = len(kept) - prompt_kept
        if supervised <= 0:
            raise SFTDataError(
                f"head 截断后没有任何监督 token：前缀占 {len(prompt_ids)} 个 token，"
                f"而 max_length={max_length}；应改用 keep-answer 或调大 max_length"
            )
        ids = kept
        prompt_labels = [ignore_index] * prompt_kept if mask_prompt else kept[:prompt_kept]
        labels = prompt_labels + kept[prompt_kept:]
        prompt_tokens = prompt_kept
        truncated = len(kept) < len(full)

    return EncodedSample(
        input_ids=tuple(ids),
        labels=tuple(labels),
        attention_mask=tuple([1] * len(ids)),
        prompt_tokens=prompt_tokens,
        supervised_tokens=sum(1 for value in labels if value != ignore_index),
        masked_prompt=mask_prompt,
        truncated=truncated,
    )


def collate(
    samples: Sequence[EncodedSample],
    *,
    pad_token_id: int = PAD_TOKEN_ID,
    pad_to: int | None = None,
) -> Batch:
    """把若干条样本 padding 成批次.

    ``pad_to=None``（缺省）按 **batch 内最长样本** 补齐（等价于
    ``padding="longest"``）：这样 padding 量由数据自己决定，不会因为
    全局 ``max_length`` 设得过大而白算。传入 ``pad_to`` 则按固定长度补齐
    （等价于 ``padding="max_length"``，用于需要静态 shape 的场景），
    此时若某条样本超过 ``pad_to`` 直接报错。

    padding 位置的三件套必须一起打：``input_ids`` 补 ``pad_token_id``、
    ``labels`` 补 ``IGNORE_INDEX``、``attention_mask`` 补 0。**三者缺一
    都会出错**：labels 没补 -100 会让 loss 去预测 pad，attention_mask 没
    补 0 会让 pad 参与注意力。
    """
    if not samples:
        raise SFTDataError("空批次：collate 至少需要一条样本")
    longest = max(len(sample.input_ids) for sample in samples)
    if pad_to is not None:
        if pad_to < longest:
            raise SFTDataError(
                f"pad_to={pad_to} 小于批内最长样本 {longest}：固定长度会截断数据，请调大 pad_to"
            )
        target = pad_to
    else:
        target = longest

    input_ids: list[tuple[int, ...]] = []
    labels: list[tuple[int, ...]] = []
    masks: list[tuple[int, ...]] = []
    for sample in samples:
        padding = target - len(sample.input_ids)
        input_ids.append(sample.input_ids + (pad_token_id,) * padding)
        labels.append(sample.labels + (IGNORE_INDEX,) * padding)
        masks.append(sample.attention_mask + (0,) * padding)
    return Batch(
        input_ids=tuple(input_ids),
        labels=tuple(labels),
        attention_mask=tuple(masks),
        pad_token_id=pad_token_id,
    )


def iter_batches(
    samples: Sequence[EncodedSample],
    *,
    batch_size: int,
    drop_last: bool = False,
    pad_token_id: int = PAD_TOKEN_ID,
) -> list[Batch]:
    """把样本切成批次.

    ``drop_last`` 的缺省是 **False**（不丢尾巴），与 PyTorch ``DataLoader``
    的缺省一致：真实数据里"最后一个不满的批次"通常是有价值的，丢掉它
    等于丢掉数据。训练循环是否丢弃由调用方决定（见 ``trainer.py`` 的
    ``drop_last_optimizer_steps`` 说明）。
    """
    if batch_size <= 0:
        raise SFTDataError(f"batch_size 必须为正整数，收到 {batch_size}")
    batches: list[Batch] = []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        if len(chunk) < batch_size and drop_last:
            continue
        batches.append(collate(chunk, pad_token_id=pad_token_id))
    return batches


@dataclass(frozen=True)
class LengthSummary:
    """渲染后的 token 长度分布（**决定 ``max_length`` 的唯一依据**）.

    day048 的 ``DatasetStats`` 给的是 ``instruction + output`` 的原始字符
    与粗略 token 估计；而真正要放进模型的序列比它长一节：

        + 系统提示词与角色标记（ChatML 约 100 字符）
        + 轮次终止符
        + 字符级分词把中文按 1 字 1 token 计（BPE 会低得多）

    实测（本课程数据集、ChatML 模板、字符级分词器）:

        ``min=219  p50=235  p90=276  p95=302  max=320  mean=248.7``

    而 day048 的画像给出的估计是 **84.24 token/条**——**相差近三倍**。
    这条差距是本课最重要的一个实操教训：**``max_length`` 必须按"渲染并
    分词之后的长度"来定，而不是按原始文本的长度来定。** 定小了会静默
    截断答案（``keep-answer`` 策略下会直接报错），定大了则白白浪费算力。
    """

    count: int
    minimum: int
    maximum: int
    mean: float
    p50: int
    p90: int
    p95: int
    prompt_mean: float
    supervised_mean: float
    #: 升序长度表（分位查询的依据；不参与 repr / 相等比较）
    lengths: tuple[int, ...] = field(default=(), repr=False, compare=False)

    def quantile(self, ratio: float) -> int:
        """按分位取长度（``ratio`` 落在 [0, 1]；用最近秩法，不插值）.

        与 day046 ``perf_baseline.percentile`` 用同一种"最近秩法"：分位数
        必须对应一个**真实存在的样本长度**，插值出来的数字不是任何一条
        样本的长度，作为"取多长能不截断"的依据会误导。
        """
        if not 0.0 <= ratio <= 1.0:
            raise SFTDataError(f"分位比例必须落在 [0, 1]，收到 {ratio}")
        if not self.lengths:
            raise SFTDataError("LengthSummary 缺少 lengths，无法查询分位")
        index = min(self.count - 1, max(0, round(ratio * (self.count - 1))))
        return self.lengths[index]

    def summary_line(self) -> str:
        """人类可读的一行摘要."""
        return (
            f"{self.count} 条 | token 长度 min={self.minimum} p50={self.p50} "
            f"p90={self.p90} p95={self.p95} max={self.maximum} mean={self.mean:.1f} | "
            f"prompt 均值 {self.prompt_mean:.1f} / 监督均值 {self.supervised_mean:.1f}"
        )

    def to_dict(self) -> dict:
        """投影为可 json.dumps 的字典（不含 lengths 明细）."""
        return {
            "count": self.count,
            "min": self.minimum,
            "p50": self.p50,
            "p90": self.p90,
            "p95": self.p95,
            "max": self.maximum,
            "mean": round(self.mean, 2),
            "prompt_mean": round(self.prompt_mean, 2),
            "supervised_mean": round(self.supervised_mean, 2),
        }


def _lengths_of(
    rendered: Sequence[RenderedSample], tokenizer: CharTokenizer
) -> tuple[list[int], list[int], list[int]]:
    """分别算出（总长度, prompt 长度, 监督长度）三个列表.

    长度按**分别编码再相加**计算，与 ``encode_supervised`` 的口径一致：
    不截断、不合并，因此这是"如果 max_length 足够大，序列会有多长"的
    上界答案——``max_length`` 就该按它来定。
    """
    totals: list[int] = []
    prompts: list[int] = []
    supervised: list[int] = []
    for item in rendered:
        prompt_len = len(tokenizer.encode(item.prompt_text))
        answer_len = len(tokenizer.encode(item.supervised_text))
        prompts.append(prompt_len)
        supervised.append(answer_len)
        totals.append(prompt_len + answer_len)
    return totals, prompts, supervised


def length_summary(rendered: Sequence[RenderedSample], tokenizer: CharTokenizer) -> LengthSummary:
    """算出渲染后的长度分布（含 p50 / p90 / p95 分位）.

    空输入抛 ``SFTDataError``：没有样本就没有分布，返回全零会让
    ``suggest_max_length`` 给出一个毫无意义的建议值。
    """
    if not rendered:
        raise SFTDataError("长度分布需要至少一条渲染样本")
    totals, prompts, supervised = _lengths_of(rendered, tokenizer)
    ordered = sorted(totals)
    count = len(ordered)

    def quantile(ratio: float) -> int:
        index = min(count - 1, max(0, round(ratio * (count - 1))))
        return ordered[index]

    summary = LengthSummary(
        count=count,
        minimum=ordered[0],
        maximum=ordered[-1],
        mean=sum(ordered) / count,
        p50=quantile(0.50),
        p90=quantile(0.90),
        p95=quantile(0.95),
        prompt_mean=sum(prompts) / count,
        supervised_mean=sum(supervised) / count,
        lengths=tuple(ordered),
    )
    return summary


def suggest_max_length(
    rendered: Sequence[RenderedSample],
    tokenizer: CharTokenizer,
    *,
    quantile: float = 0.95,
    granularity: int = 32,
    minimum: int = 64,
) -> int:
    """按长度分位数给出 ``max_length`` 建议值（向上取整到 ``granularity``）.

    三个参数各有理由：

    - ``quantile=0.95``：取 p95 而不是最大值，是为了**容忍个别超长样本**
      （长尾样本往往正是"答案啰嗦"或"指令异常"的脏数据，为它们把整批
      数据的 ``max_length`` 抬高，代价是全部样本的算力）。被切掉的那 5%
      会在编码阶段被如实记录（``EncodingReport.truncated``），**是可观测的**；
    - ``granularity=32``：取整到 32 的倍数，让后续的 shape/对齐处理更整齐，
      也避免"``max_length`` 是 302 这种由某一条样本决定的魔法数字"；
    - ``minimum=64``：兜底下限，避免极小数据集给出一个放不下系统提示词的
      建议值。

    要"一条都不截断"就传 ``quantile=1.0``。
    """
    if granularity <= 0:
        raise SFTDataError(f"granularity 必须为正整数，收到 {granularity}")
    summary = length_summary(rendered, tokenizer)
    needed = summary.quantile(quantile)
    rounded = max(minimum, -(-needed // granularity) * granularity)
    return int(rounded)
