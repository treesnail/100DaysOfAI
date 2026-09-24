"""模型侧探针测试（M5-D5）：白盒信号必须能手算核对.

``finetune_eval/model_probe.py`` 量的是"模型对领域答案有多熟"，它比文本指标
多一层要求：**数字必须能被手算复现**，否则"困惑度降了 0.3"这种结论无从核对。
本文件因此全面使用**自造的假模型**——``vocab_size`` + ``logits(ctx)`` 两个成员
就够了（``ContextModel`` 协议就只声明了两个），于是每个对数概率都能写成一个
闭式表达式：

.. code-block:: text

    均匀分布模型   logprob = -(n-1)·ln(V)      （V = 词表大小）
    后继模型       logprob = -ln((V-1)·e^-1 + 1)  （目标正好是 argmax 时）

三条承担"证明结论"角色的用例：

1. :meth:`TestSequenceLogprob.test_position_zero_is_skipped` —— 探针只对
   ``t >= 1`` 计分，且**上下文取自 ``ids[t-1]``**（``model.calls`` 必须严格
   等于 ``ids[:-1]``）。位置 0 没有前文，把它算进去会让分母凭空多 1；
2. :meth:`TestProbeResult.test_mean_logprob_is_token_weighted` —— 聚合是
   **按监督 token 数加权**，不是"先按条目平均再对条目平均"（两条记录手算
   ``-1.8`` vs 条目平均 ``-1.5``，两者必须能分辨）；
3. :meth:`TestCompareProbes.test_improved_true_even_if_accuracy_falls` /
   :meth:`TestCompareProbes.test_improved_false_even_if_accuracy_rises` ——
   ``improved`` 的唯一判据是**困惑度下降**，与命中率涨跌无关（字符级词表上
   命中率天然不敏感，用它当判据会把"概率被抬高了"这件事判成没进步）。

另外两条刻意记录的**实现口径**（测试按实现断言，不改源码）：

- ``sequence_logprob`` 只跳过**目标**位置的 ``IGNORE_INDEX``，被屏蔽的位置
  仍然会作为**下一个位置的上下文**参与计算——因此含中部屏蔽位的序列会以
  "上下文 token -100 超出词表"报错（见
  :meth:`TestSequenceLogprob.test_middle_ignore_index_breaks_the_next_position`）；
- ``token_accuracy`` 没有 ``ignore_index`` 参数，遇到 ``IGNORE_INDEX`` 会直接
  报"超出词表"而不是跳过，与 ``sequence_logprob`` 的口径不同（见
  :meth:`TestTokenAccuracy.test_ignore_index_is_not_skipped`）。
"""

from __future__ import annotations

import json
import math

import pytest

from smart_research_agent.finetune_eval.metrics import FinetuneEvalError
from smart_research_agent.finetune_eval.model_probe import (
    MIN_PROBE_TOKENS,
    PROBE_FIELDS,
    ProbeRecord,
    ProbeResult,
    compare_probes,
    encode_probe_batches,
    probe_items,
    sequence_logprob,
    token_accuracy,
)
from smart_research_agent.finetune_eval.suites import EvalItem
from smart_research_agent.sft import ReferenceSFTModel
from smart_research_agent.sft.encoding import IGNORE_INDEX, CharTokenizer

#: 假模型的默认词表大小（与"均匀分布 → 困惑度 == 词表大小"这条不变式对应）
FAKE_VOCAB = 4


# ---------------------------------------------------------------------------
# 自造的假模型：只实现 ``ContextModel`` 协议声明的两个成员
# ---------------------------------------------------------------------------


class UniformModel:
    """均匀分布模型：任何上下文的 logits 全为 0 → 每个 token 概率 ``1/V``.

    它让"对数概率"变成一行算术：``-(n-1)·ln(V)``。``calls`` 记录被查询过的
    上下文，用来证明"探针取了哪些位置"。
    """

    def __init__(self, vocab_size: int = FAKE_VOCAB) -> None:
        self._vocab_size = vocab_size
        self.calls: list[int] = []

    @property
    def vocab_size(self) -> int:
        """词表大小."""
        return self._vocab_size

    def logits(self, context_token: int) -> list[float]:
        """返回全 0 向量（softmax 之后是均匀分布）."""
        self.calls.append(context_token)
        return [0.0] * self._vocab_size


class SuccessorModel:
    """后继模型：``logits(ctx)`` 在 ``(ctx + 1) % V`` 处为 1.0，其余为 0.

    于是 ``argmax`` 恒为 ``(ctx + 1) % V``：命中率因此完全由数据决定，可以
    按"哪几个位置的 ``ids[t-1] + 1 == ids[t]``"逐个数出来。
    """

    def __init__(self, vocab_size: int = FAKE_VOCAB) -> None:
        self._vocab_size = vocab_size
        self.calls: list[int] = []

    @property
    def vocab_size(self) -> int:
        """词表大小."""
        return self._vocab_size

    def logits(self, context_token: int) -> list[float]:
        """把最大 logit 放在"后继"位置."""
        self.calls.append(context_token)
        row = [0.0] * self._vocab_size
        row[(context_token + 1) % self._vocab_size] = 1.0
        return row


def successor_hit_logprob(vocab_size: int) -> float:
    """后继模型在"目标正好是 argmax"时的对数概率（闭式）."""
    return -math.log((vocab_size - 1) * math.exp(-1.0) + 1.0)


def successor_miss_logprob(vocab_size: int) -> float:
    """后继模型在"目标不是 argmax"时的对数概率（闭式）."""
    return -1.0 + successor_hit_logprob(vocab_size)


# ---------------------------------------------------------------------------
# 用例与分词器
# ---------------------------------------------------------------------------


def build_items() -> list[EvalItem]:
    """两条覆盖不同桶、不同长度的用例（探针吃的是 ``EvalItem``）."""
    return [
        EvalItem(
            id="cite-01",
            bucket="citation",
            difficulty="normal",
            instruction="LoRA 的增量矩阵是怎么来的？",
            reference="低秩矩阵 A 与 B 的乘积。",
        ),
        EvalItem(
            id="fmt-01",
            bucket="format",
            difficulty="easy",
            instruction="用一句话说明 RAG 的作用。",
            reference="结论：RAG 把检索到的证据拼进提示词。",
        ),
    ]


def build_tokenizer(items: list[EvalItem]) -> CharTokenizer:
    """按用例文本现造字符级词表（与 ``probe_items`` 的使用口径一致）."""
    return CharTokenizer.from_texts(
        [text for item in items for text in (item.instruction, item.reference)]
    )


def build_records(
    logprobs: tuple[float, ...],
    accuracies: tuple[float, ...],
    *,
    supervised: tuple[int, ...] | None = None,
) -> list[ProbeRecord]:
    """按位置手工造一组记录（``item_id`` 固定为 ``a``、``b``、``c``…）."""
    counts = supervised if supervised is not None else tuple(4 for _ in logprobs)
    return [
        ProbeRecord(
            item_id=chr(ord("a") + index),
            bucket="citation" if index % 2 == 0 else "format",
            difficulty="normal",
            tokens=counts[index] + 1,
            supervised=counts[index],
            logprob=logprob,
            accuracy=accuracies[index],
        )
        for index, logprob in enumerate(logprobs)
    ]


def build_result(
    label: str, logprobs: tuple[float, ...], accuracies: tuple[float, ...]
) -> ProbeResult:
    """按位置手工造一份探针结果（每条 4 个监督位置）."""
    return ProbeResult(label=label, records=build_records(logprobs, accuracies))


@pytest.fixture
def items() -> list[EvalItem]:
    """两条种子用例（每个用例新建一份，避免状态泄漏）."""
    return build_items()


@pytest.fixture
def tokenizer(items: list[EvalItem]) -> CharTokenizer:
    """与 ``items`` 配套的字符级分词器."""
    return build_tokenizer(items)


# ---------------------------------------------------------------------------
# sequence_logprob
# ---------------------------------------------------------------------------


class TestSequenceLogprob:
    """序列对数概率之和与**分母**（监督位置数）."""

    @pytest.mark.parametrize(
        ("length", "vocab_size"),
        [(2, 4), (3, 4), (5, 4), (6, 8), (4, 16)],
    )
    def test_uniform_model_matches_manual_formula(self, length, vocab_size):
        """均匀分布下 ``logprob == -(n-1)·ln(V)``、``count == n-1``（逐位手算）."""
        model = UniformModel(vocab_size)
        token_ids = [index % vocab_size for index in range(length)]
        logprob, count = sequence_logprob(model, token_ids)
        assert count == length - 1
        assert logprob == pytest.approx(-(length - 1) * math.log(vocab_size))

    def test_returns_sum_and_count(self):
        """返回值是 ``(概率和, 分母)``：分母单独交出来，才能算平均."""
        model = UniformModel(FAKE_VOCAB)
        logprob, count = sequence_logprob(model, [1, 2, 3])
        assert (logprob, count) == (pytest.approx(-2 * math.log(FAKE_VOCAB)), 2)
        assert isinstance(count, int)

    @pytest.mark.parametrize("token_ids", [[], [3]])
    def test_short_sequence_rejected(self, token_ids):
        """短于 ``MIN_PROBE_TOKENS`` 的序列没有"下一个 token"，直接报错."""
        model = UniformModel(FAKE_VOCAB)
        with pytest.raises(FinetuneEvalError, match="至少需要"):
            sequence_logprob(model, token_ids)

    def test_min_tokens_constant_is_two(self):
        """``MIN_PROBE_TOKENS = 2``：1 个 token 的序列分母必为 0，所以下限是 2."""
        assert MIN_PROBE_TOKENS == 2

    @pytest.mark.parametrize("token_ids", [[4, 1], [-1, 1], [IGNORE_INDEX, 1], [99, 2]])
    def test_context_out_of_range_rejected(self, token_ids):
        """上下文 token 越界 → 报错（用错词表的 tokenizer 会得到"模型很差"的假象）."""
        model = UniformModel(FAKE_VOCAB)
        with pytest.raises(FinetuneEvalError, match="上下文 token"):
            sequence_logprob(model, token_ids)

    @pytest.mark.parametrize("token_ids", [[1, 4], [1, -1], [1, 99]])
    def test_target_out_of_range_rejected(self, token_ids):
        """目标 token 越界 → 报错（id 根本不属于这个模型）."""
        model = UniformModel(FAKE_VOCAB)
        with pytest.raises(FinetuneEvalError, match="目标 token"):
            sequence_logprob(model, token_ids)

    @pytest.mark.parametrize(
        "token_ids",
        [[IGNORE_INDEX, IGNORE_INDEX], [0, IGNORE_INDEX, IGNORE_INDEX]],
    )
    def test_all_positions_masked_rejected(self, token_ids):
        """全部被屏蔽 → 没有任何可计分的位置，报错（而不是返回 0.0）."""
        model = UniformModel(FAKE_VOCAB)
        with pytest.raises(FinetuneEvalError, match="没有任何可计分的位置"):
            sequence_logprob(model, token_ids)

    def test_position_zero_is_skipped(self):
        """位置 0 没有前文：上下文只能取自 ``ids[t-1]``（``t >= 1``）."""
        model = UniformModel(FAKE_VOCAB)
        token_ids = [1, 2, 3, 0]
        sequence_logprob(model, token_ids)
        assert model.calls == token_ids[:-1]

    def test_ignore_index_target_is_skipped(self):
        """尾部的 ``IGNORE_INDEX`` 目标被跳过：分母变小、和不变（等价于截断）."""
        model = UniformModel(FAKE_VOCAB)
        masked = sequence_logprob(model, [1, 2, IGNORE_INDEX])
        truncated = sequence_logprob(model, [1, 2])
        assert masked == pytest.approx(truncated)
        assert masked[1] == 1

    def test_ignore_index_only_shrinks_the_denominator(self):
        """同一个位置被屏蔽前后：和相同，分母差 1."""
        model = UniformModel(FAKE_VOCAB)
        logprob, count = sequence_logprob(model, [1, 2, IGNORE_INDEX])
        assert logprob == pytest.approx(-math.log(FAKE_VOCAB))
        assert count == 1

    def test_middle_ignore_index_breaks_the_next_position(self):
        """**实现口径**：被屏蔽的位置仍会作为下一个位置的上下文，于是直接报错.

        跳过只发生在**目标**上（``sequence_logprob`` 拿不到 labels），所以含
        中部屏蔽位的序列无法用它计分。真实调用方（``probe_items``）传的是
        tokenizer 产出的原始 id，不含 -100，因此不受影响。
        """
        model = UniformModel(FAKE_VOCAB)
        with pytest.raises(FinetuneEvalError, match="上下文 token -100"):
            sequence_logprob(model, [1, IGNORE_INDEX, 3])

    def test_custom_ignore_index_is_honoured(self):
        """``ignore_index`` 是可传参的：把某个正常 id 当屏蔽位时，该位置被跳过."""
        model = UniformModel(FAKE_VOCAB)
        default, default_count = sequence_logprob(model, [1, 2, 3])
        custom, custom_count = sequence_logprob(model, [1, 2, 3], ignore_index=2)
        assert custom_count == 1
        assert custom == pytest.approx(default / 2)
        assert default_count == 2

    def test_explicit_default_ignore_index_is_equivalent(self):
        """显式传 ``IGNORE_INDEX`` 与使用默认值必须完全等价."""
        model = UniformModel(FAKE_VOCAB)
        token_ids = [1, 2, 3]
        assert sequence_logprob(model, token_ids, ignore_index=IGNORE_INDEX) == pytest.approx(
            sequence_logprob(model, token_ids)
        )

    def test_successor_model_hit_is_hand_computed(self):
        """后继模型命中时 ``logprob == -ln((V-1)·e^-1 + 1)``（手算闭式）."""
        model = SuccessorModel(FAKE_VOCAB)
        logprob, count = sequence_logprob(model, [2, 3])
        assert count == 1
        assert logprob == pytest.approx(successor_hit_logprob(FAKE_VOCAB))

    def test_successor_model_miss_is_hand_computed(self):
        """后继模型未命中时，目标 token 的 logit 是 0 → 多减一个 1（手算闭式）."""
        model = SuccessorModel(FAKE_VOCAB)
        logprob, count = sequence_logprob(model, [2, 0])
        assert count == 1
        assert logprob == pytest.approx(successor_miss_logprob(FAKE_VOCAB))

    def test_successor_model_sums_positions(self):
        """多位序列的和 = 各位置之和（``2→3`` 命中、``3→1`` 未命中）."""
        model = SuccessorModel(FAKE_VOCAB)
        logprob, count = sequence_logprob(model, [2, 3, 1])
        assert count == 2
        assert logprob == pytest.approx(
            successor_hit_logprob(FAKE_VOCAB) + successor_miss_logprob(FAKE_VOCAB)
        )

    def test_larger_vocab_lowers_the_logprob(self):
        """词表越大，均匀分布下的对数概率越低（``-ln(V)`` 单调下降）."""
        token_ids = [1, 2]
        small, _ = sequence_logprob(UniformModel(4), token_ids)
        large, _ = sequence_logprob(UniformModel(16), token_ids)
        assert large < small

    def test_logprob_is_never_positive(self):
        """对数概率恒非正（概率不超过 1）——它决定了 ``loss = -mean_logprob >= 0``."""
        assert sequence_logprob(UniformModel(FAKE_VOCAB), [0, 1, 2])[0] <= 0.0

    def test_accepts_list_and_tuple(self):
        """签名收的是 ``Sequence[int]``：列表与元组必须给出同一结果."""
        model = UniformModel(FAKE_VOCAB)
        assert sequence_logprob(model, [1, 2, 3]) == pytest.approx(
            sequence_logprob(model, (1, 2, 3))
        )


# ---------------------------------------------------------------------------
# token_accuracy
# ---------------------------------------------------------------------------


class TestTokenAccuracy:
    """下一 token 命中率：``argmax logits == 目标`` 的比例."""

    def test_successor_model_is_perfect_on_successors(self):
        """后继模型的 logits 恰好指向下一个 id → 全命中."""
        assert token_accuracy(SuccessorModel(FAKE_VOCAB), [0, 1, 2]) == (1.0, 2)

    def test_successor_model_all_misses(self):
        """目标与 argmax 处处不同 → 命中率 0.0（分母仍是 2）."""
        assert token_accuracy(SuccessorModel(FAKE_VOCAB), [0, 2, 1]) == (0.0, 2)

    def test_partial_hits_hand_counted(self):
        """手数：``[0,1,3,4]`` 在 V=8 下只有第 0→1、第 3→4 两个位置命中 → 2/3."""
        assert token_accuracy(SuccessorModel(8), [0, 1, 3, 4]) == (pytest.approx(2 / 3), 3)

    @pytest.mark.parametrize(
        ("token_ids", "expected"),
        [([3, 0], 1.0), ([3, 1], 0.0), ([3, 0, 1], 0.5)],
    )
    def test_uniform_model_tie_breaks_to_first_index(self, token_ids, expected):
        """全 0 logits 时 ``argmax`` 取第一个下标（并列行为必须是确定的）."""
        assert token_accuracy(UniformModel(FAKE_VOCAB), token_ids) == (expected, len(token_ids) - 1)

    @pytest.mark.parametrize("token_ids", [[], [2]])
    def test_short_sequence_rejected(self, token_ids):
        """长度不足 → 报错（分母会是 0）."""
        with pytest.raises(FinetuneEvalError, match="至少需要"):
            token_accuracy(UniformModel(FAKE_VOCAB), token_ids)

    @pytest.mark.parametrize("token_ids", [[-1, 0], [0, 9], [9, 0], [0, -100]])
    def test_out_of_range_rejected(self, token_ids):
        """任一 token 越界 → 报错（与 ``sequence_logprob`` 用同一句话说明原因）."""
        with pytest.raises(FinetuneEvalError, match="超出模型词表"):
            token_accuracy(UniformModel(FAKE_VOCAB), token_ids)

    def test_ignore_index_is_not_skipped(self):
        """**实现口径**：``token_accuracy`` 没有 ``ignore_index`` 参数，屏蔽位会报错.

        这与 ``sequence_logprob``（跳过被屏蔽的**目标**）口径不同，因此两者
        不能互相替代；真实调用方传的是不含 -100 的原始 id 序列。
        """
        with pytest.raises(FinetuneEvalError, match="超出模型词表"):
            token_accuracy(UniformModel(FAKE_VOCAB), [1, IGNORE_INDEX])

    def test_denominator_is_length_minus_one(self):
        """分母恒为 ``len(ids) - 1``（位置 0 不算）."""
        _, count = token_accuracy(SuccessorModel(8), [0, 1, 2, 3, 4])
        assert count == 4

    @pytest.mark.parametrize("token_ids", [[0, 1, 2], [3, 2, 1, 0], [1, 1, 1]])
    def test_stays_within_unit_interval(self, token_ids):
        """命中率必须落在 [0, 1]."""
        accuracy, count = token_accuracy(SuccessorModel(FAKE_VOCAB), token_ids)
        assert 0.0 <= accuracy <= 1.0
        assert count == len(token_ids) - 1

    def test_accuracy_and_logprob_measure_different_things(self):
        """同一个模型上："全命中"与"对数概率为负"同时成立——两者量的不是一件事."""
        model = SuccessorModel(FAKE_VOCAB)
        accuracy, _ = token_accuracy(model, [0, 1, 2])
        logprob, _ = sequence_logprob(SuccessorModel(FAKE_VOCAB), [0, 1, 2])
        assert accuracy == 1.0
        assert logprob < 0.0


# ---------------------------------------------------------------------------
# encode_probe_batches
# ---------------------------------------------------------------------------


class TestEncodeProbeBatches:
    """把用例编码成**单样本** Batch，并把 prompt 段屏蔽为 ``IGNORE_INDEX``."""

    def test_prompt_segment_is_masked(self, items, tokenizer):
        """前 ``len(prompt_ids)`` 个位置的 label 必须全为 ``IGNORE_INDEX``."""
        item = items[0]
        batch = encode_probe_batches([item], tokenizer)[0]
        prompt_length = len(tokenizer.encode(item.instruction))
        labels = batch.labels[0]
        assert labels[:prompt_length] == (IGNORE_INDEX,) * prompt_length

    def test_answer_segment_keeps_original_ids(self, items, tokenizer):
        """答案段的 label 就是答案 id 本身（它们才是要学的目标）."""
        item = items[0]
        batch = encode_probe_batches([item], tokenizer)[0]
        prompt_length = len(tokenizer.encode(item.instruction))
        assert list(batch.labels[0][prompt_length:]) == tokenizer.encode(item.reference)

    def test_input_ids_is_prompt_plus_answer(self, items, tokenizer):
        """``input_ids`` 是"前缀编码 + 答案编码"的拼接（分别编码再拼，不按字符偏移切）."""
        item = items[0]
        batch = encode_probe_batches([item], tokenizer)[0]
        expected = tokenizer.encode(item.instruction) + tokenizer.encode(item.reference)
        assert list(batch.input_ids[0]) == expected

    def test_attention_mask_is_all_ones(self, items, tokenizer):
        """没有 padding → attention_mask 全 1（长度与 input_ids 相同）."""
        batch = encode_probe_batches(items, tokenizer)[0]
        assert set(batch.attention_mask[0]) == {1}
        assert len(batch.attention_mask[0]) == len(batch.input_ids[0])

    def test_pad_token_id_matches_tokenizer(self, items, tokenizer):
        """``pad_token_id`` 必须取自 tokenizer（两边不一致会让下游错位）."""
        for batch in encode_probe_batches(items, tokenizer):
            assert batch.pad_token_id == tokenizer.pad_token_id

    def test_one_batch_per_item(self, items, tokenizer):
        """逐条处理：每条用例一个 Batch，批大小为 1（不 padding）."""
        batches = encode_probe_batches(items, tokenizer)
        assert len(batches) == len(items)
        assert [batch.batch_size for batch in batches] == [1, 1]

    def test_no_padding_tokens(self, items, tokenizer):
        """单样本批次的 padding 量恒为 0（长度由数据自己决定）."""
        for batch in encode_probe_batches(items, tokenizer):
            assert batch.padding_tokens == 0

    def test_lengths_match_encoded_lengths(self, items, tokenizer):
        """批次长度等于"前缀 + 答案"的编码长度（不做截断）."""
        for item, batch in zip(items, encode_probe_batches(items, tokenizer)):
            expected = len(tokenizer.encode(item.instruction)) + len(
                tokenizer.encode(item.reference)
            )
            assert batch.lengths == (expected,)

    def test_label_length_equals_input_length(self, items, tokenizer):
        """``input_ids`` 与 ``labels`` 必须等长（长度不一致是送错数据的典型方式）."""
        for batch in encode_probe_batches(items, tokenizer):
            assert len(batch.labels[0]) == len(batch.input_ids[0])

    def test_ignored_positions_equal_prompt_length(self, items, tokenizer):
        """被屏蔽的位置数**恰好**等于前缀长度（多一个就丢信号，少一个就学到提问）."""
        item = items[0]
        batch = encode_probe_batches([item], tokenizer)[0]
        assert batch.labels[0].count(IGNORE_INDEX) == len(tokenizer.encode(item.instruction))

    def test_supervised_tokens_equals_answer_length(self, items, tokenizer):
        """监督 token 数等于答案长度（prompt 一个都不算）."""
        for item, batch in zip(items, encode_probe_batches(items, tokenizer)):
            assert batch.supervised_tokens == len(tokenizer.encode(item.reference))

    def test_empty_items_gives_empty_list(self, tokenizer):
        """空用例列表 → 空批次列表（不报错：这里是纯编码，没有"评估集为空"的语义）."""
        assert encode_probe_batches([], tokenizer) == []

    def test_unknown_characters_become_unk_and_stay_masked(self):
        """词表外字符映射为 ``<unk>``，且仍落在被屏蔽的前缀段内."""
        tokenizer = CharTokenizer.from_texts(["ab"])
        item = EvalItem(
            id="unk-01",
            bucket="citation",
            difficulty="easy",
            instruction="axb",
            reference="ab",
        )
        batch = encode_probe_batches([item], tokenizer)[0]
        assert batch.input_ids[0][:3] == (2, 1, 3)  # a / <unk> / b
        assert batch.labels[0][:3] == (IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX)
        assert batch.supervised_tokens == 2

    def test_batch_is_a_legal_training_batch(self, items, tokenizer):
        """产出的批次必须能被参考模型直接 ``accumulate``（"它就是一份合法训练批次"）."""
        batch = encode_probe_batches([items[0]], tokenizer)[0]
        model = ReferenceSFTModel(tokenizer.vocab_size, seed=42)
        _, count = model.accumulate(batch)
        assert count == len(tokenizer.encode(items[0].reference))


# ---------------------------------------------------------------------------
# probe_items
# ---------------------------------------------------------------------------


class TestProbeItems:
    """逐条探针并返回带聚合的结果."""

    @pytest.mark.parametrize("probe_field", ["answer", "", "REFERENCE", "Reference", "output"])
    def test_unknown_probe_field_rejected(self, probe_field, items, tokenizer):
        """未知字段直接报错，并把可选值写进消息（不静默退回缺省值）."""
        model = UniformModel(tokenizer.vocab_size)
        with pytest.raises(FinetuneEvalError, match="未知的探针字段"):
            probe_items(model, tokenizer, items, probe_field=probe_field)

    def test_probe_fields_constant(self):
        """可探测的字段只有参考答案与指令两种（顺序即文档顺序）."""
        assert PROBE_FIELDS == ("reference", "instruction")

    def test_empty_items_rejected(self, tokenizer):
        """空评估集 → 报错（返回空结果会让"困惑度"无从计算）."""
        model = UniformModel(tokenizer.vocab_size)
        with pytest.raises(FinetuneEvalError, match="不能对空评估集做探针"):
            probe_items(model, tokenizer, [])

    def test_default_field_is_reference(self, items, tokenizer):
        """缺省探"参考答案"：显式传 ``"reference"`` 必须等价."""
        model = UniformModel(tokenizer.vocab_size)
        default = probe_items(model, tokenizer, items, label="probe")
        explicit = probe_items(model, tokenizer, items, label="probe", probe_field="reference")
        assert default == explicit

    def test_reference_field_counts_reference_tokens(self, items, tokenizer):
        """探参考答案时，``tokens`` 等于参考答案的编码长度."""
        result = probe_items(UniformModel(tokenizer.vocab_size), tokenizer, items)
        for item, record in zip(items, result.records):
            assert record.tokens == len(tokenizer.encode(item.reference))

    def test_instruction_field_counts_instruction_tokens(self, items, tokenizer):
        """探指令时 ``tokens`` 换成指令的编码长度（同一段代码、不同字段）."""
        result = probe_items(
            UniformModel(tokenizer.vocab_size), tokenizer, items, probe_field="instruction"
        )
        for item, record in zip(items, result.records):
            assert record.tokens == len(tokenizer.encode(item.instruction))

    def test_two_fields_give_different_numbers(self, items, tokenizer):
        """两个字段的读数**不同**：探答案是"对领域答案多熟"，探指令是"对提问分布多熟"."""
        model = UniformModel(tokenizer.vocab_size)
        by_reference = probe_items(model, tokenizer, items)
        by_instruction = probe_items(model, tokenizer, items, probe_field="instruction")
        assert by_reference.total_tokens != by_instruction.total_tokens

    def test_record_fields_are_filled(self, items, tokenizer):
        """``ProbeRecord`` 的六个字段逐一对上用例与模型读数."""
        record = probe_items(UniformModel(tokenizer.vocab_size), tokenizer, items).records[0]
        assert (record.item_id, record.bucket, record.difficulty) == (
            items[0].id,
            items[0].bucket,
            items[0].difficulty,
        )
        assert record.tokens == len(tokenizer.encode(items[0].reference))
        assert record.supervised == record.tokens - 1

    def test_records_follow_item_order(self, items, tokenizer):
        """逐条读数保持用例顺序（报告要能逐行比对）."""
        result = probe_items(UniformModel(tokenizer.vocab_size), tokenizer, items)
        assert [record.item_id for record in result.records] == [item.id for item in items]

    def test_label_is_recorded(self, items, tokenizer):
        """``label`` 用来区分"微调前 / 微调后"，必须原样带进结果."""
        result = probe_items(UniformModel(tokenizer.vocab_size), tokenizer, items, label="after")
        assert result.label == "after"

    def test_uniform_model_logprob_matches_manual_formula(self, items, tokenizer):
        """每条记录的对数概率都是 ``-(tokens-1)·ln(V)``（均匀分布的手算值）."""
        vocab_size = tokenizer.vocab_size
        result = probe_items(UniformModel(vocab_size), tokenizer, items)
        for record in result.records:
            assert record.logprob == pytest.approx(-(record.tokens - 1) * math.log(vocab_size))

    def test_record_mean_logprob_is_logprob_over_supervised(self, items, tokenizer):
        """单条记录的平均对数概率 = 概率和 / 监督位置数（分母单独带出来的直接用途）."""
        result = probe_items(UniformModel(tokenizer.vocab_size), tokenizer, items)
        for record in result.records:
            assert record.mean_logprob == pytest.approx(record.logprob / record.supervised)

    def test_record_perplexity_is_exp_of_negative_mean(self, items, tokenizer):
        """单条记录的困惑度 = ``exp(-平均对数概率)``."""
        result = probe_items(UniformModel(tokenizer.vocab_size), tokenizer, items)
        record = result.records[0]
        assert record.perplexity == pytest.approx(math.exp(-record.mean_logprob))

    def test_uniform_model_perplexity_equals_vocab_size(self, items, tokenizer):
        """均匀分布下困惑度等于**词表大小**（这是困惑度最好解释的一种情形）."""
        result = probe_items(UniformModel(tokenizer.vocab_size), tokenizer, items)
        assert result.perplexity == pytest.approx(tokenizer.vocab_size)

    def test_uniform_model_accuracy_is_zero(self, items, tokenizer):
        """均匀分布时 ``argmax`` 恒为下标 0，而字符 id 从 1 起 → 命中率为 0."""
        result = probe_items(UniformModel(tokenizer.vocab_size), tokenizer, items)
        assert result.accuracy == 0.0

    def test_successor_model_accuracy_and_perplexity_hand_computed(self):
        """后继模型 + 2 字答案：命中率 1.0、困惑度 ``(V-1)/e + 1``（手算闭式）."""
        item = EvalItem(
            id="tiny-01",
            bucket="citation",
            difficulty="easy",
            instruction="ab",
            reference="ab",
        )
        tokenizer = CharTokenizer.from_texts(["ab"])
        model = SuccessorModel(tokenizer.vocab_size)
        result = probe_items(model, tokenizer, [item])
        record = result.records[0]
        assert (record.tokens, record.supervised) == (2, 1)
        assert record.accuracy == 1.0
        assert record.logprob == pytest.approx(successor_hit_logprob(tokenizer.vocab_size))
        assert record.perplexity == pytest.approx((tokenizer.vocab_size - 1) * math.exp(-1.0) + 1.0)

    def test_successor_model_misses_on_non_successor_ids(self):
        """词表按 ``ba`` 构造，于是 ``ab`` 的 id 是 ``(3, 2)``：``3`` 的 argmax 是 0，必不命中."""
        item = EvalItem(
            id="tiny-02",
            bucket="citation",
            difficulty="easy",
            instruction="ba",
            reference="ab",
        )
        tokenizer = CharTokenizer.from_texts(["ba"])
        model = SuccessorModel(tokenizer.vocab_size)
        assert tokenizer.encode("ab") == [3, 2]
        record = probe_items(model, tokenizer, [item]).records[0]
        assert record.accuracy == 0.0
        assert record.logprob == pytest.approx(successor_miss_logprob(tokenizer.vocab_size))

    def test_result_totals_match_records(self, items, tokenizer):
        """聚合分母 = 各记录 ``supervised`` 之和（逐条与聚合不能各算各的）."""
        result = probe_items(UniformModel(tokenizer.vocab_size), tokenizer, items)
        assert result.total_tokens == sum(record.supervised for record in result.records)
        assert result.total_tokens == sum(record.tokens - 1 for record in result.records)


# ---------------------------------------------------------------------------
# ProbeResult 聚合
# ---------------------------------------------------------------------------


class TestProbeResult:
    """``ProbeResult`` 的聚合口径：**按监督 token 数加权**."""

    def test_total_tokens_is_sum_of_supervised(self):
        """分母是监督位置总数，不是条目数、也不是总 token 数."""
        result = ProbeResult(
            label="x",
            records=build_records((-8.0, -1.0), (0.5, 1.0), supervised=(4, 1)),
        )
        assert result.total_tokens == 5
        assert len(result.records) == 2

    def test_mean_logprob_is_token_weighted(self):
        """**加权口径**：``(-8 + -1) / (4 + 1) = -1.8``，而不是条目平均的 ``-1.5``."""
        result = ProbeResult(
            label="x",
            records=build_records((-8.0, -1.0), (0.5, 1.0), supervised=(4, 1)),
        )
        item_average = sum(record.mean_logprob for record in result.records) / len(result.records)
        assert result.mean_logprob == pytest.approx(-1.8)
        assert item_average == pytest.approx(-1.5)
        assert result.mean_logprob != pytest.approx(item_average)

    def test_accuracy_is_token_weighted(self):
        """命中率同样按监督 token 数加权：``(0.5·4 + 1.0·1) / 5 = 0.6``."""
        result = ProbeResult(
            label="x",
            records=build_records((-8.0, -1.0), (0.5, 1.0), supervised=(4, 1)),
        )
        assert result.accuracy == pytest.approx(0.6)

    def test_perplexity_is_exp_of_negative_mean(self):
        """整体困惑度 = ``exp(-平均对数概率)`` = ``exp(1.8)``."""
        result = ProbeResult(
            label="x",
            records=build_records((-8.0, -1.0), (0.5, 1.0), supervised=(4, 1)),
        )
        assert result.perplexity == pytest.approx(math.exp(1.8))

    def test_single_record_aggregation(self):
        """单条记录的聚合就是它自己（没有加权可谈时的边界）."""
        result = ProbeResult(label="x", records=build_records((-4.0,), (0.25,)))
        assert result.mean_logprob == pytest.approx(-1.0)
        assert result.accuracy == pytest.approx(0.25)
        assert result.total_tokens == 4

    def test_by_bucket_groups_and_weights(self):
        """按桶聚合：每桶内部仍按 token 加权，且键齐全."""
        result = ProbeResult(
            label="x",
            records=build_records((-8.0, -1.0), (0.5, 1.0), supervised=(4, 1)),
        )
        table = result.by_bucket()
        assert set(table) == {"citation", "format"}
        assert table["citation"] == {
            "items": 1,
            "supervised": 4,
            "mean_logprob": pytest.approx(-2.0),
            "perplexity": pytest.approx(math.exp(2.0)),
            "accuracy": pytest.approx(0.5),
        }
        assert table["format"]["items"] == 1
        assert table["format"]["mean_logprob"] == pytest.approx(-1.0)
        assert table["format"]["accuracy"] == pytest.approx(1.0)

    def test_by_bucket_is_sorted(self):
        """桶按名字排序（报告里的表格顺序必须稳定）."""
        result = ProbeResult(
            label="x",
            records=build_records((-8.0, -1.0), (0.5, 1.0), supervised=(4, 1)),
        )
        assert list(result.by_bucket()) == ["citation", "format"]

    def test_by_bucket_merges_same_bucket_records(self):
        """同一个桶里的多条记录被合并（条目数、分母与加权读数都对）."""
        records = [
            ProbeRecord("a", "citation", "easy", 5, 4, -4.0, 0.5),
            ProbeRecord("b", "citation", "hard", 3, 2, -2.0, 1.0),
        ]
        table = ProbeResult(label="x", records=records).by_bucket()
        assert table["citation"]["items"] == 2
        assert table["citation"]["supervised"] == 6
        assert table["citation"]["mean_logprob"] == pytest.approx(-1.0)
        assert table["citation"]["accuracy"] == pytest.approx((0.5 * 4 + 1.0 * 2) / 6)

    def test_by_bucket_is_empty_for_empty_result(self):
        """空结果没有桶（不报错，与"求平均"的报错语义区分开）."""
        assert ProbeResult(label="x").by_bucket() == {}

    def test_to_dict_keys_complete(self):
        """``to_dict`` 的键集合完整（逐条 + 聚合一起留档）."""
        payload = ProbeResult(label="x", records=build_records((-4.0,), (0.5,))).to_dict()
        assert set(payload) == {
            "label",
            "total_tokens",
            "mean_logprob",
            "perplexity",
            "accuracy",
            "by_bucket",
            "records",
        }
        assert payload["label"] == "x"
        assert payload["total_tokens"] == 4

    def test_to_dict_contains_record_projection(self):
        """``records`` 是逐条 ``to_dict`` 的列表（字段名可核对）."""
        records = build_records((-4.0,), (0.5,))
        payload = ProbeResult(label="x", records=records).to_dict()
        assert payload["records"] == [records[0].to_dict()]

    def test_to_dict_is_json_serializable(self):
        """投影结果必须能 ``json.dumps``（报告要落盘）."""
        payload = ProbeResult(label="x", records=build_records((-4.0, -1.0), (0.5, 1.0))).to_dict()
        restored = json.loads(json.dumps(payload, ensure_ascii=False))
        assert restored["total_tokens"] == payload["total_tokens"]
        assert restored["by_bucket"]["citation"]["items"] == 1

    def test_record_to_dict_keys_complete(self):
        """单条记录的投影包含派生量（``mean_logprob`` 与 ``perplexity``）."""
        record = build_records((-4.0,), (0.5,))[0]
        payload = record.to_dict()
        assert set(payload) == {
            "item_id",
            "bucket",
            "difficulty",
            "tokens",
            "supervised",
            "logprob",
            "mean_logprob",
            "perplexity",
            "accuracy",
        }
        assert payload["mean_logprob"] == pytest.approx(-1.0)
        assert payload["perplexity"] == pytest.approx(math.exp(1.0))

    def test_empty_result_mean_logprob_rejected(self):
        """空结果的 ``mean_logprob`` 抛 ``FinetuneEvalError``（不是 nan）."""
        with pytest.raises(FinetuneEvalError, match="空探针结果没有平均对数概率"):
            ProbeResult(label="empty").mean_logprob

    def test_empty_result_perplexity_rejected(self):
        """空结果的困惑度同样抛错（它由平均对数概率派生）."""
        with pytest.raises(FinetuneEvalError, match="空探针结果没有平均对数概率"):
            ProbeResult(label="empty").perplexity

    def test_empty_result_accuracy_rejected(self):
        """空结果的命中率也抛错（分母为 0 时不给"0.0"这种假答案）."""
        with pytest.raises(FinetuneEvalError, match="没有任何监督位置"):
            ProbeResult(label="empty").accuracy

    def test_record_without_supervised_positions_rejected(self):
        """``supervised = 0`` 的记录在**构造期**就被拒绝（分母不能是 0）.

        ``probe_items`` 走 ``sequence_logprob``，它保证 ``supervised >= 1``，
        所以这条只在手工构造时才会命中。把它拦在构造期而不是等到
        ``mean_logprob``，是为了让错误类型说人话——真正的问题是"这条记录
        没有可计分的位置"，而不是一句读起来像"代码写错了"的
        ``ZeroDivisionError``。
        """
        with pytest.raises(FinetuneEvalError, match="supervised 必须为正整数"):
            ProbeRecord("a", "citation", "easy", 1, 0, 0.0, 0.0)

    def test_record_with_too_few_tokens_rejected(self):
        """``tokens = 1`` 的记录同样在构造期被拒绝：连一个"下一个 token"都没有."""
        with pytest.raises(FinetuneEvalError, match="tokens 至少为"):
            ProbeRecord("a", "citation", "easy", 1, 1, 0.0, 0.0)

    def test_valid_record_is_accepted(self):
        """正对照：合法的记录必须能构造出来，且平均对数概率 = 和 / 监督位置数."""
        record = ProbeRecord("a", "citation", "easy", 5, 4, -4.0, 0.5)
        assert record.mean_logprob == pytest.approx(record.logprob / 4)
        assert record.mean_logprob == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# compare_probes
# ---------------------------------------------------------------------------


class TestCompareProbes:
    """微调前后的对比：``improved`` 的判据是**困惑度下降**."""

    def test_mismatched_id_sets_rejected(self):
        """两次探针的用例集合必须一致（否则差值没有意义）."""
        before = build_result("before", (-4.0, -4.0), (0.0, 0.0))
        after = build_result("after", (-3.0, -3.0), (0.0, 0.0))
        extra = ProbeResult(
            label="after",
            records=[*after.records, ProbeRecord("zz", "format", "easy", 5, 4, -1.0, 1.0)],
        )
        with pytest.raises(FinetuneEvalError, match="用例集合不一致"):
            compare_probes(before, extra)

    def test_missing_ids_are_listed(self):
        """差异清单里要把对不上的 ``item_id`` 写出来（可核查）."""
        before = ProbeResult(label="before", records=build_records((-4.0,), (0.0,)))
        after = ProbeResult(
            label="after",
            records=[ProbeRecord("other", "citation", "normal", 5, 4, -3.0, 0.0)],
        )
        with pytest.raises(FinetuneEvalError, match="a, other"):
            compare_probes(before, after)

    def test_identical_id_sets_are_accepted(self):
        """集合一致时正常返回（正面用例，避免"只会报错"的假通过）."""
        before = build_result("before", (-4.0, -4.0), (0.0, 0.0))
        after = build_result("after", (-3.0, -3.0), (1.0, 1.0))
        assert compare_probes(before, after)["items"] == 2

    def test_result_keys_complete(self):
        """返回的字段集合完整（报告模板依赖它）."""
        report = compare_probes(
            build_result("before", (-4.0, -4.0), (0.0, 0.0)),
            build_result("after", (-2.0, -6.0), (0.0, 0.0)),
        )
        assert set(report) == {
            "items",
            "before",
            "after",
            "perplexity_delta",
            "perplexity_ratio",
            "accuracy_delta",
            "item_wins",
            "item_losses",
            "improved",
        }
        assert set(report["before"]) == {"label", "perplexity", "mean_logprob", "accuracy"}
        assert set(report["after"]) == {"label", "perplexity", "mean_logprob", "accuracy"}

    def test_labels_are_preserved(self):
        """两边的 ``label``（"微调前 / 微调后"）原样带进报告."""
        report = compare_probes(
            build_result("before", (-4.0,), (0.0,)),
            build_result("after", (-3.0,), (0.0,)),
        )
        assert (report["before"]["label"], report["after"]["label"]) == ("before", "after")

    def test_item_wins_and_losses_are_counted(self):
        """逐条胜负按对数概率数出来：赢 1、输 1、打平 0（打平不算输赢）."""
        before = build_result("before", (-4.0, -4.0, -4.0), (0.0, 0.0, 0.0))
        after = build_result("after", (-2.0, -6.0, -4.0), (0.0, 0.0, 0.0))
        report = compare_probes(before, after)
        assert (report["item_wins"], report["item_losses"]) == (1, 1)
        assert report["items"] == 3

    def test_item_wins_ignore_accuracy(self):
        """胜负只看对数概率：命中率掉下去不妨碍某条被判为"赢"."""
        before = build_result("before", (-4.0,), (1.0,))
        after = build_result("after", (-3.0,), (0.0,))
        report = compare_probes(before, after)
        assert report["item_wins"] == 1
        assert report["accuracy_delta"] == pytest.approx(-1.0)

    def test_improved_true_when_perplexity_drops(self):
        """对数概率升高（困惑度下降）→ ``improved`` 为真."""
        report = compare_probes(
            build_result("before", (-4.0, -4.0), (0.0, 0.0)),
            build_result("after", (-3.0, -3.0), (0.0, 0.0)),
        )
        assert report["perplexity_delta"] > 0
        assert report["improved"] is True

    def test_improved_false_when_perplexity_rises(self):
        """对数概率下降（困惑度上升）→ ``improved`` 为假."""
        report = compare_probes(
            build_result("before", (-4.0, -2.0), (0.0, 0.0)),
            build_result("after", (-3.0, -4.0), (0.0, 0.0)),
        )
        assert report["perplexity_delta"] < 0
        assert report["improved"] is False

    def test_improved_true_even_if_accuracy_falls(self):
        """**判据只看困惑度**：命中率从 1.0 掉到 0.0，只要困惑度降了就算进步."""
        report = compare_probes(
            build_result("before", (-4.0, -4.0), (1.0, 1.0)),
            build_result("after", (-2.0, -2.0), (0.0, 0.0)),
        )
        assert report["accuracy_delta"] == pytest.approx(-1.0)
        assert report["perplexity_delta"] > 0
        assert report["improved"] is True

    def test_improved_false_even_if_accuracy_rises(self):
        """反方向也要钉住：命中率涨了、但困惑度涨了 → 仍判"没进步"."""
        report = compare_probes(
            build_result("before", (-2.0, -2.0), (0.0, 0.0)),
            build_result("after", (-4.0, -4.0), (1.0, 1.0)),
        )
        assert report["accuracy_delta"] == pytest.approx(1.0)
        assert report["perplexity_delta"] < 0
        assert report["improved"] is False

    def test_perplexity_delta_is_before_minus_after(self):
        """``perplexity_delta = 微调前困惑度 - 微调后困惑度``（正数 = 变好）."""
        before = build_result("before", (-4.0, -4.0), (0.0, 0.0))
        after = build_result("after", (-3.0, -3.0), (0.0, 0.0))
        report = compare_probes(before, after)
        assert report["perplexity_delta"] == pytest.approx(before.perplexity - after.perplexity)

    def test_perplexity_ratio_is_after_over_before(self):
        """``perplexity_ratio = 微调后 / 微调前``（小于 1 = 困惑度变低）."""
        before = build_result("before", (-4.0, -4.0), (0.0, 0.0))
        after = build_result("after", (-3.0, -3.0), (0.0, 0.0))
        report = compare_probes(before, after)
        assert report["perplexity_ratio"] == pytest.approx(after.perplexity / before.perplexity)
        assert report["perplexity_ratio"] < 1.0

    def test_accuracy_delta_is_after_minus_before(self):
        """``accuracy_delta = 微调后命中率 - 微调前命中率``."""
        report = compare_probes(
            build_result("before", (-4.0, -4.0), (0.0, 1.0)),
            build_result("after", (-4.0, -4.0), (1.0, 1.0)),
        )
        assert report["accuracy_delta"] == pytest.approx(0.5)

    def test_identical_results_show_no_change(self):
        """同一份结果与自身比较：零差值、胜败皆 0、``improved`` 为假."""
        report = compare_probes(
            build_result("before", (-4.0, -2.0), (0.5, 0.5)),
            build_result("after", (-4.0, -2.0), (0.5, 0.5)),
        )
        assert report["perplexity_delta"] == pytest.approx(0.0)
        assert report["perplexity_ratio"] == pytest.approx(1.0)
        assert report["accuracy_delta"] == pytest.approx(0.0)
        assert (report["item_wins"], report["item_losses"]) == (0, 0)
        assert report["improved"] is False

    def test_both_empty_results_rejected(self):
        """两边都是空结果时同样报错（困惑度无定义，不能给出"零进步"的假结论）."""
        with pytest.raises(FinetuneEvalError, match="空探针结果没有平均对数概率"):
            compare_probes(ProbeResult(label="before"), ProbeResult(label="after"))

    def test_compare_on_real_probe_items(self, items, tokenizer):
        """端到端一次：两个假模型的探针对比，内部算术必须自洽."""
        vocab_size = tokenizer.vocab_size
        before = probe_items(UniformModel(vocab_size), tokenizer, items, label="before")
        after = probe_items(SuccessorModel(vocab_size), tokenizer, items, label="after")
        report = compare_probes(before, after)
        assert report["items"] == len(items)
        assert report["perplexity_ratio"] == pytest.approx(after.perplexity / before.perplexity)
        assert report["perplexity_delta"] == pytest.approx(before.perplexity - after.perplexity)
        assert report["accuracy_delta"] == pytest.approx(after.accuracy - before.accuracy)
        assert report["item_wins"] + report["item_losses"] <= report["items"]
